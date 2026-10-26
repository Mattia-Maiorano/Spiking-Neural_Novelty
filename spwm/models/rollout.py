"""
Unified Action-Conditioned Autoregressive Latent Rollout Engine (Layer 1 — Models).

Single source of truth for autonomous latent rollout mechanics across
training (M1), validation, and evaluation.
Depends exclusively on Layer 0 (`spwm.contracts`).
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from spwm.contracts.states import DynamicsState, RolloutResult, TelemetryContainer
from spwm.contracts.tensors import J, assert_tensor_contract


@torch.no_grad()
def step_grounded(
    model: nn.Module,
    event_frame: torch.Tensor,
    dyn_state: DynamicsState,
) -> Tuple[torch.Tensor, DynamicsState]:
    """Execute a single grounded sensory-driven step using frozen encoder and dynamics."""
    sx, skp = model.encoder.encode_step(event_frame)
    eps = sx - model.sensory_predictor(dyn_state.z_prev)
    return model.dynamics.step(eps, state=dyn_state, sensory_keypoints=skp)


@torch.no_grad()
def ground_initial_state(
    model: nn.Module,
    events: torch.Tensor,
    t_start: int,
) -> Tuple[torch.Tensor, DynamicsState, torch.Tensor]:
    """
    Ground initial latent state z_start and dynamics state from initial frames t in [0, t_start].
    Returns (z_start, dyn_state, z_prefix) where z_prefix is [B, t_start + 1, latent_dim].
    """
    B = events.shape[0]
    device = events.device
    dyn_state = model.dynamics.init_state(B, device=device)
    curr_z = dyn_state.z_prev
    latents = []
    for t in range(t_start + 1):
        curr_z, dyn_state = step_grounded(model, events[:, t], dyn_state)
        latents.append(curr_z)
    z_prefix = torch.stack(latents, dim=1)
    return curr_z, dyn_state, z_prefix


def compute_rollout_schedule(
    phase_flags: Optional[torch.Tensor],
    sequence_length: int,
    temporal_burn_in: int = 5,
    phase_horizon: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Computes canonical (t_start, eval_horizon) for multi-phase robotic manipulation trajectories.
    Identical across training and evaluation.
    """
    T = sequence_length
    if phase_flags is not None and bool((phase_flags == 1).any()):
        first_heft = int((phase_flags == 1).nonzero(as_tuple=False)[:, 1].min().item())
        t_start = min(3, max(1, first_heft - 1)) - 1
        horizon = T - 1 - t_start
    else:
        burn_in = min(temporal_burn_in, max(0, T - 2))
        t_start = burn_in
        horizon = T - 1 - t_start
        if phase_horizon is not None:
            horizon = min(phase_horizon, horizon)
    return t_start, horizon


def rollout_latent(
    model: nn.Module,
    z_init: torch.Tensor,
    horizon: int,
    actions: Optional[torch.Tensor] = None,
    with_grad: bool = False,
    decode_kinematics: bool = True,
    initial_decoder_state: Optional[torch.Tensor] = None,
    telemetry: Optional[Union[TelemetryContainer, Dict[str, List[torch.Tensor]]]] = None,
    start_step: int = 0,
    fixed_predictor: Optional[nn.Module] = None,
    events: Optional[torch.Tensor] = None,
    phase_flags: Optional[torch.Tensor] = None,
    initial_dynamics_state: Optional[DynamicsState] = None,
) -> RolloutResult:
    """Execute an action-conditioned multi-step rollout in latent space starting from z_init.

    Supports Active Perception / Sensory Heft injection when events and phase_flags are provided:
    - On steps with flag == 1 (Sensory Heft): executes grounded sensory step on real event frame.
    - On steps with flag != 1 (Blind movement): executes autonomous M1 predictor step.

    Args:
        model: World model exposing `predictor` / `m1` (and optionally `physical_decoder`).
        z_init: Initial latent state [B, latent_dim].
        horizon: Number of rollout steps (> 0).
        actions: Optional motor action tensor [B, H, action_dim] or [B, action_dim].
        with_grad: If True, preserves the autograd graph during rollout (used by M1 training).
        decode_kinematics: If True and model has `physical_decoder`, decodes [B, H, 2 * joint_count] kinematics.
        initial_decoder_state: Optional warm EMA filter state [B, p_dim] for `physical_decoder`.
        telemetry: Optional TelemetryContainer or dict for logging step states.
        start_step: Optional step offset (retained for signature compatibility, default 0).
        fixed_predictor: Optional predictor module (e.g., `model.m1`) to invoke directly.
        events: Optional event frame tensor [B, H, C, H, W] for active sensory injection.
        phase_flags: Optional phase flags tensor [B, H] in {0, 1, 2}.
        initial_dynamics_state: Optional DynamicsState container for recurrent dynamics sync.

    Returns:
        RolloutResult containing trajectory, actions, decoded_kinematics, active_models, and joint_count.
    """
    if horizon <= 0:
        raise ValueError(f"Rollout horizon must be positive, got {horizon}")
    assert_tensor_contract(z_init, (None, None), "rollout_latent.z_init")

    if fixed_predictor is None and not hasattr(model, "predictor"):
        raise AttributeError("Model must have a 'predictor' attribute when fixed_predictor is not provided.")

    B = z_init.shape[0]
    if actions is not None:
        if actions.dim() == 2:
            assert_tensor_contract(actions, (B, None), "rollout_latent.actions_step")
        elif actions.dim() == 3:
            assert_tensor_contract(actions, (B, None, None), "rollout_latent.actions_seq")
            if actions.shape[1] < horizon:
                raise ValueError(
                    f"rollout_latent: actions sequence length ({actions.shape[1]}) < horizon ({horizon})"
                )
        else:
            raise ValueError(f"rollout_latent: expected 2D or 3D actions tensor, got {actions.dim()}D")

    if events is not None and events.shape[1] < horizon:
        raise ValueError(
            f"rollout_latent: events sequence length ({events.shape[1]}) < horizon ({horizon})"
        )
    if phase_flags is not None and phase_flags.shape[1] < horizon:
        raise ValueError(
            f"rollout_latent: phase_flags sequence length ({phase_flags.shape[1]}) < horizon ({horizon})"
        )

    dyn_state = initial_dynamics_state
    if dyn_state is None and events is not None and hasattr(model, "dynamics") and model.dynamics is not None:
        dyn_state = model.dynamics.init_state(B, device=z_init.device)
        dyn_state = DynamicsState(dyn_state.memory_state, z_init.detach(), dyn_state.ema_spikes)

    grad_context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with grad_context:
        predictions: List[torch.Tensor] = []
        active_models: List[str] = []
        curr_z = z_init

        predictor_fn = fixed_predictor if fixed_predictor is not None else model.predictor

        has_active_perception = (
            events is not None
            and phase_flags is not None
            and hasattr(model, "encoder")
            and hasattr(model, "dynamics")
            and hasattr(model, "sensory_predictor")
        )

        for step in range(horizon):
            u_step = actions[:, step] if (actions is not None and actions.dim() == 3) else actions

            if has_active_perception:
                flag_step = phase_flags[:, step] if phase_flags.dim() == 2 else phase_flags
                event_step = events[:, step]
                is_heft_all = bool((flag_step == 1).all())
                is_heft_any = bool((flag_step == 1).any())

                if is_heft_all:
                    sync_st = DynamicsState(dyn_state.memory_state, curr_z.detach(), dyn_state.ema_spikes) if dyn_state is not None else None
                    curr_z, dyn_state = step_grounded(model, event_step, sync_st)
                    active_models.append("grounded")
                elif is_heft_any:
                    pred_out = predictor_fn(curr_z, u_step) if u_step is not None else predictor_fn(curr_z)
                    m1_z = pred_out.predicted_latent
                    sync_st = DynamicsState(dyn_state.memory_state, curr_z.detach(), dyn_state.ema_spikes) if dyn_state is not None else None
                    g_z, dyn_state = step_grounded(model, event_step, sync_st)
                    curr_z = torch.where((flag_step == 1).unsqueeze(-1), g_z, m1_z)
                    active_models.append("m1")
                else:
                    pred_out = predictor_fn(curr_z, u_step) if u_step is not None else predictor_fn(curr_z)
                    curr_z = pred_out.predicted_latent
                    if dyn_state is not None:
                        dyn_state = DynamicsState(dyn_state.memory_state, curr_z.detach(), dyn_state.ema_spikes)
                    active_models.append("m1")
            else:
                pred_out = predictor_fn(curr_z, u_step) if u_step is not None else predictor_fn(curr_z)
                curr_z = pred_out.predicted_latent
                active_models.append("m1")

            predictions.append(curr_z)

        trajectory = torch.stack(predictions, dim=1)

        decoded = None
        joint_count = J
        if decode_kinematics and hasattr(model, "physical_decoder") and model.physical_decoder is not None:
            decoded = model.physical_decoder(trajectory, initial_filter_state=initial_decoder_state)
            joint_count = model.physical_decoder.joint_count

        sliced_actions = actions[:, :horizon] if (actions is not None and actions.dim() == 3) else actions

        return RolloutResult(
            trajectory=trajectory,
            fast_trajectory=trajectory,
            actions=sliced_actions,
            spikes=None,
            gain_masks=None,
            decoded_kinematics=decoded,
            telemetry=telemetry,
            active_models=active_models,
            joint_count=joint_count,
        )


__all__ = [
    "compute_rollout_schedule",
    "ground_initial_state",
    "rollout_latent",
    "step_grounded",
    "RolloutResult",
    "TelemetryContainer",
]
