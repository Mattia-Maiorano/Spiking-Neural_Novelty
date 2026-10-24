"""
Unified Action-Conditioned Autoregressive Latent Rollout Engine (Layer 1 — Models).

Single source of truth for autonomous latent rollout mechanics across
training (M1), validation, and evaluation.
Depends exclusively on Layer 0 (`spwm.contracts`).
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from spwm.contracts.states import RolloutResult, TelemetryContainer
from spwm.contracts.tensors import assert_tensor_contract


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
) -> RolloutResult:
    """Execute an action-conditioned multi-step rollout in latent space starting from z_init.

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

    grad_context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with grad_context:
        predictions: List[torch.Tensor] = []
        active_models: List[str] = []
        curr_z = z_init

        predictor_fn = fixed_predictor if fixed_predictor is not None else model.predictor

        for step in range(horizon):
            if actions is not None:
                u_step = actions[:, step] if actions.dim() == 3 else actions
                pred_out = predictor_fn(curr_z, u_step)
            else:
                pred_out = predictor_fn(curr_z)
            curr_z = pred_out.predicted_latent
            predictions.append(curr_z)
            active_models.append("m1")

        trajectory = torch.stack(predictions, dim=1)

        decoded = None
        joint_count = 8
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
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
