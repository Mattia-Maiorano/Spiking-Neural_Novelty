"""
Unified Autoregressive Latent Rollout Engine (Layer 1 — Models).

Single source of truth for autonomous latent rollout mechanics across
training phases (M1, M2, M3), validation, and evaluation.
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
    with_grad: bool = False,
    decode_kinematics: bool = True,
    initial_decoder_state: Optional[torch.Tensor] = None,
    telemetry: Optional[Union[TelemetryContainer, Dict[str, List[torch.Tensor]]]] = None,
    start_step: int = 0,
    fixed_predictor: Optional[nn.Module] = None,
) -> RolloutResult:
    """Execute an autonomous multi-step rollout in latent space starting from z_init.

    Args:
        model: World model exposing `predictor` (and optionally `physical_decoder`).
        z_init: Initial latent state [B, latent_dim].
        horizon: Number of rollout steps (> 0).
        with_grad: If True, preserves the autograd graph during rollout (used by M1/M2/M3 training).
        decode_kinematics: If True and model has `physical_decoder`, decodes [B, H, 4 * N] kinematics.
        initial_decoder_state: Optional warm EMA filter state [B, p_dim] for `physical_decoder`.
        telemetry: Optional TelemetryContainer or dict for logging step states.
        start_step: Step offset in the CascadedTemporalRelay cycle (default 0).
        fixed_predictor: Optional sub-predictor module (e.g., `model.m1` or `model.m2`) to invoke
            directly instead of cycling through `model.predictor`.

    Returns:
        RolloutResult containing trajectory, decoded_kinematics, active_models, and num_objects.
    """
    if horizon <= 0:
        raise ValueError(f"Rollout horizon must be positive, got {horizon}")
    assert_tensor_contract(z_init, (None, None), "rollout_latent.z_init")

    if fixed_predictor is None and not hasattr(model, "predictor"):
        raise AttributeError("Model must have a 'predictor' attribute when fixed_predictor is not provided.")

    grad_context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with grad_context:
        predictions: List[torch.Tensor] = []
        active_models: List[str] = []
        curr_z = z_init

        for step_offset in range(horizon):
            t = start_step + step_offset
            if fixed_predictor is not None:
                pred_out = fixed_predictor(curr_z)
                active_name = "fixed"
            elif hasattr(model.predictor, "get_active_model"):
                active_mod, active_name, _ = model.predictor.get_active_model(t)
                pred_out = active_mod(curr_z)
            else:
                pred_out = model.predictor(curr_z)
                active_name = "predictor"

            z_next = pred_out.predicted_latent
            curr_z = z_next
            predictions.append(curr_z)
            active_models.append(active_name)

        trajectory = torch.stack(predictions, dim=1)

        decoded = None
        num_objects = 1
        if decode_kinematics and hasattr(model, "physical_decoder") and model.physical_decoder is not None:
            decoded = model.physical_decoder(trajectory, initial_filter_state=initial_decoder_state)
            num_objects = model.physical_decoder.num_objects

        return RolloutResult(
            trajectory=trajectory,
            fast_trajectory=trajectory,
            spikes=None,
            gain_masks=None,
            decoded_kinematics=decoded,
            telemetry=telemetry,
            active_models=active_models,
            num_objects=num_objects,
        )


__all__ = [
    "rollout_latent",
    "RolloutResult",
    "TelemetryContainer",
]
