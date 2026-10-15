"""
Unified Rollout Engine for SPWM.
Single source of truth for autonomous rollout mechanics, burn-in masking,
state advancing, and physical decoding across training, validation, and evaluation.
"""

from __future__ import annotations
import contextlib
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from spwm.core.corrector import SlowCorrectorState


@dataclass
class RolloutResult:
    """Standardized output container for autonomous latent rollouts."""
    trajectory: torch.Tensor                         # [B, H, latent_dim]
    fast_trajectory: torch.Tensor                    # [B, H, latent_dim] (unmodulated proposal)
    spikes: Optional[torch.Tensor] = None            # [B, H, corrector_dim]
    corrector_state: Optional[SlowCorrectorState] = None
    decoded_kinematics: Optional[torch.Tensor] = None # [B, H, 4 * num_objects]

    @property
    def positions(self) -> Optional[torch.Tensor]:
        if self.decoded_kinematics is None:
            return None
        # First 2 coords per object
        return self.decoded_kinematics[..., :2]

    @property
    def velocities(self) -> Optional[torch.Tensor]:
        if self.decoded_kinematics is None:
            return None
        return self.decoded_kinematics[..., 2:4]


def rollout_latent(
    model: nn.Module,
    z_init: torch.Tensor,
    horizon: int,
    use_corrector: bool = True,
    burn_in_steps: int = 0,
    with_grad: bool = False,
    initial_corrector_state: Optional[SlowCorrectorState] = None,
    decode_kinematics: bool = True,
) -> RolloutResult:
    """
    Executes an autonomous multi-step rollout in latent space starting from z_init.
    Unified generator for inference, evaluation, and training.
    """
    if horizon <= 0:
        raise ValueError(f"Rollout horizon must be positive, got {horizon}")

    grad_context = contextlib.nullcontext() if with_grad else torch.no_grad()
    with grad_context:
        predictions = []
        fast_predictions = []
        spikes_list = []
        curr_z = z_init

        corrector = getattr(model, "corrector", None)
        corr_state = initial_corrector_state
        has_active_corrector = use_corrector and corrector is not None

        if has_active_corrector and corr_state is None:
            corr_state = corrector.init_state(
                batch_size=z_init.shape[0],
                device=z_init.device,
                initial_latent=z_init,
            )

        predictor = getattr(model, "predictor", None)
        if predictor is None:
            raise AttributeError("Model must have a 'predictor' module for rollout.")

        for t in range(horizon):
            pred_out = predictor(curr_z)
            z_fast_next = pred_out.predicted_latent
            fast_predictions.append(z_fast_next)

            if has_active_corrector and corr_state is not None:
                is_burn_in = (t < burn_in_steps)
                z_next, corr_out, corr_state = corrector.forward_step(
                    z_prev=curr_z,
                    z_fast_proposed=z_fast_next,
                    state=corr_state,
                    step_idx=t,
                    is_burn_in=is_burn_in,
                )
                spikes_list.append(corr_out.spikes)
            else:
                z_next = z_fast_next

            curr_z = z_next
            predictions.append(curr_z)

        trajectory = torch.stack(predictions, dim=1)
        fast_trajectory = torch.stack(fast_predictions, dim=1)
        spikes = torch.stack(spikes_list, dim=1) if spikes_list else None

        decoded = None
        if decode_kinematics and hasattr(model, "physical_decoder") and model.physical_decoder is not None:
            decoded = model.physical_decoder(trajectory)

        return RolloutResult(
            trajectory=trajectory,
            fast_trajectory=fast_trajectory,
            spikes=spikes,
            corrector_state=corr_state,
            decoded_kinematics=decoded,
        )
