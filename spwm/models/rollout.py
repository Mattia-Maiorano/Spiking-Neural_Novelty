"""
Unified Rollout Engine for SPWM.
Single source of truth for autonomous rollout mechanics, burn-in masking,
state advancing, and physical decoding across training, validation, and evaluation.
"""

from __future__ import annotations
import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from spwm.core.corrector import SlowCorrectorState


@dataclass
class TelemetryContainer:
    """Container for logging fine-grained ALIF and corrector states across rollout horizons."""
    spikes: List[torch.Tensor] = field(default_factory=list)      # (B, corrector_dim)
    v_mem: List[torch.Tensor] = field(default_factory=list)       # (B, corrector_dim)
    v_th: List[torch.Tensor] = field(default_factory=list)        # (B, corrector_dim)
    gain_mask: List[torch.Tensor] = field(default_factory=list)   # (B, latent_dim)

    def append(
        self,
        spikes: torch.Tensor,
        v_mem: torch.Tensor,
        v_th: torch.Tensor,
        gain_mask: torch.Tensor,
    ) -> None:
        self.spikes.append(spikes.detach().cpu())
        self.v_mem.append(v_mem.detach().cpu())
        self.v_th.append(v_th.detach().cpu())
        self.gain_mask.append(gain_mask.detach().cpu())

    def as_stacked_tensors(self) -> Dict[str, torch.Tensor]:
        """Stacks recorded timesteps into [H, B, D] tensors."""
        return {
            "spikes": torch.stack(self.spikes, dim=0) if self.spikes else torch.empty(0),
            "v_mem": torch.stack(self.v_mem, dim=0) if self.v_mem else torch.empty(0),
            "v_th": torch.stack(self.v_th, dim=0) if self.v_th else torch.empty(0),
            "gain_mask": torch.stack(self.gain_mask, dim=0) if self.gain_mask else torch.empty(0),
        }


@dataclass
class RolloutResult:
    """Standardized output container for autonomous latent rollouts."""
    trajectory: torch.Tensor                         # [B, H, latent_dim]
    fast_trajectory: torch.Tensor                    # [B, H, latent_dim] (unmodulated proposal)
    spikes: Optional[torch.Tensor] = None            # [B, H, corrector_dim]
    gain_masks: Optional[torch.Tensor] = None        # [B, H, latent_dim]
    corrector_state: Optional[SlowCorrectorState] = None
    decoded_kinematics: Optional[torch.Tensor] = None # [B, H, 4 * num_objects]
    telemetry: Optional[Union[TelemetryContainer, Dict[str, List[torch.Tensor]]]] = None

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
    initial_decoder_state: Optional[torch.Tensor] = None,
    telemetry: Optional[Union[TelemetryContainer, Dict[str, List[torch.Tensor]]]] = None,
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
        gain_masks_list = []
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
                    horizon=horizon,
                )
                spikes_list.append(corr_out.spikes)
                gain_masks_list.append(corr_out.gain_mask)

                if telemetry is not None:
                    # Adaptive threshold: V_th,t = V_th0 + gamma * A_t (if ALIF-based)
                    if hasattr(corrector, "alif") and corr_state.alif_state is not None:
                        alif = corrector.alif
                        v_th_val = alif.v_th0 + alif.gamma * corr_state.alif_state.a_adapt
                    else:
                        v_th_val = torch.zeros_like(corr_out.v_mem)

                    if isinstance(telemetry, TelemetryContainer):
                        telemetry.append(
                            spikes=corr_out.spikes,
                            v_mem=corr_out.v_mem,
                            v_th=v_th_val,
                            gain_mask=corr_out.gain_mask,
                        )
                    elif isinstance(telemetry, dict):
                        telemetry.setdefault("spikes", []).append(corr_out.spikes.detach().cpu())
                        telemetry.setdefault("v_mem", []).append(corr_out.v_mem.detach().cpu())
                        telemetry.setdefault("v_th", []).append(v_th_val.detach().cpu())
                        telemetry.setdefault("gain_mask", []).append(corr_out.gain_mask.detach().cpu())
            else:
                z_next = z_fast_next

            curr_z = z_next
            predictions.append(curr_z)

        trajectory = torch.stack(predictions, dim=1)
        fast_trajectory = torch.stack(fast_predictions, dim=1)
        spikes = torch.stack(spikes_list, dim=1) if spikes_list else None
        gain_masks = torch.stack(gain_masks_list, dim=1) if gain_masks_list else None

        decoded = None
        if decode_kinematics and hasattr(model, "physical_decoder") and model.physical_decoder is not None:
            decoded = model.physical_decoder(trajectory, initial_filter_state=initial_decoder_state)

        return RolloutResult(
            trajectory=trajectory,
            fast_trajectory=fast_trajectory,
            spikes=spikes,
            gain_masks=gain_masks,
            corrector_state=corr_state,
            decoded_kinematics=decoded,
            telemetry=telemetry,
        )
