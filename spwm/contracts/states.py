"""
Layer 0 Immutable State & Output Containers for SPWM.
Single source of truth for all neural state and step/sequence output structures.
Contains zero model logic and zero upward imports.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch


@dataclass
class ALIFState:
    """Biophysical state of an Adaptive Leaky Integrate-and-Fire (ALIF) neuron population."""
    v_mem: torch.Tensor    # [B, size] membrane potential
    a_adapt: torch.Tensor  # [B, size] adaptive threshold variable
    spikes: torch.Tensor   # [B, size] binary emitted spikes {0.0, 1.0}

    def detach(self) -> "ALIFState":
        return ALIFState(
            v_mem=self.v_mem.detach(),
            a_adapt=self.a_adapt.detach(),
            spikes=self.spikes.detach(),
        )


@dataclass
class MultiTimescaleState:
    """State container for multi-timescale ALIF memory populations."""
    population_states: List[ALIFState]

    @property
    def v_mems(self) -> List[torch.Tensor]:
        return [s.v_mem for s in self.population_states]

    @property
    def a_adapts(self) -> List[torch.Tensor]:
        return [s.a_adapt for s in self.population_states]

    @property
    def spikes(self) -> List[torch.Tensor]:
        return [s.spikes for s in self.population_states]

    @property
    def concatenated_spikes(self) -> torch.Tensor:
        """Concatenates spikes across all timescale tiers [B, sum(dim_k)]."""
        return torch.cat(self.spikes, dim=-1)

    @property
    def concatenated_mems(self) -> torch.Tensor:
        """Concatenates membrane potentials across all tiers [B, sum(dim_k)]."""
        return torch.cat(self.v_mems, dim=-1)

    @property
    def fast_spikes(self) -> torch.Tensor:
        return self.spikes[0]

    @property
    def slow_spikes(self) -> torch.Tensor:
        return self.spikes[-1]

    def detach(self) -> "MultiTimescaleState":
        return MultiTimescaleState(
            population_states=[s.detach() for s in self.population_states]
        )


@dataclass
class DynamicsState:
    """Recurrent state container for SpikingLatentDynamics."""
    memory_state: MultiTimescaleState
    z_prev: torch.Tensor      # [B, latent_dim] (concatenated q and p)
    ema_spikes: torch.Tensor  # [B, total_memory_dim] EMA of spikes

    def detach(self) -> "DynamicsState":
        return DynamicsState(
            memory_state=self.memory_state.detach(),
            z_prev=self.z_prev.detach(),
            ema_spikes=self.ema_spikes.detach(),
        )


@dataclass
class DynamicsOutput:
    """Output container for SpikingLatentDynamics sequence forward pass."""
    latent_states: torch.Tensor    # [B, T, latent_dim]
    fast_spikes: torch.Tensor      # [B, T, fast_dim]
    slow_spikes: torch.Tensor      # [B, T, slow_dim]
    fast_mems: torch.Tensor        # [B, T, fast_dim]
    slow_mems: torch.Tensor        # [B, T, slow_dim]
    mean_spike_rate: torch.Tensor  # scalar tensor


@dataclass
class PredictorOutput:
    """Container for action-conditioned latent predictor output."""
    predicted_latent: torch.Tensor  # [B, ..., latent_dim]
    mean: torch.Tensor              # [B, ..., latent_dim]
    log_var: Optional[torch.Tensor] = None


@dataclass
class SPWMState:
    """Complete recurrent state of the SPWM world model, including action tensor u_t."""
    encoder_states: Optional[Any]
    dynamics_state: DynamicsState
    action: Optional[torch.Tensor] = None  # [B, action_dim] (u_t)


@dataclass
class SPWMStepOutput:
    """Output from a single step execution in SPWM."""
    latent: torch.Tensor                        # [B, latent_dim] (z_t)
    predicted_next_latent: torch.Tensor         # [B, latent_dim] (z_hat_(t+1))
    error_neurons: torch.Tensor                 # [B, encoder_dim] (epsilon_t = x_t - W_pred z_(t-1))
    prediction_error: Optional[torch.Tensor]    # [B] if target latent is available
    decoded_kinematics: Optional[torch.Tensor]  # [B, 2 * joint_count] (joint angles q* and angular velocities p*)
    ema_spikes: Optional[torch.Tensor] = None   # [B, total_memory_dim]
    keypoints: Optional[torch.Tensor] = None    # [B, q_dim]
    action: Optional[torch.Tensor] = None       # [B, action_dim] (u_t)


@dataclass
class SPWMSequenceOutput:
    """Output from a full sequence forward pass in SPWM."""
    latent_states: torch.Tensor          # [B, T, latent_dim]
    predicted_latents: torch.Tensor      # [B, T, latent_dim]
    prediction_errors: torch.Tensor      # [B, T-1]
    error_neurons: torch.Tensor          # [B, T, encoder_dim]
    fast_spikes: torch.Tensor            # [B, T, dim_fast]
    slow_spikes: torch.Tensor            # [B, T, dim_slow]
    encoder_spike_rate: torch.Tensor
    dynamics_spike_rate: torch.Tensor
    mean_spike_rate: torch.Tensor
    decoded_kinematics: Optional[torch.Tensor] = None
    keypoints: Optional[torch.Tensor] = None


@dataclass
class TelemetryContainer:
    """Container for logging fine-grained spiking and latent telemetry across rollout horizons."""
    spikes: List[torch.Tensor] = field(default_factory=list)
    v_mem: List[torch.Tensor] = field(default_factory=list)
    v_th: List[torch.Tensor] = field(default_factory=list)
    gain_mask: List[torch.Tensor] = field(default_factory=list)

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
    """Standardized output container for action-conditioned autonomous latent rollouts."""
    trajectory: torch.Tensor                          # [B, H, latent_dim]
    fast_trajectory: torch.Tensor                     # [B, H, latent_dim]
    actions: Optional[torch.Tensor] = None            # [B, H, action_dim] or [B, action_dim]
    spikes: Optional[torch.Tensor] = None
    gain_masks: Optional[torch.Tensor] = None
    decoded_kinematics: Optional[torch.Tensor] = None # [B, H, 2 * joint_count]
    telemetry: Optional[Union[TelemetryContainer, Dict[str, List[torch.Tensor]]]] = None
    active_models: Optional[List[str]] = None
    joint_count: int = 8

    @property
    def positions(self) -> Optional[torch.Tensor]:
        """Decoded joint angles q* in [B, H, joint_count]."""
        if self.decoded_kinematics is None:
            return None
        return self.decoded_kinematics[..., : self.joint_count]

    @property
    def velocities(self) -> Optional[torch.Tensor]:
        """Decoded joint angular velocities p* in [B, H, joint_count]."""
        if self.decoded_kinematics is None:
            return None
        return self.decoded_kinematics[..., self.joint_count : 2 * self.joint_count]
