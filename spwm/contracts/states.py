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
    grounded_steps: Optional[torch.Tensor] = None  # [B, 1] count of consecutive grounded steps

    def detach(self) -> "DynamicsState":
        return DynamicsState(
            memory_state=self.memory_state.detach(),
            z_prev=self.z_prev.detach(),
            ema_spikes=self.ema_spikes.detach(),
            grounded_steps=self.grounded_steps.detach() if self.grounded_steps is not None else None,
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
    error_neurons: Optional[torch.Tensor]                 # [B, encoder_dim] (epsilon_t = x_t - W_pred z_(t-1))
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
    error_neurons: Optional[torch.Tensor]          # [B, T, encoder_dim]
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
    joint_count: int = 6

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


@dataclass(frozen=True)
class ClosedLoopTrajectory:
    """Typed closed-loop MPPI or expert trajectory container for DAgger replay."""

    z_init: torch.Tensor
    actions: torch.Tensor
    targets: torch.Tensor
    length: int
    trajectory_id: int
    is_expert: Union[bool, str] = False
    elevation_plane: str = "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain primitive + detached CPU tensor dictionary for safe serialization."""
        return {
            "z_init": self.z_init.detach().cpu().clone(),
            "actions": self.actions.detach().cpu().clone(),
            "targets": self.targets.detach().cpu().clone(),
            "length": int(self.length),
            "trajectory_id": int(self.trajectory_id),
            "is_expert": bool(self.is_expert) if isinstance(self.is_expert, bool) else str(self.is_expert),
            "elevation_plane": str(self.elevation_plane),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        legacy_unannotated: bool = False,
    ) -> "ClosedLoopTrajectory":
        """Construct and validate a ClosedLoopTrajectory from a serialized or caller dictionary."""
        actions_t = data["actions"].detach().cpu()
        targets_t = data["targets"].detach().cpu()
        if actions_t.ndim == 3 and actions_t.shape[1] == 1:
            actions_t = actions_t.squeeze(1)
        if targets_t.ndim == 3 and targets_t.shape[1] == 1:
            targets_t = targets_t.squeeze(1)

        if "is_expert" in data:
            raw_expert = data["is_expert"]
            is_expert: Union[bool, str] = bool(raw_expert) if isinstance(raw_expert, bool) else str(raw_expert)
        else:
            is_expert = "UNANNOTATED_LEGACY_ORIGIN" if legacy_unannotated else False

        if "elevation_plane" in data:
            elevation_plane = str(data["elevation_plane"])
        else:
            elevation_plane = "UNANNOTATED_LEGACY_ELEVATION" if legacy_unannotated else "Unknown"

        return cls(
            z_init=data["z_init"].detach().cpu().view(-1),
            actions=actions_t,
            targets=targets_t,
            length=int(data["length"]),
            trajectory_id=int(data["trajectory_id"]),
            is_expert=is_expert,
            elevation_plane=elevation_plane,
        )

    def __getitem__(self, key: str) -> Any:
        if key == "z_init":
            return self.z_init
        if key == "actions":
            return self.actions
        if key == "targets":
            return self.targets
        if key == "length":
            return self.length
        if key == "trajectory_id":
            return self.trajectory_id
        if key == "is_expert":
            return self.is_expert
        if key == "elevation_plane":
            return self.elevation_plane
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in (
            "z_init",
            "actions",
            "targets",
            "length",
            "trajectory_id",
            "is_expert",
            "elevation_plane",
        )


@dataclass(frozen=True)
class Tier2EvalResult:
    """Typed cross-layer result container for Tier 2 closed-loop MPPI validation."""

    mppi_score: float
    mppi_structural_score: float
    mppi_cost: float
    mppi_success_rate: float
    mppi_mean_error_cm: float
    mppi_grasp_error_cm: float
    mppi_grasp_rot_err_rad: float
    mppi_terminal_error_cm: float
    mppi_steady_state_error_cm: float
    mppi_min_error_cm: float
    mppi_final_vel: float
    mppi_final_rot: float
    mppi_pos_penalty: float
    mppi_vel_penalty: float
    mppi_rot_penalty: float
    mppi_evaluated_trajectories: float
    mppi_expert_trajectories: float
    mppi_expert_blocks: List[str]
    mppi_expert_elevation_counts: Dict[str, int]
    mppi_expert_trajectory_ids: List[int]
    mppi_natural_blocks: List[str]
    mppi_natural_trajectory_ids: List[int]
    mppi_trials: List[Dict[str, Any]]
    mppi_elevation_breakdown: Dict[str, Dict[str, Any]]
    transitions: List[Dict[str, torch.Tensor]] = field(default_factory=list)
    trajectories: List[ClosedLoopTrajectory] = field(default_factory=list)

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Return loggable Tier 2 metrics dictionary (excluding raw replay payloads)."""
        return {
            "mppi_score": float(self.mppi_score),
            "mppi_structural_score": float(self.mppi_structural_score),
            "mppi_cost": float(self.mppi_cost),
            "mppi_success_rate": float(self.mppi_success_rate),
            "success_rate": float(self.mppi_success_rate),
            "mppi_mean_error_cm": float(self.mppi_mean_error_cm),
            "mean_cartesian_err_cm": float(self.mppi_mean_error_cm),
            "mppi_grasp_error_cm": float(self.mppi_grasp_error_cm),
            "grasp_error_cm": float(self.mppi_grasp_error_cm),
            "mppi_grasp_rot_err_rad": float(self.mppi_grasp_rot_err_rad),
            "grasp_rot_err_rad": float(self.mppi_grasp_rot_err_rad),
            "mppi_terminal_error_cm": float(self.mppi_terminal_error_cm),
            "mppi_steady_state_error_cm": float(self.mppi_steady_state_error_cm),
            "mppi_min_error_cm": float(self.mppi_min_error_cm),
            "mppi_final_vel": float(self.mppi_final_vel),
            "mppi_final_rot": float(self.mppi_final_rot),
            "mppi_rot_err_rad": float(self.mppi_final_rot),
            "rotational_err_rad": float(self.mppi_final_rot),
            "mppi_pos_penalty": float(self.mppi_pos_penalty),
            "mppi_vel_penalty": float(self.mppi_vel_penalty),
            "mppi_rot_penalty": float(self.mppi_rot_penalty),
            "mppi_evaluated_trajectories": float(self.mppi_evaluated_trajectories),
            "mppi_expert_trajectories": float(self.mppi_expert_trajectories),
            "mppi_expert_blocks": list(self.mppi_expert_blocks),
            "mppi_expert_elevation_counts": dict(self.mppi_expert_elevation_counts),
            "mppi_expert_trajectory_ids": list(self.mppi_expert_trajectory_ids),
            "mppi_natural_blocks": list(self.mppi_natural_blocks),
            "mppi_natural_trajectory_ids": list(self.mppi_natural_trajectory_ids),
            "mppi_trials": list(self.mppi_trials),
            "mppi_elevation_breakdown": dict(self.mppi_elevation_breakdown),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return full primitive/tensor dictionary representation including replay trajectories."""
        payload = self.to_metrics_dict()
        payload["transitions"] = list(self.transitions)
        payload["trajectories"] = [t.to_dict() for t in self.trajectories]
        return payload

    def _as_view(self) -> Dict[str, Any]:
        payload = self.to_metrics_dict()
        payload["transitions"] = self.transitions
        payload["trajectories"] = self.trajectories
        return payload

    def __getitem__(self, key: str) -> Any:
        if key == "transitions":
            return self.transitions
        if key == "trajectories":
            return self.trajectories
        metrics = self.to_metrics_dict()
        if key in metrics:
            return metrics[key]
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._as_view()

    def keys(self):
        return self._as_view().keys()

    def items(self):
        return self._as_view().items()

    def values(self):
        return self._as_view().values()

    def __iter__(self):
        return iter(self._as_view())

    def __len__(self) -> int:
        return len(self._as_view())
