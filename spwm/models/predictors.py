"""
Layer 1 Latent Predictors, Cascaded Temporal Relay (M1, M2, M3), and Physical Kinematic Decoder.
Pure neural modules with strict tensor shape assertions and zero training/loss couplings.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spwm.contracts.states import PredictorOutput
from spwm.contracts.tensors import assert_phase_space_contract, assert_tensor_contract


class LatentPredictor(nn.Module):
    """
    Predictive Transition Head: p(z_(t+1) | z_t).
    Uses a residual MLP block to predict the temporal differential or next latent state:
        z_hat_(t+1) = z_t + delta_z(z_t).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"latent_dim ({latent_dim}) and hidden_dim ({hidden_dim}) must be > 0")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.residual = residual

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> PredictorOutput:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"LatentPredictor: expected last dimension {self.latent_dim}, got {z.shape[-1]} (shape {tuple(z.shape)})"
            )
        delta_z = self.net(z)
        z_next = z + delta_z if self.residual else delta_z
        return PredictorOutput(
            predicted_latent=z_next,
            mean=z_next,
            log_var=None,
        )


class CascadedTemporalRelay(nn.Module):
    """
    Cascaded Temporal Relay of stateless predictors (M1, M2, M3).
    Enables autonomous, self-recovering infinite-horizon rollouts (t -> inf) without catastrophic drift.

    Entities:
      - M1 (Nominal Predictor): High-fidelity physics engine for steps t=0 to phase_horizon - 1.
      - M2 (Recovery Predictor): Shock absorber for drifted states in alternating windows.
      - M3 (Sustained Predictor): Infinite cruiser maintaining predictions from recovered states.

    Cyclic Relay Schedule:
      - step < H: M1 (nominal, steps 0..H-1)
      - For step >= H:
          Let k = step - H
          If (k // H) % 2 == 0: M2
          Else: M3
          Local step in H-step window: k % H
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        residual: bool = True,
        phase_horizon: int = 10,
    ) -> None:
        super().__init__()
        if phase_horizon <= 0:
            raise ValueError(f"CascadedTemporalRelay: phase_horizon must be > 0, got {phase_horizon}")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.phase_horizon = phase_horizon

        self.m1 = LatentPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim, residual=residual)
        self.m2 = LatentPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim, residual=residual)
        self.m3 = LatentPredictor(latent_dim=latent_dim, hidden_dim=hidden_dim, residual=residual)

    def get_active_model(self, step: int) -> Tuple[LatentPredictor, str, int]:
        """
        Determines active model, name, and local step within the dynamic H-step window.
        Uses modular arithmetic so the system operates autonomously across infinite horizons.
        """
        if step < 0:
            raise ValueError(f"Rollout step index must be >= 0, got {step}")
        H = self.phase_horizon
        if step < H:
            return self.m1, "m1", step
        k = step - H
        cycle_idx = k // H
        step_in_window = k % H
        if cycle_idx % 2 == 0:
            return self.m2, "m2", step_in_window
        else:
            return self.m3, "m3", step_in_window

    def forward(self, z: torch.Tensor, step: Optional[int] = None) -> PredictorOutput:
        """
        Predicts next latent state. If step is provided, uses the Cyclic Relay schedule.
        If step is None, uses nominal predictor M1.
        """
        if step is None:
            return self.m1(z)
        model, _, _ = self.get_active_model(step)
        return model(z)

    def init_m3_from_m1(self) -> None:
        """Initializes M3 using a strict copy of M1's pre-trained weights."""
        self.m3.load_state_dict(self.m1.state_dict(), strict=True)


class PhysicalDecoder(nn.Module):
    """
    Diagnostic Linear Probe for Physical Kinematics (x, y, vx, vy).
      - pos_pred = self.pos_head(q_t)           [B, ..., 2 * num_objects]
      - vel_pred = self.vel_head(smooth_p)      [B, ..., 2 * num_objects]
      - decoded  = torch.cat([pos_pred, vel_pred], dim=-1)
    Strictly requires q_dim + p_dim == latent_dim.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        q_dim: int = 32,
        p_dim: int = 96,
        num_objects: int = 1,
        alpha: float = 0.85,
    ) -> None:
        super().__init__()
        assert_phase_space_contract(latent_dim=latent_dim, q_dim=q_dim, p_dim=p_dim)
        if num_objects <= 0:
            raise ValueError(f"PhysicalDecoder: num_objects must be > 0, got {num_objects}")

        self.latent_dim = latent_dim
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.num_objects = num_objects
        self.alpha = float(alpha)
        self.out_dim = 4 * num_objects

        self.pos_head = nn.Linear(self.q_dim, 2 * num_objects)
        self.vel_head = nn.Linear(self.p_dim, 2 * num_objects)

    def decode_pos(self, q: torch.Tensor) -> torch.Tensor:
        """Decodes position kinematics from coordinate subspace q."""
        return self.pos_head(q)

    def decode_vel(self, p: torch.Tensor) -> torch.Tensor:
        """Decodes velocity kinematics from momentum subspace p."""
        return self.vel_head(p)

    def warm_up_filter(self, z_prefix: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Runs the EMA momentum filter across preceding latent states z_prefix [B, T_pre, D]
        to compute the warm filter state r [B, p_dim].
        """
        if z_prefix.numel() == 0:
            if initial_state is not None:
                return initial_state
            B = z_prefix.shape[0] if z_prefix.dim() >= 1 else 1
            return torch.zeros(B, self.p_dim, device=z_prefix.device, dtype=z_prefix.dtype)

        if z_prefix.shape[-1] != self.latent_dim:
            raise ValueError(
                f"PhysicalDecoder.warm_up_filter: expected latent_dim {self.latent_dim}, got {z_prefix.shape[-1]}"
            )

        p = z_prefix[..., self.q_dim :]
        if p.dim() == 2:
            p = p.unsqueeze(1)

        B, T, _ = p.shape
        r = initial_state if initial_state is not None else torch.zeros(B, self.p_dim, device=p.device, dtype=p.dtype)
        for t in range(T):
            r = self.alpha * r + (1.0 - self.alpha) * p[:, t]
        return r

    def forward(self, z: torch.Tensor, initial_filter_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z.dim() == 3:
            assert_tensor_contract(z, (None, None, self.latent_dim), "PhysicalDecoder.z_sequence")
        elif z.dim() == 2:
            assert_tensor_contract(z, (None, self.latent_dim), "PhysicalDecoder.z_step")
        else:
            raise ValueError(f"PhysicalDecoder: expected 2D or 3D latent tensor, got {z.dim()}D")

        q = z[..., : self.q_dim]
        p = z[..., self.q_dim :]

        pos_pred = self.pos_head(q)

        if z.dim() == 3:
            B, T, _ = p.shape
            filtered_p = []
            r = initial_filter_state if initial_filter_state is not None else torch.zeros(B, self.p_dim, device=z.device, dtype=z.dtype)
            for t in range(T):
                r = self.alpha * r + (1.0 - self.alpha) * p[:, t]
                filtered_p.append(r)
            smooth_p = torch.stack(filtered_p, dim=1)
            vel_pred = self.vel_head(smooth_p)
        else:
            if initial_filter_state is not None:
                r = self.alpha * initial_filter_state + (1.0 - self.alpha) * p
                vel_pred = self.vel_head(r)
            else:
                vel_pred = self.vel_head(p)

        return torch.cat([pos_pred, vel_pred], dim=-1)


def extrapolate_latent(
    z_anchor: torch.Tensor,
    t: int,
    t_anchor: int = 10,
    z_prev: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linearly extrapolate a latent state from z_anchor at step t_anchor to step t >= t_anchor.

    Formula:
        Extrapolate(z_anchor, t) = z_anchor + (t - t_anchor) * (z_anchor - z_prev)
    """
    assert_tensor_contract(z_anchor, (None, None), "extrapolate_latent.z_anchor")
    dt = float(t - t_anchor)
    if z_prev is None or dt == 0.0:
        return z_anchor
    assert_tensor_contract(z_prev, tuple(z_anchor.shape), "extrapolate_latent.z_prev")
    delta_z = z_anchor - z_prev
    return z_anchor + dt * delta_z


def compute_convergence_funnel_target(
    z_anchor: torch.Tensor,
    z_prev: Optional[torch.Tensor],
    z_ground_truth: torch.Tensor,
    t_anchor: int = 10,
    phase_horizon: Optional[int] = None,
) -> torch.Tensor:
    r"""Compute the interpolated Convergence Funnel target sequence for M2 recovery.

    For step index k in [0, H-1] (corresponding to t = t_anchor + k + 1):
        \tau_k = (k + 1) / H \in (0, 1]
        \tilde{z}^*_t = (1 - \tau_k) \cdot \text{Extrapolate}(z_{\text{anchor}}, t) + \tau_k \cdot z^*_t
    """
    assert_tensor_contract(z_anchor, (None, None), "compute_convergence_funnel_target.z_anchor")
    assert_tensor_contract(
        z_ground_truth,
        (z_anchor.shape[0], None, z_anchor.shape[1]),
        "compute_convergence_funnel_target.z_ground_truth",
    )
    _, H_seq, _ = z_ground_truth.shape
    denom = float(phase_horizon) if phase_horizon is not None else float(max(H_seq, 1))

    blended_steps = []
    for idx in range(H_seq):
        t = t_anchor + idx + 1
        tau = min(1.0, max(0.0, float(idx + 1) / denom))
        z_ext = extrapolate_latent(z_anchor=z_anchor, t=t, t_anchor=t_anchor, z_prev=z_prev)
        z_target = (1.0 - tau) * z_ext + tau * z_ground_truth[:, idx]
        blended_steps.append(z_target)

    return torch.stack(blended_steps, dim=1)


__all__ = [
    "LatentPredictor",
    "CascadedTemporalRelay",
    "PhysicalDecoder",
    "PredictorOutput",
    "extrapolate_latent",
    "compute_convergence_funnel_target",
]


