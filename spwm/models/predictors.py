"""
Layer 1 Action-Conditioned Latent Attractor Predictor (M1) and Articulated Joint PhysicalDecoder.
Pure neural modules with strict tensor shape assertions and zero training/loss couplings.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from spwm.contracts.states import PredictorOutput
from spwm.contracts.tensors import assert_phase_space_contract, assert_tensor_contract


class LatentPredictor(nn.Module):
    """
    Action-Conditioned Latent Attractor Predictor (M1):
        z_hat_(t+1) = M_1(z_t, u_t)

    Relaxes strictly conservative symplectic Euler integration to learn both dissipative
    (phase-space volume contracting) and restorative vector fields driven by motor actions u_t:
        h_t = Backbone([z_t, u_t])
        gamma_t = sigmoid(dissipation_head(h_t))
        f_p = force_head(h_t)
        f_q = restorative_q_head(h_t)
        p_(t+1) = (1 - dt * gamma_t) * p_t + f_p
        q_(t+1) = q_t + dt * (p_(t+1)[..., :q_dim] + f_q)
        z_hat_(t+1) = [q_(t+1), p_(t+1)]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        action_dim: int = 8,
        hidden_dim: int = 256,
        residual: bool = True,
        symplectic: bool = True,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        dt: float = 0.05,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0 or action_dim <= 0:
            raise ValueError(
                f"latent_dim ({latent_dim}), action_dim ({action_dim}), and hidden_dim ({hidden_dim}) must be > 0"
            )
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.symplectic = symplectic
        self.dt = float(dt)

        self.q_dim = q_dim if q_dim is not None else (latent_dim // 4)
        self.p_dim = p_dim if p_dim is not None else (latent_dim - self.q_dim)
        if self.q_dim + self.p_dim != latent_dim:
            raise ValueError(f"q_dim ({self.q_dim}) + p_dim ({self.p_dim}) != latent_dim ({latent_dim})")

        in_dim = latent_dim + action_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        if symplectic:
            self.force_head = nn.Linear(hidden_dim, self.p_dim)
            self.dissipation_head = nn.Linear(hidden_dim, self.p_dim)
            self.restorative_q_head = nn.Linear(hidden_dim, self.q_dim)
            nn.init.constant_(self.dissipation_head.bias, -2.0)
            nn.init.zeros_(self.restorative_q_head.weight)
            nn.init.zeros_(self.restorative_q_head.bias)
        else:
            self.net = nn.Sequential(
                self.backbone,
                nn.Linear(hidden_dim, latent_dim),
            )

    def forward(self, z: torch.Tensor, action: Optional[torch.Tensor] = None) -> PredictorOutput:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"LatentPredictor: expected last dimension {self.latent_dim}, got {z.shape[-1]} (shape {tuple(z.shape)})"
            )
        if action is None:
            action = torch.zeros(*z.shape[:-1], self.action_dim, device=z.device, dtype=z.dtype)
        else:
            expected_action_shape = tuple(z.shape[:-1]) + (self.action_dim,)
            assert_tensor_contract(action, expected_action_shape, "LatentPredictor.action")

        zu = torch.cat([z, action], dim=-1)

        if self.symplectic:
            q_t = z[..., : self.q_dim]
            p_t = z[..., self.q_dim :]
            h = self.backbone(zu)

            f_p = self.force_head(h)
            gamma = torch.sigmoid(self.dissipation_head(h))
            f_q = self.restorative_q_head(h)

            p_next = (1.0 - self.dt * gamma) * p_t + f_p
            p_for_q = p_next[..., : self.q_dim] if self.p_dim != self.q_dim else p_next
            q_next = q_t + self.dt * (p_for_q + f_q)
            z_next = torch.cat([q_next, p_next], dim=-1)
        else:
            delta_z = self.net(zu)
            z_next = z + delta_z if self.residual else delta_z

        return PredictorOutput(
            predicted_latent=z_next,
            mean=z_next,
            log_var=None,
        )


class PhysicalDecoder(nn.Module):
    """
    Articulated Robotic Joint Angles & Angular Velocities Decoder.
    Maps from the partitioned latent phase space z_t = [q_t, p_t] back to complex
    3D articulated joint angles q* and angular velocities p*:
      - pos_pred = self.pos_head(q_t)           [B, ..., joint_count]
      - vel_pred = self.vel_head(smooth_p)      [B, ..., joint_count]
      - decoded  = torch.cat([pos_pred, vel_pred], dim=-1)  [B, ..., 2 * joint_count]
    Strictly requires q_dim + p_dim == latent_dim.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        q_dim: int = 32,
        p_dim: int = 96,
        joint_count: int = 8,
        hidden_dim: int = 64,
        alpha: float = 0.85,
    ) -> None:
        super().__init__()
        assert_phase_space_contract(latent_dim=latent_dim, q_dim=q_dim, p_dim=p_dim)
        if joint_count <= 0:
            raise ValueError(f"PhysicalDecoder: joint_count must be > 0, got {joint_count}")

        self.latent_dim = latent_dim
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.joint_count = joint_count
        self.alpha = float(alpha)
        self.out_dim = 2 * joint_count

        self.pos_head = nn.Sequential(
            nn.Linear(self.q_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.joint_count),
        )
        self.vel_head = nn.Sequential(
            nn.Linear(self.p_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.joint_count),
        )

    def decode_pos(self, q: torch.Tensor) -> torch.Tensor:
        """Decodes articulated joint angles q* from coordinate subspace q."""
        return self.pos_head(q)

    def decode_vel(self, p: torch.Tensor) -> torch.Tensor:
        """Decodes articulated joint angular velocities p* from momentum subspace p."""
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


__all__ = [
    "LatentPredictor",
    "PhysicalDecoder",
    "PredictorOutput",
]
