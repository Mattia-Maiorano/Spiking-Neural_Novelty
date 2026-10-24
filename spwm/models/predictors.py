"""
Layer 1 Symplectic Latent Predictor (M1) and Physical Kinematic Decoder.
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
    Can run in standard residual mode:
        z_hat_(t+1) = z_t + delta_z(z_t)
    or in symplectic semi-implicit Euler mode (z_t = [q_t, p_t]):
        delta_p_t = MLP(z_t)
        p_(t+1) = p_t + delta_p_t
        q_(t+1) = q_t + dt * p_(t+1)
        z_(t+1) = [q_(t+1), p_(t+1)]
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        residual: bool = True,
        symplectic: bool = False,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        dt: float = 0.05,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"latent_dim ({latent_dim}) and hidden_dim ({hidden_dim}) must be > 0")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.symplectic = symplectic
        self.dt = float(dt)

        if symplectic:
            self.q_dim = q_dim if q_dim is not None else (latent_dim // 4)
            self.p_dim = p_dim if p_dim is not None else (latent_dim - self.q_dim)
            if self.q_dim + self.p_dim != latent_dim:
                raise ValueError(f"q_dim ({self.q_dim}) + p_dim ({self.p_dim}) != latent_dim ({latent_dim})")
            out_dim = self.p_dim
        else:
            self.q_dim = q_dim or (latent_dim // 4)
            self.p_dim = p_dim or (latent_dim - self.q_dim)
            out_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> PredictorOutput:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"LatentPredictor: expected last dimension {self.latent_dim}, got {z.shape[-1]} (shape {tuple(z.shape)})"
            )
        if self.symplectic:
            q_t = z[..., : self.q_dim]
            p_t = z[..., self.q_dim :]
            delta_p = self.net(z)
            p_next = p_t + delta_p
            p_for_q = p_next[..., : self.q_dim] if self.p_dim != self.q_dim else p_next
            q_next = q_t + self.dt * p_for_q
            z_next = torch.cat([q_next, p_next], dim=-1)
        else:
            delta_z = self.net(z)
            z_next = z + delta_z if self.residual else delta_z

        return PredictorOutput(
            predicted_latent=z_next,
            mean=z_next,
            log_var=None,
        )


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


class ManifoldProjector(nn.Module):
    """
    Stateless Manifold Projector (Geometric Denoiser) - M2.

    Takes an off-manifold latent state z_noisy (produced by autoregressive M1 rollouts
    or injected perturbations) and projects it onto the closest valid ground-truth
    phase space manifold z_clean, without advancing the time index t.

    Forward logic:
        z_projected = z + self.net(z)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        bottleneck_dim: Optional[int] = None,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"latent_dim ({latent_dim}) and hidden_dim ({hidden_dim}) must be > 0")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.q_dim = q_dim if q_dim is not None else (latent_dim // 4)
        self.p_dim = p_dim if p_dim is not None else (latent_dim - self.q_dim)

        if self.q_dim + self.p_dim != latent_dim:
            raise ValueError(f"q_dim ({self.q_dim}) + p_dim ({self.p_dim}) != latent_dim ({latent_dim})")

        self.bottleneck_dim = bottleneck_dim if bottleneck_dim is not None else max(16, latent_dim // 2)

        # Bottleneck / residual MLP (autoencoder-style projection or residual denoiser)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.SiLU(),
            nn.Linear(self.bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Projects off-manifold latent state z onto the manifold without advancing time."""
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"ManifoldProjector: expected last dimension {self.latent_dim}, got {z.shape[-1]} (shape {tuple(z.shape)})"
            )
        z_projected = z + self.net(z)
        return z_projected


M2Projector = ManifoldProjector
LatentDenoiser = ManifoldProjector


__all__ = [
    "LatentPredictor",
    "PhysicalDecoder",
    "PredictorOutput",
    "ManifoldProjector",
    "M2Projector",
    "LatentDenoiser",
]


