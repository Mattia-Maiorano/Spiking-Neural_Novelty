"""
Latent Predictor and Physical Ground-Truth Decoding Heads.
Predicts future latent state p(z_(t+1) | z_t) and decodes physical kinematic variables.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class PredictorOutput:
    """Container for latent prediction output."""
    predicted_latent: torch.Tensor  # [B, ..., latent_dim]
    mean: torch.Tensor
    log_var: Optional[torch.Tensor] = None


class LatentPredictor(nn.Module):
    """
    Predictive Transition Head: p(z_(t+1) | z_t).
    Uses a residual MLP block to predict the temporal differential or next latent state.
    Residual dynamics: z_hat_(t+1) = z_t + delta_z(z_t).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
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
    Strictly requires q_dim + p_dim == latent_dim. No fallback guessing.
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
        if q_dim + p_dim != latent_dim:
            raise ValueError(f"PhysicalDecoder: q_dim ({q_dim}) + p_dim ({p_dim}) != latent_dim ({latent_dim})")

        self.latent_dim = latent_dim
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.num_objects = num_objects
        self.alpha = alpha
        self.out_dim = 4 * num_objects

        self.pos_head = nn.Linear(self.q_dim, 2 * num_objects)
        self.vel_head = nn.Linear(self.p_dim, 2 * num_objects)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"Input latent dimension ({z.shape[-1]}) does not match expected ({self.latent_dim})")

        q = z[..., :self.q_dim]
        p = z[..., self.q_dim:]

        pos_pred = self.pos_head(q)

        if z.dim() == 3:
            B, T, _ = p.shape
            filtered_p = []
            r = torch.zeros(B, self.p_dim, device=z.device, dtype=z.dtype)
            for t in range(T):
                r = self.alpha * r + (1.0 - self.alpha) * p[:, t]
                filtered_p.append(r)
            smooth_p = torch.stack(filtered_p, dim=1)
            vel_pred = self.vel_head(smooth_p)
        else:
            vel_pred = self.vel_head(p)

        return torch.cat([pos_pred, vel_pred], dim=-1)
