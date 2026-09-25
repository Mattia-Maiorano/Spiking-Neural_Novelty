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
    Residual dynamics: z_hat_(t+1) = z_t + Δz(z_t).
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
        """
        z: [B, latent_dim] or [B, T, latent_dim]
        Returns: PredictorOutput with predicted_latent [B, ..., latent_dim]
        """
        delta_z = self.net(z)
        if self.residual:
            z_next = z + delta_z
        else:
            z_next = delta_z

        return PredictorOutput(
            predicted_latent=z_next,
            mean=z_next,
            log_var=None,  # Prepared for V4 probabilistic world model
        )


class PhysicalDecoder(nn.Module):
    """
    Physical Ground-Truth Probe.
    Decodes the abstract latent representation z_t into physical kinematic variables
    (x, y, vx, vy) to verify that the latent state captures true environment physics.
    Does NOT backpropagate into latent dynamics when used as a probe.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_objects: int = 1,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.out_dim = 4 * num_objects  # (x, y, vx, vy) per object
        self.probe = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes z [B, ..., latent_dim] -> [B, ..., 4 * num_objects]."""
        return self.probe(z)
