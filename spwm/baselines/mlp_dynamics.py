"""
Baseline 1: Feed-Forward MLP Dynamics World Model.
Maps observation embeddings to latent states and predicts next state with feed-forward MLP.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from spwm.models.predictor import LatentPredictor, PhysicalDecoder


@dataclass
class BaselineOutput:
    latent_states: torch.Tensor  # [B, T, latent_dim]
    predicted_latents: torch.Tensor  # [B, T, latent_dim]
    prediction_errors: torch.Tensor  # [B, T-1]
    decoded_kinematics: Optional[torch.Tensor] = None
    mean_spike_rate: torch.Tensor = torch.tensor(0.0)


class MLPDynamicsWorldModel(nn.Module):
    """
    Feed-Forward ANN Baseline.
    Encodes event frames with CNN and predicts next latent state using an MLP.
    No recurrent memory.
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        latent_dim: int = 128,
        num_objects: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Standard non-spiking CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (height // 4) * (width // 4), latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Feed-forward dynamics predictor
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=256, residual=True)
        self.physical_decoder = PhysicalDecoder(latent_dim=latent_dim, num_objects=num_objects)

    def forward(self, event_sequence: torch.Tensor) -> BaselineOutput:
        B, T, C, H, W = event_sequence.shape

        # Flatten sequence into batch for CNN encoder
        flat_events = event_sequence.view(B * T, C, H, W)
        latents_flat = self.encoder(flat_events)
        latent_states = latents_flat.view(B, T, self.latent_dim)

        # Predict next latent state
        pred_out = self.predictor(latent_states)
        predicted_latents = pred_out.predicted_latent

        # Compute prediction errors
        z_targets = latent_states[:, 1:].detach()
        z_preds = predicted_latents[:, :-1]
        pred_errors = torch.norm(z_preds - z_targets, dim=-1)

        decoded_kinematics = self.physical_decoder(latent_states)

        return BaselineOutput(
            latent_states=latent_states,
            predicted_latents=predicted_latents,
            prediction_errors=pred_errors,
            decoded_kinematics=decoded_kinematics,
            mean_spike_rate=torch.tensor(0.0, device=event_sequence.device),
        )

    def predict_future(self, initial_latent: torch.Tensor, horizon: int = 50) -> torch.Tensor:
        """Autonomous rollout in latent space."""
        preds = []
        curr_z = initial_latent
        for _ in range(horizon):
            pred_out = self.predictor(curr_z)
            curr_z = pred_out.predicted_latent
            preds.append(curr_z)
        return torch.stack(preds, dim=1)
