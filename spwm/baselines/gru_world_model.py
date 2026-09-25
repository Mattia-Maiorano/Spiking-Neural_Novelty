"""
Baseline 2: GRU Recurrent World Model (Continuous ANN).
Standard recurrent baseline using continuous activations and Gated Recurrent Units.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
from spwm.baselines.mlp_dynamics import BaselineOutput
from spwm.models.predictor import LatentPredictor, PhysicalDecoder


class GRUWorldModel(nn.Module):
    """
    Continuous Recurrent ANN World Model.
    Encoder (CNN) -> GRU Recurrent Dynamics -> Predictor -> Future States.
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        encoder_dim: int = 128,
        latent_dim: int = 128,
        num_objects: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Non-spiking CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (height // 4) * (width // 4), encoder_dim),
            nn.ReLU(),
        )

        # Continuous GRU dynamics
        self.gru = nn.GRU(input_size=encoder_dim, hidden_size=latent_dim, batch_first=True)

        # Predictor and probe
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden_dim=256, residual=True)
        self.physical_decoder = PhysicalDecoder(latent_dim=latent_dim, num_objects=num_objects)

    def forward(self, event_sequence: torch.Tensor) -> BaselineOutput:
        B, T, C, H, W = event_sequence.shape

        flat_events = event_sequence.view(B * T, C, H, W)
        encodings = self.encoder(flat_events).view(B, T, -1)

        # Unroll GRU
        latent_states, _ = self.gru(encodings)  # [B, T, latent_dim]

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
        """Autonomous rollout in continuous latent space."""
        preds = []
        curr_z = initial_latent
        for _ in range(horizon):
            pred_out = self.predictor(curr_z)
            curr_z = pred_out.predicted_latent
            preds.append(curr_z)
        return torch.stack(preds, dim=1)
