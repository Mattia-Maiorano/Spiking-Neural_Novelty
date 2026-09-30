"""
Baseline 3: Vanilla Single-Timescale Recurrent SNN.
Recurrent Spiking Neural Network without multi-timescale memory separation.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from spwm.models.world_model import SPWM, SPWMSequenceOutput


class VanillaRecurrentSNN(nn.Module):
    """
    Vanilla Recurrent Spiking Neural Network baseline.
    Uses identical architecture and parameter budget as SPWM,
    but all recurrent neurons operate on a single homogeneous timescale (no fast/slow distinction).
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        encoder_dim: int = 128,
        latent_dim: int = 128,
        beta: float = 0.85,
        threshold: float = 1.0,
        surrogate_name: str = "atan",
        num_objects: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Configure underlying SPWM with single timescale
        self.snn = SPWM(
            in_channels=in_channels,
            height=height,
            width=width,
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            timescale_dims=(latent_dim // 2, latent_dim // 2),
            betas=(beta, beta),  # Homogeneous single timescale
            threshold=threshold,
            surrogate_name=surrogate_name,
            num_objects=num_objects,
        )

    def forward(self, event_sequence: torch.Tensor) -> SPWMSequenceOutput:
        return self.snn(event_sequence)

    def predict_future(self, initial_latent: torch.Tensor, horizon: int = 50) -> torch.Tensor:
        return self.snn.predict_future(initial_latent, horizon=horizon)

    @property
    def predictor(self):
        return self.snn.predictor

    @property
    def physical_decoder(self):
        return self.snn.physical_decoder
