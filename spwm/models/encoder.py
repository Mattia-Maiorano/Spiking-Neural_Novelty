"""
Spiking Sensory Event Encoders.
Transforms dense differential event camera frames into temporal spiking feature vectors.
Preserves temporal dimension and supports both sequential and online single-frame step modes.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
from spwm.models.neurons import LIFCell, NeuronState


class EventEncoder(nn.Module):
    """
    Convolutional Spiking Event Encoder.
    Processes [B, 2, H, W] differential polarity frames into spatial feature representations.
    Uses convolutional layers followed by LIF spiking neurons.
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        conv_channels: Tuple[int, int] = (32, 64),
        out_dim: int = 128,
        beta: float = 0.85,
        threshold: float = 1.0,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Conv layer 1: [B, 2, H, W] -> [B, C1, H/2, W/2]
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, stride=2, padding=1)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_name=surrogate_name, surrogate_alpha=surrogate_alpha)

        # Conv layer 2: [B, C1, H/2, W/2] -> [B, C2, H/4, W/4]
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_name=surrogate_name, surrogate_alpha=surrogate_alpha)

        # Spatial pooling / projection to out_dim
        h_out = height // 4
        w_out = width // 4
        flattened_dim = conv_channels[1] * h_out * w_out
        self.fc = nn.Linear(flattened_dim, out_dim)
        self.lif_out = LIFCell(beta=beta, threshold=threshold, surrogate_name=surrogate_name, surrogate_alpha=surrogate_alpha)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[NeuronState, NeuronState, NeuronState]:
        """Initializes internal LIF states of the encoder layers."""
        # Layer 1 shape: [B, C1, H/2, W/2]
        # We initialize dynamically or lazily in forward
        return None  # Managed in forward if state is None

    def step(
        self,
        event_frame: torch.Tensor,
        states: Optional[Tuple[NeuronState, NeuronState, NeuronState]] = None,
    ) -> Tuple[torch.Tensor, Tuple[NeuronState, NeuronState, NeuronState]]:
        """
        Processes single event frame [B, 2, H, W] -> sensory embedding [B, out_dim].
        """
        s1, s2, s3 = states if states is not None else (None, None, None)

        x1 = self.conv1(event_frame)
        spk1, new_s1 = self.lif1(x1, s1)

        x2 = self.conv2(spk1)
        spk2, new_s2 = self.lif2(x2, s2)

        flat = spk2.flatten(start_dim=1)
        x3 = self.fc(flat)
        spk3, new_s3 = self.lif_out(x3, s3)

        return spk3, (new_s1, new_s2, new_s3)

    def forward(
        self,
        event_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes sequence of event frames [B, T, 2, H, W].
        Returns:
            encoded_sequence: [B, T, out_dim]
            mean_spike_rate: scalar tensor
        """
        B, T, C, H, W = event_sequence.shape
        encoded_steps = []
        states = None

        total_spikes = 0
        total_neurons = 0

        for t in range(T):
            frame_t = event_sequence[:, t]
            embedding_t, states = self.step(frame_t, states)
            encoded_steps.append(embedding_t)

            total_spikes += (states[0].spikes.sum() + states[1].spikes.sum() + states[2].spikes.sum())
            total_neurons += (states[0].spikes.numel() + states[1].spikes.numel() + states[2].spikes.numel())

        encoded_seq = torch.stack(encoded_steps, dim=1)  # [B, T, out_dim]
        mean_spike_rate = total_spikes / max(1, total_neurons)

        return encoded_seq, mean_spike_rate
