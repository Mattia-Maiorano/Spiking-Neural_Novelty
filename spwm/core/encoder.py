"""
Continuous Convolutional Event Encoders with Spatial Softmax.
Transforms differential event camera frames into continuous spatial feature vectors
via differentiable Spatial Softmax (Keypoint Bottleneck).
"""

from __future__ import annotations
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """
    Differentiable Spatial Softmax (Spatial Keypoint Bottleneck).
    Computes analytical 2D center of mass (u_k, v_k) in [-1, 1]^2 for each feature channel.
    """

    def __init__(self, height: int = 8, width: int = 8, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.height = height
        self.width = width

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        flat_features = features.view(B * C, H * W) / self.temperature
        softmax_attention = F.softmax(flat_features, dim=-1)

        expected_x = torch.sum(softmax_attention * self.pos_x, dim=-1)
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=-1)

        keypoints = torch.stack([expected_x, expected_y], dim=-1)
        return keypoints.view(B, C * 2)


class EventEncoder(nn.Module):
    """
    Spatial Softmax Keypoint Event Encoder.
    Processes [B, 2, H, W] differential polarity frames into structured keypoint-based representations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv_channels: Tuple[int, int] = (16, 32),
        num_keypoints: int = 16,
        out_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.out_dim = out_dim

        c1, c2 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(4, c1), num_channels=c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=min(4, c2), num_channels=c2),
            nn.SiLU(),
            nn.Conv2d(c2, num_keypoints, kernel_size=1),
        )
        self.spatial_softmax = SpatialSoftmax(height=8, width=8, temperature=temperature)
        self.proj = nn.Sequential(
            nn.Linear(num_keypoints * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )
        self.last_keypoints: Optional[torch.Tensor] = None

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        return None

    def step(
        self,
        event_frame: torch.Tensor,
        states: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, None]:
        feat = self.conv(event_frame)
        kp = self.spatial_softmax(feat)
        self.last_keypoints = kp
        out = self.proj(kp)
        return out, None

    def forward(
        self,
        x: torch.Tensor,
        return_keypoints: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feat = self.conv(x_flat)
            kp = self.spatial_softmax(feat)
            out = self.proj(kp).view(B, T, -1)
            if return_keypoints:
                return out, kp.view(B, T, -1)
            return out
        feat = self.conv(x)
        kp = self.spatial_softmax(feat)
        out = self.proj(kp)
        if return_keypoints:
            return out, kp
        return out
