"""
Layer 1 Continuous Convolutional Event Encoder with Spatial Softmax.
Transforms differential event camera frames into continuous spatial feature vectors
and analytical 2D centers of mass (keypoints) in [-1, 1]^2.
Zero mutable side-effect attributes (`last_keypoints` removed).
"""

from __future__ import annotations
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.tensors import assert_tensor_contract, assert_euclidean_coordinates


class SpatialSoftmax(nn.Module):
    """
    Differentiable Spatial Softmax (Spatial Keypoint Bottleneck).
    Computes analytical 2D center of mass (u_k, v_k) in [-1, 1]^2 for each feature channel.
    """

    def __init__(self, height: int = 8, width: int = 8, temperature: float = 1.0) -> None:
        super().__init__()
        if height <= 0 or width <= 0:
            raise ValueError(f"SpatialSoftmax height ({height}) and width ({width}) must be > 0")
        if temperature <= 0.0:
            raise ValueError(f"SpatialSoftmax temperature ({temperature}) must be > 0")
        self.temperature = float(temperature)
        self.height = int(height)
        self.width = int(width)

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.width),
            torch.linspace(-1.0, 1.0, self.height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert_tensor_contract(
            features,
            (None, None, self.height, self.width),
            "SpatialSoftmax.features",
        )
        B, C, H, W = features.shape
        flat_features = features.reshape(B * C, H * W) / self.temperature
        softmax_attention = F.softmax(flat_features, dim=-1)

        expected_x = torch.sum(softmax_attention * self.pos_x, dim=-1)
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=-1)

        keypoints = torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)
        assert_euclidean_coordinates(keypoints, "SpatialSoftmax.keypoints")
        return keypoints


class EventEncoder(nn.Module):
    """
    Spatial Softmax Keypoint Event Encoder.
    Processes [B, C, H, W] or [B, T, C, H, W] differential polarity frames into
    structured visual features [..., out_dim] and topological keypoints [..., 2 * num_keypoints].
    """

    def __init__(
        self,
        in_channels: int = 2,
        conv_channels: Tuple[int, int] = (16, 32),
        num_keypoints: int = 16,
        out_dim: int = 128,
        input_height: int = 32,
        input_width: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_height <= 0 or input_height % 4 != 0:
            raise ValueError(f"EventEncoder input_height ({input_height}) must be positive and divisible by 4")
        if input_width <= 0 or input_width % 4 != 0:
            raise ValueError(f"EventEncoder input_width ({input_width}) must be positive and divisible by 4")
        if len(conv_channels) != 2:
            raise ValueError(f"EventEncoder conv_channels must have length 2, got {conv_channels}")

        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.out_dim = out_dim
        self.input_height = input_height
        self.input_width = input_width

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
        grid_h = input_height // 4
        grid_w = input_width // 4
        self.spatial_softmax = SpatialSoftmax(height=grid_h, width=grid_w, temperature=temperature)
        self.proj = nn.Sequential(
            nn.Linear(num_keypoints * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def encode_step(self, event_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pure single-step encoding:
        Input:  event_frame [B, in_channels, input_height, input_width]
        Output: (features [B, out_dim], keypoints [B, 2 * num_keypoints])
        """
        assert_tensor_contract(
            event_frame,
            (None, self.in_channels, self.input_height, self.input_width),
            "EventEncoder.event_frame",
        )
        feat = self.conv(event_frame)
        kp = self.spatial_softmax(feat)
        out = self.proj(kp)
        return out, kp

    def forward(
        self,
        x: torch.Tensor,
        return_keypoints: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() == 5:
            assert_tensor_contract(
                x,
                (None, None, self.in_channels, self.input_height, self.input_width),
                "EventEncoder.sequence_input",
            )
            B, T, C, H, W = x.shape
            x_flat = x.reshape(B * T, C, H, W)
            out_flat, kp_flat = self.encode_step(x_flat)
            out = out_flat.reshape(B, T, self.out_dim)
            if return_keypoints:
                return out, kp_flat.reshape(B, T, self.num_keypoints * 2)
            return out

        out, kp = self.encode_step(x)
        if return_keypoints:
            return out, kp
        return out
