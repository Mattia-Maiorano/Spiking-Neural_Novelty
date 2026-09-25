"""
Event Camera Simulator for Neuromorphic Vision.
Transforms continuous dynamical trajectories into spatio-temporal event streams.
Supports both dense tensor ([B, T, 2, H, W]) and sparse event list formats.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from spwm.data.synthetic_world import Trajectory


@dataclass
class EventBatch:
    """Represents a batch of events with both dense and sparse accessor methods."""
    dense_events: torch.Tensor  # [B, T, 2, H, W] or [T, 2, H, W]
    sparse_events: Optional[List[List[Tuple[int, int, int, int]]]] = None  # (t, p, y, x)
    trajectory_ids: Optional[torch.Tensor] = None

    @property
    def shape(self) -> torch.Size:
        return self.dense_events.shape


class EventCameraSimulator:
    """
    Simulates a differential neuromorphic event camera sensor.
    Renders object scenes into continuous intensity fields L(t), calculates temporal changes:
        ΔL = L(t) - L(t - Δt)
    and emits binary polarity spikes when |ΔL| > threshold:
        Channel 0: Positive polarity (intensity brightening)
        Channel 1: Negative polarity (intensity dimming)
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        event_threshold: float = 0.08,
        smooth_sigma_ratio: float = 0.8,
    ) -> None:
        self.height = height
        self.width = width
        self.event_threshold = event_threshold
        self.smooth_sigma_ratio = smooth_sigma_ratio

        # Precompute coordinate grid in normalized coordinates [-1.0, 1.0]
        y = np.linspace(-1.0, 1.0, height)
        x = np.linspace(-1.0, 1.0, width)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def render_intensity_frame(self, positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """
        Renders continuous object positions into a 2D intensity field L in [0, 1].
        Uses smooth Gaussian kernels for anti-aliasing and realistic edge gradients.
        positions: [N_objects, 2]
        radii: [N_objects]
        """
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        for i in range(positions.shape[0]):
            cx, cy = positions[i]
            r = radii[i]
            sigma = r * self.smooth_sigma_ratio
            dist_sq = (self.grid_x - cx) ** 2 + (self.grid_y - cy) ** 2
            # Gaussian bell profile capped at 1.0
            obj_field = np.exp(-0.5 * dist_sq / (sigma ** 2 + 1e-8))
            frame = np.maximum(frame, obj_field)
        return frame

    def render_trajectory_intensity(self, trajectory: Trajectory) -> np.ndarray:
        """Renders intensity frames for an entire trajectory: [T, H, W]."""
        T = len(trajectory.states)
        intensity_frames = np.zeros((T, self.height, self.width), dtype=np.float32)
        for t, state in enumerate(trajectory.states):
            intensity_frames[t] = self.render_intensity_frame(state.positions, state.radii)
        return intensity_frames

    def trajectory_to_events(
        self,
        trajectory: Trajectory,
        return_sparse: bool = False,
    ) -> EventBatch:
        """
        Converts a continuous trajectory into neuromorphic event streams.
        Returns dense tensor [T, 2, H, W] (pos/neg channels) and optionally sparse events.
        """
        intensity_frames = self.render_trajectory_intensity(trajectory)  # [T, H, W]
        T = intensity_frames.shape[0]

        # Calculate temporal differential ΔL
        # At t=0, ΔL = 0 (no prior baseline)
        delta_L = np.zeros_like(intensity_frames)
        delta_L[1:] = intensity_frames[1:] - intensity_frames[:-1]

        pos_spikes = (delta_L > self.event_threshold).astype(np.float32)
        neg_spikes = (delta_L < -self.event_threshold).astype(np.float32)

        # Dense tensor [T, 2, H, W]
        dense_events = np.stack([pos_spikes, neg_spikes], axis=1)
        dense_tensor = torch.from_numpy(dense_events).float()

        sparse_list = None
        if return_sparse:
            sparse_list = []
            for t in range(T):
                step_events = []
                # Find non-zero event coordinates
                pos_y, pos_x = np.where(pos_spikes[t] > 0)
                for py, px in zip(pos_y, pos_x):
                    step_events.append((t, 1, int(py), int(px)))
                neg_y, neg_x = np.where(neg_spikes[t] > 0)
                for ny, nx in zip(neg_y, neg_x):
                    step_events.append((t, 0, int(ny), int(nx)))
                sparse_list.append(step_events)

        return EventBatch(
            dense_events=dense_tensor,
            sparse_events=sparse_list,
            trajectory_ids=torch.tensor([trajectory.trajectory_id], dtype=torch.long),
        )

    def dense_to_sparse_tensor(self, dense_events: torch.Tensor) -> torch.Tensor:
        """
        Converts dense events [B, T, 2, H, W] to sparse coordinate tensor [N_total_events, 5]
        containing (batch_idx, t, polarity, y, x).
        Allows seamless transition to event-sparse processing.
        """
        indices = torch.nonzero(dense_events > 0, as_tuple=False)
        return indices
