"""
Event Camera Simulator & Neuromorphic Vision Bridge for 3D Articulated Robotics.
Transforms exported Unity grayscale image sequences or 3D articulated robotic joint trajectories
into spatio-temporal neuromorphic event streams ([B, T, 2, H, W]).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch


@dataclass
class RoboticTrajectory:
    """Ground-truth 3D articulated robotic trajectory exported from Unity or synthesized."""
    trajectory_id: int
    joint_positions: np.ndarray   # [T, joint_count] (q^*)
    joint_velocities: np.ndarray  # [T, joint_count] (p^*)
    actions: np.ndarray           # [T, action_dim] (u_t)
    times: np.ndarray             # [T]
    joint_count: int = 8
    action_dim: int = 8
    grayscale_frames: Optional[np.ndarray] = None  # [T, H, W]
    dense_events: Optional[np.ndarray] = None      # [T, 2, H, W]

    @property
    def positions(self) -> np.ndarray:
        """Joint positions tensor [T, joint_count] (q^*)."""
        return self.joint_positions

    @property
    def velocities(self) -> np.ndarray:
        """Joint angular velocities tensor [T, joint_count] (p^*)."""
        return self.joint_velocities

    @property
    def flat_kinematics(self) -> np.ndarray:
        """Concatenated [joint_positions, joint_velocities] array [T, 2 * joint_count]."""
        return np.concatenate([self.joint_positions, self.joint_velocities], axis=-1)


Trajectory = RoboticTrajectory


@dataclass
class EventBatch:
    """Represents a batch of neuromorphic events with both dense and sparse accessor methods."""
    dense_events: torch.Tensor  # [B, T, 2, H, W] or [T, 2, H, W]
    sparse_events: Optional[List[List[Tuple[int, int, int, int]]]] = None  # (t, p, y, x)
    trajectory_ids: Optional[torch.Tensor] = None

    @property
    def shape(self) -> torch.Size:
        return self.dense_events.shape


class EventCameraSimulator:
    """
    Simulates a differential neuromorphic event camera sensor bridging Unity 3D robotics to SNNs.
    Converts high-FPS grayscale luminance sequences or 3D articulated forward-kinematic projections
    into log-intensity fields L(t), calculates temporal changes:
        ΔL = L(t) - L(t - Δt)
    and emits binary polarity spikes when |ΔL| > threshold (default θ = 0.08):
        Channel 0: Positive polarity (intensity brightening)
        Channel 1: Negative polarity (intensity dimming)
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        event_threshold: float = 0.08,
        link_radius: float = 0.10,
    ) -> None:
        self.height = height
        self.width = width
        self.event_threshold = event_threshold
        self.link_radius = link_radius

        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def render_articulated_frame(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Projects an articulated kinematic chain with joint angles q^* in R^J onto the 2D sensor plane [-1, 1]^2
        via forward kinematics of two bilateral limb chains and renders anti-aliased Gaussian link profiles.
        """
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        dof = int(joint_positions.shape[0])
        half_dof = max(1, dof // 2)
        sigma_sq = (self.link_radius ** 2) + 1e-8

        # Render two bilateral kinematic chains (left/right limbs anchored near pelvis)
        for chain_idx, base_x in enumerate((-0.22, 0.22)):
            curr_x, curr_y = float(base_x), 0.05
            cum_angle = -0.5 * np.pi
            start_j = 0 if chain_idx == 0 else half_dof
            end_j = half_dof if chain_idx == 0 else dof
            n_links = max(1, end_j - start_j)
            link_len = 0.68 / float(n_links)

            for j in range(start_j, end_j):
                cum_angle = cum_angle + float(joint_positions[j])
                next_x = np.clip(curr_x + link_len * np.cos(cum_angle), -0.92, 0.92)
                next_y = np.clip(curr_y + link_len * np.sin(cum_angle), -0.92, 0.92)
                mid_x = 0.5 * (curr_x + next_x)
                mid_y = 0.5 * (curr_y + next_y)

                dist_sq_joint = (self.grid_x - next_x) ** 2 + (self.grid_y - next_y) ** 2
                dist_sq_mid = (self.grid_x - mid_x) ** 2 + (self.grid_y - mid_y) ** 2
                blob = np.maximum(
                    np.exp(-0.5 * dist_sq_joint / sigma_sq),
                    0.85 * np.exp(-0.5 * dist_sq_mid / sigma_sq),
                )
                frame = np.maximum(frame, blob.astype(np.float32))
                curr_x, curr_y = float(next_x), float(next_y)

        return frame

    def render_trajectory_intensity(self, trajectory: RoboticTrajectory) -> np.ndarray:
        """Returns intensity frames [T, H, W] from Unity grayscale export or forward-kinematic projection."""
        if trajectory.grayscale_frames is not None:
            return trajectory.grayscale_frames.astype(np.float32)
        T = trajectory.joint_positions.shape[0]
        intensity_frames = np.zeros((T, self.height, self.width), dtype=np.float32)
        for t in range(T):
            intensity_frames[t] = self.render_articulated_frame(trajectory.joint_positions[t])
        return intensity_frames

    def frames_to_events(
        self,
        intensity_frames: np.ndarray,
        trajectory_id: int = 0,
        return_sparse: bool = False,
    ) -> EventBatch:
        """
        Computes temporal log-intensity differencing across [T, H, W] frames and thresholds at θ = 0.08.
        """
        T = intensity_frames.shape[0]
        delta_L = np.zeros_like(intensity_frames, dtype=np.float32)
        delta_L[1:] = intensity_frames[1:] - intensity_frames[:-1]

        pos_spikes = (delta_L > self.event_threshold).astype(np.float32)
        neg_spikes = (delta_L < -self.event_threshold).astype(np.float32)

        dense_events = np.stack([pos_spikes, neg_spikes], axis=1)
        dense_tensor = torch.from_numpy(dense_events).float()

        sparse_list = None
        if return_sparse:
            sparse_list = []
            for t in range(T):
                step_events = []
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
            trajectory_ids=torch.tensor([trajectory_id], dtype=torch.long),
        )

    def trajectory_to_events(
        self,
        trajectory: RoboticTrajectory,
        return_sparse: bool = False,
    ) -> EventBatch:
        """
        Converts a RoboticTrajectory into neuromorphic event streams [T, 2, H, W].
        Uses pre-computed Unity ComputeShader events if present, else bridges grayscale/kinematic frames.
        """
        if trajectory.dense_events is not None and not return_sparse:
            return EventBatch(
                dense_events=torch.from_numpy(trajectory.dense_events).float(),
                sparse_events=None,
                trajectory_ids=torch.tensor([trajectory.trajectory_id], dtype=torch.long),
            )
        intensity_frames = self.render_trajectory_intensity(trajectory)
        return self.frames_to_events(
            intensity_frames=intensity_frames,
            trajectory_id=trajectory.trajectory_id,
            return_sparse=return_sparse,
        )

    def dense_to_sparse_tensor(self, dense_events: torch.Tensor) -> torch.Tensor:
        """Converts dense events [B, T, 2, H, W] to sparse coordinate tensor [N_events, 5]."""
        return torch.nonzero(dense_events > 0, as_tuple=False)
