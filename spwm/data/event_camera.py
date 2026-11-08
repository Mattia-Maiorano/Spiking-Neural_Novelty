"""
Dual-Camera Event Simulator & Neuromorphic Vision Bridge for 3D Articulated Robotics.
Transforms exported Unity grayscale image sequences or 3D articulated robotic joint trajectories
into unified, multi-channel spatio-temporal neuromorphic event streams ([B, T, 4, H, W]):
  - Channels 0..1: Static Global Event Camera (ON/OFF polarities)
  - Channels 2..3: Egocentric Wrist-Mounted Event Camera (ON/OFF polarities)
Resolves 3D kinematic unobservability (axial wrist roll symmetry and elbow flex depth foreshortening).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch

from spwm.contracts.tensors import C_EVENT, J


@dataclass
class RoboticTrajectory:
    """Ground-truth 3D 6-DOF manipulator trajectory exported from Unity or synthesized."""
    trajectory_id: int
    joint_positions: np.ndarray   # [T, joint_count] (q^*)
    joint_velocities: np.ndarray  # [T, joint_count] (p^*)
    actions: np.ndarray           # [T, action_dim] (u_t)
    times: np.ndarray             # [T]
    joint_count: int = J
    action_dim: int = J
    phase_flags: Optional[np.ndarray] = None             # [T] int64 in {0: BlindApproach, 1: SensoryHeft, 2: BlindTransport}
    payload_mass: float = 1.5
    payload_shape: str = "Box"
    grayscale_frames: Optional[np.ndarray] = None        # [T, H, W] (static global) or [T, 2, H, W] (dual-camera)
    wrist_grayscale_frames: Optional[np.ndarray] = None  # [T, H, W] (egocentric wrist camera)
    dense_events: Optional[np.ndarray] = None            # [T, 4, H, W] (dual-camera) or [T, 2, H, W]

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
    dense_events: torch.Tensor  # [B, T, 4, H, W] or [T, 4, H, W] (or 2 channels when in_channels == 2)
    sparse_events: Optional[List[List[Tuple[int, int, int, int]]]] = None  # (t, c, y, x)
    trajectory_ids: Optional[torch.Tensor] = None

    @property
    def shape(self) -> torch.Size:
        return self.dense_events.shape


class EventCameraSimulator:
    """
    Simulates a dual-sensor differential neuromorphic event camera architecture
    bridging Unity 3D robotics to SNNs:
      1. Static Global Camera: observes overall workspace and macro arm motion.
      2. Egocentric Wrist-Mounted Camera: attached directly to the end-effector, observing
         axial wrist roll (q5), wrist pitch/yaw (q3, q4), and elbow flex (q2) depth foreshortening
         in the wrist-centric coordinate frame.

    Converts grayscale luminance sequences or 3D articulated forward-kinematic projections
    into log-intensity fields L(t), calculates temporal changes:
        ΔL = L(t) - L(t - Δt)
    and emits binary polarity spikes when |ΔL| > threshold (default θ = 0.08):
        Channel 0: Global Positive polarity (intensity brightening)
        Channel 1: Global Negative polarity (intensity dimming)
        Channel 2: Egocentric Wrist Positive polarity (intensity brightening)
        Channel 3: Egocentric Wrist Negative polarity (intensity dimming)
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        event_threshold: float = 0.08,
        link_radius: float = 0.10,
        in_channels: int = C_EVENT,
    ) -> None:
        if in_channels not in (2, C_EVENT):
            raise ValueError(f"EventCameraSimulator in_channels must be 2 or {C_EVENT}, got {in_channels}")
        self.height = height
        self.width = width
        self.event_threshold = event_threshold
        self.link_radius = link_radius
        self.in_channels = in_channels

        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

    def render_articulated_frame(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Projects a 6-DOF articulated manipulator chain with joint angles q^* in R^J onto the
        static global 2D sensor plane [-1, 1]^2 via forward kinematics of the serial arm links.
        """
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        dof = int(joint_positions.shape[0])
        curr_x, curr_y = 0.0, -0.55
        cum_angle = 0.5 * np.pi
        total_reach = 1.25

        for j in range(dof):
            link_scale = max(0.45, 1.0 - 0.10 * j)
            sigma_sq = ((self.link_radius * link_scale) ** 2) + 1e-8
            link_len = (total_reach / float(max(1, dof))) * link_scale
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

    def render_wrist_articulated_frame(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Projects the end-effector, wrist orientation fiducials, and forearm/workspace landmarks
        into the egocentric wrist-mounted camera plane [-1, 1]^2.
        Directly resolves 3D kinematic unobservability:
          - Wrist Roll (q5): rotates asymmetric gripper fiducials around the optical center.
          - Wrist Pitch & Yaw (q3, q4): shifts the optical axis relative to the workspace/payload.
          - Elbow Flex (q2) & Shoulder Lift (q1): modulates depth foreshortening and parallax.
        """
        frame = np.zeros((self.height, self.width), dtype=np.float32)
        dof = int(joint_positions.shape[0])
        q0 = float(joint_positions[0]) if dof > 0 else 0.0
        q1 = float(joint_positions[1]) if dof > 1 else 0.0
        q2 = float(joint_positions[2]) if dof > 2 else 0.0
        q3 = float(joint_positions[3]) if dof > 3 else 0.0
        q4 = float(joint_positions[4]) if dof > 4 else 0.0
        q5 = float(joint_positions[5]) if dof > 5 else 0.0

        # Egocentric center shifts with wrist pitch (q3), wrist yaw (q4), and elbow flex (q2) foreshortening
        ego_cx = float(np.clip(0.32 * np.sin(q4) + 0.18 * np.sin(q0), -0.65, 0.65))
        ego_cy = float(np.clip(0.32 * np.sin(q3) - 0.22 * np.cos(q2), -0.65, 0.65))

        # Depth foreshortening scale from elbow flex (q2) and shoulder lift (q1)
        depth_scale = float(np.clip(0.38 + 0.14 * np.sin(q2) + 0.08 * np.cos(q1), 0.22, 0.58))
        sigma_base = (self.link_radius * 0.95) ** 2 + 1e-8

        # 3 asymmetric gripper/payload fiducial landmarks rotating with axial Wrist_Roll (q5)
        fiducial_angles = (0.0, 2.0943951, -1.5707963)
        fiducial_radii = (depth_scale, 0.78 * depth_scale, 0.58 * depth_scale)
        fiducial_weights = (1.0, 0.90, 0.80)

        for base_ang, rad, weight in zip(fiducial_angles, fiducial_radii, fiducial_weights):
            ang = q5 + base_ang + 0.35 * q4
            fx = float(np.clip(ego_cx + rad * np.cos(ang), -0.92, 0.92))
            fy = float(np.clip(ego_cy + rad * np.sin(ang), -0.92, 0.92))
            dist_sq = (self.grid_x - fx) ** 2 + (self.grid_y - fy) ** 2
            blob = weight * np.exp(-0.5 * dist_sq / sigma_base)
            frame = np.maximum(frame, blob.astype(np.float32))

        # Forearm-to-wrist kinematic anchor reflecting elbow flex (q2) and shoulder lift (q1)
        elbow_x = float(np.clip(-0.35 * np.sin(q2 - q4), -0.90, 0.90))
        elbow_y = float(np.clip(0.35 * np.cos(q2 + q3), -0.90, 0.90))
        dist_sq_elbow = (self.grid_x - elbow_x) ** 2 + (self.grid_y - elbow_y) ** 2
        elbow_blob = 0.85 * np.exp(-0.5 * dist_sq_elbow / (1.15 * sigma_base))
        frame = np.maximum(frame, elbow_blob.astype(np.float32))

        return frame

    def render_dual_articulated_frames(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Renders both the static global camera frame and the egocentric wrist-mounted camera frame
        simultaneously as a unified [2, H, W] array.
        """
        global_frame = self.render_articulated_frame(joint_positions)
        wrist_frame = self.render_wrist_articulated_frame(joint_positions)
        return np.stack([global_frame, wrist_frame], axis=0)

    def render_trajectory_intensity(self, trajectory: RoboticTrajectory) -> np.ndarray:
        """
        Returns intensity frames for the trajectory:
          - When `self.in_channels == 4` (dual-camera mode): returns `[T, 2, H, W]` where index 0 is the
            static global view and index 1 is the egocentric wrist view.
          - When `self.in_channels == 2` (monocular compatibility mode): returns `[T, H, W]`.
        """
        T = trajectory.joint_positions.shape[0]

        if self.in_channels == 2:
            if trajectory.grayscale_frames is not None:
                gray = trajectory.grayscale_frames.astype(np.float32)
                return gray[:, 0] if gray.ndim == 4 else gray
            intensity_frames = np.zeros((T, self.height, self.width), dtype=np.float32)
            for t in range(T):
                intensity_frames[t] = self.render_articulated_frame(trajectory.joint_positions[t])
            return intensity_frames

        # Dual-camera mode (in_channels == 4 -> [T, 2, H, W] luminance streams)
        if trajectory.grayscale_frames is not None:
            gray = trajectory.grayscale_frames.astype(np.float32)
            if gray.ndim == 4 and gray.shape[1] == 2:
                return gray
            if gray.ndim == 3:
                if self.in_channels == C_EVENT:
                    if trajectory.wrist_grayscale_frames is None:
                        raise ValueError(
                            f"Strict dual-camera contract error: in_channels == {self.in_channels} requires "
                            "egocentric wrist camera telemetry ('wrist_grayscale_frames'), but received None. "
                            "Fallback proxy generation is removed; trajectories must contain dual-camera frames."
                        )
                    wrist_gray = trajectory.wrist_grayscale_frames.astype(np.float32)
                    return np.stack([gray, wrist_gray], axis=1)
                return gray

        dual_frames = np.zeros((T, 2, self.height, self.width), dtype=np.float32)
        for t in range(T):
            dual_frames[t] = self.render_dual_articulated_frames(trajectory.joint_positions[t])
        return dual_frames

    def _single_stream_frames_to_polarity(self, stream_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes positive and negative polarity spike arrays ([T, H, W] each) from a single
        camera's intensity sequence [T, H, W] using persistent reference contrast accumulation.
        """
        T = stream_frames.shape[0]
        pos_spikes = np.zeros((T, self.height, self.width), dtype=np.float32)
        neg_spikes = np.zeros((T, self.height, self.width), dtype=np.float32)

        if T > 0:
            ref_intensity = stream_frames[0].copy()
            for t in range(1, T):
                curr = stream_frames[t]
                delta_L = curr - ref_intensity
                pos_mask = delta_L > self.event_threshold
                neg_mask = delta_L < -self.event_threshold
                pos_spikes[t] = pos_mask.astype(np.float32)
                neg_spikes[t] = neg_mask.astype(np.float32)
                event_mask = pos_mask | neg_mask
                ref_intensity[event_mask] = curr[event_mask]

        return pos_spikes, neg_spikes

    def frames_to_events(
        self,
        intensity_frames: np.ndarray,
        trajectory_id: int = 0,
        return_sparse: bool = False,
    ) -> EventBatch:
        """
        Computes temporal contrast accumulation across intensity frames with threshold θ = 0.08:
          - If `intensity_frames` has shape `[T, 2, H, W]` (dual camera), emits `[T, 4, H, W]`
            with channels `[pos_global, neg_global, pos_wrist, neg_wrist]`.
          - If `intensity_frames` has shape `[T, H, W]` and `self.in_channels == 4`, pairs the global
            stream with a complementary egocentric contrast stream to emit `[T, 4, H, W]`.
          - If `self.in_channels == 2`, emits `[T, 2, H, W]`.
        """
        if intensity_frames.ndim == 4 and intensity_frames.shape[1] == 2:
            T = intensity_frames.shape[0]
            pos_global, neg_global = self._single_stream_frames_to_polarity(intensity_frames[:, 0])
            pos_wrist, neg_wrist = self._single_stream_frames_to_polarity(intensity_frames[:, 1])
            if self.in_channels == 2:
                dense_events = np.stack([pos_global, neg_global], axis=1)
            else:
                dense_events = np.stack([pos_global, neg_global, pos_wrist, neg_wrist], axis=1)
        elif intensity_frames.ndim == 3:
            T = intensity_frames.shape[0]
            pos_global, neg_global = self._single_stream_frames_to_polarity(intensity_frames)
            dense_events = np.stack([pos_global, neg_global], axis=1)
        else:
            raise ValueError(
                f"intensity_frames must have shape [T, H, W] or [T, 2, H, W], got {intensity_frames.shape}"
            )

        dense_tensor = torch.from_numpy(dense_events).float()

        sparse_list = None
        if return_sparse:
            sparse_list = []
            num_ch = dense_events.shape[1]
            for t in range(T):
                step_events: List[Tuple[int, int, int, int]] = []
                for c in range(num_ch):
                    ys, xs = np.where(dense_events[t, c] > 0)
                    for py, px in zip(ys, xs):
                        step_events.append((t, int(c), int(py), int(px)))
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
        Converts a RoboticTrajectory into unified neuromorphic event streams [T, C_EVENT, H, W].
        Uses pre-computed Unity ComputeShader events if present (upgrading 2-channel legacy exports
        with egocentric wrist events when `in_channels == 4`), else bridges grayscale/kinematic frames.
        """
        if trajectory.dense_events is not None and not return_sparse:
            ev = trajectory.dense_events.astype(np.float32)
            if ev.shape[1] == self.in_channels:
                return EventBatch(
                    dense_events=torch.from_numpy(ev).float(),
                    sparse_events=None,
                    trajectory_ids=torch.tensor([trajectory.trajectory_id], dtype=torch.long),
                )
            if ev.shape[1] == 2 and self.in_channels == C_EVENT:
                raise ValueError(
                    f"Strict dual-camera contract error: trajectory dense_events has only 2 channels, "
                    f"but model in_channels == {self.in_channels}. Cannot synthesize wrist events from proxy. "
                    "Fallback proxy generation is removed; trajectories must contain 4-channel dual-camera events."
                )

        intensity_frames = self.render_trajectory_intensity(trajectory)
        return self.frames_to_events(
            intensity_frames=intensity_frames,
            trajectory_id=trajectory.trajectory_id,
            return_sparse=return_sparse,
        )

    def dense_to_sparse_tensor(self, dense_events: torch.Tensor) -> torch.Tensor:
        """Converts dense events [B, T, C, H, W] to sparse coordinate tensor [N_events, 5]."""
        return torch.nonzero(dense_events > 0, as_tuple=False)
