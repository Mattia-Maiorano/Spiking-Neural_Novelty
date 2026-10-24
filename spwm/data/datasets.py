"""
PyTorch Dataset and DataLoader pipelines for SPWM-v0.4.0 (3D Unity Robotics & Attractor Dynamics).
Parses exported Unity robotic trajectories (joint positions q*, joint velocities p*, motor actions u_t,
and neuromorphic event streams) while enforcing strict trajectory-level partitioning to prevent temporal leakage.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spwm.data.event_camera import EventBatch, EventCameraSimulator, RoboticTrajectory


def partition_trajectories(
    total_trajectories: int = 1000,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Dict[str, Tuple[int, int]]:
    """
    Partitions trajectory IDs strictly by trajectory index to eliminate temporal leakage.
    Example: 1000 trajectories -> train [0, 800), val [800, 900), test [900, 1000).
    """
    n_train = int(total_trajectories * train_split)
    n_val = int(total_trajectories * val_split)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total_trajectories),
    }


def simulate_articulated_trajectory(
    trajectory_id: int,
    length: int = 105,
    joint_count: int = 8,
    action_dim: int = 8,
    velocity_range: Tuple[float, float] = (-1.0, 1.0),
    dt: float = 0.05,
    damping: float = 0.25,
    seed: Optional[int] = None,
) -> RoboticTrajectory:
    """
    Simulates an action-conditioned 3D articulated robotic trajectory matching the Unity C# physics controller:
      - Even trajectory IDs form rhythmic CPG limit-cycle walking gait attractors.
      - Odd trajectory IDs form damped PD posture fixed-point attractors under gravity and joint friction.
    """
    rng = np.random.default_rng(seed if seed is not None else trajectory_id)
    v_scale = max(0.2, 0.5 * (abs(velocity_range[1]) + abs(velocity_range[0])))

    q = rng.uniform(-0.45, 0.45, size=(joint_count,)).astype(np.float32)
    p = rng.uniform(-0.35 * v_scale, 0.35 * v_scale, size=(joint_count,)).astype(np.float32)

    half_dof = max(1, joint_count // 2)
    phase_offsets = np.array(
        [(np.pi if j >= half_dof else 0.0) + (j % half_dof) * 0.45 for j in range(joint_count)],
        dtype=np.float32,
    ) + rng.uniform(-0.2, 0.2, size=(joint_count,)).astype(np.float32)
    equilibrium_q = rng.uniform(-0.5, 0.5, size=(joint_count,)).astype(np.float32)
    omega = float(2.0 * np.pi * (1.2 + 0.3 * (trajectory_id % 3)))

    q_steps = np.zeros((length, joint_count), dtype=np.float32)
    p_steps = np.zeros((length, joint_count), dtype=np.float32)
    u_steps = np.zeros((length, action_dim), dtype=np.float32)
    times = (np.arange(length, dtype=np.float32) * float(dt))

    is_limit_cycle = (trajectory_id % 2 == 0)
    stiffness = 6.5
    gravity_torque_scale = 0.35

    for t in range(length):
        time_s = float(times[t])
        if is_limit_cycle:
            # CPG rhythmic motor target command -> stable limit-cycle attractor
            raw_u = 0.75 * np.sin(omega * time_s + phase_offsets) + 0.20 * np.cos(2.0 * omega * time_s + phase_offsets)
        else:
            # Postural PD command -> stable fixed-point attractor with smooth exploratory excitation
            target_q = equilibrium_q + 0.30 * np.sin(0.6 * omega * time_s + phase_offsets)
            raw_u = np.clip(1.2 * (target_q - q) - 0.35 * p, -1.0, 1.0)

        u_t = np.zeros((action_dim,), dtype=np.float32)
        common_d = min(joint_count, action_dim)
        u_t[:common_d] = np.clip(raw_u[:common_d], -1.0, 1.0)
        if action_dim > joint_count:
            u_t[joint_count:] = np.sin(omega * time_s + np.arange(action_dim - joint_count, dtype=np.float32))

        q_steps[t] = q.copy()
        p_steps[t] = p.copy()
        u_steps[t] = u_t.copy()

        # Dissipative articulated joint dynamics: motor drive + restorative spring + gravity - viscous damping
        drive_target = u_t[:common_d] if common_d == joint_count else np.resize(u_t, (joint_count,))
        gravity_term = -gravity_torque_scale * np.sin(q)
        coulomb_friction = -0.05 * np.tanh(5.0 * p)
        accel = stiffness * (drive_target - q) + gravity_term + coulomb_friction - (damping * 6.0) * p

        p = p + float(dt) * accel
        q = np.clip(q + float(dt) * p, -1.0, 1.0)

    return RoboticTrajectory(
        trajectory_id=trajectory_id,
        joint_positions=q_steps,
        joint_velocities=p_steps,
        actions=u_steps,
        times=times,
        joint_count=joint_count,
        action_dim=action_dim,
    )


def export_unity_trajectory_file(
    trajectory: RoboticTrajectory,
    file_path: Union[str, Path],
    sensor_height: int = 32,
    sensor_width: int = 32,
    dt: float = 0.05,
    event_threshold: float = 0.08,
) -> Path:
    """Serializes a RoboticTrajectory to Unity JSON or NPZ telemetry format on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T = trajectory.joint_positions.shape[0]

    if path.suffix.lower() == ".npz":
        payload: Dict[str, Any] = {
            "trajectory_id": np.array(trajectory.trajectory_id, dtype=np.int64),
            "joint_positions": trajectory.joint_positions.astype(np.float32),
            "joint_velocities": trajectory.joint_velocities.astype(np.float32),
            "actions": trajectory.actions.astype(np.float32),
            "times": trajectory.times.astype(np.float32),
        }
        if trajectory.grayscale_frames is not None:
            payload["grayscale_frames"] = trajectory.grayscale_frames.astype(np.float32)
        if trajectory.dense_events is not None:
            payload["dense_events"] = trajectory.dense_events.astype(np.float32)
        np.savez_compressed(path, **payload)
        return path

    steps = []
    for t in range(T):
        step_rec: Dict[str, Any] = {
            "step": t,
            "timestamp": float(trajectory.times[t]),
            "joint_positions": trajectory.joint_positions[t].tolist(),
            "joint_velocities": trajectory.joint_velocities[t].tolist(),
            "actions": trajectory.actions[t].tolist(),
        }
        if trajectory.grayscale_frames is not None:
            step_rec["grayscale_frame"] = trajectory.grayscale_frames[t].reshape(-1).tolist()
        if trajectory.dense_events is not None:
            step_rec["dense_events"] = trajectory.dense_events[t].reshape(-1).tolist()
        steps.append(step_rec)

    doc = {
        "trajectory_id": int(trajectory.trajectory_id),
        "archetype": "Humanoid",
        "attractor_regime": "LimitCycleWalkingGait" if trajectory.trajectory_id % 2 == 0 else "FixedPointBalancing",
        "sequence_length": int(T),
        "joint_count": int(trajectory.joint_count),
        "action_dim": int(trajectory.action_dim),
        "sensor_height": int(sensor_height),
        "sensor_width": int(sensor_width),
        "dt": float(dt),
        "event_threshold": float(event_threshold),
        "steps": steps,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def parse_unity_trajectory_file(
    file_path: Union[str, Path],
    sequence_length: Optional[int] = None,
    sensor_height: int = 32,
    sensor_width: int = 32,
) -> RoboticTrajectory:
    """Parses an exported Unity trajectory (.json or .npz) into a RoboticTrajectory."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Unity trajectory file not found: {path}")

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            traj_id = int(data["trajectory_id"])
            q = data["joint_positions"].astype(np.float32)
            p = data["joint_velocities"].astype(np.float32)
            u = data["actions"].astype(np.float32)
            times = data["times"].astype(np.float32) if "times" in data else np.arange(len(q), dtype=np.float32) * 0.05
            gray = data["grayscale_frames"].astype(np.float32) if "grayscale_frames" in data else None
            ev = data["dense_events"].astype(np.float32) if "dense_events" in data else None
    else:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        traj_id = int(doc["trajectory_id"])
        H = int(doc.get("sensor_height", sensor_height))
        W = int(doc.get("sensor_width", sensor_width))
        dt = float(doc.get("dt", 0.05))
        steps = doc["steps"]

        q = np.array([s["joint_positions"] for s in steps], dtype=np.float32)
        p = np.array([s["joint_velocities"] for s in steps], dtype=np.float32)
        u = np.array([s["actions"] for s in steps], dtype=np.float32)
        times = np.array([s.get("timestamp", idx * dt) for idx, s in enumerate(steps)], dtype=np.float32)

        gray = None
        if steps and "grayscale_frame" in steps[0] and steps[0]["grayscale_frame"]:
            gray = np.array([s["grayscale_frame"] for s in steps], dtype=np.float32).reshape(len(steps), H, W)
        ev = None
        if steps and "dense_events" in steps[0] and steps[0]["dense_events"]:
            ev = np.array([s["dense_events"] for s in steps], dtype=np.float32).reshape(len(steps), 2, H, W)

    if sequence_length is not None and q.shape[0] > sequence_length:
        q = q[:sequence_length]
        p = p[:sequence_length]
        u = u[:sequence_length]
        times = times[:sequence_length]
        if gray is not None:
            gray = gray[:sequence_length]
        if ev is not None:
            ev = ev[:sequence_length]

    return RoboticTrajectory(
        trajectory_id=traj_id,
        joint_positions=q,
        joint_velocities=p,
        actions=u,
        times=times,
        joint_count=int(q.shape[-1]),
        action_dim=int(u.shape[-1]),
        grayscale_frames=gray,
        dense_events=ev,
    )


class EventWorldDataset(Dataset):
    """
    Dataset of neuromorphic event streams and articulated robotic telemetry (q*, p*, u_t).
    Parses exported Unity trajectory files from `dataset_dir` when present, and falls back to
    deterministic 3D articulated robotic attractor simulation seeded strictly by trajectory_id.
    """

    def __init__(
        self,
        trajectory_indices: List[int],
        dataset_dir: Optional[Union[str, Path]] = None,
        event_simulator: Optional[EventCameraSimulator] = None,
        sequence_length: int = 105,
        joint_count: int = 8,
        action_dim: int = 8,
        velocity_range: Tuple[float, float] = (-1.0, 1.0),
        dt: float = 0.05,
        cache_data: bool = True,
    ) -> None:
        super().__init__()
        self.trajectory_indices = list(trajectory_indices)
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else None
        self.event_simulator = event_simulator or EventCameraSimulator()
        self.sequence_length = sequence_length
        self.joint_count = joint_count
        self.action_dim = action_dim
        self.velocity_range = velocity_range
        self.dt = float(dt)
        self.cache_data = cache_data

        self._cache: Dict[int, Dict[str, Any]] = {}
        if cache_data:
            for traj_id in self.trajectory_indices:
                self._cache[traj_id] = self._load_or_simulate_item(traj_id)

    def _resolve_unity_file(self, traj_id: int) -> Optional[Path]:
        if self.dataset_dir is None or not self.dataset_dir.is_dir():
            return None
        for candidate in (
            self.dataset_dir / f"trajectory_{traj_id:05d}.npz",
            self.dataset_dir / f"trajectory_{traj_id:05d}.json",
            self.dataset_dir / f"trajectory_{traj_id}.npz",
            self.dataset_dir / f"trajectory_{traj_id}.json",
        ):
            if candidate.is_file():
                return candidate
        return None

    def _load_or_simulate_item(self, traj_id: int) -> Dict[str, Any]:
        unity_file = self._resolve_unity_file(traj_id)
        if unity_file is not None:
            traj = parse_unity_trajectory_file(
                unity_file,
                sequence_length=self.sequence_length,
                sensor_height=self.event_simulator.height,
                sensor_width=self.event_simulator.width,
            )
        else:
            traj = simulate_articulated_trajectory(
                trajectory_id=traj_id,
                length=self.sequence_length,
                joint_count=self.joint_count,
                action_dim=self.action_dim,
                velocity_range=self.velocity_range,
                dt=self.dt,
                seed=traj_id,
            )

        event_batch: EventBatch = self.event_simulator.trajectory_to_events(traj, return_sparse=False)
        q_tensor = torch.from_numpy(traj.joint_positions).float()
        p_tensor = torch.from_numpy(traj.joint_velocities).float()
        u_tensor = torch.from_numpy(traj.actions).float()
        flat_kin = torch.cat([q_tensor, p_tensor], dim=-1)

        return {
            "events": event_batch.dense_events,      # [T, 2, H, W]
            "joint_positions": q_tensor,             # [T, joint_count] (q^*)
            "joint_velocities": p_tensor,            # [T, joint_count] (p^*)
            "positions": q_tensor,                   # [T, joint_count]
            "velocities": p_tensor,                  # [T, joint_count]
            "actions": u_tensor,                     # [T, action_dim] (u_t)
            "flat_kinematics": flat_kin,             # [T, 2 * joint_count]
            "trajectory_id": traj_id,
        }

    def __len__(self) -> int:
        return len(self.trajectory_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_id = self.trajectory_indices[idx]
        if self.cache_data and traj_id in self._cache:
            return self._cache[traj_id]
        return self._load_or_simulate_item(traj_id)


def collate_event_batches(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collates a list of trajectory items into a batched dictionary with actions u_t."""
    events = torch.stack([item["events"] for item in batch], dim=0)                      # [B, T, 2, H, W]
    flat_kinematics = torch.stack([item["flat_kinematics"] for item in batch], dim=0)    # [B, T, 2 * J]
    joint_positions = torch.stack([item["joint_positions"] for item in batch], dim=0)    # [B, T, J]
    joint_velocities = torch.stack([item["joint_velocities"] for item in batch], dim=0)  # [B, T, J]
    actions = torch.stack([item["actions"] for item in batch], dim=0)                    # [B, T, D_u]
    trajectory_ids = torch.tensor([item["trajectory_id"] for item in batch], dtype=torch.long)

    return {
        "events": events,
        "flat_kinematics": flat_kinematics,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "positions": joint_positions,
        "velocities": joint_velocities,
        "actions": actions,
        "trajectory_ids": trajectory_ids,
    }


def create_dataloaders(
    total_trajectories: int = 1000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    sequence_length: int = 105,
    batch_size: int = 32,
    height: int = 32,
    width: int = 32,
    joint_count: int = 8,
    action_dim: int = 8,
    dataset_dir: Optional[Union[str, Path]] = None,
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0),
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0),
    event_threshold: float = 0.08,
    dt: float = 0.05,
    num_workers: int = 0,
    cache_data: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, test, and extrapolation DataLoaders with strict trajectory-level partitioning.
    """
    splits = partition_trajectories(total_trajectories, train_split, val_split)
    train_ids = list(range(splits["train"][0], splits["train"][1]))
    val_ids = list(range(splits["val"][0], splits["val"][1]))
    test_ids = list(range(splits["test"][0], splits["test"][1]))
    extrap_ids = list(range(total_trajectories, total_trajectories + 100))

    event_sim = EventCameraSimulator(height=height, width=width, event_threshold=event_threshold)

    train_ds = EventWorldDataset(
        trajectory_indices=train_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=standard_velocity_range,
        dt=dt,
        cache_data=cache_data,
    )
    val_ds = EventWorldDataset(
        trajectory_indices=val_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=standard_velocity_range,
        dt=dt,
        cache_data=cache_data,
    )
    test_ds = EventWorldDataset(
        trajectory_indices=test_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=standard_velocity_range,
        dt=dt,
        cache_data=cache_data,
    )
    extrap_ds = EventWorldDataset(
        trajectory_indices=extrap_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=extrapolation_velocity_range,
        dt=dt,
        cache_data=cache_data,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_event_batches, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "extrapolation": DataLoader(extrap_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
    }
