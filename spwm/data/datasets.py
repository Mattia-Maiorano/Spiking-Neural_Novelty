"""
PyTorch Dataset and DataLoader pipelines for SPWM (6-DOF Active Perception Grasping).
Parses exported Unity robotic trajectories (joint positions q* in R^6, joint velocities p* in R^6,
motor actions u_t in R^6, phase flags in {0, 1, 2}, and neuromorphic event streams) while enforcing
strict trajectory-level partitioning to prevent temporal leakage.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.tensors import C_EVENT, J, assert_tensor_contract
from spwm.data.event_camera import EventBatch, EventCameraSimulator, RoboticTrajectory
from spwm.data.io_unity import export_unity_trajectory_file, parse_unity_trajectory_file
from spwm.data.sim_6dof import (
    _compute_default_phase_schedule,
    _solve_6dof_industrial_ik,
    simulate_articulated_trajectory,
)

__all__ = [
    "RoboticTrajectory",
    "EventBatch",
    "EventCameraSimulator",
    "EventWorldDataset",
    "collate_event_batches",
    "create_dataloaders",
    "create_dataloaders_from_config",
    "partition_trajectories",
    "simulate_articulated_trajectory",
    "export_unity_trajectory_file",
    "parse_unity_trajectory_file",
]


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


class EventWorldDataset(Dataset):
    """
    Dataset of neuromorphic event streams and 6-DOF articulated manipulator telemetry (q*, p*, u_t, phase_flags).
    Parses exported Unity trajectory files from `dataset_dir` when present, and falls back to
    deterministic 6-DOF active perception grasping simulation seeded strictly by trajectory_id.
    """

    def __init__(
        self,
        trajectory_indices: List[int],
        dataset_dir: Optional[Union[str, Path]] = None,
        event_simulator: Optional[EventCameraSimulator] = None,
        sequence_length: int = 105,
        joint_count: int = J,
        action_dim: int = J,
        velocity_range: Tuple[float, float] = (-1.0, 1.0),
        dt: float = 0.05,
        cache_data: bool = True,
        strict_dataset: bool = True,
        pre_contact_sensory_steps: int = 7,
        post_contact_sensory_steps: int = 7,
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
        self.strict_dataset = strict_dataset
        self.pre_contact_sensory_steps = int(pre_contact_sensory_steps)
        self.post_contact_sensory_steps = int(post_contact_sensory_steps)

        if self.strict_dataset:
            if self.dataset_dir is None:
                raise FileNotFoundError(
                    "Strict dataset validation error: `dataset_dir` is None. "
                    "SPWM contracts require exported Unity trajectories (e.g. at 'sim/ExportedTrajectories'). "
                    "Synthetic simulation fallback is disabled."
                )
            if not self.dataset_dir.is_dir():
                raise FileNotFoundError(
                    f"Strict dataset validation error: `dataset_dir` '{self.dataset_dir}' does not exist. "
                    "Please generate the real Unity trajectory dataset using the Unity physics simulation batch export."
                )
            missing = [tid for tid in self.trajectory_indices if self._resolve_unity_file(tid) is None]
            if missing:
                missing_preview = missing[:10]
                suffix = "..." if len(missing) > 10 else ""
                raise FileNotFoundError(
                    f"Strict dataset validation error: Missing {len(missing)} Unity trajectory files in '{self.dataset_dir}'. "
                    f"Sample missing IDs: {missing_preview}{suffix}. "
                    "Ensure Unity batch export has run for all requested trajectory IDs."
                )

        self._cache: Dict[int, Dict[str, Any]] = {}
        if cache_data:
            for traj_id in self.trajectory_indices:
                item = self._load_or_simulate_item(traj_id)
                # Store binary {0, 1} events as uint8 to cut RAM usage 4x (~1.7 GB vs ~6.8 GB).
                item["events"] = item["events"].to(torch.uint8)
                self._cache[traj_id] = item

    def _resolve_unity_file(self, traj_id: int) -> Optional[Path]:
        if self.dataset_dir is None or not self.dataset_dir.is_dir():
            return None
        for candidate in (
            self.dataset_dir / f"trajectory_{traj_id:05d}.json.gz",
            self.dataset_dir / f"trajectory_{traj_id:05d}.npz",
            self.dataset_dir / f"trajectory_{traj_id:05d}.json",
            self.dataset_dir / f"trajectory_{traj_id}.json.gz",
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
                pre_contact_sensory_steps=self.pre_contact_sensory_steps,
                post_contact_sensory_steps=self.post_contact_sensory_steps,
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

        if traj.phase_flags is not None:
            phase_flags_np = traj.phase_flags
        else:
            phase_flags_np = _compute_default_phase_schedule(q_tensor.shape[0])[3]
        phase_flags_tensor = torch.from_numpy(np.asarray(phase_flags_np, dtype=np.int64)).long()

        T_actual = q_tensor.shape[0]
        assert_tensor_contract(
            event_batch.dense_events,
            (T_actual, self.event_simulator.in_channels, self.event_simulator.height, self.event_simulator.width),
            "EventWorldDataset.events",
            check_finite=True,
        )
        assert_tensor_contract(q_tensor, (T_actual, self.joint_count), "EventWorldDataset.joint_positions", check_finite=True)
        assert_tensor_contract(p_tensor, (T_actual, self.joint_count), "EventWorldDataset.joint_velocities", check_finite=True)
        assert_tensor_contract(u_tensor, (T_actual, self.action_dim), "EventWorldDataset.actions", check_finite=True)
        assert_tensor_contract(flat_kin, (T_actual, 2 * self.joint_count), "EventWorldDataset.flat_kinematics", check_finite=True)
        assert_tensor_contract(phase_flags_tensor, (T_actual,), "EventWorldDataset.phase_flags")

        return {
            "events": event_batch.dense_events,      # [T, 4, H, W] (dual-camera: global + wrist)
            "joint_positions": q_tensor,             # [T, joint_count] (q^*)
            "joint_velocities": p_tensor,            # [T, joint_count] (p^*)
            "positions": q_tensor,                   # [T, joint_count]
            "velocities": p_tensor,                  # [T, joint_count]
            "actions": u_tensor,                     # [T, action_dim] (u_t)
            "flat_kinematics": flat_kin,             # [T, 2 * joint_count]
            "phase_flags": phase_flags_tensor,       # [T] int64 in {0, 1, 2}
            "trajectory_id": traj_id,
            "payload_mass": float(traj.payload_mass),
        }

    def __len__(self) -> int:
        return len(self.trajectory_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_id = self.trajectory_indices[idx]
        if self.cache_data and traj_id in self._cache:
            cached = self._cache[traj_id]
            out = dict(cached)
            out["events"] = cached["events"].float()
            return out
        return self._load_or_simulate_item(traj_id)


def collate_event_batches(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collates a list of trajectory items into a batched dictionary with actions u_t and phase_flags."""
    events = torch.stack([item["events"] for item in batch], dim=0)                      # [B, T, 2, H, W]
    flat_kinematics = torch.stack([item["flat_kinematics"] for item in batch], dim=0)    # [B, T, 2 * J]
    joint_positions = torch.stack([item["joint_positions"] for item in batch], dim=0)    # [B, T, J]
    joint_velocities = torch.stack([item["joint_velocities"] for item in batch], dim=0)  # [B, T, J]
    actions = torch.stack([item["actions"] for item in batch], dim=0)                    # [B, T, D_u]
    phase_flags = torch.stack([item["phase_flags"] for item in batch], dim=0)            # [B, T]
    trajectory_ids = torch.tensor([item["trajectory_id"] for item in batch], dtype=torch.long)

    return {
        "events": events,
        "flat_kinematics": flat_kinematics,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "positions": joint_positions,
        "velocities": joint_velocities,
        "actions": actions,
        "phase_flags": phase_flags,
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
    joint_count: int = J,
    action_dim: int = J,
    in_channels: int = C_EVENT,
    dataset_dir: Optional[Union[str, Path]] = None,
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0),
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0),
    event_threshold: float = 0.08,
    dt: float = 0.05,
    num_workers: int = 0,
    cache_data: bool = True,
    strict_dataset: bool = True,
    weighted_sampling: bool = False,
    heavy_weight: float = 2.5,
    mid_weight: float = 2.5,
    joint_heavy_mid_weight: float = 5.0,
    pre_contact_sensory_steps: int = 7,
    post_contact_sensory_steps: int = 7,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, test, and extrapolation DataLoaders with strict trajectory-level partitioning.
    Optionally applies weighted oversampling on the training split to expose M_1 to heavy payloads and mid-plane reach trajectories.
    """
    splits = partition_trajectories(total_trajectories, train_split, val_split)
    train_ids = list(range(splits["train"][0], splits["train"][1]))
    val_ids = list(range(splits["val"][0], splits["val"][1]))
    test_ids = list(range(splits["test"][0], splits["test"][1]))
    # For extrapolation evaluation, use the test partition IDs if dataset files are bounded to total_trajectories
    extrap_ids = test_ids[:min(len(test_ids), 100)] if (strict_dataset and dataset_dir is not None) else list(range(total_trajectories, total_trajectories + 100))

    event_sim = EventCameraSimulator(
        height=height,
        width=width,
        event_threshold=event_threshold,
        in_channels=in_channels,
    )

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
        strict_dataset=strict_dataset,
        pre_contact_sensory_steps=pre_contact_sensory_steps,
        post_contact_sensory_steps=post_contact_sensory_steps,
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
        strict_dataset=strict_dataset,
        pre_contact_sensory_steps=pre_contact_sensory_steps,
        post_contact_sensory_steps=post_contact_sensory_steps,
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
        strict_dataset=strict_dataset,
        pre_contact_sensory_steps=pre_contact_sensory_steps,
        post_contact_sensory_steps=post_contact_sensory_steps,
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
        strict_dataset=strict_dataset,
        pre_contact_sensory_steps=pre_contact_sensory_steps,
        post_contact_sensory_steps=post_contact_sensory_steps,
    )

    train_sampler = None
    train_shuffle = True
    if weighted_sampling:
        from torch.utils.data import WeightedRandomSampler
        from spwm.models.kinematics import DifferentiableForwardKinematics
        fk = DifferentiableForwardKinematics()
        weights: List[float] = []
        for i in range(len(train_ds)):
            item = train_ds[i]
            mass = float(item["payload_mass"]) if "payload_mass" in item else 1.5
            flags = item["phase_flags"]
            p1 = (flags == 1).nonzero()
            t_c = int(p1[0].item()) if len(p1) > 0 else min(50, len(flags) - 1)
            q_contact = item["joint_positions"][t_c : t_c + 1]
            pos_3d = fk(q_contact)[0]
            target_y = float(pos_3d[1].item())
            is_heavy = (mass >= 3.40)
            is_mid = (1.08 <= target_y < 1.18)
            if is_heavy and is_mid:
                w = joint_heavy_mid_weight
            elif is_heavy:
                w = heavy_weight
            elif is_mid:
                w = mid_weight
            else:
                w = 1.0
            weights.append(w)
        sampler_weights = torch.as_tensor(weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(sampler_weights, num_samples=len(train_ds), replacement=True)
        train_shuffle = False

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            collate_fn=collate_event_batches,
            num_workers=num_workers,
        ),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "extrapolation": DataLoader(extrap_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
    }


def create_dataloaders_from_config(
    cfg: ExperimentConfig,
    *,
    total_trajectories: Optional[int] = None,
    sequence_length: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    cache_data: Optional[bool] = None,
    strict_dataset: Optional[bool] = None,
    weighted_sampling: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, test, and extrapolation DataLoaders directly from an ExperimentConfig (SSOT).
    Optional keyword overrides adjust sequence_length, batch_size, or caching without manual field unpacking.
    """
    return create_dataloaders(
        total_trajectories=cfg.data.total_trajectories if total_trajectories is None else total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=cfg.data.sequence_length if sequence_length is None else sequence_length,
        batch_size=cfg.data.batch_size if batch_size is None else batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        joint_count=cfg.environment.joint_count,
        action_dim=cfg.environment.action_dim,
        in_channels=cfg.model.in_channels,
        dataset_dir=cfg.data.dataset_dir,
        standard_velocity_range=cfg.data.standard_velocity_range,
        extrapolation_velocity_range=cfg.data.extrapolation_velocity_range,
        event_threshold=cfg.environment.event_threshold,
        dt=cfg.environment.dt,
        num_workers=cfg.data.num_workers if num_workers is None else num_workers,
        cache_data=cfg.data.cache_data if cache_data is None else cache_data,
        strict_dataset=cfg.data.strict_dataset if strict_dataset is None else strict_dataset,
        weighted_sampling=cfg.data.weighted_sampling if weighted_sampling is None else weighted_sampling,
        pre_contact_sensory_steps=cfg.data.pre_contact_sensory_steps,
        post_contact_sensory_steps=cfg.data.post_contact_sensory_steps,
    )
