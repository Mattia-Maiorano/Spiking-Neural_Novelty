"""
PyTorch Dataset and DataLoader pipelines for SPWM.
Ensures zero temporal leakage via strict trajectory-level partitioning.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from spwm.data.synthetic_world import MovingObjectsWorld, Trajectory, partition_trajectories
from spwm.data.event_camera import EventCameraSimulator


class EventWorldDataset(Dataset):
    """
    Dataset of event streams generated from MovingObjectsWorld.
    Trajectories are strictly indexed and generated deterministically.
    """

    def __init__(
        self,
        trajectory_indices: List[int],
        world: Optional[MovingObjectsWorld] = None,
        event_simulator: Optional[EventCameraSimulator] = None,
        sequence_length: int = 50,
        num_objects: int = 1,
        velocity_range: Tuple[float, float] = (-1.0, 1.0),
        acceleration_std: float = 0.1,
        cache_data: bool = True,
    ) -> None:
        super().__init__()
        self.trajectory_indices = trajectory_indices
        self.world = world or MovingObjectsWorld()
        self.event_simulator = event_simulator or EventCameraSimulator()
        self.sequence_length = sequence_length
        self.num_objects = num_objects
        self.velocity_range = velocity_range
        self.acceleration_std = acceleration_std
        self.cache_data = cache_data

        self._cache: Dict[int, Dict[str, Any]] = {}
        if cache_data:
            # Pre-generate dataset
            for traj_id in self.trajectory_indices:
                self._cache[traj_id] = self._generate_item(traj_id)

    def _generate_item(self, traj_id: int) -> Dict[str, Any]:
        traj = self.world.generate_trajectory(
            trajectory_id=traj_id,
            length=self.sequence_length,
            num_objects=self.num_objects,
            velocity_range=self.velocity_range,
            acceleration_std=self.acceleration_std,
            seed=traj_id,
        )
        event_batch = self.event_simulator.trajectory_to_events(traj, return_sparse=False)

        return {
            "events": event_batch.dense_events,  # [T, 2, H, W]
            "flat_kinematics": torch.from_numpy(traj.flat_kinematics).float(),  # [T, 4 * num_objects]
            "positions": torch.from_numpy(traj.positions).float(),  # [T, num_objects, 2]
            "velocities": torch.from_numpy(traj.velocities).float(),  # [T, num_objects, 2]
            "trajectory_id": traj_id,
        }

    def __len__(self) -> int:
        return len(self.trajectory_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_id = self.trajectory_indices[idx]
        if self.cache_data and traj_id in self._cache:
            return self._cache[traj_id]
        return self._generate_item(traj_id)


def collate_event_batches(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collates a list of trajectory items into a batched dictionary."""
    events = torch.stack([item["events"] for item in batch], dim=0)  # [B, T, 2, H, W]
    flat_kinematics = torch.stack([item["flat_kinematics"] for item in batch], dim=0)  # [B, T, 4 * N]
    positions = torch.stack([item["positions"] for item in batch], dim=0)  # [B, T, N, 2]
    velocities = torch.stack([item["velocities"] for item in batch], dim=0)  # [B, T, N, 2]
    trajectory_ids = torch.tensor([item["trajectory_id"] for item in batch], dtype=torch.long)

    return {
        "events": events,
        "flat_kinematics": flat_kinematics,
        "positions": positions,
        "velocities": velocities,
        "trajectory_ids": trajectory_ids,
    }


def create_dataloaders(
    total_trajectories: int = 1000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    sequence_length: int = 50,
    batch_size: int = 32,
    height: int = 32,
    width: int = 32,
    num_objects: int = 1,
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0),
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0),
    num_workers: int = 0,
    cache_data: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, test, and extrapolation DataLoaders with zero temporal leakage.
    """
    splits = partition_trajectories(total_trajectories, train_split, val_split)
    train_ids = list(range(splits["train"][0], splits["train"][1]))
    val_ids = list(range(splits["val"][0], splits["val"][1]))
    test_ids = list(range(splits["test"][0], splits["test"][1]))
    # Extrapolation uses separate trajectory IDs (e.g. 1000 to 1100) with higher velocity bounds
    extrap_ids = list(range(total_trajectories, total_trajectories + 100))

    world = MovingObjectsWorld()
    event_sim = EventCameraSimulator(height=height, width=width)

    train_ds = EventWorldDataset(
        trajectory_indices=train_ids,
        world=world,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        num_objects=num_objects,
        velocity_range=standard_velocity_range,
        cache_data=cache_data,
    )
    val_ds = EventWorldDataset(
        trajectory_indices=val_ids,
        world=world,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        num_objects=num_objects,
        velocity_range=standard_velocity_range,
        cache_data=cache_data,
    )
    test_ds = EventWorldDataset(
        trajectory_indices=test_ids,
        world=world,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        num_objects=num_objects,
        velocity_range=standard_velocity_range,
        cache_data=cache_data,
    )
    extrap_ds = EventWorldDataset(
        trajectory_indices=extrap_ids,
        world=world,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        num_objects=num_objects,
        velocity_range=extrapolation_velocity_range,
        cache_data=cache_data,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_event_batches, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "extrapolation": DataLoader(extrap_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
    }
