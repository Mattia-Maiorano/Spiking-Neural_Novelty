from pathlib import Path
import torch
import numpy as np
import pytest
from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import (
    EventWorldDataset,
    collate_event_batches,
    export_unity_trajectory_file,
    parse_unity_trajectory_file,
    simulate_articulated_trajectory,
)


def test_event_camera_simulator_shapes_and_sparsity():
    traj = simulate_articulated_trajectory(
        trajectory_id=1,
        length=25,
        joint_count=8,
        action_dim=8,
    )
    sim = EventCameraSimulator(height=32, width=32, event_threshold=0.08)

    batch = sim.trajectory_to_events(traj, return_sparse=True)
    events = batch.dense_events

    assert events.shape == (25, 2, 32, 32)
    # Check binary polarity: only 0.0 or 1.0 values
    unique_vals = torch.unique(events)
    assert all(val in [0.0, 1.0] for val in unique_vals.tolist())

    # Sparse representation check
    assert batch.sparse_events is not None
    assert len(batch.sparse_events) == 25

    # Check sparsity: event camera should be sparse (< 35% of pixels firing)
    active_fraction = (events > 0).float().mean().item()
    assert 0.001 < active_fraction < 0.35, f"Unexpected active fraction: {active_fraction}"


def test_unity_telemetry_serialization_and_dataset_collation(tmp_path: Path):
    sim = EventCameraSimulator(height=24, width=24, event_threshold=0.08)
    for tid in range(4):
        traj = simulate_articulated_trajectory(
            trajectory_id=tid,
            length=15,
            joint_count=6,
            action_dim=6,
        )
        export_unity_trajectory_file(traj, tmp_path / f"trajectory_{tid:05d}.json")

    parsed = parse_unity_trajectory_file(tmp_path / "trajectory_00000.json")
    assert parsed.trajectory_id == 0
    assert parsed.joint_positions.shape == (15, 6)
    assert parsed.joint_velocities.shape == (15, 6)
    assert parsed.actions.shape == (15, 6)

    ds = EventWorldDataset(
        trajectory_indices=[0, 1, 2, 3],
        event_simulator=sim,
        sequence_length=15,
        joint_count=6,
        action_dim=6,
        dataset_dir=tmp_path,
        cache_data=True,
    )

    assert len(ds) == 4
    item = ds[0]
    assert item["events"].shape == (15, 2, 24, 24)
    assert item["flat_kinematics"].shape == (15, 12)
    assert item["actions"].shape == (15, 6)

    # Test collation
    batch = collate_event_batches([ds[0], ds[1]])
    assert batch["events"].shape == (2, 15, 2, 24, 24)
    assert batch["flat_kinematics"].shape == (2, 15, 12)
    assert batch["actions"].shape == (2, 15, 6)
    assert batch["trajectory_ids"].shape == (2,)
