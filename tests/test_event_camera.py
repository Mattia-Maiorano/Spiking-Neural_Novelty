import torch
import numpy as np
import pytest
from spwm.data.synthetic_world import MovingObjectsWorld
from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import EventWorldDataset, collate_event_batches


def test_event_camera_simulator_shapes_and_sparsity():
    world = MovingObjectsWorld()
    traj = world.generate_trajectory(trajectory_id=1, length=25, num_objects=1)
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

    # Check sparsity: event camera should be sparse (< 25% of pixels firing)
    active_fraction = (events > 0).float().mean().item()
    assert 0.001 < active_fraction < 0.25, f"Unexpected active fraction: {active_fraction}"


def test_dataset_and_dataloader_collation():
    world = MovingObjectsWorld()
    sim = EventCameraSimulator(height=24, width=24)
    ds = EventWorldDataset(
        trajectory_indices=[0, 1, 2, 3],
        world=world,
        event_simulator=sim,
        sequence_length=15,
        num_objects=1,
        cache_data=True,
    )

    assert len(ds) == 4
    item = ds[0]
    assert item["events"].shape == (15, 2, 24, 24)
    assert item["flat_kinematics"].shape == (15, 4)

    # Test collation
    batch = collate_event_batches([ds[0], ds[1]])
    assert batch["events"].shape == (2, 15, 2, 24, 24)
    assert batch["flat_kinematics"].shape == (2, 15, 4)
    assert batch["trajectory_ids"].shape == (2,)
