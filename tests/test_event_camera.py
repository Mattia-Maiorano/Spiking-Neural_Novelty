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
        joint_count=6,
        action_dim=6,
    )
    assert traj.phase_flags is not None
    assert traj.phase_flags.shape == (25,)
    assert set(np.unique(traj.phase_flags).tolist()).issubset({0, 1, 2})
    assert {0, 1, 2}.issubset(set(np.unique(traj.phase_flags).tolist()))

    sim = EventCameraSimulator(height=32, width=32, event_threshold=0.08)

    batch = sim.trajectory_to_events(traj, return_sparse=True)
    events = batch.dense_events

    assert events.shape == (25, 4, 32, 32)
    # Check binary polarity: only 0.0 or 1.0 values
    unique_vals = torch.unique(events)
    assert all(val in [0.0, 1.0] for val in unique_vals.tolist())

    # Sparse representation check
    assert batch.sparse_events is not None
    assert len(batch.sparse_events) == 25

    # Check sparsity: event camera should be sparse (< 35% of pixels firing)
    active_fraction = (events > 0).float().mean().item()
    assert 0.001 < active_fraction < 0.35, f"Unexpected active fraction: {active_fraction}"

    # Verify both global (channels 0..1) and egocentric wrist (channels 2..3) streams emit events
    assert (events[:, :2] > 0).any(), "Global static camera stream should emit events!"
    assert (events[:, 2:4] > 0).any(), "Egocentric wrist camera stream should emit events!"


def test_dual_camera_resolves_wrist_roll_unobservability():
    """Verifies that pure axial wrist roll (q[5]) produces distinct egocentric wrist camera events."""
    sim = EventCameraSimulator(height=32, width=32, event_threshold=0.08)
    q_a = np.array([0.0, 0.35, 0.85, 0.35, 0.0, -0.80], dtype=np.float32)
    q_b = np.array([0.0, 0.35, 0.85, 0.35, 0.0, +0.80], dtype=np.float32)
    dual_a = sim.render_dual_articulated_frames(q_a)
    dual_b = sim.render_dual_articulated_frames(q_b)
    assert dual_a.shape == (2, 32, 32)
    assert dual_b.shape == (2, 32, 32)
    wrist_diff = float(np.abs(dual_a[1] - dual_b[1]).max())
    assert wrist_diff > 0.25, f"Egocentric wrist camera must strongly distinguish axial wrist roll, got {wrist_diff}"


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
    assert parsed.phase_flags is not None
    assert parsed.phase_flags.shape == (15,)

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
    assert item["events"].shape == (15, 4, 24, 24)
    assert item["flat_kinematics"].shape == (15, 12)
    assert item["actions"].shape == (15, 6)
    assert item["phase_flags"].shape == (15,)
    assert item["phase_flags"].dtype == torch.long

    # Test collation
    batch = collate_event_batches([ds[0], ds[1]])
    assert batch["events"].shape == (2, 15, 4, 24, 24)
    assert batch["flat_kinematics"].shape == (2, 15, 12)
    assert batch["actions"].shape == (2, 15, 6)
    assert batch["phase_flags"].shape == (2, 15)
    assert batch["trajectory_ids"].shape == (2,)


def test_event_camera_temporal_contrast_accumulation():
    """Verifies that slow continuous contrast accumulates until exceeding threshold and fires."""
    sim = EventCameraSimulator(height=4, width=4, event_threshold=0.1)
    frames = np.zeros((5, 4, 4), dtype=np.float32)
    for t in range(5):
        frames[t, 0, 0] = t * 0.03  # t=0: 0, t=1: 0.03, t=2: 0.06, t=3: 0.09, t=4: 0.12

    batch = sim.frames_to_events(frames)
    ev = batch.dense_events.numpy()

    # t=0..3: difference from ref (0.0) is < 0.1 -> no events
    assert ev[0, 0, 0, 0] == 0.0
    assert ev[1, 0, 0, 0] == 0.0
    assert ev[2, 0, 0, 0] == 0.0
    assert ev[3, 0, 0, 0] == 0.0
    # t=4: difference is 0.12 >= 0.1 -> positive event fires and ref updates to 0.12
    assert ev[4, 0, 0, 0] == 1.0
