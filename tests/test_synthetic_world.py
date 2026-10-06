import numpy as np
import pytest
from spwm.data.synthetic_world import MovingObjectsWorld, partition_trajectories


def test_trajectory_generation_shapes_and_bounds():
    world = MovingObjectsWorld(box_size=1.0, default_radius=0.1)
    traj = world.generate_trajectory(trajectory_id=42, length=30, num_objects=2, seed=42)

    assert len(traj.states) == 30
    assert traj.positions.shape == (30, 2, 2)
    assert traj.velocities.shape == (30, 2, 2)
    assert traj.flat_kinematics.shape == (30, 8)

    # Check bounds: all positions within [-1.0, 1.0] taking radius into account
    max_pos = np.max(np.abs(traj.positions))
    assert max_pos <= 1.0


def test_reproducibility():
    world = MovingObjectsWorld()
    traj1 = world.generate_trajectory(trajectory_id=10, length=20, seed=10)
    traj2 = world.generate_trajectory(trajectory_id=10, length=20, seed=10)
    traj3 = world.generate_trajectory(trajectory_id=11, length=20, seed=11)

    np.testing.assert_allclose(traj1.positions, traj2.positions)
    np.testing.assert_allclose(traj1.velocities, traj2.velocities)
    # Different seed should differ
    assert not np.allclose(traj1.positions, traj3.positions)


def test_trajectory_partitioning_disjoint():
    splits = partition_trajectories(total_trajectories=1000, train_split=0.8, val_split=0.1)
    train_range = splits["train"]
    val_range = splits["val"]
    test_range = splits["test"]

    assert train_range == (0, 800)
    assert val_range == (800, 900)
    assert test_range == (900, 1000)

    train_set = set(range(*train_range))
    val_set = set(range(*val_range))
    test_set = set(range(*test_range))

    # Strict disjointness
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0


def test_drift_injection_and_perturbations():
    world = MovingObjectsWorld()
    # Nominal trajectory (drift_prob = 0)
    traj_nom = world.generate_trajectory(trajectory_id=5, length=50, drift_injection_prob=0.0, seed=5)
    # Trajectory with synthetic drift injection
    traj_drift = world.generate_trajectory(trajectory_id=5, length=50, drift_injection_prob=0.3, drift_magnitude=0.1, seed=5)

    assert len(traj_drift.states) == 50
    # Both trajectories start identically at t=0
    np.testing.assert_allclose(traj_nom.positions[0], traj_drift.positions[0])
    # Over time, synthetic drift perturbates the trajectory forcing OOD divergence
    assert not np.allclose(traj_nom.positions, traj_drift.positions)

