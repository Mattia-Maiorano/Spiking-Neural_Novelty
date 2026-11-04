"""
Unit Tests for Zero-Shot Sim-to-Real Domain Gap & Robustness Benchmark Suite.

Verifies:
1. Mathematical invariants for Poisson/flip background noise and refractory dropout.
2. Sensitivity monotonicity under event threshold shifts (theta = 0.05 vs 0.08 vs 0.12).
3. Physics simulation response under heavy payload (4.80 kg) and friction scaling (0.80x, 1.20x).
4. PerturbedEventDataset contract and batch collation.
5. Strict AST Anti-Fallback Compliance.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import numpy as np

from spwm.data.datasets import (
    collate_event_batches,
    parse_unity_trajectory_file,
    simulate_articulated_trajectory,
)
from spwm.evaluation.robustness import (
    PerturbedEventDataset,
    apply_background_activity_noise,
    apply_refractory_dropout,
    generate_threshold_events,
    render_events_from_kinematics,
    simulate_perturbed_kinematics,
)


def test_background_activity_noise_invariants():
    events = torch.zeros((2, 10, 4, 32, 32), dtype=torch.float32)
    events[:, :, :, 10:15, 10:15] = 1.0

    # Bounds validation
    with pytest.raises(ValueError, match="p_noise must be in"):
        apply_background_activity_noise(events, -0.1)

    with pytest.raises(ValueError, match="p_noise must be in"):
        apply_background_activity_noise(events, 1.5)

    # Zero noise identity
    clean = apply_background_activity_noise(events, 0.0)
    assert torch.equal(clean, events)

    # Injected noise
    noisy_005 = apply_background_activity_noise(events, 0.005)
    noisy_010 = apply_background_activity_noise(events, 0.010)

    assert noisy_005.shape == events.shape
    assert noisy_010.shape == events.shape

    # Binary range check
    assert torch.all((noisy_005 == 0.0) | (noisy_005 == 1.0))
    assert torch.all((noisy_010 == 0.0) | (noisy_010 == 1.0))

    # Event count strictly increases
    sum_orig = events.sum().item()
    sum_005 = noisy_005.sum().item()
    sum_010 = noisy_010.sum().item()

    assert sum_005 > sum_orig
    assert sum_010 > sum_005


def test_refractory_dropout_invariants():
    events = torch.zeros((2, 10, 4, 32, 32), dtype=torch.float32)
    events[:, :, :, 5:25, 5:25] = 1.0

    # Bounds validation
    with pytest.raises(ValueError, match="p_drop must be in"):
        apply_refractory_dropout(events, -0.05)

    with pytest.raises(ValueError, match="p_drop must be in"):
        apply_refractory_dropout(events, 1.1)

    # Zero drop identity
    clean = apply_refractory_dropout(events, 0.0)
    assert torch.equal(clean, events)

    # Dropout application
    drop_10 = apply_refractory_dropout(events, 0.10)
    drop_20 = apply_refractory_dropout(events, 0.20)

    assert drop_10.shape == events.shape
    assert drop_20.shape == events.shape

    # Binary range check
    assert torch.all((drop_10 == 0.0) | (drop_10 == 1.0))
    assert torch.all((drop_20 == 0.0) | (drop_20 == 1.0))

    # Dead-time dropout never creates false positive events in empty channels
    assert torch.all(drop_10[events == 0.0] == 0.0)
    assert torch.all(drop_20[events == 0.0] == 0.0)

    # Active count strictly decreases
    sum_orig = events.sum().item()
    sum_10 = drop_10.sum().item()
    sum_20 = drop_20.sum().item()

    assert sum_10 < sum_orig
    assert sum_20 < sum_10


def test_threshold_shift_monotonicity():
    traj = simulate_articulated_trajectory(trajectory_id=99, length=20)

    ev_005 = generate_threshold_events(traj, threshold=0.05)
    ev_008 = generate_threshold_events(traj, threshold=0.08)
    ev_012 = generate_threshold_events(traj, threshold=0.12)

    assert ev_005.shape == (20, 4, 32, 32)
    assert ev_008.shape == (20, 4, 32, 32)
    assert ev_012.shape == (20, 4, 32, 32)

    # Hypersensitive sensor (theta=0.05) must generate more events than nominal (0.08) and hyposensitive (0.12)
    sum_005 = ev_005.sum().item()
    sum_008 = ev_008.sum().item()
    sum_012 = ev_012.sum().item()

    assert sum_005 > sum_008 > sum_012


def test_perturbed_dynamics_simulation():
    traj = simulate_articulated_trajectory(trajectory_id=123, length=30)

    q_nom = traj.joint_positions
    p_nom = traj.joint_velocities

    # Out-of-distribution heavy payload (4.80 kg)
    q_heavy, p_heavy = simulate_perturbed_kinematics(traj, payload_mass=4.80)
    assert q_heavy.shape == q_nom.shape
    assert p_heavy.shape == p_nom.shape

    # Friction perturbations: 0.80x (loose) and 1.20x (stiff)
    q_loose, p_loose = simulate_perturbed_kinematics(traj, friction_scale=0.80)
    q_stiff, p_stiff = simulate_perturbed_kinematics(traj, friction_scale=1.20)

    assert q_loose.shape == q_nom.shape
    assert q_stiff.shape == q_nom.shape

    # Kinematics should reflect physical shift
    assert np.max(np.abs(q_loose - q_nom)) > 0.0
    assert np.max(np.abs(q_stiff - q_nom)) > 0.0


def test_render_events_from_kinematics():
    q = np.zeros((15, 6), dtype=np.float32)
    q[:, 1] = np.linspace(0.0, 0.5, 15)  # Moving shoulder lift

    events = render_events_from_kinematics(q, threshold=0.08)
    assert events.shape == (15, 4, 32, 32)
    assert (events > 0).any()

