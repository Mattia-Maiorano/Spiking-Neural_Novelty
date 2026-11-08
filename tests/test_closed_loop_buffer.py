"""Unit tests for ClosedLoopBuffer and DAgger-style closed-loop replay buffer in SPWM."""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from spwm.training.engine import ClosedLoopBuffer
from spwm.contracts.config import ExperimentConfig, TrainingConfig, LossConfig, EnvironmentConfig, DataConfig, ModelConfig
from spwm.training.phases.m1_nominal import M1PhaseStrategy
from spwm.models.spwm import SPWM


def test_closed_loop_buffer_push_sample_eviction():
    buf = ClosedLoopBuffer(capacity=5)
    assert len(buf) == 0

    transitions = [
        {"z": torch.randn(128), "u": torch.randn(6), "z_next": torch.randn(128)}
        for _ in range(3)
    ]
    buf.push(transitions)
    assert len(buf) == 3

    # Sample batch
    batch = buf.sample(batch_size=2)
    assert batch["z"].shape == (2, 128)
    assert batch["u"].shape == (2, 6)
    assert batch["z_next"].shape == (2, 128)

    # Push more to test FIFO circular buffer eviction
    more_transitions = [
        {"z": torch.randn(128), "u": torch.randn(6), "z_next": torch.randn(128)}
        for _ in range(4)
    ]
    buf.push(more_transitions)
    assert len(buf) == 5

    # Test state_dict and load_state_dict
    state = buf.state_dict()
    new_buf = ClosedLoopBuffer(capacity=5)
    new_buf.load_state_dict(state)
    assert len(new_buf) == 5


def test_m1_phase_strategy_mixed_batch():
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(height=32, width=32, joint_count=6, action_dim=6),
        data=DataConfig(sequence_length=15, joint_count=6, action_dim=6),
        model=ModelConfig(in_channels=2, encoder_dim=32, latent_dim=64, q_dim=16, p_dim=48, num_keypoints=8),
        training=TrainingConfig(mode="train_m1", phase_horizon=4, learning_rate=1e-3),
        loss=LossConfig(),
    )
    model = SPWM.from_config(cfg).to(device)
    strategy = M1PhaseStrategy(model=model, cfg=cfg, loss_fn=nn.Identity(), device=device)
    strategy.setup()

    B, T = 2, 15
    events = torch.randn(B, T, 2, 32, 32)
    actions = torch.randn(B, T, 6)

    # Standard batch without dagger
    batch_standard = {"events": events, "actions": actions}
    m1 = strategy.train_step(batch_standard, epoch=1)
    assert "train_loss" in m1

    # Mixed batch with dagger closed-loop transitions
    dagger_batch = {
        "z": torch.randn(4, 64),
        "u": torch.randn(4, 6),
        "z_next": torch.randn(4, 64),
    }
    batch_mixed = {"events": events, "actions": actions, "closed_loop": dagger_batch}
    m2 = strategy.train_step(batch_mixed, epoch=1)
    assert "train_loss" in m2
    assert "train_dagger_batch_size" in m2
    assert m2["train_dagger_batch_size"] == 4.0


def test_closed_loop_buffer_trajectories_bptt():
    buf = ClosedLoopBuffer(capacity=100)
    trajs = [
        {
            "z_init": torch.randn(64),
            "actions": torch.randn(15, 6),
            "targets": torch.randn(15, 64),
            "length": 15,
            "trajectory_id": i,
        }
        for i in range(4)
    ]
    buf.push_trajectories(trajs)
    assert len(buf.trajectories) == 4
    assert len(buf) == 60  # 4 trajs * 15 transitions

    # Sample trajectory batch for BPTT
    H = 5
    cl_batch = buf.sample_trajectories(batch_size=3, horizon=H)
    assert cl_batch["z_init"].shape == (3, 64)
    assert cl_batch["actions"].shape == (3, H, 6)
    assert cl_batch["targets"].shape == (3, H, 64)

    # Test with un-squeezed targets [L, 1, D] and verify shape alignment across non-zero start_idx
    buf3d = ClosedLoopBuffer(capacity=100)
    trajs3d = [
        {
            "z_init": torch.randn(64),
            "actions": torch.randn(20, 1, 6),
            "targets": torch.randn(20, 1, 64),
            "length": 20,
            "trajectory_id": i,
        }
        for i in range(5)
    ]
    buf3d.push_trajectories(trajs3d)
    for _ in range(10):
        cl3d = buf3d.sample_trajectories(batch_size=4, horizon=H)
        assert cl3d["z_init"].shape == (4, 64)
        assert cl3d["actions"].shape == (4, H, 6)
        assert cl3d["targets"].shape == (4, H, 64)

    # Test M1 Phase Strategy with multi-step BPTT
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(height=32, width=32, joint_count=6, action_dim=6),
        data=DataConfig(sequence_length=15, joint_count=6, action_dim=6),
        model=ModelConfig(in_channels=2, encoder_dim=32, latent_dim=64, q_dim=16, p_dim=48, num_keypoints=8),
        training=TrainingConfig(mode="train_m1", phase_horizon=H, learning_rate=1e-3),
        loss=LossConfig(),
    )
    model = SPWM.from_config(cfg).to(device)
    strategy = M1PhaseStrategy(model=model, cfg=cfg, loss_fn=nn.Identity(), device=device)
    strategy.setup()

    events = torch.randn(2, 15, 2, 32, 32)
    actions = torch.randn(2, 15, 6)
    batch = {"events": events, "actions": actions, "closed_loop": cl_batch}

    metrics = strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics
    assert metrics["train_dagger_batch_size"] == 3.0

    # Verify gradients were accumulated on M1 parameters through BPTT
    grad_found = False
    for p in model.m1.parameters():
        if p.grad is not None and torch.norm(p.grad) > 0:
            grad_found = True
            break
    assert grad_found, "M1 predictor must have non-zero gradients from BPTT!"


def test_resolve_closed_loop_buffer(tmp_path):
    from spwm.training.engine import resolve_closed_loop_buffer

    # Empty dir returns None
    res = resolve_closed_loop_buffer("spwm_v0_8_3", "train_m1", results_root=tmp_path)
    assert res is None

    # Sibling run buffer in results_root must NOT be loaded on fresh run (anti-contamination rule)
    cand_dir = tmp_path / "spwm_v0_8_3_train_m1_BEFOREBPTT"
    cand_dir.mkdir(parents=True)
    buf_path = cand_dir / "closed_loop_buffer.pt"
    buf = ClosedLoopBuffer(capacity=10)
    buf.push([{"z": torch.randn(10), "u": torch.randn(2), "z_next": torch.randn(10)} for _ in range(5)])
    buf.save(buf_path)

    # Fresh run without explicit path or is_resuming ignores older run
    res = resolve_closed_loop_buffer("spwm_v0_8_3", "train_m1", results_root=tmp_path)
    assert res is None

    # Explicit path loads it
    res_exp = resolve_closed_loop_buffer("spwm_v0_8_3", "train_m1", explicit_path=buf_path)
    assert res_exp == buf_path

    # Resuming run in save_dir loads it
    res_resuming = resolve_closed_loop_buffer(
        "spwm_v0_8_3", "train_m1", save_dir=cand_dir, is_resuming=True
    )
    assert res_resuming == buf_path


def test_closed_loop_buffer_generation_fifo_rolling_window():
    """Test strict FIFO rolling window across Tier 2 evaluation generations."""
    buf = ClosedLoopBuffer(capacity=1000, max_eval_generations=2)
    assert buf.max_eval_generations == 2
    assert len(buf.eval_generations) == 0

    # Generation 1 (traj ids 101, 102)
    gen1 = [
        {"z_init": torch.randn(32), "actions": torch.randn(10, 4), "targets": torch.randn(10, 32), "length": 10, "trajectory_id": 101},
        {"z_init": torch.randn(32), "actions": torch.randn(10, 4), "targets": torch.randn(10, 32), "length": 10, "trajectory_id": 102},
    ]
    buf.push_trajectories(gen1)
    assert len(buf.eval_generations) == 1
    assert len(buf.trajectories) == 2
    assert len(buf) == 20
    assert {t["trajectory_id"] for t in buf.trajectories} == {101, 102}

    # Generation 2 (traj ids 201, 202)
    gen2 = [
        {"z_init": torch.randn(32), "actions": torch.randn(10, 4), "targets": torch.randn(10, 32), "length": 10, "trajectory_id": 201},
        {"z_init": torch.randn(32), "actions": torch.randn(10, 4), "targets": torch.randn(10, 32), "length": 10, "trajectory_id": 202},
    ]
    buf.push_trajectories(gen2)
    assert len(buf.eval_generations) == 2
    assert len(buf.trajectories) == 4
    assert len(buf) == 40
    assert {t["trajectory_id"] for t in buf.trajectories} == {101, 102, 201, 202}

    # Generation 3 (traj ids 301, 302): Gen 1 should be cleanly discarded!
    gen3 = [
        {"z_init": torch.randn(32), "actions": torch.randn(10, 4), "targets": torch.randn(10, 32), "length": 10, "trajectory_id": 301},
        {"z_init": torch.randn(32), "actions": torch.randn(10, 4), "targets": torch.randn(10, 32), "length": 10, "trajectory_id": 302},
    ]
    buf.push_trajectories(gen3)
    assert len(buf.eval_generations) == 2
    assert len(buf.trajectories) == 4
    assert len(buf) == 40
    # Strictly Gen 2 and Gen 3 remain; Gen 1 is completely purged
    assert {t["trajectory_id"] for t in buf.trajectories} == {201, 202, 301, 302}

    # Verify serialization preserves generations
    sd = buf.state_dict()
    restored = ClosedLoopBuffer(capacity=1000, max_eval_generations=2)
    restored.load_state_dict(sd)
    assert len(restored.eval_generations) == 2
    assert {t["trajectory_id"] for t in restored.trajectories} == {201, 202, 301, 302}


def test_dynamic_dagger_ratio_scheduler():
    """Test linear epoch-based DAgger batch ratio scheduler."""
    from spwm.training.noise import compute_dagger_ratio

    # Epochs 0-10 strictly 0%
    for ep in range(0, 11):
        assert compute_dagger_ratio(ep) == 0.0

    # Epoch 11: 15%
    assert abs(compute_dagger_ratio(11) - 0.15) < 1e-6

    # Epoch 40: 60%
    assert abs(compute_dagger_ratio(40) - 0.60) < 1e-6

    # Epoch > 40: hard capped at 60%
    assert abs(compute_dagger_ratio(50) - 0.60) < 1e-6
    assert abs(compute_dagger_ratio(100) - 0.60) < 1e-6

    # Monotonically increasing from 11 to 40
    prev = compute_dagger_ratio(11)
    for ep in range(12, 41):
        curr = compute_dagger_ratio(ep)
        assert curr > prev
        prev = curr


def test_recalibrated_mppi_noise_curriculum():
    """Test recalibrated MPPI noise curriculum parameters."""
    from spwm.training.noise import compute_mppi_noise_alpha

    # Default parameters: warmup 10, ramp 40, target alpha 0.5
    assert compute_mppi_noise_alpha(1) == 0.0
    assert compute_mppi_noise_alpha(10) == 0.0
    assert abs(compute_mppi_noise_alpha(30) - 0.25) < 1e-6
    assert abs(compute_mppi_noise_alpha(50) - 0.50) < 1e-6
    assert abs(compute_mppi_noise_alpha(70) - 0.50) < 1e-6



