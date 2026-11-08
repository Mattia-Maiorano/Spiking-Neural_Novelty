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

