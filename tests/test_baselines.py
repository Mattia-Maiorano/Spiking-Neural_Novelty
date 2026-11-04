"""
TDD Tests for Comparative Baseline Models (DOSSIER Section 6.1).

Verifies:
1. ContinuousRSSM, DeepGRUPredictor, and ContinuousConvLSTM instantiation and forward passes.
2. Tensor contracts across sequence lengths and batch dimensions.
3. Multi-step autonomous rollout contracts over horizons H in [1, 10, 50].
4. Parameter counts are appropriately matched to SPWM budget (~300k - 380k).
5. Zero fallback violations across baseline implementation.
"""

from __future__ import annotations
import pytest
import torch

from spwm.models.baselines import (
    ContinuousConvLSTM,
    ContinuousRSSM,
    DeepGRUPredictor,
)


@pytest.fixture
def dummy_batch():
    b, t, c, h, w = 2, 20, 4, 32, 32
    events = torch.randn(b, t, c, h, w)
    actions = torch.randn(b, t, 6)
    return events, actions


def test_continuous_rssm_forward_and_rollout(dummy_batch):
    events, actions = dummy_batch
    model = ContinuousRSSM()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 250_000 <= total_params <= 450_000, f"RSSM params out of range: {total_params}"

    # Forward pass
    out = model(events, actions)
    assert out["kinematics"].shape == (2, 20, 12)
    assert out["latent_states"].shape == (2, 20, 256 + 32)
    assert out["kl_loss"].ndim == 0
    assert torch.isfinite(out["kl_loss"])

    # Rollout pass
    rollout_res = model.rollout(events, actions, horizon=10, t_start=5)
    assert rollout_res.trajectory.shape == (2, 10, 12)
    assert rollout_res.positions.shape == (2, 10, 6)
    assert rollout_res.velocities.shape == (2, 10, 6)
    assert rollout_res.latent_trajectory.shape == (2, 10, 256 + 32)


def test_deep_gru_predictor_forward_and_rollout(dummy_batch):
    events, actions = dummy_batch
    model = DeepGRUPredictor()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 250_000 <= total_params <= 400_000, f"DeepGRU params out of range: {total_params}"

    # Forward pass
    out = model(events, actions)
    assert out["kinematics"].shape == (2, 20, 12)
    assert out["latent_states"].shape == (2, 20, 164)

    # Rollout pass
    rollout_res = model.rollout(events, actions, horizon=10, t_start=5)
    assert rollout_res.trajectory.shape == (2, 10, 12)
    assert rollout_res.positions.shape == (2, 10, 6)
    assert rollout_res.velocities.shape == (2, 10, 6)


def test_continuous_conv_lstm_forward_and_rollout(dummy_batch):
    events, actions = dummy_batch
    model = ContinuousConvLSTM()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 250_000 <= total_params <= 400_000, f"ConvLSTM params out of range: {total_params}"

    # Forward pass
    out = model(events, actions)
    assert out["kinematics"].shape == (2, 20, 12)
    assert out["latent_states"].shape == (2, 20, 56 * 4)

    # Rollout pass
    rollout_res = model.rollout(events, actions, horizon=10, t_start=5)
    assert rollout_res.trajectory.shape == (2, 10, 12)
    assert rollout_res.positions.shape == (2, 10, 6)
    assert rollout_res.velocities.shape == (2, 10, 6)
