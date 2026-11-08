"""Unit & Contract Tests for ANNWorldModel Continuous Leaky Rate Ablation (DOSSIER Section 6.2)."""

from __future__ import annotations
import pytest
import torch

from spwm.contracts.protocols import WorldModelProtocol
from spwm.models.ann_ablation import (
    AblationRolloutResult,
    ANNWorldModel,
    ContinuousRateDynamics,
    ContinuousRateMemory,
    ContinuousRateState,
)
from spwm.models.world_model import SPWM


def test_ann_world_model_parameter_parity():
    """ANNWorldModel must have 100% parameter count parity with SPWM."""
    spwm = SPWM()
    ann = ANNWorldModel()

    spwm_params = sum(p.numel() for p in spwm.parameters() if p.requires_grad)
    ann_params = sum(p.numel() for p in ann.parameters() if p.requires_grad)

    assert spwm_params == 328652, f"SPWM parameter count changed unexpectedly: {spwm_params}"
    assert ann_params == spwm_params, f"Parameter count mismatch: SPWM={spwm_params} vs ANN={ann_params}"


def test_ann_world_model_protocol_compliance():
    """ANNWorldModel must satisfy the Layer 0 WorldModelProtocol contract."""
    ann = ANNWorldModel()
    assert isinstance(ann, WorldModelProtocol)
    assert hasattr(ann, "m1")
    assert ann.latent_dim == 128
    assert ann.q_dim == 32
    assert ann.p_dim == 96
    assert ann.encoder_dim == 128
    assert ann.joint_count == 6


def test_continuous_rate_memory_dynamics():
    """ContinuousRateMemory must strictly compute m_t = beta * m_{t-1} + (1 - beta) * I_t and r_t = sigmoid(m_t)."""
    mem = ContinuousRateMemory(timescale_dims=(4, 4), betas=(0.5, 0.8))
    state = mem.init_state(batch_size=2)

    # First step with constant input
    inp = torch.ones(2, 8)
    rates, next_state = mem(inp, state=state)

    assert rates.shape == (2, 8)
    assert (rates > 0.0).all() and (rates < 1.0).all()

    # Verify Tier 0: beta=0.5 -> m_1 = 0.5 * 0 + 0.5 * 1.0 = 0.5
    tier0_m = next_state.m_mem[0]
    expected_m0 = torch.full((2, 4), 0.5)
    assert torch.allclose(tier0_m, expected_m0)
    assert torch.allclose(next_state.rates[0], torch.sigmoid(expected_m0))

    # Verify Tier 1: beta=0.8 -> m_1 = 0.8 * 0 + 0.2 * 1.0 = 0.2
    tier1_m = next_state.m_mem[1]
    expected_m1 = torch.full((2, 4), 0.2)
    assert torch.allclose(tier1_m, expected_m1)
    assert torch.allclose(next_state.rates[1], torch.sigmoid(expected_m1))


def test_ann_world_model_forward_and_rollout():
    """ANNWorldModel forward pass and multi-step rollout test."""
    model = ANNWorldModel(
        latent_dim=64,
        q_dim=16,
        p_dim=48,
        num_keypoints=8,
        timescale_dims=(32, 32),
        predictor_hidden_dim=128,
    )
    b, t = 2, 8
    events = torch.randn(b, t, 4, 32, 32)
    actions = torch.randn(b, t, 6)

    # Forward
    out = model(events, actions=actions)
    assert out["positions"].shape == (b, t, 6)
    assert out["velocities"].shape == (b, t, 6)
    assert out["latents"].shape == (b, t, 64)

    # Rollout
    res = model.rollout(events, actions, horizon=5, t_start=2)
    assert isinstance(res, AblationRolloutResult)
    assert res.trajectory.shape == (b, 5, 12)
    assert res.positions.shape == (b, 5, 6)
    assert res.velocities.shape == (b, 5, 6)
    assert res.latent_trajectory.shape == (b, 5, 64)

