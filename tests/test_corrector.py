"""
Unit Tests for SPWM-v8 Slow Neuromorphic Corrector & Two-Timescale Rollouts.
"""

import pytest
import torch
import torch.nn as nn

from spwm.models.corrector import SlowCorrector, SlowCorrectorState, SlowCorrectorOutput
from spwm.models.world_model import SPWM
from spwm.learning.losses import _frozen


def test_slow_corrector_initialization_and_shapes():
    B = 4
    q_dim = 32
    p_dim = 96
    corrector_dim = 64
    cadence = 5

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=corrector_dim,
        cadence=cadence,
        beta_mem=0.95,
        beta_adapt=0.995,
        v_th0=1.5,
    )

    state = corrector.init_state(B)
    assert state.step_count == 0
    assert state.alif_state.v_mem.shape == (B, corrector_dim)
    assert state.alif_state.spikes.shape == (B, corrector_dim)

    z_prev = torch.zeros(B, q_dim + p_dim)
    z_proposed = torch.zeros(B, q_dim + p_dim)

    z_corr, out, new_state = corrector.forward_step(z_prev, z_proposed, state)

    assert z_corr.shape == (B, q_dim + p_dim)
    assert out.delta_v.shape == (B, q_dim)
    assert out.delta_p.shape == (B, p_dim)
    assert out.damp_factor.shape == (B, p_dim)
    assert out.spikes.shape == (B, corrector_dim)
    assert out.intervened is True
    assert new_state.step_count == 1


def test_slow_corrector_quiescence_on_nominal_trajectory():
    """Under nominal trajectory (near-zero drift), elevated threshold ensures quiescence (zero spikes)."""
    B = 2
    corrector = SlowCorrector(
        q_dim=16,
        p_dim=48,
        corrector_dim=32,
        cadence=3,
        v_th0=2.0,  # High threshold
    )

    state = corrector.init_state(B)
    z_prev = torch.randn(B, 64) * 0.01
    z_prop = z_prev + torch.randn(B, 64) * 0.001  # very tiny deviation

    z_corr, out, new_state = corrector.forward_step(z_prev, z_prop, state)

    # Spikes should be exactly zero under quiescent baseline
    assert out.spikes.sum().item() == 0.0


def test_slow_corrector_cadence_and_decay():
    """Corrector intervenes on cadence (t % Delta == 0) and smoothly decays in between."""
    B = 2
    cadence = 4
    decay = 0.8
    corrector = SlowCorrector(
        q_dim=8,
        p_dim=24,
        corrector_dim=16,
        cadence=cadence,
        inter_step_decay=decay,
    )

    state = corrector.init_state(B)
    z = torch.randn(B, 32)

    interventions = []
    delta_vs = []

    for t in range(8):
        z_prop = z + 0.05
        z, out, state = corrector.forward_step(z, z_prop, state)
        interventions.append(out.intervened)
        delta_vs.append(out.delta_v.abs().mean().item())

    # Interventions should occur exactly at t = 0, 4 (step_count % 4 == 0)
    assert interventions == [True, False, False, False, True, False, False, False]

    # At step 1, delta_v should be decayed from step 0
    assert delta_vs[1] <= delta_vs[0] + 1e-6


def test_non_destructive_continuous_coupling():
    """Position coordinate q must not jump abruptly and must remain bounded in [-1, 1]."""
    B = 3
    q_dim = 8
    p_dim = 24
    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=16,
        max_gain_v=0.05,
    )

    state = corrector.init_state(B)
    q0 = torch.full((B, q_dim), 0.98)
    p0 = torch.zeros(B, p_dim)
    z_prev = torch.cat([q0, p0], dim=-1)

    # Large proposed jump that would overshoot 1.0
    q_prop = torch.full((B, q_dim), 1.2)
    p_prop = torch.zeros(B, p_dim)
    z_prop = torch.cat([q_prop, p_prop], dim=-1)

    z_corr, out, _ = corrector.forward_step(z_prev, z_prop, state)
    q_corr = z_corr[:, :q_dim]

    # Clamped within valid bounds [-1, 1]
    assert (q_corr <= 1.0).all()
    assert (q_corr >= -1.0).all()


def test_spwm_two_timescale_rollout():
    """SPWM predict_future executes two-timescale rollout when corrector is enabled."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
        corrector_cadence=5,
    )

    z0 = torch.randn(2, 32)

    # 1. Rollout with corrector enabled
    traj_with_corr = model.predict_future(z0, horizon=25, use_corrector=True)
    assert traj_with_corr.shape == (2, 25, 32)
    assert not torch.isnan(traj_with_corr).any()

    # 2. Rollout without corrector
    traj_no_corr = model.predict_future(z0, horizon=25, use_corrector=False)
    assert traj_no_corr.shape == (2, 25, 32)
    assert not torch.isnan(traj_no_corr).any()

    # 3. Differentiable predict_rollout
    traj_diff, corr_spikes = model.predict_rollout(z0, horizon=15, use_corrector=True)
    assert traj_diff.shape == (2, 15, 32)
    assert corr_spikes is not None
    assert corr_spikes.shape == (2, 15, 16)


def test_protected_block_gradient_isolation():
    """Verifies that freezing the fast predictor prevents long-horizon rollout gradients
    from reaching the fast predictor parameters during corrector optimization."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
        corrector_cadence=3,
    )

    z0 = torch.randn(2, 32, requires_grad=False)
    target = torch.randn(2, 10, 32)

    corrector_optimizer = torch.optim.Adam(model.corrector.parameters(), lr=1e-3)

    # Execute protected rollout with fast predictor frozen
    with _frozen(model.predictor):
        pred_rollout, spikes = model.predict_rollout(z0, horizon=10, use_corrector=True)
        loss = nn.functional.mse_loss(pred_rollout, target) + 0.1 * spikes.mean()

        corrector_optimizer.zero_grad()
        loss.backward()

    # Corrector parameters MUST have gradients
    corrector_has_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.corrector.parameters()
    )
    assert corrector_has_grads, "Corrector parameters should receive gradients!"

    # Fast predictor parameters MUST NOT have gradients
    for p in model.predictor.parameters():
        assert p.grad is None, "Fast predictor parameters must remain strictly gradient-isolated!"
