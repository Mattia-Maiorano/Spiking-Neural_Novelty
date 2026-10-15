"""
Unit Tests for SPWM-v0.1.0 Neuromodulatory Gain Field Corrector & Two-Timescale Rollouts.
"""

import pytest
import torch
import torch.nn as nn

from spwm.core.corrector import SlowCorrector, SlowCorrectorState, SlowCorrectorOutput
from spwm.models.world_model import SPWM
from spwm.learning.losses import _frozen


def test_slow_corrector_initialization_and_shapes():
    """Verifies that SlowCorrector initializes with correct shapes for v0.1.0 Gain Field."""
    B = 4
    q_dim = 32
    p_dim = 96
    corrector_dim = 128
    latent_dim = q_dim + p_dim

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=corrector_dim,
        beta_mem=0.95,
        beta_adapt=0.94,
        v_th0=1.0,
        gamma=0.25,
        alpha_gain=0.5,
    )

    state = corrector.init_state(B)
    assert state.step_count == 0
    assert state.alif_state.v_mem.shape == (B, corrector_dim)
    assert state.alif_state.spikes.shape == (B, corrector_dim)
    assert state.last_gain_mask.shape == (B, latent_dim)

    z_prev = torch.zeros(B, latent_dim)
    z_proposed = torch.zeros(B, latent_dim)

    z_corr, out, new_state = corrector.forward_step(z_prev, z_proposed, state)

    assert z_corr.shape == (B, latent_dim)
    assert out.gain_mask.shape == (B, latent_dim)
    assert out.spikes.shape == (B, corrector_dim)
    assert out.v_mem.shape == (B, corrector_dim)
    assert out.intervened is True
    assert new_state.step_count == 1
    assert new_state.last_gain_mask.shape == (B, latent_dim)


def test_gain_mask_emission_bounded():
    """Verifies that the gain mask is strictly bounded within [-1.0, 1.0]."""
    B = 4
    q_dim = 16
    p_dim = 48
    corrector_dim = 32

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=corrector_dim,
        alpha_gain=0.5,
    )

    state = corrector.init_state(B)

    # Test across multiple steps with large inputs
    for _ in range(10):
        z_prev = torch.randn(B, q_dim + p_dim) * 5.0
        z_proposed = z_prev + torch.randn(B, q_dim + p_dim) * 2.0
        z_corr, out, state = corrector.forward_step(z_prev, z_proposed, state)

        assert torch.all(out.gain_mask >= -1.0), "gain_mask below -1.0!"
        assert torch.all(out.gain_mask <= 1.0), "gain_mask above 1.0!"
        assert not torch.isnan(out.gain_mask).any()


def test_proportional_gain_modulation():
    """Verifies that proportional modulation satisfies Δz_mod = Δz_fast * (1 + alpha_gain * G_t)."""
    B = 2
    q_dim = 8
    p_dim = 24
    latent_dim = q_dim + p_dim
    corrector_dim = 16
    alpha_gain = 0.4

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=corrector_dim,
        alpha_gain=alpha_gain,
    )

    state = corrector.init_state(B)
    # Stay strictly within boundary [-0.5, 0.5] so clamping doesn't trigger
    z_prev = torch.zeros(B, latent_dim)
    delta_z_fast = torch.full((B, latent_dim), 0.05)
    z_fast_proposed = z_prev + delta_z_fast

    z_corr, out, new_state = corrector.forward_step(z_prev, z_fast_proposed, state, is_burn_in=False)

    expected_delta_mod = delta_z_fast * (1.0 + alpha_gain * out.gain_mask)
    expected_z_corr = z_prev + expected_delta_mod

    assert torch.allclose(z_corr, expected_z_corr, atol=1e-5)


def test_burn_in_period():
    """Verifies that is_burn_in=True ignores the mask without altering the trajectory."""
    B = 3
    q_dim = 8
    p_dim = 24
    latent_dim = q_dim + p_dim
    corrector_dim = 16

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=corrector_dim,
        alpha_gain=0.8,
    )

    state = corrector.init_state(B)
    z_prev = torch.randn(B, latent_dim) * 0.1
    z_proposed = z_prev + torch.randn(B, latent_dim) * 0.1

    z_corr, out, new_state = corrector.forward_step(
        z_prev, z_proposed, state, is_burn_in=True
    )

    # During burn-in, trajectory is untouched
    assert torch.allclose(z_corr, z_proposed)
    assert out.intervened is False
    # But internal state and gain_mask must still advance
    assert new_state.step_count == 1
    assert out.gain_mask.shape == (B, latent_dim)
    assert not torch.isnan(new_state.alif_state.v_mem).any()


def test_euclidean_boundary_clamping():
    """Verifies Euclidean boundary clamping [-1.0, 1.0] on q and momentum dissipation on collision."""
    B = 2
    q_dim = 4
    p_dim = 12
    latent_dim = q_dim + p_dim

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=16,
        alpha_gain=0.5,
    )

    state = corrector.init_state(B)

    # Create proposed state with q exceeding boundaries
    z_prev = torch.zeros(B, latent_dim)
    q_overshoot = torch.tensor([[1.5, -1.8, 0.5, -0.2], [2.0, -2.0, 0.0, 0.0]])
    p_initial = torch.ones(B, p_dim)
    z_proposed = torch.cat([q_overshoot, p_initial], dim=-1)

    z_corr, out, _ = corrector.forward_step(z_prev, z_proposed, state, is_burn_in=False)

    q_corr = z_corr[:, :q_dim]
    p_corr = z_corr[:, q_dim:]

    # q must be strictly clamped in [-1.0, 1.0]
    assert (q_corr <= 1.0).all()
    assert (q_corr >= -1.0).all()

    # Hit upper on index 0, hit lower on index 1 -> momentum components corresponding to collisions dissipate
    assert (p_corr[:, :6] < p_initial[:, :6]).all()


def test_spwm_two_timescale_rollout():
    """SPWM predict_future executes two-timescale rollout when corrector is enabled."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
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


def test_chained_window_rollout_contraction():
    """Verifies that chained window rollout executes Window A and Window B with detached intermediate state."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )

    B = 2
    H = 5
    z0 = torch.randn(B, 32)
    target_A = torch.randn(B, H, 32)
    target_B = torch.randn(B, H, 32)

    optimizer = torch.optim.Adam(model.corrector.parameters(), lr=1e-3)

    with _frozen(model.predictor):
        # Window A
        pred_A, spikes_A, corr_state_A = model.predict_rollout(
            z0, horizon=H, use_corrector=True, return_corrector_state=True
        )
        loss_A = nn.functional.mse_loss(pred_A, target_A)

        # Detached state transfer
        z_init_B = pred_A[:, -1].detach()
        corr_state_B = corr_state_A.detach()
        assert not z_init_B.requires_grad
        assert not corr_state_B.alif_state.v_mem.requires_grad

        # Window B
        pred_B, spikes_B = model.predict_rollout(
            z_init_B, horizon=H, use_corrector=True, initial_corrector_state=corr_state_B
        )
        loss_B = nn.functional.mse_loss(pred_B, target_B)

        total_loss = loss_A + loss_B
        optimizer.zero_grad()
        total_loss.backward()

    # Corrector must have gradients
    corrector_grads = [p.grad for p in model.corrector.parameters() if p.grad is not None]
    assert len(corrector_grads) > 0
    assert any(g.abs().sum() > 0 for g in corrector_grads)

    # Predictor must NOT have gradients
    assert all(p.grad is None for p in model.predictor.parameters())


def test_deep_horizon_rollout_h100():
    """Verifies autonomous and differentiable rollout up to extended deep horizon H=100 (v0.1.0)."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )

    B = 2
    z0 = torch.randn(B, 32)

    # 1. Autonomous rollout for 100 steps
    traj_100 = model.predict_future(z0, horizon=100, use_corrector=True)
    assert traj_100.shape == (B, 100, 32)
    assert torch.isfinite(traj_100).all()

    # 2. Differentiable rollout with fast trajectory tracking
    with _frozen(model.predictor):
        pred_rollout, spikes, fast_rollout = model.predict_rollout(
            z0, horizon=50, use_corrector=True, return_fast_trajectory=True
        )
        assert pred_rollout.shape == (B, 50, 32)
        assert fast_rollout.shape == (B, 50, 32)
        assert spikes.shape == (B, 50, 16)
        loss = pred_rollout.sum()
        loss.backward()

    corrector_grads = [p.grad for p in model.corrector.parameters() if p.grad is not None]
    assert len(corrector_grads) > 0
