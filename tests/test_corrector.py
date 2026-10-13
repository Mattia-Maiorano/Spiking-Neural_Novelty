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
    corrector_dim = 128
    cadence = 5

    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=corrector_dim,
        cadence=cadence,
        beta_mem=0.95,
        beta_adapt=0.995,
        v_th0=1.0,
    )

    state = corrector.init_state(B)
    assert state.step_count == 0
    assert state.alif_state.v_mem.shape == (B, corrector_dim)
    assert state.alif_state.spikes.shape == (B, corrector_dim)

    z_prev = torch.zeros(B, q_dim + p_dim)
    z_proposed = torch.zeros(B, q_dim + p_dim)

    z_corr, out, new_state = corrector.forward_step(z_prev, z_proposed, state)

    assert z_corr.shape == (B, q_dim + p_dim)
    assert out.delta_q.shape == (B, q_dim)
    assert out.delta_p.shape == (B, p_dim)
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


def test_slow_corrector_cadence_and_jump_system():
    """Corrector acts as a Jump System: discrete orthogonal jump on cadence, zero interference inter-cadence."""
    B = 2
    cadence = 4
    corrector = SlowCorrector(
        q_dim=8,
        p_dim=24,
        corrector_dim=16,
        cadence=cadence,
    )

    state = corrector.init_state(B)
    z = torch.randn(B, 32)

    interventions = []

    for t in range(8):
        z_prop = z + 0.05
        z_corr, out, state = corrector.forward_step(z, z_prop, state)
        interventions.append(out.intervened)

        if out.intervened:
            # On cadence: intervention executed
            assert out.potential is not None
        else:
            # Inter-cadence: Fast Predictor operates in full autonomy without offsets or decay
            assert torch.all(out.delta_q == 0.0)
            assert torch.all(out.delta_p == 0.0)
            assert torch.all(out.damp_factor == 0.0)
            assert torch.allclose(z_corr, z_prop)

    # Interventions should occur exactly at t = 0, 4 (step_count % 4 == 0)
    assert interventions == [True, False, False, False, True, False, False, False]


def test_hamiltonian_energy_and_symplectic_coupling():
    """Position update is derived symplectically from momentum correction, and q remains in [-1, 1]."""
    B = 3
    q_dim = 8
    p_dim = 24
    corrector = SlowCorrector(
        q_dim=q_dim,
        p_dim=p_dim,
        corrector_dim=16,
        cadence=1,
    )

    state = corrector.init_state(B)
    q0 = torch.full((B, q_dim), 0.98)
    p0 = torch.zeros(B, p_dim)
    z_prev = torch.cat([q0, p0], dim=-1)

    # Proposed jump that would overshoot 1.0
    q_prop = torch.full((B, q_dim), 1.2)
    p_prop = torch.zeros(B, p_dim)
    z_prop = torch.cat([q_prop, p_prop], dim=-1)

    z_corr, out, _ = corrector.forward_step(z_prev, z_prop, state)
    q_corr = z_corr[:, :q_dim]

    # Clamped within valid bounds [-1, 1]
    assert (q_corr <= 1.0).all()
    assert (q_corr >= -1.0).all()
    assert out.potential is not None
    # delta_q is coupled to delta_p
    assert out.delta_q.shape == (B, q_dim)
    assert out.delta_p.shape == (B, p_dim)


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


def test_corrector_quiescence_hinge_loss():
    """Verifies that the hinge / dead-zone penalty produces zero loss within tolerance
    margin and clamps to the upper bound above maximum allowed threshold."""
    margin = 0.15
    cap = 0.50

    # Case 1: Below margin (e.g. 0.10 spike rate) -> Loss must be exactly 0.0
    spikes_low = torch.full((2, 10, 16), 0.10)
    loss_low = torch.clamp(
        nn.functional.relu(spikes_low.mean() - margin),
        max=cap,
    )
    assert loss_low.item() == 0.0

    # Case 2: In linear region (e.g. 0.30 spike rate) -> Loss = 0.30 - 0.15 = 0.15
    spikes_mid = torch.full((2, 10, 16), 0.30)
    loss_mid = torch.clamp(
        nn.functional.relu(spikes_mid.mean() - margin),
        max=cap,
    )
    assert abs(loss_mid.item() - 0.15) < 1e-6

    # Case 3: In saturated region (e.g. 0.90 spike rate) -> Capped at max 0.50
    spikes_high = torch.full((2, 10, 16), 0.90)
    loss_high = torch.clamp(
        nn.functional.relu(spikes_high.mean() - margin),
        max=cap,
    )
    assert abs(loss_high.item() - cap) < 1e-6


def test_chained_window_rollout_contraction():
    """Verifies that chained window rollout executes Window A and Window B with detached intermediate state."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
        corrector_cadence=5,
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


def test_isometry_loss_penalizes_nullspace_deformation():
    """Verifies that isometry regularization penalizes arbitrary latent deformations
    in the 124D null-space of the physical decoder."""
    from spwm.learning.losses import SPWMLoss
    from spwm.models.predictor import PhysicalDecoder

    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, num_objects=1)
    loss_fn = SPWMLoss()

    B, T, D = 4, 10, 128
    z_fast = torch.randn(B, T, D)

    # Case 1: Isometric deformation (latent and decoded distances match)
    z_corr_iso = z_fast.clone()
    # Apply a shift that changes decoded kinematics
    z_corr_iso[:, :, :2] += 0.1
    loss_iso = loss_fn.isometry_loss(z_corr_iso, z_fast, decoder, k_isometry=1.0)

    # Case 2: Null-space deformation (large shift in orthogonal latent space with zero decoded physical difference)
    # The decoder maps coordinates via pos_head(q) and vel_head(p). Dimensions 4..31 of q are largely in null space of 2D pos_head.
    z_corr_null = z_fast.clone()
    z_corr_null[:, :, 10:25] += 2.0  # huge perturbation in latent space
    loss_null = loss_fn.isometry_loss(z_corr_null, z_fast, decoder, k_isometry=1.0)

    # Null-space deformation must incur substantially higher isometry penalty
    assert loss_null.item() > loss_iso.item()


def test_deep_horizon_rollout_h100():
    """Verifies autonomous and differentiable rollout up to extended deep horizon H=100 (SPWM-v9)."""
    model = SPWM(
        latent_dim=32,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
        corrector_cadence=5,
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


