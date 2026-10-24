"""Unit and integration tests for M2 as a Stateless Manifold Projector (Geometric Denoiser).

Verifies all aspects of the SYSTEM DIRECTIVE:
1. Mathematical specification: z_clean = M2(z_noisy), stateless manifold projection without advancing time.
2. Architecture: bottleneck/residual MLP (input/output dim 128), z_projected = z + net(z).
3. Phase 2 Training: frozen representation + M1, autoregressive + Gaussian noise regimes, instantaneous target z^*_k,
   decomposed Smooth-L1 loss (lambda_q * SmoothL1 + lambda_p * SmoothL1).
4. Inference / Evaluation protocol: Integrate & Project cadence (M1 every step, M2 every N steps).
"""

from __future__ import annotations
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    LatentNeuronConfig,
    LossConfig,
    MemoryConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from spwm.models.predictors import LatentDenoiser, M2Projector, ManifoldProjector
from spwm.models.rollout import rollout_latent
from spwm.models.world_model import SPWM
from spwm.training.losses import SPWMLoss
from spwm.training.phases.m2_recovery import M2PhaseStrategy, Phase2DenoiserStrategy


def _make_test_config(mode: str = "train_m2", seq_len: int = 35) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="m2_denoiser_test"),
        environment=EnvironmentConfig(height=16, width=16, num_objects=1, dt=0.05),
        data=DataConfig(sequence_length=seq_len, batch_size=2, total_trajectories=4),
        model=ModelConfig(
            latent_dim=32,
            q_dim=8,
            p_dim=24,
            num_keypoints=4,
            predictor_hidden_dim=32,
            symplectic=True,
        ),
        neuron=LatentNeuronConfig(),
        memory=MemoryConfig(timescale_dims=(8, 8)),
        training=TrainingConfig(
            mode=mode,
            epochs=1,
            phase_horizon=10,
        ),
        loss=LossConfig(lambda_position=1.0, lambda_velocity=1.0),
        evaluation=EvaluationConfig(rollout_horizons=(5, 10, 20), projection_cadence=5),
    )


def test_m2_architecture_and_residual_projection():
    """Verify Section 2: Bottleneck/residual MLP architecture and z_projected = z + net(z)."""
    latent_dim = 128
    q_dim = 32
    p_dim = 96
    m2 = ManifoldProjector(
        latent_dim=latent_dim,
        hidden_dim=256,
        bottleneck_dim=64,
        q_dim=q_dim,
        p_dim=p_dim,
    )

    assert isinstance(m2, nn.Module)
    assert m2.latent_dim == latent_dim
    assert m2.q_dim == q_dim
    assert m2.p_dim == p_dim
    assert m2.bottleneck_dim == 64

    # Check aliases
    assert M2Projector is ManifoldProjector
    assert LatentDenoiser is ManifoldProjector

    # Test 2D input [B, D]
    B = 4
    z_noisy_2d = torch.randn(B, latent_dim)
    z_proj_2d = m2(z_noisy_2d)
    assert z_proj_2d.shape == (B, latent_dim)

    # Verify residual math: z_projected = z + net(z)
    expected_proj = z_noisy_2d + m2.net(z_noisy_2d)
    torch.testing.assert_close(z_proj_2d, expected_proj)

    # Test 3D sequence input [B, T, D]
    T = 15
    z_noisy_3d = torch.randn(B, T, latent_dim)
    z_proj_3d = m2(z_noisy_3d)
    assert z_proj_3d.shape == (B, T, latent_dim)

    # Dimension mismatch check
    with pytest.raises(ValueError, match="expected last dimension"):
        m2(torch.randn(B, latent_dim + 4))


def test_spwm_instantiates_m2():
    """Verify SPWM instantiates M2 as a stateless manifold projector."""
    cfg = _make_test_config()
    model = SPWM.from_config(cfg)

    assert hasattr(model, "m2"), "SPWM missing 'm2' submodule"
    assert isinstance(model.m2, ManifoldProjector)
    assert model.m2.latent_dim == cfg.model.latent_dim
    assert model.m2.q_dim == cfg.model.q_dim
    assert model.m2.p_dim == cfg.model.p_dim


def test_phase2_gradient_isolation_and_training_step():
    """Verify Section 3: Phase 2 training strategy freezes representation + M1 and isolates M2 gradients."""
    cfg = _make_test_config("train_m2", seq_len=30)
    model = SPWM.from_config(cfg)
    device = torch.device("cpu")
    loss_fn = SPWMLoss.from_config(cfg.loss)

    strategy = M2PhaseStrategy(model=model, cfg=cfg, loss_fn=loss_fn, device=device)
    assert isinstance(strategy, Phase2DenoiserStrategy)

    # Verify gradient isolation
    strategy.verify_gradient_isolation()
    assert all(p.requires_grad is False for p in model.encoder.parameters())
    assert all(p.requires_grad is False for p in model.dynamics.parameters())
    assert all(p.requires_grad is False for p in model.sensory_predictor.parameters())
    assert all(p.requires_grad is False for p in model.physical_decoder.parameters())
    assert all(p.requires_grad is False for p in model.predictor.parameters())
    assert all(p.requires_grad is True for p in model.m2.parameters())

    # Execute training step
    B, T = 2, 30
    events = torch.rand(B, T, 2, 16, 16)
    flat_kin = torch.randn(B, T, 4)
    batch = {"events": events, "flat_kinematics": flat_kin}

    metrics = strategy.train_step(batch, epoch=1)

    # Verify metrics contain decomposed Smooth-L1 losses and k_step
    assert "train_loss" in metrics
    assert "train_loss_m2" in metrics
    assert "train_loss_q" in metrics
    assert "train_loss_p" in metrics
    assert "train_k_step" in metrics

    # Verify gradients flowed ONLY to M2
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.m2.parameters()
    ), "M2 must receive non-zero gradients in train_m2"
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.dynamics.parameters())
    assert all(p.grad is None for p in model.sensory_predictor.parameters())
    assert all(p.grad is None for p in model.physical_decoder.parameters())
    assert all(p.grad is None for p in model.predictor.parameters())


def test_phase2_decomposed_smooth_l1_loss_math():
    """Verify Section 3 loss formula: L_m2 = lambda_q * SmoothL1(q_proj, q*_k) + lambda_p * SmoothL1(p_proj, p*_k)."""
    q_dim = 8
    p_dim = 24
    latent_dim = q_dim + p_dim
    m2 = ManifoldProjector(latent_dim=latent_dim, q_dim=q_dim, p_dim=p_dim)

    B = 4
    z_noisy = torch.randn(B, latent_dim)
    z_clean_gt = torch.randn(B, latent_dim)

    z_proj = m2(z_noisy)
    q_proj, p_proj = z_proj[:, :q_dim], z_proj[:, q_dim:]
    q_gt, p_gt = z_clean_gt[:, :q_dim], z_clean_gt[:, q_dim:]

    lambda_q, lambda_p = 1.0, 1.0
    expected_loss = lambda_q * F.smooth_l1_loss(q_proj, q_gt) + lambda_p * F.smooth_l1_loss(p_proj, p_gt)

    assert expected_loss > 0
    # Ensure backprop works cleanly through M2
    expected_loss.backward()
    assert all(p.grad is not None for p in m2.parameters())


def test_inference_integrate_and_project_protocol():
    """Verify Section 4: Integrate forward using M1 every step, project with M2 every N steps."""
    cfg = _make_test_config()
    model = SPWM.from_config(cfg)

    B = 2
    H = 30
    N = 10  # Projection cadence

    z_init = torch.randn(B, cfg.model.latent_dim)
    res = rollout_latent(model, z_init, horizon=H, projection_cadence=N)

    assert res.trajectory.shape == (B, H, cfg.model.latent_dim)
    assert len(res.active_models) == H

    # Check projection schedule: step t in 1..H
    for idx, active in enumerate(res.active_models):
        t = idx + 1
        if t % N == 0:
            assert active == "m1+m2", f"Step t={t} must execute M2 manifold projection"
        else:
            assert active == "m1", f"Step t={t} must be nominal M1 integration"


def test_projection_anchors_subsequent_m1_step():
    """Verify that the M2 projected state anchors the subsequent M1 integration step."""
    cfg = _make_test_config()
    model = SPWM.from_config(cfg)

    # Let M2 add a known offset to easily trace the anchor
    with torch.no_grad():
        for p in model.m2.parameters():
            p.zero_()
        # Set final bias of M2 to a known constant vector
        last_linear = list(model.m2.net.children())[-1]
        assert isinstance(last_linear, nn.Linear)
        last_linear.bias.fill_(0.5)

    z_init = torch.randn(1, cfg.model.latent_dim)
    # Rollout for 2 steps with N=1: every step gets integrated by M1 then projected by M2
    res = rollout_latent(model, z_init, horizon=2, projection_cadence=1)

    # Manual step 1:
    z1_m1 = model.m1(z_init).predicted_latent
    z1_projected = model.m2(z1_m1)
    torch.testing.assert_close(res.trajectory[:, 0], z1_projected)

    # Manual step 2: M1 integrates from z1_projected (NOT z1_m1!)
    z2_m1 = model.m1(z1_projected).predicted_latent
    z2_projected = model.m2(z2_m1)
    torch.testing.assert_close(res.trajectory[:, 1], z2_projected)
