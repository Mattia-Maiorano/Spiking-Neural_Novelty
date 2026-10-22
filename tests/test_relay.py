"""
Tests for Cascaded Temporal Relay (M1, M2, M3) Architecture.

Verifies:
1. Cyclic Relay routing and modular sub-step calculation.
2. Naive kinematic extrapolation and Convergence Funnel mathematical target blend.
3. Strict parameter isolation across training phases (no joint BPTT).
4. M3 initialization from M1 pre-trained weights.
5. Autonomous multi-step and infinite-horizon rollouts (t -> inf).
"""

import pytest
import torch
import torch.nn as nn

from spwm.config.schema import (
    ExperimentConfig,
    ProjectConfig,
    EnvironmentConfig,
    DataConfig,
    ModelConfig,
    LatentNeuronConfig,
    MemoryConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
)
from spwm.core.predictor import (
    CascadedTemporalRelay,
    LatentPredictor,
    extrapolate_latent,
    compute_convergence_funnel_target,
)
from spwm.learning.losses import ConvergenceFunnelLoss, SPWMLoss
from spwm.learning.phases.predictor_phase import PredictorPhaseHandler
from spwm.models.world_model import SPWM
from spwm.models.rollout import rollout_latent, RolloutResult


def test_cascaded_temporal_relay_sub_step_routing():
    """Verifies autonomous cyclic routing between M1, M2, and M3 without absolute step count."""
    relay = CascadedTemporalRelay(latent_dim=32, hidden_dim=64, residual=True)

    # 1. t in [0, 9] -> M1 (nominal physics)
    for t in range(10):
        mod, name, local_step = relay.get_active_model(t)
        assert mod is relay.m1
        assert name == "m1"
        assert local_step == t

    # 2. t in [10, 19] -> M2 (recovery)
    for t in range(10, 20):
        mod, name, local_step = relay.get_active_model(t)
        assert mod is relay.m2
        assert name == "m2"
        assert local_step == (t - 10)

    # 3. t in [20, 29] -> M3 (sustained)
    for t in range(20, 30):
        mod, name, local_step = relay.get_active_model(t)
        assert mod is relay.m3
        assert name == "m3"
        assert local_step == (t - 20)

    # 4. t in [30, 39] -> M2 (recovery again)
    for t in range(30, 40):
        mod, name, local_step = relay.get_active_model(t)
        assert mod is relay.m2
        assert name == "m2"
        assert local_step == (t - 30)

    # 5. Specific prompt requirement: residual at t=43 -> M3 with local_step=3
    mod, name, local_step = relay.get_active_model(43)
    assert mod is relay.m3
    assert name == "m3"
    assert local_step == 3


def test_extrapolate_latent_linear_kinematics():
    """Verifies naive kinematic projection: Extrapolate(z_10, t) = z_10 + (t - 10) * (z_10 - z_9)."""
    B, D = 2, 8
    z_9 = torch.tensor([[1.0] * D, [2.0] * D])
    z_10 = torch.tensor([[1.5] * D, [3.0] * D])  # delta is +0.5, +1.0

    # At t=10: extrapolate should be exactly z_10
    ext_10 = extrapolate_latent(z_anchor=z_10, t=10, t_anchor=10, z_prev=z_9)
    assert torch.allclose(ext_10, z_10)

    # At t=11: delta * 1
    ext_11 = extrapolate_latent(z_anchor=z_10, t=11, t_anchor=10, z_prev=z_9)
    assert torch.allclose(ext_11, torch.tensor([[2.0] * D, [4.0] * D]))

    # At t=20: delta * 10
    ext_20 = extrapolate_latent(z_anchor=z_10, t=20, t_anchor=10, z_prev=z_9)
    assert torch.allclose(ext_20, torch.tensor([[6.5] * D, [13.0] * D]))


def test_convergence_funnel_target_mathematical_blend():
    """Verifies that M2 target smoothly interpolates from dirty extrapolation to true manifold without snap."""
    B, H, D = 2, 10, 4
    z_prev = torch.zeros(B, D)
    z_anchor = torch.ones(B, D)  # delta = +1.0 per step

    # Ground truth: constant 5.0
    z_ground_truth = torch.full((B, H, D), 5.0)

    targets = compute_convergence_funnel_target(
        z_anchor=z_anchor,
        z_prev=z_prev,
        z_ground_truth=z_ground_truth,
        t_anchor=10,
    )
    assert targets.shape == (B, H, D)

    # Step t=11 (index 0): tau = 1/10 = 0.1
    # Extrapolated at t=11 is z_anchor + 1 * 1 = 2.0
    # Blend: (1 - 0.1) * 2.0 + 0.1 * 5.0 = 1.8 + 0.5 = 2.3
    expected_t11 = (1.0 - 0.1) * 2.0 + 0.1 * 5.0
    assert torch.allclose(targets[:, 0], torch.tensor([[expected_t11] * D, [expected_t11] * D]))

    # Step t=20 (index 9): tau = 10/10 = 1.0
    # Blend: (1 - 1.0) * Extrapolate + 1.0 * z_gt = 5.0 (pure ground truth)
    assert torch.allclose(targets[:, 9], z_ground_truth[:, 9])


def test_convergence_funnel_loss():
    """Verifies ConvergenceFunnelLoss computes MSE against the mathematically blended target."""
    loss_fn = ConvergenceFunnelLoss(t_anchor=10)
    B, H, D = 3, 10, 16
    predicted = torch.randn(B, H, D)
    z_anchor = torch.randn(B, D)
    z_prev = torch.randn(B, D)
    z_gt = torch.randn(B, H, D)

    target = compute_convergence_funnel_target(z_anchor, z_prev, z_gt, t_anchor=10)
    loss_out = loss_fn(predicted, z_anchor, z_prev, z_gt)

    assert "total_loss" in loss_out
    assert "funnel_loss" in loss_out
    assert "raw_gt_loss" in loss_out
    expected_mse = torch.nn.functional.mse_loss(predicted, target)
    assert torch.allclose(loss_out["funnel_loss"], expected_mse)


def make_relay_test_cfg(
    mode: str,
    phase_horizon: int = 10,
    max_rollout_horizon: int = 30,
    seq_len: int = 45,
) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="relay_test"),
        environment=EnvironmentConfig(num_objects=1),
        data=DataConfig(sequence_length=seq_len, batch_size=2, total_trajectories=4),
        model=ModelConfig(latent_dim=32, q_dim=8, p_dim=24, num_keypoints=4),
        neuron=LatentNeuronConfig(),
        memory=MemoryConfig(timescale_dims=(16, 16)),
        training=TrainingConfig(
            mode=mode,
            epochs=1,
            phase_horizon=phase_horizon,
            max_rollout_horizon=max_rollout_horizon,
        ),
        loss=LossConfig(),
        evaluation=EvaluationConfig(rollout_horizons=(10, 20, 50)),
    )


def test_strict_isolation_phase_1_train_m1():
    """Phase 1: Train M1 strictly confined to one phase. M2 and M3 must be frozen."""
    cfg = make_relay_test_cfg("train_m1", phase_horizon=8)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # M1, encoder, dynamics, sensory_predictor, physical_decoder parameters must require grad
    assert any(p.requires_grad for p in model.m1.parameters())
    assert any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.dynamics.parameters())
    assert any(p.requires_grad for p in model.sensory_predictor.parameters())
    assert any(p.requires_grad for p in model.physical_decoder.parameters())
    # M2 and M3 parameters must be strictly frozen
    assert all(not p.requires_grad for p in model.m2.parameters())
    assert all(not p.requires_grad for p in model.m3.parameters())


def test_strict_isolation_phase_2_train_m2():
    """Phase 2: Train M2 strictly confined to 1 recovery phase (H -> 2H). M1 and M3 must be frozen."""
    cfg = make_relay_test_cfg("train_m2", phase_horizon=8)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # M2 parameters must require grad
    assert any(p.requires_grad for p in model.m2.parameters())
    # M1 and M3 must be strictly frozen
    assert all(not p.requires_grad for p in model.m1.parameters())
    assert all(not p.requires_grad for p in model.m3.parameters())


def test_train_m3_joint_m2_m3_training_and_weight_init():
    """Phase 3: Train M3 via max rollout where BOTH M2 and M3 are trained, and M1 is frozen."""
    cfg = make_relay_test_cfg("train_m3", phase_horizon=8, max_rollout_horizon=25)
    model = SPWM.from_config(cfg)

    # Set distinct weights in M1
    with torch.no_grad():
        for p in model.m1.parameters():
            p.fill_(0.42)

    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # Verify M3 received M1's pre-trained weights
    for p in model.m3.parameters():
        assert torch.allclose(p, torch.tensor(0.42))

    # BOTH M2 and M3 parameters must require grad!
    assert any(p.requires_grad for p in model.m2.parameters())
    assert any(p.requires_grad for p in model.m3.parameters())
    # M1 must remain strictly frozen
    assert all(not p.requires_grad for p in model.m1.parameters())


def test_dynamic_phase_horizon_routing():
    """Verifies that dynamic phase_horizon (e.g. H=5, H=12) alters relay windows correctly."""
    for H in [5, 12]:
        relay = CascadedTemporalRelay(latent_dim=16, hidden_dim=32, phase_horizon=H)

        # [0, H-1] -> M1
        for t in range(H):
            mod, name, local_step = relay.get_active_model(t)
            assert mod is relay.m1
            assert name == "m1"
            assert local_step == t

        # [H, 2H-1] -> M2
        for t in range(H, 2 * H):
            mod, name, local_step = relay.get_active_model(t)
            assert mod is relay.m2
            assert name == "m2"
            assert local_step == (t - H)

        # [2H, 3H-1] -> M3
        for t in range(2 * H, 3 * H):
            mod, name, local_step = relay.get_active_model(t)
            assert mod is relay.m3
            assert name == "m3"
            assert local_step == (t - 2 * H)

        # [3H, 4H-1] -> M2
        for t in range(3 * H, 4 * H):
            mod, name, local_step = relay.get_active_model(t)
            assert mod is relay.m2
            assert name == "m2"
            assert local_step == (t - 3 * H)


def test_train_m3_gradient_flow_to_both_m2_and_m3():
    """Verifies that in train_m3, gradients flow to BOTH M2 and M3, and NOT to M1."""
    cfg = make_relay_test_cfg("train_m3", phase_horizon=5, max_rollout_horizon=15, seq_len=20)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    batch = {
        "events": torch.randn(2, 20, 2, 32, 32),
        "flat_kinematics": torch.randn(2, 20, 4),
    }

    metrics = handler.train_step(batch, epoch=1)
    assert "train_loss" in metrics

    # Verify M2 and M3 have non-zero gradients
    m2_grads = [p.grad for p in model.m2.parameters() if p.grad is not None]
    m3_grads = [p.grad for p in model.m3.parameters() if p.grad is not None]
    m1_grads = [p.grad for p in model.m1.parameters() if p.grad is not None]

    assert len(m2_grads) > 0, "M2 received no gradients during train_m3!"
    assert len(m3_grads) > 0, "M3 received no gradients during train_m3!"
    assert len(m1_grads) == 0, "M1 leaked gradients during train_m3!"


def test_autonomous_infinite_rollout_cyclic_relay():
    """Tests autonomous infinite-horizon rollout (e.g. H=100) cycling through M1, M2, M3."""
    B, D, H = 2, 32, 100
    relay = CascadedTemporalRelay(latent_dim=D, hidden_dim=64, residual=True)

    class DummyModel(nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

    model = DummyModel(relay)
    z_init = torch.randn(B, D)

    result = rollout_latent(
        model=model,
        z_init=z_init,
        horizon=H,
        decode_kinematics=False,
    )

    assert isinstance(result, RolloutResult)
    assert result.trajectory.shape == (B, H, D)
    assert not torch.isnan(result.trajectory).any()

    # Verify active models list tracks the 10-step relay cycles correctly
    assert result.active_models is not None
    assert len(result.active_models) == H

    # t=0..9 -> m1
    assert all(m == "m1" for m in result.active_models[0:10])
    # t=10..19 -> m2
    assert all(m == "m2" for m in result.active_models[10:20])
    # t=20..29 -> m3
    assert all(m == "m3" for m in result.active_models[20:30])
    # t=30..39 -> m2
    assert all(m == "m2" for m in result.active_models[30:40])
    # t=40..49 -> m3
    assert all(m == "m3" for m in result.active_models[40:50])
