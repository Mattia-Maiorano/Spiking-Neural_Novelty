"""Tests for Action-Conditioned Latent Attractor Predictor (M1) Architecture and Autonomous Rollouts.

Verifies:
1. Action-conditioned dissipative & restorative phase-space integration:
   p_{t+1} = (1 - dt * gamma(q_t, u_t)) * p_t + dt * f_restore(q_t, p_t, u_t)
   q_{t+1} = q_t + dt * p_{t+1}[:, :q_dim] + dt * g_q(q_t, p_{t+1}, u_t)
2. SPWM initialization with action-conditioned predictor from config.
3. Strict parameter isolation across training phases (representation vs m1).
4. Autonomous multi-step and long-horizon rollouts conditioned on action sequences u_{0:H}.
5. Numerical stability and absence of dead M2/relay/funnel constructs.
"""

import pytest
import torch
import torch.nn as nn

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
from spwm.models.predictors import LatentPredictor, PhysicalDecoder, PredictorOutput
from spwm.models.rollout import RolloutResult, rollout_latent
from spwm.models.world_model import SPWM
from spwm.training.losses import SPWMLoss
from spwm.training.phases import PredictorPhaseHandler


def make_test_cfg(
    mode: str,
    phase_horizon: int = 10,
    seq_len: int = 45,
) -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="predictor_test"),
        environment=EnvironmentConfig(joint_count=6, action_dim=6, dt=0.05),
        data=DataConfig(sequence_length=seq_len, batch_size=2, total_trajectories=4, joint_count=6, action_dim=6),
        model=ModelConfig(
            latent_dim=32,
            q_dim=8,
            p_dim=24,
            num_keypoints=4,
            symplectic=True,
        ),
        neuron=LatentNeuronConfig(),
        memory=MemoryConfig(timescale_dims=(16, 16)),
        training=TrainingConfig(
            mode=mode,
            epochs=1,
            phase_horizon=phase_horizon,
        ),
        loss=LossConfig(),
        evaluation=EvaluationConfig(rollout_horizons=(10, 20, 50)),
    )


def test_action_conditioned_dissipative_predictor_integration():
    """Verify action-conditioned dissipative & restorative integration:
    h_t = backbone([z_t, u_t])
    gamma_t = sigmoid(dissipation_head(h_t))
    f_p = force_head(h_t)
    f_q = restorative_q_head(h_t)
    p_{t+1} = (1 - dt * gamma_t) * p_t + f_p
    q_{t+1} = q_t + dt * (p_{t+1}[:, :q_dim] + f_q)
    """
    latent_dim = 128
    q_dim = 32
    p_dim = 96
    action_dim = 8
    dt = 0.05
    predictor = LatentPredictor(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=64,
        symplectic=True,
        q_dim=q_dim,
        p_dim=p_dim,
        dt=dt,
    )
    B = 4
    z = torch.randn(B, latent_dim)
    u = torch.randn(B, action_dim)
    out = predictor(z, u)
    z_next = out.predicted_latent

    q_t = z[:, :q_dim]
    p_t = z[:, q_dim:]
    q_next = z_next[:, :q_dim]
    p_next = z_next[:, q_dim:]

    zu = torch.cat([z, u], dim=-1)
    h = predictor.backbone(zu)
    f_p = predictor.force_head(h)
    gamma_t = torch.sigmoid(predictor.dissipation_head(h))
    f_q = predictor.restorative_q_head(h)
    expected_p = (1.0 - dt * gamma_t) * p_t + f_p
    expected_q = q_t + dt * (expected_p[:, :q_dim] + f_q)

    torch.testing.assert_close(p_next, expected_p)
    torch.testing.assert_close(q_next, expected_q)


def test_symplectic_spwm_from_config():
    """Verify SPWM instantiates with action-conditioned M1 from config without old M2/relay constructs."""
    cfg = ExperimentConfig(
        model=ModelConfig(
            latent_dim=128,
            q_dim=32,
            p_dim=96,
            num_keypoints=16,
            symplectic=True,
        )
    )
    model = SPWM.from_config(cfg)
    assert model.m1.symplectic is True
    assert model.predictor.symplectic is True
    assert model.m1 is model.predictor
    assert model.m1.dt == cfg.environment.dt
    assert model.m1.action_dim == cfg.environment.action_dim

    # Legacy M2, relay, and m3 must not exist
    assert not hasattr(model, "m2")
    assert not hasattr(model, "m3")
    assert not hasattr(model, "relay")

    z = torch.randn(2, 128)
    u = torch.randn(2, cfg.environment.action_dim)
    out = model.m1(z, u)
    assert out.predicted_latent.shape == (2, 128)


def test_strict_isolation_phase_0_train_representation():
    """Phase 0: Train representation (encoder, dynamics, sensory_predictor, physical_decoder). Predictor frozen."""
    cfg = make_test_cfg("train_representation", phase_horizon=8)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # Active representation modules
    assert any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.dynamics.parameters())
    assert any(p.requires_grad for p in model.sensory_predictor.parameters())
    assert any(p.requires_grad for p in model.physical_decoder.parameters())
    # Predictor must be strictly frozen
    assert all(not p.requires_grad for p in model.predictor.parameters())
    assert all(not p.requires_grad for p in model.m1.parameters())


def test_strict_isolation_phase_1_train_m1():
    """Phase 1: Train M1 predictor. Representation modules must be frozen."""
    cfg = make_test_cfg("train_m1", phase_horizon=8)
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    handler = PredictorPhaseHandler(model=model, cfg=cfg, loss_fn=loss_fn, device=torch.device("cpu"))

    # Only M1 / predictor parameters must require grad
    assert any(p.requires_grad for p in model.predictor.parameters())
    assert any(p.requires_grad for p in model.m1.parameters())
    # Representation modules must be strictly frozen
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.dynamics.parameters())
    assert all(not p.requires_grad for p in model.sensory_predictor.parameters())
    assert all(not p.requires_grad for p in model.physical_decoder.parameters())


def test_autonomous_rollout_with_action_conditioned_predictor():
    """Tests autonomous long-horizon rollout (e.g. H=100) conditioned on action sequence u_{0:H}."""
    B, D, H = 2, 32, 100
    cfg = make_test_cfg("train_m1", phase_horizon=10)
    model = SPWM.from_config(cfg)
    z_init = torch.randn(B, D)
    actions = torch.randn(B, H, cfg.environment.action_dim)

    result = rollout_latent(
        model=model,
        z_init=z_init,
        horizon=H,
        actions=actions,
        decode_kinematics=False,
    )

    assert isinstance(result, RolloutResult)
    assert result.trajectory.shape == (B, H, D)
    assert result.actions is not None
    assert result.actions.shape == (B, H, cfg.environment.action_dim)
    assert not torch.isnan(result.trajectory).any()

    # Active models list should cleanly reflect M1 predictor throughout
    assert result.active_models is not None
    assert len(result.active_models) == H
    assert all(m == "m1" for m in result.active_models)
