"""
Unit tests for the Unified Rollout Engine (spwm/models/rollout.py).
Verifies that rollout_latent is the single source of truth and behaves identically
for inference, evaluation, and training across the Cascaded Temporal Relay.
"""

import torch
from spwm.models.world_model import SPWM
from spwm.models.rollout import rollout_latent, RolloutResult


def test_unified_rollout_output_shapes_and_types():
    B = 2
    latent_dim = 32
    q_dim = 8
    p_dim = 24
    H = 25

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=q_dim,
        p_dim=p_dim,
        timescale_dims=(16, 16),
    )

    z_init = torch.randn(B, latent_dim)
    res = rollout_latent(
        model=model,
        z_init=z_init,
        horizon=H,
        with_grad=False,
        decode_kinematics=True,
    )

    assert isinstance(res, RolloutResult)
    assert res.trajectory.shape == (B, H, latent_dim)
    assert res.fast_trajectory.shape == (B, H, latent_dim)
    assert res.decoded_kinematics is not None
    assert res.decoded_kinematics.shape == (B, H, 4)
    assert res.positions.shape == (B, H, 2)
    assert res.velocities.shape == (B, H, 2)
    assert res.active_models is not None
    assert len(res.active_models) == H


def test_world_model_delegates_to_rollout_latent():
    B = 2
    latent_dim = 32
    H = 10

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
    )

    z_init = torch.randn(B, latent_dim)
    traj1 = model.predict_future(z_init, horizon=H)
    res = rollout_latent(model, z_init, horizon=H, with_grad=False)
    assert torch.allclose(traj1, res.trajectory)


def test_rollout_fixed_predictor_and_start_step():
    B = 2
    latent_dim = 32
    H = 10

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        phase_horizon=5,
    )

    z_init = torch.randn(B, latent_dim)
    res_offset = rollout_latent(model, z_init, horizon=H, start_step=5, with_grad=True)
    assert res_offset.active_models == ["m2"] * 5 + ["m3"] * 5
    assert res_offset.trajectory.requires_grad
