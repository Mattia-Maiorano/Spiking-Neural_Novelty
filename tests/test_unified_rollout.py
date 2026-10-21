"""
Unit tests for the Unified Rollout Engine (spwm/models/rollout.py).
Verifies that rollout_latent is the single source of truth and behaves identically
for inference, evaluation, and training.
"""

import torch
import pytest
from spwm.models.world_model import SPWM
from spwm.models.rollout import rollout_latent, RolloutResult


def test_unified_rollout_output_shapes_and_types():
    B = 2
    latent_dim = 32
    q_dim = 8
    p_dim = 24
    H = 15

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=q_dim,
        p_dim=p_dim,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )

    z_init = torch.randn(B, latent_dim)
    res = rollout_latent(
        model=model,
        z_init=z_init,
        horizon=H,
        use_corrector=True,
        burn_in_steps=3,
        with_grad=False,
        decode_kinematics=True,
    )

    assert isinstance(res, RolloutResult)
    assert res.trajectory.shape == (B, H, latent_dim)
    assert res.fast_trajectory.shape == (B, H, latent_dim)
    assert res.spikes is not None
    assert res.spikes.shape == (B, H, 16)
    assert res.decoded_kinematics is not None
    assert res.decoded_kinematics.shape == (B, H, 4)
    assert res.positions.shape == (B, H, 2)
    assert res.velocities.shape == (B, H, 2)


def test_world_model_delegates_to_rollout_latent():
    B = 2
    latent_dim = 32
    H = 10

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )

    z_init = torch.randn(B, latent_dim)
    # Predict future
    traj1 = model.predict_future(z_init, horizon=H, use_corrector=False)
    res = rollout_latent(model, z_init, horizon=H, use_corrector=False, with_grad=False)
    assert torch.allclose(traj1, res.trajectory)


def test_rollout_burn_in_bypasses_corrector():
    B = 2
    latent_dim = 32
    H = 10
    burn_in = 5

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        enable_corrector=True,
        corrector_dim=16,
    )

    z_init = torch.randn(B, latent_dim)
    res = rollout_latent(model, z_init, horizon=H, use_corrector=True, burn_in_steps=burn_in, with_grad=False)

    # During burn-in, trajectory must exactly match fast_trajectory and gain masks must be zero
    assert torch.allclose(res.trajectory[:, :burn_in], res.fast_trajectory[:, :burn_in])
    assert res.gain_masks is not None
    assert torch.all(res.gain_masks[:, :burn_in] == 0.0)
