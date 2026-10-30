"""Unit tests for the Unified Rollout Engine (spwm/models/rollout.py).
Verifies that rollout_latent is the single source of truth and behaves identically
for inference, evaluation, and training using the Action-Conditioned Latent Attractor Predictor (M1).
"""

import torch
from spwm.models.rollout import RolloutResult, rollout_latent
from spwm.models.world_model import SPWM


def test_unified_rollout_output_shapes_and_types():
    B = 2
    latent_dim = 32
    q_dim = 8
    p_dim = 24
    joint_count = 6
    action_dim = 6
    H = 25

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=q_dim,
        p_dim=p_dim,
        timescale_dims=(16, 16),
        joint_count=joint_count,
        action_dim=action_dim,
    )

    z_init = torch.randn(B, latent_dim)
    actions = torch.randn(B, H, action_dim)
    res = rollout_latent(
        model=model,
        z_init=z_init,
        horizon=H,
        actions=actions,
        with_grad=False,
        decode_kinematics=True,
    )

    assert isinstance(res, RolloutResult)
    assert res.trajectory.shape == (B, H, latent_dim)
    assert res.fast_trajectory.shape == (B, H, latent_dim)
    assert res.decoded_kinematics is not None
    assert res.decoded_kinematics.shape == (B, H, 2 * joint_count)
    assert res.positions.shape == (B, H, joint_count)
    assert res.velocities.shape == (B, H, joint_count)
    assert res.actions is not None
    assert res.actions.shape == (B, H, action_dim)
    assert res.active_models is not None
    assert len(res.active_models) == H
    assert all(m == "m1" for m in res.active_models)


def test_world_model_delegates_to_rollout_latent():
    B = 2
    latent_dim = 32
    action_dim = 6
    H = 10

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        joint_count=6,
        action_dim=action_dim,
    )

    z_init = torch.randn(B, latent_dim)
    actions = torch.randn(B, H, action_dim)
    traj1 = model.predict_future(z_init, horizon=H, actions=actions)
    res = rollout_latent(model, z_init, horizon=H, actions=actions, with_grad=False)
    assert torch.allclose(traj1, res.trajectory)


def test_rollout_with_grad_and_start_step():
    B = 2
    latent_dim = 32
    action_dim = 6
    H = 10

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        joint_count=6,
        action_dim=action_dim,
        phase_horizon=5,
    )

    z_init = torch.randn(B, latent_dim)
    actions = torch.randn(B, H, action_dim)
    res_offset = rollout_latent(model, z_init, horizon=H, actions=actions, start_step=5, with_grad=True)
    assert res_offset.active_models == ["m1"] * 10
    assert res_offset.trajectory.requires_grad


def test_rollout_active_perception_memory_synchronization():
    B = 2
    latent_dim = 32
    action_dim = 6
    H = 6

    model = SPWM(
        latent_dim=latent_dim,
        q_dim=8,
        p_dim=24,
        timescale_dims=(16, 16),
        joint_count=6,
        action_dim=action_dim,
        height=16,
        width=16,
    )

    z_init = torch.randn(B, latent_dim)
    actions = torch.randn(B, H, action_dim)
    events = torch.rand(B, H, 4, 16, 16)
    # 3 steps approach (0), 2 steps heft (1), 1 step transport (2)
    phase_flags = torch.tensor([[0, 0, 0, 1, 1, 2]] * B, dtype=torch.long)
    init_dyn = model.dynamics.init_state(B)

    # Initial membrane potential at step 0
    init_v_mem = init_dyn.memory_state.v_mems[0].clone()

    res = rollout_latent(
        model=model,
        z_init=z_init,
        horizon=H,
        actions=actions,
        events=events,
        phase_flags=phase_flags,
        initial_dynamics_state=init_dyn,
    )

    assert res.active_models == ["m1", "m1", "m1", "grounded", "grounded", "m1"]
    assert res.trajectory.shape == (B, H, latent_dim)
