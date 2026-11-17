"""
Unit Tests for Latent MPPI Controller.

Verifies:
1. Shape contracts and action limits for candidate rollouts.
2. Isolated mental unroll using LatentPredictor (M1) and PhysicalDecoder.
3. Cost function penalties (Cartesian goal, velocity damping, action smoothness).
4. Closed-loop multi-step optimization and latency tracking.
"""

from __future__ import annotations
import pytest
import torch

from spwm.contracts.config import ExperimentConfig, DataConfig, TrainingConfig, MPPIConfig
from spwm.evaluation.mppi import (
    LatentMPPIController,
    MPPIPlanResult,
    MPPIStructuralScorer,
    compute_rot_error,
)
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentCritic, LatentPredictor, PhysicalDecoder
from spwm.models.world_model import SPWM


@pytest.fixture
def mppi_setup():
    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=256, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=64)
    fk = DifferentiableForwardKinematics()
    cfg = MPPIConfig(
        num_samples=32,  # Smaller sample size for quick test execution
        horizon=10,
        num_iterations=2,
    )
    controller = LatentMPPIController(
        predictor=predictor,
        physical_decoder=decoder,
        config=cfg,
        forward_kinematics=fk,
    )
    return controller, predictor, decoder, fk


def test_mppi_single_step_planning(mppi_setup):
    controller, _, _, _ = mppi_setup
    controller.reset()

    z_t = torch.randn(128)
    target_pos = torch.tensor([-0.18, 1.15, 0.46])

    result = controller.plan_step(z_t, target_pos)

    assert isinstance(result, MPPIPlanResult)
    assert result.action.shape == (6,)
    assert result.action_chunk.shape == (controller.chunk_size, 6)
    assert torch.allclose(result.action, result.action_chunk[0])
    assert torch.all(result.action_chunk >= -1.0) and torch.all(result.action_chunk <= 1.0)
    assert result.nominal_action_sequence.shape == (10, 6)
    assert result.predicted_latent_trajectory.shape == (10, 128)
    assert result.predicted_ee_trajectory.shape == (10, 3)
    assert result.min_cost > 0.0
    assert result.latency_ms > 0.0


def test_mppi_critic_terminal_bootstrapping():
    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=256, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=64)
    fk = DifferentiableForwardKinematics()
    critic = LatentCritic(latent_dim=128)
    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()
        critic.net[-1].bias.fill_(10.0)

    cfg = MPPIConfig(chunk_size=4, gamma_discount=0.95, num_samples=8, horizon=5, w_critic=1.0)
    ctrl_no_critic = LatentMPPIController(
        predictor=predictor, physical_decoder=decoder, config=cfg, forward_kinematics=fk,
        critic=None,
    )
    ctrl_with_critic = LatentMPPIController(
        predictor=predictor, physical_decoder=decoder, config=cfg, forward_kinematics=fk,
        critic=critic,
    )
    ctrl_disabled_critic = LatentMPPIController(
        predictor=predictor, physical_decoder=decoder, config=MPPIConfig(chunk_size=4, gamma_discount=0.95, num_samples=8, horizon=5, w_critic=0.0), forward_kinematics=fk,
        critic=critic,
    )

    ee_pos = torch.zeros(8, 5, 3)
    target_pos = torch.zeros(3)
    u_seq = torch.zeros(8, 5, 6)
    u_prev = torch.zeros(6)
    z_trajectories = torch.randn(8, 5, 128)

    cost_no_critic = ctrl_no_critic.compute_trajectory_cost(
        ee_pos, target_pos, u_seq, u_prev, z_trajectories=z_trajectories
    )
    cost_with_critic = ctrl_with_critic.compute_trajectory_cost(
        ee_pos, target_pos, u_seq, u_prev, z_trajectories=z_trajectories
    )
    cost_disabled_critic = ctrl_disabled_critic.compute_trajectory_cost(
        ee_pos, target_pos, u_seq, u_prev, z_trajectories=z_trajectories
    )
    expected_terminal_add = (0.95 ** 5) * ctrl_with_critic.w_pos * 10.0
    assert torch.allclose(cost_with_critic - cost_no_critic, torch.full((8,), expected_terminal_add), atol=1e-5)
    assert torch.allclose(cost_disabled_critic, cost_no_critic, atol=1e-5)


def test_mppi_structural_scorer_action_chunking_execution():
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        data=DataConfig(sequence_length=25),
        training=TrainingConfig(mode="train_m1", phase_horizon=6),
    )
    model = SPWM.from_config(cfg).to(device)
    model.eval()
    fk = DifferentiableForwardKinematics().to(device)

    fixed_batch = {
        "trajectory_id": 0,
        "events": torch.rand(1, 25, 4, model.height, model.width),
        "joint_positions": torch.randn(1, 25, 6) * 0.1,
        "actions": torch.randn(1, 25, 6) * 0.1,
        "phase_flags": torch.tensor([[0] * 10 + [1] * 3 + [2] * 12], dtype=torch.long),
        "elevation_plane": "Mid",
    }
    mppi_sub_cfg = MPPIConfig(
        chunk_size=4,
        num_samples=8,
        horizon=6,
        num_iterations=1,
    )
    scorer = MPPIStructuralScorer(
        fixed_batches=[fixed_batch],
        device=device,
        frequency=10,
        fk=fk,
        approach_window=5,
        config=mppi_sub_cfg,
    )
    ctrl = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        config=mppi_sub_cfg,
        forward_kinematics=fk,
        critic=model.critic,
        device=device,
    )
    plan_calls = 0
    orig_plan_step = ctrl.plan_step

    def counted_plan_step(*args, **kwargs):
        nonlocal plan_calls
        plan_calls += 1
        return orig_plan_step(*args, **kwargs)

    ctrl.plan_step = counted_plan_step  # type: ignore[method-assign]
    res = scorer.evaluate_trajectory(model, ctrl, fixed_batch, seed=42)
    # t_c=10, t_start=max(5, 10-5)=5, t_grasp=min(17, 24)=17 -> 13 steps (t=5..17)
    # With chunk_size=4, 13 steps require ceil(13 / 4) = 4 calls to plan_step (instead of 13!)
    assert plan_calls == 4
    assert len(res["transitions"]) == 13
    assert res["trajectory"]["actions"].shape == (13, 6)


def test_mppi_se3_orientation_planning(mppi_setup):
    controller, _, _, fk = mppi_setup
    controller.reset()

    z_t = torch.randn(128)
    target_q = torch.tensor([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    target_T = fk.forward_homogeneous(target_q.unsqueeze(0))[0]

    # Test passing 4x4 SE(3) matrix
    result_se3 = controller.plan_step(z_t, target_T)
    assert isinstance(result_se3, MPPIPlanResult)
    assert result_se3.action.shape == (6,)
    assert result_se3.min_cost > 0.0

    # Test passing 3D pos + 3x3 rotation matrix
    target_pos = target_T[:3, 3]
    target_rot = target_T[:3, :3]
    result_pos_rot = controller.plan_step(z_t, target_pos, target_rot=target_rot)
    assert isinstance(result_pos_rot, MPPIPlanResult)
    assert result_pos_rot.action.shape == (6,)
    assert result_pos_rot.min_cost > 0.0


def test_mppi_directional_progress_reward(mppi_setup):
    controller, _, _, _ = mppi_setup
    assert controller.w_progress == 50.0
    assert controller.w_jerk == 4.0

    target_pos = torch.tensor([1.0, 0.0, 0.0])
    p_init = torch.tensor([0.0, 0.0, 0.0])
    u_seq = torch.zeros(1, 3, 6)
    u_prev = torch.zeros(6)

    # Trajectory A moves towards target along +X
    ee_pos_towards = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]])
    # Trajectory B moves away from target along -X
    ee_pos_away = torch.tensor([[[-0.1, 0.0, 0.0], [-0.2, 0.0, 0.0], [-0.3, 0.0, 0.0]]])

    cost_towards = controller.compute_trajectory_cost(
        ee_pos=ee_pos_towards,
        target_coords=target_pos,
        u_seq=u_seq,
        u_prev=u_prev,
        p_init=p_init,
        w_progress=50.0,
    )
    cost_away = controller.compute_trajectory_cost(
        ee_pos=ee_pos_away,
        target_coords=target_pos,
        u_seq=u_seq,
        u_prev=u_prev,
        p_init=p_init,
        w_progress=50.0,
    )

    # Cost towards must be significantly lower than cost away due to progress reward
    assert cost_towards.item() < cost_away.item()

    # Verify zero progress weight yields higher cost than positive progress weight for forward motion
    cost_towards_no_prog = controller.compute_trajectory_cost(
        ee_pos=ee_pos_towards,
        target_coords=target_pos,
        u_seq=u_seq,
        u_prev=u_prev,
        p_init=p_init,
        w_progress=0.0,
    )
    assert cost_towards.item() < cost_towards_no_prog.item()


def test_mppi_deadband_attenuation_and_cartesian_damping(mppi_setup):
    controller, _, _, _ = mppi_setup
    assert controller.w_jerk == 4.0
    assert controller.w_damping == 12.0

    target_pos = torch.tensor([0.0, 0.0, 0.0])
    p_init = torch.tensor([0.02, 0.0, 0.0])  # Within 2cm of target
    ee_pos_close = torch.tensor([[[0.015, 0.0, 0.0], [0.010, 0.0, 0.0], [0.005, 0.0, 0.0]]])  # In deadband <= 3cm

    u_seq_zero = torch.zeros(1, 3, 6)
    u_prev_zero = torch.zeros(6)

    # In deadband (<= 3cm), progress reward drops to 0
    cost_prog = controller.compute_trajectory_cost(
        ee_pos=ee_pos_close,
        target_coords=target_pos,
        u_seq=u_seq_zero,
        u_prev=u_prev_zero,
        p_init=p_init,
        w_progress=50.0,
        w_damping=0.0,
    )
    cost_no_prog = controller.compute_trajectory_cost(
        ee_pos=ee_pos_close,
        target_coords=target_pos,
        u_seq=u_seq_zero,
        u_prev=u_prev_zero,
        p_init=p_init,
        w_progress=0.0,
        w_damping=0.0,
    )
    assert torch.allclose(cost_prog, cost_no_prog, atol=1e-5)

    # Cartesian velocity damping activates inside deadband when w_damping > 0
    cost_damped = controller.compute_trajectory_cost(
        ee_pos=ee_pos_close,
        target_coords=target_pos,
        u_seq=u_seq_zero,
        u_prev=u_prev_zero,
        p_init=p_init,
        w_progress=0.0,
        w_damping=25.0,
    )
    # delta_p = [[-0.005, 0, 0], [-0.005, 0, 0], [-0.005, 0, 0]], norm = 0.005 per step
    # dist_factor = 0 in deadband, (1 - dist_factor) = 1.0
    # damping_penalty = 25.0 * (0.005 * 3) = 0.375
    assert torch.allclose(cost_damped - cost_no_prog, torch.tensor([0.375]), atol=1e-5)


def test_mppi_config_anisotropic_sigma_and_gamma_wiring():
    """Verify MPPIConfig validates and wires nominal_update_gamma and anisotropic sigma multipliers."""
    with pytest.raises(ValueError, match="nominal_update_gamma"):
        MPPIConfig(nominal_update_gamma=0.0)
    with pytest.raises(ValueError, match="nominal_update_gamma"):
        MPPIConfig(nominal_update_gamma=1.5)
    with pytest.raises(ValueError, match="shoulder_sigma_multiplier"):
        MPPIConfig(shoulder_sigma_multiplier=0.0)
    with pytest.raises(ValueError, match="elbow_sigma_multiplier"):
        MPPIConfig(elbow_sigma_multiplier=-1.0)

    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=64, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=32)
    cfg = MPPIConfig(
        num_samples=8,
        horizon=4,
        num_iterations=1,
        noise_sigma=0.2,
        shoulder_sigma_multiplier=4.0,
        elbow_sigma_multiplier=2.5,
        nominal_update_gamma=0.6,
    )
    ctrl = LatentMPPIController(predictor=predictor, physical_decoder=decoder, config=cfg)
    assert ctrl.nominal_update_gamma == pytest.approx(0.6)
    expected_sigmas = torch.tensor([0.2, 0.8, 0.5, 0.2, 0.2, 0.2], dtype=torch.float32)
    assert torch.allclose(ctrl.sigma_tensor, expected_sigmas)


def test_export_mppi_episode_replans_post_grasp_full_horizon():
    """Verify export_mppi_episode executes full-horizon closed-loop control via M1 without freezing at t_grasp."""
    from experiments.export_mppi_trajectory import export_mppi_episode

    torch.manual_seed(0)
    horizon = 26
    cfg = ExperimentConfig(
        data=DataConfig(sequence_length=horizon),
        training=TrainingConfig(mode="train_m1", phase_horizon=6),
        mppi=MPPIConfig(num_samples=8, horizon=4, num_iterations=1, chunk_size=1),
    )
    model = SPWM.from_config(cfg).eval()
    fk = DifferentiableForwardKinematics()
    ctrl = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        config=cfg.mppi,
        forward_kinematics=fk,
        critic=model.critic,
        device=torch.device("cpu"),
    )

    plan_calls = 0
    orig_plan_step = ctrl.plan_step

    def tracked_plan_step(z_t, target_pose, prev_action=None):
        nonlocal plan_calls
        plan_calls += 1
        return orig_plan_step(z_t, target_pose, prev_action=prev_action)

    ctrl.plan_step = tracked_plan_step  # type: ignore[method-assign]

    pf = torch.zeros(horizon, dtype=torch.long)
    # Contact at t_c = 10, grasp at t_grasp = 17, leaving 8 post-grasp frames before horizon=26
    pf[10:18] = 1
    pf[18:] = 2

    batch = {
        "events": torch.randn(1, horizon, 4, 32, 32) * 0.05,
        "joint_positions": torch.zeros(1, horizon, 6),
        "joint_velocities": torch.zeros(1, horizon, 6),
        "actions": torch.zeros(1, horizon, 6),
        "phase_flags": pf.unsqueeze(0),
        "trajectory_ids": torch.tensor([3622], dtype=torch.long),
    }

    ghost = export_mppi_episode(
        model=model,
        controller=ctrl,
        fk=fk,
        batch=batch,
        horizon=horizon,
        model_version="test_mppi",
        approach_window=4,
        seed=42,
    )

    # t_start = max(5, 10 - 4) = 6; loop runs t = 6 .. 24 -> 19 full closed-loop replans (past t_grasp=17)
    assert len(ghost.frames) == horizon
    assert plan_calls == (horizon - 1 - 6)
    assert ghost.metadata["model_version"] == "test_mppi"
    assert ghost.metadata["heft_start_frame_index"] == 10
    assert ghost.metadata["grasp_lock_frame_index"] == 17
    assert ghost.metadata["sensory_release_frame_index"] == 17
    assert ghost.metadata["horizon"] == horizon


def test_mppi_joint_smoothness_limits_and_slew_rate():
    """Verify joint-space acceleration penalty, physical joint limits barrier, receding shift, and slew clamp."""
    with pytest.raises(ValueError, match="w_joint_accel"):
        MPPIConfig(w_joint_accel=-1.0)
    with pytest.raises(ValueError, match="w_joint_limit"):
        MPPIConfig(w_joint_limit=-0.5)
    with pytest.raises(ValueError, match="max_action_delta"):
        MPPIConfig(max_action_delta=0.0)

    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=64, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=32)
    fk = DifferentiableForwardKinematics()
    cfg = MPPIConfig(
        num_samples=16,
        horizon=6,
        num_iterations=2,
        max_action_delta=0.05,
        w_joint_accel=40.0,
        w_joint_limit=50.0,
    )
    ctrl = LatentMPPIController(predictor=predictor, physical_decoder=decoder, config=cfg, forward_kinematics=fk)
    ctrl.reset()

    ee_pos = torch.zeros(2, 4, 3)
    target_pos = torch.zeros(3)
    u_seq = torch.zeros(2, 4, 6)
    u_prev = torch.zeros(6)
    q_prev = torch.zeros(6)
    q_init = torch.tensor([0.1, 0.0, -0.2, 0.0, 0.0, 0.0])

    # Candidate 0: smooth constant velocity (ddq == 0); Candidate 1: oscillatory zig-zag
    q_hat = torch.tensor(
        [
            [
                [0.2, 0.0, -0.2, 0.0, 0.0, 0.0],
                [0.3, 0.0, -0.2, 0.0, 0.0, 0.0],
                [0.4, 0.0, -0.2, 0.0, 0.0, 0.0],
                [0.5, 0.0, -0.2, 0.0, 0.0, 0.0],
            ],
            [
                [0.3, 0.0, -0.2, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.2, 0.0, 0.0, 0.0],
                [0.3, 0.0, -0.2, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.2, 0.0, 0.0, 0.0],
            ],
        ]
    )
    costs = ctrl.compute_trajectory_cost(
        ee_pos=ee_pos,
        target_coords=target_pos,
        u_seq=u_seq,
        u_prev=u_prev,
        q_hat=q_hat,
        q_init=q_init,
        q_prev=q_prev,
        w_jerk=0.0,
        w_progress=0.0,
        w_damping=0.0,
        w_joint_accel=40.0,
        w_joint_limit=0.0,
    )
    assert costs[1] > costs[0]

    # Joint limit barrier: candidate 1 violates Elbow_Flex upper bound (+15 deg = 0.2618 rad)
    q_lim = torch.zeros(2, 4, 6)
    q_lim[1, :, 2] = 0.8
    lim_costs = ctrl.compute_trajectory_cost(
        ee_pos=ee_pos,
        target_coords=target_pos,
        u_seq=u_seq,
        u_prev=u_prev,
        q_hat=q_lim,
        w_jerk=0.0,
        w_progress=0.0,
        w_damping=0.0,
        w_joint_accel=0.0,
        w_joint_limit=50.0,
    )
    assert lim_costs[0].item() == pytest.approx(0.0, abs=1e-6)
    assert lim_costs[1].item() > 10.0

    # Slew-rate limit & receding-horizon shift check on plan_step
    z_t = torch.randn(128)
    res = ctrl.plan_step(z_t, torch.tensor([-0.18, 1.15, 0.46]), prev_action=torch.zeros(6))
    assert torch.max(torch.abs(res.action)).item() <= 0.05 + 1e-6
    assert torch.allclose(ctrl.nominal_actions[:-1], res.nominal_action_sequence[1:])
    assert torch.allclose(ctrl.nominal_actions[-1], res.nominal_action_sequence[-1])





