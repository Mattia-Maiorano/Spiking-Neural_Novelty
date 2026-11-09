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

from spwm.evaluation.mppi import LatentMPPIController, MPPIPlanResult, MPPIStructuralScorer
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentCritic, LatentPredictor, PhysicalDecoder
from spwm.models.world_model import SPWM
from spwm.contracts.config import ExperimentConfig, DataConfig, TrainingConfig


@pytest.fixture
def mppi_setup():
    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=256, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=64)
    fk = DifferentiableForwardKinematics()
    controller = LatentMPPIController(
        predictor=predictor,
        physical_decoder=decoder,
        forward_kinematics=fk,
        num_samples=32,  # Smaller sample size for quick test execution
        horizon=10,
        num_iterations=2,
    )
    return controller, predictor, decoder, fk


def test_mppi_instantiation_validation():
    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=256, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=64)

    with pytest.raises(ValueError, match="num_samples must be > 0"):
        LatentMPPIController(predictor, decoder, num_samples=0)

    with pytest.raises(ValueError, match="horizon must be > 0"):
        LatentMPPIController(predictor, decoder, horizon=0)

    with pytest.raises(ValueError, match="temperature must be > 0.0"):
        LatentMPPIController(predictor, decoder, temperature=-1.0)


def test_mppi_single_step_planning(mppi_setup):
    controller, _, _, _ = mppi_setup
    controller.reset()

    z_t = torch.randn(128)
    target_pos = torch.tensor([-0.18, 1.15, 0.46])

    result = controller.plan_step(z_t, target_pos)

    assert isinstance(result, MPPIPlanResult)
    assert result.action.shape == (6,)
    assert result.action_chunk.shape == (4, 6)
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

    ctrl_no_critic = LatentMPPIController(
        predictor=predictor, physical_decoder=decoder, forward_kinematics=fk,
        critic=None, chunk_size=4, gamma_discount=0.95, num_samples=8, horizon=5,
    )
    ctrl_with_critic = LatentMPPIController(
        predictor=predictor, physical_decoder=decoder, forward_kinematics=fk,
        critic=critic, chunk_size=4, gamma_discount=0.95, num_samples=8, horizon=5,
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
    expected_terminal_add = (0.95 ** 5) * 10.0
    assert torch.allclose(cost_with_critic - cost_no_critic, torch.full((8,), expected_terminal_add), atol=1e-5)


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
    scorer = MPPIStructuralScorer(
        fixed_batches=[fixed_batch],
        device=device,
        frequency=10,
        num_samples=8,
        horizon=6,
        num_iterations=1,
        fk=fk,
        approach_window=5,
    )
    ctrl = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        forward_kinematics=fk,
        critic=model.critic,
        chunk_size=4,
        num_samples=8,
        horizon=6,
        num_iterations=1,
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
    assert len(res["transitions"]) == 12
    assert res["trajectory"]["actions"].shape == (12, 6)


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


def test_mppi_reset_with_bias(mppi_setup):
    controller, _, _, _ = mppi_setup
    bias = torch.tensor([0.2, -0.3, 0.1, 0.0, 0.5, -0.1])
    controller.reset(initial_action_bias=bias)

    assert torch.allclose(controller.nominal_actions[0], bias)
    assert torch.allclose(controller.prev_action, bias)


def test_mppi_invalid_inputs(mppi_setup):
    controller, _, _, _ = mppi_setup

    with pytest.raises(ValueError, match="LatentMPPIController.z_init"):
        controller.plan_step(torch.randn(64), torch.tensor([0.0, 1.0, 0.5]))

    with pytest.raises(ValueError, match="LatentMPPIController.target_pos"):
        controller.plan_step(torch.randn(128), torch.tensor([0.0, 1.0]))

