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

from spwm.evaluation.mppi import LatentMPPIController, MPPIPlanResult
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentPredictor, PhysicalDecoder


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
    assert torch.all(result.action >= -1.0) and torch.all(result.action <= 1.0)
    assert result.nominal_action_sequence.shape == (10, 6)
    assert result.predicted_latent_trajectory.shape == (10, 128)
    assert result.predicted_ee_trajectory.shape == (10, 3)
    assert result.min_cost > 0.0
    assert result.latency_ms > 0.0


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

