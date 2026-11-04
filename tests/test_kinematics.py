"""
Unit Tests for Differentiable Forward Kinematics (FK).

Verifies mathematical equivalence between PyTorch DifferentiableForwardKinematics
and Unity PhysX 4 ArticulatedRobotRig kinematics.
"""

from __future__ import annotations
import pytest
import torch

from spwm.models.kinematics import (
    CANONICAL_ANCHORS,
    CANONICAL_JOINT_AXES,
    DifferentiableForwardKinematics,
    END_EFFECTOR_TIP_OFFSET,
)


def test_fk_canonical_constants():
    """Verify kinematic link specs and axis dimensions."""
    assert len(CANONICAL_ANCHORS) == 7
    assert len(CANONICAL_JOINT_AXES) == 6
    assert len(END_EFFECTOR_TIP_OFFSET) == 3


def test_fk_zero_pose_matches_unity():
    """At q = zeros(6), the arm stands vertically upright at y = 2.02m."""
    fk = DifferentiableForwardKinematics()
    q_zero = torch.zeros(1, 6)
    tip = fk(q_zero)

    expected = torch.tensor([[0.000000, 2.020000, 0.000000]])
    assert torch.allclose(tip, expected, atol=1e-4), f"Got {tip}, expected {expected}"


def test_fk_neutral_pose_matches_unity():
    """Canonical anthropomorphic neutral resting pose matches Unity C# reference."""
    fk = DifferentiableForwardKinematics()
    q_neutral = torch.tensor([[1.42, 3.00, -1.78, 0.05, -0.70, -0.25]])
    tip = fk(q_neutral)

    # Reference from ArticulatedRobotRig.ComputeForwardKinematics in Unity
    expected = torch.tensor([[0.020663, 1.155607, 0.369924]])
    assert torch.allclose(tip, expected, atol=1e-3), f"Got {tip}, expected {expected}"


def test_fk_arbitrary_pose_matches_unity():
    """Arbitrary joint pose matches Unity C# reference."""
    fk = DifferentiableForwardKinematics()
    q_arb = torch.tensor([[0.35, 2.1, -1.1, 0.2, -0.4, 0.8]])
    tip = fk(q_arb)

    expected = torch.tensor([[-0.137852, 1.217260, 0.544298]])
    assert torch.allclose(tip, expected, atol=1e-3), f"Got {tip}, expected {expected}"


def test_fk_batched_sequence_shape():
    """Batched sequence [B, H, 6] produces [B, H, 3]."""
    fk = DifferentiableForwardKinematics()
    B, H = 8, 20
    q = torch.randn(B, H, 6)
    tip = fk(q)

    assert tip.shape == (B, H, 3)
    assert not torch.isnan(tip).any()


def test_fk_gradient_flow():
    """Gradients propagate through FK back to joint angles q."""
    fk = DifferentiableForwardKinematics()
    q = torch.randn(4, 20, 6, requires_grad=True)
    tip = fk(q)
    loss = tip.sum()
    loss.backward()

    assert q.grad is not None
    assert q.grad.shape == (4, 20, 6)
    assert not torch.isnan(q.grad).any()
    assert (q.grad.abs() > 0.0).any()


def test_fk_homogeneous_matrix():
    """Verify 4x4 homogeneous transformation matrix output."""
    fk = DifferentiableForwardKinematics()
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    T = fk.forward_homogeneous(q)

    assert T.shape == (1, 4, 4)
    assert torch.allclose(T[:, 3, :], torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert torch.allclose(T[:, :3, 3], torch.tensor([[0.0, 2.02, 0.0]]), atol=1e-4)


def test_fk_invalid_joint_count():
    """Reject mismatched joint angles without fallback."""
    fk = DifferentiableForwardKinematics()
    with pytest.raises(ValueError, match="expected joint_count"):
        fk(torch.randn(2, 5))

