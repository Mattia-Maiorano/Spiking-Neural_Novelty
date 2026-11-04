"""
Differentiable Forward Kinematics (FK) for Anthropomorphic 6-DOF Manipulator Arm.

Extracted with mathematical equivalence from `sim/Assets/Scripts/ArticulatedRobotRig.cs`.
Computes exact 3D Cartesian coordinates (x, y, z) of the End-Effector tip
from batched joint angles q* via batched 4x4 homogeneous transformation matrices.
Layer 1 (Models) module with zero AST fallbacks.
"""

from __future__ import annotations
from typing import Sequence, Tuple
import torch
import torch.nn as nn

from spwm.contracts.tensors import J, assert_tensor_contract

# Exact link lengths and mounting elevations matching ArticulatedRobotRig.cs (meters)
SHOULDER_MOUNT_HEIGHT: float = 1.25
BASE_HEIGHT: float = 0.05
TURRET_HEIGHT: float = 0.05
UPPER_ARM_LENGTH: float = 0.28
FOREARM_LENGTH: float = 0.25
WRIST_PITCH_LENGTH: float = 0.04
WRIST_YAW_LENGTH: float = 0.04
WRIST_ROLL_LENGTH: float = 0.06

# Canonical local anchor offsets from parent link pivot to child joint pivot
CANONICAL_ANCHORS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, SHOULDER_MOUNT_HEIGHT, 0.0),  # Base_Root
    (0.0, BASE_HEIGHT, 0.0),            # Shoulder_Pan (Joint 0)
    (0.0, TURRET_HEIGHT, 0.0),          # Shoulder_Lift (Joint 1)
    (0.0, UPPER_ARM_LENGTH, 0.0),       # Elbow_Flex (Joint 2)
    (0.0, FOREARM_LENGTH, 0.0),         # Wrist_Pitch (Joint 3)
    (0.0, WRIST_PITCH_LENGTH, 0.0),     # Wrist_Yaw (Joint 4)
    (0.0, WRIST_YAW_LENGTH, 0.0),       # Wrist_Roll (Joint 5)
)

# End-Effector Tip offset extending along the Wrist_Roll local +Y axis
END_EFFECTOR_TIP_OFFSET: Tuple[float, float, float] = (0.0, WRIST_ROLL_LENGTH, 0.0)

# Canonical joint rotation axes extracted from Quaternion.Euler(spec.AxisRotationEuler) * Vector3.right
CANONICAL_JOINT_AXES: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 1.0, 0.0),                             # Joint 0: Shoulder_Pan (Euler 0, 0, 90)
    (1.0, 0.0, 0.0),                             # Joint 1: Shoulder_Lift (Euler 0, 0, 0)
    (0.5375313, -0.2756374, -0.7969221),         # Joint 2: Elbow_Flex (Euler 0, 56, -16)
    (0.5375313, -0.2756374, -0.7969221),         # Joint 3: Wrist_Pitch (Euler 0, 56, -16)
    (0.0, 0.0, -1.0),                            # Joint 4: Wrist_Yaw (Euler 0, 90, 0)
    (0.0, 1.0, 0.0),                             # Joint 5: Wrist_Roll (Euler 0, 0, 90)
)


class DifferentiableForwardKinematics(nn.Module):
    """
    Differentiable Forward Kinematics solver for the 6-DOF Anthropomorphic Manipulator Arm.

    Computes the 3D world Cartesian end-effector coordinates:
        (x, y, z) = FK(q_hat)
    using batched 4x4 homogeneous transformation matrices.
    Supports arbitrary batch shapes: [B, H, 6], [B, 6], or [N, 6].
    """

    def __init__(
        self,
        base_origin: Sequence[float] = (0.0, 0.0, 0.0),
        joint_count: int = J,
    ) -> None:
        super().__init__()
        if joint_count != 6:
            raise ValueError(f"DifferentiableForwardKinematics strictly supports 6-DOF arm, got {joint_count}")
        if len(base_origin) != 3:
            raise ValueError(f"base_origin must be a 3D vector, got length {len(base_origin)}")

        self.joint_count = joint_count

        # Register kinematic geometry as non-trainable persistent buffers
        anchors_tensor = torch.tensor(CANONICAL_ANCHORS, dtype=torch.float32)  # [7, 3]
        axes_tensor = torch.tensor(CANONICAL_JOINT_AXES, dtype=torch.float32)      # [6, 3]
        # Normalize axes explicitly to guarantee unit norm
        axes_tensor = axes_tensor / torch.norm(axes_tensor, dim=-1, keepdim=True)
        tip_offset_tensor = torch.tensor(END_EFFECTOR_TIP_OFFSET, dtype=torch.float32)  # [3]
        base_origin_tensor = torch.tensor(list(base_origin), dtype=torch.float32)       # [3]

        self.register_buffer("anchors", anchors_tensor)
        self.register_buffer("axes", axes_tensor)
        self.register_buffer("tip_offset", tip_offset_tensor)
        self.register_buffer("base_origin", base_origin_tensor)

    def _build_rodrigues_matrices(self, axes: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
        """
        Builds batched 3x3 rotation matrices via Rodrigues' formula.
        axes: [6, 3]
        thetas: [N, 6]
        Returns: [6, N, 3, 3]
        """
        N = thetas.shape[0]
        device = thetas.device
        dtype = thetas.dtype

        I = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(N, -1, -1)  # [N, 3, 3]
        R_list = []

        for j in range(self.joint_count):
            ax = axes[j, 0]
            ay = axes[j, 1]
            az = axes[j, 2]
            # Skew-symmetric cross product matrix [a]_x
            K = torch.tensor(
                [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]],
                dtype=dtype,
                device=device,
            ).unsqueeze(0)  # [1, 3, 3]
            K2 = torch.matmul(K, K)  # [1, 3, 3]

            theta_j = thetas[:, j].view(N, 1, 1)  # [N, 1, 1]
            sin_j = torch.sin(theta_j)
            cos_j = torch.cos(theta_j)

            R_j = I + sin_j * K + (1.0 - cos_j) * K2  # [N, 3, 3]
            R_list.append(R_j)

        return torch.stack(R_list, dim=0)  # [6, N, 3, 3]

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes forward kinematics for generalized joint angles q.

        Args:
            q: Generalized joint angles [..., 6] in radians (e.g., [B, H, 6] or [B, 6]).

        Returns:
            End-effector tip 3D Cartesian coordinates [..., 3] in meters.
        """
        if q.shape[-1] != self.joint_count:
            raise ValueError(
                f"DifferentiableForwardKinematics: expected joint_count {self.joint_count}, got {q.shape[-1]}"
            )

        orig_shape = q.shape[:-1]
        q_flat = q.reshape(-1, self.joint_count)
        N = q_flat.shape[0]
        device = q.device
        dtype = q.dtype

        anchors = self.anchors.to(device=device, dtype=dtype)
        axes = self.axes.to(device=device, dtype=dtype)
        tip_offset = self.tip_offset.to(device=device, dtype=dtype)
        base_origin = self.base_origin.to(device=device, dtype=dtype)

        # 1. Base transformation: Translation by base_origin + Base_Root anchor
        T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)  # [N, 4, 4]
        T[:, :3, 3] = base_origin + anchors[0]

        # 2. Compute all 6 joint rotation matrices: [6, N, 3, 3]
        R_joints = self._build_rodrigues_matrices(axes, q_flat)

        # 3. Step sequentially through each articulated joint link:
        # Each link applies local translation by anchor[j + 1] followed by joint rotation R_j.
        # In homogeneous representation: T_link = [R_j, anchor; 0, 1]
        for j in range(self.joint_count):
            T_link = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)  # [N, 4, 4]
            T_link[:, :3, :3] = R_joints[j]
            T_link[:, :3, 3] = anchors[j + 1]
            T = torch.bmm(T, T_link)

        # 4. Final end-effector tip translation along Wrist_Roll local frame
        T_tip = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)  # [N, 4, 4]
        T_tip[:, :3, 3] = tip_offset
        T = torch.bmm(T, T_tip)

        tip_pos = T[:, :3, 3]
        out_shape = tuple(orig_shape) + (3,)
        result = tip_pos.view(*out_shape)
        assert_tensor_contract(result, out_shape, "DifferentiableForwardKinematics.tip_position")
        return result

    def forward_homogeneous(self, q: torch.Tensor) -> torch.Tensor:
        """
        Computes the full 4x4 homogeneous pose matrix for the end-effector tip.

        Args:
            q: Generalized joint angles [..., 6].

        Returns:
            End-effector tip 4x4 homogeneous transformation matrix [..., 4, 4].
        """
        if q.shape[-1] != self.joint_count:
            raise ValueError(
                f"DifferentiableForwardKinematics: expected joint_count {self.joint_count}, got {q.shape[-1]}"
            )

        orig_shape = q.shape[:-1]
        q_flat = q.reshape(-1, self.joint_count)
        N = q_flat.shape[0]
        device = q.device
        dtype = q.dtype

        anchors = self.anchors.to(device=device, dtype=dtype)
        axes = self.axes.to(device=device, dtype=dtype)
        tip_offset = self.tip_offset.to(device=device, dtype=dtype)
        base_origin = self.base_origin.to(device=device, dtype=dtype)

        T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        T[:, :3, 3] = base_origin + anchors[0]

        R_joints = self._build_rodrigues_matrices(axes, q_flat)

        for j in range(self.joint_count):
            T_link = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
            T_link[:, :3, :3] = R_joints[j]
            T_link[:, :3, 3] = anchors[j + 1]
            T = torch.bmm(T, T_link)

        T_tip = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        T_tip[:, :3, 3] = tip_offset
        T = torch.bmm(T, T_tip)

        out_shape = tuple(orig_shape) + (4, 4)
        return T.view(*out_shape)


__all__ = [
    "CANONICAL_ANCHORS",
    "CANONICAL_JOINT_AXES",
    "DifferentiableForwardKinematics",
    "END_EFFECTOR_TIP_OFFSET",
]

