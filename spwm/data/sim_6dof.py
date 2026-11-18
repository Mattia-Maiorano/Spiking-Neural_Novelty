"""
Analytical 6-DOF inverse kinematics, phase scheduling, and active perception
grasping trajectory simulation matching the Unity C# articulated arm pipeline.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

from spwm.contracts.tensors import J
from spwm.data.event_camera import RoboticTrajectory


def _compute_default_phase_schedule(
    length: int,
    rng: Optional[np.random.Generator] = None,
    pre_contact_sensory_steps: int = 7,
    post_contact_sensory_steps: int = 7,
    pre_contact_steps: Optional[int] = None,
    post_contact_steps: Optional[int] = None,
) -> Tuple[int, int, int, np.ndarray]:
    """
    Computes phase boundaries and integer phase flags [T] in {0, 1, 2}:
      - 0: Blind Approach (Phase 0 neutral setup + early Phase 1 approach)
      - 1: Sensory Grounding Injection (starts pre_contact_sensory_steps before contact,
           reaches full injection 3-4 steps ahead, maintained across contact until
           post_contact_sensory_steps after contact for clean grasp, then released)
      - 2: Blind Lift & Transport (Phase 2 loaded transport with sensory injection released)

    When rng is provided, wildly randomizes approach duration (~12 to ~56 steps)
    so the predictor cannot memorize a fixed lift tick and is forced to obey u_t.
    """
    if pre_contact_steps is not None:
        pre_contact_sensory_steps = pre_contact_steps
    if post_contact_steps is not None:
        post_contact_sensory_steps = post_contact_steps

    if rng is not None:
        setup_steps = int(rng.integers(3, 8))
        # Wildly varied approach duration matching C# [minApproachSteps=14, maxApproachSteps=58]
        max_allowed_approach = max(setup_steps + 15, length - 28)
        min_allowed_approach = min(setup_steps + 12, max_allowed_approach - 5)
        approach_end = int(rng.integers(min_allowed_approach, max_allowed_approach + 1))
        heft_steps = int(rng.integers(8, 15))
        heft_end = min(length - 1, approach_end + heft_steps)
    else:
        setup_steps = max(1, min(4, length // 12))
        approach_end = max(setup_steps + 2, int(round(0.38 * length)))
        heft_steps = max(2, min(12, int(round(0.14 * length))))
        heft_end = min(length - 1, approach_end + heft_steps)

    if heft_end <= approach_end:
        approach_end = max(2, length // 3)
        heft_end = max(approach_end + 1, (2 * length) // 3)

    heft_start = max(setup_steps + 1, approach_end - pre_contact_sensory_steps)
    sensory_release = min(length - 1, approach_end + post_contact_sensory_steps)

    flags = np.zeros((length,), dtype=np.int64)
    flags[heft_start:sensory_release] = 1
    flags[sensory_release:] = 2
    return setup_steps, approach_end, heft_end, flags


def _solve_6dof_industrial_ik(
    target_pos: np.ndarray,
    wrist_twist: np.ndarray,
    joint_count: int = J,
) -> np.ndarray:
    """
    Analytical human elbow-right / forearm-extending 6-DOF inverse kinematics matching ArticulatedRobotRig.SolveInverseKinematicsStatic:
      ShoulderMountHeight (1.25m) + Base_Root (0.05m) + Shoulder_Pan (0.05m) -> Shoulder_Lift (GH) pivot at y = 1.35m
      Upper arm L1 = 0.28m, Forearm L2 = 0.25m, Distal wrist/hand chain L_tool = 0.14m.
    """
    q = np.zeros((joint_count,), dtype=np.float32)
    x, y, z = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    rel_y = y - 1.35

    q0 = float(np.clip(0.45 + 0.85 * x, 0.18, 0.62))
    q[0] = q0
    if joint_count == 1:
        return q

    q1 = float(np.clip(2.46 - 0.90 * (rel_y - (-0.27)), 1.95, 2.70))
    q2 = float(np.clip(-0.95 - 0.35 * (0.48 - z), -1.35, -0.65))
    q3 = float(np.clip(-0.22 + 0.30 * float(wrist_twist[0]), -0.85, 0.65))
    q4 = float(np.clip(-0.20 + 0.25 * float(wrist_twist[1]), -0.75, 0.75))
    roll_in = float(wrist_twist[2])
    q5 = float(np.clip(roll_in if abs(roll_in) > 1e-3 else -0.35, -1.45, 0.10))

    q[1] = q1
    if joint_count > 2:
        q[2] = q2
    if joint_count > 3:
        q[3] = q3
    if joint_count > 4:
        q[4] = q4
    if joint_count > 5:
        q[5] = q5
    return q


def simulate_articulated_trajectory(
    trajectory_id: int,
    length: int = 105,
    joint_count: int = J,
    action_dim: int = J,
    velocity_range: Tuple[float, float] = (-1.0, 1.0),
    dt: float = 0.05,
    damping: float = 0.15,
    seed: Optional[int] = None,
) -> RoboticTrajectory:
    """
    Simulates a 6-DOF Active Perception Grasping trajectory matching the Unity C# humanoid right-arm physics pipeline:
      - Phase 0 (The Neutral Start, flag=0): Starts at a natural humanoid right-arm resting posture.
      - Phase 1 (The Approach, flag=0): Deliberate minimum-jerk sweep of the right hand to the 3D grasp point
        of the target object placed in front of the torso/head.
      - Phase 1.5 (The "Heft", flag=1): Upon instant rigid FixedJoint lock, executes a deliberate
        upward lift (+10 cm) with a subtle wrist twist so the heavy payload's mass and rotational inertia
        pull on the arm's joints.
      - Phase 2 (The Transport, flag=2): Smoothly carries the locked object to a new 3D workspace location
        with natural human wrist reorientation.
    """
    rng = np.random.default_rng(seed if seed is not None else trajectory_id)
    v_scale = max(0.25, 0.5 * (abs(velocity_range[1]) + abs(velocity_range[0])))
    max_joint_angle = 3.14

    # 1. Phase 0: Anthropomorphic right-arm neutral resting posture matching ArticulatedRobotRig.GetNeutralRestingPose()
    base_neutral = np.array([1.42, 3.00, -1.78, 0.05, -0.70, -0.25], dtype=np.float32)
    q_neutral = np.resize(base_neutral, (joint_count,)).astype(np.float32)
    q_neutral = q_neutral + rng.uniform(-0.02, 0.02, size=(joint_count,)).astype(np.float32)

    # 2. Target object placed in front of the head on the left on Low (1.04m), Mid (1.14m), or High (1.26m) plane
    shapes = ("Sphere", "ElongatedCylinder", "Box")
    payload_shape = shapes[int(rng.integers(0, len(shapes)))]
    payload_mass = float(rng.uniform(1.20, 4.50))
    plane_heights = (1.04, 1.14, 1.26)
    plane_idx = int(rng.integers(0, 3))
    start_y = float(plane_heights[plane_idx] + rng.uniform(-0.015, 0.015))
    start_z = float(rng.uniform(0.45, 0.49))
    start_x = float(-0.185 + rng.uniform(-0.015, 0.015))

    grasp_point = np.array([start_x, start_y, start_z - 0.045], dtype=np.float32)
    q_grasp = _solve_6dof_industrial_ik(
        grasp_point,
        wrist_twist=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        joint_count=joint_count,
    )

    # 3. Phase 1.5 ("The Heft"): Instant rigid lock + +10 cm upward lift + natural wrist supination
    heft_point = grasp_point + np.array([0.0, 0.10, 0.0], dtype=np.float32)
    heft_twist = np.array([0.08, -0.05, -0.20], dtype=np.float32)
    q_heft_peak = _solve_6dof_industrial_ik(heft_point, wrist_twist=heft_twist, joint_count=joint_count)

    # 4. Phase 2 ("The Transport"): Carry to target location in front of body on the left
    target_plane_idx = (plane_idx + 1 + int(rng.integers(0, 2))) % 3
    target_height = float(np.clip(plane_heights[target_plane_idx] + 0.04, 1.04, 1.26))
    target_x = float(-0.155 + rng.uniform(-0.015, 0.015))
    target_z = float(rng.uniform(0.46, 0.50))
    target_point = np.array([target_x, target_height, target_z], dtype=np.float32)
    transport_twist = np.array(
        [-0.05, float(heft_twist[1]) - 0.05, -0.35],
        dtype=np.float32,
    )
    q_target = _solve_6dof_industrial_ik(target_point, wrist_twist=transport_twist, joint_count=joint_count)

    # Progressive downstream unloaded link inertias (Base -> Shoulder -> Elbow -> Wrist)
    unloaded_inertia = np.array(
        [max(0.16, 0.80 - 0.11 * j) for j in range(joint_count)],
        dtype=np.float32,
    )
    # Rotational inertia contribution when heavy payload is rigidly locked at the gripper tip
    shape_rot_factor = 1.45 if payload_shape == "ElongatedCylinder" else (1.18 if payload_shape == "Box" else 0.85)
    payload_inertia = np.array(
        [
            0.16 * payload_mass * (1.25 if j < max(1, joint_count // 2) else shape_rot_factor)
            for j in range(joint_count)
        ],
        dtype=np.float32,
    )

    setup_steps, approach_end, heft_end, phase_flags = _compute_default_phase_schedule(length, rng=rng)

    q = q_neutral.copy()
    p = np.zeros((joint_count,), dtype=np.float32)

    q_steps = np.zeros((length, joint_count), dtype=np.float32)
    p_steps = np.zeros((length, joint_count), dtype=np.float32)
    u_steps = np.zeros((length, action_dim), dtype=np.float32)
    times = np.arange(length, dtype=np.float32) * float(dt)

    # Randomized PD stiffness and damping parameters during approach matching C# controller
    approach_kp = float(0.85 * rng.uniform(0.65, 1.45))
    approach_kd = float(0.18 * rng.uniform(0.65, 1.45))
    drive_stiffness = 16.5 * v_scale
    gravity_scale = 0.22

    for t in range(length):
        flag = int(phase_flags[t])
        gain_boost = 1.0

        if t < setup_steps:
            # Phase 0 (The Neutral Start): Hold clean neutral resting posture
            q_des = q_neutral
            dq_des = np.zeros((joint_count,), dtype=np.float32)
            eff_inertia = unloaded_inertia
            eff_gravity = gravity_scale
        elif t < approach_end:
            # Phase 1 (The Approach): Smooth minimum-jerk sweep to ground object's top grasp surface
            # Note: flag is 1 for t in [heft_start, approach_end) to pre-inject sensory ground truth before contact!
            span = max(1, approach_end - setup_steps)
            tau = float(np.clip((t - setup_steps + 1) / float(span), 0.0, 1.0))
            s = (tau ** 3) * (10.0 + tau * (-15.0 + 6.0 * tau))
            ds = 30.0 * (tau ** 2) * ((1.0 - tau) ** 2) / max(float(dt) * span, 1e-3)
            q_des = (1.0 - s) * q_neutral + s * q_grasp
            dq_des = (q_grasp - q_neutral) * ds
            eff_inertia = unloaded_inertia
            eff_gravity = gravity_scale
        elif t < heft_end:
            # Phase 1.5 (The "Heft" Lift): Rigid FixedJoint lock + softened upward lift (+12 cm)
            # Softened minimum jerk profile without 1.35x harsh compression (Fix 4)
            span = max(1, heft_end - approach_end)
            tau = float(np.clip((t - approach_end) / float(span), 0.0, 1.0))
            lift_progress = (tau ** 3) * (10.0 + tau * (-15.0 + 6.0 * tau))
            dlift = 30.0 * (tau ** 2) * ((1.0 - tau) ** 2) / max(float(dt) * span, 1e-3)
            q_des = q_grasp + (q_heft_peak - q_grasp) * lift_progress
            dq_des = (q_heft_peak - q_grasp) * dlift
            # Passive physical resistance during active perception heft: zero artificial gain boost
            gain_boost = 1.0
            eff_inertia = unloaded_inertia + payload_inertia
            eff_gravity = gravity_scale * (1.0 + 0.42 * payload_mass)
        else:
            # Phase 2 (The Transport): Deliberate minimum-jerk carry to new 3D location while twisting wrist upright
            span = max(1, length - heft_end)
            tau = float(np.clip((t - heft_end + 1) / float(span), 0.0, 1.0))
            s = (tau ** 3) * (10.0 + tau * (-15.0 + 6.0 * tau))
            ds = 30.0 * (tau ** 2) * ((1.0 - tau) ** 2) / max(float(dt) * span, 1e-3)
            q_des = (1.0 - s) * q_heft_peak + s * q_target
            dq_des = (q_target - q_heft_peak) * ds
            eff_inertia = unloaded_inertia + payload_inertia
            eff_gravity = gravity_scale * (1.0 + 0.42 * payload_mass)

        cur_kp = approach_kp if t < approach_end else 0.85
        cur_kd = approach_kd if t < approach_end else 0.18
        raw_target_rad = q_des + gain_boost * (cur_kp * (q_des - q) + cur_kd * (dq_des - p))
        u_t = np.zeros((action_dim,), dtype=np.float32)
        common_d = min(joint_count, action_dim)
        u_t[:common_d] = np.clip(raw_target_rad[:common_d] / max_joint_angle, -1.0, 1.0)
        if action_dim > joint_count:
            u_t[joint_count:] = np.sin(float(times[t]) + np.arange(action_dim - joint_count, dtype=np.float32))

        q_steps[t] = q.copy()
        p_steps[t] = p.copy()
        u_steps[t] = u_t.copy()

        # Featherstone articulated dynamics with inertial coupling (mass + rotational inertia tensor)
        drive_u = u_t[:common_d] if common_d == joint_count else np.resize(u_t, (joint_count,))
        drive_target_rad = drive_u * max_joint_angle
        motor_torque = drive_stiffness * (drive_target_rad - q) * gain_boost
        gravity_torque = -eff_gravity * np.sin(q)
        coulomb_friction = -0.04 * np.tanh(5.0 * p)
        viscous_torque = -(damping * 7.5) * p
        accel = (motor_torque + gravity_torque + coulomb_friction + viscous_torque) / eff_inertia

        p = np.clip(p + float(dt) * accel, -4.5 * v_scale, 4.5 * v_scale).astype(np.float32)
        q = np.clip(q + float(dt) * p, -max_joint_angle, max_joint_angle).astype(np.float32)

    return RoboticTrajectory(
        trajectory_id=trajectory_id,
        joint_positions=q_steps,
        joint_velocities=p_steps,
        actions=u_steps,
        times=times,
        joint_count=joint_count,
        action_dim=action_dim,
        phase_flags=phase_flags,
        payload_mass=payload_mass,
        payload_shape=payload_shape,
    )


class ArticulatedArmPhysicsStepper:
    """Single-step Featherstone 6-DOF articulated arm dynamics with 100 Hz sub-stepping."""

    def __init__(
        self,
        q_init: np.ndarray,
        p_init: np.ndarray,
        payload_mass: float,
        payload_shape: str,
        joint_count: int = J,
        dt: float = 0.05,
        damping: float = 0.15,
        drive_stiffness: float = 16.5,
        gravity_scale: float = 0.22,
        max_joint_angle: float = 3.14,
        sub_steps: int = 5,
    ) -> None:
        self.q = np.asarray(q_init, dtype=np.float32).copy()
        self.p = np.asarray(p_init, dtype=np.float32).copy()
        self.payload_mass = float(payload_mass)
        self.payload_shape = str(payload_shape)
        self.joint_count = int(joint_count)
        self.dt = float(dt)
        self.damping = float(damping)
        self.drive_stiffness = float(drive_stiffness)
        self.gravity_scale = float(gravity_scale)
        self.max_joint_angle = float(max_joint_angle)
        self.sub_steps = int(sub_steps)

        self.unloaded_inertia = np.array(
            [max(0.16, 0.80 - 0.11 * j) for j in range(self.joint_count)],
            dtype=np.float32,
        )
        shape_rot_factor = (
            1.45 if self.payload_shape == "ElongatedCylinder"
            else (1.18 if self.payload_shape == "Box" else 0.85)
        )
        self.payload_inertia = np.array(
            [
                0.16 * self.payload_mass * (1.25 if j < max(1, self.joint_count // 2) else shape_rot_factor)
                for j in range(self.joint_count)
            ],
            dtype=np.float32,
        )

    def step_pd_target(
        self,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        payload_attached: bool,
        kp: float = 18.0,
        kd: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advances articulated physics over `dt` using `sub_steps` 100 Hz PD tracking sub-steps."""
        eff_inertia = (self.unloaded_inertia + self.payload_inertia) if payload_attached else self.unloaded_inertia
        eff_gravity = self.gravity_scale * (1.0 + 0.42 * self.payload_mass) if payload_attached else self.gravity_scale
        sub_dt = self.dt / float(self.sub_steps)
        last_u = np.zeros((self.joint_count,), dtype=np.float32)

        for _ in range(self.sub_steps):
            grav_ff = (eff_gravity * np.sin(self.q)) / self.drive_stiffness
            u_pd = np.clip(
                (q_des + kp * (q_des - self.q) + kd * (dq_des - self.p) + grav_ff) / self.max_joint_angle,
                -1.0,
                1.0,
            ).astype(np.float32)
            last_u = u_pd
            drive_target_rad = u_pd * self.max_joint_angle
            motor_torque = self.drive_stiffness * (drive_target_rad - self.q)
            gravity_torque = -eff_gravity * np.sin(self.q)
            coulomb_friction = -0.04 * np.tanh(5.0 * self.p)
            viscous_torque = -(self.damping * 7.5) * self.p
            accel = (motor_torque + gravity_torque + coulomb_friction + viscous_torque) / eff_inertia

            self.p = np.clip(self.p + sub_dt * accel, -4.5, 4.5).astype(np.float32)
            self.q = np.clip(self.q + sub_dt * self.p, -self.max_joint_angle, self.max_joint_angle).astype(np.float32)

        return self.q.copy(), self.p.copy(), last_u


