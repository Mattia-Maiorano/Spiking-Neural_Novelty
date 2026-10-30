"""
PyTorch Dataset and DataLoader pipelines for SPWM-v0.5.0 (6-DOF Active Perception Grasping).
Parses exported Unity robotic trajectories (joint positions q* in R^6, joint velocities p* in R^6,
motor actions u_t in R^6, phase flags in {0, 1, 2}, and neuromorphic event streams) while enforcing
strict trajectory-level partitioning to prevent temporal leakage.
"""

from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spwm.contracts.tensors import C_EVENT, J, assert_tensor_contract
from spwm.data.event_camera import EventBatch, EventCameraSimulator, RoboticTrajectory


def partition_trajectories(
    total_trajectories: int = 1000,
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Dict[str, Tuple[int, int]]:
    """
    Partitions trajectory IDs strictly by trajectory index to eliminate temporal leakage.
    Example: 1000 trajectories -> train [0, 800), val [800, 900), test [900, 1000).
    """
    n_train = int(total_trajectories * train_split)
    n_val = int(total_trajectories * val_split)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, total_trajectories),
    }


def _compute_default_phase_schedule(
    length: int,
    rng: Optional[np.random.Generator] = None,
    pre_contact_steps: int = 5,
) -> Tuple[int, int, int, np.ndarray]:
    """
    Computes phase boundaries and integer phase flags [T] in {0, 1, 2}:
      - 0: Blind Approach (Phase 0 neutral setup + early Phase 1 approach)
      - 1: Sensory Heft (Phase 1.5 entered n steps before contact for full ground truth injection,
           followed by rigid lock and smooth upward heft lift)
      - 2: Blind Transport (Phase 2 loaded transport with active wrist rotation)

    When rng is provided, wildly randomizes approach duration (~12 to ~56 steps)
    so the predictor cannot memorize a fixed lift tick and is forced to obey u_t.
    """
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

    heft_start = max(setup_steps + 1, approach_end - pre_contact_steps)

    flags = np.zeros((length,), dtype=np.int64)
    flags[heft_start:heft_end] = 1
    flags[heft_end:] = 2
    return setup_steps, approach_end, heft_end, flags


def _solve_6dof_industrial_ik(
    target_pos: np.ndarray,
    wrist_twist: np.ndarray,
    joint_count: int = J,
) -> np.ndarray:
    """
    Analytical elbow-up 6-DOF inverse kinematics matching ArticulatedRobotRig.SolveInverseKinematics6D:
      Base_Root (0.16m) + Shoulder_Pan (0.14m) -> Shoulder_Lift pivot at y = 0.30m
      Upper arm L1 = 0.30m, Forearm L2 = 0.26m, Distal wrist chain L_tool = 0.28m.
    """
    q = np.zeros((joint_count,), dtype=np.float32)
    x, y, z = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])

    # 1. Joint 0 (Shoulder_Pan): rotate base yaw so arm plane aligns with (x, z)
    q0 = float(np.clip(np.arctan2(x, z), -2.60, 2.60))
    q[0] = q0
    if joint_count == 1:
        return q

    r_tip = float(np.sqrt(x * x + z * z))
    y_shoulder = 0.30
    l1, l2, l_tool = 0.30, 0.26, 0.28

    # Orient distal wrist links ~29 deg from vertical downward toward ground/workspace target
    alpha = 0.50
    r_w = max(0.05, r_tip - l_tool * float(np.sin(alpha)))
    y_w = y + l_tool * float(np.cos(alpha))
    dy = y_w - y_shoulder
    dist = float(np.sqrt(r_w * r_w + dy * dy))

    max_reach = (l1 + l2) * 0.992
    min_reach = abs(l1 - l2) * 1.05
    if dist > max_reach and dist > 1e-5:
        scale = max_reach / dist
        r_w *= scale
        dy *= scale
        dist = max_reach
    elif dist < min_reach and dist > 1e-5:
        scale = min_reach / dist
        r_w *= scale
        dy *= scale
        dist = min_reach

    cos_q2 = float(np.clip((dist * dist - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0))
    q2 = float(np.clip(np.arccos(cos_q2), 0.0, 2.60))
    beta = float(np.arctan2(r_w, dy))
    psi = float(np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2)))
    q1 = float(np.clip(beta - psi, -2.60, 2.60))
    q3_base = float(np.clip((np.pi - alpha) - (q1 + q2), -2.60, 2.60))

    q[1] = q1
    if joint_count > 2:
        q[2] = q2
    if joint_count > 3:
        q[3] = float(np.clip(q3_base + float(wrist_twist[0]), -2.60, 2.60))
    if joint_count > 4:
        q[4] = float(np.clip(float(wrist_twist[1]), -2.60, 2.60))
    if joint_count > 5:
        q[5] = float(np.clip(float(wrist_twist[2]), -2.60, 2.60))
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
    Simulates a 6-DOF Active Perception Grasping trajectory matching the Unity C# industrial physics pipeline:
      - Phase 0 (The Neutral Start, flag=0): Starts at a clean, poised neutral resting posture.
      - Phase 1 (The Approach, flag=0): Deliberate minimum-jerk sweep of the end-effector tip to the exact
        3D location of the target object sitting on the ground.
      - Phase 1.5 (The "Heft", flag=1): Upon instant rigid FixedJoint lock, executes a sharp, deliberate
        upward lift (+12 cm) with a crisp wrist twist so the heavy payload's mass and rotational inertia
        pull on the arm's joints.
      - Phase 2 (The Transport, flag=2): Smoothly carries the locked object to a new 3D workspace location
        while deliberately twisting the wrist joints (Pitch, Yaw, Roll) to rotate the object from flat to upright.
    """
    rng = np.random.default_rng(seed if seed is not None else trajectory_id)
    v_scale = max(0.25, 0.5 * (abs(velocity_range[1]) + abs(velocity_range[0])))
    max_joint_angle = 2.60

    # 1. Phase 0: Clean neutral resting posture matching ArticulatedRobotRig.GetNeutralRestingPose()
    base_neutral = np.array([0.0, 0.35, 0.85, 0.35, 0.0, 0.0], dtype=np.float32)
    q_neutral = np.resize(base_neutral, (joint_count,)).astype(np.float32)
    q_neutral = q_neutral + rng.uniform(-0.02, 0.02, size=(joint_count,)).astype(np.float32)

    # 2. Target object resting flat on the ground plane (y = half_height)
    shapes = ("Sphere", "ElongatedCylinder", "Box")
    payload_shape = shapes[int(rng.integers(0, len(shapes)))]
    payload_mass = float(rng.uniform(1.20, 4.50))
    half_height = 0.035

    # Wildly varied distance and azimuth matching C# spawner [minReachRadius=0.22, maxReachRadius=0.52]
    start_radius = float(rng.uniform(0.22, 0.52))
    start_azimuth = float(rng.uniform(-0.75, 0.75))
    if abs(start_azimuth) < 0.12:
        start_azimuth = 0.18 if start_azimuth >= 0.0 else -0.18

    grasp_point = np.array(
        [
            start_radius * np.sin(start_azimuth),
            2.0 * half_height,  # Top surface of the object sitting on the ground at y = 0
            start_radius * np.cos(start_azimuth),
        ],
        dtype=np.float32,
    )
    q_grasp = _solve_6dof_industrial_ik(
        grasp_point,
        wrist_twist=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        joint_count=joint_count,
    )

    # 3. Phase 1.5 ("The Heft"): Instant rigid lock + sharp +12 cm upward lift + crisp wrist twist
    heft_point = grasp_point + np.array([0.0, 0.12, 0.0], dtype=np.float32)
    heft_twist = np.array([0.24, -0.28, 0.36], dtype=np.float32)
    q_heft_peak = _solve_6dof_industrial_ik(heft_point, wrist_twist=heft_twist, joint_count=joint_count)

    # 4. Phase 2 ("The Transport"): Carry to new elevated 3D location on opposite side while twisting wrist upright
    target_radius = float(rng.uniform(0.30, 0.42))
    target_azimuth = -float(np.sign(start_azimuth)) * float(rng.uniform(0.35, 0.65))
    target_height = float(rng.uniform(0.26, 0.38))
    target_point = np.array(
        [
            target_radius * np.sin(target_azimuth),
            target_height,
            target_radius * np.cos(target_azimuth),
        ],
        dtype=np.float32,
    )
    side_sign = 1.0 if target_point[0] >= 0.0 else -1.0
    transport_twist = np.array(
        [-0.55, float(heft_twist[1]) + side_sign * 0.95, float(heft_twist[2]) + side_sign * 1.45],
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
            gain_boost = 1.2
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


def export_unity_trajectory_file(
    trajectory: RoboticTrajectory,
    file_path: Union[str, Path],
    sensor_height: int = 32,
    sensor_width: int = 32,
    dt: float = 0.05,
    event_threshold: float = 0.08,
) -> Path:
    """Serializes a RoboticTrajectory to Unity JSON or NPZ telemetry format on disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    T = trajectory.joint_positions.shape[0]
    phase_flags = (
        trajectory.phase_flags.astype(np.int64)
        if trajectory.phase_flags is not None
        else _compute_default_phase_schedule(T)[3]
    )

    if path.suffix.lower() == ".npz":
        payload: Dict[str, Any] = {
            "trajectory_id": np.array(trajectory.trajectory_id, dtype=np.int64),
            "joint_positions": trajectory.joint_positions.astype(np.float32),
            "joint_velocities": trajectory.joint_velocities.astype(np.float32),
            "actions": trajectory.actions.astype(np.float32),
            "phase_flags": phase_flags,
            "times": trajectory.times.astype(np.float32),
        }
        if trajectory.grayscale_frames is not None:
            payload["grayscale_frames"] = trajectory.grayscale_frames.astype(np.float32)
        if trajectory.wrist_grayscale_frames is not None:
            payload["wrist_grayscale_frames"] = trajectory.wrist_grayscale_frames.astype(np.float32)
        if trajectory.dense_events is not None:
            payload["dense_events"] = trajectory.dense_events.astype(np.float32)
        np.savez_compressed(path, **payload)
        return path

    steps = []
    for t in range(T):
        step_rec: Dict[str, Any] = {
            "step": t,
            "timestamp": float(trajectory.times[t]),
            "phase_flag": int(phase_flags[t]),
            "joint_positions": trajectory.joint_positions[t].tolist(),
            "joint_velocities": trajectory.joint_velocities[t].tolist(),
            "actions": trajectory.actions[t].tolist(),
        }
        if trajectory.grayscale_frames is not None:
            if trajectory.grayscale_frames.ndim == 4 and trajectory.grayscale_frames.shape[1] == 2:
                step_rec["grayscale_frame"] = trajectory.grayscale_frames[t, 0].reshape(-1).tolist()
                step_rec["wrist_grayscale_frame"] = trajectory.grayscale_frames[t, 1].reshape(-1).tolist()
            else:
                step_rec["grayscale_frame"] = trajectory.grayscale_frames[t].reshape(-1).tolist()
        if trajectory.wrist_grayscale_frames is not None and "wrist_grayscale_frame" not in step_rec:
            step_rec["wrist_grayscale_frame"] = trajectory.wrist_grayscale_frames[t].reshape(-1).tolist()
        if trajectory.dense_events is not None:
            step_rec["dense_events"] = trajectory.dense_events[t].reshape(-1).tolist()
        steps.append(step_rec)

    doc = {
        "trajectory_id": int(trajectory.trajectory_id),
        "archetype": "ManipulatorArm",
        "payload_shape": trajectory.payload_shape,
        "payload_mass": float(trajectory.payload_mass),
        "sequence_length": int(T),
        "joint_count": int(trajectory.joint_count),
        "action_dim": int(trajectory.action_dim),
        "sensor_channels": int(
            trajectory.dense_events.shape[1]
            if trajectory.dense_events is not None and trajectory.dense_events.ndim == 4
            else C_EVENT
        ),
        "sensor_height": int(sensor_height),
        "sensor_width": int(sensor_width),
        "dt": float(dt),
        "event_threshold": float(event_threshold),
        "phase_flags": phase_flags.tolist(),
        "steps": steps,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return path


def parse_unity_trajectory_file(
    file_path: Union[str, Path],
    sequence_length: Optional[int] = None,
    sensor_height: int = 32,
    sensor_width: int = 32,
) -> RoboticTrajectory:
    """Parses an exported Unity trajectory (.json, .json.gz, or .npz) into a RoboticTrajectory."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Unity trajectory file not found: {path}")

    payload_mass = 1.5
    payload_shape = "Box"

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            traj_id = int(data["trajectory_id"])
            q = data["joint_positions"].astype(np.float32)
            p = data["joint_velocities"].astype(np.float32)
            u = data["actions"].astype(np.float32)
            times = data["times"].astype(np.float32) if "times" in data else np.arange(len(q), dtype=np.float32) * 0.05
            phase_flags = (
                data["phase_flags"].astype(np.int64)
                if "phase_flags" in data
                else _compute_default_phase_schedule(len(q))[3]
            )
            gray = data["grayscale_frames"].astype(np.float32) if "grayscale_frames" in data else None
            wrist_gray = data["wrist_grayscale_frames"].astype(np.float32) if "wrist_grayscale_frames" in data else None
            ev = data["dense_events"].astype(np.float32) if "dense_events" in data else None
    else:
        if path.suffix.lower() == ".gz" or str(path).lower().endswith(".json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                doc = json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        traj_id = int(doc["trajectory_id"])
        H = int(doc.get("sensor_height", sensor_height))
        W = int(doc.get("sensor_width", sensor_width))
        dt = float(doc.get("dt", 0.05))
        payload_mass = float(doc.get("payload_mass", 1.5))
        payload_shape = str(doc.get("payload_shape", "Box"))
        steps = doc["steps"]

        q = np.array([s["joint_positions"] for s in steps], dtype=np.float32)
        p = np.array([s["joint_velocities"] for s in steps], dtype=np.float32)
        u = np.array([s["actions"] for s in steps], dtype=np.float32)
        times = np.array([s.get("timestamp", idx * dt) for idx, s in enumerate(steps)], dtype=np.float32)

        if steps and "phase_flag" in steps[0]:
            phase_flags = np.array([int(s["phase_flag"]) for s in steps], dtype=np.int64)
        elif "phase_flags" in doc and doc["phase_flags"]:
            phase_flags = np.array(doc["phase_flags"], dtype=np.int64)
        else:
            phase_flags = _compute_default_phase_schedule(len(steps))[3]

        gray = None
        wrist_gray = None
        if steps and "grayscale_frame" in steps[0] and steps[0]["grayscale_frame"]:
            raw_gray = np.array([s["grayscale_frame"] for s in steps], dtype=np.float32)
            if raw_gray.shape[-1] == 2 * H * W:
                dual_gray = raw_gray.reshape(len(steps), 2, H, W)
                if float(np.sum(dual_gray)) > 0.0:
                    gray = dual_gray[:, 0]
                    wrist_gray = dual_gray[:, 1]
            else:
                gray_arr = raw_gray.reshape(len(steps), H, W)
                if float(np.sum(gray_arr)) > 0.0:
                    gray = gray_arr

        if steps and "wrist_grayscale_frame" in steps[0] and steps[0]["wrist_grayscale_frame"]:
            wrist_arr = np.array([s["wrist_grayscale_frame"] for s in steps], dtype=np.float32).reshape(len(steps), H, W)
            if float(np.sum(wrist_arr)) > 0.0:
                wrist_gray = wrist_arr

        ev = None
        if steps and "dense_events" in steps[0] and steps[0]["dense_events"]:
            raw_ev = np.array([s["dense_events"] for s in steps], dtype=np.float32)
            ev_channels = max(2, int(raw_ev.shape[-1] // max(1, H * W)))
            ev_arr = raw_ev.reshape(len(steps), ev_channels, H, W)
            if float(np.sum(ev_arr)) > 0.0:
                ev = ev_arr

    if sequence_length is not None and q.shape[0] > sequence_length:
        q = q[:sequence_length]
        p = p[:sequence_length]
        u = u[:sequence_length]
        times = times[:sequence_length]
        phase_flags = phase_flags[:sequence_length]
        if gray is not None:
            gray = gray[:sequence_length]
        if wrist_gray is not None:
            wrist_gray = wrist_gray[:sequence_length]
        if ev is not None:
            ev = ev[:sequence_length]

    return RoboticTrajectory(
        trajectory_id=traj_id,
        joint_positions=q,
        joint_velocities=p,
        actions=u,
        times=times,
        joint_count=int(q.shape[-1]),
        action_dim=int(u.shape[-1]),
        phase_flags=phase_flags,
        payload_mass=payload_mass,
        payload_shape=payload_shape,
        grayscale_frames=gray,
        wrist_grayscale_frames=wrist_gray,
        dense_events=ev,
    )


class EventWorldDataset(Dataset):
    """
    Dataset of neuromorphic event streams and 6-DOF articulated manipulator telemetry (q*, p*, u_t, phase_flags).
    Parses exported Unity trajectory files from `dataset_dir` when present, and falls back to
    deterministic 6-DOF active perception grasping simulation seeded strictly by trajectory_id.
    """

    def __init__(
        self,
        trajectory_indices: List[int],
        dataset_dir: Optional[Union[str, Path]] = None,
        event_simulator: Optional[EventCameraSimulator] = None,
        sequence_length: int = 105,
        joint_count: int = J,
        action_dim: int = J,
        velocity_range: Tuple[float, float] = (-1.0, 1.0),
        dt: float = 0.05,
        cache_data: bool = True,
        strict_dataset: bool = True,
    ) -> None:
        super().__init__()
        self.trajectory_indices = list(trajectory_indices)
        self.dataset_dir = Path(dataset_dir) if dataset_dir is not None else None
        self.event_simulator = event_simulator or EventCameraSimulator()
        self.sequence_length = sequence_length
        self.joint_count = joint_count
        self.action_dim = action_dim
        self.velocity_range = velocity_range
        self.dt = float(dt)
        self.cache_data = cache_data
        self.strict_dataset = strict_dataset

        if self.strict_dataset:
            if self.dataset_dir is None:
                raise FileNotFoundError(
                    "Strict dataset validation error: `dataset_dir` is None. "
                    "SPWM contracts require exported Unity trajectories (e.g. at 'sim/ExportedTrajectories'). "
                    "Synthetic simulation fallback is disabled."
                )
            if not self.dataset_dir.is_dir():
                raise FileNotFoundError(
                    f"Strict dataset validation error: `dataset_dir` '{self.dataset_dir}' does not exist. "
                    "Please generate the real Unity trajectory dataset using the Unity physics simulation batch export."
                )
            missing = [tid for tid in self.trajectory_indices if self._resolve_unity_file(tid) is None]
            if missing:
                missing_preview = missing[:10]
                suffix = "..." if len(missing) > 10 else ""
                raise FileNotFoundError(
                    f"Strict dataset validation error: Missing {len(missing)} Unity trajectory files in '{self.dataset_dir}'. "
                    f"Sample missing IDs: {missing_preview}{suffix}. "
                    "Ensure Unity batch export has run for all requested trajectory IDs."
                )

        self._cache: Dict[int, Dict[str, Any]] = {}
        if cache_data:
            for traj_id in self.trajectory_indices:
                self._cache[traj_id] = self._load_or_simulate_item(traj_id)

    def _resolve_unity_file(self, traj_id: int) -> Optional[Path]:
        if self.dataset_dir is None or not self.dataset_dir.is_dir():
            return None
        for candidate in (
            self.dataset_dir / f"trajectory_{traj_id:05d}.json.gz",
            self.dataset_dir / f"trajectory_{traj_id:05d}.npz",
            self.dataset_dir / f"trajectory_{traj_id:05d}.json",
            self.dataset_dir / f"trajectory_{traj_id}.json.gz",
            self.dataset_dir / f"trajectory_{traj_id}.npz",
            self.dataset_dir / f"trajectory_{traj_id}.json",
        ):
            if candidate.is_file():
                return candidate
        return None

    def _load_or_simulate_item(self, traj_id: int) -> Dict[str, Any]:
        unity_file = self._resolve_unity_file(traj_id)
        if unity_file is not None:
            traj = parse_unity_trajectory_file(
                unity_file,
                sequence_length=self.sequence_length,
                sensor_height=self.event_simulator.height,
                sensor_width=self.event_simulator.width,
            )
        else:
            traj = simulate_articulated_trajectory(
                trajectory_id=traj_id,
                length=self.sequence_length,
                joint_count=self.joint_count,
                action_dim=self.action_dim,
                velocity_range=self.velocity_range,
                dt=self.dt,
                seed=traj_id,
            )

        event_batch: EventBatch = self.event_simulator.trajectory_to_events(traj, return_sparse=False)
        q_tensor = torch.from_numpy(traj.joint_positions).float()
        p_tensor = torch.from_numpy(traj.joint_velocities).float()
        u_tensor = torch.from_numpy(traj.actions).float()
        flat_kin = torch.cat([q_tensor, p_tensor], dim=-1)

        if traj.phase_flags is not None:
            phase_flags_np = traj.phase_flags
        else:
            phase_flags_np = _compute_default_phase_schedule(q_tensor.shape[0])[3]
        phase_flags_tensor = torch.from_numpy(np.asarray(phase_flags_np, dtype=np.int64)).long()

        T_actual = q_tensor.shape[0]
        assert_tensor_contract(
            event_batch.dense_events,
            (T_actual, self.event_simulator.in_channels, self.event_simulator.height, self.event_simulator.width),
            "EventWorldDataset.events",
            check_finite=True,
        )
        assert_tensor_contract(q_tensor, (T_actual, self.joint_count), "EventWorldDataset.joint_positions", check_finite=True)
        assert_tensor_contract(p_tensor, (T_actual, self.joint_count), "EventWorldDataset.joint_velocities", check_finite=True)
        assert_tensor_contract(u_tensor, (T_actual, self.action_dim), "EventWorldDataset.actions", check_finite=True)
        assert_tensor_contract(flat_kin, (T_actual, 2 * self.joint_count), "EventWorldDataset.flat_kinematics", check_finite=True)
        assert_tensor_contract(phase_flags_tensor, (T_actual,), "EventWorldDataset.phase_flags")

        return {
            "events": event_batch.dense_events,      # [T, 4, H, W] (dual-camera: global + wrist)
            "joint_positions": q_tensor,             # [T, joint_count] (q^*)
            "joint_velocities": p_tensor,            # [T, joint_count] (p^*)
            "positions": q_tensor,                   # [T, joint_count]
            "velocities": p_tensor,                  # [T, joint_count]
            "actions": u_tensor,                     # [T, action_dim] (u_t)
            "flat_kinematics": flat_kin,             # [T, 2 * joint_count]
            "phase_flags": phase_flags_tensor,       # [T] int64 in {0, 1, 2}
            "trajectory_id": traj_id,
        }

    def __len__(self) -> int:
        return len(self.trajectory_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_id = self.trajectory_indices[idx]
        if self.cache_data and traj_id in self._cache:
            return self._cache[traj_id]
        return self._load_or_simulate_item(traj_id)


def collate_event_batches(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collates a list of trajectory items into a batched dictionary with actions u_t and phase_flags."""
    events = torch.stack([item["events"] for item in batch], dim=0)                      # [B, T, 2, H, W]
    flat_kinematics = torch.stack([item["flat_kinematics"] for item in batch], dim=0)    # [B, T, 2 * J]
    joint_positions = torch.stack([item["joint_positions"] for item in batch], dim=0)    # [B, T, J]
    joint_velocities = torch.stack([item["joint_velocities"] for item in batch], dim=0)  # [B, T, J]
    actions = torch.stack([item["actions"] for item in batch], dim=0)                    # [B, T, D_u]
    phase_flags = torch.stack([item["phase_flags"] for item in batch], dim=0)            # [B, T]
    trajectory_ids = torch.tensor([item["trajectory_id"] for item in batch], dtype=torch.long)

    return {
        "events": events,
        "flat_kinematics": flat_kinematics,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,
        "positions": joint_positions,
        "velocities": joint_velocities,
        "actions": actions,
        "phase_flags": phase_flags,
        "trajectory_ids": trajectory_ids,
    }


def create_dataloaders(
    total_trajectories: int = 1000,
    train_split: float = 0.8,
    val_split: float = 0.1,
    sequence_length: int = 105,
    batch_size: int = 32,
    height: int = 32,
    width: int = 32,
    joint_count: int = J,
    action_dim: int = J,
    in_channels: int = C_EVENT,
    dataset_dir: Optional[Union[str, Path]] = None,
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0),
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0),
    event_threshold: float = 0.08,
    dt: float = 0.05,
    num_workers: int = 0,
    cache_data: bool = True,
    strict_dataset: bool = True,
) -> Dict[str, DataLoader]:
    """
    Creates train, val, test, and extrapolation DataLoaders with strict trajectory-level partitioning.
    """
    splits = partition_trajectories(total_trajectories, train_split, val_split)
    train_ids = list(range(splits["train"][0], splits["train"][1]))
    val_ids = list(range(splits["val"][0], splits["val"][1]))
    test_ids = list(range(splits["test"][0], splits["test"][1]))
    # For extrapolation evaluation, use the test partition IDs if dataset files are bounded to total_trajectories
    extrap_ids = test_ids[:min(len(test_ids), 100)] if (strict_dataset and dataset_dir is not None) else list(range(total_trajectories, total_trajectories + 100))

    event_sim = EventCameraSimulator(
        height=height,
        width=width,
        event_threshold=event_threshold,
        in_channels=in_channels,
    )

    train_ds = EventWorldDataset(
        trajectory_indices=train_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=standard_velocity_range,
        dt=dt,
        cache_data=cache_data,
        strict_dataset=strict_dataset,
    )
    val_ds = EventWorldDataset(
        trajectory_indices=val_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=standard_velocity_range,
        dt=dt,
        cache_data=cache_data,
        strict_dataset=strict_dataset,
    )
    test_ds = EventWorldDataset(
        trajectory_indices=test_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=standard_velocity_range,
        dt=dt,
        cache_data=cache_data,
        strict_dataset=strict_dataset,
    )
    extrap_ds = EventWorldDataset(
        trajectory_indices=extrap_ids,
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=sequence_length,
        joint_count=joint_count,
        action_dim=action_dim,
        velocity_range=extrapolation_velocity_range,
        dt=dt,
        cache_data=cache_data,
        strict_dataset=strict_dataset,
    )

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_event_batches, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
        "extrapolation": DataLoader(extrap_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches, num_workers=num_workers),
    }
