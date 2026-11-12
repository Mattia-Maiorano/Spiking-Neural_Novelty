"""
Unity trajectory serialization and parsing (.json, .json.gz, and .npz)
for 6-DOF Active Perception Grasping telemetry.
"""

from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

from spwm.contracts.tensors import C_EVENT
from spwm.data.event_camera import RoboticTrajectory
from spwm.data.sim_6dof import _compute_default_phase_schedule


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
    pre_contact_sensory_steps: Optional[int] = None,
    post_contact_sensory_steps: Optional[int] = None,
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
            pre_s = pre_contact_sensory_steps if pre_contact_sensory_steps is not None else 7
            post_s = post_contact_sensory_steps if post_contact_sensory_steps is not None else 7
            phase_flags = _compute_default_phase_schedule(
                len(steps),
                pre_contact_sensory_steps=pre_s,
                post_contact_sensory_steps=post_s,
            )[3]

        if pre_contact_sensory_steps is not None or post_contact_sensory_steps is not None:
            pre_s = pre_contact_sensory_steps if pre_contact_sensory_steps is not None else 7
            post_s = post_contact_sensory_steps if post_contact_sensory_steps is not None else 7
            expected_p1_len = pre_s + post_s
            p1_indices = np.where(phase_flags == 1)[0]
            if len(p1_indices) > 0 and len(p1_indices) != expected_p1_len:
                t_old_start = int(p1_indices[0])
                # In legacy exports, t_old_start was t_contact - 5 (duration was 17)
                t_contact = t_old_start + 5 if len(p1_indices) == 17 else int(p1_indices[0])
                t_inj_start = max(1, t_contact - pre_s)
                t_inj_end = min(len(steps) - 1, t_contact + post_s)
                phase_flags = np.zeros(len(steps), dtype=np.int64)
                phase_flags[t_inj_start:t_inj_end] = 1
                phase_flags[t_inj_end:] = 2

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

