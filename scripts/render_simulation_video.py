"""
Render Robotic Arm Simulation Video CLI (SPWM-v0.5.0)

Generates a standalone video (.mp4) of the 6-DOF industrial robotic arm simulation
directly from the terminal without launching Unity.

Features:
- Pure simulation playback of ground-truth physical trajectory (no ghost prediction overlays).
- Full 3D kinematic model matching ArticulatedRobotRig (links, rotating revolute joints, gripper, floor).
- Payload tracking (resting on ground during approach, picked up at grasp, hefted and transported).
- Synchronized multi-joint angle dynamics telemetry plot on the side.
- Accepts exported JSON trajectory, simulated synthetic trajectory, or custom trajectory ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
from scipy.spatial.transform import Rotation as R

from spwm.data.datasets import simulate_articulated_trajectory

# Canonical 6-DOF Manipulator link specs matching ArticulatedRobotRig.cs
SPECS = [
    {
        "name": "Base_Root",
        "parent": -1,
        "anchor": np.array([0.0, 0.0, 0.0]),
        "axis": None,
        "dim": np.array([0.30, 0.16, 0.30]),
        "color": "#334155",  # Slate
    },
    {
        "name": "Shoulder_Pan",
        "parent": 0,
        "anchor": np.array([0.0, 0.16, 0.0]),
        "axis": np.array([0.0, 1.0, 0.0]),  # Yaw around Y
        "dim": np.array([0.14, 0.14, 0.14]),
        "color": "#475569",
    },
    {
        "name": "Shoulder_Lift",
        "parent": 1,
        "anchor": np.array([0.0, 0.14, 0.0]),
        "axis": np.array([1.0, 0.0, 0.0]),  # Pitch around X
        "dim": np.array([0.11, 0.30, 0.11]),
        "color": "#2563EB",  # Industrial blue
    },
    {
        "name": "Elbow_Flex",
        "parent": 2,
        "anchor": np.array([0.0, 0.30, 0.0]),
        "axis": np.array([1.0, 0.0, 0.0]),  # Pitch around X
        "dim": np.array([0.09, 0.26, 0.09]),
        "color": "#3B82F6",
    },
    {
        "name": "Wrist_Pitch",
        "parent": 3,
        "anchor": np.array([0.0, 0.26, 0.0]),
        "axis": np.array([1.0, 0.0, 0.0]),  # Pitch around X
        "dim": np.array([0.075, 0.11, 0.075]),
        "color": "#60A5FA",
    },
    {
        "name": "Wrist_Yaw",
        "parent": 4,
        "anchor": np.array([0.0, 0.11, 0.0]),
        "axis": np.array([0.0, 0.0, -1.0]),  # Yaw around Z in local frame
        "dim": np.array([0.06, 0.09, 0.06]),
        "color": "#93C5FD",
    },
    {
        "name": "Wrist_Roll",
        "parent": 5,
        "anchor": np.array([0.0, 0.09, 0.0]),
        "axis": np.array([0.0, 1.0, 0.0]),  # Roll along link axis Y
        "dim": np.array([0.048, 0.08, 0.048]),
        "color": "#CBD5E1",
    },
]

JOINT_NAMES = [
    "Shoulder_Pan",
    "Shoulder_Lift",
    "Elbow_Flex",
    "Wrist_Pitch",
    "Wrist_Yaw",
    "Wrist_Roll",
]


def compute_arm_kinematics(q: np.ndarray) -> Tuple[List[np.ndarray], List[R], np.ndarray]:
    """
    Computes exact forward kinematics for all links and end-effector tip given joint vector q [6].
    """
    positions = []
    rotations = []
    joint_idx = 0

    for s in SPECS:
        p = s["parent"]
        anchor = s["anchor"]
        if p == -1:
            curr_pos = anchor.copy()
            curr_rot = R.identity()
        else:
            parent_pos = positions[p]
            parent_rot = rotations[p]
            curr_pos = parent_pos + parent_rot.apply(anchor)
            q_val = float(q[joint_idx])
            axis = s["axis"]
            j_rot = R.from_rotvec(axis * q_val)
            curr_rot = parent_rot * j_rot
            joint_idx += 1
        positions.append(curr_pos)
        rotations.append(curr_rot)

    tip_pos = positions[-1] + rotations[-1].apply(np.array([0.0, SPECS[-1]["dim"][1], 0.0]))
    return positions, rotations, tip_pos


def build_cylinder_mesh(p0: np.ndarray, p1: np.ndarray, radius: float, n_seg: int = 12) -> Poly3DCollection:
    """Creates a 3D polygonal cylinder mesh between points p0 and p1."""
    v = p1 - p0
    height = np.linalg.norm(v)
    if height < 1e-6:
        v = np.array([0.0, 1.0, 0.0])
        height = 1e-6
    v_norm = v / height

    # Basis perpendicular to v
    not_v = np.array([1.0, 0.0, 0.0]) if abs(v_norm[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    n1 = np.cross(v_norm, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v_norm, n1)

    theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    circle = np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2)

    bottom = p0 + radius * circle
    top = p1 + radius * circle

    faces = []
    for i in range(n_seg):
        nxt = (i + 1) % n_seg
        faces.append([bottom[i], bottom[nxt], top[nxt], top[i]])
    faces.append(bottom)
    faces.append(top)
    return Poly3DCollection(faces)


def render_simulation_video(
    joint_trajectory: np.ndarray,
    times: np.ndarray,
    output_path: Path,
    fps: int = 20,
    grasp_frame: int = 35,
    elevation: float = 24.0,
    azimuth: float = 48.0,
    title: str = "6-DOF Industrial Robotic Arm Simulation",
) -> None:
    """
    Renders high-definition simulation video of the moving robotic arm.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = len(joint_trajectory)

    # Precompute tip positions to locate object at grasp frame
    tip_positions = []
    for f in range(num_frames):
        _, _, tip = compute_arm_kinematics(joint_trajectory[f])
        tip_positions.append(tip)
    tip_positions = np.array(tip_positions)

    # Payload ground position = tip at grasp frame snapped to ground level
    grasp_idx = min(grasp_frame, num_frames - 1)
    payload_ground = tip_positions[grasp_idx].copy()
    payload_ground[1] = 0.04  # half height on ground

    print(f"[Renderer] Rendering {num_frames} frames to {output_path} at {fps} FPS...")

    # Set up dark aesthetic figure matching workstation simulator style
    fig = plt.figure(figsize=(15, 8), facecolor="#0B0F17")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0], height_ratios=[1.0, 1.0])

    ax_3d = fig.add_subplot(gs[:, 0], projection="3d", facecolor="#0B0F17")
    ax_telemetry = fig.add_subplot(gs[0, 1], facecolor="#111827")
    ax_polar = fig.add_subplot(gs[1, 1], facecolor="#111827")

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )

    joint_colors = ["#38BDF8", "#818CF8", "#34D399", "#FBBF24", "#F472B6", "#A78BFA"]

    for frame_idx in range(num_frames):
        q_now = joint_trajectory[frame_idx]
        t_now = times[frame_idx] if frame_idx < len(times) else frame_idx * 0.05
        positions, rotations, tip_pos = compute_arm_kinematics(q_now)

        # ---------------- 3D ROBOT VIEW ----------------
        ax_3d.clear()
        ax_3d.set_facecolor("#0B0F17")

        # Ground plane workcell floor
        floor_x = np.array([-0.6, 0.6, 0.6, -0.6])
        floor_z = np.array([-0.6, -0.6, 0.6, 0.6])
        floor_y = np.array([0.0, 0.0, 0.0, 0.0])
        floor_poly = Poly3DCollection([list(zip(floor_x, floor_y, floor_z))], alpha=0.35, facecolor="#1E293B", edgecolor="#334155", linewidth=1.5)
        ax_3d.add_collection3d(floor_poly)

        # Grid lines on floor
        for grid_val in np.linspace(-0.5, 0.5, 5):
            ax_3d.plot([grid_val, grid_val], [0, 0], [-0.5, 0.5], color="#334155", lw=0.6, alpha=0.5)
            ax_3d.plot([-0.5, 0.5], [0, 0], [grid_val, grid_val], color="#334155", lw=0.6, alpha=0.5)

        # Render each link cylinder
        for i in range(len(SPECS)):
            p_start = positions[i]
            if i == len(SPECS) - 1:
                p_end = tip_pos
            else:
                p_end = positions[i + 1]

            dim = SPECS[i]["dim"]
            radius = dim[0] * 0.5
            col = SPECS[i]["color"]

            # Visual shaft
            cyl = build_cylinder_mesh(p_start, p_end, radius, n_seg=10)
            cyl.set_facecolor(col)
            cyl.set_edgecolor("#0F172A")
            cyl.set_linewidth(0.8)
            cyl.set_alpha(0.95)
            ax_3d.add_collection3d(cyl)

            # Bolted joint spherical housing at pivot
            if i > 0:
                ax_3d.scatter([p_start[0]], [p_start[1]], [p_start[2]], color="#1E293B", s=140, edgecolors="#64748B", linewidths=1.2, depthshade=True)

        # Base Flange
        base_cyl = build_cylinder_mesh(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.02, 0.0]), 0.22, n_seg=12)
        base_cyl.set_facecolor("#1E293B")
        base_cyl.set_edgecolor("#475569")
        base_cyl.set_alpha(0.9)
        ax_3d.add_collection3d(base_cyl)

        # End Effector gripper plate
        gripper_cyl = build_cylinder_mesh(tip_pos - rotations[-1].apply(np.array([0.0, 0.01, 0.0])), tip_pos, 0.045, n_seg=10)
        gripper_cyl.set_facecolor("#F59E0B")
        gripper_cyl.set_edgecolor("#78350F")
        ax_3d.add_collection3d(gripper_cyl)

        # Payload logic (resting on ground or locked to tip)
        if frame_idx < grasp_frame:
            payload_pos = payload_ground
            payload_status = "Approaching (Phase 0/1)"
            status_color = "#94A3B8"
        else:
            payload_pos = tip_pos + np.array([0.0, -0.04, 0.0])
            payload_status = "Grasped & Transporting (Phase 1.5/2)"
            status_color = "#10B981"

        # Payload cylinder
        payload_mesh = build_cylinder_mesh(payload_pos - np.array([0, 0.035, 0]), payload_pos + np.array([0, 0.035, 0]), 0.035, n_seg=10)
        payload_mesh.set_facecolor("#EA580C")
        payload_mesh.set_edgecolor("#7C2D12")
        payload_mesh.set_alpha(0.95)
        ax_3d.add_collection3d(payload_mesh)

        # End-effector trajectory trail
        trail = tip_positions[: frame_idx + 1]
        if len(trail) > 1:
            ax_3d.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#38BDF8", lw=1.8, linestyle="--", alpha=0.85)

        # Camera view bounds & angles
        ax_3d.set_xlim([-0.55, 0.55])
        ax_3d.set_ylim([0.0, 0.85])
        ax_3d.set_zlim([-0.55, 0.55])
        ax_3d.view_init(elev=elevation, azim=azimuth)

        ax_3d.set_xlabel("X (m)", color="#64748B", labelpad=-8)
        ax_3d.set_ylabel("Y (Height m)", color="#64748B", labelpad=-8)
        ax_3d.set_zlabel("Z (m)", color="#64748B", labelpad=-8)
        ax_3d.tick_params(colors="#64748B", labelsize=8)
        ax_3d.set_title(
            f"{title}\nFrame {frame_idx + 1}/{num_frames} · t = {t_now:.2f}s · {payload_status}",
            color="#F8FAFC",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        # ---------------- TELEMETRY 1: JOINT ANGLES ----------------
        ax_telemetry.clear()
        ax_telemetry.set_facecolor("#111827")
        ax_telemetry.grid(True, color="#1F2937", linestyle="--", alpha=0.6)

        t_axis = times[:num_frames] if len(times) >= num_frames else np.arange(num_frames) * 0.05
        for j in range(6):
            ax_telemetry.plot(t_axis, joint_trajectory[:, j], color=joint_colors[j], lw=1.2, alpha=0.35)
            # Active line up to current frame
            ax_telemetry.plot(t_axis[: frame_idx + 1], joint_trajectory[: frame_idx + 1, j], color=joint_colors[j], lw=2.0, label=JOINT_NAMES[j])
            # Current value dot
            ax_telemetry.scatter([t_axis[frame_idx]], [q_now[j]], color=joint_colors[j], s=35, zorder=5)

        ax_telemetry.axvline(x=t_axis[frame_idx], color="#F3F4F6", linestyle=":", lw=1.0, alpha=0.7)
        ax_telemetry.set_title("Joint Kinematics Telemetry (q* rad)", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_telemetry.set_ylabel("Angle (rad)", color="#94A3B8", fontsize=9)
        ax_telemetry.tick_params(colors="#94A3B8", labelsize=8)
        ax_telemetry.set_xlim([t_axis[0], t_axis[-1]])
        ax_telemetry.set_ylim([-2.7, 2.7])
        ax_telemetry.legend(loc="upper right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=7, ncol=2)

        # ---------------- TELEMETRY 2: TIP HEIGHT & VELOCITY ----------------
        ax_polar.clear()
        ax_polar.set_facecolor("#111827")
        ax_polar.grid(True, color="#1F2937", linestyle="--", alpha=0.6)

        # End effector height trajectory
        tip_y = tip_positions[:, 1]
        ax_polar.plot(t_axis, tip_y, color="#34D399", lw=1.2, alpha=0.35)
        ax_polar.plot(t_axis[: frame_idx + 1], tip_y[: frame_idx + 1], color="#34D399", lw=2.2, label="Tip Height Y (m)")
        ax_polar.scatter([t_axis[frame_idx]], [tip_y[frame_idx]], color="#34D399", s=40, zorder=5)

        # Grasp indicator vertical line
        t_grasp = t_axis[grasp_idx]
        ax_polar.axvline(x=t_grasp, color="#F59E0B", linestyle="--", lw=1.2, alpha=0.8, label=f"Grasp Trigger (t={t_grasp:.2f}s)")
        ax_polar.axvline(x=t_axis[frame_idx], color="#F3F4F6", linestyle=":", lw=1.0, alpha=0.7)

        ax_polar.set_title("End-Effector Trajectory & Grasp Phase", color="#F8FAFC", fontsize=10, fontweight="bold")
        ax_polar.set_xlabel("Time (s)", color="#94A3B8", fontsize=9)
        ax_polar.set_ylabel("Height (m)", color="#94A3B8", fontsize=9)
        ax_polar.tick_params(colors="#94A3B8", labelsize=8)
        ax_polar.set_xlim([t_axis[0], t_axis[-1]])
        ax_polar.set_ylim([0.0, 0.75])
        ax_polar.legend(loc="upper right", facecolor="#1F2937", edgecolor="#374151", labelcolor="#F3F4F6", fontsize=7)

        fig.canvas.draw()
        rgba_image = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(rgba_image[:, :, :3])

        if (frame_idx + 1) % 25 == 0 or frame_idx == num_frames - 1:
            print(f"  -> Processed {frame_idx + 1}/{num_frames} frames...")

    writer.close()
    plt.close(fig)
    print(f"[Renderer] Successfully generated video: {output_path.resolve()} ({output_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a video of the robotic arm simulation from the terminal.")
    parser.add_argument(
        "--source",
        type=str,
        default="json",
        choices=["json", "sim"],
        help="Source of trajectory: 'json' (from sim/Assets/StreamingAssets/...) or 'sim' (fresh simulated trajectory)",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="sim/Assets/StreamingAssets/trajectory_ghost_v0_4_0.json",
        help="Path to JSON trajectory file (used when --source json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/videos/robot_arm_simulation.mp4",
        help="Path where output MP4 video will be written",
    )
    parser.add_argument("--trajectory-id", type=int, default=900, help="Trajectory ID for simulation (used when --source sim)")
    parser.add_argument("--horizon", type=int, default=100, help="Number of frames to render")
    parser.add_argument("--fps", type=int, default=20, help="Video frames per second")
    parser.add_argument("--grasp-frame", type=int, default=35, help="Frame at which payload is grasped")
    parser.add_argument("--elevation", type=float, default=24.0, help="3D camera elevation angle in degrees")
    parser.add_argument("--azimuth", type=float, default=48.0, help="3D camera azimuth angle in degrees")
    args = parser.parse_args()

    out_path = Path(args.output)

    if args.source == "json":
        json_path = Path(args.json_file)
        if not json_path.exists():
            print(f"Error: Trajectory JSON not found at {json_path}", file=sys.stderr)
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        if not frames:
            print(f"Error: No frames found in {json_path}", file=sys.stderr)
            sys.exit(1)

        # Ground truth simulation joint angles ONLY (strictly no ghost prediction)
        joint_trajectory = np.array([fr["ground_truth_q"] for fr in frames], dtype=np.float32)
        times = np.array([fr.get("time_s", i * 0.05) for i, fr in enumerate(frames)], dtype=np.float32)
        title = f"6-DOF Robotic Arm Simulation (GT Trajectory ID {data.get('metadata', {}).get('trajectory_id', 'N/A')})"
    else:
        print(f"[Simulator] Synthesizing 6-DOF trajectory (ID={args.trajectory_id}, length={args.horizon})...")
        traj = simulate_articulated_trajectory(trajectory_id=args.trajectory_id, length=args.horizon)
        joint_trajectory = traj.joint_positions
        times = traj.times
        title = f"6-DOF Robotic Arm Simulation (PhysX Pipeline - Trajectory ID {args.trajectory_id})"

    if args.horizon < len(joint_trajectory):
        joint_trajectory = joint_trajectory[: args.horizon]
        times = times[: args.horizon]

    render_simulation_video(
        joint_trajectory=joint_trajectory,
        times=times,
        output_path=out_path,
        fps=args.fps,
        grasp_frame=args.grasp_frame,
        elevation=args.elevation,
        azimuth=args.azimuth,
        title=title,
    )


if __name__ == "__main__":
    main()
