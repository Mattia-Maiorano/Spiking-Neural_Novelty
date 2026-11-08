"""
Render Robotic Arm Simulation Video CLI

Enforces rendering directly from the Unity simulation output, capturing
100% fidelity realization of what Unity shows (kinematics, materials,
lighting, ground plane, magnetic gripper, payload, HUD telemetry, and camera).
No fallback, no alternative source: directly executes Unity simulation playback.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import imageio.v2 as imageio

DEFAULT_UNITY_BINARY = "/Applications/Unity/Hub/Editor/6000.5.7f1/Unity.app/Contents/MacOS/Unity"


def capture_unity_simulation_frames(
    project_path: Path,
    json_path: Path,
    output_frames_dir: Path,
    unity_bin: str = DEFAULT_UNITY_BINARY,
    width: int = 1280,
    height: int = 720,
    compare_models: bool = False,
) -> List[Path]:
    """
    Executes Unity in headless batchmode to step through the trajectory
    and render camera output directly from the scene.
    """
    if not Path(unity_bin).exists():
        raise FileNotFoundError(
            f"Unity binary not found at '{unity_bin}'. "
            "Unity is strictly required to produce the simulation video with 100% engine fidelity."
        )

    if not json_path.exists():
        raise FileNotFoundError(
            f"Simulation trajectory not found at '{json_path}'. "
            "The video generator script enforces using directly the output of Unity."
        )

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    for f in output_frames_dir.glob("frame_*.png"):
        f.unlink()

    # Ensure StreamingAssets has the requested trajectory file
    streaming_assets_target = project_path / "Assets" / "StreamingAssets" / "trajectory_ghost.json"
    streaming_assets_target.parent.mkdir(parents=True, exist_ok=True)
    if json_path.resolve() != streaming_assets_target.resolve():
        shutil.copy2(json_path, streaming_assets_target)

    cmd = [
        unity_bin,
        "-batchmode",
        "-projectPath",
        str(project_path.resolve()),
        "-executeMethod",
        "SPWM.Editor.GhostVisualizerSceneBuilder.HeadlessRenderVideo",
        "-framesDir",
        str(output_frames_dir.resolve()),
        "-trajectoryJson",
        str(streaming_assets_target.resolve()),
        "-width",
        str(width),
        "-height",
        str(height),
        "-logfile",
        str(output_frames_dir / "unity_render.log"),
    ]
    if compare_models:
        cmd.append("-compareModels")  # Trigger dual-rig generation in C#
    cmd.append("-quit")

    print(f"[Unity Renderer] Launching Unity simulation capture ({width}x{height})...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        log_content = ""
        log_file = output_frames_dir / "unity_render.log"
        if log_file.exists():
            log_content = log_file.read_text(encoding="utf-8", errors="replace")[-1000:]
        raise RuntimeError(
            f"Unity headless rendering failed with returncode {res.returncode}.\n"
            f"Log tail:\n{log_content}"
        )

    frame_files = sorted(output_frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise RuntimeError(
            f"No rendered frames produced in {output_frames_dir}. "
            "Unity output capture failed to record the simulation."
        )

    print(f"[Unity Renderer] Captured {len(frame_files)} frames directly from Unity scene.")
    return frame_files


def encode_frames_to_video(
    frame_files: List[Path],
    output_path: Path,
    fps: int = 20,
) -> None:
    """
    Compiles captured Unity simulation frames directly into MP4 video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Video Encoder] Encoding {len(frame_files)} Unity frames to {output_path} at {fps} FPS...")

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
    )

    for i, frame_path in enumerate(frame_files):
        img = imageio.imread(frame_path)
        writer.append_data(img)
        if (i + 1) % 25 == 0 or i == len(frame_files) - 1:
            print(f"  -> Encoded {i + 1}/{len(frame_files)} frames...")

    writer.close()
    print(f"[Video Encoder] Successfully generated video: {output_path.resolve()} ({output_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render video of the robotic arm simulation directly from Unity output (100% fidelity)."
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="sim/Assets/StreamingAssets/trajectory_ghost.json",
        help="Path to Unity trajectory file (default: sim/Assets/StreamingAssets/trajectory_ghost.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/videos/robot_arm_simulation.mp4",
        help="Path where output MP4 video will be written",
    )
    parser.add_argument(
        "--unity-project",
        type=str,
        default="sim",
        help="Path to Unity project root directory",
    )
    parser.add_argument(
        "--unity-bin",
        type=str,
        default=DEFAULT_UNITY_BINARY,
        help="Path to Unity Editor binary",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video frames per second (default: 20)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width in pixels (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height in pixels (default: 720)")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Max number of frames/horizon steps to render (optional)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Retain raw exported frame PNGs on disk after encoding",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Enable dual-model trajectory overlay with distinct holographic materials",
    )
    args = parser.parse_args()

    project_path = Path(args.unity_project)
    json_path = Path(args.json_file)
    output_path = Path(args.output)

    if not json_path.exists():
        print(f"Error: Trajectory JSON not found at {json_path}", file=sys.stderr)
        sys.exit(1)

    temp_dir_obj = None
    if args.keep_frames:
        frames_dir = output_path.parent / "unity_frames"
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="unity_sim_frames_")
        frames_dir = Path(temp_dir_obj.name)

    try:
        frame_files = capture_unity_simulation_frames(
            project_path=project_path,
            json_path=json_path,
            output_frames_dir=frames_dir,
            unity_bin=args.unity_bin,
            width=args.width,
            height=args.height,
            compare_models=args.compare_models,
        )

        encode_frames_to_video(
            frame_files=frame_files,
            output_path=output_path,
            fps=args.fps,
        )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()

