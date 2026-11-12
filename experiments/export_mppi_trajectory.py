"""
Latent MPPI Trajectory Exporter for Unity Simulation Video.

Executes closed-loop Latent MPPI control on a specified test trajectory and exports
the resulting joint angles, velocities, and instantaneous reaching metrics into
the Unity Ghost Trajectory format (`spwm.ghost_trajectory/1.0`).

Zero AST Fallbacks strictly enforced.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import chronicle
import numpy as np
import torch

from spwm.contracts.config import ExperimentConfig, MPPIConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders_from_config
from spwm.evaluation.mppi import LatentMPPIController
from spwm.evaluation.visualizer import (
    DEFAULT_GHOST_FILENAME,
    DEFAULT_UNITY_PROJECT_DIR,
    GHOST_SCHEMA_VERSION,
    GhostTrajectory,
)
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import ground_initial_state
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed

MANIPULATOR_JOINT_NAMES = [
    "Shoulder_Pan", "Shoulder_Lift", "Elbow_Flex",
    "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll",
]


def _finite(val: float, precision: int = 6) -> float:
    f = float(val)
    if np.isnan(f) or np.isinf(f):
        raise ValueError(f"Ghost trajectory value is non-finite: {f}")
    return round(f, precision)


def _vec(arr: Sequence[float], precision: int = 6) -> List[float]:
    return [_finite(x, precision) for x in arr]


def export_mppi_episode(
    model: SPWM,
    controller: LatentMPPIController,
    fk: DifferentiableForwardKinematics,
    batch: Mapping[str, Any],
    horizon: int = 100,
    model_version: Optional[str] = None,
    dt: float = 0.05,
    precision: int = 6,
    approach_window: int = 20,
    seed: Optional[int] = None,
) -> GhostTrajectory:
    device = controller.device
    events = batch["events"].to(device)
    q_gt_full = (batch["joint_positions"] if "joint_positions" in batch else batch["positions"])[0].to(device)
    p_gt_full = (batch["joint_velocities"] if "joint_velocities" in batch else batch["velocities"])[0].to(device)
    actions = batch["actions"].to(device)
    pf = batch["phase_flags"][0].to(device)
    traj_id = int(batch["trajectory_ids"][0].item())

    if horizon < 2:
        raise ValueError(f"horizon must be >= 2, got {horizon}")
    if events.shape[1] < horizon or q_gt_full.shape[0] < horizon or p_gt_full.shape[0] < horizon:
        raise ValueError(
            f"Trajectory {traj_id} sequence length ({events.shape[1]}) is shorter than requested horizon ({horizon})."
        )

    q_gt = q_gt_full[:horizon]
    p_gt = p_gt_full[:horizon]
    max_t = horizon - 1

    p1_indices = (pf == 1).nonzero()
    t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
    t_grasp = min(t_c + 7, q_gt_full.shape[0] - 1)
    t_c_idx = min(t_c, max_t)
    t_grasp_idx = min(t_grasp, max_t)
    t_start = min(max(5, t_c - approach_window), max_t - 1)

    # Precompute grounded latent sequence and reachable 4x4 SE(3) target poses across the full horizon
    with torch.no_grad():
        _, _, z_gt_seq = ground_initial_state(model, events, t_start=max_t, actions=actions)
        q_reachable_seq = model.physical_decoder(z_gt_seq.squeeze(0))[..., :6]
        target_pose_seq = fk.forward_homogeneous(q_reachable_seq).detach()

    z_start = z_gt_seq[:, t_start]
    dec_start = model.physical_decoder(z_start)
    initial_bias = dec_start[0, :6] / 3.14
    controller.reset(initial_action_bias=initial_bias, seed=seed)
    prev_u = initial_bias.clone()

    predicted_q_list: List[np.ndarray] = []
    predicted_p_list: List[np.ndarray] = []
    ee_errors_cm: List[float] = []

    target_pos_c = target_pose_seq[t_c_idx, :3, 3]

    # Before t_start, state is tracking ground truth
    for t in range(t_start):
        predicted_q_list.append(q_gt[t].detach().cpu().numpy())
        predicted_p_list.append(p_gt[t].detach().cpu().numpy())
        ee_pos = fk.forward_homogeneous(q_gt[t : t + 1])[0, :3, 3].detach()
        err_cm = float((torch.norm(ee_pos - target_pos_c) * 100.0).item())
        ee_errors_cm.append(err_cm)

    # Emit grounded initial state at t_start
    predicted_q_list.append(dec_start[0, :6].detach().cpu().numpy())
    predicted_p_list.append(dec_start[0, 6:12].detach().cpu().numpy())
    ee_start = fk.forward_homogeneous(dec_start[:, :6])[0, :3, 3].detach()
    ee_errors_cm.append(float((torch.norm(ee_start - target_pos_c) * 100.0).item()))

    # From t_start to horizon - 1: full-horizon closed-loop execution via M1 + Phase-1 active-perception blending
    curr_z = z_start.clone()
    current_chunk: Optional[torch.Tensor] = None
    chunk_idx = 0

    for t in range(t_start, max_t):
        if t <= t_c_idx:
            target_pose_mat = target_pose_seq[t_c_idx]
        elif t <= t_grasp_idx:
            target_pose_mat = target_pose_seq[t_grasp_idx]
        else:
            target_pose_mat = target_pose_seq[min(t + 1, max_t)]

        # Invalidate action chunk at contact and post-grasp phase transitions
        if t == t_c_idx + 1 or t == t_grasp_idx + 1:
            current_chunk = None
            chunk_idx = 0

        if current_chunk is None or chunk_idx >= controller.chunk_size:
            res = controller.plan_step(curr_z, target_pose_mat, prev_action=prev_u)
            current_chunk = res.action_chunk
            chunk_idx = 0

        u_exec = current_chunk[chunk_idx]
        chunk_idx += 1
        prev_u = u_exec
        controller.prev_action = u_exec.clone()

        z_in = curr_z if curr_z.dim() == 2 else curr_z.unsqueeze(0)
        u_in = u_exec if u_exec.dim() == 2 else u_exec.unsqueeze(0)
        z_gt_next = z_gt_seq[:, t + 1]
        with torch.no_grad():
            pred_out = model.predictor(z_in, u_in)
            z_pred = pred_out.predicted_latent
            if int(pf[t].item()) == 1:
                z_next = 0.40 * z_pred + 0.60 * z_gt_next
            else:
                z_next = z_pred

        dec_next = model.physical_decoder(z_next)
        predicted_q_list.append(dec_next[0, :6].detach().cpu().numpy())
        predicted_p_list.append(dec_next[0, 6:12].detach().cpu().numpy())

        ee_next = fk.forward_homogeneous(dec_next[:, :6])[0, :3, 3].detach()
        err_cm = float((torch.norm(ee_next - target_pose_mat[:3, 3]) * 100.0).item())
        ee_errors_cm.append(err_cm)

        curr_z = z_next

    # Compute metrics and serialize frames
    frames: List[Dict[str, Any]] = []
    pos_errs = []
    vel_errs = []

    for t in range(horizon):
        q_g = q_gt[t].detach().cpu().numpy()
        p_g = p_gt[t].detach().cpu().numpy()
        q_p = predicted_q_list[t]
        p_p = predicted_p_list[t]

        pos_e = float(np.linalg.norm(q_p - q_g))
        vel_e = float(np.linalg.norm(p_p - p_g))
        pos_errs.append(pos_e)
        vel_errs.append(vel_e)

        pos_rel = pos_e / (pos_errs[0] if pos_errs[0] > 1e-8 else 1.0)
        vel_rel = vel_e / (vel_errs[0] if vel_errs[0] > 1e-8 else 1.0)

        frames.append({
            "t": t + 1,
            "time_s": _finite((t + 1) * dt, precision),
            "ground_truth_q": _vec(q_g, precision),
            "predicted_q": _vec(q_p, precision),
            "ground_truth_p": _vec(p_g, precision),
            "predicted_p": _vec(p_p, precision),
            "instantaneous_metrics": {
                "latent_error": _finite(ee_errors_cm[t] / 100.0, precision), # Cartesian reaching error in meters
                "position_abs_error": _finite(pos_e, precision),
                "velocity_abs_error": _finite(vel_e, precision),
                "velocity_relative_degradation": _finite(vel_rel, precision),
                "position_relative_degradation": _finite(pos_rel, precision),
            },
        })

    summary = {
        "mean_latent_error": _finite(float(np.mean(ee_errors_cm)) / 100.0, precision),
        "final_latent_error": _finite(ee_errors_cm[-1] / 100.0, precision),
        "max_latent_error": _finite(float(np.max(ee_errors_cm)) / 100.0, precision),
        "mean_position_abs_error": _finite(float(np.mean(pos_errs)), precision),
        "final_position_abs_error": _finite(pos_errs[-1], precision),
        "max_position_abs_error": _finite(float(np.max(pos_errs)), precision),
        "mean_velocity_abs_error": _finite(float(np.mean(vel_errs)), precision),
        "final_velocity_abs_error": _finite(vel_errs[-1], precision),
        "max_velocity_abs_error": _finite(float(np.max(vel_errs)), precision),
        "final_velocity_relative_degradation": _finite(vel_errs[-1] / (vel_errs[0] if vel_errs[0] > 1e-8 else 1.0), precision),
        "final_position_relative_degradation": _finite(pos_errs[-1] / (pos_errs[0] if pos_errs[0] > 1e-8 else 1.0), precision),
    }

    resolved_version = model_version if model_version is not None else f"{GLOBAL_VERSION.snake_case}_mppi"
    metadata = {
        "schema": GHOST_SCHEMA_VERSION,
        "model_version": resolved_version,
        "horizon": horizon,
        "dt": dt,
        "joint_count": 6,
        "joint_names": MANIPULATOR_JOINT_NAMES,
        "angle_unit": "rad",
        "has_velocities": True,
        "t_start": t_start,
        "sample_index": 0,
        "trajectory_id": traj_id,
        "batch_index": 0,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "has_sensory_injection": True,
        "heft_start_frame_index": t_c,
        "grasp_lock_frame_index": t_grasp,
        "sensory_release_frame_index": min(t_grasp + 3, horizon - 1),
    }

    return GhostTrajectory(metadata=metadata, frames=frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SPWM MPPI ghost trajectory for Unity playback.")
    parser.add_argument("--checkpoint", type=str, default=str(GLOBAL_VERSION.m1_checkpoint()), help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=str(GLOBAL_VERSION.m1_config()), help="Path to config")
    parser.add_argument("--trajectory-id", type=int, default=3622, help="Trajectory ID to evaluate (e.g., 3622)")
    parser.add_argument("--horizon", type=int, default=100, help="Simulation export horizon steps")
    parser.add_argument("--num-samples", type=int, default=None, help="Override cfg.mppi.num_samples (K)")
    parser.add_argument("--mppi-horizon", type=int, default=None, help="Override cfg.mppi.horizon (H)")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override cfg.mppi.num_iterations")
    parser.add_argument("--model-version", type=str, default=None, help="Metadata model_version string")
    parser.add_argument("--output", type=str, default="sim/Assets/StreamingAssets/trajectory_ghost_mppi.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device (cpu, cuda, mps)")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.project.seed)
    device = torch.device(args.device)

    chronicle.log_info(f"Loading checkpoint strictly: {args.checkpoint} on {device}")
    model = SPWM.from_config(cfg).to(device)
    ckpt_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt_payload["model_state"] if isinstance(ckpt_payload, dict) and "model_state" in ckpt_payload else ckpt_payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fk = DifferentiableForwardKinematics().to(device)
    mppi_cfg = replace(
        cfg.mppi,
        num_samples=args.num_samples if args.num_samples is not None else cfg.mppi.num_samples,
        horizon=args.mppi_horizon if args.mppi_horizon is not None else cfg.mppi.horizon,
        num_iterations=args.num_iterations if args.num_iterations is not None else cfg.mppi.num_iterations,
    )

    controller = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        config=mppi_cfg,
        forward_kinematics=fk,
        critic=model.critic,
        device=device,
    )

    loaders = create_dataloaders_from_config(
        cfg,
        sequence_length=max(cfg.data.sequence_length, args.horizon),
        batch_size=1,
        cache_data=False,
    )

    target_batch = None
    for batch in loaders["test"]:
        tid = int(batch["trajectory_ids"][0].item())
        if tid == args.trajectory_id:
            target_batch = batch
            break

    if target_batch is None:
        raise ValueError(f"Trajectory ID {args.trajectory_id} not found in test partition")

    resolved_version = args.model_version if args.model_version is not None else f"{cfg.project.name}_mppi"
    chronicle.log_info(
        f"Exporting MPPI trajectory for trajectory_id={args.trajectory_id} "
        f"(K={mppi_cfg.num_samples}, H={mppi_cfg.horizon}, iters={mppi_cfg.num_iterations}, "
        f"version={resolved_version})..."
    )
    ghost = export_mppi_episode(
        model=model,
        controller=controller,
        fk=fk,
        batch=target_batch,
        horizon=args.horizon,
        model_version=resolved_version,
        dt=cfg.environment.dt,
        seed=cfg.project.seed + args.trajectory_id,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ghost.to_dict(), f, separators=(",", ":"), allow_nan=False)

    chronicle.log_success(f"MPPI Ghost Trajectory JSON exported successfully to: {out_path}")


if __name__ == "__main__":
    main()
