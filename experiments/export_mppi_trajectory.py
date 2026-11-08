"""
Latent MPPI Trajectory Exporter for Unity Simulation Video (SPWM-v0.7.3).

Executes closed-loop Latent MPPI control on a specified test trajectory and exports
the resulting joint angles, velocities, and instantaneous reaching metrics into
the Unity Ghost Trajectory format (`spwm.ghost_trajectory/1.0`).

Zero AST Fallbacks strictly enforced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import chronicle
import numpy as np
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.mppi import LatentMPPIController
from spwm.evaluation.visualizer import (
    DEFAULT_GHOST_FILENAME,
    DEFAULT_UNITY_PROJECT_DIR,
    GHOST_SCHEMA_VERSION,
    GhostTrajectory,
)
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import ground_initial_state, step_grounded
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
) -> GhostTrajectory:
    events = batch["events"]
    q_gt = batch["positions"][0]
    p_gt = batch["velocities"][0]
    actions = batch["actions"]
    pf = batch["phase_flags"][0]
    traj_id = int(batch["trajectory_ids"][0].item())

    p1_indices = (pf == 1).nonzero()
    t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
    target_pos_t = fk(q_gt[t_c : t_c + 1])[0].detach()

    # Match benchmark_mppi_control approach window: ground state up to t_start
    t_start = max(5, t_c - approach_window)
    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=t_start, actions=actions)

    dec_start = model.physical_decoder(z_start)
    initial_bias = dec_start[0, :6] / 3.14
    controller.reset(initial_action_bias=initial_bias)
    prev_u = initial_bias.clone()

    predicted_q_list: List[np.ndarray] = []
    predicted_p_list: List[np.ndarray] = []
    ee_errors_cm: List[float] = []

    # Before t_start, state is tracking ground truth
    for t in range(t_start):
        predicted_q_list.append(q_gt[t].detach().cpu().numpy())
        predicted_p_list.append(p_gt[t].detach().cpu().numpy())
        ee_pos = fk(q_gt[t : t + 1])[0].detach()
        err_cm = float((torch.norm(ee_pos - target_pos_t) * 100.0).item())
        ee_errors_cm.append(err_cm)

    # From t_start to horizon: closed-loop execution
    for t in range(t_start, horizon):
        z_t, dyn_state = step_grounded(
            model,
            events[:, t],
            dyn_state,
            in_sensory_phase=True,
            unconstrained=True,
            action=prev_u.unsqueeze(0),
        )

        # Latent MPPI mental unroll planning up to grasp; hold control action after grasp
        if t <= t_grasp:
            res = controller.plan_step(z_t, target_pos_t, prev_action=prev_u)
            prev_u = res.action

        dec_t = model.physical_decoder(z_t)
        q_pred_t = dec_t[0, :6].detach().cpu().numpy()
        p_pred_t = dec_t[0, 6:12].detach().cpu().numpy()
        predicted_q_list.append(q_pred_t)
        predicted_p_list.append(p_pred_t)

        ee_t = fk(dec_t[:, :6])[0].detach()
        err_cm = float((torch.norm(ee_t - target_pos_t) * 100.0).item())
        ee_errors_cm.append(err_cm)

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
    parser.add_argument("--horizon", type=int, default=100, help="Horizon steps")
    parser.add_argument("--num-samples", type=int, default=256, help="MPPI candidate rollouts K")
    parser.add_argument("--mppi-horizon", type=int, default=20, help="MPPI planning horizon H")
    parser.add_argument("--num-iterations", type=int, default=3, help="MPPI iterations")
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
    # Construct the anisotropic noise vector from base, shoulder, and elbow
    s_base = 0.1226
    s_shoulder = 0.5894
    s_elbow = 0.2172
    anisotropic_sigmas = [s_base, s_shoulder, s_elbow, s_base, s_base, s_base]

    controller = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        forward_kinematics=fk,
        num_samples=args.num_samples,
        horizon=args.horizon,
        num_iterations=args.num_iterations,
        noise_sigma=anisotropic_sigmas,
        noise_beta=0.3778,
        temperature=0.0410,
        w_pos=1.0,
        w_rot=2.4498,
        w_term_pos=5.7514,
        w_term_rot=2.4498 * 4.0,  # 9.799
        w_vel=0.00231,
        w_act=0.00224,
        device=device,
    )


    loaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=max(cfg.data.sequence_length, args.horizon),
        batch_size=1,
        height=cfg.environment.height,
        width=cfg.environment.width,
        joint_count=cfg.environment.joint_count,
        action_dim=cfg.environment.action_dim,
        in_channels=cfg.model.in_channels,
        event_threshold=cfg.environment.event_threshold,
        dt=cfg.environment.dt,
        dataset_dir=cfg.data.dataset_dir,
        strict_dataset=cfg.data.strict_dataset,
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

    chronicle.log_info(f"Exporting MPPI trajectory for trajectory_id={args.trajectory_id}...")
    ghost = export_mppi_episode(
        model=model,
        controller=controller,
        fk=fk,
        batch=target_batch,
        horizon=args.horizon,
        model_version=f"{GLOBAL_VERSION.snake_case}_mppi",
        dt=cfg.environment.dt,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ghost.to_dict(), f, separators=(",", ":"), allow_nan=False)

    chronicle.log_success(f"MPPI Ghost Trajectory JSON exported successfully to: {out_path}")


if __name__ == "__main__":
    main()
