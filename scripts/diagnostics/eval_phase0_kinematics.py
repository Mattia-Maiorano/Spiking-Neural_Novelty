"""
Phase 0 Grounded Kinematic & Cartesian End-Effector Diagnostic Evaluator.

Evaluates Phase 0 (`train_representation`) checkpoints on the critical benchmark
trajectories (Traj 3200, Traj 3204, Traj 3211):
  1. Loads the Phase 0 checkpoint in `eval()` mode and extracts the latent sequence
     via `model.forward_representation(events, actions=actions)`.
  2. Decodes joint positions with `model.physical_decoder.pos_head(z_seq[:, :, :32])`.
  3. Computes homogeneous Forward Kinematics (`fk.forward_homogeneous`) on the final
     frame T_end for both decoded predictions and Ground Truth.
  4. Reports terminal Euclidean Cartesian error ||p_dec - p_gt||_2 (cm), geodesic
     SO(3) rotational error theta_err (rad and deg), and per-joint absolute errors
     |q_hat_i - q*_i| (rad and deg) for all 6 joints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import EventWorldDataset
from spwm.data.event_camera import EventCameraSimulator
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.world_model import SPWM


JOINT_NAMES: Tuple[str, ...] = (
    "J0 (Shoulder_Pan)",
    "J1 (Shoulder_Lift)",
    "J2 (Elbow_Flex)",
    "J3 (Wrist_Pitch)",
    "J4 (Wrist_Yaw)",
    "J5 (Wrist_Roll)",
)


def evaluate_phase0_checkpoint(
    checkpoint_path: Path,
    config_path: Optional[Path] = None,
    trajectory_ids: Sequence[int] = (3200, 3204, 3211),
    dataset_dir: str = "sim/ExportedTrajectories",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluates a Phase 0 representation checkpoint on benchmark trajectories."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Phase 0 checkpoint not found at: {checkpoint_path}")

    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if config_path is not None and config_path.is_file():
        cfg = ExperimentConfig.load(config_path)
    elif isinstance(ckpt, dict) and "config" in ckpt and isinstance(ckpt["config"], dict):
        cfg = ExperimentConfig.from_dict(ckpt["config"])
    else:
        cfg = ExperimentConfig()

    model = SPWM.from_config(cfg).to(device)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fk = DifferentiableForwardKinematics(joint_count=cfg.environment.joint_count).to(device)
    event_sim = EventCameraSimulator(
        height=cfg.environment.height,
        width=cfg.environment.width,
        event_threshold=cfg.environment.event_threshold,
        in_channels=cfg.model.in_channels,
    )
    dataset = EventWorldDataset(
        trajectory_indices=list(trajectory_ids),
        dataset_dir=dataset_dir,
        event_simulator=event_sim,
        sequence_length=cfg.data.sequence_length,
        joint_count=cfg.environment.joint_count,
        action_dim=cfg.environment.action_dim,
        velocity_range=cfg.data.standard_velocity_range,
        dt=cfg.environment.dt,
        cache_data=False,
        strict_dataset=True,
    )

    results_per_traj: Dict[int, Dict[str, Any]] = {}
    q_dim = cfg.model.q_dim

    with torch.no_grad():
        for idx, traj_id in enumerate(trajectory_ids):
            item = dataset[idx]
            events = item["events"].unsqueeze(0).to(device)          # [1, T, C, H, W]
            actions = item["actions"].unsqueeze(0).to(device)        # [1, T, D_u]
            q_gt_seq = item["joint_positions"].unsqueeze(0).to(device)  # [1, T, 6]

            # 1. Extract latent sequence via forward_representation
            z_seq, _ = model.forward_representation(events, actions=actions)

            # 2. Decode joint positions with physical_decoder.pos_head(z_seq[:, :, :32])
            q_dec_seq = model.physical_decoder.pos_head(z_seq[:, :, :q_dim])  # [1, T, 6]

            # 3. Extract terminal step T_end (index -1) and compute homogeneous FK
            q_dec_end = q_dec_seq[:, -1, :]  # [1, 6]
            q_gt_end = q_gt_seq[:, -1, :]    # [1, 6]

            T_dec_end = fk.forward_homogeneous(q_dec_end)[0]  # [4, 4]
            T_gt_end = fk.forward_homogeneous(q_gt_end)[0]    # [4, 4]

            p_dec = T_dec_end[:3, 3]
            p_gt = T_gt_end[:3, 3]
            R_dec = T_dec_end[:3, :3]
            R_gt = T_gt_end[:3, :3]

            # 4a. Euclidean Cartesian error ||p_dec - p_gt||_2 (in cm)
            cart_err_m = float(torch.norm(p_dec - p_gt, p=2).item())
            cart_err_cm = cart_err_m * 100.0

            # 4b. Geodesic SO(3) rotational error: arccos((Tr(R_dec @ R_gt^T) - 1) / 2)
            r_rel = torch.matmul(R_dec, R_gt.transpose(-1, -2))
            trace_val = float(torch.diagonal(r_rel, dim1=-2, dim2=-1).sum().item())
            cos_theta = max(-1.0, min(1.0, (trace_val - 1.0) * 0.5))
            rot_err_rad = math.acos(cos_theta)
            rot_err_deg = math.degrees(rot_err_rad)

            # 4c. Per-joint absolute error |q_hat_i - q*_i| (in rad and deg)
            joint_abs_err_rad = (q_dec_end[0] - q_gt_end[0]).abs().cpu().tolist()
            joint_abs_err_deg = [math.degrees(val) for val in joint_abs_err_rad]

            # Sequence-wide mean metrics for additional context
            T_dec_all = fk.forward_homogeneous(q_dec_seq)[0]
            T_gt_all = fk.forward_homogeneous(q_gt_seq)[0]
            seq_cart_err_cm = float(torch.norm(T_dec_all[:, :3, 3] - T_gt_all[:, :3, 3], p=2, dim=-1).mean().item() * 100.0)
            seq_j2_err_rad = float((q_dec_seq[0, :, 2] - q_gt_seq[0, :, 2]).abs().mean().item())

            results_per_traj[int(traj_id)] = {
                "trajectory_id": int(traj_id),
                "cartesian_err_cm": cart_err_cm,
                "rot_err_rad": rot_err_rad,
                "rot_err_deg": rot_err_deg,
                "joint_err_rad": joint_abs_err_rad,
                "joint_err_deg": joint_abs_err_deg,
                "q_pred_end": q_dec_end[0].cpu().tolist(),
                "q_gt_end": q_gt_end[0].cpu().tolist(),
                "p_dec_end": p_dec.cpu().tolist(),
                "p_gt_end": p_gt.cpu().tolist(),
                "seq_mean_cartesian_err_cm": seq_cart_err_cm,
                "seq_mean_j2_err_rad": seq_j2_err_rad,
            }

    max_j2_err_rad = max(r["joint_err_rad"][2] for r in results_per_traj.values())
    max_cart_err_cm = max(r["cartesian_err_cm"] for r in results_per_traj.values())
    mean_j2_err_rad = sum(r["joint_err_rad"][2] for r in results_per_traj.values()) / len(results_per_traj)
    mean_cart_err_cm = sum(r["cartesian_err_cm"] for r in results_per_traj.values()) / len(results_per_traj)
    mean_rot_err_rad = sum(r["rot_err_rad"] for r in results_per_traj.values()) / len(results_per_traj)

    passed_j2 = max_j2_err_rad < 0.035
    passed_cart = max_cart_err_cm < 1.2

    return {
        "checkpoint": str(checkpoint_path),
        "trajectories": results_per_traj,
        "summary": {
            "mean_j2_err_rad": mean_j2_err_rad,
            "mean_j2_err_deg": math.degrees(mean_j2_err_rad),
            "max_j2_err_rad": max_j2_err_rad,
            "max_j2_err_deg": math.degrees(max_j2_err_rad),
            "mean_cartesian_err_cm": mean_cart_err_cm,
            "max_cartesian_err_cm": max_cart_err_cm,
            "mean_rot_err_rad": mean_rot_err_rad,
            "mean_rot_err_deg": math.degrees(mean_rot_err_rad),
            "acceptance_j2_lt_0_035rad": passed_j2,
            "acceptance_cart_lt_1_2cm": passed_cart,
            "all_criteria_passed": bool(passed_j2 and passed_cart),
        },
    }


def print_diagnostic_report(report: Dict[str, Any], title: str = "PHASE 0 KINEMATIC DIAGNOSTIC") -> None:
    """Formats and prints the Phase 0 kinematic diagnostic report to stdout."""
    print("=" * 88)
    print(f"  {title}")
    print(f"  Checkpoint: {report['checkpoint']}")
    print("=" * 88)

    for tid, data in report["trajectories"].items():
        print(f"\n--- Trajectory {tid} (Terminal Frame T_end) ---")
        print(
            f"  Cartesian Error ||p_dec - p_gt||_2 : {data['cartesian_err_cm']:6.3f} cm  "
            f"(Seq Mean: {data['seq_mean_cartesian_err_cm']:.3f} cm)"
        )
        print(
            f"  Geodesic SO(3) Error theta_err     : {data['rot_err_rad']:6.4f} rad "
            f"({data['rot_err_deg']:5.2f} deg)"
        )
        print("  Per-Joint Absolute Errors |q_hat_i - q*_i|:")
        for j in range(6):
            r_err = data["joint_err_rad"][j]
            d_err = data["joint_err_deg"][j]
            q_p = data["q_pred_end"][j]
            q_g = data["q_gt_end"][j]
            flag = " <-- [CRITICAL J2]" if j == 2 else ""
            print(
                f"    {JOINT_NAMES[j]:<20}: {r_err:6.4f} rad ({d_err:5.2f} deg)  "
                f"[pred={q_p:+7.4f}, gt={q_g:+7.4f}]{flag}"
            )

    s = report["summary"]
    print("\n" + "-" * 88)
    print("  SUMMARY & ACCEPTANCE CRITERIA:")
    print(
        f"    Max J2 Error (Target < 0.035 rad / 2.00 deg) : "
        f"{s['max_j2_err_rad']:.4f} rad ({s['max_j2_err_deg']:.2f} deg) -> "
        f"{'PASS' if s['acceptance_j2_lt_0_035rad'] else 'FAIL'}"
    )
    print(
        f"    Max Cartesian Error (Target < 1.20 cm)       : "
        f"{s['max_cartesian_err_cm']:.3f} cm (Mean: {s['mean_cartesian_err_cm']:.3f} cm) -> "
        f"{'PASS' if s['acceptance_cart_lt_1_2cm'] else 'FAIL'}"
    )
    print(
        f"    Mean Geodesic SO(3) Rotation Error           : "
        f"{s['mean_rot_err_rad']:.4f} rad ({s['mean_rot_err_deg']:.2f} deg)"
    )
    print("=" * 88 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 0 kinematic and Cartesian end-effector accuracy.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(GLOBAL_VERSION.representation_checkpoint()),
        help="Path to Phase 0 model.pt checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config.yaml.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Optional path to baseline Phase 0 checkpoint (e.g. v0.11.0) for before/after comparison.",
    )
    parser.add_argument(
        "--trajectories",
        type=int,
        nargs="+",
        default=[3200, 3204, 3211],
        help="Benchmark trajectory IDs to evaluate (default: 3200 3204 3211).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="sim/ExportedTrajectories",
        help="Path to exported Unity trajectories.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (cpu, cuda, mps).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save diagnostic report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config is not None else None

    if args.compare_baseline is not None:
        base_path = Path(args.compare_baseline)
        base_report = evaluate_phase0_checkpoint(
            checkpoint_path=base_path,
            config_path=None,
            trajectory_ids=args.trajectories,
            dataset_dir=args.dataset_dir,
            device_str=args.device,
        )
        print_diagnostic_report(base_report, title="BASELINE PHASE 0 KINEMATIC DIAGNOSTIC (BEFORE)")

    report = evaluate_phase0_checkpoint(
        checkpoint_path=ckpt_path,
        config_path=cfg_path,
        trajectory_ids=args.trajectories,
        dataset_dir=args.dataset_dir,
        device_str=args.device,
    )
    print_diagnostic_report(report, title="PHASE 0 KINEMATIC DIAGNOSTIC")

    if args.output_json is not None:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

