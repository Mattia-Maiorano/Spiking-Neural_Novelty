"""
Forensic Failure-Mode Decomposition for Continuous RSSM and Deep GRU in PIL.

Inspects all 50 PIL episodes ([3600, 3649]) for Continuous RSSM (Seeds 42, 101, 2024),
Deep GRU (Seed 42), ANN-Rate (Seed 42), and SPWM (Seed 42), decomposing failures into:
  (a) Distance Failure: min_t ||x_ee(t) - x_target||_2 > 3.0 cm
  (b) Velocity Stability Failure: min_t <= 3.0 cm AND ||q_dot(t_grasp)||_2 >= tau_vel
  (c) Post-Contact Impulse Divergence: min_{t <= t_c} <= 3.0 cm, then terminal_err > 3.0 cm
      after rigid payload inertia attaches at t >= t_c and ev_online updates state.

Also outputs step-by-step error trajectories around t_c and t_grasp.
Zero AST fallbacks strictly enforced.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spwm.data.datasets import EventWorldDataset
from spwm.evaluation.mppi_scorer import resolve_effective_vel_threshold


def load_gt_episode_metadata(
    dataset_dir: Path,
    traj_ids: List[int],
    approach_window: int = 20,
) -> Dict[int, Dict[str, Any]]:
    """Extract exact t_c, t_start, t_grasp, and effective velocity threshold from GT dataset."""
    ds = EventWorldDataset(traj_ids, dataset_dir=dataset_dir)
    meta: Dict[int, Dict[str, Any]] = {}
    for idx, traj_id in enumerate(traj_ids):
        sample = ds[idx]
        phase_flags = sample["phase_flags"]
        q_gt = sample["joint_positions"]
        p_gt = sample["joint_velocities"]

        p1_indices = (phase_flags == 1).nonzero()
        t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
        t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
        t_start = max(5, t_c - approach_window)
        gt_final_vel = float(torch.norm(p_gt[t_grasp, :6]).item())
        eff_vel_thresh = resolve_effective_vel_threshold(1.5, gt_final_vel)

        meta[traj_id] = {
            "t_c": t_c,
            "t_start": t_start,
            "t_grasp": t_grasp,
            "idx_tc": t_c - t_start,
            "gt_final_vel": gt_final_vel,
            "effective_vel_threshold": eff_vel_thresh,
        }
    return meta


def analyze_model_pil_log(
    label: str,
    json_path: Path,
    gt_meta: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Decompose all 50 episodes into exact failure modes and step-by-step error around t_c."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    episodes = data["episodes"]
    n_eps = len(episodes)

    n_success = 0
    # Failure mode flags (not mutually exclusive, plus mutually exclusive primary decomposition)
    n_dist_fail = 0                 # (a) min_t err > 3.0 cm
    n_vel_fail_given_reach = 0      # (b) min_t err <= 3.0 cm AND final_vel >= tau_vel
    n_vel_fail_all = 0              # final_vel >= tau_vel across all episodes
    n_settling_fail_given_reach = 0 # min_t err <= 3.0 cm AND not all(last 3 < 3.0 cm)
    n_post_contact_div = 0          # (c) min_{t <= t_c} err <= 3.0 cm AND terminal_err > 3.0 cm
    n_post_contact_div_only_after = 0  # reached <= 3.0 only after t_c then diverged
    n_rot_fail_only = 0             # spatially & kinematically stable, failed orientation

    # Mutually exclusive decomposition of failed episodes:
    # 1. Never reached <= 3.0 cm (Distance Failure)
    # 2. Reached <= 3.0 cm (at or before t_c) -> Post-Contact Impulse Divergence (settling failed, term > 3.0 cm)
    #    - Sub-case: also violated velocity threshold
    # 3. Reached <= 3.0 cm only after t_c -> still failed settling
    # 4. Passed settling window (< 3.0 cm for last 3 steps), failed Velocity Stability ONLY
    # 5. Passed settling & velocity, failed Orientation ONLY
    excl_never_reached = 0
    excl_post_contact_div_both_vel = 0
    excl_post_contact_div_spatial_only = 0
    excl_post_tc_reach_then_div = 0
    excl_vel_only = 0
    excl_rot_only = 0

    offsets = list(range(-3, 8))  # t_c - 3 .. t_c + 7 (= t_grasp)
    err_by_offset: Dict[int, List[float]] = {off: [] for off in offsets}

    min_errs: List[float] = []
    pre_tc_min_errs: List[float] = []
    tc_errs: List[float] = []
    term_errs: List[float] = []
    final_vels: List[float] = []
    vel_thresholds: List[float] = []
    accels: List[float] = []
    jerks: List[float] = []
    flips: List[int] = []
    err_jump_tc_to_grasp: List[float] = []

    for ep in episodes:
        tid = int(ep["trajectory_id"])
        m = gt_meta[tid]
        idx_tc = m["idx_tc"]
        tau_vel = m["effective_vel_threshold"]

        step_errs = [float(x) for x in ep["step_errors_cm"]]
        min_err = float(ep["min_placement_error_cm"])
        term_err = float(ep["terminal_placement_error_cm"])
        final_vel = float(ep["terminal_angular_velocity"])
        grasp_err = float(ep["grasp_error_cm"])
        is_succ = bool(ep["grasp_success"])

        pre_tc_slice = step_errs[: idx_tc + 1]
        min_pre_tc = min(pre_tc_slice)

        min_errs.append(min_err)
        pre_tc_min_errs.append(min_pre_tc)
        tc_errs.append(grasp_err)
        term_errs.append(term_err)
        final_vels.append(final_vel)
        vel_thresholds.append(tau_vel)
        accels.append(float(ep["mean_joint_accel_rad_s2"]))
        jerks.append(float(ep["mean_joint_jerk_rad_s3"]))
        flips.append(int(ep["direction_flips"]))
        err_jump_tc_to_grasp.append(term_err - grasp_err)

        for off in offsets:
            pos = idx_tc + off
            if 0 <= pos < len(step_errs):
                err_by_offset[off].append(step_errs[pos])

        if is_succ:
            n_success += 1
            continue

        window_errs = step_errs[-3:] if len(step_errs) >= 3 else step_errs
        is_spatially_stable = all(e < 3.0 for e in window_errs)
        is_kinematically_stable = final_vel < tau_vel

        if not is_kinematically_stable:
            n_vel_fail_all += 1

        if min_err > 3.0:
            n_dist_fail += 1
            excl_never_reached += 1
        else:
            if not is_kinematically_stable:
                n_vel_fail_given_reach += 1
            if not is_spatially_stable:
                n_settling_fail_given_reach += 1
            if min_pre_tc <= 3.0 and term_err > 3.0:
                n_post_contact_div += 1
                if not is_kinematically_stable:
                    excl_post_contact_div_both_vel += 1
                else:
                    excl_post_contact_div_spatial_only += 1
            elif not is_spatially_stable:
                n_post_contact_div_only_after += 1
                excl_post_tc_reach_then_div += 1
            elif not is_kinematically_stable:
                excl_vel_only += 1
            else:
                n_rot_fail_only += 1
                excl_rot_only += 1

    n_fail = n_eps - n_success
    step_profile = {
        f"t_c{off:+d}" if off != 0 else "t_c": {
            "mean_cm": float(np.mean(err_by_offset[off])),
            "std_cm": float(np.std(err_by_offset[off])),
            "median_cm": float(np.median(err_by_offset[off])),
            "n": len(err_by_offset[off]),
        }
        for off in offsets
    }

    return {
        "label": label,
        "n_episodes": n_eps,
        "n_success": n_success,
        "n_fail": n_fail,
        "success_rate_pct": 100.0 * n_success / n_eps,
        "failure_modes_non_exclusive": {
            "a_distance_failure_gt_3cm": {"count": n_dist_fail, "pct": 100.0 * n_dist_fail / n_eps},
            "b_velocity_failure_given_reach_le_3cm": {"count": n_vel_fail_given_reach, "pct": 100.0 * n_vel_fail_given_reach / n_eps},
            "b_velocity_failure_all_episodes": {"count": n_vel_fail_all, "pct": 100.0 * n_vel_fail_all / n_eps},
            "c_post_contact_impulse_divergence": {"count": n_post_contact_div, "pct": 100.0 * n_post_contact_div / n_eps},
            "settling_window_failure_given_reach_le_3cm": {"count": n_settling_fail_given_reach, "pct": 100.0 * n_settling_fail_given_reach / n_eps},
            "reached_only_after_tc_then_diverged": {"count": n_post_contact_div_only_after, "pct": 100.0 * n_post_contact_div_only_after / n_eps},
        },
        "failure_modes_mutually_exclusive": {
            "1_distance_never_reached_3cm": {"count": excl_never_reached, "pct": 100.0 * excl_never_reached / n_eps},
            "2a_post_contact_div_plus_vel_violation": {"count": excl_post_contact_div_both_vel, "pct": 100.0 * excl_post_contact_div_both_vel / n_eps},
            "2b_post_contact_div_spatial_only": {"count": excl_post_contact_div_spatial_only, "pct": 100.0 * excl_post_contact_div_spatial_only / n_eps},
            "3_reached_post_tc_then_diverged": {"count": excl_post_tc_reach_then_div, "pct": 100.0 * excl_post_tc_reach_then_div / n_eps},
            "4_settling_passed_vel_failed_only": {"count": excl_vel_only, "pct": 100.0 * excl_vel_only / n_eps},
            "5_settling_and_vel_passed_rot_failed_only": {"count": excl_rot_only, "pct": 100.0 * excl_rot_only / n_eps},
        },
        "summary_stats": {
            "min_err_cm_mean": float(np.mean(min_errs)),
            "min_err_cm_std": float(np.std(min_errs)),
            "pre_tc_min_err_cm_mean": float(np.mean(pre_tc_min_errs)),
            "tc_err_cm_mean": float(np.mean(tc_errs)),
            "tc_err_cm_std": float(np.std(tc_errs)),
            "term_err_cm_mean": float(np.mean(term_errs)),
            "term_err_cm_std": float(np.std(term_errs)),
            "err_jump_tc_to_grasp_cm_mean": float(np.mean(err_jump_tc_to_grasp)),
            "final_vel_mean": float(np.mean(final_vels)),
            "final_vel_std": float(np.std(final_vels)),
            "effective_vel_threshold_mean": float(np.mean(vel_thresholds)),
            "mean_joint_accel_rad_s2": float(np.mean(accels)),
            "mean_joint_jerk_rad_s3": float(np.mean(jerks)),
            "mean_direction_flips": float(np.mean(flips)),
        },
        "step_error_profile_around_tc": step_profile,
    }


def main() -> None:
    dataset_dir = REPO_ROOT / "sim/ExportedTrajectories"
    traj_ids = list(range(3600, 3650))
    gt_meta = load_gt_episode_metadata(dataset_dir, traj_ids)

    targets = [
        ("Continuous RSSM Un-Gated (Seed 42)", REPO_ROOT / "results/mppi_benchmark/physics_in_loop/rssm_ungated/mppi_benchmark_metrics.json"),
        ("Continuous RSSM Un-Gated (Seed 101)", REPO_ROOT / "results/multiseed_rssm/seed_101/mppi_pil_ungated/mppi_benchmark_metrics.json"),
        ("Continuous RSSM Un-Gated (Seed 2024)", REPO_ROOT / "results/multiseed_rssm/seed_2024/mppi_pil_ungated/mppi_benchmark_metrics.json"),
        ("Continuous RSSM Residual-Gated (Seed 42)", REPO_ROOT / "results/mppi_benchmark/physics_in_loop/rssm_residual_gated/mppi_benchmark_metrics.json"),
        ("Continuous RSSM Residual-Gated (Seed 101)", REPO_ROOT / "results/multiseed_rssm/seed_101/mppi_pil_residual_gated/mppi_benchmark_metrics.json"),
        ("Continuous RSSM Residual-Gated (Seed 2024)", REPO_ROOT / "results/multiseed_rssm/seed_2024/mppi_pil_residual_gated/mppi_benchmark_metrics.json"),
        ("Deep GRU Un-Gated (Seed 42)", REPO_ROOT / "results/mppi_benchmark/physics_in_loop/deep_gru_ungated/mppi_benchmark_metrics.json"),
        ("Deep GRU Un-Gated (Seed 101)", REPO_ROOT / "results/multiseed_rssm/seed_101/deep_gru_mppi_pil_ungated/mppi_benchmark_metrics.json"),
        ("Deep GRU Un-Gated (Seed 2024)", REPO_ROOT / "results/multiseed_rssm/seed_2024/deep_gru_mppi_pil_ungated/mppi_benchmark_metrics.json"),
        ("Deep GRU Residual-Gated (Seed 42)", REPO_ROOT / "results/mppi_benchmark/physics_in_loop/deep_gru_residual_gated/mppi_benchmark_metrics.json"),
        ("Deep GRU Residual-Gated (Seed 101)", REPO_ROOT / "results/multiseed_rssm/seed_101/deep_gru_mppi_pil_residual_gated/mppi_benchmark_metrics.json"),
        ("Deep GRU Residual-Gated (Seed 2024)", REPO_ROOT / "results/multiseed_rssm/seed_2024/deep_gru_mppi_pil_residual_gated/mppi_benchmark_metrics.json"),
        ("ANN-Rate (Seed 42)", REPO_ROOT / "results/ann_ablation/seed_42_mppi_pil/mppi_benchmark_metrics.json"),
        ("ANN-Rate (Seed 101)", REPO_ROOT / "results/ann_ablation/seed_101_mppi_pil/mppi_benchmark_metrics.json"),
        ("ANN-Rate (Seed 2024)", REPO_ROOT / "results/ann_ablation/seed_2024_mppi_pil/mppi_benchmark_metrics.json"),
        ("SPWM N=25 Val-Sel (Seed 42)", REPO_ROOT / "results/ablations/multiseed/stage4b_full_seed42/mppi_benchmark_metrics.json"),
        ("SPWM N=25 Val-Sel (Seed 101)", REPO_ROOT / "results/ablations/multiseed/stage4b_full_seed101/mppi_benchmark_metrics.json"),
        ("SPWM N=25 Val-Sel (Seed 2024)", REPO_ROOT / "results/ablations/multiseed/stage4b_full_seed2024/mppi_benchmark_metrics.json"),
    ]

    report: Dict[str, Any] = {}
    for label, path in targets:
        if not path.is_file():
            raise FileNotFoundError(f"Missing PIL metrics JSON: {path}")
        res = analyze_model_pil_log(label, path, gt_meta)
        report[label] = res
        ne = res["failure_modes_non_exclusive"]
        ex = res["failure_modes_mutually_exclusive"]
        st = res["summary_stats"]
        print(f"\n=== {label} (Success: {res['n_success']}/{res['n_episodes']} = {res['success_rate_pct']:.1f}%) ===")
        print(f"  Min Err: {st['min_err_cm_mean']:.2f} cm | Err@t_c: {st['tc_err_cm_mean']:.2f} cm | Term Err@t_grasp: {st['term_err_cm_mean']:.2f} cm (Delta: {st['err_jump_tc_to_grasp_cm_mean']:+.2f} cm)")
        print(f"  Final Vel: {st['final_vel_mean']:.2f} rad/s (tau_vel mean: {st['effective_vel_threshold_mean']:.2f}) | |d2q|: {st['mean_joint_accel_rad_s2']:.5f} | Flips: {st['mean_direction_flips']:.2f}")
        print(f"  Non-excl: (a) Dist>3cm={ne['a_distance_failure_gt_3cm']['count']}/50 ({ne['a_distance_failure_gt_3cm']['pct']:.1f}%), "
              f"(b) VelFail|reach={ne['b_velocity_failure_given_reach_le_3cm']['count']}/50 ({ne['b_velocity_failure_given_reach_le_3cm']['pct']:.1f}%), "
              f"(c) PostContactDiv={ne['c_post_contact_impulse_divergence']['count']}/50 ({ne['c_post_contact_impulse_divergence']['pct']:.1f}%)")
        print(f"  Mutually-excl fails: never_reached={ex['1_distance_never_reached_3cm']['count']}, "
              f"post_tc_div+vel={ex['2a_post_contact_div_plus_vel_violation']['count']}, "
              f"post_tc_div_spatial_only={ex['2b_post_contact_div_spatial_only']['count']}, "
              f"post_tc_reach_then_div={ex['3_reached_post_tc_then_div' if '3_reached_post_tc_then_div' in ex else '3_reached_post_tc_then_diverged']['count']}, "
              f"vel_only={ex['4_settling_passed_vel_failed_only']['count']}, "
              f"rot_only={ex['5_settling_and_vel_passed_rot_failed_only']['count']}")
        prof = res["step_error_profile_around_tc"]
        prof_str = " -> ".join(f"{k}:{prof[k]['mean_cm']:.2f}" for k in ["t_c-2", "t_c-1", "t_c", "t_c+1", "t_c+2", "t_c+4", "t_c+7"])
        print(f"  Step profile (cm): {prof_str}")

    # Aggregate 3-seed summaries across all equalized models & SPWM/ANN-Rate
    agg_groups = {
        "Continuous RSSM Un-Gated (3-Seed Aggregate)": [
            "Continuous RSSM Un-Gated (Seed 42)",
            "Continuous RSSM Un-Gated (Seed 101)",
            "Continuous RSSM Un-Gated (Seed 2024)",
        ],
        "Continuous RSSM Residual-Gated (3-Seed Aggregate)": [
            "Continuous RSSM Residual-Gated (Seed 42)",
            "Continuous RSSM Residual-Gated (Seed 101)",
            "Continuous RSSM Residual-Gated (Seed 2024)",
        ],
        "Deep GRU Un-Gated (3-Seed Aggregate)": [
            "Deep GRU Un-Gated (Seed 42)",
            "Deep GRU Un-Gated (Seed 101)",
            "Deep GRU Un-Gated (Seed 2024)",
        ],
        "Deep GRU Residual-Gated (3-Seed Aggregate)": [
            "Deep GRU Residual-Gated (Seed 42)",
            "Deep GRU Residual-Gated (Seed 101)",
            "Deep GRU Residual-Gated (Seed 2024)",
        ],
        "ANN-Rate (3-Seed Aggregate)": [
            "ANN-Rate (Seed 42)",
            "ANN-Rate (Seed 101)",
            "ANN-Rate (Seed 2024)",
        ],
        "SPWM N=25 Val-Sel (3-Seed Aggregate)": [
            "SPWM N=25 Val-Sel (Seed 42)",
            "SPWM N=25 Val-Sel (Seed 101)",
            "SPWM N=25 Val-Sel (Seed 2024)",
        ],
    }
    for agg_name, keys in agg_groups.items():
        report[agg_name] = {
            "total_episodes": 150,
            "success_total": sum(report[k]["n_success"] for k in keys),
            "reach_le_3cm_total": 150 - sum(report[k]["failure_modes_non_exclusive"]["a_distance_failure_gt_3cm"]["count"] for k in keys),
            "a_distance_failure_gt_3cm_total": sum(report[k]["failure_modes_non_exclusive"]["a_distance_failure_gt_3cm"]["count"] for k in keys),
            "b_velocity_failure_all_total": sum(report[k]["failure_modes_non_exclusive"]["b_velocity_failure_all_episodes"]["count"] for k in keys),
            "b_velocity_failure_given_reach_total": sum(report[k]["failure_modes_non_exclusive"]["b_velocity_failure_given_reach_le_3cm"]["count"] for k in keys),
            "c_post_contact_impulse_divergence_total": sum(report[k]["failure_modes_non_exclusive"]["c_post_contact_impulse_divergence"]["count"] for k in keys),
            "excl_never_reached_total": sum(report[k]["failure_modes_mutually_exclusive"]["1_distance_never_reached_3cm"]["count"] for k in keys),
            "excl_post_contact_div_plus_vel_total": sum(report[k]["failure_modes_mutually_exclusive"]["2a_post_contact_div_plus_vel_violation"]["count"] for k in keys),
            "excl_post_contact_div_spatial_only_total": sum(report[k]["failure_modes_mutually_exclusive"]["2b_post_contact_div_spatial_only"]["count"] for k in keys),
            "excl_post_tc_reach_then_div_total": sum(report[k]["failure_modes_mutually_exclusive"]["3_reached_post_tc_then_diverged"]["count"] for k in keys),
            "excl_vel_only_total": sum(report[k]["failure_modes_mutually_exclusive"]["4_settling_passed_vel_failed_only"]["count"] for k in keys),
            "excl_rot_only_total": sum(report[k]["failure_modes_mutually_exclusive"]["5_settling_and_vel_passed_rot_failed_only"]["count"] for k in keys),
        }

    out_path = REPO_ROOT / "results/diagnostics/forensic_rssm_gru_failure.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved forensic failure report to {out_path}")


if __name__ == "__main__":
    main()
