"""
Metrics Resume Generator and Schema for SPWM.
Produces fixed-granularity summary tables (metrics_v*.json) mimicking visualization plots:
1. Training progression downsampled to 10 segments (with initial, final, best, and pct changes).
2. Multi-step autonomous rollout evaluation breakdown across horizons.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from spwm.evaluation.evaluator import RolloutEvaluationResult


def get_canonical_metrics_filename(version_tag: str) -> str:
    """
    Return clean canonical filename metrics_<tag>.json avoiding duplicate 'v' prefixes.
    For example:
        'spwm_v0_1_0' -> 'metrics_v0_1_0.json'
        'v0_1_0'      -> 'metrics_v0_1_0.json'
        '0_1_0'       -> 'metrics_v0_1_0.json'
    """
    clean_tag = version_tag[5:] if version_tag.startswith("spwm_") else version_tag
    clean_tag = clean_tag if clean_tag.startswith("v") else f"v{clean_tag}"
    return f"metrics_{clean_tag}.json"


def generate_metrics_resume(
    history: Optional[List[Dict[str, Any]]] = None,
    evaluation_result: Optional[RolloutEvaluationResult] = None,
    parameters: Optional[int] = None,
    num_segments: int = 10,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    evaluation_result_no_corrector: Optional[RolloutEvaluationResult] = None,
    pos_span: Optional[float] = None,
    vel_span: Optional[float] = None,
    mppi_evaluation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a fixed-granularity data table resume mimicking the figure plots.
    
    Args:
        history: Epoch-by-epoch training logs.
        evaluation_result: RolloutEvaluationResult from test/eval set.
        parameters: Total parameter count of the model.
        num_segments: Number of segments to split the epoch range into (default: 10).
        model_name: Identifier of the experiment/model.
        version: Version string for the metrics resume.
        evaluation_result_no_corrector: Optional RolloutEvaluationResult evaluated without corrector.
        pos_span: Optional total physical position coordinate span for relative normalization.
        vel_span: Optional total physical velocity span for relative normalization.
        mppi_evaluation_result: Optional closed-loop MPPI evaluation metrics dictionary.
    """
    total_epochs = len(history) if history else 0
    resume: Dict[str, Any] = {
        "version": version or "v0_1_0",
        "model_name": model_name or "SPWM",
        "total_epochs": total_epochs,
        "parameters": parameters,
    }

    def _eval_result_to_dict(eval_res: RolloutEvaluationResult) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "teacher_forcing_mse": float(eval_res.teacher_forcing_mse),
            "mean_spike_rate": float(eval_res.mean_spike_rate),
            "rollout_mae": float(eval_res.rollout_mae),
            "rollout_score": float(eval_res.dynamic_score),
            "dynamic_score": float(eval_res.dynamic_score),
            "drift_ratio": float(eval_res.drift_ratio),
            "horizons": list(eval_res.horizons),
            "latent_mse_per_horizon": {str(k): float(v) for k, v in eval_res.latent_mse_per_horizon.items()},
            "position_error_per_horizon": {str(k): float(v) for k, v in eval_res.position_error_per_horizon.items()},
            "velocity_error_per_horizon": {str(k): float(v) for k, v in eval_res.velocity_error_per_horizon.items()},
        }
        if eval_res.position_mse_per_horizon is not None:
            d["position_mse_per_horizon"] = {str(k): float(v) for k, v in eval_res.position_mse_per_horizon.items()}
        if eval_res.velocity_mse_per_horizon is not None:
            d["velocity_mse_per_horizon"] = {str(k): float(v) for k, v in eval_res.velocity_mse_per_horizon.items()}
        if eval_res.q_mse_per_horizon is not None:
            d["q_mse_per_horizon"] = {str(k): float(v) for k, v in eval_res.q_mse_per_horizon.items()}
        if eval_res.p_mse_per_horizon is not None:
            d["p_mse_per_horizon"] = {str(k): float(v) for k, v in eval_res.p_mse_per_horizon.items()}

        # Physical rollout degradation (normalized / relative degradation matching visualizer curves)
        h_vals = sorted(list(eval_res.horizons))
        if h_vals:
            # Physical degradation
            pos_span_denom = pos_span if (pos_span is not None and pos_span > 0) else None
            vel_span_denom = vel_span if (vel_span is not None and vel_span > 0) else None

            pos_h1 = eval_res.position_error_per_horizon.get(h_vals[0], 1.0)
            pos_h1_denom = pos_h1 if (pos_h1 > 0) else 1.0
            vel_h1 = eval_res.velocity_error_per_horizon.get(h_vals[0], 1.0)
            vel_h1_denom = vel_h1 if (vel_h1 > 0) else 1.0

            pos_norm_denom = pos_span_denom if pos_span_denom is not None else pos_h1_denom
            vel_norm_denom = vel_span_denom if vel_span_denom is not None else vel_h1_denom

            d["physical_rollout_degradation"] = {
                "position_relative": {
                    str(h): float(eval_res.position_error_per_horizon.get(h, 0.0) / pos_h1_denom)
                    for h in h_vals
                },
                "velocity_relative": {
                    str(h): float(eval_res.velocity_error_per_horizon.get(h, 0.0) / vel_h1_denom)
                    for h in h_vals
                },
                "position_normalized": {
                    str(h): float(eval_res.position_error_per_horizon.get(h, 0.0) / pos_norm_denom)
                    for h in h_vals
                },
                "velocity_normalized": {
                    str(h): float(eval_res.velocity_error_per_horizon.get(h, 0.0) / vel_norm_denom)
                    for h in h_vals
                },
            }

            # Latent rollout degradation
            lat_h1 = eval_res.latent_mse_per_horizon.get(h_vals[0], 1.0)
            lat_h1_denom = lat_h1 if (lat_h1 > 0) else 1.0

            latent_deg: Dict[str, Any] = {
                "latent_relative": {
                    str(h): float(eval_res.latent_mse_per_horizon.get(h, 0.0) / lat_h1_denom)
                    for h in h_vals
                }
            }

            if eval_res.q_mse_per_horizon is not None:
                q_h1 = eval_res.q_mse_per_horizon.get(h_vals[0], 1.0)
                q_h1_denom = q_h1 if (q_h1 > 0) else 1.0
                latent_deg["q_relative"] = {
                    str(h): float(eval_res.q_mse_per_horizon.get(h, 0.0) / q_h1_denom)
                    for h in h_vals
                }

            if eval_res.p_mse_per_horizon is not None:
                p_h1 = eval_res.p_mse_per_horizon.get(h_vals[0], 1.0)
                p_h1_denom = p_h1 if (p_h1 > 0) else 1.0
                latent_deg["p_relative"] = {
                    str(h): float(eval_res.p_mse_per_horizon.get(h, 0.0) / p_h1_denom)
                    for h in h_vals
                }

            d["latent_rollout_degradation"] = latent_deg

        return d

    if evaluation_result is not None:
        resume["evaluation"] = _eval_result_to_dict(evaluation_result)

    if evaluation_result_no_corrector is not None:
        resume["evaluation_no_corrector"] = _eval_result_to_dict(evaluation_result_no_corrector)

    if mppi_evaluation_result is not None:
        resume["mppi_evaluation"] = mppi_evaluation_result

    if history and total_epochs > 0:
        epochs = [r.get("epoch", i + 1) for i, r in enumerate(history)]

        # Sample indices into fixed segments (e.g. 10 segments -> 10 checkpoints)
        if total_epochs <= num_segments:
            sample_indices = list(range(total_epochs))
        else:
            sample_indices = [
                round(i * (total_epochs - 1) / (num_segments - 1))
                for i in range(num_segments)
            ]

        # Extract primary metrics tracked across epochs and figures
        metrics_keys = [
            ("train_loss", "train_loss"),
            ("val_loss", "val_loss"),
            ("train_pos_err", "train_pos_err"),
            ("val_pos_err", "val_pos_err"),
            ("train_vel_err", "train_vel_err"),
            ("val_vel_err", "val_vel_err"),
            ("val_drift_ratio", "val_drift_ratio"),
            ("score", "score"),
            ("mppi_score", "mppi_score"),
            ("mppi_cost", "mppi_cost"),
            ("mppi_success_rate", "mppi_success_rate"),
            ("mppi_mean_error_cm", "mppi_mean_error_cm"),
            ("mppi_grasp_error_cm", "mppi_grasp_error_cm"),
            ("mppi_grasp_rot_err_rad", "mppi_grasp_rot_err_rad"),
            ("mppi_terminal_error_cm", "mppi_terminal_error_cm"),
            ("mppi_steady_state_error_cm", "mppi_steady_state_error_cm"),
            ("mppi_min_error_cm", "mppi_min_error_cm"),
            ("mppi_rot_err_rad", "mppi_rot_err_rad"),
            ("mppi_final_vel", "mppi_final_vel"),
            ("mppi_final_rot", "mppi_final_rot"),
            ("mppi_pos_penalty", "mppi_pos_penalty"),
            ("mppi_vel_penalty", "mppi_vel_penalty"),
            ("mppi_rot_penalty", "mppi_rot_penalty"),
            ("train_mppi_noise_alpha", "train_mppi_noise_alpha"),
            ("mppi_expert_trajectories", "mppi_expert_trajectories"),
        ]

        # Filter keys actually present in records
        available_keys = [k for k, _ in metrics_keys if any(k in r for r in history)]

        sampled_table: List[Dict[str, Any]] = []
        for segment_idx, idx in enumerate(sample_indices):
            rec = history[idx]
            ep = rec.get("epoch", idx + 1)
            pct = round((segment_idx / max(1, len(sample_indices) - 1)) * 100.0, 1)
            entry = {
                "segment": segment_idx + 1,
                "epoch": ep,
                "progress_pct": f"{pct}%",
            }
            for k in available_keys:
                if k in rec and rec[k] is not None:
                    val = rec[k]
                    if isinstance(val, (int, float, np.number)):
                        entry[k] = round(float(val), 6) if np.isfinite(val) else None
                    else:
                        entry[k] = val

            if "mppi_score" in rec and "mppi_expert_trajectories" in rec:
                n_exp = int(float(rec.get("mppi_expert_trajectories", 0.0)))
                if "mppi_expert_blocks" in rec and isinstance(rec["mppi_expert_blocks"], list):
                    entry["mppi_expert_blocks"] = rec["mppi_expert_blocks"]
                elif n_exp > 0:
                    entry["mppi_expert_blocks"] = ["Unknown"] * n_exp
                else:
                    entry["mppi_expert_blocks"] = []

            sampled_table.append(entry)

        # Compute summary stats (initial, final, best, pct_change)
        higher_is_better_keys = {"mppi_success_rate", "success_rate", "accuracy"}

        def compute_stat(key: str) -> Optional[Dict[str, Any]]:
            vals = [
                (epochs[i], float(r[key]))
                for i, r in enumerate(history)
                if key in r and r[key] is not None and isinstance(r[key], (int, float, np.number)) and np.isfinite(r[key])
            ]
            if not vals:
                return None
            initial_val = vals[0][1]
            final_val = vals[-1][1]
            if key in higher_is_better_keys:
                best_ep, best_val = max(vals, key=lambda x: x[1])
            else:
                best_ep, best_val = min(vals, key=lambda x: x[1])
            change = final_val - initial_val
            pct_change = (change / abs(initial_val) * 100.0) if initial_val != 0 else 0.0
            return {
                "initial": round(float(initial_val), 6),
                "final": round(float(final_val), 6),
                "best": round(float(best_val), 6),
                "best_epoch": int(best_ep),
                "change": round(float(change), 6),
                "pct_change": f"{pct_change:+.2f}%",
            }

        trends = {k: compute_stat(k) for k in available_keys}
        trends = {k: v for k, v in trends.items() if v is not None}

        # Best selection score & epoch
        scores = [(epochs[i], r["score"]) for i, r in enumerate(history) if "score" in r and r["score"] is not None and np.isfinite(r["score"])]
        best_score_info = None
        if scores:
            best_score_ep, best_score_val = min(scores, key=lambda x: x[1])
            best_score_info = {
                "epoch": int(best_score_ep),
                "score": round(float(best_score_val), 6),
            }

        resume["training_progression"] = {
            "total_epochs": total_epochs,
            "num_segments": len(sample_indices),
            "best_checkpoint": best_score_info,
            "trends": trends,
            "sampled_segments": sampled_table,
        }

        # Dedicated MPPI Closed-Loop Progression Summary
        mppi_entries = [
            r for r in history
            if ("mppi_score" in r and r["mppi_score"] is not None)
            or ("mppi_success_rate" in r and r["mppi_success_rate"] is not None)
        ]
        if mppi_entries:
            last_mppi = mppi_entries[-1]
            mppi_scores = [
                (r.get("epoch", i + 1), r["mppi_score"])
                for i, r in enumerate(mppi_entries)
                if "mppi_score" in r and r["mppi_score"] is not None and np.isfinite(r["mppi_score"])
            ]
            if mppi_scores:
                best_mppi_ep, best_mppi_score = min(mppi_scores, key=lambda x: x[1])
            else:
                best_mppi_ep = int(last_mppi.get("epoch", 0))
                best_mppi_score = float(last_mppi.get("score", 0.0))

            final_succ_raw = float(last_mppi["mppi_success_rate"]) if "mppi_success_rate" in last_mppi and last_mppi["mppi_success_rate"] is not None else 0.0
            final_succ_pct = round(final_succ_raw * 100.0 if final_succ_raw <= 1.0 else final_succ_raw, 2)

            final_term_err = float(last_mppi["mppi_terminal_error_cm"]) if "mppi_terminal_error_cm" in last_mppi and last_mppi["mppi_terminal_error_cm"] is not None else (
                float(last_mppi["mppi_mean_error_cm"]) if "mppi_mean_error_cm" in last_mppi and last_mppi["mppi_mean_error_cm"] is not None else float("inf")
            )
            final_rot = float(last_mppi["mppi_rot_err_rad"]) if "mppi_rot_err_rad" in last_mppi and last_mppi["mppi_rot_err_rad"] is not None else (
                float(last_mppi["mppi_final_rot"]) if "mppi_final_rot" in last_mppi and last_mppi["mppi_final_rot"] is not None else float("inf")
            )
            final_vel = float(last_mppi["mppi_final_vel"]) if "mppi_final_vel" in last_mppi and last_mppi["mppi_final_vel"] is not None else float("inf")

            succ_vals = [
                float(r["mppi_success_rate"])
                for r in mppi_entries
                if "mppi_success_rate" in r and r["mppi_success_rate"] is not None and np.isfinite(r["mppi_success_rate"])
            ]
            best_succ_raw = max(succ_vals) if succ_vals else 0.0
            best_succ_pct = round(best_succ_raw * 100.0 if best_succ_raw <= 1.0 else best_succ_raw, 2)

            all_known_blocks: List[str] = []
            untracked_expert_count = 0
            total_expert_count = 0
            valid_elevation_planes = {"Low", "Mid", "High"}
            for r in mppi_entries:
                n_exp = int(float(r.get("mppi_expert_trajectories", 0.0)))
                total_expert_count += n_exp
                blocks = r.get("mppi_expert_blocks")
                if isinstance(blocks, list) and blocks:
                    all_known_blocks.extend([b for b in blocks if b in valid_elevation_planes])
                    untracked_expert_count += sum(1 for b in blocks if b not in valid_elevation_planes)
                elif n_exp > 0:
                    untracked_expert_count += n_exp

            init_expert = int(float(mppi_entries[0].get("mppi_expert_trajectories", 0.0)))
            final_expert = int(float(last_mppi.get("mppi_expert_trajectories", 0.0)))

            expert_blocks_final = [
                b for b in last_mppi.get("mppi_expert_blocks", []) if b in valid_elevation_planes
            ] if isinstance(last_mppi.get("mppi_expert_blocks"), list) else []

            tracked_counts = {
                "Low": all_known_blocks.count("Low"),
                "Mid": all_known_blocks.count("Mid"),
                "High": all_known_blocks.count("High"),
            }

            expert_summary: Dict[str, Any] = {
                "initial_expert_trajectories": init_expert,
                "final_expert_trajectories": final_expert,
                "curriculum_annealed_to_autonomous": bool(final_expert == 0),
                "total_overwritten_trajectories": total_expert_count,
                "tracked_elevation_counts": tracked_counts,
                "final_natural_trajectory_ids": last_mppi.get("mppi_natural_trajectory_ids", []),
                "final_natural_blocks": last_mppi.get("mppi_natural_blocks", []),
                "final_expert_trajectory_ids": last_mppi.get("mppi_expert_trajectory_ids", []),
            }
            if untracked_expert_count > 0:
                expert_summary["unannotated_legacy_overwritten_trajectories"] = untracked_expert_count
            if expert_blocks_final:
                expert_summary["final_overwritten_blocks"] = expert_blocks_final
            if "mppi_elevation_breakdown" in last_mppi and isinstance(last_mppi["mppi_elevation_breakdown"], dict):
                clean_final_breakdown = {
                    k: v for k, v in last_mppi["mppi_elevation_breakdown"].items()
                    if any(k.endswith(f"_{p.lower()}") for p in ("low", "mid", "high"))
                }
                if clean_final_breakdown:
                    expert_summary["final_elevation_breakdown"] = clean_final_breakdown
            if "mppi_trials" in last_mppi and isinstance(last_mppi["mppi_trials"], list):
                expert_summary["final_trials"] = last_mppi["mppi_trials"]

            # Per-epoch evaluation history with trial-by-trial and elevation value breakdowns
            evaluations_history: List[Dict[str, Any]] = []
            for r in mppi_entries:
                succ_r = float(r["mppi_success_rate"]) if "mppi_success_rate" in r and r["mppi_success_rate"] is not None else 0.0
                succ_p = round(succ_r * 100.0 if succ_r <= 1.0 else succ_r, 2)
                hist_item: Dict[str, Any] = {
                    "epoch": int(r.get("epoch", 0)),
                    "mppi_score": round(float(r["mppi_score"]), 6) if "mppi_score" in r and r["mppi_score"] is not None else None,
                    "success_rate_pct": succ_p,
                    "mean_cartesian_error_cm": round(float(r["mppi_mean_error_cm"]), 3) if "mppi_mean_error_cm" in r and r["mppi_mean_error_cm"] is not None else None,
                    "grasp_error_cm": round(float(r["mppi_grasp_error_cm"]), 3) if "mppi_grasp_error_cm" in r and r["mppi_grasp_error_cm"] is not None else None,
                    "terminal_error_cm": round(float(r["mppi_terminal_error_cm"]), 3) if "mppi_terminal_error_cm" in r and r["mppi_terminal_error_cm"] is not None else None,
                    "natural_trajectories_count": int(float(r.get("mppi_evaluated_trajectories", 0.0))),
                    "natural_blocks": r.get("mppi_natural_blocks", []),
                    "natural_trajectory_ids": r.get("mppi_natural_trajectory_ids", []),
                    "expert_trajectories_count": int(float(r.get("mppi_expert_trajectories", 0.0))),
                    "overwritten_blocks": [b for b in r.get("mppi_expert_blocks", []) if b in valid_elevation_planes],
                    "overwritten_trajectory_ids": r.get("mppi_expert_trajectory_ids", []),
                }
                if "mppi_elevation_breakdown" in r and isinstance(r["mppi_elevation_breakdown"], dict):
                    clean_ep_breakdown = {
                        k: v for k, v in r["mppi_elevation_breakdown"].items()
                        if any(k.endswith(f"_{p.lower()}") for p in ("low", "mid", "high"))
                    }
                    if clean_ep_breakdown:
                        hist_item["elevation_breakdown"] = clean_ep_breakdown
                if "mppi_trials" in r and isinstance(r["mppi_trials"], list):
                    hist_item["trials"] = r["mppi_trials"]
                evaluations_history.append(hist_item)

            resume["mppi_progression"] = {
                "total_evaluations": len(mppi_entries),
                "best_checkpoint": {
                    "epoch": int(best_mppi_ep),
                    "score": round(float(best_mppi_score), 6),
                    "best_success_rate_pct": best_succ_pct,
                },
                "final_metrics": {
                    "mppi_score": round(float(last_mppi["mppi_score"]), 6) if "mppi_score" in last_mppi and last_mppi["mppi_score"] is not None else None,
                    "success_rate_pct": final_succ_pct,
                    "mean_cartesian_error_cm": round(float(last_mppi["mppi_mean_error_cm"]), 3) if "mppi_mean_error_cm" in last_mppi and last_mppi["mppi_mean_error_cm"] is not None else None,
                    "grasp_error_cm": round(float(last_mppi["mppi_grasp_error_cm"]), 3) if "mppi_grasp_error_cm" in last_mppi and last_mppi["mppi_grasp_error_cm"] is not None else None,
                    "grasp_rot_err_rad": round(float(last_mppi["mppi_grasp_rot_err_rad"]), 4) if "mppi_grasp_rot_err_rad" in last_mppi and last_mppi["mppi_grasp_rot_err_rad"] is not None else None,
                    "terminal_error_cm": round(float(last_mppi["mppi_terminal_error_cm"]), 3) if "mppi_terminal_error_cm" in last_mppi and last_mppi["mppi_terminal_error_cm"] is not None else None,
                    "rotational_error_rad": round(float(final_rot), 4) if np.isfinite(final_rot) else None,
                    "terminal_velocity_rad_s": round(float(final_vel), 4) if np.isfinite(final_vel) else None,
                },
                "expert_execution": expert_summary,
                "evaluations_history": evaluations_history,
                "spec_verification": {
                    "grasp_success_rate_ge_80pct": bool(final_succ_raw >= 0.80 or best_succ_raw >= 0.80),
                    "placement_accuracy_lt_3cm": bool(final_term_err < 3.0),
                    "rotational_alignment_lt_0_05rad": bool(final_rot < 0.05),
                    "terminal_velocity_lt_1_5rad_s": bool(final_vel < 1.5),
                },
            }

    return resume


def save_metrics_resume(
    target_path: Union[str, Path],
    history: Optional[List[Dict[str, Any]]] = None,
    evaluation_result: Optional[RolloutEvaluationResult] = None,
    parameters: Optional[int] = None,
    num_segments: int = 10,
    model_name: Optional[str] = None,
    version: Optional[str] = None,
    evaluation_result_no_corrector: Optional[RolloutEvaluationResult] = None,
    pos_span: Optional[float] = None,
    vel_span: Optional[float] = None,
    mppi_evaluation_result: Optional[Dict[str, Any]] = None,
) -> Path:
    """Convenience helper to build and dump metrics resume JSON."""
    target_path = Path(target_path)
    resume = generate_metrics_resume(
        history=history,
        evaluation_result=evaluation_result,
        parameters=parameters,
        num_segments=num_segments,
        model_name=model_name,
        version=version,
        evaluation_result_no_corrector=evaluation_result_no_corrector,
        pos_span=pos_span,
        vel_span=vel_span,
        mppi_evaluation_result=mppi_evaluation_result,
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2)
    return target_path
