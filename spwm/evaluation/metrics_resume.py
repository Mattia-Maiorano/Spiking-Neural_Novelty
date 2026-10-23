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
            "gate_passed": bool(eval_res.dynamic_score <= 1.0),
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
                    entry[k] = round(float(val), 6) if np.isfinite(val) else None
            sampled_table.append(entry)

        # Compute summary stats (initial, final, best, pct_change)
        def compute_stat(key: str) -> Optional[Dict[str, Any]]:
            vals = [(epochs[i], r[key]) for i, r in enumerate(history) if key in r and r[key] is not None and np.isfinite(r[key])]
            if not vals:
                return None
            initial_val = vals[0][1]
            final_val = vals[-1][1]
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
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2)
    return target_path
