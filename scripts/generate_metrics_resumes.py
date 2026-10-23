"""
Generator script for metrics_v*.json resume files across experiment results directories.

Usage:
    python scripts/generate_metrics_resumes.py [--pattern "spwm_v0_2_0*"] [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import chronicle
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import (
    RolloutEvaluationResult,
    RolloutEvaluator,
    get_phase_eval_horizons,
)
from spwm.evaluation.metrics_resume import (
    get_canonical_metrics_filename,
    save_metrics_resume,
)
from spwm.evaluation.visualizer import (
    plot_latent_rollout_degradation,
    plot_multi_step_degradation,
    plot_physical_rollout_degradation,
    plot_relay_rollout,
    plot_training_curves,
)
from spwm.models.rollout import rollout_latent
from spwm.models.world_model import SPWM


def process_folder(
    folder: Path,
    device: torch.device,
    force_eval: bool = False,
    num_segments: int = 10,
) -> Optional[Dict[str, Any]]:
    chronicle.log_info(f"Processing folder: {folder.name}")

    log_file = folder / "training_log.json"
    history = None
    if log_file.is_file():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            chronicle.log_warning(f"Could not load training_log.json in {folder}: {e}")

    cfg_file = folder / "config.yaml"
    cfg = None
    if cfg_file.is_file():
        try:
            cfg = ExperimentConfig.load(cfg_file)
        except Exception as e:
            chronicle.log_warning(f"Could not load config.yaml in {folder}: {e}")

    # Strictly use model.pt (best validation checkpoint), never checkpoint.pt
    model_file = folder / "model.pt"

    eval_result: Optional[RolloutEvaluationResult] = None
    total_params: Optional[int] = None

    if cfg is not None and model_file.is_file():
        try:
            model = SPWM.from_config(cfg).to(device)
            ckpt = torch.load(model_file, map_location=device, weights_only=False)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=True)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())
            mode = cfg.training.mode
            expected_horizons = get_phase_eval_horizons(
                mode=mode,
                phase_horizon=cfg.training.phase_horizon,
                config_horizons=cfg.evaluation.rollout_horizons,
            )

            existing_metrics_files = list(folder.glob("metrics*.json"))
            existing_eval = None
            if existing_metrics_files and not force_eval:
                for emf in existing_metrics_files:
                    try:
                        with open(emf, "r", encoding="utf-8") as f:
                            ed = json.load(f)
                        if "test" in ed or "evaluation" in ed:
                            cand_eval = ed.get("test", ed.get("evaluation", {}))
                            cand_h = [int(k) for k in cand_eval.get("position_error_per_horizon", {}).keys()]
                            if sorted(cand_h) == sorted(expected_horizons):
                                existing_eval = cand_eval
                                break
                    except Exception as exc:
                        chronicle.log_warning(f"Could not read metrics file {emf}: {exc}")

            if existing_eval is not None and "position_error_per_horizon" in existing_eval:
                chronicle.log_info(f"Using existing evaluation numbers from {folder.name}")
                eval_result = RolloutEvaluationResult(
                    horizons=[int(k) for k in existing_eval["position_error_per_horizon"].keys()],
                    latent_mse_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("latent_mse_per_horizon", {}).items()
                    },
                    position_error_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("position_error_per_horizon", {}).items()
                    },
                    velocity_error_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("velocity_error_per_horizon", {}).items()
                    },
                    teacher_forcing_mse=float(existing_eval.get("teacher_forcing_mse", 0.0)),
                    mean_spike_rate=float(existing_eval.get("mean_spike_rate", 0.0)),
                    position_mse_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("position_mse_per_horizon", {}).items()
                    } if "position_mse_per_horizon" in existing_eval else None,
                    velocity_mse_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("velocity_mse_per_horizon", {}).items()
                    } if "velocity_mse_per_horizon" in existing_eval else None,
                    q_mse_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("q_mse_per_horizon", {}).items()
                    } if "q_mse_per_horizon" in existing_eval else None,
                    p_mse_per_horizon={
                        int(k): float(v) for k, v in existing_eval.get("p_mse_per_horizon", {}).items()
                    } if "p_mse_per_horizon" in existing_eval else None,
                )
            else:
                chronicle.log_info(
                    f"Running RolloutEvaluator on test dataset for {folder.name} [{mode}] "
                    f"with horizons={expected_horizons}..."
                )
                eval_seq_len = max(
                    cfg.data.sequence_length,
                    cfg.training.temporal_burn_in + max(expected_horizons) + 1,
                )
                dataloaders = create_dataloaders(
                    total_trajectories=cfg.data.total_trajectories,
                    train_split=cfg.data.train_split,
                    val_split=cfg.data.val_split,
                    sequence_length=eval_seq_len,
                    batch_size=cfg.data.batch_size,
                    height=cfg.environment.height,
                    width=cfg.environment.width,
                    num_objects=cfg.environment.num_objects,
                    standard_velocity_range=cfg.data.standard_velocity_range,
                    extrapolation_velocity_range=cfg.data.extrapolation_velocity_range,
                    drift_injection_prob=0.0,
                    drift_magnitude=cfg.data.drift_magnitude,
                    mixed_drift=cfg.data.mixed_drift,
                    event_threshold=cfg.environment.event_threshold,
                    box_size=cfg.environment.box_size,
                    dt=cfg.environment.dt,
                    num_workers=0,
                    cache_data=cfg.data.cache_data,
                )
                fixed_predictor_eval = model.m1 if mode == "train_m1" else None
                evaluator = RolloutEvaluator(
                    model=model,
                    device=device,
                    horizons=expected_horizons,
                    num_objects=cfg.environment.num_objects,
                    temporal_burn_in=cfg.training.temporal_burn_in,
                    fixed_predictor=fixed_predictor_eval,
                )
                eval_result = evaluator.evaluate_dataset(dataloaders["test"])
                chronicle.log_success(
                    f"Evaluation complete: Horizons={eval_result.horizons} | "
                    f"Rollout MAE = {eval_result.rollout_mae:.5f} | "
                    f"Rollout Score (Dynamic) = {eval_result.dynamic_score:.5f} | "
                    f"Drift Ratio = {eval_result.drift_ratio:.2f}x"
                )

                figures_dir = folder / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                if history:
                    plot_training_curves(history, figures_dir / "training_curves.png")
                plot_multi_step_degradation(
                    horizons=eval_result.horizons,
                    pos_errors=eval_result.position_error_per_horizon,
                    vel_errors=eval_result.velocity_error_per_horizon,
                    latent_errors=eval_result.latent_mse_per_horizon,
                    save_path=figures_dir / "rollout_degradation.png",
                )
                pos_span = 2.0 * float(cfg.environment.box_size)
                vel_span = float(cfg.data.standard_velocity_range[1] - cfg.data.standard_velocity_range[0])
                plot_physical_rollout_degradation(
                    horizons=eval_result.horizons,
                    pos_errors=eval_result.position_error_per_horizon,
                    vel_errors=eval_result.velocity_error_per_horizon,
                    save_path=figures_dir / "physical_rollout_degradation.png",
                    relative=True,
                    pos_span=pos_span,
                    vel_span=vel_span,
                )
                plot_latent_rollout_degradation(
                    horizons=eval_result.horizons,
                    latent_errors=eval_result.latent_mse_per_horizon,
                    q_errors=eval_result.q_mse_per_horizon,
                    p_errors=eval_result.p_mse_per_horizon,
                    save_path=figures_dir / "latent_rollout_degradation.png",
                    relative=True,
                )
                sample_batch = next(iter(dataloaders["test"]))
                events = sample_batch["events"].to(device)
                flat_kin = sample_batch["flat_kinematics"].to(device)
                burn_in = min(cfg.training.temporal_burn_in, max(0, events.shape[1] - 2))
                with torch.no_grad():
                    out = model(events)
                    z_start = out.latent_states[:, burn_in]
                    decoder_state = model.physical_decoder.warm_up_filter(out.latent_states[:, : burn_in + 1])
                    eval_h = min(max(expected_horizons), events.shape[1] - 1 - burn_in)
                    roll_res = rollout_latent(
                        model=model,
                        z_init=z_start,
                        horizon=eval_h,
                        with_grad=False,
                        decode_kinematics=True,
                        initial_decoder_state=decoder_state,
                        fixed_predictor=fixed_predictor_eval,
                    )
                    if roll_res.decoded_kinematics is not None:
                        gt_kin = flat_kin[:, burn_in + 1 : burn_in + 1 + eval_h]
                        plot_relay_rollout(
                            pred_traj=roll_res.decoded_kinematics,
                            gt_traj=gt_kin,
                            phase_horizon=cfg.training.phase_horizon,
                            save_path=figures_dir / "relay_rollout.png",
                        )
        except Exception as e:
            chronicle.log_warning(f"Could not run/load evaluation for {folder.name}: {e}")

    version_tag = cfg.project.name if cfg is not None else "v0_2_0"
    model_name = folder.name
    pos_span = (2.0 * float(cfg.environment.box_size)) if cfg is not None else None
    vel_span = float(cfg.data.standard_velocity_range[1] - cfg.data.standard_velocity_range[0]) if cfg is not None else None

    out_file1 = folder / f"metrics_{version_tag}.json"
    save_metrics_resume(
        target_path=out_file1,
        history=history,
        evaluation_result=eval_result,
        parameters=total_params,
        num_segments=num_segments,
        model_name=model_name,
        version=version_tag,
        pos_span=pos_span,
        vel_span=vel_span,
    )

    canonical_name = get_canonical_metrics_filename(version_tag)
    out_file2 = folder / canonical_name
    if out_file2 != out_file1:
        save_metrics_resume(
            target_path=out_file2,
            history=history,
            evaluation_result=eval_result,
            parameters=total_params,
            num_segments=num_segments,
            model_name=model_name,
            version=version_tag,
            pos_span=pos_span,
            vel_span=vel_span,
        )

    chronicle.log_success(f"Generated {out_file1.name} (and {out_file2.name}) for {folder.name}")
    with open(out_file2, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_combined_cascade_report(
    results_dir: Path,
    version_tag: str,
    phase_resumes: Dict[str, Dict[str, Any]],
) -> None:
    """Create a combined cascade report uniting M1 (0-10), M2 (0-20), and M3 (full relay 0-100)."""
    if not phase_resumes:
        return
    clean_tag = version_tag[5:] if version_tag.startswith("spwm_") else version_tag
    clean_tag = clean_tag if clean_tag.startswith("v") else f"v{clean_tag}"
    combined_path = results_dir / f"metrics_{clean_tag}_combined.json"

    combined_epochs = sum(int(r.get("total_epochs", 0)) for r in phase_resumes.values())
    m3_eval = phase_resumes.get("train_m3", {}).get("evaluation")
    combined_payload = {
        "version": version_tag,
        "model_name": f"{version_tag}_combined_cascade",
        "total_epochs_across_phases": combined_epochs,
        "phases": phase_resumes,
        "full_relay_evaluation": m3_eval,
    }
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2)
    chronicle.log_success(f"Generated combined 3-phase cascade report: {combined_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metrics_v*.json resume tables for experiment folders.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--pattern", type=str, default="spwm_v0_2_0*", help="Glob pattern for folder names")
    parser.add_argument("--device", type=str, default=None, help="Device to use for evaluation (mps, cuda, cpu)")
    parser.add_argument("--segments", type=int, default=10, help="Number of segments to split epochs into")
    parser.add_argument("--force-eval", action="store_true", help="Force re-evaluation of model on test dataset")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    matched_folders = sorted([d for d in results_dir.glob(args.pattern) if d.is_dir()])
    if not matched_folders:
        chronicle.log_warning(f"No folders matched pattern '{args.pattern}' in {results_dir}")
        return

    chronicle.log_info(f"Found {len(matched_folders)} folder(s) matching '{args.pattern}' on device {device}")
    phase_resumes: Dict[str, Dict[str, Any]] = {}
    version_tag = "spwm_v0_2_0"
    for folder in matched_folders:
        res = process_folder(
            folder=folder,
            device=device,
            force_eval=args.force_eval,
            num_segments=args.segments,
        )
        if res is not None:
            version_tag = res.get("version", version_tag)
            for pm in ("train_m1", "train_m2", "train_m3"):
                if folder.name.endswith(pm):
                    phase_resumes[pm] = res

    if len(phase_resumes) >= 2:
        generate_combined_cascade_report(results_dir, version_tag, phase_resumes)


if __name__ == "__main__":
    main()
