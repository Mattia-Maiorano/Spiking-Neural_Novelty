"""
Generator script for metrics_v*.json resume files across experiment results directories.
Usage:
    python scripts/generate_metrics_resumes.py [--pattern "spwm_v0_1_0*"] [--results-dir results]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import chronicle
import torch

from spwm.config.schema import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.models.world_model import SPWM
from spwm.evaluation.evaluator import RolloutEvaluator, RolloutEvaluationResult
from spwm.evaluation.metrics_resume import save_metrics_resume


def process_folder(
    folder: Path,
    device: torch.device,
    force_eval: bool = False,
    num_segments: int = 10,
) -> None:
    chronicle.log_info(f"Processing folder: {folder.name}")

    # Load history if present
    log_file = folder / "training_log.json"
    history = None
    if log_file.is_file():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            chronicle.log_warning(f"Could not load training_log.json in {folder}: {e}")

    # Check for config
    cfg_file = folder / "config.yaml"
    cfg = None
    if cfg_file.is_file():
        try:
            cfg = ExperimentConfig.load(cfg_file)
        except Exception as e:
            chronicle.log_warning(f"Could not load config.yaml in {folder}: {e}")

    # Check for model checkpoint
    model_file = folder / "model.pt"
    if not model_file.is_file():
        model_file = folder / "checkpoint.pt"

    eval_result: Optional[RolloutEvaluationResult] = None
    total_params: Optional[int] = None

    if cfg is not None and model_file.is_file():
        try:
            mode = cfg.training.mode
            enable_corrector = (mode == "train_corrector") or ("corrector" in folder.name.lower())
            model = SPWM.from_config(cfg, enable_corrector=enable_corrector).to(device)
            ckpt = torch.load(model_file, map_location=device)
            state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())

            # Check if existing metrics already have evaluation
            existing_metrics_files = list(folder.glob("metrics*.json"))
            existing_eval = None
            if existing_metrics_files and not force_eval:
                for emf in existing_metrics_files:
                    try:
                        with open(emf, "r", encoding="utf-8") as f:
                            ed = json.load(f)
                        if "test" in ed or "evaluation" in ed:
                            existing_eval = ed.get("test", ed.get("evaluation", {}))
                            break
                    except Exception:
                        pass

            if existing_eval is not None and "position_error_per_horizon" in existing_eval:
                chronicle.log_info(f"Using existing evaluation numbers from {folder.name}")
                eval_result = RolloutEvaluationResult(
                    horizons=[int(k) for k in existing_eval["position_error_per_horizon"].keys()],
                    latent_mse_per_horizon={int(k): float(v) for k, v in existing_eval.get("latent_mse_per_horizon", {}).items()},
                    position_error_per_horizon={int(k): float(v) for k, v in existing_eval.get("position_error_per_horizon", {}).items()},
                    velocity_error_per_horizon={int(k): float(v) for k, v in existing_eval.get("velocity_error_per_horizon", {}).items()},
                    teacher_forcing_mse=float(existing_eval.get("teacher_forcing_mse", 0.0)),
                    mean_spike_rate=float(existing_eval.get("mean_spike_rate", 0.0)),
                )
            else:
                chronicle.log_info(f"Running RolloutEvaluator on test dataset for {folder.name}...")
                drift_prob = 0.0 if mode == "train_predictor" else cfg.data.drift_injection_prob
                dataloaders = create_dataloaders(
                    total_trajectories=cfg.data.total_trajectories,
                    train_split=cfg.data.train_split,
                    val_split=cfg.data.val_split,
                    sequence_length=cfg.data.sequence_length,
                    batch_size=cfg.data.batch_size,
                    height=cfg.environment.height,
                    width=cfg.environment.width,
                    num_objects=cfg.environment.num_objects,
                    standard_velocity_range=cfg.data.standard_velocity_range,
                    extrapolation_velocity_range=cfg.data.extrapolation_velocity_range,
                    drift_injection_prob=drift_prob,
                    drift_magnitude=cfg.data.drift_magnitude,
                    mixed_drift=cfg.data.mixed_drift,
                    event_threshold=cfg.environment.event_threshold,
                    box_size=cfg.environment.box_size,
                    dt=cfg.environment.dt,
                    num_workers=0,
                    cache_data=cfg.data.cache_data,
                )
                evaluator = RolloutEvaluator(
                    model=model,
                    device=device,
                    horizons=cfg.evaluation.rollout_horizons,
                    num_objects=cfg.environment.num_objects,
                    use_corrector=enable_corrector,
                    temporal_burn_in=cfg.training.temporal_burn_in,
                )
                eval_result = evaluator.evaluate_dataset(dataloaders["test"])
                chronicle.log_success(
                    f"Evaluation complete: Rollout MAE = {eval_result.rollout_mae:.5f} | "
                    f"Rollout Score (Dynamic) = {eval_result.dynamic_score:.5f} | "
                    f"Drift Ratio = {eval_result.drift_ratio:.2f}x"
                )
                pos_parts = [f"H={h}: {eval_result.position_error_per_horizon.get(h, float('nan')):.5f}" for h in eval_result.horizons]
                vel_parts = [f"H={h}: {eval_result.velocity_error_per_horizon.get(h, float('nan')):.5f}" for h in eval_result.horizons]
                chronicle.log_detail("📍 Position Errors (MAE)", " | ".join(pos_parts), indent_level=1)
                chronicle.log_detail("💨 Velocity Errors (MAE)", " | ".join(vel_parts), indent_level=1)

        except Exception as e:
            chronicle.log_warning(f"Could not run/load evaluation for {folder.name}: {e}")

    # Determine version tag
    version_tag = cfg.project.name if cfg is not None else "v0_1_0"
    model_name = folder.name

    # Save target files
    out_file1 = folder / f"metrics_{version_tag}.json"
    save_metrics_resume(
        target_path=out_file1,
        history=history,
        evaluation_result=eval_result,
        parameters=total_params,
        num_segments=num_segments,
        model_name=model_name,
        version=version_tag,
    )

    from spwm.evaluation.metrics_resume import get_canonical_metrics_filename
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
        )

    chronicle.log_success(f"Generated {out_file1.name} (and {out_file2.name}) for {folder.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metrics_v*.json resume tables for experiment folders.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--pattern", type=str, default="spwm_v0_1_0*", help="Glob pattern for folder names")
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
    for folder in matched_folders:
        process_folder(
            folder=folder,
            device=device,
            force_eval=args.force_eval,
            num_segments=args.segments,
        )


if __name__ == "__main__":
    main()
