"""
Main Training Entrypoint for SPWM.
Streamlined, config-first single source of truth.
Usage:
    python experiments/train.py --config configs/experiments/spwm_v0_1_0.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import chronicle
import torch

from spwm.config.schema import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import (
    plot_training_curves,
    plot_multi_step_degradation,
    plot_corrector_telemetry,
)
from spwm.evaluation.metrics_resume import save_metrics_resume
from spwm.learning.engine import BaseTrainer
from spwm.models.world_model import SPWM
from spwm.models.rollout import TelemetryContainer, rollout_latent
from spwm.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPWM with typed config schema.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config YAML, Python script/module, or omit for active ExperimentConfig")
    parser.add_argument("--mode", type=str, default=None, choices=["train_representation", "train_predictor", "train_joint", "train_corrector"], help="Override training mode")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--predictor-checkpoint", type=str, default=None, help="Predictor checkpoint for corrector phase")
    parser.add_argument("--representation-checkpoint", type=str, default=None, help="Representation checkpoint for predictor phase (defaults to results/{project_name}_train_representation/model.pt)")
    parser.add_argument("--temporal-burn-in", type=int, default=None, help="Override temporal burn-in steps for recurrent warm-up")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is not None:
        cfg = ExperimentConfig.load(args.config)
    else:
        # Default Python-defined single source of truth configuration
        cfg = ExperimentConfig()

    # Apply explicit CLI overrides
    if args.seed is not None:
        cfg.project.seed = args.seed
    if args.mode is not None:
        cfg.training.mode = args.mode
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.predictor_checkpoint is not None:
        cfg.training.predictor_checkpoint = args.predictor_checkpoint
    if args.representation_checkpoint is not None:
        cfg.training.representation_checkpoint = args.representation_checkpoint
    if args.temporal_burn_in is not None:
        cfg.training.temporal_burn_in = args.temporal_burn_in

    set_seed(cfg.project.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mode = cfg.training.mode
    if args.output_dir is not None:
        save_dir = Path(args.output_dir)
    else:
        save_dir = Path("results") / f"{cfg.project.name}_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Build Dataloaders: nominal clean in predictor/representation/joint mode, drift in corrector mode
    drift_prob = 0.0 if mode in ("train_predictor", "train_representation", "train_joint") else cfg.data.drift_injection_prob
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
        num_workers=cfg.data.num_workers,
        cache_data=cfg.data.cache_data,
    )

    # Instantiate Model
    enable_corrector = (mode == "train_corrector")
    model = SPWM.from_config(cfg, enable_corrector=enable_corrector).to(device)

    # In predictor phase, load pre-trained representation checkpoint
    if mode == "train_predictor":
        ckpt_path = cfg.training.representation_checkpoint
        if not ckpt_path:
            # Default to results/{project_name}_train_representation/model.pt if exists
            default_path = Path("results") / f"{cfg.project.name}_train_representation" / "model.pt"
            if default_path.is_file():
                ckpt_path = str(default_path)

        if ckpt_path:
            ckpt_file = Path(ckpt_path)
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"Representation checkpoint not found at: {ckpt_file}")

            ckpt_data = torch.load(ckpt_file, map_location=device)
            state_dict = ckpt_data["model_state"] if isinstance(ckpt_data, dict) and "model_state" in ckpt_data else ckpt_data
            model.load_state_dict(state_dict, strict=False)
            chronicle.log_success(f"Loaded representation checkpoint into predictor: {ckpt_file}")
        else:
            chronicle.log_warning(
                "No representation checkpoint specified (--representation-checkpoint) "
                f"and default '{Path('results') / f'{cfg.project.name}_train_representation' / 'model.pt'}' was not found. "
                "Training predictor from randomly initialized representation!"
            )

    # In corrector phase, load pre-trained predictor checkpoint
    if mode == "train_corrector":
        ckpt_path = cfg.training.predictor_checkpoint
        if not ckpt_path:
            # Default to results/{project_name}_train_predictor/model.pt if exists
            default_path = Path("results") / f"{cfg.project.name}_train_predictor" / "model.pt"
            if default_path.is_file():
                ckpt_path = str(default_path)

        if not ckpt_path:
            raise ValueError(
                "Mode 'train_corrector' requires 'predictor_checkpoint' set in config or passed via --predictor-checkpoint."
            )

        ckpt_file = Path(ckpt_path)
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Predictor checkpoint not found at: {ckpt_file}")

        ckpt_data = torch.load(ckpt_file, map_location=device)
        state_dict = ckpt_data["model_state"] if isinstance(ckpt_data, dict) and "model_state" in ckpt_data else ckpt_data
        model.load_state_dict(state_dict, strict=False)
        chronicle.log_success(f"Loaded predictor checkpoint: {ckpt_file}")

    # Instantiate BaseTrainer Engine
    engine = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        save_dir=save_dir,
    )

    # Resume checkpoint if present in save_dir
    resume_ckpt = save_dir / "checkpoint.pt"
    if resume_ckpt.is_file():
        engine.load_checkpoint(resume_ckpt)
        chronicle.log_info(f"Resumed from checkpoint: {resume_ckpt} at epoch {engine.start_epoch}")

    # Train
    history = engine.fit()

    # Save active config with run
    cfg.save(save_dir / "config.yaml")

    # Evaluate on test set (full 100 steps benchmark)
    eval_horizons = sorted(list(set(list(cfg.evaluation.rollout_horizons) + [100])))
    
    # 1. Main evaluation (with corrector if enabled)
    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=eval_horizons,
        num_objects=cfg.environment.num_objects,
        use_corrector=enable_corrector,
        temporal_burn_in=cfg.training.temporal_burn_in,
    )
    test_metrics = evaluator.evaluate_dataset(dataloaders["test"])
    chronicle.log_success(
        f"Test Evaluation (with corrector={enable_corrector}): Rollout MAE = {test_metrics.rollout_mae:.5f} | "
        f"Rollout Score (Dynamic) = {test_metrics.dynamic_score:.5f} | "
        f"Drift Ratio = {test_metrics.drift_ratio:.2f}x | "
    )

    # Detailed autonomous rollout breakdown across horizons
    chronicle.log_section_header("Multi-Horizon Autonomous Rollout Errors" + (" (With Corrector)" if enable_corrector else ""))
    pos_err_parts = [
        f"H={h}: {test_metrics.position_error_per_horizon.get(h, float('nan')):.5f}"
        for h in test_metrics.horizons
    ]
    vel_err_parts = [
        f"H={h}: {test_metrics.velocity_error_per_horizon.get(h, float('nan')):.5f}"
        for h in test_metrics.horizons
    ]
    chronicle.log_detail("📍 Position Errors (MAE / L2)", " | ".join(pos_err_parts), indent_level=1)
    chronicle.log_detail("💨 Velocity Errors (MAE / L2)", " | ".join(vel_err_parts), indent_level=1)
    if test_metrics.position_mse_per_horizon:
        pos_mse_parts = [
            f"H={h}: {test_metrics.position_mse_per_horizon.get(h, float('nan')):.5f}"
            for h in test_metrics.horizons
        ]
        vel_mse_parts = [
            f"H={h}: {test_metrics.velocity_mse_per_horizon.get(h, float('nan')):.5f}"
            for h in test_metrics.horizons
        ]
        chronicle.log_detail("🎯 Position Errors (MSE)", " | ".join(pos_mse_parts), indent_level=1)
        chronicle.log_detail("🎯 Velocity Errors (MSE)", " | ".join(vel_mse_parts), indent_level=1)
    chronicle.log_newline()

    # 2. Evaluation without corrector
    test_metrics_no_corrector = None
    if enable_corrector:
        evaluator_no_corr = RolloutEvaluator(
            model=model,
            device=device,
            horizons=eval_horizons,
            num_objects=cfg.environment.num_objects,
            use_corrector=False,
            temporal_burn_in=cfg.training.temporal_burn_in,
        )
        test_metrics_no_corrector = evaluator_no_corr.evaluate_dataset(dataloaders["test"])
        chronicle.log_success(
            f"Test Evaluation (without corrector): Rollout MAE = {test_metrics_no_corrector.rollout_mae:.5f} | "
            f"Rollout Score (Dynamic) = {test_metrics_no_corrector.dynamic_score:.5f} | "
            f"Drift Ratio = {test_metrics_no_corrector.drift_ratio:.2f}x | "
        )
        chronicle.log_section_header("Multi-Horizon Autonomous Rollout Errors (Without Corrector)")
        pos_err_no_corr = [
            f"H={h}: {test_metrics_no_corrector.position_error_per_horizon.get(h, float('nan')):.5f}"
            for h in test_metrics_no_corrector.horizons
        ]
        vel_err_no_corr = [
            f"H={h}: {test_metrics_no_corrector.velocity_error_per_horizon.get(h, float('nan')):.5f}"
            for h in test_metrics_no_corrector.horizons
        ]
        chronicle.log_detail("📍 Position Errors (MAE / L2)", " | ".join(pos_err_no_corr), indent_level=1)
        chronicle.log_detail("💨 Velocity Errors (MAE / L2)", " | ".join(vel_err_no_corr), indent_level=1)
        chronicle.log_newline()

    # Plot figures
    if history:
        plot_training_curves(history, figures_dir / "training_curves.png")

    if enable_corrector and test_metrics_no_corrector is not None:
        plot_multi_step_degradation(
            {
                "With Corrector": test_metrics.position_error_per_horizon,
                "Without Corrector": test_metrics_no_corrector.position_error_per_horizon,
            },
            save_path=figures_dir / "rollout_degradation.png",
        )
    else:
        plot_multi_step_degradation(
            horizons=eval_horizons,
            pos_errors=test_metrics.position_error_per_horizon,
            save_path=figures_dir / "rollout_degradation.png",
        )

    # Diagnostic Telemetry for SlowCorrector (if active in current model)
    if enable_corrector and getattr(model, "corrector", None) is not None:
        try:
            sample_batch = next(iter(dataloaders["test"]))
            events = sample_batch["events"].to(device)
            burn_in = min(cfg.training.temporal_burn_in, max(0, events.shape[1] - 2))
            with torch.no_grad():
                out = model(events, accumulate_local_updates=False)
                z_start = out.latent_states[:, burn_in]
                decoder_state = None
                if hasattr(model, "physical_decoder") and model.physical_decoder is not None:
                    decoder_state = model.physical_decoder.warm_up_filter(out.latent_states[:, : burn_in + 1])
                telemetry = TelemetryContainer()
                rollout_latent(
                    model=model,
                    z_init=z_start,
                    horizon=max(eval_horizons),
                    use_corrector=True,
                    burn_in_steps=0,
                    with_grad=False,
                    decode_kinematics=False,
                    initial_decoder_state=decoder_state,
                    telemetry=telemetry,
                )
                telemetry_path = figures_dir / "corrector_telemetry.png"
                plot_corrector_telemetry(telemetry, batch_idx=0, save_path=telemetry_path)
                chronicle.log_success(f"Generated SlowCorrector telemetry plot: {telemetry_path.name}")
        except Exception as e:
            chronicle.log_warning(f"Could not generate corrector telemetry plot: {e}")

    # Save metrics resume mimicking figures (metrics_v*.json)
    from spwm.evaluation.metrics_resume import get_canonical_metrics_filename
    total_params = sum(p.numel() for p in model.parameters())
    version_tag = cfg.project.name
    model_display_name = f"{cfg.project.name}_{mode}"
    metrics_resume_path = save_dir / f"metrics_{version_tag}.json"
    save_metrics_resume(
        target_path=metrics_resume_path,
        history=history,
        evaluation_result=test_metrics,
        parameters=total_params,
        num_segments=10,
        model_name=model_display_name,
        version=version_tag,
        evaluation_result_no_corrector=test_metrics_no_corrector,
    )
    # Also save canonical metrics_v*.json avoiding duplicate 'v' prefixes
    canonical_name = get_canonical_metrics_filename(version_tag)
    canonical_path = save_dir / canonical_name
    if canonical_path != metrics_resume_path:
        save_metrics_resume(
            target_path=canonical_path,
            history=history,
            evaluation_result=test_metrics,
            parameters=total_params,
            num_segments=10,
            model_name=model_display_name,
            version=version_tag,
            evaluation_result_no_corrector=test_metrics_no_corrector,
        )
    chronicle.log_success(f"Generated metrics resume: {metrics_resume_path.name}")


if __name__ == "__main__":
    main()