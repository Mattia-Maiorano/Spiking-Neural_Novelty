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
from spwm.evaluation.visualizer import plot_training_curves, plot_multi_step_degradation
from spwm.learning.engine import BaseTrainer
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPWM with typed config schema.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config YAML, Python script/module, or omit for active ExperimentConfig")
    parser.add_argument("--mode", type=str, default=None, choices=["train_predictor", "train_corrector"], help="Override training mode")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--predictor-checkpoint", type=str, default=None, help="Predictor checkpoint for corrector phase")
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

    # Build Dataloaders: nominal clean in predictor mode, drift in corrector mode
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
        num_workers=cfg.data.num_workers,
        cache_data=cfg.data.cache_data,
    )

    # Instantiate Model
    enable_corrector = (mode == "train_corrector")
    model = SPWM.from_config(cfg, enable_corrector=enable_corrector).to(device)

    # In corrector phase, load pre-trained predictor checkpoint
    if mode == "train_corrector":
        ckpt_path = cfg.training.predictor_checkpoint
        if not ckpt_path:
            raise ValueError("Mode 'train_corrector' requires 'predictor_checkpoint' set in config or passed via --predictor-checkpoint.")

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

    # Evaluate on test set
    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=cfg.evaluation.rollout_horizons,
        num_objects=cfg.environment.num_objects,
        use_corrector=enable_corrector,
    )
    test_metrics = evaluator.evaluate_dataset(dataloaders["test"])
    chronicle.log_success(
        f"Test Evaluation: Rollout MAE = {test_metrics.rollout_mae:.5f} | Drift Ratio = {test_metrics.drift_ratio:.2f}x"
    )

    # Plot figures
    if history:
        plot_training_curves(history, figures_dir / "training_curves.png")
    plot_multi_step_degradation(
        horizons=cfg.evaluation.rollout_horizons,
        pos_errors=test_metrics.position_error_per_horizon,
        save_path=figures_dir / "rollout_degradation.png",
    )


if __name__ == "__main__":
    main()