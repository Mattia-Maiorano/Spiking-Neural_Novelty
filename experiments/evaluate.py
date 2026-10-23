"""
Evaluation Entrypoint for SPWM Checkpoints (v0.2.0).

Usage:
    python experiments/evaluate.py --checkpoint results/<run_id>/model.pt --config results/<run_id>/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import chronicle
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator, get_phase_eval_horizons
from spwm.evaluation.visualizer import (
    plot_latent_rollout_degradation,
    plot_multi_step_degradation,
    plot_physical_rollout_degradation,
)
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SPWM model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.project.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "model.pt"
    out_dir = ckpt_path.parent
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    chronicle.log_info(f"Loading checkpoint strictly: {ckpt_path} on {device}")
    model = SPWM.from_config(cfg).to(device)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    eval_horizons = get_phase_eval_horizons(
        mode=cfg.training.mode,
        phase_horizon=cfg.training.phase_horizon,
        config_horizons=cfg.evaluation.rollout_horizons,
    )
    fixed_predictor = model.m1 if cfg.training.mode == "train_m1" else None
    eval_seq_len = max(cfg.data.sequence_length, cfg.training.temporal_burn_in + max(eval_horizons) + 1)

    dataloaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=eval_seq_len,
        batch_size=cfg.data.batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        num_objects=cfg.environment.num_objects,
        event_threshold=cfg.environment.event_threshold,
        box_size=cfg.environment.box_size,
        dt=cfg.environment.dt,
        drift_injection_prob=0.0,
        mixed_drift=cfg.data.mixed_drift,
    )

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=eval_horizons,
        num_objects=cfg.environment.num_objects,
        temporal_burn_in=cfg.training.temporal_burn_in,
        fixed_predictor=fixed_predictor,
    )
    results = evaluator.evaluate_dataset(dataloaders["test"])

    chronicle.log_success(f"Evaluated Horizons: {results.horizons}")
    chronicle.log_success(f"Teacher Forcing MSE: {results.teacher_forcing_mse:.5f}")
    chronicle.log_success(f"Rollout MAE: {results.rollout_mae:.5f}")
    chronicle.log_success(f"Drift Ratio: {results.drift_ratio:.2f}x")

    plot_multi_step_degradation(
        horizons=results.horizons,
        pos_errors=results.position_error_per_horizon,
        vel_errors=results.velocity_error_per_horizon,
        latent_errors=results.latent_mse_per_horizon,
        save_path=figures_dir / "eval_rollout_degradation.png",
    )
    pos_span = 2.0 * float(cfg.environment.box_size)
    vel_span = float(cfg.data.standard_velocity_range[1] - cfg.data.standard_velocity_range[0])
    plot_physical_rollout_degradation(
        horizons=results.horizons,
        pos_errors=results.position_error_per_horizon,
        vel_errors=results.velocity_error_per_horizon,
        save_path=figures_dir / "eval_physical_rollout_degradation.png",
        relative=True,
        pos_span=pos_span,
        vel_span=vel_span,
    )
    plot_latent_rollout_degradation(
        horizons=results.horizons,
        latent_errors=results.latent_mse_per_horizon,
        q_errors=results.q_mse_per_horizon,
        p_errors=results.p_mse_per_horizon,
        save_path=figures_dir / "eval_latent_rollout_degradation.png",
        relative=True,
    )


if __name__ == "__main__":
    main()
