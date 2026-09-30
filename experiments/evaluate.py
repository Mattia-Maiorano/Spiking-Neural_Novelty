"""
Evaluation Entrypoint for SPWM and Baseline Checkpoints.
Usage:
    python experiments/evaluate.py --checkpoint results/<run_id>/model.pt --config results/<run_id>/config.yaml
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

from spwm.utils.reproducibility import set_seed
from spwm.utils.config import load_config
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.rollout import RolloutEvaluator
from spwm.evaluation.plots import plot_multi_step_degradation
from experiments.train import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained world model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_path = Path(args.checkpoint)
    out_dir = ckpt_path.parent
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path} on {device}")
    model = build_model(config, device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Create test datasets
    data_cfg = config.get("data", {})
    env_cfg = config.get("environment", {})
    dataloaders = create_dataloaders(
        total_trajectories=data_cfg.get("total_trajectories", 600),
        train_split=data_cfg.get("train_split", 0.8),
        val_split=data_cfg.get("val_split", 0.1),
        sequence_length=data_cfg.get("sequence_length", 30),
        batch_size=32,
        height=env_cfg.get("height", 32),
        width=env_cfg.get("width", 32),
        num_objects=env_cfg.get("num_objects", 1),
    )

    horizons = config.get("evaluation", {}).get("rollout_horizons", [1, 5, 10, 25])
    evaluator = RolloutEvaluator(model=model, device=device, horizons=horizons, num_objects=env_cfg.get("num_objects", 1))

    print("Evaluating autonomous rollouts on test set...")
    test_results = evaluator.evaluate_dataset(dataloaders["test"])
    print("Evaluating autonomous rollouts on extrapolation set...")
    extrap_results = evaluator.evaluate_dataset(dataloaders["extrapolation"])

    metrics = {
        "test": {
            "teacher_forcing_mse": test_results.teacher_forcing_mse,
            "mean_spike_rate": test_results.mean_spike_rate,
            "latent_mse_per_horizon": test_results.latent_mse_per_horizon,
            "position_error_per_horizon": test_results.position_error_per_horizon,
            "velocity_error_per_horizon": test_results.velocity_error_per_horizon,
        },
        "extrapolation": {
            "teacher_forcing_mse": extrap_results.teacher_forcing_mse,
            "latent_mse_per_horizon": extrap_results.latent_mse_per_horizon,
            "position_error_per_horizon": extrap_results.position_error_per_horizon,
            "velocity_error_per_horizon": extrap_results.velocity_error_per_horizon,
        },
    }

    # Save test_metrics.json
    metrics_path = out_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved test metrics to {metrics_path}")

    # Generate sample rollout predictions for visualization
    with torch.no_grad():
        sample_batch = next(iter(dataloaders["test"]))
        sample_events = sample_batch["events"][:8].to(device)
        out = model(sample_events)
        z0 = out.latent_states[:, 5]
        rollout_preds = model.predict_future(z0, horizon=20).cpu().numpy()
        true_latents = out.latent_states[:, 6:26].cpu().numpy()
        np.savez(out_dir / "rollout_predictions.npz", predictions=rollout_preds, ground_truth=true_latents)

    # Plot degradation curve
    curve_data = {config.get("project", {}).get("name", "Model"): test_results.latent_mse_per_horizon}
    plot_multi_step_degradation(curve_data, figures_dir / "fig3_rollout_degradation.png")

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
