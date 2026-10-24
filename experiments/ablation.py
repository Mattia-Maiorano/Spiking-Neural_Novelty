"""
Automated Ablation Study Runner for SPWM (v0.2.0).
Executes systematic architectural and learning ablations using immutable ExperimentConfig overrides.

Usage:
    python experiments/ablation.py --base-config configs/experiments/spwm_v0_1_0.yaml --epochs 10
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict
import chronicle
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import plot_ablation_comparison
from spwm.models.world_model import SPWM
from spwm.training.engine import BaseTrainer
from spwm.utils.reproducibility import set_seed


def run_single_ablation(
    name: str,
    cfg: ExperimentConfig,
    dataloaders: Dict[str, Any],
    device: torch.device,
    epochs: int,
    results_dir: Path,
) -> float:
    run_cfg = cfg.with_overrides(
        name=f"ablation_{name.lower().replace(' ', '_')}",
        epochs=epochs,
    )
    ablation_dir = results_dir / run_cfg.project.name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    chronicle.log_info(f"--- Running Ablation: {name} ({epochs} epochs) ---")
    set_seed(run_cfg.project.seed)

    model = SPWM.from_config(run_cfg).to(device)
    engine = BaseTrainer(
        model=model,
        cfg=run_cfg,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        save_dir=ablation_dir,
    )
    engine.fit(epochs=epochs)

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=[1, 5, 10],
        joint_count=run_cfg.environment.joint_count,
    )
    test_res = evaluator.evaluate_dataset(dataloaders["test"])
    rollout_h10 = test_res.latent_mse_per_horizon.get(10, test_res.teacher_forcing_mse)

    chronicle.log_success(f"Result {name}: Rollout MSE (H=10) = {rollout_h10:.4f}")
    return rollout_h10


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPWM systematic ablation studies.")
    parser.add_argument("--base-config", type=str, default=None, help="Base config path or YAML")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per ablation run")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--results-dir", type=str, default="results/ablations", help="Results directory")
    args = parser.parse_args()

    base_cfg = ExperimentConfig.load(args.base_config) if args.base_config is not None else ExperimentConfig()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataloaders = create_dataloaders(
        total_trajectories=base_cfg.data.total_trajectories,
        train_split=base_cfg.data.train_split,
        val_split=base_cfg.data.val_split,
        sequence_length=base_cfg.data.sequence_length,
        batch_size=base_cfg.data.batch_size,
        height=base_cfg.environment.height,
        width=base_cfg.environment.width,
        joint_count=base_cfg.environment.joint_count,
        action_dim=base_cfg.environment.action_dim,
        dataset_dir=base_cfg.data.dataset_dir,
    )

    ablation_matrix = {
        "Full SPWM (Nominal)": base_cfg,
        "Ablation A: Fast Memory Only": replace(
            base_cfg, memory=replace(base_cfg.memory, betas=(0.90, 0.90))
        ),
        "Ablation B: Short Relay Horizon": base_cfg.with_overrides(phase_horizon=5),
        "Ablation C: No Keypoint Supervision": replace(
            base_cfg, loss=replace(base_cfg.loss, lambda_encoder=0.0)
        ),
    }

    scores = {}
    for name, cfg in ablation_matrix.items():
        score = run_single_ablation(name, cfg, dataloaders, device, args.epochs, results_dir)
        scores[name] = score

    plot_ablation_comparison(scores, results_dir / "ablation_comparison.png")


if __name__ == "__main__":
    main()
