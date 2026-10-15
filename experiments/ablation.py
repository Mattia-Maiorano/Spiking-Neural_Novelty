"""
Automated Ablation Study Runner for SPWM.
Executes systematic architectural and learning ablations.
Usage:
    python experiments/ablation.py --base-config configs/experiments/spwm_v0_1_0.yaml --epochs 10
"""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
from typing import Any, Dict
import chronicle
import torch

from spwm.config.schema import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import plot_ablation_comparison
from spwm.learning.engine import BaseTrainer
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def run_single_ablation(
    name: str,
    cfg: ExperimentConfig,
    dataloaders: Dict[str, Any],
    device: torch.device,
    epochs: int,
    results_dir: Path,
) -> float:
    cfg.project.name = f"ablation_{name.lower().replace(' ', '_')}"
    cfg.training.epochs = epochs
    ablation_dir = results_dir / cfg.project.name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    chronicle.log_info(f"--- Running Ablation: {name} ({epochs} epochs) ---")
    set_seed(cfg.project.seed)

    model = SPWM.from_config(cfg).to(device)
    engine = BaseTrainer(
        model=model,
        cfg=cfg,
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
        num_objects=cfg.environment.num_objects,
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

    if args.base_config is not None:
        base_cfg = ExperimentConfig.load(args.base_config)
    else:
        base_cfg = ExperimentConfig()
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
        num_objects=base_cfg.environment.num_objects,
        drift_injection_prob=0.0,
    )

    ablation_matrix = {
        "Full SPWM (Nominal)": copy.deepcopy(base_cfg),
        "Ablation A: Fast Only": copy.deepcopy(base_cfg),
        "Ablation B: No Multi-step": copy.deepcopy(base_cfg),
        "Ablation C: No Sparsity": copy.deepcopy(base_cfg),
    }

    # Configure overrides
    ablation_matrix["Ablation A: Fast Only"].memory.betas = (0.90, 0.90)
    ablation_matrix["Ablation B: No Multi-step"].loss.lambda_multi = 0.0
    ablation_matrix["Ablation C: No Sparsity"].loss.lambda_sparse = 0.0

    scores = {}
    for name, cfg in ablation_matrix.items():
        score = run_single_ablation(name, cfg, dataloaders, device, args.epochs, results_dir)
        scores[name] = score

    plot_ablation_comparison(scores, results_dir / "ablation_comparison.png")


if __name__ == "__main__":
    main()
