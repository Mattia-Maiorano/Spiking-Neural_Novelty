"""
Automated Ablation Study Runner.
Executes the required architectural and learning ablation matrix:
- Ablation A: No slow memory (both populations fast)
- Ablation B: No fast memory (both populations slow)
- Ablation C: Single-timescale SNN
- Ablation D: No prediction loss
- Ablation E: No sparsity regularization
- Ablation F: No multi-step loss
- Ablation G: FastSigmoid surrogate gradient
- Ablation H: Sigmoid surrogate gradient

Usage:
    python experiments/ablation.py --base-config configs/experiments/spwm_v1.yaml --epochs 10
"""

from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict
import torch

from spwm.utils.reproducibility import set_seed
from spwm.utils.config import load_config, merge_configs
from spwm.data.datasets import create_dataloaders
from spwm.learning.losses import SPWMLoss
from spwm.learning.trainer import Trainer
from spwm.evaluation.rollout import RolloutEvaluator
from spwm.evaluation.plots import plot_ablation_comparison
from experiments.train import build_model


def run_single_ablation(
    name: str,
    override_cfg: Dict[str, Any],
    base_config: Dict[str, Any],
    dataloaders: Dict[str, Any],
    device: torch.device,
    epochs: int,
    results_dir: Path,
) -> float:
    """Trains and evaluates a single ablation configuration, returning rollout MSE (H=10)."""
    cfg = merge_configs(copy.deepcopy(base_config), override_cfg)
    cfg["project"]["name"] = f"ablation_{name.lower().replace(' ', '_')}"
    ablation_dir = results_dir / cfg["project"]["name"]
    ablation_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Running Ablation: {name} ({epochs} epochs) ---")
    set_seed(cfg.get("project", {}).get("seed", 42))

    model = build_model(cfg, device=device)
    loss_cfg = cfg.get("loss", {})
    loss_fn = SPWMLoss(
        lambda_pred=loss_cfg.get("lambda_pred", 1.0),
        lambda_multi=loss_cfg.get("lambda_multi", 0.5),
        lambda_var=loss_cfg.get("lambda_var", 0.1),
        lambda_sparse=loss_cfg.get("lambda_sparse", 0.001),
        lambda_probe=loss_cfg.get("lambda_probe", 0.5),
        multi_step_horizon=loss_cfg.get("multi_step_horizon", 3),
    )

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        loss_fn=loss_fn,
        learning_rate=cfg.get("training", {}).get("learning_rate", 1e-3),
        device=device,
        save_dir=str(ablation_dir),
        tensorboard_logging=False,
    )

    trainer.fit(epochs=epochs)

    # Evaluate on test set
    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=[1, 5, 10],
        num_objects=cfg.get("environment", {}).get("num_objects", 1),
    )
    test_res = evaluator.evaluate_dataset(dataloaders["test"])
    rollout_h10 = test_res.latent_mse_per_horizon.get(10, test_res.teacher_forcing_mse)

    print(f"Result {name}: Rollout MSE (H=10) = {rollout_h10:.4f}")
    return rollout_h10


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPWM systematic ablation studies.")
    parser.add_argument("--base-config", type=str, required=True, help="Base config path")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per ablation run")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--results-dir", type=str, default="results/ablations", help="Results directory")
    args = parser.parse_args()

    base_config = load_config(args.base_config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data loaders
    data_cfg = base_config.get("data", {})
    env_cfg = base_config.get("environment", {})
    dataloaders = create_dataloaders(
        total_trajectories=data_cfg.get("total_trajectories", 600),
        train_split=0.8,
        val_split=0.1,
        sequence_length=data_cfg.get("sequence_length", 30),
        batch_size=32,
        height=env_cfg.get("height", 32),
        width=env_cfg.get("width", 32),
        num_objects=env_cfg.get("num_objects", 1),
    )

    # Define Ablations Matrix
    ablations = [
        ("Full SPWM-v1", {}),
        ("A: No Slow Memory", {"memory": {"betas": [0.80, 0.80]}}),
        ("B: No Fast Memory", {"memory": {"betas": [0.98, 0.98]}}),
        ("C: Single-Timescale SNN", {"model": {"type": "vanilla_snn", "beta": 0.85}}),
        ("D: No Prediction Loss", {"loss": {"lambda_pred": 0.0}}),
        ("E: No Sparsity Reg", {"loss": {"lambda_sparse": 0.0}}),
        ("F: No Multi-Step Loss", {"loss": {"lambda_multi": 0.0}}),
        ("G: FastSigmoid Surrogate", {"neuron": {"surrogate": "fast_sigmoid"}}),
        ("H: Sigmoid Surrogate", {"neuron": {"surrogate": "sigmoid"}}),
    ]

    ablation_results: Dict[str, float] = {}

    for name, override in ablations:
        rollout_err = run_single_ablation(
            name=name,
            override_cfg=override,
            base_config=base_config,
            dataloaders=dataloaders,
            device=device,
            epochs=args.epochs,
            results_dir=results_dir,
        )
        ablation_results[name] = float(rollout_err)

    # Save summary JSON
    summary_path = results_dir / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    # Generate Figure 7
    plot_ablation_comparison(ablation_results, results_dir / "fig7_ablation_comparison.png")

    print("\n=== Ablation Matrix Complete ===")
    print(f"Summary saved to {summary_path}")
    print(f"Figure saved to {results_dir / 'fig7_ablation_comparison.png'}")


if __name__ == "__main__":
    main()
