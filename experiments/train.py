"""
Main Training Entrypoint for SPWM and Baselines.
Usage:
    python experiments/train.py --config configs/experiments/spwm_v1.yaml --seed 42
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict
import torch

from spwm.utils.reproducibility import set_seed
from spwm.utils.config import load_config, save_config, merge_configs
from spwm.data.datasets import create_dataloaders
from spwm.models.world_model import SPWM
from spwm.baselines.gru_world_model import GRUWorldModel
from spwm.baselines.mlp_dynamics import MLPDynamicsWorldModel
from spwm.baselines.recurrent_snn import VanillaRecurrentSNN
from spwm.learning.losses import SPWMLoss
from spwm.learning.trainer import Trainer
from spwm.learning.metrics import parameter_count
from spwm.evaluation.rollout import RolloutEvaluator
from spwm.evaluation.plots import (
    plot_training_curves,
    plot_architecture_diagram,
    plot_spike_raster,
    plot_fast_vs_slow_memory,
)


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash if available."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.stdout.strip() if res.returncode == 0 else "git_not_committed"
    except Exception:
        return "git_unavailable"


def build_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Builds appropriate model based on configuration."""
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "spwm").lower()
    env_cfg = config.get("environment", {})
    mem_cfg = config.get("memory", {})
    neuron_cfg = config.get("neuron", {})

    height = env_cfg.get("height", 32)
    width = env_cfg.get("width", 32)
    num_objects = env_cfg.get("num_objects", 1)
    in_channels = model_cfg.get("in_channels", 2)
    latent_dim = model_cfg.get("latent_dim", 128)
    encoder_dim = model_cfg.get("encoder_dim", 128)

    if model_type == "spwm":
        model = SPWM(
            in_channels=in_channels,
            height=height,
            width=width,
            encoder_conv_channels=tuple(model_cfg.get("encoder_conv_channels", (32, 64))),
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            timescale_dims=tuple(mem_cfg.get("timescale_dims", (latent_dim // 2, latent_dim // 2))),
            betas=tuple(mem_cfg.get("betas", (0.90, 0.985))),
            beta_mem=neuron_cfg.get("beta_mem", 0.80),
            threshold=neuron_cfg.get("threshold", 1.0),
            gamma=neuron_cfg.get("gamma", 0.18),
            surrogate_name=neuron_cfg.get("surrogate", "atan"),
            surrogate_alpha=neuron_cfg.get("surrogate_alpha", 2.0),
            predictor_hidden_dim=model_cfg.get("predictor_hidden_dim", 256),
            num_objects=num_objects,
            local_lr=model_cfg.get("local_lr", 1e-3),
        )
    elif model_type == "gru":
        model = GRUWorldModel(
            in_channels=in_channels,
            height=height,
            width=width,
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            num_objects=num_objects,
        )
    elif model_type == "vanilla_snn":
        model = VanillaRecurrentSNN(
            in_channels=in_channels,
            height=height,
            width=width,
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            beta=model_cfg.get("beta", 0.85),
            threshold=neuron_cfg.get("threshold", 1.0),
            surrogate_name=neuron_cfg.get("surrogate", "atan"),
            num_objects=num_objects,
        )
    elif model_type == "mlp":
        model = MLPDynamicsWorldModel(
            in_channels=in_channels,
            height=height,
            width=width,
            latent_dim=latent_dim,
            num_objects=num_objects,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPWM or baseline model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output results directory")
    args = parser.parse_args()




    # 1. Load configuration
    config = load_config(args.config)
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Output directory (reuse existing if present)
    raw_exp_name = config.get("project", {}).get("name", "experiment")
    # If the experiment name contains "v2", rename it to "v3" to avoid overwriting corrupted v2 results
    exp_name = raw_exp_name.replace("v2", "v3")
    if args.output_dir is not None:
        save_dir = Path(args.output_dir)
    else:
        base = Path("results")
        pattern = f"{exp_name}_seed{seed}_"
        candidate_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(pattern)]
        if candidate_dirs:
            # Pick the most recently modified folder
            save_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)
            print(f"🔁 Reusing existing folder {save_dir} for resume")
        else:
            run_id = f"{exp_name}_seed{seed}_{int(time.time())}"
            save_dir = base / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Experiment: {exp_name} ===")
    print(f"Device: {device} | Seed: {seed} | Save Dir: {save_dir}")

    # 2. Build DataLoaders
    data_cfg = config.get("data", {})
    env_cfg = config.get("environment", {})
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    seq_len = data_cfg.get("sequence_length", 30)

    dataloaders = create_dataloaders(
        total_trajectories=data_cfg.get("total_trajectories", 600),
        train_split=data_cfg.get("train_split", 0.8),
        val_split=data_cfg.get("val_split", 0.1),
        sequence_length=seq_len,
        batch_size=batch_size,
        height=env_cfg.get("height", 32),
        width=env_cfg.get("width", 32),
        num_objects=env_cfg.get("num_objects", 1),
        standard_velocity_range=tuple(data_cfg.get("standard_velocity_range", (-1.0, 1.0))),
        extrapolation_velocity_range=tuple(data_cfg.get("extrapolation_velocity_range", (-2.0, 2.0))),
        num_workers=data_cfg.get("num_workers", 0),
        cache_data=data_cfg.get("cache_data", True),
    )

    # 3. Build Model & Loss
    model = build_model(config, device=device)
    loss_cfg = config.get("loss", {})
    loss_fn = SPWMLoss(
        lambda_pred=loss_cfg.get("lambda_pred", 1.0),
        lambda_multi=loss_cfg.get("lambda_multi", 0.5),
        lambda_var=loss_cfg.get("lambda_var", 0.1),
        lambda_sparse=loss_cfg.get("lambda_sparse", 0.001),
        lambda_probe=loss_cfg.get("lambda_probe", 0.5),
        multi_step_horizon=loss_cfg.get("multi_step_horizon", 3),
        target_variance=loss_cfg.get("target_variance", 1.0),
    )

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 20)
    lr = train_cfg.get("learning_rate", 1e-3)

    # 4. Train Model
    # Load existing model.pt if present
    model_path = save_dir / "model.pt"
    ckpt = None
    if model_path.is_file():
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            print(f"🔁 Loaded checkpoint (state dict) from {model_path}")
        else:
            model.load_state_dict(ckpt)
            print(f"🔁 Loaded plain model.pt from {model_path}")
    else:
        print("⚡ No existing model.pt, starting fresh.")
    start_epoch = 1
    best_val_loss = float("inf")
    history = None
    optimizer_state = None
    
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        loss_fn=loss_fn,
        learning_rate=lr,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        learning_algorithm=train_cfg.get("learning_algorithm", "online_eprop"),
        device=device,
        save_dir=str(save_dir),
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        history=history,
        optimizer_state=optimizer_state,
    )

    # If we loaded only a plain model.pt (no full checkpoint), compute its validation loss as baseline
    if ckpt is not None and not (isinstance(ckpt, dict) and "model_state" in ckpt):
        baseline = trainer.evaluate()
        trainer.best_val_loss = baseline["val_total_loss"]
        print(f"Baseline validation loss from existing model: {trainer.best_val_loss:.4e}")

    history = trainer.fit(epochs=epochs)

    # 5. Evaluate on Test and Extrapolation sets
    print("\n--- Running Rollout & Generalization Evaluation ---")
    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=config.get("evaluation", {}).get("rollout_horizons", [1, 5, 10, 25]),
        num_objects=env_cfg.get("num_objects", 1),
    )

    test_results = evaluator.evaluate_dataset(dataloaders["test"])
    extrap_results = evaluator.evaluate_dataset(dataloaders["extrapolation"])

    print(f"Test TF MSE: {test_results.teacher_forcing_mse:.4e} | Mean Spike Rate: {test_results.mean_spike_rate:.3f}")
    print(f"Extrapolation TF MSE: {extrap_results.teacher_forcing_mse:.4e}")
    if test_results.position_error_per_horizon:
        print(f"Test Pos Error by Horizon: {test_results.position_error_per_horizon}")


    # 6. Save Artifacts & Metadata
    save_config(config, save_dir / "config.yaml")

    metrics_payload = {
        "parameters": parameter_count(model),
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

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    metadata = {
        "run_id": save_dir.name,
        "experiment_name": exp_name,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "git_commit": get_git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 7. Generate Figures
    plot_training_curves(history, figures_dir / "fig2_training_curves.png")
    plot_architecture_diagram(figures_dir / "fig1_architecture.png")

    if hasattr(model, "dynamics") and hasattr(model.dynamics, "memory"):
        # Sample trajectory forward pass to extract spike raster and membrane potentials
        with torch.no_grad():
            eval_loader = dataloaders.get("test") or dataloaders.get("val") or dataloaders.get("train")
            if eval_loader is not None and len(eval_loader) > 0:
                sample_batch = next(iter(eval_loader))
                sample_events = sample_batch["events"][:1].to(device)
                sample_out = model(sample_events)
                fast_spk = sample_out.fast_spikes[0].cpu().numpy()
                slow_spk = sample_out.slow_spikes[0].cpu().numpy()
                plot_spike_raster(fast_spk, slow_spk, figures_dir / "fig4_spike_raster.png")

    print(f"=== Experiment {exp_name} Completed Successfully ===")
    print(f"Saved artifacts to {save_dir}")


if __name__ == "__main__":
    main()
