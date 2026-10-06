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

import chronicle
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
            q_dim=model_cfg.get("q_dim", None),
            p_dim=model_cfg.get("p_dim", None),
            ema_decay=model_cfg.get("ema_decay", 0.9),
            timescale_dims=tuple(mem_cfg.get("timescale_dims", (latent_dim // 2, latent_dim // 2))),
            betas=tuple(mem_cfg.get("betas", (0.90, 0.985))),
            beta_mem=neuron_cfg.get("beta_mem", 0.80),
            threshold=neuron_cfg.get("threshold", 1.0),
            gamma=neuron_cfg.get("gamma", 0.18),
            surrogate_name=neuron_cfg.get("surrogate", "atan"),
            surrogate_alpha=neuron_cfg.get("surrogate_alpha", 2.0),
            num_keypoints=model_cfg.get("num_keypoints", 16),
            predictor_hidden_dim=model_cfg.get("predictor_hidden_dim", 256),
            num_objects=num_objects,
            local_lr=model_cfg.get("local_lr", 1e-3),
            # v3.4: kinematic feedback into e-prop learning signal (0.0 disables for backward-compat)
            lambda_kin_feedback=model_cfg.get("lambda_kin_feedback", 0.0),
            # v3.5: online RLS decoder
            rls_enabled=model_cfg.get("rls_enabled", False),
            rls_forgetting=model_cfg.get("rls_forgetting", 0.99),
            rls_delta=model_cfg.get("rls_delta", 1.0),
            # v8: Slow Neuromorphic Corrector
            enable_corrector=config.get("corrector", {}).get("enabled", True),
            corrector_dim=config.get("corrector", {}).get("dim", 64),
            corrector_cadence=config.get("corrector", {}).get("cadence", 5),
            corrector_beta_mem=config.get("corrector", {}).get("beta_mem", 0.95),
            corrector_beta_adapt=config.get("corrector", {}).get("beta_adapt", 0.995),
            corrector_v_th0=config.get("corrector", {}).get("v_th0", 1.5),
            corrector_gamma=config.get("corrector", {}).get("gamma", 0.25),
            corrector_max_gain_v=config.get("corrector", {}).get("max_gain_v", 0.08),
            corrector_max_gain_p=config.get("corrector", {}).get("max_gain_p", 0.08),
            corrector_max_damp=config.get("corrector", {}).get("max_damp", 0.40),
            corrector_inter_step_decay=config.get("corrector", {}).get("inter_step_decay", 0.85),
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


def get_latest_config_path(configs_dir: Path = Path("configs/experiments")) -> Path:
    """Finds the latest experiment configuration file based on versioning."""
    import re
    if not configs_dir.is_dir():
        raise FileNotFoundError(f"Configs directory not found at: {configs_dir}")

    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML configuration files found in: {configs_dir}")

    def parse_version_key(path: Path):
        stem = path.stem
        m = re.match(r"^(.*?)(?:_v?(\d+.*))?$", stem)
        if not m or not m.group(2):
            return (0, stem, ())
        prefix, v_str = m.group(1), m.group(2)
        tokens = re.findall(r"[0-9]+|[a-zA-Z]+", v_str)
        parsed = []
        for t in tokens:
            if t.isdigit():
                parsed.append((1, int(t)))
            else:
                parsed.append((2, t.upper()))
        return (1, prefix, parsed)

    return max(yaml_files, key=parse_version_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPWM or baseline model.")
    parser.add_argument(
        "--config",
        type=str,
        default="latest",
        help="Path to experiment config YAML, or 'latest' to automatically use the highest version (default: 'latest')",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output results directory")
    args = parser.parse_args()

    # 1. Resolve and load configuration
    if args.config.lower() == "latest":
        config_path = get_latest_config_path()
        chronicle.log_info(f"Using latest configuration: {config_path}")
    else:
        config_path = Path(args.config)
        if not config_path.is_file() and not config_path.suffix:
            # Check if it was provided without .yaml extension
            potential = Path("configs/experiments") / f"{args.config}.yaml"
            if potential.is_file():
                config_path = potential

    config = load_config(config_path)
    if args.seed is not None:
        config["project"]["seed"] = args.seed
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    # Hardware & Device Selection (Cross-Platform)
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # Attiva l'autotuner CuDNN su Windows/Linux per ottimizzare le convoluzioni
        torch.backends.cudnn.benchmark = True 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Output directory (matches config stem if not explicitly provided)
    exp_name = config.get("project", {}).get("name", config_path.stem)
    if args.output_dir is not None:
        save_dir = Path(args.output_dir)
    else:
        save_dir = Path("results") / config_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    chronicle.log_application_title(f"Starting Experiment: {config_path.stem}")
    chronicle.log_detail("Device", device)
    chronicle.log_detail("Seed", seed)
    chronicle.log_detail("Save Dir", save_dir)

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
        drift_injection_prob=data_cfg.get("drift_injection_prob", 0.0),
        drift_magnitude=data_cfg.get("drift_magnitude", 0.05),
        mixed_drift=data_cfg.get("mixed_drift", False),
        num_workers=data_cfg.get("num_workers", 0),
        cache_data=data_cfg.get("cache_data", True),
    )

    # 3. Build Model & Loss
    model = build_model(config, device=device)
    model.init_buffer()
    
    loss_cfg = config.get("loss", {})
    model_cfg = config.get("model", {})
    loss_fn = SPWMLoss(
        lambda_pred=loss_cfg.get("lambda_pred", 1.0),
        lambda_multi=loss_cfg.get("lambda_multi", 0.5),
        lambda_var=loss_cfg.get("lambda_var", 0.1),
        lambda_sparse=loss_cfg.get("lambda_sparse", 0.5),
        lambda_vel=loss_cfg.get("lambda_vel", 0.5),
        lambda_probe=loss_cfg.get("lambda_probe", 0.5),
        lambda_coord=loss_cfg.get("lambda_coord", 0.0),
        beta_v=loss_cfg.get("beta_v", 2.0),
        sigma_q=loss_cfg.get("sigma_q", 1.0),
        sigma_v=loss_cfg.get("sigma_v", 1.0),
        use_empirical_variance=loss_cfg.get("use_empirical_variance", False),
        multi_step_horizon=loss_cfg.get("multi_step_horizon", 3),
        target_variance=loss_cfg.get("target_variance", 1.0),
        target_spike_rate=loss_cfg.get("target_spike_rate", 0.10),
        q_dim=model_cfg.get("q_dim", 32),
        p_dim=model_cfg.get("p_dim", 96),
    ).to(device)

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 20)
    lr = train_cfg.get("learning_rate", 1e-3)

    # 4. Train Model (Resume checkpoint or load existing model & history)
    model_path = save_dir / "model.pt"
    checkpoint_path = save_dir / "checkpoint.pt"
    log_json_path = save_dir / "training_log.json"

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_pos_err = float("inf")
    best_val_vel_err = float("inf")
    best_combined_score = float("inf")
    best_rollout_mae = float("inf")
    best_drift_ratio = float("inf")
    best_gate_passed = False
    history = []
    optimizer_state = None
    loaded_existing = False

    # Check for full training checkpoint first
    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_val_pos_err = ckpt.get("best_val_pos_err", float("inf"))
            best_val_vel_err = ckpt.get("best_val_vel_err", float("inf"))
            best_combined_score = ckpt.get("best_combined_score", float("inf"))
            best_rollout_mae = ckpt.get("best_rollout_mae", float("inf"))
            best_drift_ratio = ckpt.get("best_drift_ratio", float("inf"))
            best_gate_passed = ckpt.get("best_gate_passed", False)
            history = ckpt.get("history", []) or []
            optimizer_state = ckpt.get("optimizer_state", None)
            loaded_existing = True
            chronicle.log_info(f"Loaded full checkpoint from {checkpoint_path} (resuming at epoch {start_epoch})")
    elif model_path.is_file():
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
            chronicle.log_info(f"Loaded model state dict from {model_path}")
        else:
            model.load_state_dict(ckpt, strict=False)
            chronicle.log_info(f"Loaded plain model.pt from {model_path}")
        loaded_existing = True

        # Try to load existing history logs if checkpoint.pt was not available
        if log_json_path.is_file():
            try:
                with open(log_json_path, "r", encoding="utf-8") as f:
                    history = json.load(f) or []
                if history:
                    start_epoch = history[-1].get("epoch", len(history)) + 1
                    chronicle.log_info(f"Loaded existing history ({len(history)} epochs) from {log_json_path}")
            except Exception as e:
                chronicle.log_warning(f"Could not load existing history: {e}")
                history = []
    else:
        chronicle.log_info("No existing model or checkpoint — starting fresh.")

    probe_lr = train_cfg.get("probe_lr", 5e-4)
    probe_weight_decay = train_cfg.get("probe_weight_decay", 1e-2)
    eval_cfg = config.get("evaluation", {})

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        loss_fn=loss_fn,
        learning_rate=lr,
        probe_lr=probe_lr,
        probe_weight_decay=probe_weight_decay,
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        learning_algorithm=train_cfg.get("learning_algorithm", "online_eprop"),
        device=device,
        save_dir=str(save_dir),
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        best_val_pos_err=best_val_pos_err,
        best_val_vel_err=best_val_vel_err,
        best_combined_score=best_combined_score,
        best_rollout_mae=best_rollout_mae,
        best_drift_ratio=best_drift_ratio,
        best_gate_passed=best_gate_passed,
        history=history,
        optimizer_state=optimizer_state,
        curriculum_multi_step=train_cfg.get("curriculum_multi_step", False),
        smooth_horizon_sampling=train_cfg.get("smooth_horizon_sampling", False),
        k_max=config.get("k_max", 50),
        corrector_lr=train_cfg.get("corrector_lr", None),
        corrector_horizon=train_cfg.get("corrector_horizon", 25),
        lambda_corrector_asymptotic=loss_cfg.get("lambda_corrector_asymptotic", 1.0),
        lambda_corrector_quiescence=loss_cfg.get("lambda_corrector_quiescence", 0.5),
        corrector_quiescence_margin=loss_cfg.get("corrector_quiescence_margin", 0.15),
        corrector_quiescence_cap=loss_cfg.get("corrector_quiescence_cap", 0.50),
        curriculum_thresholds=loss_cfg.get("curriculum_thresholds", None),
        encoder_warmup_epochs=train_cfg.get("encoder_warmup_epochs", 60),
        max_drift_ratio=eval_cfg.get("max_drift_ratio", 5.0),
        rollout_horizons=eval_cfg.get("rollout_horizons", [1, 5, 10, 25, 50]),
        save_best_metric=eval_cfg.get("save_best_metric", "pareto_rollout"),
        num_objects=env_cfg.get("num_objects", 1),
    )

    if loaded_existing and (best_val_pos_err == float("inf") or best_val_loss == float("inf")):
        baseline = trainer.evaluate()
        trainer.best_val_loss = baseline["val_total_loss"]
        trainer.best_val_pos_err = baseline.get("val_pos_err", float("inf"))
        chronicle.log_detail("Baseline Val Loss", f"{trainer.best_val_loss:.4e}")
        chronicle.log_detail("Baseline Val Pos Err", f"{trainer.best_val_pos_err:.4e}")

    history = trainer.fit(epochs=epochs)

    # Ricarica il miglior modello prima della valutazione finale
    best_weights_path = save_dir / "model.pt"
    if best_weights_path.is_file():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        chronicle.log_info(f"Loaded best checkpoint from {best_weights_path} for final evaluation.")

    # 5. Evaluate on Test and Extrapolation sets
    try:
        chronicle.log_section_header("Rollout & Generalization Evaluation")
        evaluator = RolloutEvaluator(
            model=model,
            device=device,
            horizons=eval_cfg.get("rollout_horizons", [1, 5, 10, 25, 50]),
            num_objects=env_cfg.get("num_objects", 1),
        )

        test_results = evaluator.evaluate_dataset(dataloaders["test"])
        extrap_results = evaluator.evaluate_dataset(dataloaders["extrapolation"])

        chronicle.log_detail("Test TF MSE", f"{test_results.teacher_forcing_mse:.4e}")
        chronicle.log_detail("Test Mean Spike Rate", f"{test_results.mean_spike_rate:.3f}")
        chronicle.log_detail("Test Rollout MAE", f"{test_results.rollout_mae:.4f}")
        chronicle.log_detail("Test Drift Ratio (H50/H1)", f"{test_results.drift_ratio:.2f}x")
        chronicle.log_detail("Extrapolation TF MSE", f"{extrap_results.teacher_forcing_mse:.4e}")
        if test_results.position_error_per_horizon:
            chronicle.log_detail("Test Pos Error by Horizon", test_results.position_error_per_horizon)

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
    except Exception as e:
        chronicle.log_warning(f"Rollout evaluation skipped or incomplete: {e}")

    # 6. Save Artifacts & Metadata
    save_config(config, save_dir / "config.yaml")

    total_epochs_completed = len(history) if history else epochs
    metadata = {
        "run_id": save_dir.name,
        "experiment_name": exp_name,
        "seed": seed,
        "total_epochs": total_epochs_completed,
        "last_session_epochs": epochs,
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
    if history:
        plot_training_curves(history, figures_dir / "fig2_training_curves.png")
    plot_architecture_diagram(figures_dir / "fig1_architecture.png")

    if hasattr(model, "dynamics") and hasattr(model.dynamics, "memory"):
        with torch.no_grad():
            try:
                eval_loader = dataloaders.get("test") or dataloaders.get("val") or dataloaders.get("train")
                if eval_loader is not None and len(eval_loader) > 0:
                    sample_batch = next(iter(eval_loader))
                    sample_events = sample_batch["events"][:1].to(device)
                    sample_out = model(sample_events)
                    fast_spk = sample_out.fast_spikes[0].cpu().numpy()
                    slow_spk = sample_out.slow_spikes[0].cpu().numpy()
                    plot_spike_raster(fast_spk, slow_spk, figures_dir / "fig4_spike_raster.png")
            except Exception as e:
                chronicle.log_warning(f"Spike raster generation skipped: {e}")

    chronicle.log_success(f"Experiment '{exp_name}' completed.")
    chronicle.log_detail("Artifacts saved to", save_dir)


if __name__ == "__main__":
    main()
