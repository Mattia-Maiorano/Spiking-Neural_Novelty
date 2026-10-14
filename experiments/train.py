"""
Main Training Entrypoint for SPWM and Baselines.
Usage:
    python experiments/train.py --config configs/experiments/spwm_v0_1_0.yaml --seed 42
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
import traceback
import numpy as np  # Added for metric summarization

from spwm.utils.reproducibility import set_seed
from spwm.utils.config import load_config, save_config, merge_configs
from spwm.data.datasets import create_dataloaders
from spwm.models.world_model import SPWM
from spwm.learning.losses import SPWMLoss
from spwm.learning.trainer import Trainer
from spwm.learning.metrics import parameter_count
from spwm.evaluation.rollout import RolloutEvaluator
from spwm.evaluation.plots import (
    plot_training_curves,
    plot_corrector_ab_impact,
    plot_spike_raster,
    plot_fast_vs_slow_memory,
)


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.stdout.strip() if res.returncode == 0 else "git_not_committed"
    except Exception:
        return "git_unavailable"


def build_model(config: Dict[str, Any], device: torch.device, mode: str = "train_predictor") -> torch.nn.Module:
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
        if mode == "train_predictor":
            enable_corrector = False
        else:
            enable_corrector = config.get("corrector", {}).get("enabled", True)

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
            lambda_kin_feedback=model_cfg.get("lambda_kin_feedback", 0.0),
            enable_corrector=enable_corrector,
            corrector_dim=config.get("corrector", {}).get("dim", 128),
            corrector_beta_mem=config.get("corrector", {}).get("beta_mem", 0.95),
            corrector_beta_adapt=config.get("corrector", {}).get("beta_adapt", 0.94),
            corrector_v_th0=config.get("corrector", {}).get("v_th0", 1.0),
            corrector_gamma=config.get("corrector", {}).get("gamma", 0.25),
            corrector_alpha_gain=config.get("corrector", {}).get("alpha_gain", 0.5),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def parse_version_key(path_or_stem: Path | str):
    import re
    stem = path_or_stem.stem if isinstance(path_or_stem, Path) else str(path_or_stem)
    clean_stem = re.sub(r"_(train_predictor|train_corrector|predictor|corrector|joint)$", "", stem)
    m = re.match(r"^(.*?)(?:_v?(\d+.*))?$", clean_stem)
    if not m or not m.group(2): return (0, clean_stem, ())
    prefix, v_str = m.group(1), m.group(2)
    tokens = re.findall(r"[0-9]+|[a-zA-Z]+", v_str)
    parsed = []
    for t in tokens:
        if t.isdigit(): parsed.append((1, int(t)))
        else: parsed.append((2, t.upper()))
    return (1, prefix, parsed)


def get_latest_config_path(configs_dir: Path = Path("configs/experiments")) -> Path:
    if not configs_dir.is_dir(): raise FileNotFoundError(f"Configs directory not found at: {configs_dir}")
    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    if not yaml_files: raise FileNotFoundError(f"No YAML configuration files found in: {configs_dir}")
    return max(yaml_files, key=parse_version_key)


def find_antecedent_predictor_checkpoint(config_path: Path, results_dir: Path = Path("results")) -> Path | None:
    if not results_dir.is_dir(): return None
    curr_key = parse_version_key(config_path)
    candidate_dirs = [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    sorted_dirs = sorted(candidate_dirs, key=parse_version_key, reverse=True)

    prioritized_dirs = []
    other_dirs = []
    for d in sorted_dirs:
        d_key = parse_version_key(d)
        if d_key[0] == 1 and curr_key[0] == 1 and d_key[1] == curr_key[1]:
            if d_key[2] <= curr_key[2]: prioritized_dirs.append(d)
            else: other_dirs.append(d)
        else:
            other_dirs.append(d)

    target_files = ["model_predictor.pt", "predictor_checkpoint.pt", "model.pt", "checkpoint.pt"]
    for d in prioritized_dirs + other_dirs:
        for fname in target_files:
            candidate_file = d / fname
            if candidate_file.is_file(): return candidate_file
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SPWM or baseline model.")
    parser.add_argument("--config", type=str, default="latest")
    parser.add_argument("--mode", type=str, default=None, choices=["train_predictor", "train_corrector", "joint"])
    parser.add_argument("--predictor-checkpoint", type=str, default=None)
    parser.add_argument("--try-antecedent-predictor", action="store_true", default=False)
    parser.add_argument("--drift-prob", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.config.lower() == "latest":
        config_path = get_latest_config_path()
        chronicle.log_info(f"Using latest configuration: {config_path}")
    else:
        config_path = Path(args.config)
        if not config_path.is_file() and not config_path.suffix:
            potential = Path("configs/experiments") / f"{args.config}.yaml"
            if potential.is_file(): config_path = potential

    config = load_config(config_path)
    if args.seed is not None: config["project"]["seed"] = args.seed
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    train_cfg = config.get("training", {})
    mode = (args.mode or train_cfg.get("mode", "train_predictor")).lower()
    
    if args.device is not None: device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True 
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    exp_name = config.get("project", {}).get("name", config_path.stem)
    if args.output_dir is not None: save_dir = Path(args.output_dir)
    else:
        mode_suffix = f"_{mode}" if mode != "joint" else ""
        save_dir = Path("results") / f"{config_path.stem}{mode_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    chronicle.log_application_title(f"Starting Experiment: {config_path.stem} [{mode.upper()}]")
    chronicle.log_detail("Device", device)
    chronicle.log_detail("Seed", seed)
    chronicle.log_detail("Save Dir", save_dir)
    
    data_cfg = config.get("data", {})
    env_cfg = config.get("environment", {})
    batch_size = train_cfg.get("batch_size", 32)
    seq_len = data_cfg.get("sequence_length", 30)

    if mode == "train_predictor":
        drift_injection_prob = 0.0
        mixed_drift = False
        drift_magnitude = 0.0
        chronicle.log_detail("Dataset Dynamics", "Nominal clean trajectories")
    elif mode == "train_corrector":
        raw_drift = args.drift_prob if args.drift_prob is not None else data_cfg.get("drift_injection_prob", 0.15)
        drift_injection_prob = max(0.15, float(raw_drift))
        mixed_drift = data_cfg.get("mixed_drift", False)
        drift_magnitude = data_cfg.get("drift_magnitude", 0.05)
        chronicle.log_detail("Dataset Dynamics", f"Drift trajectories (p={drift_injection_prob:.2f})")
    else:
        drift_injection_prob = data_cfg.get("drift_injection_prob", 0.0)
        mixed_drift = data_cfg.get("mixed_drift", False)
        drift_magnitude = data_cfg.get("drift_magnitude", 0.05)

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
        drift_injection_prob=drift_injection_prob,
        drift_magnitude=drift_magnitude,
        mixed_drift=mixed_drift,
        event_threshold=env_cfg.get("event_threshold", 0.08),
        box_size=env_cfg.get("box_size", 1.0),
        dt=env_cfg.get("dt", 0.05),
        num_workers=data_cfg.get("num_workers", 0),
        cache_data=data_cfg.get("cache_data", True),
    )

    model = build_model(config, device=device, mode=mode)
    model.init_buffer()

    if mode == "train_corrector":
        predictor_ckpt = args.predictor_checkpoint
        if args.try_antecedent_predictor:
            antecedent_ckpt = find_antecedent_predictor_checkpoint(config_path)
            if antecedent_ckpt is not None: predictor_ckpt = str(antecedent_ckpt)
        if not predictor_ckpt: predictor_ckpt = train_cfg.get("predictor_checkpoint")
        if not predictor_ckpt:
            antecedent_ckpt = find_antecedent_predictor_checkpoint(config_path)
            if antecedent_ckpt is not None: predictor_ckpt = str(antecedent_ckpt)
            else:
                candidates = [
                    save_dir / "predictor_checkpoint.pt",
                    save_dir / "model_predictor.pt",
                    save_dir / "model.pt",
                    Path("results") / f"{config_path.stem}_train_predictor" / "model.pt",
                    Path("results") / f"{config_path.stem}_predictor" / "model.pt",
                ]
                for cand in candidates:
                    if cand.is_file():
                        predictor_ckpt = str(cand)
                        break

        if not predictor_ckpt:
            raise ValueError("Mode '--mode train_corrector' requires a pre-trained predictor checkpoint.")
        ckpt_file = Path(predictor_ckpt)
        if not ckpt_file.is_file(): raise FileNotFoundError(f"Pre-trained predictor checkpoint not found at: {predictor_ckpt}")

        loaded_data = torch.load(ckpt_file, map_location=device)
        state_dict_to_load = loaded_data["model_state"] if (isinstance(loaded_data, dict) and "model_state" in loaded_data) else loaded_data
        missing_keys, _ = model.load_state_dict(state_dict_to_load, strict=False)
        chronicle.log_success(f"Loaded predictor checkpoint: {ckpt_file}")

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

    if checkpoint_path.is_file():
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if "model_state" in ckpt: model.load_state_dict(ckpt["model_state"], strict=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_val_pos_err = ckpt.get("best_val_pos_err", float("inf"))
            best_val_vel_err = ckpt.get("best_val_vel_err", float("inf"))
            best_combined_score = ckpt.get("best_combined_score", float("inf"))
            best_rollout_mae = ckpt.get("best_rollout_mae", float("inf"))
            best_drift_ratio = ckpt.get("best_drift_ratio", float("inf"))
            best_gate_passed = ckpt.get("best_gate_passed", False)
            best_epoch = ckpt.get("best_epoch", None)
            history = ckpt.get("history", []) or []
            optimizer_state = ckpt.get("optimizer_state", None)
            loaded_existing = True
    elif model_path.is_file():
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt: model.load_state_dict(ckpt["model_state"], strict=False)
        else: model.load_state_dict(ckpt, strict=False)
        loaded_existing = True
        if log_json_path.is_file():
            try:
                with open(log_json_path, "r", encoding="utf-8") as f: history = json.load(f) or []
                if history: start_epoch = history[-1].get("epoch", len(history)) + 1
            except Exception:
                history = []

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
        best_epoch=best_epoch,
        history=history,
        optimizer_state=optimizer_state,
        curriculum_multi_step=train_cfg.get("curriculum_multi_step", False),
        smooth_horizon_sampling=train_cfg.get("smooth_horizon_sampling", False),
        k_max=config.get("k_max", 50),
        corrector_lr=train_cfg.get("corrector_lr", None),
        corrector_horizon=train_cfg.get("corrector_horizon", 100),
        corrector_burn_in_steps=train_cfg.get("corrector_burn_in_steps", 15),
        lambda_corrector_asymptotic=loss_cfg.get("lambda_corrector_asymptotic", 1.0),
        perturb_corrector_states=train_cfg.get("perturb_corrector_states", True),
        corrector_perturbation_std=train_cfg.get("corrector_perturbation_std", 0.05),
        curriculum_thresholds=loss_cfg.get("curriculum_thresholds", None),
        encoder_warmup_epochs=train_cfg.get("encoder_warmup_epochs", 60),
        max_drift_ratio=eval_cfg.get("max_drift_ratio", 5.0),
        rollout_horizons=eval_cfg.get("rollout_horizons", [1, 5, 10, 25, 50, 100]),
        num_objects=env_cfg.get("num_objects", 1),
        mode=mode,
    )

    if loaded_existing and (best_val_pos_err == float("inf") or best_val_loss == float("inf")):
        baseline = trainer.evaluate()
        trainer.best_val_loss = baseline["val_total_loss"]
        trainer.best_val_pos_err = baseline.get("val_pos_err", float("inf"))

    history = trainer.fit(epochs=epochs)

    best_weights_path = save_dir / "model.pt"
    if best_weights_path.is_file():
        try:
            ckpt_state = torch.load(best_weights_path, map_location=device)
            if isinstance(ckpt_state, dict) and "model_state" in ckpt_state: ckpt_state = ckpt_state["model_state"]
            model.load_state_dict(ckpt_state, strict=False)
        except Exception as e:
            chronicle.log_warning(f"Could not load best checkpoint from {best_weights_path}: {e}")

    try:
        chronicle.log_section_header("Rollout Evaluation")
        eval_horizons = eval_cfg.get("rollout_horizons", [1, 5, 10, 25, 50])
        num_objs = env_cfg.get("num_objects", 1)
        evaluator = RolloutEvaluator(model=model, device=device, horizons=eval_horizons, num_objects=num_objs, use_corrector=(mode != "train_predictor"))
        
        test_results = evaluator.evaluate_dataset(dataloaders["test"])
        extrap_results = evaluator.evaluate_dataset(dataloaders["extrapolation"])

        ab_comparison = {}
        if mode == "train_corrector":
            evaluator_no_corr = RolloutEvaluator(model=model, device=device, horizons=eval_horizons, num_objects=num_objs, use_corrector=False)
            test_no_corr = evaluator_no_corr.evaluate_dataset(dataloaders["test"])
            extrap_no_corr = evaluator_no_corr.evaluate_dataset(dataloaders["extrapolation"])
            ab_comparison = {
                "predictor_only": {
                    "test_rollout_mae": test_no_corr.rollout_mae,
                    "test_drift_ratio": test_no_corr.drift_ratio,
                    "test_pos_error_per_horizon": test_no_corr.position_error_per_horizon,
                    "extrap_rollout_mae": extrap_no_corr.rollout_mae,
                    "extrap_drift_ratio": extrap_no_corr.drift_ratio,
                },
                "predictor_with_corrector": {
                    "test_rollout_mae": test_results.rollout_mae,
                    "test_drift_ratio": test_results.drift_ratio,
                    "test_pos_error_per_horizon": test_results.position_error_per_horizon,
                    "extrap_rollout_mae": extrap_results.rollout_mae,
                    "extrap_drift_ratio": extrap_results.drift_ratio,
                }
            }

            chronicle.log_detail("Test TF MSE", f"{test_results.teacher_forcing_mse:.4e}")
            chronicle.log_detail("Test Mean Spike Rate", f"{test_results.mean_spike_rate:.3f}")
            chronicle.log_detail("Test Rollout MAE", f"{test_results.rollout_mae:.4f}")
            chronicle.log_detail("Test Drift Ratio", f"{test_results.drift_ratio:.2f}x")
            chronicle.log_detail("Extrapolation TF MSE", f"{extrap_results.teacher_forcing_mse:.4e}")
            chronicle.log_detail("Predictor ONLY - Test Rollout MAE", f"{test_no_corr.rollout_mae:.4f}")
            chronicle.log_detail("Predictor ONLY - Test Drift Ratio", f"{test_no_corr.drift_ratio:.2f}x")
            chronicle.log_detail("Predictor + Corrector - Test Rollout MAE", f"{test_results.rollout_mae:.4f}")
            chronicle.log_detail("Predictor + Corrector - Test Drift Ratio", f"{test_results.drift_ratio:.2f}x")

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
        if ab_comparison: metrics_payload["ab_comparison"] = ab_comparison
        
        # --- BLENDED METRICS PROGRESS SCRIPT ---
        if history:
            epochs_list = [r.get("epoch", i) for i, r in enumerate(history)]
            
            def series(key): return [r.get(key, None) for r in history]
            def finite(vals): return [v for v in vals if v is not None and np.isfinite(v)]
            
            def improvement(vals):
                vals_f = finite(vals)
                if not vals_f: return None
                return round(vals_f[0] - min(vals_f), 6)
            
            def pct_improvement(vals):
                vals_f = finite(vals)
                if not vals_f or vals_f[0] == 0: return None
                return round(100.0 * (vals_f[0] - min(vals_f)) / abs(vals_f[0]), 2)
            
            def epoch_of_best(vals):
                vals_clean = [(e, v) for e, v in zip(epochs_list, vals) if v is not None and np.isfinite(v)]
                if not vals_clean: return None
                return min(vals_clean, key=lambda x: x[1])[0]

            obj_key = "l_pred" if mode == "train_predictor" else "l_corr"
            train_obj = series(obj_key)
            rollout_mae = series("val_rollout_mae")
            sel_score = series("selection_score")
            val_pos = series("val_pos_err")
            val_vel = series("val_vel_err")
            drift = series("val_drift_ratio")

            horizon_keys = [1, 5, 10, 25, 50, 100]
            rollout_h_best = {}
            for h in horizon_keys:
                vals_h = series(f"val_pos_err_h{h}")
                best_val_h = min(finite(vals_h)) if finite(vals_h) else None
                rollout_h_best[str(h)] = round(best_val_h, 6) if best_val_h is not None else None

            best_epoch = epoch_of_best(sel_score)
            best_score = round(min(finite(sel_score)), 6) if finite(sel_score) else None
            total_epochs = epochs_list[-1] if epochs_list else 0
            training_time_s = history[-1].get("elapsed_time", None)

            stride = max(1, len(history) // 40)
            sampled = history[::stride]
            if not sampled or history[-1] != sampled[-1]:
                sampled.append(history[-1])

            def compact_curve(key, records):
                return {str(r.get("epoch", i)): round(r.get(key, 0.0), 6) for i, r in enumerate(records) if r.get(key) is not None}

            progress = {
                "training_progress": {
                    "total_epochs": total_epochs,
                    "best_epoch": best_epoch,
                    "best_selection_score": best_score,
                    "training_time_s": round(training_time_s, 1) if training_time_s else None,
                    "panel_A_objective": {
                        f"train_{obj_key}_initial": round(finite(train_obj)[0], 6) if finite(train_obj) else None,
                        f"train_{obj_key}_final": round(finite(train_obj)[-1], 6) if finite(train_obj) else None,
                        f"train_{obj_key}_best": round(min(finite(train_obj)), 6) if finite(train_obj) else None,
                        f"train_{obj_key}_improvement": improvement(train_obj),
                        f"train_{obj_key}_improvement_pct": pct_improvement(train_obj),
                        "curve": compact_curve(obj_key, sampled),
                    },
                    "panel_B_rollout_degradation": {
                        "val_rollout_mae_initial": round(finite(rollout_mae)[0], 6) if finite(rollout_mae) else None,
                        "val_rollout_mae_final": round(finite(rollout_mae)[-1], 6) if finite(rollout_mae) else None,
                        "val_rollout_mae_best": round(min(finite(rollout_mae)), 6) if finite(rollout_mae) else None,
                        "val_rollout_mae_improvement_pct": pct_improvement(rollout_mae),
                        "best_rollout_mae_by_horizon": rollout_h_best,
                        "selection_score_initial": round(finite(sel_score)[0], 6) if finite(sel_score) else None,
                        "selection_score_best": best_score,
                        "curve_rollout_mae": compact_curve("val_rollout_mae", sampled),
                        "curve_selection_score": compact_curve("selection_score", sampled),
                    },
                    "panel_C_kinematic_probe": {
                        "val_pos_err_initial": round(finite(val_pos)[0], 6) if finite(val_pos) else None,
                        "val_pos_err_final": round(finite(val_pos)[-1], 6) if finite(val_pos) else None,
                        "val_pos_err_best": round(min(finite(val_pos)), 6) if finite(val_pos) else None,
                        "val_vel_err_initial": round(finite(val_vel)[0], 6) if finite(val_vel) else None,
                        "val_vel_err_final": round(finite(val_vel)[-1], 6) if finite(val_vel) else None,
                        "val_vel_err_best": round(min(finite(val_vel)), 6) if finite(val_vel) else None,
                        "curve_pos_err": compact_curve("val_pos_err", sampled),
                        "curve_vel_err": compact_curve("val_vel_err", sampled),
                    },
                    "panel_D_drift_stability": {
                        "val_drift_ratio_initial": round(finite(drift)[0], 6) if finite(drift) else None,
                        "val_drift_ratio_final": round(finite(drift)[-1], 6) if finite(drift) else None,
                        "val_drift_ratio_best": round(min(finite(drift)), 6) if finite(drift) else None,
                        "val_drift_ratio_improvement_pct": pct_improvement(drift),
                        "curve_drift_ratio": compact_curve("val_drift_ratio", sampled),
                    },
                }
            }
            metrics_payload.update(progress)
            chronicle.log_success(f"Appended training_progress metrics. Total epochs: {total_epochs}, best epoch: {best_epoch}, sampled points: {len(sampled)}")
        # --- END OF BLENDED SCRIPT ---

        with open(save_dir / "metrics.json", "w", encoding="utf-8") as f: json.dump(metrics_payload, f, indent=2)
    except Exception as e:
        traceback.print_exc()
        chronicle.log_warning(f"Rollout evaluation failed or skipped: {e}")

    save_config(config, save_dir / "config.yaml")

    metadata = {
        "run_id": save_dir.name,
        "experiment_name": exp_name,
        "seed": seed,
        "total_epochs": len(history) if history else epochs,
        "last_session_epochs": epochs,
        "learning_rate": lr,
        "device": str(device),
        "pytorch_version": torch.__version__,
        "git_commit": get_git_commit_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(save_dir / "run_metadata.json", "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)

    mode_tag = "corrector" if mode == "train_corrector" else ("predictor" if mode == "train_predictor" else "joint")
    if history: plot_training_curves(history, figures_dir / f"fig2_training_curves_{mode_tag}.png", mode=mode)
    if mode == "train_corrector" and ab_comparison:
        try: plot_corrector_ab_impact(ab_comparison, figures_dir / "fig3_corrector_ab_impact.png")
        except Exception: pass

if __name__ == "__main__":
    main()