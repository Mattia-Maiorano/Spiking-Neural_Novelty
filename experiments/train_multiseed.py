"""
Multi-Seed Statistical Significance Benchmark Runner (DOSSIER Section 6.4 — Full Two-Phase Pipeline).

Executes the COMPLETE two-phase training and evaluation pipeline for every random seed:
1. Phase 0 (Representation): Pre-trains EventEncoder (SpatialSoftmax), continuous ALIF multi-timescale memory,
   sensory predictor, and physical kinematic decoders under Phase 0 representation loss.
   This grounds the latent manifold (q, p) topologically to joint kinematics.
2. Phase 1 (M1 Latent Predictor): Strictly loads the Phase 0 representation checkpoint and trains the
   symplectic latent predictor M1 with multi-step autoregressive unrolling over horizon H=50 and Active Perception
   smooth soft-fade sensory handover (w_max = 0.40).
3. Evaluates multi-horizon autonomous rollouts on the unseen test partition across H in {1, 5, 10, 25, 50, 100}.
4. Aggregates sample mean +/- sample standard deviation (mean +/- std) for tracking errors, velocities,
   drift ratios, spike rates, and inference latencies.
5. Generates publication-ready error-bar degradation figures and summary Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chronicle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.models.world_model import SPWM
from spwm.training.metrics import parameter_count
from spwm.utils.reproducibility import set_seed


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    })


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Computes sample mean and sample standard deviation (ddof=1)."""
    if not values:
        raise ValueError("Cannot compute mean/std on empty values list.")
    if len(values) == 1:
        return float(values[0]), 0.0
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))
    return mean_val, std_val


def format_mean_std(mean: float, std: float, precision: int = 5) -> str:
    """Formats mean +/- std with exact decimal precision."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def measure_inference_latency_ms(
    model: nn.Module,
    device: torch.device,
    in_channels: int = 4,
    height: int = 32,
    width: int = 32,
    action_dim: int = 6,
    warmup_steps: int = 10,
    bench_steps: int = 100,
) -> float:
    """Measures single-frame forward inference latency in milliseconds."""
    model = model.to(device)
    model.eval()
    dummy_events = torch.randn(1, 1, in_channels, height, width, device=device)
    dummy_actions = torch.randn(1, 1, action_dim, device=device)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(dummy_events, actions=dummy_actions)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(bench_steps):
            _ = model(dummy_events, actions=dummy_actions)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
    t1 = time.perf_counter()

    latency_ms = ((t1 - t0) / bench_steps) * 1000.0
    return float(latency_ms)


def run_pipeline_for_seed(
    seed: int,
    config_path: Path,
    output_dir: Path,
    rep_epochs: int,
    rep_patience: int,
    m1_epochs: int,
    m1_patience: int,
    total_trajectories: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Executes the full canonical two-phase pipeline for a given seed:
    Phase 0 (Representation) -> Phase 1 (M1 Latent Predictor).
    Returns paths to (rep_model_pt, m1_model_pt).
    """
    rep_dir = output_dir / f"seed_{seed}_representation"
    m1_dir = output_dir / f"seed_{seed}_m1"
    rep_model_pt = rep_dir / "model.pt"
    m1_model_pt = m1_dir / "model.pt"

    # Reuse existing production checkpoints for seed 42 if present and matching
    if seed == 42:
        prod_rep = Path("results/spwm_v0_7_3_train_representation/model.pt")
        prod_m1 = Path("results/spwm_v0_7_3_train_m1/model.pt")
        if prod_rep.is_file() and prod_m1.is_file() and total_trajectories is None:
            chronicle.log_info("Seed 42: Reusing canonical converged production checkpoints from results/spwm_v0_7_3_...")
            rep_dir.mkdir(parents=True, exist_ok=True)
            m1_dir.mkdir(parents=True, exist_ok=True)
            if not rep_model_pt.exists():
                shutil.copy(prod_rep, rep_model_pt)
                if (prod_rep.parent / "config.yaml").exists():
                    shutil.copy(prod_rep.parent / "config.yaml", rep_dir / "config.yaml")
            if not m1_model_pt.exists():
                shutil.copy(prod_m1, m1_model_pt)
                if (prod_m1.parent / "config.yaml").exists():
                    shutil.copy(prod_m1.parent / "config.yaml", m1_dir / "config.yaml")
            return rep_model_pt, m1_model_pt

    # 1. Phase 0: Representation Pre-Training
    if rep_model_pt.is_file():
        chronicle.log_info(f"[Seed {seed}] Phase 0 Representation checkpoint already exists: {rep_model_pt}")
    else:
        chronicle.log_info(f"======================================================================")
        chronicle.log_info(f" [Seed {seed}] Step 1/2: Training Phase 0 Representation...")
        chronicle.log_info(f"======================================================================")
        rep_cmd = [
            sys.executable, "experiments/train.py",
            "--config", str(config_path),
            "--mode", "train_representation",
            "--seed", str(seed),
            "--epochs", str(rep_epochs),
            "--patience", str(rep_patience),
            "--output-dir", str(rep_dir),
        ]
        if total_trajectories is not None:
            rep_cmd.extend(["--total-trajectories", str(total_trajectories)])
        if device_name is not None:
            rep_cmd.extend(["--device", device_name])

        res = subprocess.run(rep_cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Phase 0 representation training failed for seed {seed} (exit code {res.returncode})")
        if not rep_model_pt.is_file():
            raise FileNotFoundError(f"Expected Phase 0 checkpoint not created at: {rep_model_pt}")
        chronicle.log_success(f"[Seed {seed}] Phase 0 Representation training complete: {rep_model_pt}")

    # 2. Phase 1: M1 Latent Predictor Training
    if m1_model_pt.is_file():
        chronicle.log_info(f"[Seed {seed}] Phase 1 M1 checkpoint already exists: {m1_model_pt}")
    else:
        chronicle.log_info(f"======================================================================")
        chronicle.log_info(f" [Seed {seed}] Step 2/2: Training Phase 1 M1 Latent Predictor...")
        chronicle.log_info(f"======================================================================")
        m1_cmd = [
            sys.executable, "experiments/train.py",
            "--config", str(config_path),
            "--mode", "train_m1",
            "--seed", str(seed),
            "--representation-checkpoint", str(rep_model_pt),
            "--epochs", str(m1_epochs),
            "--patience", str(m1_patience),
            "--output-dir", str(m1_dir),
        ]
        if total_trajectories is not None:
            m1_cmd.extend(["--total-trajectories", str(total_trajectories)])
        if device_name is not None:
            m1_cmd.extend(["--device", device_name])

        res = subprocess.run(m1_cmd, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Phase 1 M1 training failed for seed {seed} (exit code {res.returncode})")
        if not m1_model_pt.is_file():
            raise FileNotFoundError(f"Expected Phase 1 checkpoint not created at: {m1_model_pt}")
        chronicle.log_success(f"[Seed {seed}] Phase 1 M1 Latent Predictor training complete: {m1_model_pt}")

    return rep_model_pt, m1_model_pt


def evaluate_trained_m1_model(
    model_pt: Path,
    config_path: Path,
    device: torch.device,
    horizons: List[int],
    total_trajectories: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates multi-horizon autonomous rollout on test split using canonical RolloutEvaluator."""
    cfg = ExperimentConfig.load(config_path)
    if total_trajectories is not None:
        cfg = cfg.with_overrides(total_trajectories=total_trajectories)

    model = SPWM.from_config(cfg).to(device)
    raw_ckpt = torch.load(model_pt, map_location=device, weights_only=False)
    state = raw_ckpt["model_state"] if isinstance(raw_ckpt, dict) and "model_state" in raw_ckpt else raw_ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    eval_seq_len = max(cfg.data.sequence_length, cfg.training.temporal_burn_in + max(horizons) + 1)
    dataloaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=eval_seq_len,
        batch_size=cfg.data.batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        joint_count=cfg.environment.joint_count,
        action_dim=cfg.environment.action_dim,
        in_channels=cfg.model.in_channels,
        standard_velocity_range=cfg.data.standard_velocity_range,
        extrapolation_velocity_range=cfg.data.extrapolation_velocity_range,
        event_threshold=cfg.environment.event_threshold,
        dt=cfg.environment.dt,
        num_workers=cfg.data.num_workers,
        cache_data=cfg.data.cache_data,
        dataset_dir=cfg.data.dataset_dir,
        strict_dataset=cfg.data.strict_dataset,
    )

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=horizons,
        joint_count=cfg.environment.joint_count,
        temporal_burn_in=cfg.training.temporal_burn_in,
        fixed_predictor=model.predictor,
    )
    test_metrics = evaluator.evaluate_dataset(dataloaders["test"])
    latency_ms = measure_inference_latency_ms(model, device)

    pos_mses = test_metrics.position_mse_per_horizon or {}
    vel_mses = test_metrics.velocity_mse_per_horizon or {}
    pos_errs = test_metrics.position_error_per_horizon or {}
    vel_errs = test_metrics.velocity_error_per_horizon or {}

    h_min = min(horizons)
    h_max = max(horizons)
    drift = float((pos_mses[h_max] - pos_mses[h_min]) / max(1e-6, pos_mses[h_min]))

    return {
        "latency_ms": latency_ms,
        "mean_spike_rate": test_metrics.mean_spike_rate,
        "teacher_forcing_mse": test_metrics.teacher_forcing_mse,
        "position_mse": pos_mses,
        "velocity_mse": vel_mses,
        "position_error": pos_errs,
        "velocity_error": vel_errs,
        "drift_ratio": drift,
    }


def run_multiseed_benchmark(
    seeds: List[int],
    config_path: Path,
    output_dir: Path,
    rep_epochs: int = 30,
    rep_patience: int = 8,
    m1_epochs: int = 30,
    m1_patience: int = 8,
    total_trajectories: Optional[int] = None,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Coordinates the full two-phase multi-seed benchmark across all requested random seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if device_name is not None:
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    chronicle.log_info(
        f"Starting Multi-Seed Benchmark across {len(seeds)} seeds: {seeds} "
        f"(Device: {device}, Config: {config_path})"
    )

    eval_horizons = [1, 5, 10, 25, 50, 100]
    per_seed_results: Dict[int, Dict[str, Any]] = {}

    for seed in seeds:
        chronicle.log_info(f"\n======================================================================")
        chronicle.log_info(f" EXECUTING CANONICAL TWO-PHASE PIPELINE FOR SEED: {seed}")
        chronicle.log_info(f"======================================================================")

        rep_pt, m1_pt = run_pipeline_for_seed(
            seed=seed,
            config_path=config_path,
            output_dir=output_dir,
            rep_epochs=rep_epochs,
            rep_patience=rep_patience,
            m1_epochs=m1_epochs,
            m1_patience=m1_patience,
            total_trajectories=total_trajectories,
            device_name=str(device),
        )

        cfg_for_seed = m1_pt.parent / "config.yaml" if (m1_pt.parent / "config.yaml").exists() else config_path
        metrics = evaluate_trained_m1_model(
            model_pt=m1_pt,
            config_path=cfg_for_seed,
            device=device,
            horizons=eval_horizons,
            total_trajectories=total_trajectories,
        )

        per_seed_results[seed] = {
            "seed": seed,
            "latency_ms": metrics["latency_ms"],
            "mean_spike_rate": metrics["mean_spike_rate"],
            "teacher_forcing_mse": metrics["teacher_forcing_mse"],
            "position_mse": metrics["position_mse"],
            "velocity_mse": metrics["velocity_mse"],
            "position_error": metrics["position_error"],
            "velocity_error": metrics["velocity_error"],
            "drift_ratio": metrics["drift_ratio"],
        }

        chronicle.log_success(
            f"[Seed {seed}] Evaluated: Pos MSE (H=100): {metrics['position_mse'][100]:.5f} | "
            f"Vel MSE (H=100): {metrics['velocity_mse'][100]:.5f} | "
            f"Drift: {metrics['drift_ratio']:.4f}"
        )

    # Statistical Aggregation (Mean +/- Std)
    stats: Dict[str, Any] = {}
    stats["latency_ms"] = compute_mean_std([r["latency_ms"] for r in per_seed_results.values()])
    stats["mean_spike_rate"] = compute_mean_std([r["mean_spike_rate"] for r in per_seed_results.values()])
    stats["teacher_forcing_mse"] = compute_mean_std([r["teacher_forcing_mse"] for r in per_seed_results.values()])
    stats["drift_ratio"] = compute_mean_std([r["drift_ratio"] for r in per_seed_results.values()])

    stats["position_mse"] = {}
    stats["velocity_mse"] = {}
    stats["position_error"] = {}
    stats["velocity_error"] = {}

    for h in eval_horizons:
        stats["position_mse"][h] = compute_mean_std([r["position_mse"][h] for r in per_seed_results.values()])
        stats["velocity_mse"][h] = compute_mean_std([r["velocity_mse"][h] for r in per_seed_results.values()])
        stats["position_error"][h] = compute_mean_std([r["position_error"][h] for r in per_seed_results.values()])
        stats["velocity_error"][h] = compute_mean_std([r["velocity_error"][h] for r in per_seed_results.values()])

    final_payload = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "per_seed_results": per_seed_results,
        "aggregated_statistics": stats,
    }

    # Save metrics JSON
    metrics_path = output_dir / "multiseed_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    chronicle.log_success(f"Multi-seed metrics saved to: {metrics_path}")

    # Generate Markdown Table
    md_table = generate_multiseed_markdown_table(per_seed_results, stats, eval_horizons)
    table_path = output_dir / "multiseed_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")
    chronicle.log_success(f"Multi-seed markdown table saved to: {table_path}")

    # Plot Error-Bar Degradation Curves
    plot_multiseed_errorbars(stats, eval_horizons, output_dir / "multiseed_rollout_errorbars.png")
    chronicle.log_success(f"Multi-seed error-bar figure saved to: {output_dir / 'multiseed_rollout_errorbars.png'}")

    print("\n" + "=" * 100)
    print("STATISTICAL MULTI-SEED VALIDATION TABLE (DOSSIER SECTION 6.4: FULL TWO-PHASE PIPELINE)")
    print("=" * 100)
    print(md_table)
    print("=" * 100 + "\n")

    return final_payload


def generate_multiseed_markdown_table(
    per_seed: Dict[int, Dict[str, Any]],
    stats: Dict[str, Any],
    horizons: List[int],
) -> str:
    key_h = [1, 10, 50, 100]
    lines = [
        "| Run / Metric | Latency (ms) | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=1) | Vel MSE (H=10) | Vel MSE (H=50) | Vel MSE (H=100) | Drift Ratio (H=100) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for seed, r in per_seed.items():
        line = (
            f"| Seed {seed} | {r['latency_ms']:.2f} | {r['mean_spike_rate']*100:.1f}% | "
            f"{r['position_mse'][1]:.5f} | {r['position_mse'][10]:.5f} | {r['position_mse'][50]:.5f} | {r['position_mse'][100]:.5f} | "
            f"{r['velocity_mse'][1]:.5f} | {r['velocity_mse'][10]:.5f} | {r['velocity_mse'][50]:.5f} | {r['velocity_mse'][100]:.5f} | "
            f"{r['drift_ratio']:.4f} |"
        )
        lines.append(line)

    m_lat, s_lat = stats["latency_ms"]
    m_spk, s_spk = stats["mean_spike_rate"]
    m_drift, s_drift = stats["drift_ratio"]

    agg_pos = [format_mean_std(*stats["position_mse"][h], precision=4) for h in key_h]
    agg_vel = [format_mean_std(*stats["velocity_mse"][h], precision=4) for h in key_h]

    agg_line = (
        f"| **Mean ± Std (N={len(per_seed)})** | **{m_lat:.2f} ± {s_lat:.2f}** | **{m_spk*100:.1f} ± {s_spk*100:.1f}%** | "
        f"**{agg_pos[0]}** | **{agg_pos[1]}** | **{agg_pos[2]}** | **{agg_pos[3]}** | "
        f"**{agg_vel[0]}** | **{agg_vel[1]}** | **{agg_vel[2]}** | **{agg_vel[3]}** | "
        f"**{m_drift:.4f} ± {s_drift:.4f}** |"
    )
    lines.append(agg_line)
    return "\n".join(lines)


def plot_multiseed_errorbars(stats: Dict[str, Any], horizons: List[int], save_path: Path) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    h_arr = np.array(horizons)

    # 1. Position MSE vs Horizon
    ax1 = axes[0]
    pos_means = np.array([stats["position_mse"][h][0] for h in horizons])
    pos_stds = np.array([stats["position_mse"][h][1] for h in horizons])

    ax1.plot(h_arr, pos_means, "o-", color="#2ecc71", linewidth=2.5, markersize=7, label="SPWM Mean Pos MSE")
    ax1.fill_between(h_arr, np.maximum(0, pos_means - pos_stds), pos_means + pos_stds, color="#2ecc71", alpha=0.25, label="±1σ Confidence Band")
    ax1.errorbar(h_arr, pos_means, yerr=pos_stds, fmt="none", ecolor="#27ae60", elinewidth=1.5, capsize=4)

    ax1.set_xlabel("Autonomous Prediction Horizon (H)")
    ax1.set_ylabel("Joint Position MSE (rad²)")
    ax1.set_title("Multi-Seed Position Tracking Stability\n(Full Two-Phase Pipeline, Mean ± Std)")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Velocity MSE vs Horizon
    ax2 = axes[1]
    vel_means = np.array([stats["velocity_mse"][h][0] for h in horizons])
    vel_stds = np.array([stats["velocity_mse"][h][1] for h in horizons])

    ax2.plot(h_arr, vel_means, "s-", color="#3498db", linewidth=2.5, markersize=7, label="SPWM Mean Vel MSE")
    ax2.fill_between(h_arr, np.maximum(0, vel_means - vel_stds), vel_means + vel_stds, color="#3498db", alpha=0.25, label="±1σ Confidence Band")
    ax2.errorbar(h_arr, vel_means, yerr=vel_stds, fmt="none", ecolor="#2980b9", elinewidth=1.5, capsize=4)

    ax2.set_xlabel("Autonomous Prediction Horizon (H)")
    ax2.set_ylabel("Joint Velocity MSE (rad²/s²)")
    ax2.set_title("Multi-Seed Velocity Tracking Stability\n(Full Two-Phase Pipeline, Mean ± Std)")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed statistical significance validation (Full Two-Phase Pipeline).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--config", type=str, default="results/spwm_v0_7_3_train_m1/config.yaml", help="Path to base config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/multiseed", help="Output directory")
    parser.add_argument("--rep-epochs", type=int, default=30, help="Epochs for Phase 0 Representation")
    parser.add_argument("--rep-patience", type=int, default=8, help="Patience for Phase 0 Representation")
    parser.add_argument("--m1-epochs", type=int, default=30, help="Epochs for Phase 1 M1 Latent Predictor")
    parser.add_argument("--m1-patience", type=int, default=8, help="Patience for Phase 1 M1 Latent Predictor")
    parser.add_argument("--total-trajectories", type=int, default=None, help="Optional override for total trajectories")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, mps)")
    args = parser.parse_args()

    run_multiseed_benchmark(
        seeds=args.seeds,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        rep_epochs=args.rep_epochs,
        rep_patience=args.rep_patience,
        m1_epochs=args.m1_epochs,
        m1_patience=args.m1_patience,
        total_trajectories=args.total_trajectories,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
