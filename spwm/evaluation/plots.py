"""
Publication-Quality Scientific Plotting Utilities for SPWM.
Generates all 7 required research figures:
- Figure 1: SPWM Architecture Diagram
- Figure 2: Training & Convergence Curves
- Figure 3: Multi-Step Prediction Degradation vs Horizon
- Figure 4: Spike Raster & Activity Histogram
- Figure 5: Fast vs Slow Memory Dynamics
- Figure 6: Baseline Comparative Benchmarking
- Figure 7: Ablation Study Results
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt


def set_paper_style():
    """Sets clean, publication-grade aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def plot_architecture_diagram(save_path: Union[str, Path]) -> None:
    """Figure 1: Visualizes the neuromorphic SPWM information flow."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Draw architecture blocks
    boxes = [
        ("Event Stream\n[B, T, 2, H, W]", (0.1, 0.5), "#E2E8F0"),
        ("Spiking Sensory\nEncoder (Conv-LIF)", (0.3, 0.5), "#CBD5E1"),
        ("Multi-Timescale Memory\n[Fast (τ=0.8) | Slow (τ=0.98)]", (0.55, 0.5), "#93C5FD"),
        ("Latent State z_t\n(Recurrent Dynamics)", (0.8, 0.5), "#60A5FA"),
        ("Predictor p(z_t+1|z_t)\nAutonomous Rollout", (0.8, 0.8), "#34D399"),
        ("Physical Probe\n(x, y, vx, vy)", (0.8, 0.2), "#FBBF24"),
    ]

    for text, (x, y), color in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="#475569", lw=1.5),
            fontsize=10, weight="bold",
        )

    # Draw arrows
    arrows = [
        ((0.18, 0.5), (0.22, 0.5)),
        ((0.38, 0.5), (0.43, 0.5)),
        ((0.67, 0.5), (0.72, 0.5)),
        ((0.8, 0.58), (0.8, 0.72)),
        ((0.8, 0.42), (0.8, 0.28)),
    ]
    for start, end in arrows:
        ax.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#1E293B"),
        )

    ax.set_title("Figure 1: SPWM-v1 Spiking Predictive World Model Architecture", pad=20)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(history: List[Dict[str, float]], save_path: Union[str, Path]) -> None:
    """Figure 2: Multi-panel convergence curves (Total, Pred, Var, Sparsity, Val)."""
    set_paper_style()
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel A: Total Train vs Val Loss
    axes[0, 0].plot(epochs, [r["total_loss"] for r in history], label="Train Total", color="#2563EB")
    axes[0, 0].plot(epochs, [r["val_total_loss"] for r in history], label="Val Total", color="#DC2626", linestyle="--")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("A. Objective Convergence")
    axes[0, 0].legend()

    # Panel B: Latent Prediction Loss
    axes[0, 1].plot(epochs, [r["l_pred"] for r in history], label="Train L_pred", color="#059669")
    axes[0, 1].plot(epochs, [r["val_l_pred"] for r in history], label="Val L_pred", color="#10B981", linestyle="--")
    axes[0, 1].set_ylabel("Prediction MSE")
    axes[0, 1].set_title("B. Latent Next-Step Prediction")
    axes[0, 1].legend()

    # Panel C: Physical Decoding Probe Errors
    axes[1, 0].plot(epochs, [r["val_pos_err"] for r in history], label="Position Error", color="#D97706")
    axes[1, 0].plot(epochs, [r["val_vel_err"] for r in history], label="Velocity Error", color="#7C3AED", linestyle="--")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Physical Error (Euclidean)")
    axes[1, 0].set_title("C. Kinematic Probe Accuracy")
    axes[1, 0].legend()

    # Panel D: Spiking Activity Rate
    axes[1, 1].plot(epochs, [r["val_spike_rate"] for r in history], label="Mean Spike Rate", color="#4B5563")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Spike Rate (spikes/step/neuron)")
    axes[1, 1].set_title("D. Neuromorphic Sparsity")
    axes[1, 1].legend()

    fig.suptitle("Figure 2: SPWM-v1 Training & Validation Dynamics", fontsize=14, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_multi_step_degradation(
    model_degradation_dict: Dict[str, Dict[int, float]],
    save_path: Union[str, Path],
) -> None:
    """
    Figure 3: Long-Horizon Prediction Degradation Curve.
    Compares autonomous rollout error vs prediction horizon H across models.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = {
        "SPWM-v1": "#2563EB",
        "GRU World Model": "#059669",
        "Vanilla SNN": "#D97706",
        "MLP Dynamics": "#DC2626",
    }
    markers = {"SPWM-v1": "o", "GRU World Model": "s", "Vanilla SNN": "^", "MLP Dynamics": "x"}

    for model_name, curve in model_degradation_dict.items():
        horizons = sorted(curve.keys())
        errors = [curve[h] for h in horizons]
        color = palette.get(model_name, "#475569")
        marker = markers.get(model_name, "o")
        ax.plot(horizons, errors, marker=marker, label=model_name, color=color)

    ax.set_xlabel("Autonomous Prediction Horizon (H timesteps)")
    ax.set_ylabel("Autonomous Rollout MSE")
    ax.set_title("Figure 3: Long-Horizon Error Degradation Curve", pad=15)
    ax.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_spike_raster(
    fast_spikes: np.ndarray,
    slow_spikes: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Figure 4: Spike raster plot and firing rate distributions."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})

    # Raster plot: sample trajectory [T, N]
    # fast neurons in bottom half, slow neurons in top half
    T, N_fast = fast_spikes.shape
    _, N_slow = slow_spikes.shape

    t_fast, n_fast = np.where(fast_spikes > 0)
    t_slow, n_slow = np.where(slow_spikes > 0)

    ax1.scatter(t_fast, n_fast, s=4, color="#3B82F6", label=f"Fast Population (τ_fast, N={N_fast})", alpha=0.8)
    ax1.scatter(t_slow, n_slow + N_fast, s=4, color="#EF4444", label=f"Slow Population (τ_slow, N={N_slow})", alpha=0.8)
    ax1.axhline(N_fast, color="#94A3B8", linestyle="--", lw=1.0)
    ax1.set_xlabel("Timestep (t)")
    ax1.set_ylabel("Neuron Index")
    ax1.set_title("Spike Raster Across Multi-Timescale Populations")
    ax1.legend(loc="upper right")

    # Histogram of firing rates
    combined_rates = np.concatenate([fast_spikes.mean(axis=0), slow_spikes.mean(axis=0)])
    ax2.hist(combined_rates, bins=15, color="#64748B", edgecolor="black")
    ax2.set_xlabel("Firing Rate")
    ax2.set_ylabel("Neuron Count")
    ax2.set_title("Activity Distribution")

    fig.suptitle("Figure 4: Neuromorphic Spike Raster and Activity Statistics", fontsize=14, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_fast_vs_slow_memory(
    fast_mems: np.ndarray,
    slow_mems: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Figure 5: Compares fast vs slow membrane potential dynamics over time."""
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    T = fast_mems.shape[0]
    timesteps = np.arange(T)

    # Plot sample neurons (mean + sample traces)
    ax1.plot(timesteps, fast_mems[:, :3], alpha=0.7)
    ax1.plot(timesteps, fast_mems.mean(axis=1), color="#1E40AF", lw=2.5, label="Population Mean")
    ax1.set_ylabel("V_mem (Fast, β=0.80)")
    ax1.set_title("Fast Neurons: High-Frequency Reactivity")
    ax1.legend(loc="upper right")

    ax2.plot(timesteps, slow_mems[:, :3], alpha=0.7)
    ax2.plot(timesteps, slow_mems.mean(axis=1), color="#B91C1C", lw=2.5, label="Population Mean")
    ax2.set_xlabel("Timestep (t)")
    ax2.set_ylabel("V_mem (Slow, β=0.98)")
    ax2.set_title("Slow Neurons: Temporal Integration & Momentum")
    ax2.legend(loc="upper right")

    fig.suptitle("Figure 5: Multi-Timescale Membrane Potential Dynamics", fontsize=14, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_baseline_comparison(
    metrics_by_model: Dict[str, Dict[str, float]],
    save_path: Union[str, Path],
) -> None:
    """Figure 6: Multi-metric benchmarking bar chart across models."""
    set_paper_style()
    models = list(metrics_by_model.keys())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Metric 1: One-Step Pred MSE
    pred_errors = [metrics_by_model[m].get("one_step_mse", 0.0) for m in models]
    axes[0].bar(models, pred_errors, color=["#2563EB", "#059669", "#D97706", "#DC2626"][:len(models)])
    axes[0].set_ylabel("One-Step MSE")
    axes[0].set_title("A. Latent Prediction Error")
    axes[0].tick_params(axis="x", rotation=25)

    # Metric 2: Multi-step Rollout Error (H=10)
    rollout_errors = [metrics_by_model[m].get("rollout_mse_h10", 0.0) for m in models]
    axes[1].bar(models, rollout_errors, color=["#2563EB", "#059669", "#D97706", "#DC2626"][:len(models)])
    axes[1].set_ylabel("Rollout MSE (H=10)")
    axes[1].set_title("B. Autonomous Rollout Error")
    axes[1].tick_params(axis="x", rotation=25)

    # Metric 3: Spike Rate (Sparsity)
    spike_rates = [metrics_by_model[m].get("spike_rate", 0.0) for m in models]
    axes[2].bar(models, spike_rates, color=["#2563EB", "#059669", "#D97706", "#DC2626"][:len(models)])
    axes[2].set_ylabel("Mean Spike Rate")
    axes[2].set_title("C. Neural Spike Sparsity")
    axes[2].tick_params(axis="x", rotation=25)

    fig.suptitle("Figure 6: Systematic Baseline Comparison", fontsize=14, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ablation_comparison(
    ablation_results: Dict[str, float],
    save_path: Union[str, Path],
) -> None:
    """Figure 7: Ablation study error impact comparison."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(ablation_results.keys())
    values = list(ablation_results.values())

    colors = ["#2563EB" if "Full" in n else "#EF4444" for n in names]
    bars = ax.barh(names, values, color=colors, height=0.6)
    ax.set_xlabel("Autonomous Rollout Error (H=10)")
    ax.set_title("Figure 7: Ablation Study — Contribution of Architectural Components", pad=15)

    # Add value annotations on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
