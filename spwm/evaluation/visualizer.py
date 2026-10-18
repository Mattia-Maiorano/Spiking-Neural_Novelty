"""
Streamlined Scientific Visualizer for SPWM.
Produces clean, publication-ready figures without deprecated baseline comparisons.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_paper_style():
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
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    boxes = [
        ("Event Stream\n[B, T, 2, H, W]", (0.1, 0.5), "#E2E8F0"),
        ("Spiking Sensory\nEncoder (SpatialSoftmax)", (0.3, 0.5), "#CBD5E1"),
        ("Multi-Timescale Memory\n[Reactive | Deep Context]", (0.55, 0.5), "#93C5FD"),
        ("Latent State z_t\n(Recurrent Dynamics)", (0.8, 0.5), "#60A5FA"),
        ("Fast Predictor\nΔz(z_t)", (0.8, 0.8), "#34D399"),
        ("Slow Corrector\nGain Mask Field", (0.55, 0.8), "#A78BFA"),
        ("Physical Probe\n(x, y, vx, vy)", (0.8, 0.2), "#FBBF24"),
    ]

    for text, (x, y), color in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="#475569", lw=1.5),
            fontsize=10, weight="bold",
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_curves(history: List[Dict[str, float]], save_path: Union[str, Path]) -> None:
    set_paper_style()
    epochs = [rec.get("epoch", i + 1) for i, rec in enumerate(history)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Loss curve
    train_losses = [r.get("train_loss", 0.0) for r in history]
    val_losses = [r.get("val_loss", 0.0) for r in history]
    axes[0].plot(epochs, train_losses, label="Train Loss", color="#2563EB")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#DC2626", linestyle="--")
    axes[0].set_title("Loss Evolution (Train / Val)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Kinematics
    axes[1].plot(epochs, [r.get("train_pos_err", 0.0) for r in history], label="Train Pos Err", color="#059669")
    axes[1].plot(epochs, [r.get("val_pos_err", 0.0) for r in history], label="Val Pos Err", color="#D97706", linestyle="--")
    axes[1].set_title("Kinematic Position Error")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Velocity Error
    axes[2].plot(epochs, [r.get("train_vel_err", 0.0) for r in history], label="Train Vel Err", color="#059669")
    axes[2].plot(epochs, [r.get("val_vel_err", 0.0) for r in history], label="Val Vel Err", color="#D97706", linestyle="--")
    axes[2].set_title("Kinematic Velocity Error")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_multi_step_degradation(
    curves_or_horizons: Optional[Union[Dict[str, Dict[int, float]], Sequence[int]]] = None,
    pos_errors_or_save_path: Optional[Union[Dict[int, float], str, Path]] = None,
    save_path: Optional[Union[str, Path]] = None,
    horizons: Optional[Sequence[int]] = None,
    pos_errors: Optional[Dict[int, float]] = None,
    label: str = "SPWM",
) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Resolve keyword args if horizons and pos_errors were passed by name
    if horizons is not None:
        curves_or_horizons = horizons
    if pos_errors is not None:
        pos_errors_or_save_path = pos_errors

    if isinstance(curves_or_horizons, dict):
        # Multi-model dictionary passed: curves_or_horizons = {"ModelA": {1: ..., 5: ...}}
        # Destination path can be passed positionally in pos_errors_or_save_path or as keyword save_path
        actual_save_path = save_path if save_path is not None else pos_errors_or_save_path
        if actual_save_path is None:
            raise ValueError("save_path must be provided when passing a dictionary of curves.")
        target_path = Path(actual_save_path)
        colors = ["#2563EB", "#DC2626", "#059669", "#D97706", "#7C3AED"]
        all_h = set()
        for idx, (m_name, m_curve) in enumerate(curves_or_horizons.items()):
            h_vals = sorted(list(m_curve.keys()))
            all_h.update(h_vals)
            errs = [m_curve[h] for h in h_vals]
            ax.plot(h_vals, errs, marker="o", color=colors[idx % len(colors)], label=m_name)
        ax.set_xticks(sorted(list(all_h)))
    elif curves_or_horizons is not None and pos_errors_or_save_path is not None:
        # horizons and pos_errors passed
        if save_path is None:
            raise ValueError("save_path must be provided when passing horizons and pos_errors.")
        target_path = Path(save_path)
        h_vals = sorted(list(curves_or_horizons))
        errs = [pos_errors_or_save_path.get(h, 0.0) for h in h_vals]  # type: ignore
        ax.plot(h_vals, errs, marker="o", color="#2563EB", label=label)
        ax.set_xticks(h_vals)
    else:
        raise ValueError("Invalid arguments provided to plot_multi_step_degradation.")

    ax.set_title("Multi-Step Prediction Degradation vs Horizon")
    ax.set_xlabel("Prediction Horizon (Steps)")
    ax.set_ylabel("Mean Position Error")
    ax.legend()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()


def plot_spike_raster(
    spikes_or_fast: np.ndarray,
    slow_spikes_or_save_path: Union[np.ndarray, str, Path],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    if isinstance(slow_spikes_or_save_path, np.ndarray):
        target_path = Path(save_path)
        fast_t, fast_n = np.where(spikes_or_fast > 0)
        slow_t, slow_n = np.where(slow_spikes_or_save_path > 0)
        ax.scatter(fast_t, fast_n, s=3, color="#2563EB", alpha=0.7, label="Reactive Pool")
        offset = spikes_or_fast.shape[-1]
        ax.scatter(slow_t, slow_n + offset, s=3, color="#DC2626", alpha=0.7, label="Deep Context Pool")
        ax.legend()
    else:
        target_path = Path(slow_spikes_or_save_path)
        t_idx, n_idx = np.where(spikes_or_fast > 0)
        ax.scatter(t_idx, n_idx, s=2, color="#0F172A", alpha=0.7)

    ax.set_title("Spiking Activity Raster")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron Index")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()



def plot_fast_vs_slow_memory(
    fast_mems: np.ndarray,
    slow_mems: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    set_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(fast_mems[:, :5], alpha=0.8)
    axes[0].set_title("Reactive Pool (Fast Memory)")
    axes[0].set_ylabel("Membrane Potential")

    axes[1].plot(slow_mems[:, :5], alpha=0.8)
    axes[1].set_title("Deep Context Pool (Slow Memory)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Membrane Potential")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_corrector_ab_impact(
    fast_trajectory: np.ndarray,
    corrected_trajectory: np.ndarray,
    ground_truth: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(ground_truth[:, 0], ground_truth[:, 1], "k-", label="Ground Truth", lw=2.5)
    ax.plot(fast_trajectory[:, 0], fast_trajectory[:, 1], "r--", label="Uncorrected (Fast)", lw=1.8)
    ax.plot(corrected_trajectory[:, 0], corrected_trajectory[:, 1], "b-", label="Gain-Modulated (SPWM)", lw=2.0)

    ax.set_title("A/B Trajectory Stabilization: Fast vs Corrected")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ablation_comparison(results: Dict[str, float], save_path: Union[str, Path]) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(results.keys())
    scores = list(results.values())

    colors = ["#2563EB" if "spwm" in n.lower() else "#94A3B8" for n in names]
    ax.barh(names, scores, color=colors)
    ax.set_title("Ablation Study Comparison (Rollout MSE H=10)")
    ax.set_xlabel("MSE (Lower is better)")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_model_comparison(metrics: Dict[str, Dict[str, float]], save_path: Union[str, Path]) -> None:
    """Figure 6: Model Comparative Benchmarking."""
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = list(metrics.keys())
    x = np.arange(len(models))

    tf_mse = [metrics[m].get("one_step_mse", 0.0) for m in models]
    h10_mse = [metrics[m].get("rollout_mse_h10", 0.0) for m in models]
    spk_rate = [metrics[m].get("spike_rate", 0.0) for m in models]

    colors = ["#3B82F6", "#8B5CF6", "#10B981"][:len(models)]
    axes[0].bar(x, tf_mse, color=colors, width=0.5)
    axes[0].set_title("1-Step Teacher Forcing MSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)

    axes[1].bar(x, h10_mse, color=colors, width=0.5)
    axes[1].set_title("Rollout MSE (H=10)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)

    axes[2].bar(x, spk_rate, color=colors, width=0.5)
    axes[2].set_title("Mean Spike Rate")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


plot_baseline_comparison = plot_model_comparison


def plot_corrector_telemetry(
    telemetry: Union[Dict[str, List[torch.Tensor]], Any],
    batch_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plots SlowCorrector diagnostic telemetry across rollout horizons:
    1. Spike Raster Plot
    2. Population Mean Membrane Potential & Adaptive Threshold
    3. Gain Mask Heatmap across latent coordinates
    """
    set_paper_style()

    # Handle TelemetryContainer or dict
    if hasattr(telemetry, "as_stacked_tensors"):
        stacked = telemetry.as_stacked_tensors()
        spikes = stacked["spikes"][:, batch_idx, :]
        v_mem = stacked["v_mem"][:, batch_idx, :]
        v_th = stacked["v_th"][:, batch_idx, :]
        gain_mask = stacked["gain_mask"][:, batch_idx, :]
    elif isinstance(telemetry, dict):
        spikes = torch.stack(telemetry["spikes"])[:, batch_idx, :]
        v_mem = torch.stack(telemetry["v_mem"])[:, batch_idx, :]
        v_th = torch.stack(telemetry["v_th"])[:, batch_idx, :]
        gain_mask = torch.stack(telemetry["gain_mask"])[:, batch_idx, :]
    else:
        raise ValueError("telemetry must be a TelemetryContainer or a dictionary of tensors.")

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 1. Spike Raster Plot
    spike_times, spike_neurons = torch.where(spikes > 0)
    axs[0].scatter(spike_times.numpy(), spike_neurons.numpy(), s=6, c="#1E293B", alpha=0.85)
    axs[0].set_ylabel("Neuron ID")
    axs[0].set_title("SlowCorrector Spike Raster")
    axs[0].set_ylim(-0.5, spikes.shape[1] - 0.5)

    # 2. Membrane Potential & Adaptive Threshold (Mean across population)
    axs[1].plot(v_mem.mean(dim=1).numpy(), label=r"Mean $V_{\mathrm{mem}}$", color="#2563EB", lw=2.0)
    axs[1].plot(v_th.mean(dim=1).numpy(), label=r"Mean $V_{\mathrm{th}}$", color="#DC2626", linestyle="--", lw=2.0)
    axs[1].set_ylabel("Voltage")
    axs[1].set_title("Population Dynamics (Membrane Potential vs Adaptive Threshold)")
    axs[1].legend(loc="upper right")

    # 3. Gain Mask Heatmap
    im = axs[2].imshow(gain_mask.T.numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axs[2].set_ylabel("Latent Dim")
    axs[2].set_xlabel("Timestep (Horizon)")
    axs[2].set_title("Gain Mask Output Field")
    fig.colorbar(im, ax=axs[2], orientation="vertical", pad=0.02)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

