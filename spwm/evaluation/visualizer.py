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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    boxes = [
        ("Event Stream\n[B, T, 2, H, W]", (0.08, 0.5), "#E2E8F0"),
        ("Spiking Sensory\nEncoder", (0.25, 0.5), "#CBD5E1"),
        ("Latent State z_t\n[B, D]", (0.42, 0.5), "#93C5FD"),
        ("M1: Nominal Predictor\n(t=0 -> H, clean manifold)", (0.72, 0.8), "#34D399"),
        ("M2: Recovery Funnel\n(Cyclic H -> 2H, Funnel Target)", (0.72, 0.5), "#FBBF24"),
        ("M3: Sustained Predictor\n(Max Rollout Relay with M2)", (0.72, 0.2), "#A78BFA"),
        ("Physical Decoder\n(x, y, vx, vy)", (0.95, 0.5), "#F472B6"),
    ]

    for text, (x, y), color in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="#475569", lw=1.5),
            fontsize=9.5, weight="bold",
        )

    # Connecting arrows
    ax.annotate("", xy=(0.17, 0.5), xytext=(0.33, 0.5), arrowprops=dict(arrowstyle="<-", lw=1.5, color="#475569"))
    ax.annotate("", xy=(0.34, 0.5), xytext=(0.50, 0.5), arrowprops=dict(arrowstyle="<-", lw=1.5, color="#475569"))
    ax.annotate("", xy=(0.50, 0.5), xytext=(0.60, 0.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("", xy=(0.50, 0.5), xytext=(0.60, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706"))
    ax.annotate("", xy=(0.50, 0.5), xytext=(0.60, 0.2), arrowprops=dict(arrowstyle="->", lw=1.5, color="#7C3AED"))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_relay_rollout(
    pred_traj: Union[np.ndarray, torch.Tensor],
    gt_traj: Union[np.ndarray, torch.Tensor],
    phase_horizon: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    channel_idx: int = 0,
) -> None:
    """
    Plots autonomous rollout across the Cascaded Temporal Relay (M1, M2, M3).
    Bands denote active model windows:
      - [0, H): M1 (Nominal Predictor)
      - [H + 2k*H, H + (2k+1)*H): M2 (Recovery Funnel)
      - [H + (2k+1)*H, H + (2k+2)*H): M3 (Sustained Autoregression)
    """
    set_paper_style()
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.detach().cpu().numpy()
    if isinstance(gt_traj, torch.Tensor):
        gt_traj = gt_traj.detach().cpu().numpy()

    if pred_traj.ndim == 3:
        pred_traj = pred_traj[0]
    if gt_traj.ndim == 3:
        gt_traj = gt_traj[0]

    T = min(len(pred_traj), len(gt_traj))
    H = phase_horizon
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot trajectories
    ax.plot(timesteps, gt_traj[:T, channel_idx], label="Ground Truth", color="#1E293B", lw=2.2, linestyle="--")
    ax.plot(timesteps, pred_traj[:T, channel_idx], label="Relay Prediction", color="#2563EB", lw=2.0)

    # Color background bands by active model
    # M1: [0, H)
    if H > 0:
        ax.axvspan(0, min(H, T), alpha=0.15, color="#10B981", label="M1 Window (Nominal)")
    
    # Cyclic M2 and M3
    m2_labeled = False
    m3_labeled = False
    t_curr = H
    cycle = 0
    while t_curr < T:
        t_next = min(t_curr + H, T)
        if cycle % 2 == 0:
            ax.axvspan(t_curr, t_next, alpha=0.15, color="#F59E0B", label="M2 Window (Funnel)" if not m2_labeled else None)
            m2_labeled = True
        else:
            ax.axvspan(t_curr, t_next, alpha=0.15, color="#8B5CF6", label="M3 Window (Sustained)" if not m3_labeled else None)
            m3_labeled = True
        ax.axvline(t_curr, color="#94A3B8", linestyle=":", lw=1.2)
        t_curr = t_next
        cycle += 1

    ax.set_title(f"Cascaded Temporal Relay Autonomous Rollout (Phase Horizon H={H})")
    ax.set_xlabel("Rollout Step (t)")
    ax.set_ylabel(f"Coordinate Value (Channel {channel_idx})")
    ax.legend(loc="upper left", frameon=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


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



