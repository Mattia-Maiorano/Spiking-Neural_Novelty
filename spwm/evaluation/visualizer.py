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
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    boxes = [
        ("Event Stream\n[B, T, 2, H, W]", (0.10, 0.5), "#E2E8F0"),
        ("SpatialSoftmax\nEventEncoder", (0.28, 0.5), "#CBD5E1"),
        ("Phase-Space State\nz_t = [q_t, p_t]", (0.48, 0.5), "#93C5FD"),
        ("Symplectic Predictor\n(M1 Semi-Implicit Euler)", (0.72, 0.5), "#34D399"),
        ("Physical Decoder\n(x, y, vx, vy)", (0.92, 0.5), "#F472B6"),
    ]

    for text, (x, y), color in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color, edgecolor="#475569", lw=1.5),
            fontsize=9.5, weight="bold",
        )

    # Connecting arrows
    ax.annotate("", xy=(0.19, 0.5), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
    ax.annotate("", xy=(0.36, 0.5), xytext=(0.39, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))
    ax.annotate("", xy=(0.58, 0.5), xytext=(0.60, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("", xy=(0.82, 0.5), xytext=(0.85, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#475569"))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rollout_trajectory(
    pred_traj: Union[np.ndarray, torch.Tensor],
    gt_traj: Union[np.ndarray, torch.Tensor],
    phase_horizon: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    channel_idx: int = 0,
) -> None:
    """
    Plots autonomous rollout from the Symplectic Latent Predictor (M1) vs Ground Truth.
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
    timesteps = np.arange(T)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot trajectories
    ax.plot(timesteps, gt_traj[:T, channel_idx], label="Ground Truth", color="#1E293B", lw=2.2, linestyle="--")
    ax.plot(timesteps, pred_traj[:T, channel_idx], label="Symplectic Prediction (M1)", color="#2563EB", lw=2.0)

    ax.set_title("Symplectic Latent Predictor (M1) Autonomous Rollout")
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


plot_relay_rollout = plot_rollout_trajectory


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
    vel_errors: Optional[Dict[int, float]] = None,
    latent_errors: Optional[Dict[int, float]] = None,
    label: str = "SPWM",
) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if horizons is not None:
        curves_or_horizons = horizons
    if pos_errors is not None:
        pos_errors_or_save_path = pos_errors

    if isinstance(curves_or_horizons, dict):
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
        if save_path is None:
            raise ValueError("save_path must be provided when passing horizons and pos_errors.")
        target_path = Path(save_path)
        h_vals = sorted(list(curves_or_horizons))
        errs = [pos_errors_or_save_path.get(h, 0.0) for h in h_vals]  # type: ignore
        ax.plot(h_vals, errs, marker="o", color="#2563EB", label=f"{label} (Position)")
        if vel_errors is not None:
            ax.plot(h_vals, [vel_errors.get(h, 0.0) for h in h_vals], marker="s", linestyle="--", color="#DC2626", label=f"{label} (Velocity)")
        if latent_errors is not None:
            ax.plot(h_vals, [latent_errors.get(h, 0.0) for h in h_vals], marker="^", linestyle="-.", color="#059669", label=f"{label} (Latent)")
        ax.set_xticks(h_vals)
    else:
        raise ValueError("Invalid arguments provided to plot_multi_step_degradation.")

    ax.set_title("Multi-Step Prediction Degradation vs Horizon")
    ax.set_xlabel("Prediction Horizon (Steps)")
    ax.set_ylabel("Error")
    ax.legend()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()


def plot_physical_rollout_degradation(
    horizons: Sequence[int],
    pos_errors: Dict[int, float],
    vel_errors: Optional[Dict[int, float]] = None,
    save_path: Union[str, Path] = "figures/physical_rollout_degradation.png",
    relative: bool = True,
    pos_span: Optional[float] = 2.0,
    vel_span: Optional[float] = 2.0,
    label: str = "SPWM",
) -> None:
    """
    Plots multi-step physical kinematic degradation (position & velocity error) vs horizon.
    If `relative=True`:
      - If `pos_span` / `vel_span` is provided, plots the error relative to each dynamic span (E_h / span).
      - Otherwise, errors are plotted as relative ratio to horizon H=1 error (E_h / E_1).
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    target_path = Path(save_path)
    h_vals = sorted(list(horizons))

    if relative:
        if pos_span is not None and pos_span > 0:
            pos_denom = pos_span
            pos_suffix = " / Pos Span"
        else:
            pos_denom = pos_errors.get(h_vals[0], 1.0) if (h_vals and pos_errors.get(h_vals[0], 0.0) > 0) else 1.0
            pos_suffix = " / H1"
        pos_err_list = [pos_errors.get(h, 0.0) / pos_denom for h in h_vals]
        pos_lbl = f"{label} (Position Relative Error{pos_suffix})"
    else:
        pos_err_list = [pos_errors.get(h, 0.0) for h in h_vals]
        pos_lbl = f"{label} (Position L2 Error)"
    ax.plot(h_vals, pos_err_list, marker="o", color="#2563EB", lw=2.2, label=pos_lbl)

    if vel_errors is not None:
        if relative:
            if vel_span is not None and vel_span > 0:
                vel_denom = vel_span
                vel_suffix = " / Vel Span"
            else:
                vel_denom = vel_errors.get(h_vals[0], 1.0) if (h_vals and vel_errors.get(h_vals[0], 0.0) > 0) else 1.0
                vel_suffix = " / H1"
            vel_err_list = [vel_errors.get(h, 0.0) / vel_denom for h in h_vals]
            vel_lbl = f"{label} (Velocity Relative Error{vel_suffix})"
        else:
            vel_err_list = [vel_errors.get(h, 0.0) for h in h_vals]
            vel_lbl = f"{label} (Velocity L2 Error)"
        ax.plot(h_vals, vel_err_list, marker="s", linestyle="--", color="#DC2626", lw=2.2, label=vel_lbl)

    ax.set_xticks(h_vals)
    ax.set_xlabel("Prediction Horizon (Steps)")
    ylabel = "Relative Error (Normalized to Span)" if (relative and pos_span is not None) else ("Relative Error (E_h / E_1)" if relative else "Euclidean Error (L2)")
    ax.set_ylabel(ylabel)
    ax.set_title("Physical Kinematics Rollout Degradation vs Horizon")
    ax.legend()

    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(target_path, dpi=300)
    plt.close()


def plot_latent_rollout_degradation(
    horizons: Sequence[int],
    latent_errors: Dict[int, float],
    q_errors: Optional[Dict[int, float]] = None,
    p_errors: Optional[Dict[int, float]] = None,
    save_path: Union[str, Path] = "figures/latent_rollout_degradation.png",
    relative: bool = True,
    label: str = "SPWM",
) -> None:
    """
    Plots multi-step latent state rollout degradation vs horizon.
    By default (`relative=True`), plots the relative error growth normalized by 1-step error (MSE(H) / MSE(H=1))
    for overall latent, coordinate subspace q, and momentum subspace p.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    target_path = Path(save_path)
    h_vals = sorted(list(horizons))

    lat_base = latent_errors.get(h_vals[0], 1.0) if (relative and h_vals and latent_errors.get(h_vals[0], 0.0) > 0) else 1.0
    lat_err_list = [(latent_errors.get(h, 0.0) / lat_base) if relative else latent_errors.get(h, 0.0) for h in h_vals]
    lat_lbl = f"{label} (Total Latent MSE{' / H1' if relative else ''})"
    ax.plot(h_vals, lat_err_list, marker="^", linestyle="-", color="#059669", lw=2.2, label=lat_lbl)

    if q_errors is not None:
        q_base = q_errors.get(h_vals[0], 1.0) if (relative and h_vals and q_errors.get(h_vals[0], 0.0) > 0) else 1.0
        q_err_list = [(q_errors.get(h, 0.0) / q_base) if relative else q_errors.get(h, 0.0) for h in h_vals]
        ax.plot(h_vals, q_err_list, marker="o", linestyle="--", color="#2563EB", lw=1.8, label=f"{label} (Coordinate q-MSE{' / H1' if relative else ''})")

    if p_errors is not None:
        p_base = p_errors.get(h_vals[0], 1.0) if (relative and h_vals and p_errors.get(h_vals[0], 0.0) > 0) else 1.0
        p_err_list = [(p_errors.get(h, 0.0) / p_base) if relative else p_errors.get(h, 0.0) for h in h_vals]
        ax.plot(h_vals, p_err_list, marker="s", linestyle=":", color="#DC2626", lw=1.8, label=f"{label} (Momentum p-MSE{' / H1' if relative else ''})")

    if relative:
        ax.axhline(1.0, color="#94A3B8", linestyle="--", lw=1.2, alpha=0.7, label="Baseline (H=1)")

    ax.set_xticks(h_vals)
    ax.set_xlabel("Prediction Horizon (Steps)")
    ax.set_ylabel("Normalized Relative Error (MSE_h / MSE_1)" if relative else "Mean Squared Error (MSE)")
    ax.set_title("Latent Space Rollout Degradation vs Horizon (Normalized Error)" if relative else "Latent Space Rollout Degradation vs Horizon")
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



