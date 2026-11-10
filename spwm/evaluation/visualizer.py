"""
Streamlined Scientific Visualizer for SPWM.
Produces clean, publication-ready figures without deprecated baseline comparisons.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spwm.evaluation.evaluator import RolloutEvaluator, TrajectoryRollout


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


def plot_training_curves(history: List[Dict[str, Any]], save_path: Union[str, Path]) -> None:
    set_paper_style()
    epochs = [rec.get("epoch", i + 1) for i, rec in enumerate(history)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def _extract_series(key: str) -> Tuple[List[int], List[float]]:
        pts = [
            (epochs[i], float(r[key]))
            for i, r in enumerate(history)
            if key in r and r[key] is not None and isinstance(r[key], (int, float, np.number)) and np.isfinite(r[key])
        ]
        if not pts:
            return [], []
        x, y = zip(*pts)
        return list(x), list(y)

    # Loss curve
    tr_ep, tr_loss = _extract_series("train_loss")
    val_ep, val_loss = _extract_series("val_loss")
    if tr_ep:
        axes[0].plot(tr_ep, tr_loss, label="Train Loss", color="#2563EB")
    if val_ep:
        axes[0].plot(val_ep, val_loss, label="Val Loss", color="#DC2626", linestyle="--")
    axes[0].set_title("Loss Evolution (Train / Val)")
    axes[0].set_xlabel("Epoch")
    if tr_ep or val_ep:
        axes[0].legend()

    # Kinematics
    tr_pos_ep, tr_pos_err = _extract_series("train_pos_err")
    val_pos_ep, val_pos_err = _extract_series("val_pos_err")
    if tr_pos_ep:
        axes[1].plot(tr_pos_ep, tr_pos_err, label="Train Pos Err", color="#059669")
    if val_pos_ep:
        axes[1].plot(val_pos_ep, val_pos_err, label="Val Pos Err", color="#D97706", linestyle="--")
    axes[1].set_title("Kinematic Position Error")
    axes[1].set_xlabel("Epoch")
    if tr_pos_ep or val_pos_ep:
        axes[1].legend()

    # Velocity Error
    tr_vel_ep, tr_vel_err = _extract_series("train_vel_err")
    val_vel_ep, val_vel_err = _extract_series("val_vel_err")
    if tr_vel_ep:
        axes[2].plot(tr_vel_ep, tr_vel_err, label="Train Vel Err", color="#059669")
    if val_vel_ep:
        axes[2].plot(val_vel_ep, val_vel_err, label="Val Vel Err", color="#D97706", linestyle="--")
    axes[2].set_title("Kinematic Velocity Error")
    axes[2].set_xlabel("Epoch")
    if tr_vel_ep or val_vel_ep:
        axes[2].legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # If MPPI evaluation metrics are present in training history, automatically generate MPPI curves
    has_mppi = any("mppi_score" in r or "mppi_success_rate" in r for r in history)
    if has_mppi:
        mppi_save_path = save_path.parent / "mppi_training_curves.png"
        plot_mppi_training_curves(history, mppi_save_path)


def plot_mppi_training_curves(history: List[Dict[str, Any]], save_path: Union[str, Path]) -> None:
    """
    Renders publication-ready diagnostic curves for Closed-Loop Latent MPPI training.
    Visualizes the Two-Stage Phasic Target Switching (Stage 1 Grasp at t_c vs Stage 2 Placement at t_grasp),
    structural score / cost convergence, success rate, and terminal kinematic stability.
    """
    set_paper_style()
    epochs = [rec.get("epoch", i + 1) for i, rec in enumerate(history)]

    def _extract_series(key: str) -> Tuple[List[int], List[float]]:
        pts = [
            (epochs[i], float(r[key]))
            for i, r in enumerate(history)
            if key in r and r[key] is not None and isinstance(r[key], (int, float, np.number)) and np.isfinite(r[key])
        ]
        if not pts:
            return [], []
        x, y = zip(*pts)
        return list(x), list(y)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel (0, 0): MPPI Cost & Structural Score Evolution
    score_ep, score_vals = _extract_series("mppi_score")
    if not score_ep:
        score_ep, score_vals = _extract_series("score")
    cost_ep, cost_vals = _extract_series("mppi_cost")
    pos_pen_ep, pos_pen_vals = _extract_series("mppi_pos_penalty")
    vel_pen_ep, vel_pen_vals = _extract_series("mppi_vel_penalty")
    rot_pen_ep, rot_pen_vals = _extract_series("mppi_rot_penalty")

    if score_ep:
        axes[0, 0].plot(score_ep, score_vals, label="MPPI Score", color="#2563EB", marker="o", lw=2.2)
        best_idx = int(np.argmin(score_vals))
        axes[0, 0].plot(score_ep[best_idx], score_vals[best_idx], marker="*", markersize=14, color="#E11D48", label=f"Best: {score_vals[best_idx]:.3f} (Ep {score_ep[best_idx]})")
    elif cost_ep:
        axes[0, 0].plot(cost_ep, cost_vals, label="MPPI Cost", color="#2563EB", marker="o", lw=2.2)

    if pos_pen_ep:
        axes[0, 0].plot(pos_pen_ep, pos_pen_vals, label="Pos Penalty", color="#D97706", linestyle="--", alpha=0.8)
    if vel_pen_ep:
        axes[0, 0].plot(vel_pen_ep, vel_pen_vals, label="Vel Penalty", color="#DC2626", linestyle=":", alpha=0.8)
    if rot_pen_ep:
        axes[0, 0].plot(rot_pen_ep, rot_pen_vals, label="Rot Penalty", color="#7C3AED", linestyle="-.", alpha=0.8)

    axes[0, 0].set_title("MPPI Structural Score & Cost")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Score / Penalty")
    axes[0, 0].legend(loc="upper right")

    # Panel (0, 1): Autonomous Closed-Loop Success Rate (%)
    succ_ep, succ_vals = _extract_series("mppi_success_rate")
    if not succ_ep:
        succ_ep, succ_vals = _extract_series("success_rate")
    if succ_ep:
        pct_vals = [v * 100.0 if v <= 1.0 else v for v in succ_vals]
        axes[0, 1].plot(succ_ep, pct_vals, label="Success Rate (%)", color="#059669", marker="s", lw=2.2)
        axes[0, 1].axhline(80.0, color="#10B981", linestyle="--", lw=1.8, label="Target Spec (80%)")
        axes[0, 1].set_ylim(-5, 105)
    axes[0, 1].set_title("Closed-Loop Grasp Success Rate")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].legend(loc="lower right")

    # Panel (1, 0): Two-Stage Phasic Tracking & Grasp Accuracy (cm)
    grasp_ep, grasp_err = _extract_series("mppi_grasp_error_cm")
    if not grasp_ep:
        grasp_ep, grasp_err = _extract_series("grasp_error_cm")
    term_ep, term_err = _extract_series("mppi_terminal_error_cm")
    if not term_ep:
        term_ep, term_err = _extract_series("terminal_error_cm")
    mean_ep, mean_err = _extract_series("mppi_mean_error_cm")
    if not mean_ep:
        mean_ep, mean_err = _extract_series("mean_cartesian_err_cm")

    if grasp_ep:
        axes[1, 0].plot(grasp_ep, grasp_err, label="Stage 1: Grasp Contact (t_c)", color="#0284C7", marker="^", lw=2.0)
    if term_ep:
        axes[1, 0].plot(term_ep, term_err, label="Stage 2: Placement (t_grasp)", color="#E11D48", marker="v", lw=2.0)
    if mean_ep:
        axes[1, 0].plot(mean_ep, mean_err, label="Mean Cartesian Error", color="#64748B", linestyle="--", lw=1.8)

    axes[1, 0].axhline(3.0, color="#F59E0B", linestyle="--", lw=1.8, label="Grasp Snap Bound (3.0 cm)")
    axes[1, 0].set_title("Two-Stage Phasic Cartesian Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Error (cm)")
    axes[1, 0].legend(loc="upper right")

    # Panel (1, 1): Terminal Kinematic & Orientation Stability
    rot_ep, rot_vals = _extract_series("mppi_rot_err_rad")
    if not rot_ep:
        rot_ep, rot_vals = _extract_series("mppi_final_rot")
    vel_ep, vel_vals = _extract_series("mppi_final_vel")

    ax_rot = axes[1, 1]
    lines = []
    labels = []
    if rot_ep:
        l1 = ax_rot.plot(rot_ep, rot_vals, label="Terminal Rot Err (rad)", color="#7C3AED", marker="o", lw=2.0)
        l2 = ax_rot.axhline(0.05, color="#A78BFA", linestyle="--", lw=1.5, label="Rot Bound (0.05 rad)")
        lines.extend(l1)
        labels.append("Terminal Rot Err (rad)")
        lines.append(l2)
        labels.append("Rot Bound (0.05 rad)")
    ax_rot.set_xlabel("Epoch")
    ax_rot.set_ylabel("Orientation Error (rad)", color="#7C3AED")
    ax_rot.tick_params(axis="y", labelcolor="#7C3AED")

    if vel_ep:
        ax_vel = ax_rot.twinx()
        l3 = ax_vel.plot(vel_ep, vel_vals, label="Terminal Joint Vel (rad/s)", color="#DC2626", marker="x", linestyle="-.", lw=2.0)
        l4 = ax_vel.axhline(1.5, color="#F87171", linestyle=":", lw=1.5, label="Vel Bound (1.5 rad/s)")
        lines.extend(l3)
        labels.append("Terminal Joint Vel (rad/s)")
        lines.append(l4)
        labels.append("Vel Bound (1.5 rad/s)")
        ax_vel.set_ylabel("Velocity Norm (rad/s)", color="#DC2626")
        ax_vel.tick_params(axis="y", labelcolor="#DC2626")
        ax_vel.grid(False)

    ax_rot.set_title("Terminal Stability (Orientation & Velocity)")
    if lines:
        ax_rot.legend(lines, labels, loc="upper right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_mppi_benchmark_results(
    results: Sequence[Any],
    save_path: Union[str, Path],
) -> None:
    """
    Renders multi-panel benchmark diagnostic visualization for Latent MPPI control episodes.
    Works with both EpisodeResult objects and dictionary records.
    """
    set_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0F172A")

    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")
        ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    def _get(obj: Any, key: str) -> Any:
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    # Panel 1: Success Rate & Mean Error by Elevation Plane
    planes = ["Low", "Mid", "High"]
    succ_rates = []
    mean_errs = []
    for p in planes:
        subset = [r for r in results if _get(r, "elevation_plane") == p]
        if subset:
            succ_rates.append(float(np.mean([1.0 if _get(r, "grasp_success") else 0.0 for r in subset]) * 100.0))
            mean_errs.append(float(np.mean([_get(r, "min_placement_error_cm") for r in subset])))
        else:
            succ_rates.append(0.0)
            mean_errs.append(0.0)

    x = np.arange(len(planes))
    width = 0.35
    axes[0].bar(x - width / 2, succ_rates, width, label="Success Rate (%)", color="#38BDF8")
    axes[0].bar(x + width / 2, mean_errs, width, label="Mean Error (cm)", color="#F472B6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(planes)
    axes[0].set_ylabel("Metric Value")
    axes[0].set_title("Performance Across Elevation Planes")
    axes[0].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC")
    axes[0].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    # Panel 2: Placement Error vs Payload Mass (1.20 - 4.50 kg)
    masses = [float(_get(r, "payload_mass")) for r in results]
    errors = [float(_get(r, "min_placement_error_cm")) for r in results]
    success_flags = [bool(_get(r, "grasp_success")) for r in results]

    colors = ["#34D399" if s else "#F87171" for s in success_flags]
    axes[1].scatter(masses, errors, c=colors, s=40, alpha=0.85, edgecolors="#0F172A")
    axes[1].axhline(3.0, color="#F59E0B", linestyle="--", linewidth=1.8, label="Grasp Snap Bound (3.0 cm)")
    axes[1].set_xlabel("Payload Mass (kg)")
    axes[1].set_ylabel("Min Placement Error (cm)")
    axes[1].set_title("Terminal Error vs. Payload Mass (1.2 - 4.5 kg)")
    axes[1].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC")
    axes[1].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    # Panel 3: MPPI Inference Latency Distribution
    latencies = [float(_get(r, "mean_latency_ms")) for r in results]
    axes[2].hist(latencies, bins=15, color="#A78BFA", edgecolor="#0F172A", alpha=0.85)
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    axes[2].axvline(mean_lat, color="#38BDF8", linestyle="--", linewidth=1.8, label=f"Mean: {mean_lat:.1f} ms")
    axes[2].set_xlabel("MPPI Step Latency (ms)")
    axes[2].set_ylabel("Count")
    axes[2].set_title("MPPI Inference Latency")
    axes[2].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC")
    axes[2].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    plt.tight_layout()
    target_path = Path(save_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(target_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
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


# ---------------------------------------------------------------------------
# Unity Ghost Execution Telemetry Export
# ---------------------------------------------------------------------------

GHOST_SCHEMA_VERSION = "spwm.ghost_trajectory/1.0"
DEFAULT_GHOST_FILENAME = "trajectory_ghost.json"
DEFAULT_UNITY_PROJECT_DIR = Path(__file__).resolve().parents[2] / "sim"


def _finite(x: float, ndigits: int = 6) -> float:
    """Unity's JsonUtility cannot parse NaN/Infinity; clamp to a finite rounded float."""
    x = float(x)
    if not math.isfinite(x):
        return 0.0
    return round(x, ndigits)


def _vec(arr: np.ndarray, ndigits: int = 6) -> List[float]:
    return [_finite(v, ndigits) for v in np.asarray(arr, dtype=np.float64).ravel()]


@dataclass
class GhostTrajectory:
    """In-memory representation of the ghost-execution JSON contract (one trajectory)."""
    metadata: Dict[str, Any]
    frames: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"metadata": self.metadata, "frames": self.frames}


class GhostVisualizerExporter:
    """Serializes one autonomous M1 rollout vs. ground truth for Unity ghost playback.

    Hooks into `RolloutEvaluator.rollout_trajectory`, so the exported trajectory is produced by
    the exact same burn-in / decoder warm-up / `rollout_latent` protocol as the benchmark metrics.

    Output contract (`trajectory_ghost_v0_4_0.json`, written to `<unity>/Assets/StreamingAssets`):
        metadata: schema, model_version, horizon (H), dt, joint_count, t_start, sample ids, summary.
        frames[H]: {
            t, time_s,
            ground_truth_q[J], predicted_q[J],            # joint angles (rad)
            ground_truth_p[J], predicted_p[J],            # joint angular velocities (rad/s)
            instantaneous_metrics: {
                position_abs_error,                       # ||q_hat_t - q*_t||_2
                velocity_abs_error,                       # ||p_hat_t - p*_t||_2
                velocity_relative_degradation,            # E_v(t) / E_v(1)  (metrics_resume convention)
                position_relative_degradation,            # E_q(t) / E_q(1)
            }
        }

    Frame `t` (1-indexed) is the prediction for simulation index `t_start + t`.
    """

    def __init__(
        self,
        evaluator: RolloutEvaluator,
        unity_project_dir: Union[str, Path] = DEFAULT_UNITY_PROJECT_DIR,
        filename: str = DEFAULT_GHOST_FILENAME,
        model_version: str = "spwm",
        model_version_b: Optional[str] = None,
        dt: float = 0.05,
        horizon: int = 100,
        joint_names: Optional[Sequence[str]] = None,
        precision: int = 6,
        pre_contact_sensory_steps: int = 7,
        post_contact_sensory_steps: int = 3,
    ) -> None:
        if horizon <= 0:
            raise ValueError(f"GhostVisualizerExporter: horizon must be positive, got {horizon}")
        if dt <= 0:
            raise ValueError(f"GhostVisualizerExporter: dt must be positive, got {dt}")
        self.evaluator = evaluator
        self.unity_project_dir = Path(unity_project_dir)
        self.filename = filename
        self.model_version = model_version
        self.model_version_b = model_version_b
        self.dt = float(dt)
        self.horizon = int(horizon)
        self.joint_names = list(joint_names) if joint_names is not None else None
        self.precision = int(precision)
        self.pre_contact_sensory_steps = int(pre_contact_sensory_steps)
        self.post_contact_sensory_steps = int(post_contact_sensory_steps)

    @property
    def output_path(self) -> Path:
        return self.unity_project_dir / "Assets" / "StreamingAssets" / self.filename

    # -- capture -----------------------------------------------------------------

    def capture_batch(
        self,
        batch: Mapping[str, Any],
        sample_index: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
        with_active_perception: bool = True,
        evaluator_b: Optional[RolloutEvaluator] = None,
    ) -> GhostTrajectory:
        """Run the benchmark rollout on `batch` and build the ghost payload for one sample."""
        traj = self.evaluator.rollout_trajectory(batch, horizon=self.horizon, with_active_perception=with_active_perception)

        traj_b = None
        if evaluator_b is not None:
            traj_b = evaluator_b.rollout_trajectory(batch, horizon=self.horizon, with_active_perception=with_active_perception)

        b = traj.target_kinematics.shape[0]

        if not 0 <= sample_index < b:
            raise IndexError(f"sample_index {sample_index} out of range for batch of size {b}")
        if traj.horizon < self.horizon:
            raise ValueError(
                f"Sequence only supports H={traj.horizon} < requested H={self.horizon}. "
                f"Increase sequence_length to >= temporal_burn_in + H + 1."
            )
        meta = dict(extra_metadata or {})
        meta["has_sensory_injection"] = bool(with_active_perception)
        if "trajectory_ids" in batch:
            meta.setdefault("trajectory_id", int(batch["trajectory_ids"][sample_index]))
        if "phase_flags" in batch and with_active_perception:
            pf = batch["phase_flags"][sample_index]
            # Find the transition from approach (flag == 0) to heft / lift (flag == 1)
            # Offset by traj.t_start if evaluated past burn-in
            t_start = int(traj.t_start)
            h = min(self.horizon, traj.horizon)
            sub_pf = pf[t_start : t_start + h].detach().cpu().numpy()
            heft_indices = np.where(sub_pf == 1)[0]
            if len(heft_indices) > 0:
                first_heft = int(heft_indices[0])
                meta["heft_start_frame_index"] = first_heft
                contact_idx = min(first_heft + self.pre_contact_sensory_steps, int(heft_indices[-1]))
                meta["grasp_lock_frame_index"] = contact_idx
                sensory_rel_idx = min(contact_idx + self.post_contact_sensory_steps, int(heft_indices[-1]))
                meta["sensory_release_frame_index"] = sensory_rel_idx
        return self.from_trajectory_rollout(traj, sample_index=sample_index, extra_metadata=meta, traj_b=traj_b)

    def capture(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        batch_index: int = 0,
        sample_index: int = 0,
        with_active_perception: bool = True,
        evaluator_b: Optional[RolloutEvaluator] = None,
    ) -> GhostTrajectory:
        """Select batch `batch_index` from `data_loader` and capture sample `sample_index`."""
        for i, batch in enumerate(data_loader):
            if i == batch_index:
                return self.capture_batch(batch, sample_index, extra_metadata={"batch_index": batch_index}, with_active_perception=with_active_perception, evaluator_b=evaluator_b)
        raise IndexError(f"batch_index {batch_index} exceeds data loader length")

    def from_trajectory_rollout(
        self,
        traj: TrajectoryRollout,
        sample_index: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
        traj_b: Optional[TrajectoryRollout] = None,
    ) -> GhostTrajectory:
        """Pure serialization of a `TrajectoryRollout` (no model execution)."""
        j = traj.joint_count
        h = min(self.horizon, traj.horizon)
        pred = traj.rollout.decoded_kinematics[sample_index, :h].detach().float().cpu().numpy()
        true = traj.target_kinematics[sample_index, :h].detach().float().cpu().numpy()
        pred_z = traj.rollout.trajectory[sample_index, :h].detach().float().cpu().numpy()
        true_z = traj.target_latents[sample_index, :h].detach().float().cpu().numpy()
        pred_b_q = None
        pred_b_p = None
        pred_b_z = None
        if traj_b is not None:
            pred_b_z = traj_b.rollout.trajectory[sample_index, :h].detach().float().cpu().numpy()
            pred_b = traj_b.rollout.decoded_kinematics[sample_index, :h].detach().float().cpu().numpy()
            pred_b_q = pred_b[:, :j]
            pred_b_p = pred_b[:, j : 2 * j]

        return self.from_arrays(
            predicted_q=pred[:, :j],
            ground_truth_q=true[:, :j],
            predicted_p=pred[:, j : 2 * j],
            ground_truth_p=true[:, j : 2 * j],
            predicted_b_q=pred_b_q,
            predicted_b_p=pred_b_p,
            predicted_z=pred_z,
            ground_truth_z=true_z,
            predicted_b_z=pred_b_z,
            extra_metadata={"t_start": int(traj.t_start), "sample_index": int(sample_index), **(extra_metadata or {})},
        )

    def from_arrays(
        self,
        predicted_q: np.ndarray,
        ground_truth_q: np.ndarray,
        predicted_p: Optional[np.ndarray] = None,
        ground_truth_p: Optional[np.ndarray] = None,
        predicted_z: Optional[np.ndarray] = None,
        ground_truth_z: Optional[np.ndarray] = None,
        predicted_b_q: Optional[np.ndarray] = None,
        predicted_b_p: Optional[np.ndarray] = None,
        predicted_b_z: Optional[np.ndarray] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> GhostTrajectory:
        """Build the payload from raw [H, J] arrays. Metrics follow `spwm.training.metrics` (per-step L2)."""
        q_hat = np.asarray(predicted_q, dtype=np.float64)
        q_star = np.asarray(ground_truth_q, dtype=np.float64)
        if q_hat.shape != q_star.shape or q_hat.ndim != 2:
            raise ValueError(f"q arrays must share shape [H, J]; got {q_hat.shape} vs {q_star.shape}")
        h, j = q_hat.shape
        has_p = predicted_p is not None and ground_truth_p is not None
        p_hat = np.asarray(predicted_p, dtype=np.float64) if has_p else np.zeros_like(q_hat)
        p_star = np.asarray(ground_truth_p, dtype=np.float64) if has_p else np.zeros_like(q_hat)

        pos_err = np.linalg.norm(q_hat - q_star, axis=-1)          # [H]
        vel_err = np.linalg.norm(p_hat - p_star, axis=-1)          # [H]
        has_z = predicted_z is not None and ground_truth_z is not None
        if has_z:
            z_hat = np.asarray(predicted_z, dtype=np.float64)
            z_star = np.asarray(ground_truth_z, dtype=np.float64)
            lat_err = np.linalg.norm(z_hat - z_star, axis=-1)       # [H] L2 norm
        else:
            lat_err = np.zeros(h, dtype=np.float64)
        pos_err_b = vel_err_b = lat_err_b = None
        if predicted_b_q is not None:
            pos_err_b = np.linalg.norm(np.asarray(predicted_b_q, dtype=np.float64) - q_star, axis=-1)
            if has_p and predicted_b_p is not None:
                vel_err_b = np.linalg.norm(np.asarray(predicted_b_p, dtype=np.float64) - p_star, axis=-1)
            if has_z and predicted_b_z is not None:
                lat_err_b = np.linalg.norm(np.asarray(predicted_b_z, dtype=np.float64) - z_star, axis=-1)
        pos_ref = pos_err[0] if pos_err[0] > 1e-8 else 1.0
        vel_ref = vel_err[0] if vel_err[0] > 1e-8 else 1.0
        pos_rel = pos_err / pos_ref
        vel_rel = vel_err / vel_ref

        nd = self.precision
        frames: List[Dict[str, Any]] = []
        for t in range(h):
            frame: Dict[str, Any] = {
                "t": t + 1,
                "time_s": _finite((t + 1) * self.dt, nd),
                "ground_truth_q": _vec(q_star[t], nd),
                "predicted_q": _vec(q_hat[t], nd),
                "ground_truth_p": _vec(p_star[t], nd),
                "predicted_p": _vec(p_hat[t], nd),
                "instantaneous_metrics": {
                    "latent_error": _finite(lat_err[t], nd),
                    "position_abs_error": _finite(pos_err[t], nd),
                    "velocity_abs_error": _finite(vel_err[t], nd),
                    "velocity_relative_degradation": _finite(vel_rel[t], nd),
                    "position_relative_degradation": _finite(pos_rel[t], nd),
                },
            }
            if pos_err_b is not None:
                frame["instantaneous_metrics"]["position_abs_error_b"] = _finite(pos_err_b[t], nd)
                if vel_err_b is not None:
                    frame["instantaneous_metrics"]["velocity_abs_error_b"] = _finite(vel_err_b[t], nd)
                if lat_err_b is not None:
                    frame["instantaneous_metrics"]["latent_error_b"] = _finite(lat_err_b[t], nd)
            if predicted_b_q is not None:
                frame["predicted_b_q"] = _vec(predicted_b_q[t], nd)
            if predicted_b_p is not None:
                frame["predicted_b_p"] = _vec(predicted_b_p[t], nd)
            frames.append(frame)

        names = self.joint_names if self.joint_names and len(self.joint_names) == j else [f"joint_{i}" for i in range(j)]
        metadata: Dict[str, Any] = {
            "schema": GHOST_SCHEMA_VERSION,
            "model_version": self.model_version,
            "horizon": int(h),
            "dt": _finite(self.dt, 6),
            "joint_count": int(j),
            "joint_names": names,
            "angle_unit": "rad",
            "has_velocities": bool(has_p),
            "t_start": 0,
            "sample_index": 0,
            "trajectory_id": -1,
            "batch_index": 0,
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": {
                "mean_latent_error": _finite(lat_err.mean(), nd),
                "final_latent_error": _finite(lat_err[-1], nd),
                "max_latent_error": _finite(lat_err.max(), nd),
                "mean_position_abs_error": _finite(pos_err.mean(), nd),
                "final_position_abs_error": _finite(pos_err[-1], nd),
                "max_position_abs_error": _finite(pos_err.max(), nd),
                "mean_velocity_abs_error": _finite(vel_err.mean(), nd),
                "final_velocity_abs_error": _finite(vel_err[-1], nd),
                "max_velocity_abs_error": _finite(vel_err.max(), nd),
                "final_velocity_relative_degradation": _finite(vel_rel[-1], nd),
                "final_position_relative_degradation": _finite(pos_rel[-1], nd),
            },
        }
        if predicted_b_q is not None:
            if self.model_version_b is None:
                raise ValueError("Dual-model export requires model_version_b so the HUD labels the correct model.")
            metadata["has_dual_models"] = True
            metadata["model_version_b"] = self.model_version_b
        metadata.update(extra_metadata or {})
        return GhostTrajectory(metadata=metadata, frames=frames)

    # -- persistence -------------------------------------------------------------

    def save(self, ghost: GhostTrajectory, path: Optional[Union[str, Path]] = None) -> Path:
        """Write the payload as JSON (default: `<unity>/Assets/StreamingAssets/<filename>`)."""
        target = Path(path) if path is not None else self.output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ghost.to_dict(), f, separators=(",", ":"), allow_nan=False)
        tmp.replace(target)  # atomic swap so Unity never reads a half-written file
        return target

    def export(
        self,
        data_loader: Iterable[Mapping[str, Any]],
        batch_index: int = 0,
        sample_index: int = 0,
        path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Capture + save in one call. Returns the written JSON path."""
        return self.save(self.capture(data_loader, batch_index, sample_index), path)



