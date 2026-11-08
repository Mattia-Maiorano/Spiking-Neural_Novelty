"""
Figure 4: Multi-Horizon Stability & Drift Ablation (CoRL Publication Asset).

Generates:
  1. Grouped log-scale velocity MSE error accumulation across prediction horizons H in {1, 10, 50, 100},
     contrasting SPWM against Continuous RSSM, Deep GRU, Continuous ConvLSTM, and ANN-Rate.
  2. Qualitative 3D scene capture at horizon H=100 displaying Ground-Truth payload,
     strictly aligned SPWM prediction marker, and continuous rate baseline prediction
     computed directly from real model rollouts and forward kinematics.

Zero AST Fallbacks strictly enforced. Multi-format vector (.pdf, .svg) and 300+ DPI (.png) export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from figure_style import PALETTE, save_publication_figure, set_paper_style
from spwm.contracts.config import ExperimentConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import EventWorldDataset
from spwm.models.ann_ablation import ANNWorldModel
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.world_model import SPWM


def load_comparative_metrics(
    baselines_json: Path = Path("results/baselines/comparative_metrics.json"),
    ablation_json: Path = Path("results/ann_ablation/ablation_metrics.json"),
) -> Dict[str, Dict[str, float]]:
    """Loads exact verified velocity MSE metrics from serialized benchmarks."""
    with open(baselines_json, "r", encoding="utf-8") as f:
        b_data = json.load(f)
    with open(ablation_json, "r", encoding="utf-8") as f:
        a_data = json.load(f)

    spwm_tag = GLOBAL_VERSION.formatted if GLOBAL_VERSION.formatted in b_data else [k for k in b_data.keys() if "SPWM" in k][0]
    return {
        f"{GLOBAL_VERSION.formatted} (Ours)": {h: float(b_data[spwm_tag]["velocity_mse_per_horizon"][h]) for h in ["1", "10", "50", "100"]},
        "Continuous RSSM": {h: float(b_data["Continuous RSSM"]["velocity_mse_per_horizon"][h]) for h in ["1", "10", "50", "100"]},
        "Deep GRU": {h: float(b_data["Deep GRU"]["velocity_mse_per_horizon"][h]) for h in ["1", "10", "50", "100"]},
        "Continuous ConvLSTM": {h: float(b_data["Continuous ConvLSTM"]["velocity_mse_per_horizon"][h]) for h in ["1", "10", "50", "100"]},
        "ANN-Rate (Continuous Leaky)": {h: float(a_data["ANN-Rate (Continuous Leaky)"]["velocity_mse_per_horizon"][h]) for h in ["1", "10", "50", "100"]},
    }


def compute_h100_spatial_states(
    config_path: Path,
    spwm_ckpt: Path,
    ann_ckpt: Path,
    trajectories_dir: Path,
    trajectory_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Computes real Cartesian spatial states at horizon H=100 using Differentiable Forward Kinematics."""
    cfg = ExperimentConfig.load(config_path)
    spwm = SPWM.from_config(cfg)
    raw_sp = torch.load(spwm_ckpt, map_location="cpu", weights_only=False)
    sd_sp = raw_sp["model_state"] if "model_state" in raw_sp else raw_sp["model_state_dict"]["model_state"]
    spwm.load_state_dict(sd_sp, strict=True)
    spwm.eval()

    ann = ANNWorldModel.from_config(cfg)
    raw_ann = torch.load(ann_ckpt, map_location="cpu", weights_only=False)
    sd_ann = raw_ann["model_state"] if "model_state" in raw_ann else raw_ann["model_state_dict"]["model_state"]
    ann.load_state_dict(sd_ann, strict=True)
    ann.eval()

    ds = EventWorldDataset([trajectory_id], dataset_dir=trajectories_dir)
    sample = ds[0]
    events = sample["events"].unsqueeze(0)
    actions = sample["actions"].unsqueeze(0)
    flags = sample["phase_flags"].unsqueeze(0)

    with torch.no_grad():
        res_sp = spwm.rollout(events=events, actions=actions, horizon=100, t_start=0, phase_flags=flags)
        res_ann = ann.rollout(events=events, actions=actions, horizon=100, t_start=0, phase_flags=flags)

    fk = DifferentiableForwardKinematics()
    gt_step_idx = min(100, sample["positions"].shape[0] - 1)
    gt_pos = fk(sample["positions"][gt_step_idx].unsqueeze(0))[0].numpy()

    assert res_sp.decoded_kinematics is not None, "SPWM decoded_kinematics is required"
    assert res_ann.decoded_kinematics is not None, "ANN decoded_kinematics is required"
    spwm_pos = fk(res_sp.decoded_kinematics[:, -1, :6])[0].numpy()
    ann_pos = fk(res_ann.decoded_kinematics[:, -1, :6])[0].numpy()

    spwm_err_cm = float(np.linalg.norm(spwm_pos - gt_pos) * 100.0)
    ann_err_cm = float(np.linalg.norm(ann_pos - gt_pos) * 100.0)

    return gt_pos, spwm_pos, ann_pos, spwm_err_cm, ann_err_cm


def draw_3d_box_on_surface(
    ax: plt.Axes,
    center_xy: Tuple[float, float],
    surface_z: float,
    size: float,
    color: str,
    alpha: float,
    edge_color: str = "#1E293B",
    z_offset: float = 0.0,
) -> None:
    """Renders a 3D payload cube resting on top of a specified surface elevation."""
    cx, cy = center_xy
    s = size / 2.0
    z_bot = surface_z + z_offset
    z_top = z_bot + size

    vertices = np.array([
        [cx - s, cy - s, z_bot],
        [cx + s, cy - s, z_bot],
        [cx + s, cy + s, z_bot],
        [cx - s, cy + s, z_bot],
        [cx - s, cy - s, z_top],
        [cx + s, cy - s, z_top],
        [cx + s, cy + s, z_top],
        [cx - s, cy + s, z_top],
    ])

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
    ]

    poly = Poly3DCollection(faces, facecolors=color, alpha=alpha, edgecolors=edge_color, linewidths=0.9)
    ax.add_collection3d(poly)


def render_figure_4(
    output_dir: Path,
    baselines_json: Path = Path("results/baselines/comparative_metrics.json"),
    ablation_json: Path = Path("results/ann_ablation/ablation_metrics.json"),
    config_path: Path = GLOBAL_VERSION.m1_config(),
    spwm_ckpt: Path = GLOBAL_VERSION.m1_checkpoint(),
    ann_ckpt: Path = Path("results/ann_ablation/ann_rate_m1/model.pt"),
    trajectories_dir: Path = Path("sim/ExportedTrajectories"),
) -> None:
    """Renders Figure 4: Multi-Horizon Stability & Drift Ablation."""
    set_paper_style()

    print("[Fig 4] Loading comparative metrics and rendering stability figures...")
    metrics = load_comparative_metrics(baselines_json, ablation_json)

    fig = plt.figure(figsize=(13.2, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1.0], wspace=0.26)

    # --------------------------------------------------------------------------
    # Panel (a): Grouped Log-Scale Multi-Horizon Velocity MSE Error Accumulation
    # --------------------------------------------------------------------------
    ax_err = fig.add_subplot(gs[0, 0])
    ax_err.set_title(
        "(a) Velocity Error Accumulation Across Horizons $H \in \{1, 10, 50, 100\}$\n"
        "SPWM Attractor Basin vs. Compounding Exposure Bias in Continuous Baselines",
        fontsize=9.5, pad=14, weight="bold",
    )

    horizons = ["1", "10", "50", "100"]
    x = np.arange(len(horizons))
    width = 0.16

    model_configs = [
        (f"{GLOBAL_VERSION.formatted} (Ours)", PALETTE["spwm"], "-", 0),
        ("Continuous RSSM", PALETTE["continuous_rssm"], "--", 1),
        ("Deep GRU", PALETTE["deep_gru"], ":", 2),
        ("Continuous ConvLSTM", PALETTE["conv_lstm"], "-.", 3),
        ("ANN-Rate (Continuous Leaky)", PALETTE["ann_rate"], "--", 4),
    ]

    for model_name, color, ls, offset_idx in model_configs:
        vals = [metrics[model_name][h] for h in horizons]
        pos = x + (offset_idx - 2) * width
        ax_err.bar(
            pos, vals, width,
            label=model_name,
            color=color, edgecolor="#1E293B", linewidth=0.7,
            alpha=0.92 if "SPWM" in model_name else 0.82,
        )

    ax_err.set_yscale("log")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(["$H=1$", "$H=10$", "$H=50$", "$H=100$"], fontsize=9.2)
    ax_err.set_ylabel("Velocity MSE (log-scale, $\\mathrm{rad}^2/\\mathrm{s}^2$)", fontsize=9.2)
    ax_err.set_ylim(1e-4, 1.0)
    ax_err.grid(True, which="both", axis="y", linestyle=":", alpha=0.6)

    # Dynamic Callout Annotations from verified data
    spwm_h100 = metrics[f"{GLOBAL_VERSION.formatted} (Ours)"]["100"]
    rssm_h100 = metrics["Continuous RSSM"]["100"]
    gru_h100 = metrics["Deep GRU"]["100"]
    ann_h100 = metrics["ANN-Rate (Continuous Leaky)"]["100"]
    ann_h1 = metrics["ANN-Rate (Continuous Leaky)"]["1"]
    ann_drift_pct = (ann_h100 - ann_h1) / ann_h1 * 100.0

    ax_err.annotate(
        f"SPWM: ${spwm_h100:.4f}$ MSE\n${rssm_h100 / spwm_h100:.1f}\\times$ lower than RSSM\n${gru_h100 / spwm_h100:.1f}\\times$ lower than GRU",
        xy=(3 + (0 - 2) * width, spwm_h100),
        xytext=(2.15, 0.0004),
        arrowprops=dict(arrowstyle="->", color=PALETTE["spwm"], lw=1.2),
        fontsize=7.8, color="#1E3A8A", weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=0.8),
    )

    ax_err.annotate(
        f"ANN-Rate: $+{ann_drift_pct:.0f}\\%$ Drift\nCompounding velocity divergence",
        xy=(3 + (4 - 2) * width, ann_h100),
        xytext=(2.1, 0.22),
        arrowprops=dict(arrowstyle="->", color=PALETTE["ann_rate"], lw=1.2),
        fontsize=7.8, color="#991B1B", weight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#FCA5A5", lw=0.8),
    )

    ax_err.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7.2)

    # --------------------------------------------------------------------------
    # Panel (b): Qualitative 3D Scene Visualization at Horizon H=100
    # --------------------------------------------------------------------------
    ax_scene = fig.add_subplot(gs[0, 1], projection="3d")
    ax_scene.set_title(
        "(b) Qualitative Spatial State at Horizon $H=100$\n"
        "Strict Spatial Alignment (SPWM) vs. Compounding Offset (Continuous)",
        fontsize=9.5, pad=14, weight="bold",
    )

    # Compute real spatial states from model rollouts
    gt_pos, spwm_pos, ann_pos, spwm_dist_cm, ann_dist_cm = compute_h100_spatial_states(
        config_path=config_path,
        spwm_ckpt=spwm_ckpt,
        ann_ckpt=ann_ckpt,
        trajectories_dir=trajectories_dir,
        trajectory_id=0,
    )

    # Table Surface Definition
    table_z = 1.05
    xt = np.linspace(-0.35, 0.25, 10)
    yt = np.linspace(0.35, 0.75, 10)
    XT, YT = np.meshgrid(xt, yt)
    ZT = np.full_like(XT, table_z)
    ax_scene.plot_surface(XT, YT, ZT, color="#F8FAFC", alpha=0.75, edgecolor="#CBD5E1", linewidth=0.5)

    # Table Perimeter Edge
    ax_scene.plot([-0.35, 0.25, 0.25, -0.35, -0.35], [0.35, 0.35, 0.75, 0.75, 0.35], [table_z, table_z, table_z, table_z, table_z],
                  color="#DC2626", linewidth=1.5, linestyle="--")
    ax_scene.text(0.0, 0.33, table_z, "Table Perimeter Edge", fontsize=7.0, color="#991B1B", style="italic")

    box_size = 0.06

    # 1. Ground Truth Payload (Real FK Coordinates: X, Z depth, Y elevation)
    gt_xy = (float(gt_pos[0]), float(gt_pos[2]))
    gt_elev = float(gt_pos[1])
    draw_3d_box_on_surface(ax_scene, gt_xy, surface_z=gt_elev - box_size / 2.0, size=box_size, color="#10B981", alpha=0.95, edge_color="#065F46", z_offset=0.0)
    ax_scene.scatter([gt_xy[0]], [gt_xy[1]], [gt_elev], color="#065F46", s=65, marker="o", label=r"Ground Truth Payload $\mathbf{x}_{\mathrm{GT}}^*$")

    # 2. SPWM Prediction Marker (Real FK Coordinates)
    spwm_xy = (float(spwm_pos[0]), float(spwm_pos[2]))
    spwm_elev = float(spwm_pos[1])
    draw_3d_box_on_surface(ax_scene, spwm_xy, surface_z=spwm_elev - box_size / 2.0, size=box_size, color=PALETTE["spwm"], alpha=0.80, edge_color="#1E3A8A", z_offset=0.0)
    ax_scene.scatter([spwm_xy[0]], [spwm_xy[1]], [spwm_elev], color=PALETTE["spwm"], s=75, marker="^", label=f"SPWM Prediction ($\Delta = {spwm_dist_cm:.1f}\\,\\mathrm{{cm}}$, Bounded)")

    # 3. Continuous Rate Model Prediction (Real FK Coordinates)
    ann_xy = (float(ann_pos[0]), float(ann_pos[2]))
    ann_elev = float(ann_pos[1])
    draw_3d_box_on_surface(ax_scene, ann_xy, surface_z=ann_elev - box_size / 2.0, size=box_size, color=PALETTE["ann_rate"], alpha=0.85, edge_color="#991B1B", z_offset=0.0)
    ax_scene.scatter([ann_xy[0]], [ann_xy[1]], [ann_elev], color=PALETTE["ann_rate"], s=85, marker="X", label=f"Continuous Rate ($\Delta = {ann_dist_cm:.1f}\\,\\mathrm{{cm}}$, Divergent)")

    # Displacement indicator lines
    gt_center = np.array([gt_xy[0], gt_xy[1], gt_elev])
    spwm_center = np.array([spwm_xy[0], spwm_xy[1], spwm_elev])
    ann_center = np.array([ann_xy[0], ann_xy[1], ann_elev])

    ax_scene.plot([gt_center[0], spwm_center[0]], [gt_center[1], spwm_center[1]], [gt_center[2], spwm_center[2]], color=PALETTE["spwm"], linewidth=1.8, linestyle=":")
    ax_scene.plot([gt_center[0], ann_center[0]], [gt_center[1], ann_center[1]], [gt_center[2], ann_center[2]], color=PALETTE["ann_rate"], linewidth=1.8, linestyle="--")

    # Callout text
    ax_scene.text(spwm_xy[0] + 0.03, spwm_xy[1], spwm_elev + box_size + 0.03, "SPWM: Bounded\nAttractor Orbit", fontsize=7.2, color="#1E3A8A", weight="bold")
    ax_scene.text(ann_xy[0] - 0.04, ann_xy[1] - 0.04, ann_elev + 0.08, "ANN: Drift Offset\n(Exposure Bias)", fontsize=7.2, color="#991B1B", weight="bold")

    ax_scene.set_xlabel("$X_{\\mathrm{world}}$ (m)", labelpad=4)
    ax_scene.set_ylabel("$Z_{\\mathrm{world}}$ Depth (m)", labelpad=4)
    ax_scene.set_zlabel("$Y_{\\mathrm{world}}$ Elevation (m)", labelpad=4)
    ax_scene.view_init(elev=22, azim=-55)
    ax_scene.set_xlim(-0.35, 0.25)
    ax_scene.set_ylim(0.25, 0.75)
    ax_scene.set_zlim(0.95, 1.35)
    ax_scene.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=7.0)

    save_publication_figure(fig, output_dir / "fig4_stability")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoRL Figure 4: Multi-Horizon Stability & Drift Ablation.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for exported figures")
    parser.add_argument("--baselines-json", type=Path, default=Path("results/baselines/comparative_metrics.json"))
    parser.add_argument("--ablation-json", type=Path, default=Path("results/ann_ablation/ablation_metrics.json"))
    parser.add_argument("--config", type=Path, default=GLOBAL_VERSION.m1_config())
    parser.add_argument("--spwm-ckpt", type=Path, default=GLOBAL_VERSION.m1_checkpoint())
    parser.add_argument("--ann-ckpt", type=Path, default=Path("results/ann_ablation/ann_rate_m1/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("sim/ExportedTrajectories"))
    args = parser.parse_args()

    render_figure_4(
        output_dir=args.output_dir,
        baselines_json=args.baselines_json,
        ablation_json=args.ablation_json,
        config_path=args.config,
        spwm_ckpt=args.spwm_ckpt,
        ann_ckpt=args.ann_ckpt,
        trajectories_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
