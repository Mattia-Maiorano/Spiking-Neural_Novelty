"""
Figure 5: Hardware Efficiency & Physical Domain Boundaries (CoRL Publication Asset).

Generates a dual-panel evaluation asset:
  - Panel A (Compute Efficiency): Horizontal comparative metric displaying energy-per-step (nJ/step)
    and dynamic power consumption (mW at 20 Hz) across Nvidia Jetson Orin edge GPU execution
    versus Intel Loihi-2 neuromorphic event-driven execution.
  - Panel B (Failure Boundary Matrix): 2D sensitivity matrix mapping downstream Latent MPPI task grasp
    success rate (%) across evaluated conditions (nominal, sensory noise, friction drift, payload extrapolation)
    and stratified slices (elevation planes and mass tiers), loaded directly from verified benchmark outputs.

Zero AST Fallbacks strictly enforced. Pure vector (.pdf, .svg) and 300+ DPI (.png) export.
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
import numpy as np

from figure_style import PALETTE, save_publication_figure, set_paper_style


def load_hardware_and_boundary_metrics(
    energy_json: Path = Path("results/energy_profiling/energy_profiling_metrics.json"),
    robustness_json: Path = Path("results/robustness_benchmark/robustness_metrics.json"),
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, List[str], List[str]]:
    """Loads exact verified hardware profiling and robustness matrix metrics."""
    with open(energy_json, "r", encoding="utf-8") as f:
        e_data = json.load(f)
    with open(robustness_json, "r", encoding="utf-8") as f:
        r_data = json.load(f)

    # 1. Panel A metrics: Energy per step & Power
    hw_metrics: Dict[str, Dict[str, float]] = {}
    for name in [
        "SPWM-v0.7.3 (Prod)",
        "SNN-ALIF (Matched)",
        "ANN-Rate (Ablation)",
        "Continuous RSSM",
        "Deep GRU",
        "Continuous ConvLSTM",
    ]:
        rec = e_data[name]
        hw_metrics[name] = {
            "energy_neuro_nj": float(rec["energy_neuro_nj"]) if rec["energy_neuro_nj"] is not None else 0.0,
            "energy_gpu_nj": float(rec["energy_gpu_nj"]),
            "power_mw": float(rec["power_20hz_mw"]),
            "ratio": float(rec["energy_ratio_vs_gpu"]),
        }

    # 2. Panel B metrics: 2D Sensitivity Matrix from real MPPI benchmark suite
    mppi_suite = r_data["mppi_suite"]
    cond_labels: List[str] = []
    col_labels = [
        "Overall\nMean",
        "Low Plane\nElevation",
        "Mid Plane\nElevation",
        "High Plane\nElevation",
        "Light Mass\n(1.2-2.3 kg)",
        "Medium Mass\n(2.3-3.4 kg)",
        "Heavy Mass\n(3.4-4.5+ kg)",
    ]

    mat_rows: List[List[float]] = []
    for m in mppi_suite:
        name = m["condition_name"]
        if "Nominal" in name:
            short_name = "Nominal Baseline\n(Clean Telemetry)"
        elif "Sensory Noise" in name:
            short_name = "Sensory Noise\n(1.0% Background)"
        elif "0.80x" in name:
            short_name = "Actuator Drift\n(0.80x Friction)"
        elif "1.20x" in name:
            short_name = "Actuator Drift\n(1.20x Friction)"
        elif "Payload" in name:
            short_name = "Payload Extrap.\n(4.80 kg OOD)"
        else:
            short_name = name

        cond_labels.append(short_name)
        row = [
            float(m["grasp_success_rate_pct"]),
            float(m["elevation_plane_breakdown"]["Low"]["success_rate_pct"]),
            float(m["elevation_plane_breakdown"]["Mid"]["success_rate_pct"]),
            float(m["elevation_plane_breakdown"]["High"]["success_rate_pct"]),
            float(m["mass_tier_breakdown"]["Light (1.2 - 2.3 kg)"]["success_rate_pct"]),
            float(m["mass_tier_breakdown"]["Medium (2.3 - 3.4 kg)"]["success_rate_pct"]),
            float(m["mass_tier_breakdown"]["Heavy (3.4 - 4.5+ kg)"]["success_rate_pct"]),
        ]
        mat_rows.append(row)

    matrix = np.array(mat_rows, dtype=np.float64)
    return hw_metrics, matrix, col_labels, cond_labels


def render_figure_5(
    output_dir: Path,
    energy_json: Path = Path("results/energy_profiling/energy_profiling_metrics.json"),
    robustness_json: Path = Path("results/robustness_benchmark/robustness_metrics.json"),
) -> None:
    """Renders Figure 5: Hardware Efficiency & Physical Domain Boundaries."""
    set_paper_style()

    print("[Fig 5] Loading hardware profiling and domain boundary metrics...")
    hw_metrics, matrix, col_labels, cond_labels = load_hardware_and_boundary_metrics(energy_json, robustness_json)

    fig = plt.figure(figsize=(14.2, 5.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.15], wspace=0.36)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.14)

    # --------------------------------------------------------------------------
    # Panel A: Compute Efficiency (Horizontal Energy & Power Comparison)
    # --------------------------------------------------------------------------
    ax_hw = fig.add_subplot(gs[0, 0])
    ax_hw.set_title(
        "(a) Compute Energy & Power Consumption Comparison\n"
        "Intel Loihi-2 Neuromorphic Execution vs. Nvidia Jetson Orin Edge GPU",
        fontsize=9.5, pad=14, weight="bold",
    )

    models_display = [
        ("SPWM-v0.7.3 (Prod)", "SPWM-v0.7.3\n(Neuromorphic SNN)", True),
        ("SNN-ALIF (Matched)", "SNN-ALIF\n(Matched Baseline)", True),
        ("ANN-Rate (Ablation)", "ANN-Rate\n(Continuous Leaky)", False),
        ("Continuous RSSM", "Continuous RSSM\n(Hafner et al.)", False),
        ("Deep GRU", "Deep GRU\n(Continuous RNN)", False),
        ("Continuous ConvLSTM", "ConvLSTM\n(Continuous Spatial)", False),
    ]

    y_pos = np.arange(len(models_display))[::-1]
    bar_height = 0.55

    for idx, (m_key, label, has_neuro) in enumerate(models_display):
        y = y_pos[idx]
        e_gpu = hw_metrics[m_key]["energy_gpu_nj"]
        e_neuro = hw_metrics[m_key]["energy_neuro_nj"]
        p_mw = hw_metrics[m_key]["power_mw"]
        ratio = hw_metrics[m_key]["ratio"]

        # 1. Edge GPU execution bar (Warm Red / Coral)
        ax_hw.barh(y, e_gpu, height=bar_height, color="#FCA5A5", edgecolor="#DC2626", linewidth=0.9, alpha=0.85, zorder=2)

        # 2. Neuromorphic execution bar (Teal / Sapphire) if supported
        if has_neuro:
            ax_hw.barh(y, e_neuro, height=bar_height, color=PALETTE["neuromorphic"], edgecolor="#0F766E", linewidth=1.1, alpha=0.95, zorder=3)
            ax_hw.text(e_neuro * 1.25, y, f"{e_neuro:.1f} nJ ({p_mw:.3f} mW)", va="center", ha="left", fontsize=7.2, color="#044E42", weight="bold")
            ax_hw.text(e_gpu * 1.15, y, f"{e_gpu:.0f} nJ (GPU: {p_mw*ratio:.1f} mW) [$\mathbf{{{ratio:.1f}\\times}}$]", va="center", ha="left", fontsize=7.0, color="#991B1B")
        else:
            ax_hw.text(e_gpu * 1.15, y, f"{e_gpu:.0f} nJ ({p_mw:.1f} mW)", va="center", ha="left", fontsize=7.0, color="#991B1B")

    ax_hw.set_xscale("log")
    ax_hw.set_xlim(1.0, 3e6)
    ax_hw.set_yticks(y_pos)
    ax_hw.set_yticklabels([m[1] for m in models_display], fontsize=8.0)
    ax_hw.set_xlabel("Recurrent Energy per Step (log-scale nJ/step)", fontsize=9.0, weight="bold")
    ax_hw.grid(True, which="both", axis="x", linestyle=":", alpha=0.5)

    leg_patches = [
        patches.Patch(facecolor=PALETTE["neuromorphic"], edgecolor="#0F766E", label="Intel Loihi-2 Neuromorphic (SOPs)"),
        patches.Patch(facecolor="#FCA5A5", edgecolor="#DC2626", label="Nvidia Jetson Orin Edge GPU (MACs)"),
    ]
    ax_hw.legend(handles=leg_patches, loc="lower right", frameon=True, framealpha=0.92, fontsize=7.5)

    # --------------------------------------------------------------------------
    # Panel B: Failure Boundary Matrix (2D Sensitivity Heatmap)
    # --------------------------------------------------------------------------
    ax_mat = fig.add_subplot(gs[0, 1])
    ax_mat.set_title(
        "(b) Zero-Shot Sim-to-Real Failure Boundary Matrix\n"
        "Task Grasp Success Rate (%) Across Sensor vs. Mechanical Domain Gaps",
        fontsize=9.5, pad=14, weight="bold",
    )

    im = ax_mat.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=100.0, aspect="auto", origin="upper")

    ax_mat.set_xticks(np.arange(len(col_labels)))
    ax_mat.set_yticks(np.arange(len(cond_labels)))
    ax_mat.set_xticklabels(col_labels, fontsize=7.0, rotation=15, ha="right")
    ax_mat.set_yticklabels(cond_labels, fontsize=7.2)
    ax_mat.set_xlabel("Empirical Evaluation Slices (Elevation & Payload Mass)", fontsize=8.8, labelpad=8, weight="bold")
    ax_mat.set_ylabel("Domain Shift Perturbations", fontsize=8.8, labelpad=14, weight="bold")

    # Print numerical success rates in each cell
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            text_color = "#FFFFFF" if val < 25.0 or val > 65.0 else "#0F172A"
            ax_mat.text(c, r, f"{val:.1f}%", ha="center", va="center", color=text_color, fontsize=8.0, weight="bold")

    # Demarcation Boundary Line dividing Invariant Sensor Domain from Mechanical Failure Domain
    ax_mat.axhline(1.5, color="#1E293B", linewidth=2.4, linestyle="--")

    # Domain Region Labels
    ax_mat.text(
        3.0, -0.42,
        "ROBUST PERCEPTION DOMAIN (Zero-Shot Invariant: 74%-100% Success)",
        ha="center", va="center", fontsize=7.2, color="#065F46", weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#10B981", lw=0.8),
    )

    ax_mat.text(
        3.0, 4.42,
        "MECHANICAL TRANSFER BOUNDARY (Sharp Drop-Off: 0%-33% Success)",
        ha="center", va="center", fontsize=7.2, color="#991B1B", weight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#EF4444", lw=0.8),
    )

    cbar = fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04)
    cbar.set_label("Task Grasp Success Rate (%)", fontsize=8.0, weight="bold")
    cbar.ax.tick_params(labelsize=7.5)

    save_publication_figure(fig, output_dir / "fig5_hardware_domain")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoRL Figure 5: Hardware Efficiency & Physical Domain Boundaries.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for exported figures")
    parser.add_argument("--energy-json", type=Path, default=Path("results/energy_profiling/energy_profiling_metrics.json"))
    parser.add_argument("--robustness-json", type=Path, default=Path("results/robustness_benchmark/robustness_metrics.json"))
    args = parser.parse_args()

    render_figure_5(
        output_dir=args.output_dir,
        energy_json=args.energy_json,
        robustness_json=args.robustness_json,
    )


if __name__ == "__main__":
    main()
