"""
Master Scientific Visualization Pipeline for CoRL Paper (SPWM-v0.7.3).

Executes all 5 publication figure generation scripts:
  - Figure 1: System Teaser & Phase-Space Concept (`visualize_fig1_teaser.py`)
  - Figure 2: End-to-End SPWM Architecture (`visualize_fig2_architecture.py`)
  - Figure 3: Active Perception & Contact Impulse Telemetry (`visualize_fig3_telemetry.py`)
  - Figure 4: Multi-Horizon Stability & Drift Ablation (`visualize_fig4_stability.py`)
  - Figure 5: Hardware Efficiency & Physical Domain Boundaries (`visualize_fig5_hardware_domain.py`)

Zero AST Fallbacks strictly enforced. Generates vector (.pdf, .svg) and 300+ DPI (.png).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from visualize_fig1_teaser import render_figure_1
from visualize_fig2_architecture import render_figure_2
from visualize_fig3_telemetry import render_figure_3
from visualize_fig4_stability import render_figure_4
from visualize_fig5_hardware_domain import render_figure_5


def generate_all_figures(
    output_dir: Path = Path("figures"),
    config_path: Path = Path("results/spwm_v0_7_3_train_m1/config.yaml"),
    spwm_ckpt: Path = Path("results/spwm_v0_7_3_train_m1/model.pt"),
    ann_ckpt: Path = Path("results/ann_ablation/ann_rate_m1/model.pt"),
    data_dir: Path = Path("sim/ExportedTrajectories"),
    baselines_json: Path = Path("results/baselines/comparative_metrics.json"),
    ablation_json: Path = Path("results/ann_ablation/ablation_metrics.json"),
    energy_json: Path = Path("results/energy_profiling/energy_profiling_metrics.json"),
    robustness_json: Path = Path("results/robustness_benchmark/robustness_metrics.json"),
) -> None:
    """Executes the complete publication figure generation pipeline."""
    start_time = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING PUBLICATION FIGURE GENERATION PIPELINE (CoRL)")
    print(f"Output Directory: {output_dir.resolve()}")
    print("=" * 80)

    # Figure 1: System Teaser & Phase-Space Concept
    print("\n>>> [1/5] Rendering Figure 1: System Teaser & Phase-Space Concept...")
    t0 = time.perf_counter()
    render_figure_1(
        output_dir=output_dir,
        config_path=config_path,
        spwm_ckpt=spwm_ckpt,
        ann_ckpt=ann_ckpt,
        trajectories_dir=data_dir,
    )
    print(f"    Completed Figure 1 in {time.perf_counter() - t0:.2f}s")

    # Figure 2: End-to-End SPWM Architecture Block Diagram
    print("\n>>> [2/5] Rendering Figure 2: End-to-End SPWM Architecture...")
    t0 = time.perf_counter()
    render_figure_2(output_dir=output_dir)
    print(f"    Completed Figure 2 in {time.perf_counter() - t0:.2f}s")

    # Figure 3: Active Perception & Contact Impulse Telemetry
    print("\n>>> [3/5] Rendering Figure 3: Active Perception & Contact Impulse Telemetry...")
    t0 = time.perf_counter()
    render_figure_3(
        output_dir=output_dir,
        config_path=config_path,
        ckpt_path=spwm_ckpt,
        data_dir=data_dir,
    )
    print(f"    Completed Figure 3 in {time.perf_counter() - t0:.2f}s")

    # Figure 4: Multi-Horizon Stability & Drift Ablation
    print("\n>>> [4/5] Rendering Figure 4: Multi-Horizon Stability & Drift Ablation...")
    t0 = time.perf_counter()
    render_figure_4(
        output_dir=output_dir,
        baselines_json=baselines_json,
        ablation_json=ablation_json,
        config_path=config_path,
        spwm_ckpt=spwm_ckpt,
        ann_ckpt=ann_ckpt,
        trajectories_dir=data_dir,
    )
    print(f"    Completed Figure 4 in {time.perf_counter() - t0:.2f}s")

    # Figure 5: Hardware Efficiency & Physical Domain Boundaries
    print("\n>>> [5/5] Rendering Figure 5: Hardware Efficiency & Physical Domain Boundaries...")
    t0 = time.perf_counter()
    render_figure_5(
        output_dir=output_dir,
        energy_json=energy_json,
        robustness_json=robustness_json,
    )
    print(f"    Completed Figure 5 in {time.perf_counter() - t0:.2f}s")

    total_sec = time.perf_counter() - start_time
    print("\n" + "=" * 80)
    print(f"ALL PUBLICATION FIGURES GENERATED SUCCESSFULLY in {total_sec:.2f}s")
    print(f"Artifacts preserved in: {output_dir.resolve()}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Scientific Visualization Pipeline for CoRL Paper.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for exported figures")
    parser.add_argument("--config", type=Path, default=Path("results/spwm_v0_7_3_train_m1/config.yaml"))
    parser.add_argument("--spwm-ckpt", type=Path, default=Path("results/spwm_v0_7_3_train_m1/model.pt"))
    parser.add_argument("--ann-ckpt", type=Path, default=Path("results/ann_ablation/ann_rate_m1/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("sim/ExportedTrajectories"))
    parser.add_argument("--baselines-json", type=Path, default=Path("results/baselines/comparative_metrics.json"))
    parser.add_argument("--ablation-json", type=Path, default=Path("results/ann_ablation/ablation_metrics.json"))
    parser.add_argument("--energy-json", type=Path, default=Path("results/energy_profiling/energy_profiling_metrics.json"))
    parser.add_argument("--robustness-json", type=Path, default=Path("results/robustness_benchmark/robustness_metrics.json"))
    args = parser.parse_args()

    generate_all_figures(
        output_dir=args.output_dir,
        config_path=args.config,
        spwm_ckpt=args.spwm_ckpt,
        ann_ckpt=args.ann_ckpt,
        data_dir=args.data_dir,
        baselines_json=args.baselines_json,
        ablation_json=args.ablation_json,
        energy_json=args.energy_json,
        robustness_json=args.robustness_json,
    )


if __name__ == "__main__":
    main()

