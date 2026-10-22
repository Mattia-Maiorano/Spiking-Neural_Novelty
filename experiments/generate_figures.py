"""
Consolidated Scientific Figure Generator for SPWM (v0.2.0).
Generates Figures 1 through 7 for SPWM paper and research report,
combining Phase 1 (M1: 0-10), Phase 2 (M2: 0-20), and Phase 3 (M3: 0-100) runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import chronicle
from spwm.evaluation.visualizer import (
    plot_ablation_comparison,
    plot_architecture_diagram,
    plot_baseline_comparison,
    plot_multi_step_degradation,
    plot_training_curves,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all scientific figures.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Target directory for figures")
    parser.add_argument("--version", type=str, default="spwm_v0_2_0", help="Target model version prefix")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chronicle.log_info(f"Generating publication figures to: {out_dir}")

    # Figure 1: Architecture
    plot_architecture_diagram(out_dir / "fig1_architecture.png")
    chronicle.log_success("Generated Figure 1: Architecture Diagram")

    # Figure 2: Combined Training Curves across M1 -> M2 -> M3
    combined_history: List[Dict[str, Any]] = []
    epoch_offset = 0
    for phase_suffix in ("train_m1", "train_m2", "train_m3"):
        phase_log = results_dir / f"{args.version}_{phase_suffix}" / "training_log.json"
        if phase_log.is_file():
            with open(phase_log, "r", encoding="utf-8") as f:
                records = json.load(f)
            for idx, rec in enumerate(records):
                entry = dict(rec)
                entry["epoch"] = epoch_offset + idx + 1
                entry["phase"] = phase_suffix
                combined_history.append(entry)
            epoch_offset += len(records)

    if combined_history:
        plot_training_curves(combined_history, out_dir / "fig2_training_curves.png")
        chronicle.log_success(
            f"Generated Figure 2: Combined M1+M2+M3 Training Curves ({len(combined_history)} total epochs)"
        )
    else:
        csv_candidates = list(results_dir.glob("**/training_log.csv"))
        if csv_candidates:
            import pandas as pd

            df = pd.read_csv(csv_candidates[0])
            history = df.to_dict(orient="records")
            plot_training_curves(history, out_dir / "fig2_training_curves.png")
            chronicle.log_success(f"Generated Figure 2: Training Curves from {csv_candidates[0]}")

    # Figure 3: Multi-Step Rollout Degradation Curve (Combined M1 [0-10], M2 [0-20], M3 [0-100])
    phase_degradation_curves: Dict[str, Dict[int, float]] = {}
    for m, label in [
        (f"{args.version}_train_m1", "Phase 1: M1 Isolated (0-10)"),
        (f"{args.version}_train_m2", "Phase 2: M1+M2 Recovery (0-20)"),
        (f"{args.version}_train_m3", "Phase 3: M1+M2+M3 Full Relay (0-100)"),
    ]:
        m_dir = results_dir / m
        candidates = sorted(m_dir.glob("metrics*.json")) if m_dir.is_dir() else []
        for mf in candidates:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                eval_data = data.get("test", data.get("evaluation", {}))
                if "position_error_per_horizon" in eval_data:
                    curve = {int(k): float(v) for k, v in eval_data["position_error_per_horizon"].items()}
                    phase_degradation_curves[label] = curve
                    break
            except Exception as exc:
                chronicle.log_warning(f"Skipping malformed metrics file {mf}: {exc}")

    if phase_degradation_curves:
        plot_multi_step_degradation(phase_degradation_curves, out_dir / "fig3_multi_step_degradation.png")
        chronicle.log_success(
            f"Generated Figure 3: Combined Degradation Curve across {len(phase_degradation_curves)} phases"
        )

    # Figure 6: Phase Comparison across M1, M2, M3
    comparison_metrics = {}
    for m, label in [
        (f"{args.version}_train_m1", "Phase 1 (M1 Nominal)"),
        (f"{args.version}_train_m2", "Phase 2 (M2 Recovery)"),
        (f"{args.version}_train_m3", "Phase 3 (M3 Sustained Relay)"),
    ]:
        candidates = sorted((results_dir / m).glob("metrics*.json")) if (results_dir / m).is_dir() else []
        for m_path in candidates:
            try:
                with open(m_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                eval_dict = d.get("test", d.get("evaluation", {}))
                comparison_metrics[label] = {
                    "one_step_mse": eval_dict.get("teacher_forcing_mse", 0.0),
                    "rollout_mse_h10": eval_dict.get("latent_mse_per_horizon", {}).get("10", 0.0),
                    "spike_rate": eval_dict.get("mean_spike_rate", 0.0),
                }
                break
            except Exception as exc:
                chronicle.log_warning(f"Could not parse comparison metrics {m_path}: {exc}")
    if comparison_metrics:
        plot_baseline_comparison(comparison_metrics, out_dir / "fig6_baseline_comparison.png")
        chronicle.log_success("Generated Figure 6: Phase Comparison Bar Charts")

    # Figure 7: Ablation Comparison
    ablation_json = results_dir / "ablations" / "ablation_summary.json"
    if ablation_json.is_file():
        with open(ablation_json, "r", encoding="utf-8") as f:
            ablation_dict = json.load(f)
        plot_ablation_comparison(ablation_dict, out_dir / "fig7_ablation_comparison.png")
        chronicle.log_success("Generated Figure 7: Ablation Study Comparison")

    chronicle.log_success(f"All figures generated successfully in {out_dir}")


if __name__ == "__main__":
    main()
