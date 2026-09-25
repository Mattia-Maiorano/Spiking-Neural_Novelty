"""
Consolidated Scientific Figure Generator.
Generates Figures 1 through 7 for SPWM-v1 paper and research report.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from spwm.evaluation.plots import (
    plot_architecture_diagram,
    plot_training_curves,
    plot_multi_step_degradation,
    plot_spike_raster,
    plot_fast_vs_slow_memory,
    plot_baseline_comparison,
    plot_ablation_comparison,
)
from spwm.data.synthetic_world import MovingObjectsWorld
from spwm.data.event_camera import EventCameraSimulator
from spwm.models.world_model import SPWM


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all scientific figures.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="results/figures", help="Target directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating publication figures to: {out_dir}")

    # Figure 1: Architecture
    plot_architecture_diagram(out_dir / "fig1_architecture.png")
    print("Generated Figure 1: Architecture Diagram")

    # Figure 2: Training Curves
    # Search for training_log.csv in results
    csv_candidates = list(results_dir.glob("**/training_log.csv"))
    if csv_candidates:
        import pandas as pd
        df = pd.read_csv(csv_candidates[0])
        history = df.to_dict(orient="records")
        plot_training_curves(history, out_dir / "fig2_training_curves.png")
        print(f"Generated Figure 2: Training Curves from {csv_candidates[0]}")

    # Figure 4 & 5: Spike Raster & Fast vs Slow Memory
    # Run a realistic sample trajectory through SPWM
    world = MovingObjectsWorld()
    traj = world.generate_trajectory(trajectory_id=99, length=40, num_objects=1)
    sim = EventCameraSimulator(height=32, width=32)
    ev_batch = sim.trajectory_to_events(traj)

    model = SPWM(latent_dim=64, timescale_dims=(32, 32), betas=(0.80, 0.98))
    model.eval()
    with torch.no_grad():
        out = model(ev_batch.dense_events.unsqueeze(0))
        fast_spk = out.fast_spikes[0].cpu().numpy()
        slow_spk = out.slow_spikes[0].cpu().numpy()
        # Extract internal membrane potential traces
        dyn_out, _ = model.dynamics(model.encoder(ev_batch.dense_events.unsqueeze(0))[0])
        fast_mem = dyn_out.fast_mems[0].cpu().numpy()
        slow_mem = dyn_out.slow_mems[0].cpu().numpy()

    plot_spike_raster(fast_spk, slow_spk, out_dir / "fig4_spike_raster.png")
    print("Generated Figure 4: Spike Raster & Activity Histogram")

    plot_fast_vs_slow_memory(fast_mem, slow_mem, out_dir / "fig5_fast_vs_slow_memory.png")
    print("Generated Figure 5: Fast vs Slow Memory Dynamics")

    # Figure 3: Multi-Step Rollout Degradation Curve
    # Check if baseline metrics exist in results_dir
    degradation_curves: Dict[str, Dict[int, float]] = {}
    metrics_files = list(results_dir.glob("**/metrics.json"))
    for mf in metrics_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_name = mf.parent.name
            if "test" in data and "latent_mse_per_horizon" in data["test"]:
                curve = {int(k): v for k, v in data["test"]["latent_mse_per_horizon"].items()}
                degradation_curves[model_name] = curve
        except Exception:
            pass

    if degradation_curves:
        plot_multi_step_degradation(degradation_curves, out_dir / "fig3_multi_step_degradation.png")
        print(f"Generated Figure 3: Degradation Curve across {len(degradation_curves)} models")

    # Figure 6: Baseline Comparison
    baseline_metrics = {}
    for m, label in [
        ("spwm_v1", "SPWM-v1"),
        ("baseline_gru", "GRU World Model"),
        ("baseline_vanilla_snn", "Vanilla SNN"),
        ("baseline_mlp", "MLP Dynamics"),
    ]:
        m_path = results_dir / m / "metrics.json"
        if m_path.is_file():
            with open(m_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            baseline_metrics[label] = {
                "one_step_mse": d.get("test", {}).get("teacher_forcing_mse", 0.0),
                "rollout_mse_h10": d.get("test", {}).get("latent_mse_per_horizon", {}).get("10", 0.0),
                "spike_rate": d.get("test", {}).get("mean_spike_rate", 0.0),
            }
    if baseline_metrics:
        plot_baseline_comparison(baseline_metrics, out_dir / "fig6_baseline_comparison.png")
        print("Generated Figure 6: Baseline Comparison Bar Charts")

    # Figure 7: Ablation Comparison
    ablation_json = results_dir / "ablations" / "ablation_summary.json"
    if ablation_json.is_file():
        with open(ablation_json, "r", encoding="utf-8") as f:
            ablation_dict = json.load(f)
        plot_ablation_comparison(ablation_dict, out_dir / "fig7_ablation_comparison.png")
        print("Generated Figure 7: Ablation Study Comparison")

    print(f"All figures generated successfully in {out_dir}")


if __name__ == "__main__":
    main()
