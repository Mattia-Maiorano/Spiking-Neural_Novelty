"""
Diagnostic Analysis Script for GRU Corrector Failure Mode Isolation.
Executes:
- Diagnostic 2: Inspect Learned Gain Field Magnitude and Trajectory (Mean abs gain, Var across time, Var across dims).
- Diagnostic 3: Gradient Norm Profiling Across Time Steps and Cosine Similarity between t=1 and t=25.

Usage:
    .venv/bin/python scripts/diagnostic_analysis.py
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.config.schema import ExperimentConfig
from spwm.data.datasets import EventWorldDataset
from spwm.models.world_model import SPWM
from spwm.models.rollout import rollout_latent
from spwm.learning.metrics import position_error, velocity_error


def load_model_and_val_batch(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[SPWM, Dict[str, torch.Tensor], ExperimentConfig]:
    cfg = ExperimentConfig.load(config_path)

    # Disable temporal phase feeding if loading an older checkpoint without it
    model = SPWM.from_config(cfg, enable_corrector=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_state = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
    
    # Check if checkpoint corrector had temporal phase or not
    if "corrector.gru_cell.weight_ih" in model_state:
        in_dim = model_state["corrector.gru_cell.weight_ih"].shape[1]
        expected_in = 2 * model.latent_dim
        if in_dim == expected_in:
            model.corrector.feed_temporal_phase = False
            model.corrector.gru_cell = nn.GRUCell(expected_in, model.corrector.corrector_dim).to(device)

    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    # Load validation data matching exact validation split indices
    val_indices = list(range(3200, 3200 + batch_size))
    val_dataset = EventWorldDataset(
        trajectory_indices=val_indices,
        sequence_length=cfg.data.sequence_length,
        drift_injection_prob=cfg.data.drift_injection_prob,
        drift_magnitude=cfg.data.drift_magnitude,
        mixed_drift=cfg.data.mixed_drift,
        cache_data=False,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(val_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    return model, batch, cfg


def run_diagnostic_2(
    model: SPWM,
    batch: Dict[str, torch.Tensor],
    horizon: int = 25,
    burn_in_steps: int = 0,
) -> Dict[str, Any]:
    """
    Diagnostic 2: Inspect the Learned Gain Field Magnitude and Trajectory.
    1. Mean absolute gain per step: \bar{g}(t) = \frac{1}{B \cdot D} \sum_{b, d} |g_{t, b, d}|
    2. Gain variance across time: Var_t(g_t)
    3. Gain variance across dimensions: Var_d(g_t)
    """
    events = batch["events"]
    flat_kin = batch["flat_kinematics"]
    B = events.shape[0]

    with torch.no_grad():
        enc_out = model(events, accumulate_local_updates=False)
        z_start = enc_out.latent_states[:, 0]

        decoder_state = None
        if hasattr(model, "physical_decoder") and model.physical_decoder is not None:
            z_prefix = enc_out.latent_states[:, :1]
            decoder_state = model.physical_decoder.warm_up_filter(z_prefix)

        rollout_res = rollout_latent(
            model=model,
            z_init=z_start,
            horizon=horizon,
            use_corrector=True,
            burn_in_steps=burn_in_steps,
            with_grad=False,
            decode_kinematics=True,
            initial_decoder_state=decoder_state,
        )

    # gain_masks shape: [B, H, D]
    gain_masks = rollout_res.gain_masks  # [B, H, D]
    if gain_masks is None:
        raise RuntimeError("rollout_res.gain_masks is None; corrector must be enabled.")

    g = gain_masks.cpu().numpy()  # [B, H, D]
    B_sz, H_sz, D_sz = g.shape

    # 1. Mean absolute gain per step: \bar{g}(t) = \frac{1}{B \cdot D} \sum_{b, d} |g_{t, b, d}|
    mean_abs_gain_per_step = np.mean(np.abs(g), axis=(0, 2)).tolist()  # shape (H,)
    mean_gain_per_step = np.mean(g, axis=(0, 2)).tolist()

    # 2. Gain variance across time: Var_t(g_t)
    # Variance across time per (b, d), then averaged over B and D:
    var_t_per_elem = np.var(g, axis=1, ddof=1)  # shape (B, D)
    mean_var_across_time = float(np.mean(var_t_per_elem))
    # Variance of the mean trajectory curve across time:
    var_of_mean_across_time = float(np.var(mean_abs_gain_per_step, ddof=1))

    # 3. Gain variance across dimensions: Var_d(g_t) for each step t
    var_d_per_step = np.mean(np.var(g, axis=2, ddof=1), axis=0).tolist()  # shape (H,)
    mean_var_across_dims = float(np.mean(var_d_per_step))

    # Additional statistics: saturation and distribution
    g_flat = g.flatten()
    sat_99 = float(np.mean(np.abs(g_flat) > 0.99) * 100)
    sat_90 = float(np.mean(np.abs(g_flat) > 0.90) * 100)
    near_zero = float(np.mean(np.abs(g_flat) < 1e-4) * 100)
    min_val = float(np.min(g_flat))
    max_val = float(np.max(g_flat))
    median_abs = float(np.median(np.abs(g_flat)))

    return {
        "horizon": horizon,
        "burn_in_steps": burn_in_steps,
        "mean_abs_gain_per_step": mean_abs_gain_per_step,
        "mean_gain_per_step": mean_gain_per_step,
        "var_across_time_elementwise": mean_var_across_time,
        "var_of_mean_across_time": var_of_mean_across_time,
        "var_across_dims_per_step": var_d_per_step,
        "mean_var_across_dims": mean_var_across_dims,
        "saturation_99_pct": sat_99,
        "saturation_90_pct": sat_90,
        "near_zero_pct": near_zero,
        "min_gain": min_val,
        "max_gain": max_val,
        "median_abs_gain": median_abs,
    }


def run_diagnostic_3(
    model: SPWM,
    batch: Dict[str, torch.Tensor],
    eval_steps: List[int] = [1, 5, 10, 15, 20, 25],
    beta_v: float = 2.0,
    burn_in_steps: int = 0,
) -> Dict[str, Any]:
    """
    Diagnostic 3: Gradient Norm Profiling Across Time Steps.
    Computes G(t) = ||\nabla_{\theta_{corr}} \mathcal{L}_t||_2 for t in eval_steps.
    Computes cosine similarity between step 1 and step 25:
    \cos(\theta) = \frac{\nabla_{\theta} \mathcal{L}_1 \cdot \nabla_{\theta} \mathcal{L}_{25}}{||\nabla_{\theta} \mathcal{L}_1|| ||\nabla_{\theta} \mathcal{L}_{25}||}
    """
    events = batch["events"]
    flat_kin = batch["flat_kinematics"]
    device = events.device
    num_objects = 1

    # Freeze base model parameters strictly
    for name, param in model.named_parameters():
        if "corrector" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    corrector_params = [p for p in model.corrector.parameters() if p.requires_grad]

    with torch.no_grad():
        enc_out = model(events, accumulate_local_updates=False)
        z_start = enc_out.latent_states[:, 0].detach()

        decoder_state = None
        if hasattr(model, "physical_decoder") and model.physical_decoder is not None:
            z_prefix = enc_out.latent_states[:, :1]
            decoder_state = model.physical_decoder.warm_up_filter(z_prefix)

    max_t = max(eval_steps)
    target_kin = flat_kin[:, 1 : 1 + max_t]

    step_grad_vectors: Dict[int, torch.Tensor] = {}
    step_grad_norms: Dict[int, float] = {}
    step_losses: Dict[int, float] = {}

    for t_step in eval_steps:
        # Zero gradients
        model.zero_grad()
        for p in corrector_params:
            if p.grad is not None:
                p.grad.zero_()

        # Differentiable rollout up to step t_step
        rollout_res = rollout_latent(
            model=model,
            z_init=z_start,
            horizon=t_step,
            use_corrector=True,
            burn_in_steps=burn_in_steps,
            with_grad=True,
            decode_kinematics=True,
            initial_decoder_state=decoder_state,
        )

        # Loss specifically at step t_step (0-indexed index: t_step - 1)
        step_idx = t_step - 1
        pred_pos_t = rollout_res.decoded_kinematics[:, step_idx, : 2 * num_objects]
        true_pos_t = target_kin[:, step_idx, : 2 * num_objects]
        pred_vel_t = rollout_res.decoded_kinematics[:, step_idx, 2 * num_objects : 4 * num_objects]
        true_vel_t = target_kin[:, step_idx, 2 * num_objects : 4 * num_objects]

        loss_pos_t = F.mse_loss(pred_pos_t, true_pos_t)
        loss_vel_t = F.mse_loss(pred_vel_t, true_vel_t)
        loss_t = loss_pos_t + beta_v * loss_vel_t

        step_losses[t_step] = float(loss_t.item())

        # Compute gradient w.r.t corrector parameters
        grads = torch.autograd.grad(
            loss_t,
            corrector_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        flat_grads = []
        for g, p in zip(grads, corrector_params):
            if g is not None:
                flat_grads.append(g.contiguous().view(-1))
            else:
                flat_grads.append(torch.zeros_like(p).view(-1))
        
        flat_grad_vec = torch.cat(flat_grads)
        norm_val = float(torch.norm(flat_grad_vec, p=2).item())
        step_grad_vectors[t_step] = flat_grad_vec
        step_grad_norms[t_step] = norm_val

    # Cosine similarities
    cos_sim_1_25 = None
    if 1 in step_grad_vectors and 25 in step_grad_vectors:
        g1 = step_grad_vectors[1]
        g25 = step_grad_vectors[25]
        norm_product = (torch.norm(g1, p=2) * torch.norm(g25, p=2)).item()
        if norm_product > 1e-12:
            cos_sim_1_25 = float((torch.dot(g1, g25) / norm_product).item())
        else:
            cos_sim_1_25 = 0.0

    # Full cosine similarity matrix between all evaluated steps
    cos_matrix: Dict[str, Dict[str, float]] = {}
    for t1 in eval_steps:
        cos_matrix[str(t1)] = {}
        for t2 in eval_steps:
            g_a = step_grad_vectors[t1]
            g_b = step_grad_vectors[t2]
            n_ab = (torch.norm(g_a, p=2) * torch.norm(g_b, p=2)).item()
            if n_ab > 1e-12:
                cos_matrix[str(t1)][str(t2)] = float((torch.dot(g_a, g_b) / n_ab).item())
            else:
                cos_matrix[str(t1)][str(t2)] = 0.0

    return {
        "eval_steps": eval_steps,
        "step_losses": step_losses,
        "gradient_norms": step_grad_norms,
        "cosine_similarity_t1_t25": cos_sim_1_25,
        "cosine_similarity_matrix": cos_matrix,
    }


def plot_diagnostics(
    d2_results: Dict[str, Any],
    d3_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Subplot 1: Gain magnitude across time
    steps = list(range(1, d2_results["horizon"] + 1))
    mean_abs_g = d2_results["mean_abs_gain_per_step"]
    axes[0].plot(steps, mean_abs_g, "o-", color="#1f77b4", linewidth=2.0, markersize=5, label=r"Mean Absolute Gain $\bar{g}(t)$")
    axes[0].set_title(r"Diagnostic 2: Gain Magnitude $\bar{g}(t)$ Across Horizon", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rollout Step $t$", fontsize=11)
    axes[0].set_ylabel(r"$\frac{1}{B \cdot D} \sum |g_{t, b, d}|$", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Subplot 2: Gain variance across dimensions
    var_d = d2_results["var_across_dims_per_step"]
    axes[1].plot(steps, var_d, "s-", color="#ff7f0e", linewidth=2.0, markersize=5, label=r"Var Across Dims $\mathrm{Var}_d(g_t)$")
    axes[1].set_title(r"Diagnostic 2: Spatial Dimensional Variance $\mathrm{Var}_d(g_t)$", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Rollout Step $t$", fontsize=11)
    axes[1].set_ylabel(r"$\mathrm{Var}_d(g_t)$", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    # Subplot 3: Gradient norm profile across time steps
    eval_steps = d3_results["eval_steps"]
    g_norms = [d3_results["gradient_norms"][t] for t in eval_steps]
    axes[2].bar([str(t) for t in eval_steps], g_norms, color="#2ca02c", alpha=0.85, edgecolor="black")
    axes[2].set_title(r"Diagnostic 3: Gradient Norm $G(t) = \|\nabla_{\theta} \mathcal{L}_t\|_2$", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Rollout Step $t$", fontsize=11)
    axes[2].set_ylabel(r"$\|\nabla_{\theta} \mathcal{L}_t\|_2$", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6, axis="y")

    # Add text box for cosine similarity
    cos_1_25 = d3_results["cosine_similarity_t1_t25"]
    axes[2].text(
        0.05, 0.90,
        f"cos(grad_1, grad_25) = {cos_1_25:+.4f}" if cos_1_25 is not None else "N/A",
        transform=axes[2].transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5),
    )

    plt.tight_layout()
    plot_path = output_dir / "diagnostics_2_and_3.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Diagnostic] Summary figure saved to: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diagnostics 2 and 3 on GRU corrector checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="results/spwm_v0_1_0_train_corrector/model.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="results/spwm_v0_1_0_train_corrector/config.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
    parser.add_argument("--horizon", type=int, default=25, help="Rollout horizon")
    parser.add_argument("--output-dir", type=str, default="results/diagnostics", help="Output directory for reports")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING DIAGNOSTIC ANALYSIS OF GRU CORRECTOR")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(f"Device:     {device}")
    print(f"Horizon:    {args.horizon}")
    print("=" * 80)

    # Load model and batch
    model, batch, cfg = load_model_and_val_batch(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=device,
        batch_size=32,
    )

    # -------------------------------------------------------------------------
    # Diagnostic 2: Inspect Learned Gain Field Magnitude and Trajectory
    # -------------------------------------------------------------------------
    print("\n>>> RUNNING DIAGNOSTIC 2: Gain Field Magnitude and Trajectory Analysis...")
    d2 = run_diagnostic_2(model, batch, horizon=args.horizon, burn_in_steps=0)

    print("\n--- Diagnostic 2 Results ---")
    print(f"Mean Abs Gain at t=1:   {d2['mean_abs_gain_per_step'][0]:.6f}")
    print(f"Mean Abs Gain at t=5:   {d2['mean_abs_gain_per_step'][4]:.6f}")
    print(f"Mean Abs Gain at t=10:  {d2['mean_abs_gain_per_step'][9]:.6f}")
    print(f"Mean Abs Gain at t=15:  {d2['mean_abs_gain_per_step'][14]:.6f}")
    print(f"Mean Abs Gain at t=20:  {d2['mean_abs_gain_per_step'][19]:.6f}")
    print(f"Mean Abs Gain at t=25:  {d2['mean_abs_gain_per_step'][24]:.6f}")
    print(f"Overall Mean Abs Gain:  {np.mean(d2['mean_abs_gain_per_step']):.6f}")
    print(f"Min / Max Gain Value:   {d2['min_gain']:.6f} / {d2['max_gain']:.6f}")
    print(f"Median Abs Gain:        {d2['median_abs_gain']:.6f}")
    print(f"Var across Time (elem): {d2['var_across_time_elementwise']:.6e}")
    print(f"Var of Mean Curve:      {d2['var_of_mean_across_time']:.6e}")
    print(f"Var across Dims (mean): {d2['mean_var_across_dims']:.6e}")
    print(f"Saturated (|g| > 0.99): {d2['saturation_99_pct']:.2f}%")
    print(f"Saturated (|g| > 0.90): {d2['saturation_90_pct']:.2f}%")
    print(f"Near Zero (|g| < 1e-4): {d2['near_zero_pct']:.2f}%")

    # -------------------------------------------------------------------------
    # Diagnostic 3: Gradient Norm Profiling Across Time Steps
    # -------------------------------------------------------------------------
    print("\n>>> RUNNING DIAGNOSTIC 3: Gradient Norm Profiling Across Time Steps...")
    eval_steps = [1, 5, 10, 15, 20, 25]
    d3 = run_diagnostic_3(model, batch, eval_steps=eval_steps, beta_v=cfg.loss.beta_v)

    print("\n--- Diagnostic 3 Results ---")
    for t in eval_steps:
        print(f"Step t={t:2d} | Loss L_t = {d3['step_losses'][t]:.6f} | Gradient Norm ||grad_t||_2 = {d3['gradient_norms'][t]:.6f}")

    cos_1_25 = d3["cosine_similarity_t1_t25"]
    print(f"\nCosine Similarity cos(grad_1, grad_25): {cos_1_25:+.6f}")
    print("\nCosine Similarity Matrix between Steps:")
    header = "      " + "".join([f"t={t:<7}" for t in eval_steps])
    print(header)
    for t1 in eval_steps:
        row = f"t={t1:<3} " + "".join([f"{d3['cosine_similarity_matrix'][str(t1)][str(t2)]:+7.3f} " for t2 in eval_steps])
        print(row)

    # -------------------------------------------------------------------------
    # Scientific Interpretations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCIENTIFIC INTERPRETATION OF DIAGNOSTICS")
    print("=" * 80)

    # D2 Interpretation
    mean_g_all = np.mean(d2["mean_abs_gain_per_step"])
    var_mean_curve = d2["var_of_mean_across_time"]
    print("\n[Diagnostic 2 Interpretation]:")
    if mean_g_all < 1e-3:
        print("-> g_t -> 0: The network has effectively shut down corrections. State noise / opposing gradients suppressed gain activity.")
    elif var_mean_curve < 1e-6:
        print(f"-> g_t is FLAT across time (Var_t = {var_mean_curve:.2e}). The GRU failed to learn sequential/recurrent dynamics and acts as a static bias layer.")
    else:
        print(f"-> g_t is ACTIVE and time-varying (Mean = {mean_g_all:.4f}, Var_t = {var_mean_curve:.2e}).")
        print("   The GRU is outputting dynamic signals, but the current formulation/gradients may be misaligned with trajectory needs.")

    # D3 Interpretation
    g1 = d3["gradient_norms"][1]
    g25 = d3["gradient_norms"][25]
    ratio_25_1 = g25 / max(g1, 1e-12)
    print("\n[Diagnostic 3 Interpretation]:")
    print(f"-> Gradient Norm Ratio G(25) / G(1) = {ratio_25_1:.4f}")
    if ratio_25_1 < 0.05:
        print("-> Severe Vanishing Gradients: Gradients from step 25 vanish before they can train the recurrent weights.")
    elif ratio_25_1 > 20.0:
        print("-> Exploding/Dominant Gradients from step 25: Gradients from step 25 heavily overpower step 1.")
    else:
        print("-> Gradient norms between step 1 and step 25 are within comparable orders of magnitude.")

    print(f"\n-> Cosine Similarity: cos(grad_1, grad_25) = {cos_1_25:+.4f}")
    if cos_1_25 is not None and cos_1_25 < -0.2:
        print("-> SEVERE GRADIENT CONFLICT: cos(theta) is negative! Updates to fix step 25 directly contradict and destroy predictions at step 1.")
        print("   Because the U-shaped loss weights both steps heavily, the optimizer oscillates or stalls in a valley between opposing objectives.")
    elif cos_1_25 is not None and abs(cos_1_25) < 0.2:
        print("-> Orthogonal Gradients: Gradients between step 1 and step 25 are nearly orthogonal (little direct destructive interference, but independent parameter competition).")
    else:
        print("-> Gradients between step 1 and step 25 are positively aligned.")

    # Save report and plot
    report = {
        "diagnostic_2": d2,
        "diagnostic_3": d3,
        "interpretations": {
            "mean_gain_overall": mean_g_all,
            "gradient_ratio_25_to_1": ratio_25_1,
            "cosine_similarity_t1_t25": cos_1_25,
        }
    }
    report_path = output_dir / "diagnostic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Diagnostic] Full report saved to: {report_path}")

    plot_diagnostics(d2, d3, output_dir)


if __name__ == "__main__":
    main()
