"""
Comparative Baseline Training & Multi-Horizon Evaluation Runner (DOSSIER Section 6.1).

Trains and benchmarks non-spiking deep recurrent world models against SPWM-v0.7.3:
1. Continuous RSSM (DreamerV3 style: h_t in R^256, s_t in R^32)
2. Deep GRU Predictor (2-layer GRU matched to ~328k parameters)
3. Continuous ConvLSTM (Spatial-temporal recurrence on 2D convolutional feature maps)

Evaluates multi-step autonomous rollout metrics across H in {1, 10, 50, 100},
computes parameter counts and single-step inference latency (ms),
and compiles a publication-ready comparative metrics table and degradation figures.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chronicle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders
from spwm.models.baselines import (
    ContinuousConvLSTM,
    ContinuousRSSM,
    DeepGRUPredictor,
)
from spwm.models.rollout import compute_rollout_schedule
from spwm.training.metrics import position_error, velocity_error
from spwm.utils.reproducibility import set_seed


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def get_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency_ms(
    model: nn.Module,
    device: torch.device,
    in_channels: int = 4,
    height: int = 32,
    width: int = 32,
    action_dim: int = 6,
    warmup_steps: int = 10,
    bench_steps: int = 100,
) -> float:
    """Measures single-frame forward inference latency in milliseconds."""
    model = model.to(device)
    model.eval()
    dummy_events = torch.randn(1, 1, in_channels, height, width, device=device)
    dummy_actions = torch.randn(1, 1, action_dim, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(dummy_events, actions=dummy_actions)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(bench_steps):
            _ = model(dummy_events, actions=dummy_actions)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
    t1 = time.perf_counter()

    latency_ms = ((t1 - t0) / bench_steps) * 1000.0
    return float(latency_ms)


def train_single_baseline(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    save_dir: Path,
    joint_count: int = 6,
) -> Dict[str, Any]:
    chronicle.log_info(f"--- Training {model_name} on {device} ({epochs} epochs max) ---")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = save_dir / "model.pt"
    history: List[Dict[str, float]] = []

    for ep in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_pos_losses: List[float] = []
        train_vel_losses: List[float] = []

        t0 = time.perf_counter()
        for batch in train_loader:
            events = batch["events"].to(device)
            actions = batch["actions"].to(device)
            kin_gt = batch["flat_kinematics"].to(device)
            pos_gt = kin_gt[..., :joint_count]
            vel_gt = kin_gt[..., joint_count:2 * joint_count]

            optimizer.zero_grad()
            out = model(events, actions=actions)
            pred_pos = out["positions"]
            pred_vel = out["velocities"]

            loss_pos = torch.mean((pred_pos - pos_gt) ** 2)
            loss_vel = torch.mean((pred_vel - vel_gt) ** 2)
            total_loss = loss_pos + 1.5 * loss_vel

            if "kl_loss" in out:
                total_loss = total_loss + 0.1 * out["kl_loss"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(total_loss.item()))
            train_pos_losses.append(float(loss_pos.item()))
            train_vel_losses.append(float(loss_vel.item()))

        # Validation
        model.eval()
        val_losses: List[float] = []
        val_pos_losses: List[float] = []
        val_vel_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                events = batch["events"].to(device)
                actions = batch["actions"].to(device)
                kin_gt = batch["flat_kinematics"].to(device)
                pos_gt = kin_gt[..., :joint_count]
                vel_gt = kin_gt[..., joint_count:2 * joint_count]

                out = model(events, actions=actions)
                loss_pos = torch.mean((out["positions"] - pos_gt) ** 2)
                loss_vel = torch.mean((out["velocities"] - vel_gt) ** 2)
                v_loss = loss_pos + 1.5 * loss_vel
                if "kl_loss" in out:
                    v_loss = v_loss + 0.1 * out["kl_loss"]

                val_losses.append(float(v_loss.item()))
                val_pos_losses.append(float(loss_pos.item()))
                val_vel_losses.append(float(loss_vel.item()))

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        dt_ep = time.perf_counter() - t0

        record = {
            "epoch": ep,
            "train_loss": mean_train,
            "val_loss": mean_val,
            "val_pos_mse": float(np.mean(val_pos_losses)),
            "val_vel_mse": float(np.mean(val_vel_losses)),
            "duration": dt_ep,
        }
        history.append(record)

        is_best = mean_val < best_val_loss
        if is_best:
            best_val_loss = mean_val
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "history": history}, best_model_path)
            star = " *"
        else:
            epochs_no_improve += 1
            star = ""

        chronicle.log_info(
            f"[{model_name}] Ep {ep:02d}/{epochs:02d} | Train: {mean_train:.4f} | Val: {mean_val:.4f} "
            f"(Pos: {record['val_pos_mse']:.4f}, Vel: {record['val_vel_mse']:.4f}) | {dt_ep:.1f}s{star}"
        )

        if epochs_no_improve >= patience:
            chronicle.log_info(f"[{model_name}] Early stopping triggered after {patience} epochs without improvement.")
            break

    # Load best checkpoint strictly
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return {"history": history, "best_val_loss": best_val_loss}


@torch.no_grad()
def evaluate_baseline_rollout(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    horizons: List[int],
    temporal_burn_in: int = 5,
    joint_count: int = 6,
) -> Dict[str, Any]:
    """Evaluates multi-step autonomous rollout across requested horizons using canonical schedule."""
    model.eval()
    max_h = max(horizons)

    pos_mse_dict: Dict[int, List[float]] = {h: [] for h in horizons}
    vel_mse_dict: Dict[int, List[float]] = {h: [] for h in horizons}
    pos_err_dict: Dict[int, List[float]] = {h: [] for h in horizons}
    vel_err_dict: Dict[int, List[float]] = {h: [] for h in horizons}

    for batch in test_loader:
        events = batch["events"].to(device)
        actions = batch["actions"].to(device)
        true_kin = batch["flat_kinematics"].to(device)
        phase_flags = batch["phase_flags"].to(device)

        b, t_seq, _, _, _ = events.shape
        t_start, max_eval_h = compute_rollout_schedule(
            phase_flags=phase_flags,
            sequence_length=t_seq,
            temporal_burn_in=temporal_burn_in,
        )

        eval_h = min(max_h, max_eval_h)
        if eval_h <= 0:
            continue

        rollout_res = model.rollout(
            events=events,
            actions=actions,
            horizon=eval_h,
            t_start=t_start,
        )

        target_kin = true_kin[:, t_start + 1 : t_start + 1 + eval_h]
        pred_kin = rollout_res.trajectory

        for h in horizons:
            if h <= eval_h:
                p_sub = pred_kin[:, :h]
                t_sub = target_kin[:, :h]

                pred_pos = p_sub[..., :joint_count]
                true_pos = t_sub[..., :joint_count]
                pred_vel = p_sub[..., joint_count:2 * joint_count]
                true_vel = t_sub[..., joint_count:2 * joint_count]

                p_mse = float(torch.mean((pred_pos - true_pos) ** 2).item())
                v_mse = float(torch.mean((pred_vel - true_vel) ** 2).item())
                p_err = position_error(p_sub, t_sub, joint_count=joint_count)
                v_err = velocity_error(p_sub, t_sub, joint_count=joint_count)

                pos_mse_dict[h].append(p_mse)
                vel_mse_dict[h].append(v_mse)
                pos_err_dict[h].append(p_err)
                vel_err_dict[h].append(v_err)

    return {
        "position_mse_per_horizon": {h: float(np.mean(pos_mse_dict[h])) if pos_mse_dict[h] else 0.0 for h in horizons},
        "velocity_mse_per_horizon": {h: float(np.mean(vel_mse_dict[h])) if vel_mse_dict[h] else 0.0 for h in horizons},
        "position_error_per_horizon": {h: float(np.mean(pos_err_dict[h])) if pos_err_dict[h] else 0.0 for h in horizons},
        "velocity_error_per_horizon": {h: float(np.mean(vel_err_dict[h])) if vel_err_dict[h] else 0.0 for h in horizons},
    }


def plot_comparative_degradation(
    benchmarks: Dict[str, Dict[str, Any]],
    save_path: Path,
) -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        GLOBAL_VERSION.formatted: "#10B981",       # Green
        "Continuous RSSM": "#3B82F6",   # Blue
        "Deep GRU": "#F59E0B",          # Orange
        "Continuous ConvLSTM": "#8B5CF6",# Purple
    }
    markers = {
        GLOBAL_VERSION.formatted: "o",
        "Continuous RSSM": "s",
        "Deep GRU": "^",
        "Continuous ConvLSTM": "d",
    }

    # 1. Position MSE plot
    ax_pos = axes[0]
    for name, data in benchmarks.items():
        pos_mses = data["position_mse_per_horizon"]
        horizons = sorted(int(h) for h in pos_mses.keys())
        values = [pos_mses[str(h) if str(h) in pos_mses else h] for h in horizons]
        col = colors.get(name, "#6B7280")
        mk = markers.get(name, "o")
        ax_pos.plot(horizons, values, marker=mk, color=col, label=f"{name}", linewidth=2.2)

    ax_pos.set_title(r"Position Rollout Degradation ($\mathrm{MSE}_{\mathrm{pos}}$)")
    ax_pos.set_xlabel(r"Rollout Horizon ($H$)")
    ax_pos.set_ylabel(r"Mean Squared Error ($\mathrm{rad}^2$)")
    ax_pos.set_yscale("log")
    ax_pos.legend(frameon=True)
    ax_pos.grid(True, which="both", linestyle="--", alpha=0.5)

    # 2. Velocity MSE plot
    ax_vel = axes[1]
    for name, data in benchmarks.items():
        vel_mses = data["velocity_mse_per_horizon"]
        horizons = sorted(int(h) for h in vel_mses.keys())
        values = [vel_mses[str(h) if str(h) in vel_mses else h] for h in horizons]
        col = colors.get(name, "#6B7280")
        mk = markers.get(name, "o")
        ax_vel.plot(horizons, values, marker=mk, color=col, label=f"{name}", linewidth=2.2)

    ax_vel.set_title(r"Velocity Rollout Degradation ($\mathrm{MSE}_{\mathrm{vel}}$)")
    ax_vel.set_xlabel(r"Rollout Horizon ($H$)")
    ax_vel.set_ylabel(r"Mean Squared Error ($(\mathrm{rad}/\mathrm{s})^2$)")
    ax_vel.set_yscale("log")
    ax_vel.legend(frameon=True)
    ax_vel.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    chronicle.log_success(f"Comparative degradation figure saved to: {save_path}")


def generate_markdown_table(benchmarks: Dict[str, Dict[str, Any]], horizons: List[int]) -> str:
    lines = []
    lines.append("| Model Architecture | Params | Latency (ms) | " + " | ".join(f"Pos MSE (H={h})" for h in horizons) + " | " + " | ".join(f"Vel MSE (H={h})" for h in horizons) + " |")
    lines.append("| :--- | :---: | :---: | " + " | ".join(":---:" for _ in horizons) + " | " + " | ".join(":---:" for _ in horizons) + " |")

    for name, data in benchmarks.items():
        params_k = f"{data['parameter_count'] / 1000.0:.1f}k"
        latency = f"{data['inference_latency_ms']:.2f}"
        pos_vals = [f"{data['position_mse_per_horizon'].get(str(h), data['position_mse_per_horizon'].get(h, 0.0)):.5f}" for h in horizons]
        vel_vals = [f"{data['velocity_mse_per_horizon'].get(str(h), data['velocity_mse_per_horizon'].get(h, 0.0)):.5f}" for h in horizons]
        row = f"| **{name}** | {params_k} | {latency} | " + " | ".join(pos_vals) + " | " + " | ".join(vel_vals) + " |"
        lines.append(row)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate comparative non-spiking baseline models.")
    parser.add_argument("--epochs", type=int, default=18, help="Max epochs per baseline")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    parser.add_argument("--total-trajectories", type=int, default=1000, help="Total trajectories to partition")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/baselines", help="Output directory")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    chronicle.log_info(f"Target execution device: {device}")

    eval_horizons = [1, 10, 50, 100]
    seq_len = 105

    chronicle.log_info(f"Creating DataLoaders: total={args.total_trajectories}, seq_len={seq_len}, batch={args.batch_size}")
    dataloaders = create_dataloaders(
        total_trajectories=args.total_trajectories,
        train_split=0.8,
        val_split=0.1,
        sequence_length=seq_len,
        batch_size=args.batch_size,
        height=32,
        width=32,
        joint_count=6,
        action_dim=6,
        in_channels=4,
        dataset_dir="sim/ExportedTrajectories",
        cache_data=True,
        strict_dataset=True,
    )

    # Models to benchmark
    models: Dict[str, nn.Module] = {
        "Continuous RSSM": ContinuousRSSM(
            in_channels=4,
            height=32,
            width=32,
            encoder_dim=128,
            deterministic_dim=256,
            stochastic_dim=32,
            action_dim=6,
            joint_count=6,
        ),
        "Deep GRU": DeepGRUPredictor(
            in_channels=4,
            height=32,
            width=32,
            encoder_dim=128,
            hidden_dim=164,
            action_dim=6,
            joint_count=6,
        ),
        "Continuous ConvLSTM": ContinuousConvLSTM(
            in_channels=4,
            height=32,
            width=32,
            conv_channels=(32, 48),
            action_dim=6,
            action_embed_dim=16,
            hidden_channels=56,
            joint_count=6,
        ),
    }

    benchmark_results: Dict[str, Dict[str, Any]] = {}

    # 1. Train and evaluate each baseline
    for name, model in models.items():
        model_slug = name.lower().replace(" ", "_").replace("-", "_")
        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        params = get_parameter_count(model)
        latency_ms = measure_inference_latency_ms(model, device)
        chronicle.log_info(f"[{name}] Trainable Parameters: {params:,} | Single-frame Latency: {latency_ms:.2f} ms")

        train_res = train_single_baseline(
            model_name=name,
            model=model,
            train_loader=dataloaders["train"],
            val_loader=dataloaders["val"],
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            save_dir=model_dir,
            joint_count=6,
        )

        eval_res = evaluate_baseline_rollout(
            model=model,
            test_loader=dataloaders["test"],
            device=device,
            horizons=eval_horizons,
            temporal_burn_in=5,
            joint_count=6,
        )

        benchmark_results[name] = {
            "parameter_count": params,
            "inference_latency_ms": latency_ms,
            "position_mse_per_horizon": eval_res["position_mse_per_horizon"],
            "velocity_mse_per_horizon": eval_res["velocity_mse_per_horizon"],
            "position_error_per_horizon": eval_res["position_error_per_horizon"],
            "velocity_error_per_horizon": eval_res["velocity_error_per_horizon"],
            "best_val_loss": train_res["best_val_loss"],
        }
        chronicle.log_success(
            f"[{name}] Evaluated: Pos MSE (H=100): {eval_res['position_mse_per_horizon'][100]:.5f} | "
            f"Vel MSE (H=100): {eval_res['velocity_mse_per_horizon'][100]:.5f}"
        )

    # 2. Incorporate SPWM production baseline metrics
    spwm_metrics_file = GLOBAL_VERSION.m1_metrics()
    if spwm_metrics_file.exists():
        with open(spwm_metrics_file, "r", encoding="utf-8") as f:
            spwm_data = json.load(f)

        # Profile SPWM latency on current device
        from spwm.contracts.config import ExperimentConfig
        from spwm.models.world_model import SPWM
        spwm_cfg = ExperimentConfig.load(GLOBAL_VERSION.m1_config())
        spwm_model = SPWM.from_config(spwm_cfg).to(device)
        spwm_ckpt = torch.load(GLOBAL_VERSION.m1_checkpoint(), map_location=device, weights_only=False)
        spwm_state = spwm_ckpt["model_state"] if "model_state" in spwm_ckpt else spwm_ckpt
        spwm_model.load_state_dict(spwm_state, strict=True)
        spwm_latency = measure_inference_latency_ms(spwm_model, device)

        spwm_eval = spwm_data["evaluation"]
        pos_mses = {int(h): spwm_eval["position_mse_per_horizon"][str(h)] for h in eval_horizons if str(h) in spwm_eval["position_mse_per_horizon"]}
        vel_mses = {int(h): spwm_eval["velocity_mse_per_horizon"][str(h)] for h in eval_horizons if str(h) in spwm_eval["velocity_mse_per_horizon"]}
        pos_errs = {int(h): spwm_eval["position_error_per_horizon"][str(h)] for h in eval_horizons if str(h) in spwm_eval["position_error_per_horizon"]}
        vel_errs = {int(h): spwm_eval["velocity_error_per_horizon"][str(h)] for h in eval_horizons if str(h) in spwm_eval["velocity_error_per_horizon"]}

        # Insert SPWM as first entry
        final_benchmarks = {
            GLOBAL_VERSION.formatted: {
                "parameter_count": spwm_data["parameters"],
                "inference_latency_ms": spwm_latency,
                "position_mse_per_horizon": pos_mses,
                "velocity_mse_per_horizon": vel_mses,
                "position_error_per_horizon": pos_errs,
                "velocity_error_per_horizon": vel_errs,
                "mean_spike_rate": spwm_eval["mean_spike_rate"],
            }
        }
        final_benchmarks.update(benchmark_results)
    else:
        final_benchmarks = benchmark_results

    # 3. Save comparative metrics JSON
    metrics_path = output_dir / "comparative_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_benchmarks, f, indent=2)
    chronicle.log_success(f"Comparative benchmark metrics saved to: {metrics_path}")

    # 4. Generate Degradation Plot
    plot_comparative_degradation(final_benchmarks, output_dir / "baseline_rollout_comparison.png")

    # 5. Output Markdown Table
    md_table = generate_markdown_table(final_benchmarks, eval_horizons)
    md_path = output_dir / "comparative_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    print("\n" + "=" * 80)
    print("COMPARATIVE BENCHMARK TABLE (DOSSIER SECTION 6.1)")
    print("=" * 80)
    print(md_table)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
