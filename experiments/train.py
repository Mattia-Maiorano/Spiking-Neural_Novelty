"""
Main Training Entrypoint for SPWM (v0.2.0).
Streamlined, config-first Single Source of Truth with immutable ExperimentConfig
and scoped strict checkpoint loading (`model.pt`) across M1 -> M2 -> M3 phases.

Usage:
    python experiments/train.py --config configs/experiments/spwm_v0_1_0.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import chronicle
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator, get_phase_eval_horizons
from spwm.evaluation.metrics_resume import (
    get_canonical_metrics_filename,
    save_metrics_resume,
)
from spwm.evaluation.visualizer import (
    plot_latent_rollout_degradation,
    plot_multi_step_degradation,
    plot_physical_rollout_degradation,
    plot_relay_rollout,
    plot_training_curves,
)
from spwm.models.rollout import rollout_latent
from spwm.models.world_model import SPWM
from spwm.training.diagnostics import (
    collect_spike_statistics,
    inspect_gradients,
)
from spwm.training.engine import (
    BaseTrainer,
    enforce_phase_prerequisites,
)
from spwm.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPWM with immutable typed config schema.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to config YAML or Python module (defaults to ExperimentConfig())",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["train_representation", "train_m1", "train_m2"],
        help="Override training mode (train_representation, train_m1, or train_m2)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience (epochs without validation improvement before switching/stopping)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--representation-checkpoint",
        type=str,
        default=None,
        help="Representation checkpoint (model.pt) for Phase 1 (train_m1) training",
    )
    parser.add_argument(
        "--m1-checkpoint",
        type=str,
        default=None,
        help="M1 checkpoint (model.pt) for Phase 2 (train_m2) training",
    )
    parser.add_argument("--phase-horizon", type=int, default=None, help="Predictor unroll horizon during training")
    parser.add_argument("--temporal-burn-in", type=int, default=None, help="Override temporal burn-in steps for recurrent warm-up")
    return parser.parse_args()


def _evaluate_and_save_phase_artifacts(
    model: SPWM,
    cfg: ExperimentConfig,
    phase_mode: str,
    save_dir: Path,
    history: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a trained phase checkpoint (`model.pt`) with its exact horizon rules and save plots/metrics."""
    best_model_pt = save_dir / "model.pt"
    if best_model_pt.is_file():
        ckpt = torch.load(best_model_pt, map_location=device, weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
    model.eval()

    figures_dir = save_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    eval_horizons = get_phase_eval_horizons(
        mode=phase_mode,
        phase_horizon=cfg.training.phase_horizon,
        config_horizons=cfg.evaluation.rollout_horizons,
    )
    fixed_predictor_eval = model.m1 if phase_mode == "train_m1" else None
    eval_seq_len = max(cfg.data.sequence_length, cfg.training.temporal_burn_in + max(eval_horizons) + 1)

    eval_loaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=eval_seq_len,
        batch_size=cfg.data.batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        num_objects=cfg.environment.num_objects,
        standard_velocity_range=cfg.data.standard_velocity_range,
        extrapolation_velocity_range=cfg.data.extrapolation_velocity_range,
        drift_injection_prob=0.0,
        drift_magnitude=cfg.data.drift_magnitude,
        mixed_drift=cfg.data.mixed_drift,
        event_threshold=cfg.environment.event_threshold,
        box_size=cfg.environment.box_size,
        dt=cfg.environment.dt,
        num_workers=cfg.data.num_workers,
        cache_data=cfg.data.cache_data,
    )

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=eval_horizons,
        num_objects=cfg.environment.num_objects,
        temporal_burn_in=cfg.training.temporal_burn_in,
        fixed_predictor=fixed_predictor_eval,
    )
    test_metrics = evaluator.evaluate_dataset(eval_loaders["test"])
    chronicle.log_success(
        f"Test Evaluation [{phase_mode.upper()}]: Horizons={test_metrics.horizons} | "
        f"Rollout MAE = {test_metrics.rollout_mae:.5f} | "
        f"Rollout Score (Dynamic) = {test_metrics.dynamic_score:.5f} | "
        f"Drift Ratio = {test_metrics.drift_ratio:.2f}x"
    )

    if history:
        plot_training_curves(history, figures_dir / "training_curves.png")

    pos_span = 2.0 * float(cfg.environment.box_size)
    vel_span = float(cfg.data.standard_velocity_range[1] - cfg.data.standard_velocity_range[0])
    plot_physical_rollout_degradation(
        horizons=test_metrics.horizons,
        pos_errors=test_metrics.position_error_per_horizon,
        vel_errors=test_metrics.velocity_error_per_horizon,
        save_path=figures_dir / "physical_rollout_degradation.png",
        relative=True,
        pos_span=pos_span,
        vel_span=vel_span,
    )
    plot_latent_rollout_degradation(
        horizons=test_metrics.horizons,
        latent_errors=test_metrics.latent_mse_per_horizon,
        q_errors=test_metrics.q_mse_per_horizon,
        p_errors=test_metrics.p_mse_per_horizon,
        save_path=figures_dir / "latent_rollout_degradation.png",
        relative=True,
    )

    sample_batch = next(iter(eval_loaders["test"]))
    events = sample_batch["events"].to(device)
    flat_kin = sample_batch["flat_kinematics"].to(device)
    burn_in = min(cfg.training.temporal_burn_in, max(0, events.shape[1] - 2))
    with torch.no_grad():
        out = model(events)
        z_start = out.latent_states[:, burn_in]
        decoder_state = model.physical_decoder.warm_up_filter(out.latent_states[:, : burn_in + 1])
        max_mode_h = max(eval_horizons)
        eval_h = min(max_mode_h, events.shape[1] - 1 - burn_in)
        roll_res = rollout_latent(
            model=model,
            z_init=z_start,
            horizon=eval_h,
            with_grad=False,
            decode_kinematics=True,
            initial_decoder_state=decoder_state,
            fixed_predictor=fixed_predictor_eval,
        )
        if roll_res.decoded_kinematics is not None:
            gt_kin = flat_kin[:, burn_in + 1 : burn_in + 1 + eval_h]
            relay_plot_path = figures_dir / "relay_rollout.png"
            plot_relay_rollout(
                pred_traj=roll_res.decoded_kinematics,
                gt_traj=gt_kin,
                phase_horizon=cfg.training.phase_horizon,
                save_path=relay_plot_path,
            )
            chronicle.log_success(f"Generated Symplectic Predictor (M1) rollout plot: {relay_plot_path.name}")

    total_params = sum(p.numel() for p in model.parameters())
    version_tag = cfg.project.name
    model_display_name = f"{cfg.project.name}_{phase_mode}"
    metrics_resume_path = save_dir / f"metrics_{version_tag}.json"
    save_metrics_resume(
        target_path=metrics_resume_path,
        history=history,
        evaluation_result=test_metrics,
        parameters=total_params,
        num_segments=10,
        model_name=model_display_name,
        version=version_tag,
        pos_span=pos_span,
        vel_span=vel_span,
    )
    chronicle.log_success(f"Generated metrics resume: {metrics_resume_path.name}")

    with torch.no_grad():
        all_spikes = torch.cat([out.fast_spikes, out.slow_spikes], dim=-1)
        spike_diag = collect_spike_statistics(all_spikes)
    grad_diag = inspect_gradients(model)
    diag_summary = {
        "model_name": model_display_name,
        "version": version_tag,
        "mode": phase_mode,
        "spike_statistics": spike_diag,
        "gradient_diagnostics": {
            "total_norm": grad_diag["total_norm"],
            "has_nan": grad_diag["has_nan"],
            "has_inf": grad_diag["has_inf"],
            "zero_gradient_ratio": grad_diag["zero_gradient_ratio"],
        },
    }
    diag_path = save_dir / "diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)
    chronicle.log_success(f"Generated run diagnostics: {diag_path.name}")
    return {"test_metrics": test_metrics, "diagnostics": diag_summary}


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.load(args.config) if args.config is not None else ExperimentConfig()

    overrides: Dict[str, Any] = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.mode is not None:
        overrides["mode"] = args.mode
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.patience is not None:
        overrides["early_stopping_patience"] = args.patience
    if args.representation_checkpoint is not None:
        overrides["representation_checkpoint"] = args.representation_checkpoint
    if args.m1_checkpoint is not None:
        overrides["m1_checkpoint"] = args.m1_checkpoint
    if args.phase_horizon is not None:
        overrides["phase_horizon"] = args.phase_horizon
    if args.temporal_burn_in is not None:
        overrides["temporal_burn_in"] = args.temporal_burn_in

    if overrides:
        cfg = cfg.with_overrides(**overrides)

    set_seed(cfg.project.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    mode = cfg.training.mode
    save_dir = Path(args.output_dir) if args.output_dir is not None else Path("results") / f"{cfg.project.name}_{mode}"
    save_dir.mkdir(parents=True, exist_ok=True)
    results_root = save_dir.parent

    dataloaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=cfg.data.sequence_length,
        batch_size=cfg.data.batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        num_objects=cfg.environment.num_objects,
        standard_velocity_range=cfg.data.standard_velocity_range,
        extrapolation_velocity_range=cfg.data.extrapolation_velocity_range,
        drift_injection_prob=0.0,
        drift_magnitude=cfg.data.drift_magnitude,
        mixed_drift=cfg.data.mixed_drift,
        event_threshold=cfg.environment.event_threshold,
        box_size=cfg.environment.box_size,
        dt=cfg.environment.dt,
        num_workers=cfg.data.num_workers,
        cache_data=cfg.data.cache_data,
    )

    model = SPWM.from_config(cfg).to(device)

    # Auto-enforced strict loading of best validation `model.pt` from matching version
    enforce_phase_prerequisites(model, cfg, device=device, results_root=results_root)

    engine = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        device=device,
        save_dir=save_dir,
        results_root=results_root,
    )

    resume_ckpt = save_dir / "checkpoint.pt"
    if resume_ckpt.is_file():
        engine.load_checkpoint(resume_ckpt)
        chronicle.log_info(f"Resumed from checkpoint: {resume_ckpt} at epoch {engine.start_epoch}")

    history = engine.fit()
    cfg.save(save_dir / "config.yaml")

    _evaluate_and_save_phase_artifacts(
        model=model,
        cfg=cfg,
        phase_mode=mode,
        save_dir=save_dir,
        history=history,
        device=device,
    )


if __name__ == "__main__":
    main()