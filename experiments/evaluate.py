"""
Evaluation Entrypoint for SPWM Checkpoints.
Usage:
    python experiments/evaluate.py --checkpoint results/<run_id>/model.pt --config results/<run_id>/config.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import chronicle
import torch

from spwm.config.schema import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import plot_multi_step_degradation
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained SPWM model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--use-corrector", action="store_true", default=None, help="Force use of corrector in rollouts")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.project.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ckpt_path = Path(args.checkpoint)
    out_dir = ckpt_path.parent
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    use_corrector = args.use_corrector
    if use_corrector is None:
        use_corrector = (cfg.training.mode == "train_corrector") or ("corrector" in str(ckpt_path).lower())

    chronicle.log_info(f"Loading checkpoint: {ckpt_path} on {device} (use_corrector: {use_corrector})")
    model = SPWM.from_config(cfg, enable_corrector=use_corrector).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    drift_prob = cfg.data.drift_injection_prob if use_corrector else 0.0
    dataloaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=cfg.data.sequence_length,
        batch_size=cfg.data.batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        num_objects=cfg.environment.num_objects,
        event_threshold=cfg.environment.event_threshold,
        box_size=cfg.environment.box_size,
        dt=cfg.environment.dt,
        drift_injection_prob=drift_prob,
        mixed_drift=cfg.data.mixed_drift,
    )

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=cfg.evaluation.rollout_horizons,
        num_objects=cfg.environment.num_objects,
        use_corrector=use_corrector,
    )
    results = evaluator.evaluate_dataset(dataloaders["test"])

    chronicle.log_success(f"Teacher Forcing MSE: {results.teacher_forcing_mse:.5f}")
    chronicle.log_success(f"Rollout MAE: {results.rollout_mae:.5f}")
    chronicle.log_success(f"Drift Ratio: {results.drift_ratio:.2f}x")

    plot_multi_step_degradation(
        horizons=cfg.evaluation.rollout_horizons,
        pos_errors=results.position_error_per_horizon,
        save_path=figures_dir / "eval_rollout_degradation.png",
    )


if __name__ == "__main__":
    main()
