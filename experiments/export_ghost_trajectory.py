"""
Ghost Execution Exporter for Unity (SPWM-v0.4.0).

Runs the canonical RolloutEvaluator protocol on one test trajectory and serializes
ground-truth vs. M1-predicted joint angles into `sim/Assets/StreamingAssets/trajectory_ghost_v0_4_0.json`.

Usage:
    python experiments/export_ghost_trajectory.py \
        --checkpoint results/spwm_v0_4_0_train_m1/model.pt \
        --config results/spwm_v0_4_0_train_m1/config.yaml \
        [--batch-index 0] [--sample-index 0] [--horizon 100]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chronicle
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import DEFAULT_UNITY_PROJECT_DIR, GhostVisualizerExporter
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed

# Actuated joint order of `ArticulatedRobotRig` (6-DOF ManipulatorArm), matching q* column order.
MANIPULATOR_JOINT_NAMES = [
    "Shoulder_Pan", "Shoulder_Lift", "Elbow_Flex",
    "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SPWM ghost trajectory for Unity playback.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt) or run dir")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--unity-project", type=str, default=str(DEFAULT_UNITY_PROJECT_DIR), help="Unity project root")
    parser.add_argument("--horizon", type=int, default=100, help="Rollout horizon H")
    parser.add_argument("--batch-index", type=int, default=0, help="Test-loader batch to sample from")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample within the batch")
    parser.add_argument("--model-version", type=str, default="spwm_v0_5_0")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
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
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "model.pt"

    chronicle.log_info(f"Loading checkpoint strictly: {ckpt_path} on {device}")
    model = SPWM.from_config(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    seq_len = max(cfg.data.sequence_length, cfg.training.temporal_burn_in + args.horizon + 1)
    dataloaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=seq_len,
        batch_size=cfg.data.batch_size,
        height=cfg.environment.height,
        width=cfg.environment.width,
        joint_count=cfg.environment.joint_count,
        action_dim=cfg.environment.action_dim,
        in_channels=cfg.model.in_channels,
        event_threshold=cfg.environment.event_threshold,
        dt=cfg.environment.dt,
        dataset_dir=cfg.data.dataset_dir,
        strict_dataset=cfg.data.strict_dataset,
        cache_data=False,
    )

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=[args.horizon],
        joint_count=cfg.environment.joint_count,
        temporal_burn_in=cfg.training.temporal_burn_in,
        fixed_predictor=model.m1,
    )
    joint_names = MANIPULATOR_JOINT_NAMES if cfg.environment.joint_count == len(MANIPULATOR_JOINT_NAMES) else None
    exporter = GhostVisualizerExporter(
        evaluator=evaluator,
        unity_project_dir=args.unity_project,
        model_version=args.model_version,
        dt=cfg.environment.dt,
        horizon=args.horizon,
        joint_names=joint_names,
    )

    ghost = exporter.capture(dataloaders["test"], batch_index=args.batch_index, sample_index=args.sample_index)
    out_path = exporter.save(ghost)

    s = ghost.metadata["summary"]
    chronicle.log_success(f"Ghost trajectory written: {out_path}")
    chronicle.log_success(
        f"H={ghost.metadata['horizon']} | traj_id={ghost.metadata.get('trajectory_id')} | "
        f"pos_err mean/final={s['mean_position_abs_error']:.4f}/{s['final_position_abs_error']:.4f} | "
        f"vel_rel final={s['final_velocity_relative_degradation']:.3f}"
    )


if __name__ == "__main__":
    main()
