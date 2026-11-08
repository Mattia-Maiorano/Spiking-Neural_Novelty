"""
Ghost Execution Exporter for Unity.

Runs the canonical RolloutEvaluator protocol on test trajectories and serializes
ground-truth vs. M1-predicted joint angles into `sim/Assets/StreamingAssets/trajectory_ghost.json`
(or multi-episode / injection comparison JSON files).

Usage:
    python experiments/export_ghost_trajectory.py \
        --checkpoint results/<model_dir>/model.pt \
        --config results/<model_dir>/config.yaml \
        [--episodes 5] [--compare-injection] [--horizon 100]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Tuple

import chronicle
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders
from spwm.data.unity_episode_meta import episode_meta, weight_class
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import DEFAULT_UNITY_PROJECT_DIR, GhostTrajectory, GhostVisualizerExporter
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed

# Actuated joint order of `ArticulatedRobotRig` (6-DOF ManipulatorArm), matching q* column order.
MANIPULATOR_JOINT_NAMES = [
    "Shoulder_Pan", "Shoulder_Lift", "Elbow_Flex",
    "Wrist_Pitch", "Wrist_Yaw", "Wrist_Roll",
]


def get_sample_from_loader(data_loader: Any, global_index: int) -> Tuple[Mapping[str, Any], int, int]:
    current = 0
    for b_idx, batch in enumerate(data_loader):
        batch_len = int(batch["events"].shape[0])
        if current + batch_len > global_index:
            s_idx = global_index - current
            return batch, b_idx, s_idx
        current += batch_len
    raise IndexError(f"global_index {global_index} exceeds total samples ({current}) in loader")


def log_ghost_summary(ghost: GhostTrajectory, out_path: Path) -> None:
    s = ghost.metadata["summary"]
    inj_str = "with_injection" if ghost.metadata.get("has_sensory_injection", False) else "no_injection"
    chronicle.log_success(f"Ghost trajectory written: {out_path} [{inj_str}]")
    chronicle.log_success(
        f"H={ghost.metadata['horizon']} | traj_id={ghost.metadata.get('trajectory_id')} | "
        f"pos_err mean/final={s['mean_position_abs_error']:.4f}/{s['final_position_abs_error']:.4f} | "
        f"vel_rel final={s['final_velocity_relative_degradation']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SPWM ghost trajectory for Unity playback.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt) or run dir")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--unity-project", type=str, default=str(DEFAULT_UNITY_PROJECT_DIR), help="Unity project root")
    parser.add_argument("--horizon", type=int, default=100, help="Rollout horizon H")
    parser.add_argument("--batch-index", type=int, default=0, help="Test-loader batch to sample from")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample within the batch")
    parser.add_argument("--episodes", type=int, default=1, help="Number of distinct test episodes to export")
    parser.add_argument("--episode-index", type=int, default=None, help="Specific episode index to export")
    parser.add_argument("--contiguous", action="store_true", help="Sample consecutive episodes instead of spreading across the test set")
    parser.add_argument("--compare-injection", action="store_true", help="Export each episode both with and without sensory injection")
    parser.add_argument("--output", type=str, default=None, help="Path to save the trajectory JSON (for single export)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for exported trajectory files")
    parser.add_argument("--model-version", type=str, default=GLOBAL_VERSION.snake_case)
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--no-injection", action="store_true", help="Disable active perception sensory injection during rollout")
    parser.add_argument("--checkpoint-b", type=str, default=None, help="Path to secondary model checkpoint")
    parser.add_argument("--config-b", type=str, default=None, help="Path to secondary config YAML")
    parser.add_argument("--model-version-b", type=str, default=f"{GLOBAL_VERSION.snake_case}_B")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma list of <low|mid|high>:<light|normal|heavy>; exports one matching test episode per variant "
        "to <output-dir>/trajectory_ghost_<height>_<weight>.json",
    )
    args = parser.parse_args()
    if bool(args.checkpoint_b) != bool(args.config_b):
        parser.error("--checkpoint-b and --config-b must be provided together")

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

    evaluator_b = None
    if args.checkpoint_b and args.config_b:
        chronicle.log_info(f"Loading Model B checkpoint: {args.checkpoint_b}")
        cfg_b = ExperimentConfig.load(args.config_b)
        ckpt_b = Path(args.checkpoint_b)
        if ckpt_b.is_dir():
            ckpt_b = ckpt_b / "model.pt"
        model_b = SPWM.from_config(cfg_b).to(device)
        state_dict_b = torch.load(ckpt_b, map_location=device, weights_only=False)
        if isinstance(state_dict_b, dict) and "model_state" in state_dict_b:
            state_dict_b = state_dict_b["model_state"]
        model_b.load_state_dict(state_dict_b, strict=True)
        model_b.eval()
        evaluator_b = RolloutEvaluator(
            model=model_b,
            device=device,
            horizons=[args.horizon],
            joint_count=cfg_b.environment.joint_count,
            temporal_burn_in=cfg_b.training.temporal_burn_in,
            fixed_predictor=model_b.m1,
        )

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
        model_version_b=args.model_version_b if evaluator_b is not None else None,
        dt=cfg.environment.dt,
        horizon=args.horizon,
        joint_names=joint_names,
    )

    target_dir = Path(args.output_dir) if args.output_dir is not None else (Path(args.unity_project) / "Assets" / "StreamingAssets")
    target_dir.mkdir(parents=True, exist_ok=True)

    is_multi_export = (args.episodes > 1) or args.compare_injection or (args.episode_index is not None)

    test_len = len(dataloaders["test"].dataset)

    if args.variants is not None:
        test_ids = list(dataloaders["test"].dataset.trajectory_indices)
        meta = {tid: episode_meta(tid) for tid in test_ids}
        for variant in args.variants.split(","):
            height, weight = variant.strip().split(":")
            matches = [
                pos for pos, tid in enumerate(test_ids)
                if meta[tid][0] == height and weight_class(meta[tid][1]) == weight
            ]
            if not matches:
                raise ValueError(f"No test episode with target height '{height}' and weight '{weight}'")
            pos = matches[0]
            batch, b_idx, s_idx = get_sample_from_loader(dataloaders["test"], pos)
            chronicle.log_info(
                f"Variant {height}_{weight}: test index {pos}, trajectory_id={test_ids[pos]}, "
                f"mass={meta[test_ids[pos]][1]:.2f} kg"
            )
            ghost = exporter.capture_batch(
                batch,
                sample_index=s_idx,
                extra_metadata={"batch_index": b_idx, "episode_index": 0, "test_dataset_index": pos,
                                "target_height": height, "weight_class": weight},
                with_active_perception=not args.no_injection,
                evaluator_b=evaluator_b,
            )
            log_ghost_summary(ghost, exporter.save(ghost, path=target_dir / f"trajectory_ghost_{height}_{weight}.json"))
    elif not is_multi_export:
        # Standard legacy single-trajectory capture
        ghost = exporter.capture(
            dataloaders["test"],
            batch_index=args.batch_index,
            sample_index=args.sample_index,
            with_active_perception=not args.no_injection,
            evaluator_b=evaluator_b,
        )
        out_path = exporter.save(ghost, path=args.output)
        log_ghost_summary(ghost, out_path)
    else:
        if args.episode_index is not None:
            episodes_to_export = [args.episode_index]
        elif args.contiguous:
            episodes_to_export = list(range(min(args.episodes, test_len)))
        else:
            # Spread episodes evenly across the full test partition for maximum trajectory and payload diversity
            if args.episodes > 1:
                step = (test_len - 1) / float(args.episodes - 1)
                episodes_to_export = [int(round(i * step)) for i in range(args.episodes)]
            else:
                episodes_to_export = [0]

        chronicle.log_info(
            f"Exporting {len(episodes_to_export)} test episode(s) (indices: {episodes_to_export}, "
            f"compare_injection={args.compare_injection}, horizon={args.horizon}) to {target_dir}..."
        )

        for i, ep_idx in enumerate(episodes_to_export):
            batch, b_idx, s_idx = get_sample_from_loader(dataloaders["test"], ep_idx)
            extra_meta = {"batch_index": b_idx, "episode_index": i, "test_dataset_index": ep_idx}

            if args.compare_injection:
                # 1. With active perception sensory injection
                ghost_with = exporter.capture_batch(
                    batch,
                    sample_index=s_idx,
                    extra_metadata=extra_meta,
                    with_active_perception=True,
                    evaluator_b=evaluator_b,
                )
                path_with = target_dir / f"trajectory_ghost_ep{i}_with_injection.json"
                saved_with = exporter.save(ghost_with, path=path_with)
                log_ghost_summary(ghost_with, saved_with)

                # 2. Without active perception sensory injection
                ghost_no = exporter.capture_batch(
                    batch,
                    sample_index=s_idx,
                    extra_metadata=extra_meta,
                    with_active_perception=False,
                    evaluator_b=evaluator_b,
                )
                path_no = target_dir / f"trajectory_ghost_ep{i}_no_injection.json"
                saved_no = exporter.save(ghost_no, path=path_no)
                log_ghost_summary(ghost_no, saved_no)

                if i == 0:
                    # Update canonical trajectory_ghost.json for default Unity playback
                    canonical_path = exporter.output_path
                    exporter.save(ghost_with, path=canonical_path)
                    chronicle.log_info(f"Updated default playback trajectory: {canonical_path}")
            else:
                active_perception = not args.no_injection
                ghost = exporter.capture_batch(
                    batch,
                    sample_index=s_idx,
                    extra_metadata=extra_meta,
                    with_active_perception=active_perception,
                    evaluator_b=evaluator_b,
                )
                tag = "with_injection" if active_perception else "no_injection"
                if len(episodes_to_export) == 1 and args.output is not None:
                    out_path = exporter.save(ghost, path=args.output)
                else:
                    out_path = exporter.save(ghost, path=target_dir / f"trajectory_ghost_ep{i}_{tag}.json")
                log_ghost_summary(ghost, out_path)

                if i == 0:
                    canonical_path = exporter.output_path
                    exporter.save(ghost, path=canonical_path)
                    chronicle.log_info(f"Updated default playback trajectory: {canonical_path}")


if __name__ == "__main__":
    main()
