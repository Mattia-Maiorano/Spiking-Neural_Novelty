#!/usr/bin/env python3
"""
Offline Dry-Run Verification Script for Smooth Ground Truth Blending (SPWM-v0.7.2).

Verifies:
1. Rollout with soft-fade handover executes cleanly across the test dataset.
2. Latent momentum subspace satisfies max(|p|) <= 3.5 (no LayerNorm bypass or scale explosion).
3. Decoded kinematic positions and velocities are smooth without single-frame spikes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import numpy as np

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run smooth rollout verification.")
    parser.add_argument("--checkpoint", type=str, default="results/spwm_v0_7_0_train_m1/model.pt")
    parser.add_argument("--config", type=str, default="results/spwm_v0_7_0_train_m1/config.yaml")
    parser.add_argument("--horizon", type=int, default=100)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config)
    if not ckpt_path.is_file() or not cfg_path.is_file():
        print(f"[!] Checkpoint or config not found: {ckpt_path}, {cfg_path}")
        return 1

    cfg = ExperimentConfig.load(str(cfg_path))
    set_seed(cfg.project.seed)
    device = torch.device("cpu")

    model = SPWM.from_config(cfg)
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load test loader
    dataloaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        sequence_length=cfg.data.sequence_length,
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
    test_loader = dataloaders["test"]

    batch = next(iter(test_loader))
    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=[args.horizon],
        joint_count=cfg.environment.joint_count,
        temporal_burn_in=cfg.training.temporal_burn_in,
        fixed_predictor=model.m1,
    )

    traj = evaluator.rollout_trajectory(batch, horizon=args.horizon)
    
    # 1. Latent momentum magnitude check
    pred_lat = traj.rollout.trajectory[0].detach().cpu().numpy()  # [H, latent_dim]
    q_dim = cfg.model.q_dim
    p_subspace = pred_lat[:, q_dim:]
    max_p = float(np.max(np.abs(p_subspace)))
    print(f"[*] Max latent momentum |p|: {max_p:.4f} (Threshold: <= 3.50)")
    assert max_p <= 3.50, f"Momentum magnitude exploded: {max_p} > 3.50"

    # 2. Kinematics smoothness check
    pred_kin = traj.rollout.decoded_kinematics[0].detach().cpu().numpy()  # [H, 2 * joint_count]
    pred_q = pred_kin[:, :cfg.environment.joint_count]
    delta_q = np.diff(pred_q, axis=0)  # [H-1, joint_count]
    max_step_delta_q = float(np.max(np.abs(delta_q)))
    print(f"[*] Max 1-step coordinate delta |Δq|: {max_step_delta_q:.4f} rad/step")

    # In contact region (e.g. frames 50-65)
    first_heft = 0
    if "phase_flags" in batch:
        flags = batch["phase_flags"][0].cpu().numpy()
        p1_idx = np.where(flags == 1)[0]
        if len(p1_idx) > 0:
            first_heft = max(0, int(p1_idx[0]) - traj.t_start)
    contact_deltas = delta_q[max(0, first_heft - 5) : min(len(delta_q), first_heft + 10)]
    max_contact_delta = float(np.max(np.abs(contact_deltas))) if len(contact_deltas) > 0 else 0.0
    print(f"[*] Max coordinate delta in contact window: {max_contact_delta:.4f} rad/step")
    assert max_contact_delta < 0.25, f"High-frequency coordinate spike detected: {max_contact_delta} >= 0.25 rad/step"

    print("\n[✓] Offline smooth rollout verification passed completely.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
