"""
SSOT Table 1 & Figure 2 Open-Loop Rollout Evaluator (N=400 Test Trajectories).

Evaluates all frozen checkpoints (SPWM, ANN-Rate, Continuous RSSM, Deep GRU, Continuous ConvLSTM)
on the exact N=400 unseen test split [3600, 3999] with t_start=4, horizon=100.
Logs mean and std across trajectories into comparative_metrics_v0_12_4.json
and saves per-step L2 rollout error curves for Figure 2.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import EventWorldDataset
from spwm.models.ann_ablation import ANNWorldModel
from spwm.models.baselines import ContinuousConvLSTM, ContinuousRSSM, DeepGRUPredictor
from spwm.models.world_model import SPWM


def _load_state(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    return ckpt


@torch.no_grad()
def evaluate_model_rollout(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizons: List[int],
    t_start: int = 4,
    max_h: int = 100,
) -> Dict[str, Any]:
    model.eval()
    pos_mse_all: List[torch.Tensor] = []
    vel_mse_all: List[torch.Tensor] = []
    pos_l2_all: List[torch.Tensor] = []

    for batch in loader:
        events = batch["events"].to(device)
        actions = batch["actions"].to(device)
        q_true = batch["joint_positions"][:, t_start + 1 : t_start + 1 + max_h, :6].to(device)
        p_true = batch["joint_velocities"][:, t_start + 1 : t_start + 1 + max_h, :6].to(device)

        res = model.rollout(events, actions=actions, horizon=max_h, t_start=t_start)
        q_pred = res.positions[..., :6]
        p_pred = res.velocities[..., :6]

        pos_mse_step = ((q_pred - q_true) ** 2).mean(dim=-1)  # [B, H]
        vel_mse_step = ((p_pred - p_true) ** 2).mean(dim=-1)  # [B, H]
        pos_l2_step = torch.norm(q_pred - q_true, dim=-1)     # [B, H]

        pos_mse_all.append(pos_mse_step.cpu())
        vel_mse_all.append(vel_mse_step.cpu())
        pos_l2_all.append(pos_l2_step.cpu())

    pos_mse_mat = torch.cat(pos_mse_all, dim=0).numpy()  # [400, 100]
    vel_mse_mat = torch.cat(vel_mse_all, dim=0).numpy()  # [400, 100]
    pos_l2_mat = torch.cat(pos_l2_all, dim=0).numpy()    # [400, 100]

    pos_mse_mean = {str(h): float(np.mean(pos_mse_mat[:, h - 1])) for h in horizons}
    pos_mse_std = {str(h): float(np.std(pos_mse_mat[:, h - 1])) for h in horizons}
    vel_mse_mean = {str(h): float(np.mean(vel_mse_mat[:, h - 1])) for h in horizons}
    vel_mse_std = {str(h): float(np.std(vel_mse_mat[:, h - 1])) for h in horizons}
    pos_l2_mean = {str(h): float(np.mean(pos_l2_mat[:, h - 1])) for h in horizons}
    pos_l2_std = {str(h): float(np.std(pos_l2_mat[:, h - 1])) for h in horizons}

    return {
        "n_test": int(pos_mse_mat.shape[0]),
        "position_mse_per_horizon": pos_mse_mean,
        "position_mse_std_per_horizon": pos_mse_std,
        "velocity_mse_per_horizon": vel_mse_mean,
        "velocity_mse_std_per_horizon": vel_mse_std,
        "position_error_per_horizon": pos_l2_mean,
        "position_error_std_per_horizon": pos_l2_std,
        "overall_l2_mean": float(np.mean(pos_l2_mat)),
        "curve_l2_mean": [float(x) for x in np.mean(pos_l2_mat, axis=0)],
        "curve_l2_q25": [float(x) for x in np.percentile(pos_l2_mat, 25, axis=0)],
        "curve_l2_q75": [float(x) for x in np.percentile(pos_l2_mat, 75, axis=0)],
    }


def _aggregate_multiseed_rollout(
    seed_results: Dict[int, Dict[str, Any]],
    horizons: List[int],
) -> Dict[str, Any]:
    """Aggregate multi-seed open-loop rollout results while keeping canonical top-level keys."""
    seeds = sorted(seed_results.keys())
    first = seed_results[seeds[0]]
    agg: Dict[str, Any] = {
        "n_test": first["n_test"],
        "seeds": seeds,
        "parameter_count": first["parameter_count"],
        "per_seed": {str(s): seed_results[s] for s in seeds},
        "position_mse_per_horizon": {},
        "position_mse_std_per_horizon": {},
        "position_mse_seed_std_per_horizon": {},
        "velocity_mse_per_horizon": {},
        "velocity_mse_std_per_horizon": {},
        "velocity_mse_seed_std_per_horizon": {},
        "position_error_per_horizon": {},
        "position_error_std_per_horizon": {},
        "position_error_seed_std_per_horizon": {},
    }
    if "active_inference_params" in first:
        agg["active_inference_params"] = first["active_inference_params"]

    ddof = 1 if len(seeds) > 1 else 0
    for h in horizons:
        hs = str(h)
        p_vals = [seed_results[s]["position_mse_per_horizon"][hs] for s in seeds]
        v_vals = [seed_results[s]["velocity_mse_per_horizon"][hs] for s in seeds]
        e_vals = [seed_results[s]["position_error_per_horizon"][hs] for s in seeds]
        agg["position_mse_per_horizon"][hs] = float(np.mean(p_vals))
        agg["position_mse_std_per_horizon"][hs] = float(np.mean([seed_results[s]["position_mse_std_per_horizon"][hs] for s in seeds]))
        agg["position_mse_seed_std_per_horizon"][hs] = float(np.std(p_vals, ddof=ddof))
        agg["velocity_mse_per_horizon"][hs] = float(np.mean(v_vals))
        agg["velocity_mse_std_per_horizon"][hs] = float(np.mean([seed_results[s]["velocity_mse_std_per_horizon"][hs] for s in seeds]))
        agg["velocity_mse_seed_std_per_horizon"][hs] = float(np.std(v_vals, ddof=ddof))
        agg["position_error_per_horizon"][hs] = float(np.mean(e_vals))
        agg["position_error_std_per_horizon"][hs] = float(np.mean([seed_results[s]["position_error_std_per_horizon"][hs] for s in seeds]))
        agg["position_error_seed_std_per_horizon"][hs] = float(np.std(e_vals, ddof=ddof))

    overalls = [seed_results[s]["overall_l2_mean"] for s in seeds]
    agg["overall_l2_mean"] = float(np.mean(overalls))
    agg["overall_l2_seed_std"] = float(np.std(overalls, ddof=ddof))
    agg["curve_l2_mean"] = first["curve_l2_mean"]
    agg["curve_l2_q25"] = first["curve_l2_q25"]
    agg["curve_l2_q75"] = first["curve_l2_q75"]
    return agg


def main() -> None:
    cfg = ExperimentConfig.load("results/spwm_v0_12_4_train_m1/config.yaml")
    device = torch.device("cpu")
    ds_test = EventWorldDataset(list(range(3600, 4000)), dataset_dir=Path("sim/ExportedTrajectories"))
    loader = DataLoader(ds_test, batch_size=32, shuffle=False)
    horizons = [1, 5, 10, 25, 50, 100]

    # 1. SPWM (Seeds 42, 101, 2024)
    spwm_ckpts = {
        42: Path("results/spwm_v0_12_4_train_m1/model.pt"),
        101: Path("results/multiseed/seed_101_m1/model.pt"),
        2024: Path("results/multiseed/seed_2024_m1/model.pt"),
    }
    spwm_per_seed: Dict[int, Dict[str, Any]] = {}
    for seed, ckpt_path in spwm_ckpts.items():
        if ckpt_path.is_file():
            spwm = SPWM.from_config(cfg).to(device)
            spwm.load_state_dict(_load_state(ckpt_path, device), strict=True)
            res = evaluate_model_rollout(spwm, loader, device, horizons)
            res["parameter_count"] = sum(p.numel() for p in spwm.parameters())
            res["active_inference_params"] = res["parameter_count"] - sum(
                p.numel() for n, p in spwm.named_parameters() if n.startswith("critic.") or n.endswith("W_vel.weight")
            )
            spwm_per_seed[seed] = res
    spwm_res = _aggregate_multiseed_rollout(spwm_per_seed, horizons)

    # 2. ANN-Rate (Seeds 42, 101, 2024)
    ann_ckpts = {
        42: Path("results/ann_ablation/seed_42_m1/model.pt"),
        101: Path("results/ann_ablation/seed_101_m1/model.pt"),
        2024: Path("results/ann_ablation/seed_2024_m1/model.pt"),
    }
    if not ann_ckpts[42].is_file():
        ann_ckpts[42] = Path("results/ann_ablation/ann_rate_m1/model.pt")
    ann_per_seed: Dict[int, Dict[str, Any]] = {}
    for seed, ckpt_path in ann_ckpts.items():
        if ckpt_path.is_file():
            ann = ANNWorldModel.from_config(cfg).to(device)
            ann.load_state_dict(_load_state(ckpt_path, device), strict=True)
            res = evaluate_model_rollout(ann, loader, device, horizons)
            res["parameter_count"] = sum(p.numel() for p in ann.parameters())
            res["active_inference_params"] = res["parameter_count"] - sum(
                p.numel() for n, p in ann.named_parameters() if n.startswith("critic.") or n.endswith("W_vel.weight")
            )
            ann_per_seed[seed] = res
    ann_res = _aggregate_multiseed_rollout(ann_per_seed, horizons)

    # 3. Continuous RSSM (Seeds 42, 101, 2024)
    rssm_ckpts = {
        42: Path("results/baselines/continuous_rssm/model.pt"),
        101: Path("results/multiseed_rssm/seed_101/continuous_rssm/model.pt"),
        2024: Path("results/multiseed_rssm/seed_2024/continuous_rssm/model.pt"),
    }
    rssm_per_seed: Dict[int, Dict[str, Any]] = {}
    for seed, ckpt_path in rssm_ckpts.items():
        if ckpt_path.is_file():
            rssm = ContinuousRSSM(
                in_channels=cfg.model.in_channels,
                height=cfg.environment.height,
                width=cfg.environment.width,
                encoder_dim=cfg.model.encoder_dim,
                deterministic_dim=256,
                stochastic_dim=32,
                action_dim=cfg.environment.action_dim,
                joint_count=cfg.environment.joint_count,
            ).to(device)
            rssm.load_state_dict(_load_state(ckpt_path, device), strict=True)
            res = evaluate_model_rollout(rssm, loader, device, horizons)
            res["parameter_count"] = sum(p.numel() for p in rssm.parameters())
            rssm_per_seed[seed] = res
    rssm_res = _aggregate_multiseed_rollout(rssm_per_seed, horizons)

    # 4. Deep GRU (Seeds 42, 101, 2024)
    gru_ckpts = {
        42: Path("results/baselines/deep_gru/model.pt"),
        101: Path("results/multiseed_rssm/seed_101/deep_gru/model.pt"),
        2024: Path("results/multiseed_rssm/seed_2024/deep_gru/model.pt"),
    }
    gru_per_seed: Dict[int, Dict[str, Any]] = {}
    for seed, ckpt_path in gru_ckpts.items():
        if ckpt_path.is_file():
            gru = DeepGRUPredictor(
                in_channels=cfg.model.in_channels,
                height=cfg.environment.height,
                width=cfg.environment.width,
                encoder_dim=cfg.model.encoder_dim,
                hidden_dim=164,
                action_dim=cfg.environment.action_dim,
                joint_count=cfg.environment.joint_count,
            ).to(device)
            gru.load_state_dict(_load_state(ckpt_path, device), strict=True)
            res = evaluate_model_rollout(gru, loader, device, horizons)
            res["parameter_count"] = sum(p.numel() for p in gru.parameters())
            gru_per_seed[seed] = res
    gru_res = _aggregate_multiseed_rollout(gru_per_seed, horizons)

    # 5. Continuous ConvLSTM
    conv = ContinuousConvLSTM(
        in_channels=cfg.model.in_channels,
        height=cfg.environment.height,
        width=cfg.environment.width,
        hidden_channels=56,
        action_dim=cfg.environment.action_dim,
        joint_count=cfg.environment.joint_count,
    ).to(device)
    conv.load_state_dict(_load_state(Path("results/baselines/continuous_convlstm/model.pt"), device), strict=True)
    conv_res = evaluate_model_rollout(conv, loader, device, horizons)
    conv_res["parameter_count"] = sum(p.numel() for p in conv.parameters())

    out = {
        "SPWM-v0.12.4": spwm_res,
        "ANN-Rate": ann_res,
        "Continuous RSSM": rssm_res,
        "Deep GRU": gru_res,
        "Continuous ConvLSTM": conv_res,
    }
    out_path = Path("results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for name, d in out.items():
        seeds_str = f"seeds={d['seeds']}" if "seeds" in d else "seed=42"
        print(
            f"{name:22s} ({seeds_str}) | params={d['parameter_count']:,} | "
            f"H1={d['position_mse_per_horizon']['1']:.5f}+-{d['position_mse_std_per_horizon']['1']:.5f} | "
            f"H100={d['position_mse_per_horizon']['100']:.5f}+-{d['position_mse_std_per_horizon']['100']:.5f} | "
            f"V100={d['velocity_mse_per_horizon']['100']:.5f}+-{d['velocity_mse_std_per_horizon']['100']:.5f}"
        )


if __name__ == "__main__":
    main()
