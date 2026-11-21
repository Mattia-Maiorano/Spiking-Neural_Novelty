"""
Comparative Baseline Training & Multi-Horizon Evaluation Runner (DOSSIER Section 6.1).

Trains and benchmarks non-spiking deep recurrent world models against canonical SPWM:
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
import sys
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmark_mppi_control import (
    _BaselineModelWrapper,
    _DeepGRUDecoderAdapter,
    _DeepGRUPredictorAdapter,
    _RSSMDecoderAdapter,
    _RSSMPredictorAdapter,
)
from spwm.contracts.config import ExperimentConfig, MPPIConfig
from spwm.contracts.states import ClosedLoopTrajectory
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders_from_config
from spwm.evaluation.mppi import LatentMPPIController
from spwm.models.baselines import (
    ContinuousConvLSTM,
    ContinuousRSSM,
    DeepGRUPredictor,
)
from spwm.models.kinematics import DifferentiableForwardKinematics, classify_elevation
from spwm.models.rollout import compute_rollout_schedule
from spwm.training.losses import _decoded_kinematic_temporal_loss
from spwm.training.metrics import position_error, velocity_error
from spwm.training.noise import (
    compute_dagger_ratio,
    compute_mppi_noise_alpha,
    sample_ar1_noise,
)
from spwm.training.replay_buffer import ClosedLoopBuffer
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


def compute_equalized_kinematic_loss(
    fk: DifferentiableForwardKinematics,
    pred_pos: torch.Tensor,
    pred_vel: torch.Tensor,
    pos_gt: torch.Tensor,
    vel_gt: torch.Tensor,
    dt: float = 0.05,
    cartesian_weight: float = 40.0,
    ee_rot_weight: float = 0.5,
    lambda_kinematic_smooth: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equalized kinematic + SE(3) EE + smoothness loss matching SPWM supervision weights."""
    loss_pos = F.mse_loss(pred_pos, pos_gt, reduction="none").sum(dim=-1).mean()
    loss_vel = F.mse_loss(pred_vel, vel_gt, reduction="none").sum(dim=-1).mean()
    t_pred = fk.forward_homogeneous(pred_pos)
    t_gt = fk.forward_homogeneous(pos_gt)
    loss_ee_pos = F.mse_loss(t_pred[..., :3, 3], t_gt[..., :3, 3], reduction="none").sum(dim=-1).mean()
    loss_ee_rot = F.mse_loss(t_pred[..., :3, :3], t_gt[..., :3, :3], reduction="none").sum(dim=(-2, -1)).mean()
    loss_cart = float(cartesian_weight) * (loss_ee_pos + float(ee_rot_weight) * loss_ee_rot)
    loss_smooth = float(lambda_kinematic_smooth) * _decoded_kinematic_temporal_loss(
        pos_pred=pred_pos,
        vel_pred=pred_vel,
        pos_true=pos_gt,
        dt=dt,
        reduce_joint_sum=True,
    )
    total = loss_pos + loss_vel + loss_cart + loss_smooth
    raw_pos_mse = torch.mean((pred_pos - pos_gt) ** 2)
    raw_vel_mse = torch.mean((pred_vel - vel_gt) ** 2)
    return total, raw_pos_mse, raw_vel_mse


def _build_baseline_adapters(
    model: nn.Module,
) -> Tuple[str, nn.Module, nn.Module]:
    """Constructs predictor and decoder adapters for ContinuousRSSM or DeepGRUPredictor."""
    if isinstance(model, ContinuousRSSM):
        return "rssm", _RSSMPredictorAdapter(model), _RSSMDecoderAdapter(model)
    if isinstance(model, DeepGRUPredictor):
        return "deep_gru", _DeepGRUPredictorAdapter(model), _DeepGRUDecoderAdapter(model)
    raise ValueError(
        f"DAgger/AR(1) closed-loop equalization requires ContinuousRSSM or DeepGRUPredictor, got {type(model).__name__}"
    )


def build_stratified_val_batches(
    val_dataset: Any,
    fk: DifferentiableForwardKinematics,
    device: torch.device,
    num_trajectories: int = 6,
) -> List[Dict[str, Any]]:
    """Extracts a balanced Low/Mid/High validation subset for baseline DAgger collection."""
    target_planes = ("Low", "Mid", "High")
    target_per_plane = max(1, num_trajectories // len(target_planes))
    by_plane: Dict[str, List[Dict[str, Any]]] = {p: [] for p in target_planes}
    other_items: List[Dict[str, Any]] = []

    for idx in range(len(val_dataset)):
        item = val_dataset[idx]
        q = item["joint_positions"]
        pf = item["phase_flags"]
        p1 = (pf == 1).nonzero()
        tc = int(p1[0].item()) if len(p1) > 0 else 50
        with torch.no_grad():
            t_mat = fk.forward_homogeneous(q[tc : tc + 1].to(device))[0]
            plane = classify_elevation(float(t_mat[1, 3].item()))

        batch_dict = {
            "events": item["events"].unsqueeze(0),
            "joint_positions": item["joint_positions"].unsqueeze(0),
            "phase_flags": item["phase_flags"].unsqueeze(0),
            "actions": item["actions"].unsqueeze(0),
            "trajectory_id": int(item["trajectory_id"]),
            "elevation_plane": plane,
        }
        if plane in by_plane and len(by_plane[plane]) < target_per_plane:
            by_plane[plane].append(batch_dict)
        else:
            other_items.append(batch_dict)

        if all(len(by_plane[p]) >= target_per_plane for p in target_planes):
            break

    selected: List[Dict[str, Any]] = []
    for k in range(target_per_plane):
        for p in target_planes:
            if k < len(by_plane[p]):
                selected.append(by_plane[p][k])
    if len(selected) < num_trajectories:
        for item in other_items:
            if len(selected) >= num_trajectories:
                break
            selected.append(item)
    return selected


@torch.no_grad()
def collect_baseline_dagger_trajectories(
    model: nn.Module,
    fixed_batches: List[Dict[str, Any]],
    fk: DifferentiableForwardKinematics,
    device: torch.device,
    mppi_cfg: MPPIConfig,
    base_seed: int = 42000,
    approach_window: int = 20,
) -> List[ClosedLoopTrajectory]:
    """Collects closed-loop MPPI validation trajectories into ClosedLoopTrajectory objects."""
    was_training = model.training
    model.eval()
    model_type, predictor, decoder = _build_baseline_adapters(model)
    wrapper = _BaselineModelWrapper(model, model_type, predictor.to(device), decoder.to(device))
    controller = LatentMPPIController(
        predictor=predictor.to(device),
        physical_decoder=decoder.to(device),
        config=mppi_cfg,
        forward_kinematics=fk,
        device=device,
    )

    trajectories: List[ClosedLoopTrajectory] = []
    for batch in fixed_batches:
        events = batch["events"].to(device)
        q_gt = batch["joint_positions"][0].to(device)
        phase_flags = batch["phase_flags"][0].to(device)
        actions = batch["actions"].to(device)
        traj_id = int(batch["trajectory_id"])

        p1_indices = (phase_flags == 1).nonzero()
        t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
        t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
        t_start = max(5, t_c - approach_window)
        max_t = min(t_grasp + 1, events.shape[1] - 1)

        z_gt_seq = wrapper.ground_sequence(events, max_t=max_t, actions=actions)
        z_start = z_gt_seq[:, t_start]

        q_reachable_grasp = decoder(z_gt_seq[:, t_c])[..., :6]
        target_grasp_mat = fk.forward_homogeneous(q_reachable_grasp)[0].detach()
        q_reachable_final = decoder(z_gt_seq[:, t_grasp])[..., :6]
        target_final_mat = fk.forward_homogeneous(q_reachable_final)[0].detach()

        dec_start = decoder(z_start)
        initial_bias = dec_start[0, :6] / 3.14
        controller.reset(initial_action_bias=initial_bias, seed=base_seed + traj_id)
        prev_u = initial_bias.clone()
        curr_z = z_start.clone()

        traj_z: List[torch.Tensor] = []
        traj_u: List[torch.Tensor] = []
        traj_targets: List[torch.Tensor] = []

        for t in range(t_start, t_grasp + 1):
            target_pose_plan = torch.stack(
                [target_grasp_mat if (t + tau) <= t_c else target_final_mat for tau in range(controller.horizon)],
                dim=0,
            )
            res = controller.plan_step(curr_z, target_pose_plan, prev_action=prev_u)
            u_exec = res.action
            prev_u = u_exec
            controller.prev_action = u_exec.clone()

            z_pred = predictor(curr_z, u_exec.unsqueeze(0)).predicted_latent
            z_gt_next = z_gt_seq[:, min(t + 1, max_t)]
            if int(phase_flags[t].item()) == 1:
                z_next = 0.40 * z_pred + 0.60 * z_gt_next
            else:
                z_next = z_pred

            traj_z.append(curr_z[0].detach().cpu())
            traj_u.append(u_exec.detach().cpu())
            traj_targets.append(z_gt_next[0].detach().cpu())
            curr_z = z_next

        trajectories.append(
            ClosedLoopTrajectory(
                z_init=traj_z[0],
                actions=torch.stack(traj_u),
                targets=torch.stack(traj_targets),
                length=len(traj_u),
                trajectory_id=traj_id,
                is_expert=False,
                elevation_plane=str(batch["elevation_plane"]),
            )
        )

    if was_training:
        model.train()
    return trajectories


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
    enable_dagger_ar1: bool = False,
    mppi_cfg: Optional[MPPIConfig] = None,
    ar1_warmup_epochs: int = 10,
    ar1_ramp_epochs: int = 20,
    ar1_target_alpha: float = 0.25,
    dagger_frequency: int = 5,
    dagger_start_epoch: int = 10,
    dagger_horizon: int = 10,
    seed: int = 42,
    prior_epochs: int = 0,
    prior_history: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    regime_desc = "Equalized SE(3) + Multi-Step Rollout + AR(1) + DAgger" if enable_dagger_ar1 else "Equalized SE(3) + Multi-Step Rollout"
    chronicle.log_info(f"--- Training {model_name} ({regime_desc}) on {device} ({epochs} epochs max, prior={prior_epochs}) ---")
    model = model.to(device)
    fk = DifferentiableForwardKinematics(joint_count=joint_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = save_dir / "model.pt"
    history: List[Dict[str, float]] = list(prior_history) if prior_history is not None else []
    train_start_time = time.perf_counter()

    replay_buffer: Optional[ClosedLoopBuffer] = None
    dagger_val_batches: List[Dict[str, Any]] = []
    dagger_generations = 0
    pred_adapter: Optional[nn.Module] = None
    dec_adapter: Optional[nn.Module] = None
    active_mppi_cfg = mppi_cfg if mppi_cfg is not None else MPPIConfig()

    if enable_dagger_ar1:
        _, pred_adapter, dec_adapter = _build_baseline_adapters(model)
        pred_adapter = pred_adapter.to(device)
        dec_adapter = dec_adapter.to(device)
        replay_buffer = ClosedLoopBuffer(capacity=50_000, max_eval_generations=3)
        dagger_val_batches = build_stratified_val_batches(val_loader.dataset, fk, device, num_trajectories=6)
        chronicle.log_info(
            f"[{model_name}] DAgger + AR(1) enabled: {len(dagger_val_batches)} stratified val trajectories, "
            f"alpha->{ar1_target_alpha}, dagger_freq={dagger_frequency}, start_ep={dagger_start_epoch}"
        )

    for ep in range(1, epochs + 1):
        alpha = (
            compute_mppi_noise_alpha(
                ep,
                warmup_epochs=ar1_warmup_epochs,
                ramp_epochs=ar1_ramp_epochs,
                target_alpha=ar1_target_alpha,
            )
            if enable_dagger_ar1
            else 0.0
        )
        dagger_ratio = 0.0
        if enable_dagger_ar1 and replay_buffer is not None and len(replay_buffer) > 0:
            dagger_ratio = compute_dagger_ratio(
                ep,
                warmup_epochs=max(0, dagger_start_epoch - 1),
                ramp_end_epoch=max(dagger_start_epoch + 1, int(round(epochs * 0.75))),
                min_ratio=0.10,
                max_ratio=0.50,
            )

        if (
            enable_dagger_ar1
            and replay_buffer is not None
            and ep >= dagger_start_epoch
            and ((ep - dagger_start_epoch) % max(1, dagger_frequency) == 0)
        ):
            cl_trajs = collect_baseline_dagger_trajectories(
                model=model,
                fixed_batches=dagger_val_batches,
                fk=fk,
                device=device,
                mppi_cfg=active_mppi_cfg,
                base_seed=42000,
            )
            replay_buffer.push_trajectories(cl_trajs)
            replay_buffer.save(save_dir / "closed_loop_buffer.pt")
            dagger_generations += 1
            dagger_ratio = compute_dagger_ratio(
                ep,
                warmup_epochs=max(0, dagger_start_epoch - 1),
                ramp_end_epoch=max(dagger_start_epoch + 1, int(round(epochs * 0.75))),
                min_ratio=0.10,
                max_ratio=0.50,
            )
            chronicle.log_info(
                f"[{model_name}] Ep {ep:02d}: Collected {len(cl_trajs)} DAgger trajectories "
                f"(buffer={len(replay_buffer)} transitions, gen={dagger_generations}, ratio={dagger_ratio:.2f})"
            )

        model.train()
        train_losses: List[float] = []

        t0 = time.perf_counter()
        for batch in train_loader:
            events = batch["events"].to(device)
            actions = batch["actions"].to(device)
            kin_gt = batch["flat_kinematics"].to(device)
            pos_gt = kin_gt[..., :joint_count]
            vel_gt = kin_gt[..., joint_count:2 * joint_count]

            optimizer.zero_grad()
            out = model(events, actions=actions)
            tf_loss, _, _ = compute_equalized_kinematic_loss(
                fk, out["positions"], out["velocities"], pos_gt, vel_gt
            )
            if "kl_loss" in out:
                tf_loss = tf_loss + 0.1 * out["kl_loss"]

            t_start = 4
            roll_h = min(80, events.shape[1] - t_start - 1)
            rollout_actions = actions
            if enable_dagger_ar1 and alpha > 0.0:
                rollout_actions = actions.clone()
                ar1 = sample_ar1_noise(
                    batch_size=events.shape[0],
                    horizon=roll_h,
                    action_dim=actions.shape[-1],
                    device=device,
                    beta=active_mppi_cfg.noise_beta,
                    base_sigma=float(active_mppi_cfg.noise_sigma),
                    shoulder_multiplier=active_mppi_cfg.shoulder_sigma_multiplier,
                    elbow_multiplier=active_mppi_cfg.elbow_sigma_multiplier,
                )
                rollout_actions[:, t_start : t_start + roll_h] = torch.clamp(
                    rollout_actions[:, t_start : t_start + roll_h] + alpha * ar1,
                    -1.0,
                    1.0,
                )

            roll_res = model.rollout_train(events=events, actions=rollout_actions, horizon=roll_h, t_start=t_start)
            roll_loss, _, _ = compute_equalized_kinematic_loss(
                fk,
                roll_res.positions,
                roll_res.velocities,
                pos_gt[:, t_start + 1 : t_start + 1 + roll_h],
                vel_gt[:, t_start + 1 : t_start + 1 + roll_h],
            )

            total_loss = 0.5 * tf_loss + 1.0 * roll_loss

            if (
                enable_dagger_ar1
                and dagger_ratio > 0.0
                and replay_buffer is not None
                and len(replay_buffer) > 0
                and pred_adapter is not None
                and dec_adapter is not None
            ):
                cl_batch = replay_buffer.sample_trajectories(
                    batch_size=min(16, events.shape[0]),
                    horizon=dagger_horizon,
                    device=device,
                )
                curr_cl_z = cl_batch["z_init"]
                cl_acts = cl_batch["actions"]
                cl_preds: List[torch.Tensor] = []
                for h_idx in range(cl_acts.shape[1]):
                    curr_cl_z = pred_adapter(curr_cl_z, cl_acts[:, h_idx]).predicted_latent
                    cl_preds.append(curr_cl_z)
                cl_pred = torch.stack(cl_preds, dim=1)
                cl_target = cl_batch["targets"]
                cl_lat_loss = F.mse_loss(cl_pred, cl_target)
                cl_kin = dec_adapter(cl_pred)
                with torch.no_grad():
                    tgt_kin = dec_adapter(cl_target)
                cl_kin_loss, _, _ = compute_equalized_kinematic_loss(
                    fk,
                    cl_kin[..., :joint_count],
                    cl_kin[..., joint_count : 2 * joint_count],
                    tgt_kin[..., :joint_count],
                    tgt_kin[..., joint_count : 2 * joint_count],
                )
                total_loss = total_loss + dagger_ratio * (cl_lat_loss + 0.5 * cl_kin_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(total_loss.item()))

        scheduler.step()

        # Validation on long-horizon autonomous rollout
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

                t_start = 4
                val_h = min(100, events.shape[1] - t_start - 1)
                roll_res = model.rollout(events=events, actions=actions, horizon=val_h, t_start=t_start)
                v_loss, loss_pos, loss_vel = compute_equalized_kinematic_loss(
                    fk,
                    roll_res.positions,
                    roll_res.velocities,
                    pos_gt[:, t_start + 1 : t_start + 1 + val_h],
                    vel_gt[:, t_start + 1 : t_start + 1 + val_h],
                )

                val_losses.append(float(v_loss.item()))
                val_pos_losses.append(float(loss_pos.item()))
                val_vel_losses.append(float(loss_vel.item()))

        mean_train = float(np.mean(train_losses))
        mean_val = float(np.mean(val_losses))
        dt_ep = time.perf_counter() - t0

        record = {
            "epoch": prior_epochs + ep,
            "stage_epoch": ep,
            "train_loss": mean_train,
            "val_loss": mean_val,
            "val_pos_mse": float(np.mean(val_pos_losses)),
            "val_vel_mse": float(np.mean(val_vel_losses)),
            "ar1_alpha": float(alpha),
            "dagger_ratio": float(dagger_ratio),
            "duration": dt_ep,
        }
        history.append(record)

        is_best = mean_val < best_val_loss
        if is_best:
            best_val_loss = mean_val
            best_epoch = prior_epochs + ep
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "history": history}, best_model_path)
            star = " *"
        else:
            epochs_no_improve += 1
            star = ""

        chronicle.log_info(
            f"[{model_name}] Ep {ep:02d}/{epochs:02d} (tot={prior_epochs + ep}) | Train: {mean_train:.4f} | "
            f"ValRoll: {mean_val:.4f} (Pos: {record['val_pos_mse']:.5f}, Vel: {record['val_vel_mse']:.5f}) | "
            f"a={alpha:.2f} d={dagger_ratio:.2f} | {dt_ep:.1f}s{star}"
        )

        if epochs_no_improve >= patience:
            chronicle.log_info(f"[{model_name}] Early stopping triggered after {patience} epochs without improvement.")
            break

    wall_clock_sec = time.perf_counter() - train_start_time
    meta = {
        "model_name": model_name,
        "seed": int(seed),
        "device": str(device),
        "enable_dagger_ar1": bool(enable_dagger_ar1),
        "base_epochs": int(prior_epochs),
        "ar1_dagger_epochs": int(len(history) - prior_epochs) if enable_dagger_ar1 else 0,
        "total_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "wall_clock_sec": float(wall_clock_sec),
        "ar1_target_alpha": float(ar1_target_alpha) if enable_dagger_ar1 else 0.0,
        "dagger_generations_collected": int(dagger_generations),
    }
    (save_dir / "training_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Load best checkpoint strictly
    if best_model_path.exists():
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return {"history": history, "best_val_loss": best_val_loss, "training_meta": meta}


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
    parser.add_argument("--epochs", type=int, default=60, help="Max epochs per baseline")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--total-trajectories", type=int, default=4000, help="Total trajectories to partition")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, mps, cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results/baselines", help="Output directory")
    parser.add_argument("--config", type=Path, default=GLOBAL_VERSION.m1_config(), help="Path to SPWM M1 config")
    parser.add_argument("--checkpoint", type=Path, default=GLOBAL_VERSION.m1_checkpoint(), help="Path to SPWM M1 checkpoint")
    parser.add_argument("--m1-metrics", type=Path, default=GLOBAL_VERSION.m1_metrics(), help="Path to SPWM M1 metrics JSON")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "rssm", "deep_gru", "convlstm"],
        help="Subset of baseline models to train and evaluate.",
    )
    parser.add_argument(
        "--enable-dagger-ar1",
        action="store_true",
        help="Enable AR(1) MPPI-matched action noise injection and ClosedLoopBuffer DAgger replay.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional pre-trained baseline checkpoint to load strictly before AR(1)+DAgger fine-tuning.",
    )
    parser.add_argument("--ar1-warmup-epochs", type=int, default=10, help="AR(1) noise warmup epochs.")
    parser.add_argument("--ar1-ramp-epochs", type=int, default=20, help="AR(1) noise ramp epochs.")
    parser.add_argument("--ar1-target-alpha", type=float, default=0.25, help="Target AR(1) action noise scale.")
    parser.add_argument("--dagger-frequency", type=int, default=5, help="Epochs between DAgger val trajectory collection.")
    parser.add_argument("--dagger-start-epoch", type=int, default=10, help="First epoch to collect DAgger trajectories.")
    parser.add_argument("--dagger-horizon", type=int, default=10, help="DAgger BPTT unroll horizon.")
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

    cfg = ExperimentConfig.load(args.config) if args.config.is_file() else ExperimentConfig()
    eval_horizons = [1, 10, 50, 100]
    seq_len = 105

    chronicle.log_info(f"Creating DataLoaders: total={args.total_trajectories}, seq_len={seq_len}, batch={args.batch_size}")
    dataloaders = create_dataloaders_from_config(
        cfg,
        total_trajectories=args.total_trajectories,
        sequence_length=seq_len,
        batch_size=args.batch_size,
        cache_data=True,
        strict_dataset=True,
    )

    # Models to benchmark
    all_models: Dict[str, nn.Module] = {
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
    key_map = {
        "rssm": "Continuous RSSM",
        "deep_gru": "Deep GRU",
        "convlstm": "Continuous ConvLSTM",
    }
    if "all" in args.models:
        models = all_models
    else:
        selected_names = [key_map[k] for k in args.models]
        models = {k: v for k, v in all_models.items() if k in selected_names}

    if args.init_checkpoint is not None and len(models) != 1:
        raise ValueError("--init-checkpoint requires selecting exactly one model via --models.")

    benchmark_results: Dict[str, Dict[str, Any]] = {}

    # 1. Train and evaluate each baseline
    for name, model in models.items():
        model_slug = name.lower().replace(" ", "_").replace("-", "_")
        model_dir = output_dir / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)

        prior_epochs = 0
        prior_history: List[Dict[str, float]] = []
        if args.init_checkpoint is not None:
            init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
            init_state = init_ckpt["model_state"] if isinstance(init_ckpt, dict) and "model_state" in init_ckpt else init_ckpt
            model.load_state_dict(init_state, strict=True)
            if isinstance(init_ckpt, dict) and "history" in init_ckpt:
                prior_history = list(init_ckpt["history"])
                prior_epochs = len(prior_history)
            chronicle.log_info(
                f"[{name}] Strictly loaded init checkpoint {args.init_checkpoint} (prior_epochs={prior_epochs})"
            )

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
            enable_dagger_ar1=args.enable_dagger_ar1,
            mppi_cfg=cfg.mppi,
            ar1_warmup_epochs=args.ar1_warmup_epochs,
            ar1_ramp_epochs=args.ar1_ramp_epochs,
            ar1_target_alpha=args.ar1_target_alpha,
            dagger_frequency=args.dagger_frequency,
            dagger_start_epoch=args.dagger_start_epoch,
            dagger_horizon=args.dagger_horizon,
            seed=args.seed,
            prior_epochs=prior_epochs,
            prior_history=prior_history,
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
            "training_meta": train_res["training_meta"],
        }
        chronicle.log_success(
            f"[{name}] Evaluated: Pos MSE (H=100): {eval_res['position_mse_per_horizon'][100]:.5f} | "
            f"Vel MSE (H=100): {eval_res['velocity_mse_per_horizon'][100]:.5f}"
        )

    # 2. Incorporate SPWM production baseline metrics
    spwm_metrics_file = args.m1_metrics
    if spwm_metrics_file.exists():
        with open(spwm_metrics_file, "r", encoding="utf-8") as f:
            spwm_data = json.load(f)

        # Profile SPWM latency on current device
        from spwm.models.world_model import SPWM
        spwm_cfg = ExperimentConfig.load(args.config)
        spwm_model = SPWM.from_config(spwm_cfg).to(device)
        spwm_ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    # 3. Save comparative metrics JSON (preserving sibling model entries on subset runs)
    metrics_path = output_dir / "comparative_metrics.json"
    if "all" not in args.models and metrics_path.is_file():
        existing_benchmarks = json.loads(metrics_path.read_text(encoding="utf-8"))
        existing_benchmarks.update(final_benchmarks)
        final_benchmarks = existing_benchmarks
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
