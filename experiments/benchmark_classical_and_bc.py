"""
Non-World-Model & Classical Control Baselines (50-Episode PIL Benchmark).

Implements and evaluates two grounded non-world-model baselines on the exact same
50 canonical PIL test episodes ([3600, 3649]) used for SPWM:
  (a) Fixed-Gain Cartesian/Joint Jacobian-IK + PD Controller calibrated to the
      mean training payload mass (2.85 kg) without online inertia adaptation.
  (b) Reactive Behavior Cloning (BC) Policy (~318k params, EventEncoder + GRU
      mapping dual-camera SpatialSoftmax features + action history directly to u_t)
      trained on the exact same training split [0, 3199].

Strict Repository AST Anti-Fallback Compliance Enforced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import chronicle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.benchmark_mppi_control import load_trajectory_metadata
from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import EventWorldDataset, partition_trajectories
from spwm.data.event_camera import EventCameraSimulator, OnlineEventCameraStream
from spwm.data.sim_6dof import ArticulatedArmPhysicsStepper
from spwm.models.encoder import EventEncoder
from spwm.evaluation.mppi import resolve_effective_vel_threshold
from spwm.models.kinematics import (
    DifferentiableForwardKinematics,
    classify_elevation,
    compute_rot_error,
)
from spwm.utils.reproducibility import set_seed


class ReactiveBCPolicy(nn.Module):
    """Parameter-matched (~318k) reactive Behavior Cloning policy (EventEncoder + GRU -> u_t)."""

    def __init__(
        self,
        in_channels: int = 4,
        height: int = 32,
        width: int = 32,
        encoder_dim: int = 128,
        action_dim: int = 6,
        hidden_dim: int = 242,
    ) -> None:
        super().__init__()
        self.encoder = EventEncoder(
            in_channels=in_channels,
            input_height=height,
            input_width=width,
            out_dim=encoder_dim,
        )
        self.gru = nn.GRUCell(encoder_dim + action_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        event_frame: torch.Tensor,
        prev_action: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out = self.encoder(event_frame)
        x = torch.cat([enc_out, prev_action], dim=-1)
        h_next = self.gru(x, h)
        u = self.head(h_next)
        return u, h_next

    def forward_sequence(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        B, T = events.shape[:2]
        h = self.init_hidden(B, events.device)
        preds: List[torch.Tensor] = []
        prev_u = torch.zeros(B, actions.shape[-1], device=events.device)
        for t in range(T):
            u_t, h = self.step(events[:, t], prev_u, h)
            preds.append(u_t)
            if scheduled_sampling_prob > 0.0:
                use_pred = (torch.rand(B, 1, device=events.device) < scheduled_sampling_prob).float()
                prev_u = use_pred * u_t.detach() + (1.0 - use_pred) * actions[:, t]
            else:
                prev_u = actions[:, t]
        return torch.stack(preds, dim=1)


def _compute_se3_jacobian_step(
    fk: DifferentiableForwardKinematics,
    q_curr: torch.Tensor,
    target_pose: torch.Tensor,
    damping: float = 0.05,
    step_gain: float = 0.65,
) -> np.ndarray:
    """Computes a damped least-squares SE(3) IK step q_des for a fixed-gain PD controller."""
    q_var = q_curr.detach().clone().requires_grad_(True)
    T_curr = fk.forward_homogeneous(q_var.unsqueeze(0))[0]
    p_err = target_pose[:3, 3] - T_curr[:3, 3]
    # Orientation error vector from skew-symmetric part of R_err = R_target @ R_curr^T
    R_err = target_pose[:3, :3] @ T_curr[:3, :3].T
    rot_err = 0.5 * torch.stack(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    err_6d = torch.cat([p_err, 0.35 * rot_err], dim=0)

    jac_rows: List[torch.Tensor] = []
    for i in range(6):
        grad_i = torch.autograd.grad(err_6d[i], q_var, retain_graph=(i < 5))[0]
        jac_rows.append(-grad_i)
    J = torch.stack(jac_rows, dim=0)  # [6, 6]
    JJt = J @ J.T + (damping ** 2) * torch.eye(6, device=q_curr.device)
    dq = J.T @ torch.linalg.solve(JJt, err_6d)
    q_des = torch.clamp(q_curr + step_gain * dq, -3.14, 3.14)
    return q_des.detach().cpu().numpy()


def _summarize_episodes(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    succ = [r["grasp_success"] for r in records]
    min_e = [r["min_placement_error_cm"] for r in records]
    term_e = [r["terminal_placement_error_cm"] for r in records]
    ddq = [r["mean_joint_accel_rad_s2"] for r in records]
    dddq = [r["mean_joint_jerk_rad_s3"] for r in records]
    flips = [r["direction_flips"] for r in records]
    lat = [r["mean_latency_ms"] for r in records]
    return {
        "model_label": label,
        "episodes_evaluated": len(records),
        "grasp_success_rate_pct": float(np.mean(succ) * 100.0),
        "terminal_placement_accuracy_cm": float(np.mean(min_e)),
        "placement_accuracy_std_cm": float(np.std(min_e)),
        "terminal_error_cm": float(np.mean(term_e)),
        "terminal_error_std_cm": float(np.std(term_e)),
        "mean_joint_accel_rad_s2": float(np.mean(ddq)),
        "std_joint_accel_rad_s2": float(np.std(ddq)),
        "mean_joint_jerk_rad_s3": float(np.mean(dddq)),
        "std_joint_jerk_rad_s3": float(np.std(dddq)),
        "mean_direction_flips": float(np.mean(flips)),
        "mppi_inference_latency_ms": float(np.mean(lat)),
    }


def eval_pil_baseline(
    mode: str,
    fk: DifferentiableForwardKinematics,
    test_ds: EventWorldDataset,
    dataset_dir: Path,
    device: torch.device,
    bc_policy: ReactiveBCPolicy | None = None,
    num_episodes: int = 50,
) -> Dict[str, Any]:
    """Evaluates Fixed-Gain PD or Reactive BC on the 50 canonical PIL test episodes."""
    records: List[Dict[str, Any]] = []
    mean_calib_mass = 2.85

    for idx in range(num_episodes):
        b = test_ds[idx]
        tid = int(b["trajectory_id"])
        mass, shape = load_trajectory_metadata(dataset_dir, tid)
        events = b["events"].unsqueeze(0).to(device)
        q_gt = b["joint_positions"].to(device)
        p_gt = b["joint_velocities"].to(device)
        actions = b["actions"].to(device)
        phase_flags = b["phase_flags"].to(device)

        p1 = (phase_flags == 1).nonzero()
        t_c = int(p1[0].item()) if len(p1) > 0 else 50
        t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
        t_start = max(5, t_c - 20)
        target_grasp_mat = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
        target_final_mat = fk.forward_homogeneous(q_gt[t_grasp : t_grasp + 1])[0].detach()

        stepper = ArticulatedArmPhysicsStepper(
            q_init=q_gt[t_start].detach().cpu().numpy(),
            p_init=p_gt[t_start].detach().cpu().numpy(),
            payload_mass=mass,
            payload_shape=shape,
        )
        sim_cam = EventCameraSimulator(
            height=events.shape[-2],
            width=events.shape[-1],
            event_threshold=0.08,
            in_channels=events.shape[2],
        )
        ev_stream = OnlineEventCameraStream(sim_cam, stepper.q)

        h_bc = None
        prev_u = actions[t_start - 1].detach().clone()
        if mode == "bc" and bc_policy is not None:
            h_bc = bc_policy.init_hidden(1, device)
            with torch.no_grad():
                u_prev_burn = torch.zeros(1, actions.shape[-1], device=device)
                for tau in range(t_start):
                    _, h_bc = bc_policy.step(events[:, tau], u_prev_burn, h_bc)
                    u_prev_burn = actions[tau : tau + 1]

        step_errors: List[float] = []
        latencies: List[float] = []
        q_history: List[np.ndarray] = []
        q_eval_t = q_gt[t_start]
        p_eval_t = p_gt[t_start]
        target_pose_eval = target_grasp_mat

        for t in range(t_start, t_grasp + 1):
            target_pose_eval = target_grasp_mat if t <= t_c else target_final_mat
            target_pos_t = target_pose_eval[:3, 3]
            t0 = time.perf_counter()
            if mode == "fixed_pd":
                q_curr_t = torch.from_numpy(stepper.q).to(device)
                q_des = _compute_se3_jacobian_step(fk, q_curr_t, target_pose_eval)
                dq_des = np.zeros_like(q_des)
                # Calibrated to mean mass 2.85 kg without online inertia adaptation
                q_phys_next, p_phys_next, _ = stepper.step_pd_target(
                    q_des=q_des,
                    dq_des=dq_des,
                    payload_attached=(t >= t_c),
                    kp=18.0 * (mean_calib_mass / 2.85),
                    kd=0.80,
                )
            elif mode == "bc" and bc_policy is not None and h_bc is not None:
                with torch.no_grad():
                    ev_in = ev_stream.step(stepper.q).to(device) if t > t_start else events[:, t_start]
                    u_bc, h_bc = bc_policy.step(ev_in, prev_u.unsqueeze(0), h_bc)
                    prev_u = u_bc[0]
                    q_des = (u_bc[0].detach().cpu().numpy() * 3.14).astype(np.float32)
                    dq_des = np.zeros_like(q_des)
                q_phys_next, p_phys_next, _ = stepper.step_pd_target(
                    q_des=q_des,
                    dq_des=dq_des,
                    payload_attached=(t >= t_c),
                )
            else:
                raise ValueError(f"Unsupported baseline mode: {mode}")
            latencies.append((time.perf_counter() - t0) * 1000.0)

            q_eval_t = torch.from_numpy(q_phys_next).to(device)
            p_eval_t = torch.from_numpy(p_phys_next).to(device)
            q_history.append(q_phys_next.copy())
            ee_t = fk(q_eval_t.unsqueeze(0))[0].detach()
            step_errors.append(float((torch.norm(ee_t - target_pos_t) * 100.0).item()))

        window = step_errors[-3:] if len(step_errors) >= 3 else step_errors
        is_spatially_stable = all(e < 3.0 for e in window)
        gt_final_vel = float(torch.norm(p_gt[t_grasp]).item())
        eff_vel_thr = resolve_effective_vel_threshold(1.5, gt_final_vel)
        final_vel = float(torch.norm(p_eval_t).item())
        is_kin_stable = final_vel < eff_vel_thr
        final_rot_err = compute_rot_error(fk, q_eval_t, target_pose_eval)
        is_success = bool(is_spatially_stable and is_kin_stable and (final_rot_err < 0.05))

        q_arr = np.stack(q_history, axis=0)
        dq = np.diff(q_arr, axis=0)
        ddq = np.diff(dq, axis=0)
        dddq = np.diff(ddq, axis=0)
        records.append({
            "trajectory_id": tid,
            "payload_mass": mass,
            "elevation": classify_elevation(float(target_grasp_mat[1, 3].item())),
            "min_placement_error_cm": min(step_errors),
            "terminal_placement_error_cm": step_errors[-1],
            "final_rot_err": final_rot_err,
            "grasp_success": is_success,
            "mean_joint_accel_rad_s2": float(np.mean(np.abs(ddq))) if ddq.shape[0] > 0 else 0.0,
            "mean_joint_jerk_rad_s3": float(np.mean(np.abs(dddq))) if dddq.shape[0] > 0 else 0.0,
            "direction_flips": int(np.sum((dq[:-1] * dq[1:]) < 0.0)) if dq.shape[0] > 1 else 0,
            "mean_latency_ms": float(np.mean(latencies)),
        })

    return _summarize_episodes(records, mode)


def train_bc_policy(
    cfg: ExperimentConfig,
    dataset_dir: Path,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
) -> ReactiveBCPolicy:
    """Trains the parameter-matched ~322k Reactive BC policy on [0, 3199] with val checkpointing."""
    splits = partition_trajectories(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
    )
    train_ids = list(range(splits["train"][0], splits["train"][1]))
    val_ids = list(range(splits["val"][0], splits["val"][1]))
    train_ds = EventWorldDataset(train_ids, dataset_dir=dataset_dir)
    val_ds = EventWorldDataset(val_ids, dataset_dir=dataset_dir)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    policy = ReactiveBCPolicy(
        in_channels=cfg.model.in_channels,
        height=cfg.environment.height,
        width=cfg.environment.width,
        encoder_dim=cfg.model.encoder_dim,
        action_dim=cfg.environment.action_dim,
        hidden_dim=242,
    ).to(device)
    param_cnt = sum(p.numel() for p in policy.parameters())
    chronicle.log_info(f"ReactiveBCPolicy parameter count: {param_cnt:,}")

    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-5)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None

    for ep in range(1, epochs + 1):
        policy.train()
        tot_loss = 0.0
        n_batches = 0
        ss_prob = min(0.5, 0.5 * (ep / max(1, epochs)))
        for batch in loader:
            ev = batch["events"].to(device)
            act = batch["actions"].to(device)
            opt.zero_grad()
            pred = policy.forward_sequence(ev, act, scheduled_sampling_prob=ss_prob)
            mse_loss = loss_fn(pred, act)
            smooth_loss = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2)
            loss = mse_loss + 0.1 * smooth_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            tot_loss += float(mse_loss.item())
            n_batches += 1
        scheduler.step()

        policy.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                ev = batch["events"].to(device)
                act = batch["actions"].to(device)
                pred = policy.forward_sequence(ev, act, scheduled_sampling_prob=0.0)
                val_loss += float(loss_fn(pred, act).item())
                val_batches += 1
        mean_val = val_loss / max(1, val_batches)
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
        chronicle.log_info(
            f"BC Epoch {ep:02d}/{epochs}: train_mse={tot_loss / max(1, n_batches):.6f} | val_mse={mean_val:.6f} (best={best_val:.6f})"
        )

    if best_state is not None:
        policy.load_state_dict(best_state, strict=True)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-World-Model & Classical Control PIL Baselines.")
    parser.add_argument("--config", type=str, default="results/spwm_v0_12_4_train_m1/config.yaml")
    parser.add_argument("--output-dir", type=str, default="results/baselines/classical_and_bc_pil")
    parser.add_argument("--bc-epochs", type=int, default=50)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.project.seed)
    device = torch.device(args.device)
    dataset_dir = Path(cfg.data.dataset_dir) if cfg.data.dataset_dir is not None else Path("sim/ExportedTrajectories")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = partition_trajectories(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
    )
    test_ids = list(range(splits["test"][0], splits["test"][1]))
    test_ds = EventWorldDataset(test_ids, dataset_dir=dataset_dir)
    fk = DifferentiableForwardKinematics(joint_count=cfg.environment.joint_count).to(device)

    chronicle.log_info("Evaluating Fixed-Gain Cartesian/Joint PD Controller (50 PIL episodes)...")
    pd_summary = eval_pil_baseline("fixed_pd", fk, test_ds, dataset_dir, device, num_episodes=args.num_episodes)
    chronicle.log_success(f"Fixed-Gain PD PIL: {pd_summary}")

    chronicle.log_info(f"Training parameter-matched Reactive BC Policy ({args.bc_epochs} epochs)...")
    bc_policy = train_bc_policy(cfg, dataset_dir, device, epochs=args.bc_epochs)
    torch.save({"model_state": bc_policy.state_dict()}, out_dir / "bc_policy.pt")

    bc_summary = eval_pil_baseline("bc", fk, test_ds, dataset_dir, device, bc_policy=bc_policy, num_episodes=args.num_episodes)
    bc_summary["parameter_count"] = sum(p.numel() for p in bc_policy.parameters())
    chronicle.log_success(f"Reactive BC PIL: {bc_summary}")

    payload = {
        "fixed_gain_pd_pil": pd_summary,
        "reactive_bc_pil": bc_summary,
    }
    with open(out_dir / "classical_bc_pil_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    chronicle.log_success(f"Saved classical & BC PIL metrics to {out_dir / 'classical_bc_pil_metrics.json'}")


if __name__ == "__main__":
    main()
