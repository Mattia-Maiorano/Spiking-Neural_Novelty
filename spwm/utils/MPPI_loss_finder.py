"""
Structural MPPI Loss Optimization (SPWM-v0.7.3).

Elevates optimization from scalar weight tuning to structural function shaping.
Dynamically alters cost exponents, norm types, and distance-gated thresholds
to discover the optimal mathematical form of the MPPI cost function.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, List, Optional

import numpy as np
import optuna
import torch

from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.mppi import LatentMPPIController, MPPIPlanResult
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import ground_initial_state, step_grounded
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


class MorphableMPPIController(LatentMPPIController):
    """
    Subclasses LatentMPPIController to intercept and dynamically shape the cost functions.
    Exposes algebraic parameters (exponents, norms, thresholds) to external optimization.
    """
    def __init__(
        self,
        *args,
        pos_gamma: float = 1.0,
        pos_norm_p: int = 2,
        vel_engagement_radius: float = 0.15,
        vel_gamma: float = 1.0,
        act_gamma: float = 2.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pos_gamma = pos_gamma
        self.pos_norm_p = pos_norm_p
        self.vel_engagement_radius = vel_engagement_radius
        self.vel_gamma = vel_gamma
        self.act_gamma = act_gamma

    @torch.no_grad()
    def plan_step(self, z_grounded: torch.Tensor, target_pos: torch.Tensor, prev_action: Optional[torch.Tensor] = None, target_rot: Optional[torch.Tensor] = None) -> MPPIPlanResult:
        # ---------------------------------------------------------
        # Pre-amble and Mental Simulation (Identical to base class)
        # ---------------------------------------------------------
        start_time = time.perf_counter()
        z_init = z_grounded[0] if z_grounded.dim() == 2 else z_grounded
        
        target_orientation = None
        if target_pos.shape == (4, 4) or (target_pos.dim() == 3 and target_pos.shape[-2:] == (4, 4)):
            t_mat = target_pos[0] if target_pos.dim() == 3 else target_pos
            target_coords = t_mat[:3, 3].to(device=self.device, dtype=torch.float32)
            target_orientation = t_mat[:3, :3].to(device=self.device, dtype=torch.float32)
        else:
            target_pos_3d = target_pos[0] if target_pos.dim() == 2 else target_pos
            target_coords = target_pos_3d.to(device=self.device, dtype=torch.float32)

        u_prev = prev_action.to(self.device) if prev_action is not None else self.prev_action
        K, H, D_u = self.num_samples, self.horizon, self.action_dim
        beta = self.noise_beta
        decay = math.sqrt(max(0.0, 1.0 - beta ** 2))

        curr_nominal = self.nominal_actions.clone()
        
        best_z_traj = torch.zeros(H, self.predictor.latent_dim, device=self.device)
        best_ee_traj = torch.zeros(H, 3, device=self.device)
        candidate_ee = torch.zeros(K, H, 3, device=self.device)
        candidate_w = torch.zeros(K, device=self.device)

        for _ in range(self.num_iterations):
            eta = torch.randn(K, H, D_u, device=self.device) * self.sigma_tensor.view(1, 1, D_u)
            noise = torch.zeros(K, H, D_u, device=self.device)
            curr_eps = eta[:, 0]
            noise[:, 0] = curr_eps
            for tau in range(1, H):
                curr_eps = beta * curr_eps + decay * eta[:, tau]
                noise[:, tau] = curr_eps
            noise[0] = 0.0

            candidates = curr_nominal.unsqueeze(0) + noise
            candidates = torch.clamp(candidates, self.action_min, self.action_max)

            z_batch = z_init.unsqueeze(0).repeat(K, 1)
            z_traj_list = []
            curr_z = z_batch
            for tau in range(H):
                pred_out = self.predictor(curr_z, candidates[:, tau])
                curr_z = pred_out.predicted_latent
                z_traj_list.append(curr_z)

            z_trajectories = torch.stack(z_traj_list, dim=1)
            decoded = self.physical_decoder(z_trajectories)
            q_hat = decoded[..., : self.physical_decoder.joint_count]
            p_hat = decoded[..., self.physical_decoder.joint_count :]

            if target_orientation is not None:
                T_ee = self.fk.forward_homogeneous(q_hat)
                ee_pos = T_ee[..., :3, 3]
                R_ee = T_ee[..., :3, :3]
            else:
                ee_pos = self.fk(q_hat)
                R_ee = None

            # ---------------------------------------------------------
            # SHAPED COST EVALUATION BLOCK (Overrides Section 5)
            # ---------------------------------------------------------
            term_slice = slice(-self.terminal_steps, None)
            delta_pos = ee_pos - target_coords.view(1, 1, 3)
            
            # A. Structurally Shaped Position Cost[span_1](start_span)[span_1](end_span)
            # Optuna controls if this is L1 (Manhattan) or L2 (Euclidean), and applies the gamma shape.
            axis_weights = torch.tensor([1.0, 2.5, 1.0], device=self.device)
            raw_pos_err = torch.norm(delta_pos * axis_weights.view(1, 1, 3), p=self.pos_norm_p, dim=-1)
            pos_err_weighted = raw_pos_err ** self.pos_gamma 
            
            running_pos_cost = self.w_pos * pos_err_weighted.sum(dim=-1)
            terminal_pos_cost = self.w_term_pos * pos_err_weighted[:, term_slice].mean(dim=-1)

            # B. SE(3) Orientation Cost (unchanged structural mapping)
            if target_orientation is not None and R_ee is not None:
                t_fwd = target_orientation[:, 2].view(1, 1, 3)
                t_up = target_orientation[:, 1].view(1, 1, 3)
                ee_fwd = R_ee[..., :, 2]
                ee_up = R_ee[..., :, 1]
                rot_err_step = (1.0 - torch.sum(ee_fwd * t_fwd, dim=-1)) + (1.0 - torch.sum(ee_up * t_up, dim=-1))
                running_rot_cost = self.w_rot * rot_err_step.sum(dim=-1)
                terminal_rot_cost = self.w_term_rot * rot_err_step[:, term_slice].mean(dim=-1)
            else:
                running_rot_cost = torch.zeros(K, device=self.device)
                terminal_rot_cost = torch.zeros(K, device=self.device)

            # C. Structurally Shaped Velocity Penalty[span_2](start_span)[span_2](end_span)
            vel_norm = torch.norm(p_hat, p=2, dim=-1)
            true_dist = torch.norm(delta_pos, p=2, dim=-1)
            
            # Optuna dictates when friction turns on, and how sharply it engages (linear vs steep curve)
            dist_scaling = torch.clamp(true_dist / self.vel_engagement_radius, min=0.0, max=1.0)
            engagement_curve = (1.0 - dist_scaling) ** self.vel_gamma
            vel_cost = self.w_vel * torch.sum(vel_norm * engagement_curve, dim=-1)

            # D. Structurally Shaped Action Smoothness[span_3](start_span)[span_3](end_span)
            delta_u_0 = candidates[:, 0] - u_prev.unsqueeze(0)
            delta_u_rest = candidates[:, 1:] - candidates[:, :-1]
            delta_u = torch.cat([delta_u_0.unsqueeze(1), delta_u_rest], dim=1)
            
            # Optuna determines if jerk is penalized quadratically (smooth) or linearly (sparse changes)
            delta_u_norm = torch.norm(delta_u, p=2, dim=-1)
            act_cost = self.w_act * torch.sum(delta_u_norm ** self.act_gamma, dim=-1)

            # E. MPPI Control Penalty[span_4](start_span)[span_4](end_span)
            variance = self.sigma_tensor ** 2
            control_penalty = self.temperature * torch.sum(
                (curr_nominal.unsqueeze(0) * noise) / variance.view(1, 1, D_u), dim=(1, 2)
            )

            total_costs = running_pos_cost + terminal_pos_cost + running_rot_cost + terminal_rot_cost + vel_cost + act_cost + control_penalty

            min_idx = torch.argmin(total_costs)
            beta_weight = total_costs[min_idx]
            weights = torch.exp(-(total_costs - beta_weight) / self.temperature)
            norm_weights = weights / (torch.sum(weights) + 1e-10)

            new_nominal = torch.sum(norm_weights.view(K, 1, 1) * candidates, dim=0)
            curr_nominal = (1.0 - 0.75) * curr_nominal + 0.75 * new_nominal
            curr_nominal = torch.clamp(curr_nominal, self.action_min, self.action_max)

        action_to_execute = curr_nominal[0].clone()
        self.nominal_actions = curr_nominal.clone()
        self.prev_action = action_to_execute.clone()

        return MPPIPlanResult(
            action=action_to_execute,
            nominal_action_sequence=curr_nominal,
            predicted_latent_trajectory=z_trajectories[min_idx],
            predicted_ee_trajectory=ee_pos[min_idx],
            min_cost=float(total_costs[min_idx].item()),
            mean_cost=float(total_costs.mean().item()),
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
            candidate_ee_trajectories=ee_pos,
            candidate_weights=norm_weights,
        )


def compute_rot_error(fk: DifferentiableForwardKinematics, q: torch.Tensor, target_pose_mat: torch.Tensor) -> float:
    T_ee = fk.forward_homogeneous(q.unsqueeze(0))[0]
    R_ee, target_rot = T_ee[:3, :3], target_pose_mat[:3, :3]
    return float((1.0 - torch.sum(R_ee[:, 2] * target_rot[:, 2], dim=-1) + 1.0 - torch.sum(R_ee[:, 1] * target_rot[:, 1], dim=-1)).item())


def evaluate_mppi_config(model: SPWM, controller: MorphableMPPIController, fk: DifferentiableForwardKinematics, batch: dict, device: torch.device) -> float:
    events = batch["events"].to(device)
    q_gt = batch["joint_positions"][0].to(device)
    phase_flags = batch["phase_flags"][0].to(device)
    actions = batch.get("actions", None)
    if actions is not None: actions = actions.to(device)

    p1_indices = (phase_flags == 1).nonzero()
    t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
    
    target_pose_mat = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
    target_pos_t = target_pose_mat[:3, 3]

    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=max(5, t_c - 20), actions=actions)
    controller.reset(initial_action_bias=model.physical_decoder(z_start)[0, :6] / 3.14)
    prev_u = controller.prev_action.clone()

    step_errors: List[float] = []
    final_vel, final_rot = 0.0, 0.0

    for t in range(max(5, t_c - 20), t_grasp + 1):
        z_t, dyn_state = step_grounded(model, events[:, t], dyn_state, in_sensory_phase=True, unconstrained=True, action=prev_u.unsqueeze(0))
        res = controller.plan_step(z_t, target_pose_mat, prev_action=prev_u)
        prev_u = res.action

        dec_t = model.physical_decoder(z_t)
        ee_t = fk(dec_t[:, :6])[0].detach()
        step_errors.append(float((torch.norm(ee_t - target_pos_t) * 100.0).item()))

        if t == t_grasp:
            final_vel = float(torch.norm(dec_t[0, 6:12]).item())
            final_rot = compute_rot_error(fk, dec_t[0, :6], target_pose_mat)

    # Calculate Continuous Shaping Score
    window_errors = step_errors[-3:] if len(step_errors) >= 3 else step_errors
    steady_state_error = np.mean(window_errors)
    
    pos_penalty = (steady_state_error / 3.0) ** 2  
    vel_penalty = (final_vel / 1.5) ** 2           
    rot_penalty = (final_rot / 0.05) ** 2          

    score = pos_penalty + vel_penalty + rot_penalty
    if all(err < 3.0 for err in window_errors) and final_vel < 1.5 and final_rot < 0.05:
        score -= 2.0  

    return float(score)


def objective(trial: optuna.Trial, model: SPWM, fk: DifferentiableForwardKinematics, val_loader: Any, device: torch.device) -> float:
    # 1. Morphological Structure Parameters
    pos_gamma = trial.suggest_float("pos_gamma", 0.5, 3.0)          # <1.0 = robust to outliers, >1.0 = steep wall
    pos_norm_p = trial.suggest_categorical("pos_norm_p", [1, 2])    # L1 vs L2 geometry
    vel_radius = trial.suggest_float("vel_radius", 0.05, 0.35)      # Engagement radius (m)
    vel_gamma = trial.suggest_float("vel_gamma", 0.5, 4.0)          # Friction engagement curve
    act_gamma = trial.suggest_float("act_gamma", 1.0, 3.0)          # Action delta penalty shape
    
    # 2. Base Weights & Noise
    w_act = trial.suggest_float("w_act", 1e-4, 5e-2, log=True)
    w_vel = trial.suggest_float("w_vel", 1e-3, 1.0, log=True)
    w_rot = trial.suggest_float("w_rot", 0.1, 3.0)
    w_term_pos = trial.suggest_float("w_term_pos", 1.0, 15.0)
    temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)
    noise_beta = trial.suggest_float("noise_beta", 0.2, 0.9)
    
    # 3. Anisotropic Noise Configuration
    sigma_base = trial.suggest_float("sigma_base", 0.1, 0.4)
    sigma_shoulder = trial.suggest_float("sigma_shoulder", 0.2, 0.8)
    sigma_elbow = trial.suggest_float("sigma_elbow", 0.2, 0.6)

    controller = MorphableMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        forward_kinematics=fk,
        num_samples=256,
        horizon=20,
        num_iterations=3,
        noise_sigma=[sigma_base, sigma_shoulder, sigma_elbow, sigma_base, sigma_base, sigma_base],
        noise_beta=noise_beta,
        temperature=temperature,
        w_pos=1.0,
        w_rot=w_rot,
        w_vel=w_vel,
        w_act=w_act,
        w_term_pos=w_term_pos,
        w_term_rot=w_rot * 4.0,
        pos_gamma=pos_gamma,
        pos_norm_p=pos_norm_p,
        vel_engagement_radius=vel_radius,
        vel_gamma=vel_gamma,
        act_gamma=act_gamma,
        device=device,
    )

    total_score = 0.0
    episodes = 10
    
    for idx, batch in enumerate(val_loader):
        if idx >= episodes: break
        total_score += evaluate_mppi_config(model, controller, fk, batch, device)
        trial.report(total_score, idx)
        if trial.should_prune(): raise optuna.TrialPruned()

    return total_score / episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="results/spwm_v0_7_3_train_m1/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="results/spwm_v0_7_3_train_m1/model.pt")
    parser.add_argument("--n-trials", type=int, default=75)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying structural optimization on {device}...")
    
    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.project.seed)
    
    model = SPWM.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
    model.eval()

    fk = DifferentiableForwardKinematics().to(device)
    loaders = create_dataloaders(
        total_trajectories=cfg.data.total_trajectories, train_split=cfg.data.train_split, val_split=cfg.data.val_split,
        sequence_length=cfg.data.sequence_length, batch_size=1, dataset_dir=cfg.data.dataset_dir, cache_data=False
    )

    study = optuna.create_study(direction="minimize", study_name="mppi_structural_optimization")
    study.optimize(lambda trial: objective(trial, model, fk, loaders["val"], device), n_trials=args.n_trials)

    print("\nOptimization Finished.")
    print(f"  Best Score (Penalty): {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
