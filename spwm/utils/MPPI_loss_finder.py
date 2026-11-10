"""
Neural MPPI Loss Optimization.

Abandons algebraic cost functions. Uses CMA-ES to evolve a Micro-MLP that maps
raw physical state features (position error, orientation error, velocity, jerk)
directly to a scalar trajectory cost.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, List, Optional

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
import torch
import torch.nn.functional as F

from spwm.contracts.config import ExperimentConfig, MPPIConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders
from spwm.evaluation.mppi import LatentMPPIController, MPPIPlanResult
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import ground_initial_state, step_grounded
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


class NeuralCostMPPIController(LatentMPPIController):
    """
    Abandons algebraic cost equations for an evolved Micro-MLP.
    Maps physical state features directly to trajectory cost.
    """
    def __init__(
        self,
        predictor: LatentPredictor,
        physical_decoder: PhysicalDecoder,
        config: MPPIConfig,
        neural_weights: List[float],
        forward_kinematics: Optional[DifferentiableForwardKinematics] = None,
        critic: Optional[LatentCritic] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            predictor=predictor,
            physical_decoder=physical_decoder,
            config=config,
            forward_kinematics=forward_kinematics,
            critic=critic,
            device=device,
            seed=seed,
        )
        
        # Reconstruct the Micro-MLP from the flat list of Optuna parameters
        # Architecture: 4 Inputs -> 6 Hidden (ELU) -> 1 Output (Softplus)
        # Total Parameters: (4 * 6) + 6 bias + (6 * 1) + 1 bias = 37 parameters
        tensor_weights = torch.tensor(neural_weights, device=self.device, dtype=torch.float32)
        
        self.W1 = tensor_weights[0:24].view(4, 6)
        self.b1 = tensor_weights[24:30]
        self.W2 = tensor_weights[30:36].view(6, 1)
        self.b2 = tensor_weights[36:37]

    @torch.no_grad()
    def plan_step(self, z_grounded: torch.Tensor, target_pos: torch.Tensor, prev_action: Optional[torch.Tensor] = None, target_rot: Optional[torch.Tensor] = None) -> MPPIPlanResult:
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
        beta, decay = self.noise_beta, math.sqrt(max(0.0, 1.0 - self.noise_beta ** 2))

        curr_nominal = self.nominal_actions.clone()
        gen = self.generator
        
        for _ in range(self.num_iterations):
            eta = torch.randn(K, H, D_u, generator=gen, device=self.device) * self.sigma_tensor.view(1, 1, D_u)
            noise = torch.zeros(K, H, D_u, device=self.device)
            curr_eps = eta[:, 0]
            noise[:, 0] = curr_eps
            for tau in range(1, H):
                curr_eps = beta * curr_eps + decay * eta[:, tau]
                noise[:, tau] = curr_eps
            noise[0] = 0.0

            candidates = torch.clamp(curr_nominal.unsqueeze(0) + noise, self.action_min, self.action_max)

            # Mental unroll
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
            # NEURAL COST FUNCTION BLOCK
            # ---------------------------------------------------------
            # Feature 1: Position Error [K, H]
            delta_pos = ee_pos - target_coords.view(1, 1, 3)
            pos_err = torch.norm(delta_pos, p=2, dim=-1)
            
            # Feature 2: Orientation Error [K, H]
            if target_orientation is not None and R_ee is not None:
                t_fwd = target_orientation[:, 2].view(1, 1, 3)
                t_up = target_orientation[:, 1].view(1, 1, 3)
                rot_err = (1.0 - torch.sum(R_ee[..., :, 2] * t_fwd, dim=-1)) + (1.0 - torch.sum(R_ee[..., :, 1] * t_up, dim=-1))
            else:
                rot_err = torch.zeros(K, H, device=self.device)
            
            # Feature 3: Velocity Magnitude [K, H]
            vel = torch.norm(p_hat, p=2, dim=-1)
            
            # Feature 4: Action Smoothness / Jerk [K, H]
            delta_u_0 = candidates[:, 0] - u_prev.unsqueeze(0)
            delta_u_rest = candidates[:, 1:] - candidates[:, :-1]
            delta_u = torch.cat([delta_u_0.unsqueeze(1), delta_u_rest], dim=1)
            act_err = torch.norm(delta_u, p=2, dim=-1)

            # Stack features: Shape [K, H, 4]
            features = torch.stack([pos_err, rot_err, vel, act_err], dim=-1)
            
            # Forward pass through Micro-MLP
            # Use ELU for smooth gradients, Softplus at output to ensure strictly positive costs
            hidden = F.elu(torch.matmul(features, self.W1) + self.b1)  # [K, H, 6]
            step_costs = F.softplus(torch.matmul(hidden, self.W2) + self.b2).squeeze(-1) # [K, H]
            
            # Sum trajectory costs (running cost + heavy terminal weight via slicing)
            term_slice = slice(-self.terminal_steps, None)
            total_costs = step_costs.sum(dim=-1) + (step_costs[:, term_slice].mean(dim=-1) * 5.0)

            # MPPI Control Penalty (Information-theoretic bound)
            variance = self.sigma_tensor ** 2
            control_penalty = self.temperature * torch.sum(
                (curr_nominal.unsqueeze(0) * noise) / variance.view(1, 1, D_u), dim=(1, 2)
            )
            total_costs += control_penalty

            # Update loop
            min_idx = torch.argmin(total_costs)
            beta_weight = total_costs[min_idx]
            weights = torch.exp(-(total_costs - beta_weight) / self.temperature)
            norm_weights = weights / (torch.sum(weights) + 1e-10)

            curr_nominal = (1.0 - 0.75) * curr_nominal + 0.75 * torch.sum(norm_weights.view(K, 1, 1) * candidates, dim=0)
            curr_nominal = torch.clamp(curr_nominal, self.action_min, self.action_max)

        action_chunk = torch.clamp(curr_nominal[: self.chunk_size].clone(), self.action_min, self.action_max)
        action_to_execute = action_chunk[0].clone()
        self.nominal_actions = curr_nominal.clone()
        self.prev_action = action_to_execute.clone()

        return MPPIPlanResult(
            action=action_to_execute,
            action_chunk=action_chunk,
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


def evaluate_mppi_config(model: SPWM, controller: NeuralCostMPPIController, fk: DifferentiableForwardKinematics, batch: dict, device: torch.device) -> float:
    events = batch["events"].to(device)
    q_gt = batch["joint_positions"][0].to(device)
    phase_flags = batch["phase_flags"][0].to(device)
    actions = batch.get("actions", None)
    if actions is not None: actions = actions.to(device)

    p1_indices = (phase_flags == 1).nonzero()
    t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
    
    target_grasp_mat = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
    target_final_mat = fk.forward_homogeneous(q_gt[t_grasp : t_grasp + 1])[0].detach()

    traj_id = int(batch["trajectory_ids"][0].item()) if "trajectory_ids" in batch else 0

    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=max(5, t_c - 20), actions=actions)
    controller.reset(initial_action_bias=model.physical_decoder(z_start)[0, :6] / 3.14, seed=42000 + traj_id)
    prev_u = controller.prev_action.clone()

    step_errors: List[float] = []
    final_vel, final_rot = 0.0, 0.0

    for t in range(max(5, t_c - 20), t_grasp + 1):
        target_pose_mat = target_grasp_mat if t <= t_c else target_final_mat
        target_pos_t = target_pose_mat[:3, 3]
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
    # 1. MPPI Hyperparameters
    temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)
    noise_beta = trial.suggest_float("noise_beta", 0.2, 0.9)
    sigma_base = trial.suggest_float("sigma_base", 0.1, 0.4)
    sigma_shoulder = trial.suggest_float("sigma_shoulder", 0.2, 0.8)
    sigma_elbow = trial.suggest_float("sigma_elbow", 0.2, 0.6)

    # 2. Evolve the 37 Neural Weights for the Cost Function
    # CMA-ES will handle these correlations beautifully
    neural_weights = [trial.suggest_float(f"nw_{i}", -2.0, 2.0) for i in range(37)]

    mppi_cfg = MPPIConfig(
        num_samples=256,
        horizon=20,
        num_iterations=3,
        noise_sigma=(sigma_base, sigma_shoulder, sigma_elbow, sigma_base, sigma_base, sigma_base),
        noise_beta=noise_beta,
        temperature=temperature,
    )

    controller = NeuralCostMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        config=mppi_cfg,
        neural_weights=neural_weights,
        forward_kinematics=fk,
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
    parser.add_argument("--config", type=str, default=str(GLOBAL_VERSION.m1_config()))
    parser.add_argument("--checkpoint", type=str, default=str(GLOBAL_VERSION.m1_checkpoint()))
    parser.add_argument("--n-trials", type=int, default=75)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deploying neural cost optimization on {device}...")
    
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

    study = optuna.create_study(direction="minimize", study_name="mppi_neural_optimization", sampler=CmaEsSampler())
    study.optimize(lambda trial: objective(trial, model, fk, loaders["val"], device), n_trials=args.n_trials)

    print("\nOptimization Finished.")
    print(f"  Best Score (Penalty): {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
