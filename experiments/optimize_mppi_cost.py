"""
Neural MPPI Loss Optimization.

Abandons algebraic cost functions. Uses CMA-ES to evolve a Micro-MLP that maps
raw physical state features (position error, orientation error, velocity, jerk)
directly to a scalar trajectory cost.
"""

from __future__ import annotations

import argparse
from typing import Any, List

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
import torch

from spwm.contracts.config import ExperimentConfig, MPPIConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import create_dataloaders_from_config
from spwm.evaluation.mppi_controller import NeuralCostMPPIController
from spwm.models.kinematics import DifferentiableForwardKinematics, compute_rot_error
from spwm.models.rollout import ground_initial_state, step_grounded
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def evaluate_mppi_config(
    model: SPWM,
    controller: NeuralCostMPPIController,
    fk: DifferentiableForwardKinematics,
    batch: dict,
    device: torch.device,
) -> float:
    events = batch["events"].to(device)
    q_gt = batch["joint_positions"][0].to(device)
    phase_flags = batch["phase_flags"][0].to(device)
    actions = batch["actions"].to(device)
    traj_id = int(batch["trajectory_ids"][0].item())

    p1_indices = (phase_flags == 1).nonzero()
    t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)

    target_grasp_mat = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
    target_final_mat = fk.forward_homogeneous(q_gt[t_grasp : t_grasp + 1])[0].detach()

    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=max(5, t_c - 20), actions=actions)
    controller.reset(initial_action_bias=model.physical_decoder(z_start)[0, :6] / 3.14, seed=42000 + traj_id)
    prev_u = controller.prev_action.clone()

    step_errors: List[float] = []
    final_vel, final_rot = 0.0, 0.0

    for t in range(max(5, t_c - 20), t_grasp + 1):
        target_pose_mat = target_grasp_mat if t <= t_c else target_final_mat
        target_pos_t = target_pose_mat[:3, 3]
        z_t, dyn_state = step_grounded(
            model,
            events[:, t],
            dyn_state,
            in_sensory_phase=True,
            unconstrained=True,
            action=prev_u.unsqueeze(0),
        )
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


def objective(
    trial: optuna.Trial,
    model: SPWM,
    fk: DifferentiableForwardKinematics,
    val_loader: Any,
    device: torch.device,
) -> float:
    # 1. MPPI Hyperparameters
    temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)
    noise_beta = trial.suggest_float("noise_beta", 0.2, 0.9)
    sigma_base = trial.suggest_float("sigma_base", 0.1, 0.4)
    sigma_shoulder = trial.suggest_float("sigma_shoulder", 0.2, 0.8)
    sigma_elbow = trial.suggest_float("sigma_elbow", 0.2, 0.6)

    # 2. Evolve the 37 Neural Weights for the Cost Function
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
        if idx >= episodes:
            break
        total_score += evaluate_mppi_config(model, controller, fk, batch, device)
        trial.report(total_score, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return total_score / episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Neural MPPI Cost Weights via Optuna CMA-ES.")
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
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    fk = DifferentiableForwardKinematics().to(device)
    loaders = create_dataloaders_from_config(cfg, batch_size=1, cache_data=False)

    study = optuna.create_study(direction="minimize", study_name="mppi_neural_optimization", sampler=CmaEsSampler())
    study.optimize(lambda trial: objective(trial, model, fk, loaders["val"], device), n_trials=args.n_trials)

    print("\nOptimization Finished.")
    print(f"  Best Score (Penalty): {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()

