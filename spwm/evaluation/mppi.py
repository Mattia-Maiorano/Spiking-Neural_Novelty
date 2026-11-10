from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from spwm.contracts.config import MPPIConfig
from spwm.contracts.protocols import Tier2EvaluatorProtocol
from spwm.contracts.tensors import J, assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentCritic, LatentPredictor, PhysicalDecoder
from spwm.models.rollout import ground_initial_state, step_grounded


@dataclass(frozen=True)
class MPPIPlanResult:
    """Diagnostic outputs from a single MPPI planning step."""

    action: torch.Tensor
    action_chunk: torch.Tensor
    nominal_action_sequence: torch.Tensor
    predicted_latent_trajectory: torch.Tensor
    predicted_ee_trajectory: torch.Tensor
    min_cost: float
    mean_cost: float
    latency_ms: float
    candidate_ee_trajectories: torch.Tensor  # [K, H, 3] FK end-effector paths of the final MPPI iteration
    candidate_weights: torch.Tensor          # [K] normalized path-integral weights of the final MPPI iteration


class LatentMPPIController:
    """
    Downstream Model Predictive Path Integral (MPPI) controller operating in SPWM latent space.

    Executes closed-loop downstream control inside SPWM's latent phase space.
    Optimizes motor action sequences u_{t:t+H}^* without external physics simulation:
      1. Receives grounded latent state z_t in R^128 (assimilated via active perception).
      2. Mental simulation: autoregressively unrolls isolated LatentPredictor (M1)
      3. Decodes kinematics via PhysicalDecoder and computes end-effector 3D position
         via DifferentiableForwardKinematics.
      4. Minimizes Cartesian goal distance, velocity damping, action smoothness costs,
         and Critic terminal cost-to-go bootstrapping.
    """

    def __init__(
        self,
        predictor: LatentPredictor,
        physical_decoder: PhysicalDecoder,
        config: MPPIConfig,
        forward_kinematics: Optional[DifferentiableForwardKinematics] = None,
        critic: Optional[LatentCritic] = None,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        if config.chunk_size <= 0 or config.chunk_size > config.horizon:
            raise ValueError(f"chunk_size must be in 1..{config.horizon}, got {config.chunk_size}")
        resolved_term_steps = int(config.terminal_steps) if config.terminal_steps is not None else min(4, config.horizon)
        if resolved_term_steps <= 0 or resolved_term_steps > config.horizon:
            raise ValueError(f"terminal_steps must be in 1..{config.horizon}, got {resolved_term_steps}")

        self.config = config
        self.predictor = predictor
        self.physical_decoder = physical_decoder
        self.fk = forward_kinematics if forward_kinematics is not None else DifferentiableForwardKinematics()
        self.critic = critic
        self.chunk_size = int(config.chunk_size)
        self.gamma_discount = float(config.gamma_discount)

        self.num_samples = config.num_samples
        self.horizon = config.horizon
        self.terminal_steps = resolved_term_steps
        self.action_dim = config.action_dim
        self.action_min = float(config.action_min)
        self.action_max = float(config.action_max)
        self.noise_beta = float(config.noise_beta)
        self.temperature = float(config.temperature)
        self.num_iterations = config.num_iterations

        # Autonomous weights definition
        self.w_pos = float(config.w_pos)
        self.w_rot = float(config.w_rot)
        self.w_jerk = float(config.w_jerk)
        self.w_damping = float(config.w_damping)
        self.w_vel = 0.0
        self.w_act = self.w_jerk
        self.w_term_pos = 0.0
        self.w_term_rot = 0.0
        self.w_altitude = 0.0
        self.w_progress = float(config.w_progress)

        dev = device if device is not None else next(predictor.parameters()).device
        self.device = dev

        # Covariance Sigma: boost variance on Shoulder_Lift (idx 1) and Elbow_Flex (idx 2)
        noise_sigma = config.noise_sigma
        action_dim = config.action_dim
        if isinstance(noise_sigma, torch.Tensor):
            if noise_sigma.numel() == 1:
                sigmas = torch.full((action_dim,), float(noise_sigma.item()), device=dev, dtype=torch.float32)
            else:
                sigmas = noise_sigma.to(device=dev, dtype=torch.float32).flatten()
        elif isinstance(noise_sigma, (list, tuple)):
            sigmas = torch.tensor(noise_sigma, device=dev, dtype=torch.float32)
        else:
            base_s = float(noise_sigma)
            sigmas = torch.full((action_dim,), base_s, device=dev, dtype=torch.float32)
            if action_dim >= 3:
                # Anisotropic covariance boost: Shoulder_Lift (idx 1) +60%, Elbow_Flex (idx 2) +45%
                sigmas[1] = base_s * 3.5
                sigmas[2] = base_s * 2.0


        if sigmas.shape[0] != action_dim:
            raise ValueError(f"noise_sigma shape {tuple(sigmas.shape)} does not match action_dim {action_dim}")

        self.noise_sigma = float(sigmas.mean().item())
        self.sigma_tensor = sigmas
        self.nominal_actions = torch.zeros(self.horizon, self.action_dim, device=self.device)
        self.prev_action = torch.zeros(self.action_dim, device=self.device)
        self.generator: Optional[torch.Generator] = None
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Sets a deterministic RNG seed for MPPI noise generation (Zero-Stochasticity Guarantee)."""
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def reset(self, initial_action_bias: Optional[torch.Tensor] = None, seed: Optional[int] = None) -> None:
        """Resets nominal action sequence and previous action buffer, optionally re-seeding RNG."""
        if seed is not None:
            self.set_seed(seed)
        if initial_action_bias is not None:
            if initial_action_bias.shape != (self.action_dim,):
                raise ValueError(
                    f"initial_action_bias must have shape ({self.action_dim},), got {tuple(initial_action_bias.shape)}"
                )
            self.nominal_actions = initial_action_bias.unsqueeze(0).repeat(self.horizon, 1).to(self.device)
            self.prev_action = initial_action_bias.clone().to(self.device)
        else:
            self.nominal_actions = torch.zeros(self.horizon, self.action_dim, device=self.device)
            self.prev_action = torch.zeros(self.action_dim, device=self.device)

    @torch.no_grad()
    def plan_step(
        self,
        z_grounded: torch.Tensor,
        target_pos: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MPPIPlanResult:
        """
        Executes MPPI optimization at control step t starting from grounded latent state z_t.

        Args:
            z_grounded: Grounded latent state z_t [128] or [1, 128].
            target_pos: Target 3D Cartesian coordinates (x, y, z) [3] or [1, 3], or full 4x4 SE(3) matrix.
            prev_action: Optional previous action u_{t-1} [action_dim].
            target_rot: Optional target 3x3 rotation matrix [3, 3] if not embedded in target_pos.

        Returns:
            MPPIPlanResult with planned immediate action u_t^* and planning diagnostics.
        """
        start_time = time.perf_counter()

        if z_grounded.dim() == 2:
            z_init = z_grounded[0]
        else:
            z_init = z_grounded

        assert_tensor_contract(z_init, (self.predictor.latent_dim,), "LatentMPPIController.z_init")

        target_orientation: Optional[torch.Tensor] = None
        if target_pos.shape == (4, 4) or (target_pos.dim() == 3 and target_pos.shape[-2:] == (4, 4)):
            # Full 4x4 SE(3) transformation matrix passed as target
            t_mat = target_pos[0] if target_pos.dim() == 3 else target_pos
            target_coords = t_mat[:3, 3].to(device=self.device, dtype=torch.float32)
            target_orientation = t_mat[:3, :3].to(device=self.device, dtype=torch.float32)
        else:
            if target_pos.dim() == 2:
                target_pos_3d = target_pos[0]
            else:
                target_pos_3d = target_pos
            assert_tensor_contract(target_pos_3d, (3,), "LatentMPPIController.target_pos")
            target_coords = target_pos_3d.to(device=self.device, dtype=torch.float32)

            if target_rot is not None:
                t_rot = target_rot[0] if target_rot.dim() == 3 else target_rot
                assert_tensor_contract(t_rot, (3, 3), "LatentMPPIController.target_rot")
                target_orientation = t_rot.to(device=self.device, dtype=torch.float32)

        u_prev = prev_action.to(self.device) if prev_action is not None else self.prev_action

        K = self.num_samples
        H = self.horizon
        D_u = self.action_dim
        beta = self.noise_beta
        decay = math.sqrt(max(0.0, 1.0 - beta ** 2))

        curr_nominal = self.nominal_actions.clone()
        min_cost_val = float("inf")
        mean_cost_val = 0.0

        best_z_traj = torch.zeros(H, self.predictor.latent_dim, device=self.device)
        best_ee_traj = torch.zeros(H, 3, device=self.device)
        candidate_ee = torch.zeros(K, H, 3, device=self.device)
        candidate_w = torch.zeros(K, device=self.device)
        gen = generator if generator is not None else self.generator

        # Compute starting Cartesian position p_0 for directional progress
        decoded_init = self.physical_decoder(z_init.unsqueeze(0))
        q_init = decoded_init[:, : self.physical_decoder.joint_count]
        p_init = self.fk(q_init)[0]

        for _ in range(self.num_iterations):
            # 1. Sample temporally correlated action perturbations: AR(1) process
            eta = torch.randn(K, H, D_u, generator=gen, device=self.device) * self.sigma_tensor.view(1, 1, D_u)
            noise = torch.zeros(K, H, D_u, device=self.device)
            curr_eps = eta[:, 0]
            noise[:, 0] = curr_eps
            for tau in range(1, H):
                curr_eps = beta * curr_eps + decay * eta[:, tau]
                noise[:, tau] = curr_eps

            # Preserve current nominal sequence as candidate 0
            noise[0] = 0.0

            candidates = curr_nominal.unsqueeze(0) + noise
            candidates = torch.clamp(candidates, self.action_min, self.action_max)

            # 2. Mental Simulation: Autoregressively unroll M1 predictor over K trajectories
            z_batch = z_init.unsqueeze(0).repeat(K, 1)  # [K, latent_dim]
            z_traj_list = []

            curr_z = z_batch
            for tau in range(H):
                u_tau = candidates[:, tau]  # [K, D_u]
                pred_out = self.predictor(curr_z, u_tau)
                curr_z = pred_out.predicted_latent
                z_traj_list.append(curr_z)

            z_trajectories = torch.stack(z_traj_list, dim=1)  # [K, H, latent_dim]

            # 3. Pass M1's latent trajectory output through PhysicalDecoder
            decoded = self.physical_decoder(z_trajectories)  # [K, H, 2 * joint_count]
            q_hat = decoded[..., : self.physical_decoder.joint_count]  # [K, H, 6]

            # 4. Compute end-effector Cartesian positions and poses via Differentiable FK
            if target_orientation is not None:
                T_ee = self.fk.forward_homogeneous(q_hat)  # [K, H, 4, 4]
                ee_pos = T_ee[..., :3, 3]                   # [K, H, 3]
                R_ee = T_ee[..., :3, :3]                    # [K, H, 3, 3]
            else:
                ee_pos = self.fk(q_hat)                     # [K, H, 3]
                R_ee = None

            # 5. Evaluate Lifting-Aware Physics-Grounded Trajectory Cost
            total_costs = self.compute_trajectory_cost(
                ee_pos=ee_pos,
                target_coords=target_coords,
                u_seq=candidates,
                u_prev=u_prev,
                R_ee=R_ee,
                target_orientation=target_orientation,
                z_trajectories=z_trajectories,
                p_init=p_init,
            )


            min_idx = torch.argmin(total_costs)
            min_cost_val = float(total_costs[min_idx].item())
            mean_cost_val = float(total_costs.mean().item())

            best_z_traj = z_trajectories[min_idx]
            best_ee_traj = ee_pos[min_idx]

            # 6. Softmax Path Integral Importance Weighting
            beta_weight = total_costs[min_idx]
            weights = torch.exp(-(total_costs - beta_weight) / self.temperature)
            denom = torch.sum(weights)
            norm_weights = weights / (denom + 1e-10)  # [K]
            candidate_ee = ee_pos
            candidate_w = norm_weights

            # 7. Update Nominal Action Sequence with EMA (Control Inertia)
            new_nominal = torch.sum(norm_weights.view(K, 1, 1) * candidates, dim=0)
            
            # gamma: Update step size (1.0 = instant jump, 0.6-0.9 = smooth tracking)
            gamma = 0.75 
            curr_nominal = (1.0 - gamma) * curr_nominal + gamma * new_nominal
            curr_nominal = torch.clamp(curr_nominal, self.action_min, self.action_max)

        # Selected action chunk and immediate action to execute, strictly clamped against EMA drift
        action_chunk = torch.clamp(curr_nominal[: self.chunk_size].clone(), self.action_min, self.action_max)
        action_to_execute = action_chunk[0].clone()

        self.nominal_actions = curr_nominal.clone()
        self.prev_action = action_to_execute.clone()


        latency = (time.perf_counter() - start_time) * 1000.0

        return MPPIPlanResult(
            action=action_to_execute,
            action_chunk=action_chunk,
            nominal_action_sequence=curr_nominal,
            predicted_latent_trajectory=best_z_traj,
            predicted_ee_trajectory=best_ee_traj,
            min_cost=min_cost_val,
            mean_cost=mean_cost_val,
            latency_ms=latency,
            candidate_ee_trajectories=candidate_ee,
            candidate_weights=candidate_w,
        )

    def compute_trajectory_cost(
        self,
        ee_pos: torch.Tensor,
        target_coords: torch.Tensor,
        u_seq: torch.Tensor,
        u_prev: torch.Tensor,
        R_ee: Optional[torch.Tensor] = None,
        target_orientation: Optional[torch.Tensor] = None,
        z_trajectories: Optional[torch.Tensor] = None,
        p_init: Optional[torch.Tensor] = None,
        w_progress: Optional[float] = None,
        w_jerk: Optional[float] = None,
        w_damping: Optional[float] = None,
    ) -> torch.Tensor:
        """Evaluates the physics-grounded, lifting-aware MPPI trajectory cost over K candidates.

        Formulation per step t:
          - Positional Cost: W_pos * ||pos_current - target_pos||_2
          - Rotational Cost: W_rot * (1 - cos(theta_error))
          - Action Smoothness (Jerk Penalty): Constant W_jerk * ||u_t - u_{t-1}||_2^2 across horizon
          - Directional Progress Reward: - W_progress * sum(progress_t * dist_factor) (deadband attenuation)
          - Cartesian Velocity Damping: W_damping * sum(||delta_p_t|| * (1 - dist_factor)) (deadband braking)
          - Terminal Critic Bootstrapping: (gamma ** H) * V(z_H)
        """
        K, H, _ = ee_pos.shape
        # 1. Positional Cost: W_pos * ||pos_current - target_pos||_2
        pos_err = torch.norm(ee_pos - target_coords.view(1, 1, 3), p=2, dim=-1)  # [K, H]
        pos_cost = self.w_pos * pos_err.sum(dim=-1)  # [K]

        # Distance-based damping factor (Deadband):
        # Full reward / zero velocity damping when >= 13cm, dropping to 0 reward / full velocity damping when within 3cm
        dist_factor = torch.clamp((pos_err - 0.03) / 0.10, 0.0, 1.0)  # [K, H]

        # 2. Rotational Cost: W_rot * (1 - cos(theta_error))
        if target_orientation is not None and R_ee is not None:
            r_rel = torch.matmul(R_ee.transpose(-1, -2), target_orientation.view(1, 1, 3, 3))
            trace_r = torch.diagonal(r_rel, dim1=-2, dim2=-1).sum(-1)  # [K, H]
            cos_theta = torch.clamp((trace_r - 1.0) * 0.5, -1.0, 1.0)
            rot_err_step = 1.0 - cos_theta
            rot_cost = self.w_rot * rot_err_step.sum(dim=-1)  # [K]
        else:
            rot_cost = torch.zeros(K, device=ee_pos.device, dtype=ee_pos.dtype)

        # 3. Action Smoothness (Jerk Penalty): Constant W_jerk across horizon to allow braking torques
        delta_u_0 = u_seq[:, 0] - u_prev.unsqueeze(0)
        delta_u_rest = u_seq[:, 1:] - u_seq[:, :-1]
        delta_u = torch.cat([delta_u_0.unsqueeze(1), delta_u_rest], dim=1)  # [K, H, D_u]
        jerk_sq = torch.sum(delta_u ** 2, dim=-1)  # [K, H]
        curr_w_jerk = self.w_jerk if w_jerk is None else float(w_jerk)
        jerk_cost = curr_w_jerk * jerk_sq.sum(dim=-1)  # [K]

        p_0 = p_init.view(1, 1, 3).expand(K, 1, 3) if p_init is not None else ee_pos[:, 0:1, :]
        p_traj = torch.cat([p_0, ee_pos], dim=1)  # [K, H + 1, 3]
        delta_p = p_traj[:, 1:, :] - p_traj[:, :-1, :]  # [K, H, 3]

        # 4. Directional Progress Reward (Attractor toward target) with Progress Attenuation (Deadband)
        curr_w_progress = self.w_progress if w_progress is None else float(w_progress)
        if curr_w_progress != 0.0:
            dir_vec = target_coords.view(1, 1, 3) - p_traj[:, :-1, :]  # [K, H, 3]
            dir_target = dir_vec / (torch.norm(dir_vec, dim=-1, keepdim=True) + 1e-6)

            step_progress = torch.sum(delta_p * dir_target, dim=-1)  # [K, H]
            progress_reward = step_progress * curr_w_progress * dist_factor  # [K, H]
            c_progress = - progress_reward.sum(dim=-1)  # [K]
        else:
            c_progress = torch.zeros(K, device=ee_pos.device, dtype=ee_pos.dtype)

        # 5. Cartesian Velocity Damping: Activates inside deadband (when dist_factor -> 0)
        curr_w_damping = self.w_damping if w_damping is None else float(w_damping)
        if curr_w_damping != 0.0:
            cart_vel = torch.norm(delta_p, dim=-1)  # [K, H]
            damping_penalty = curr_w_damping * cart_vel * (1.0 - dist_factor)  # [K, H]
            c_damping = damping_penalty.sum(dim=-1)  # [K]
        else:
            c_damping = torch.zeros(K, device=ee_pos.device, dtype=ee_pos.dtype)

        total_cost = pos_cost + rot_cost + jerk_cost + c_progress + c_damping

        # 5. Terminal Value Bootstrapping (TD-MPC Critic)
        if self.critic is not None and z_trajectories is not None:
            terminal_z = z_trajectories[:, -1, :]  # [K, latent_dim]
            raw_terminal_costs = self.critic(terminal_z).squeeze(-1)  # [K]
            scaled_terminal_costs = self.w_pos * raw_terminal_costs
            total_cost = total_cost + (self.gamma_discount ** H) * scaled_terminal_costs

        return total_cost


def compute_rot_error(
    fk: DifferentiableForwardKinematics,
    q: torch.Tensor,
    target_pose_mat: torch.Tensor,
) -> float:
    """Computes geodesic SO(3) rotational error theta_error in radians."""
    t_input = q.unsqueeze(0) if q.dim() == 1 else q
    T_ee = fk.forward_homogeneous(t_input)[0]
    R_ee = T_ee[:3, :3]
    target_rot = target_pose_mat[:3, :3]
    r_rel = torch.matmul(R_ee.transpose(-1, -2), target_rot)
    trace_r = torch.diagonal(r_rel, dim1=-2, dim2=-1).sum(-1)
    cos_theta = torch.clamp((trace_r - 1.0) * 0.5, -1.0, 1.0)
    theta_err = torch.acos(cos_theta)
    return float(theta_err.item())


def classify_elevation(y: float) -> str:
    """Classifies vertical Cartesian elevation into Low, Mid, or High plane."""
    if y < 1.08:
        return "Low"
    elif y < 1.18:
        return "Mid"
    else:
        return "High"


def compute_structural_score(
    step_errors: Sequence[float],
    final_vel: float,
    final_rot: float,
    settling_window: int = 3,
    pos_threshold_cm: float = 3.0,
    vel_threshold: float = 1.5,
    rot_threshold: float = 0.05,
) -> Tuple[float, bool]:
    """Computes continuous structural shaping score for closed-loop MPPI reaching.

    Penalizes steady-state position error, terminal joint velocity, and terminal rotation error.
    Grants a discrete success bonus when all spatial, velocity, and orientation criteria are met.
    """
    window_errors = step_errors[-settling_window:] if len(step_errors) >= settling_window else step_errors
    steady_state_error = float(np.mean(window_errors))

    pos_penalty = (steady_state_error / pos_threshold_cm) ** 2
    vel_penalty = (final_vel / vel_threshold) ** 2
    rot_penalty = (final_rot / rot_threshold) ** 2

    raw_score = pos_penalty + vel_penalty + rot_penalty
    is_success = bool(
        all(err < pos_threshold_cm for err in window_errors)
        and (final_vel < vel_threshold)
        and (final_rot < rot_threshold)
    )
    score = raw_score
    return float(score), is_success


class MPPIStructuralScorer:
    """
    Tier 2 Closed-Loop MPPI Structural Validation Scorer.

    Evaluates downstream closed-loop MPPI control on a tiny, fixed validation subset
    (covering Low, Mid, and High vertical elevations) at fixed epoch intervals (e.g. every 10 epochs).

    Zero-Stochasticity Guarantee:
    Explicitly re-seeds the MPPI noise generator per trajectory (`base_seed + traj_id`)
    so the exact same perturbation sequence is evaluated across epochs, preventing
    random noise variance from corrupting the validation signal.
    """

    def __init__(
        self,
        fixed_batches: List[Dict[str, Any]],
        device: torch.device,
        frequency: int = 10,
        base_seed: int = 42000,
        num_samples: int = 256,
        horizon: int = 10,
        num_iterations: int = 3,
        fk: Optional[DifferentiableForwardKinematics] = None,
        approach_window: int = 20,
        config: Optional[MPPIConfig] = None,
    ) -> None:
        if not fixed_batches:
            raise ValueError("fixed_batches cannot be empty for MPPIStructuralScorer.")
        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency}")

        self.fixed_batches = fixed_batches
        self.device = device
        self.frequency = int(frequency)
        self.base_seed = int(base_seed)
        self.num_samples = int(num_samples)
        self.horizon = int(horizon)
        self.num_iterations = int(num_iterations)
        self.fk = fk if fk is not None else DifferentiableForwardKinematics().to(device)
        self.approach_window = int(approach_window)
        self.config = config if config is not None else MPPIConfig(
            num_samples=self.num_samples,
            horizon=self.horizon,
            num_iterations=self.num_iterations,
        )
        self.eval_count: int = 0
        self.phase4_eval_count: int = 0
        self.last_natural_success_rate: float = 0.0

    @classmethod
    def from_dataset(
        cls,
        val_dataset: Any,
        device: torch.device,
        frequency: int = 10,
        num_trajectories: int = 6,
        base_seed: int = 42000,
        num_samples: int = 256,
        horizon: int = 10,
        num_iterations: int = 3,
        fk: Optional[DifferentiableForwardKinematics] = None,
        approach_window: int = 20,
        config: Optional[MPPIConfig] = None,
    ) -> "MPPIStructuralScorer":
        """Builds an MPPIStructuralScorer by extracting a balanced, fixed subset of trajectories covering Low, Mid, and High elevations."""
        target_planes = ("Low", "Mid", "High")
        target_per_plane = max(1, num_trajectories // len(target_planes))
        by_plane: Dict[str, List[Dict[str, Any]]] = {p: [] for p in target_planes}
        other_items: List[Dict[str, Any]] = []

        forward_kin = fk if fk is not None else DifferentiableForwardKinematics().to(device)

        for idx in range(len(val_dataset)):
            item = val_dataset[idx]
            q = item["joint_positions"]
            pf = item["phase_flags"]
            p1 = (pf == 1).nonzero()
            tc = int(p1[0].item()) if len(p1) > 0 else 50

            with torch.no_grad():
                q_tc = q[tc : tc + 1].to(device)
                t_mat = forward_kin.forward_homogeneous(q_tc)[0]
                y_val = float(t_mat[1, 3].item())
            plane = classify_elevation(y_val)

            batch_dict = {
                "events": item["events"].unsqueeze(0).to(device),
                "joint_positions": item["joint_positions"].unsqueeze(0).to(device),
                "phase_flags": item["phase_flags"].unsqueeze(0).to(device),
                "actions": item["actions"].unsqueeze(0).to(device),
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
        for p in target_planes:
            selected.extend(by_plane[p])

        if len(selected) < num_trajectories:
            for item in other_items:
                if len(selected) >= num_trajectories:
                    break
                selected.append(item)

        scorer_cfg = config if config is not None else MPPIConfig(
            num_samples=num_samples,
            horizon=horizon,
            num_iterations=num_iterations,
        )

        return cls(
            fixed_batches=selected,
            device=device,
            frequency=frequency,
            base_seed=base_seed,
            num_samples=num_samples,
            horizon=horizon,
            num_iterations=num_iterations,
            fk=forward_kin,
            approach_window=approach_window,
            config=scorer_cfg,
        )

    def evaluate_trajectory(
        self,
        model: nn.Module,
        controller: LatentMPPIController,
        batch: Dict[str, Any],
        seed: int,
        use_expert_actions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates MPPI closed-loop control (or teacher-forced expert actions) on a single trajectory."""
        events = batch["events"]
        q_gt = batch["joint_positions"][0]
        phase_flags = batch["phase_flags"][0]
        actions = batch["actions"]

        p1_indices = (phase_flags == 1).nonzero()
        t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
        t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
        t_start = max(5, t_c - self.approach_window)

        # Project targets onto the reachable manifold of PhysicalDecoder via grounded latent states
        with torch.no_grad():
            z_start, _, _ = ground_initial_state(model, events, t_start=t_start, actions=actions)
            z_gt_c, _, _ = ground_initial_state(model, events, t_start=t_c, actions=actions)
            z_gt_grasp, _, _ = ground_initial_state(model, events, t_start=t_grasp, actions=actions)

            q_reachable_grasp = model.physical_decoder(z_gt_c)[..., :6]
            target_grasp_mat = self.fk.forward_homogeneous(q_reachable_grasp)[0].detach()

            q_reachable_final = model.physical_decoder(z_gt_grasp)[..., :6]
            target_final_mat = self.fk.forward_homogeneous(q_reachable_final)[0].detach()

        elevation_plane = classify_elevation(float(target_grasp_mat[1, 3].item()))

        dec_start = model.physical_decoder(z_start)
        initial_bias = dec_start[0, :6] / 3.14
        ee_prev = self.fk.forward_homogeneous(dec_start[:, :6])[0, :3, 3].detach()

        # Zero-Stochasticity Guarantee: seed controller explicitly before trajectory unrolling
        controller.reset(initial_action_bias=initial_bias, seed=seed)
        prev_u = actions[0, t_start - 1].clone() if use_expert_actions else initial_bias.clone()

        traj_z: List[torch.Tensor] = []
        traj_u: List[torch.Tensor] = []
        traj_z_next: List[torch.Tensor] = []

        step_errors: List[float] = []
        step_rot_errors_rad: List[float] = []
        step_mppi_costs: List[float] = []
        grasp_error_cm = 0.0
        grasp_rot_err_rad = 0.0
        final_vel = 0.0
        final_rot = 0.0

        current_chunk: Optional[torch.Tensor] = None
        chunk_idx = 0

        curr_z = z_start.clone()

        for t in range(t_start, t_grasp + 1):
            # Target is first the object to grasp (t <= t_c), then the final position and rotation (t > t_c)
            target_pose_mat = target_grasp_mat if t <= t_c else target_final_mat
            target_pos_t = target_pose_mat[:3, 3]

            # Invalidate chunk at transition to immediately replan towards final position/rotation
            if t == t_c + 1:
                current_chunk = None
                chunk_idx = 0

            z_t = curr_z

            u_before = prev_u.clone()
            if use_expert_actions:
                u_exec = actions[0, t].clone()
            else:
                if current_chunk is None or chunk_idx >= controller.chunk_size:
                    res = controller.plan_step(z_t, target_pose_mat, prev_action=prev_u)
                    current_chunk = res.action_chunk
                    chunk_idx = 0

                u_exec = current_chunk[chunk_idx]
                chunk_idx += 1

            prev_u = u_exec
            controller.prev_action = u_exec.clone()

            # Autoregressively advance latent state via dynamics predictor M1
            z_in = z_t if z_t.dim() == 2 else z_t.unsqueeze(0)
            u_in = u_exec if u_exec.dim() == 2 else u_exec.unsqueeze(0)
            with torch.no_grad():
                pred_out = model.predictor(z_in, u_in)
            z_next = pred_out.predicted_latent

            traj_z.append(z_t.detach().cpu())
            traj_u.append(u_exec.detach().cpu())
            traj_z_next.append(z_next.detach().cpu())

            dec_t = model.physical_decoder(z_t)
            T_ee_t = self.fk.forward_homogeneous(dec_t[:, :6])[0].detach()
            ee_t = T_ee_t[:3, 3]
            R_ee_t = T_ee_t[:3, :3]
            pos_err_m = float(torch.norm(ee_t - target_pos_t, p=2).item())
            err_cm = pos_err_m * 100.0
            step_errors.append(err_cm)

            rot_err_rad = compute_rot_error(self.fk, dec_t[0, :6], target_pose_mat)
            step_rot_errors_rad.append(rot_err_rad)

            # Evaluate the Lifting-Aware MPPI Cost at step t (computed only for natural MPPI trajectories to save evaluation time)
            if not use_expert_actions:
                step_cost = float(
                    controller.compute_trajectory_cost(
                        ee_pos=ee_t.view(1, 1, 3),
                        target_coords=target_pos_t,
                        u_seq=u_exec.view(1, 1, -1),
                        u_prev=u_before,
                        R_ee=R_ee_t.view(1, 1, 3, 3),
                        target_orientation=target_pose_mat[:3, :3],
                        p_init=ee_prev,
                    )[0].item()
                )
                step_mppi_costs.append(step_cost)
            ee_prev = ee_t.clone()

            if t == t_c:
                grasp_error_cm = err_cm
                grasp_rot_err_rad = rot_err_rad

            if t == t_grasp:
                final_vel = float(torch.norm(dec_t[0, 6:12]).item())
                final_rot = rot_err_rad

            # Advance to next latent state
            curr_z = z_next

        if len(traj_z) > len(traj_z_next):
            traj_z.pop()
            traj_u.pop()

        packed_transitions = [
            {
                "z": z.squeeze(0),
                "u": u.view(-1),
                "z_next": zn.squeeze(0),
            }
            for z, u, zn in zip(traj_z, traj_u, traj_z_next)
        ]
        packed_trajectory = {
            "z_init": traj_z[0].squeeze(0).detach().cpu(),
            "actions": torch.stack(traj_u).detach().cpu(),
            "targets": torch.stack(traj_z_next).detach().cpu(),
            "length": len(traj_u),
            "trajectory_id": int(batch["trajectory_id"]),
            "is_expert": bool(use_expert_actions),
            "elevation_plane": elevation_plane,
        }

        mppi_cost_score = float(np.mean(step_mppi_costs)) if step_mppi_costs else 0.0
        structural_score_val, is_success = compute_structural_score(step_errors, final_vel, final_rot)
        window_errors = step_errors[-3:] if len(step_errors) >= 3 else step_errors
        steady_state_error = float(np.mean(window_errors))
        mean_cartesian_err_cm = float(np.mean(step_errors))
        mean_rot_err_rad = float(np.mean(step_rot_errors_rad))

        return {
            "score": structural_score_val,
            "mppi_cost": mppi_cost_score,
            "success": 1.0 if is_success else 0.0,
            "min_error_cm": min(step_errors),
            "grasp_error_cm": grasp_error_cm,
            "grasp_rot_err_rad": grasp_rot_err_rad,
            "mean_cartesian_err_cm": mean_cartesian_err_cm,
            "terminal_error_cm": step_errors[-1],
            "steady_state_error_cm": steady_state_error,
            "final_vel": final_vel,
            "final_rot": final_rot,
            "rot_err_rad": mean_rot_err_rad,
            "pos_penalty": (steady_state_error / 3.0) ** 2,
            "vel_penalty": (final_vel / 1.5) ** 2,
            "rot_penalty": (final_rot / 0.05) ** 2,
            "is_expert": bool(use_expert_actions),
            "elevation_plane": elevation_plane,
            "trajectory_id": int(batch["trajectory_id"]),
            "transitions": packed_transitions,
            "trajectory": packed_trajectory,
        }

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        curriculum_phase: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes Tier 2 closed-loop MPPI scoring with Teacher-Forced action blending and strict metric isolation."""
        model.eval()

        controller = LatentMPPIController(
            predictor=model.m1,
            physical_decoder=model.physical_decoder,
            config=self.config,
            forward_kinematics=self.fk,
            critic=model.critic,
            device=self.device,
        )

        total_trajs = len(self.fixed_batches)
        eval_idx = self.eval_count
        self.eval_count += 1

        if curriculum_phase is None:
            num_expert = 0
        elif curriculum_phase < 4:
            num_expert = min(4, max(0, total_trajs - 2)) if total_trajs >= 3 else 0
        else:
            base_expert = min(4, max(0, total_trajs - 2)) if total_trajs >= 3 else 0
            if self.last_natural_success_rate > 0.0:
                self.phase4_eval_count += 1
            num_expert = max(0, base_expert - self.phase4_eval_count)

        num_natural = max(1, total_trajs - num_expert)
        natural_indices = {(eval_idx + i) % total_trajs for i in range(num_natural)}

        natural_results: List[Dict[str, Any]] = []
        all_transitions: List[Dict[str, torch.Tensor]] = []
        all_trajectories: List[Dict[str, Any]] = []
        expert_blocks: List[str] = []
        expert_traj_ids: List[int] = []
        natural_blocks: List[str] = []
        natural_traj_ids: List[int] = []
        all_trials: List[Dict[str, Any]] = []
        for idx, batch in enumerate(self.fixed_batches):
            use_expert = idx not in natural_indices
            traj_id = int(batch["trajectory_id"])
            traj_seed = self.base_seed + traj_id
            res = self.evaluate_trajectory(
                model,
                controller,
                batch,
                seed=traj_seed,
                use_expert_actions=use_expert,
            )
            elev = str(res["elevation_plane"])
            trial_record = {
                "trajectory_id": traj_id,
                "elevation_plane": elev,
                "is_expert": bool(use_expert),
                "score": round(float(res["score"]), 6),
                "mean_cartesian_err_cm": round(float(res["mean_cartesian_err_cm"]), 3),
                "terminal_error_cm": round(float(res["terminal_error_cm"]), 3),
                "final_vel": round(float(res["final_vel"]), 4),
                "final_rot": round(float(res["final_rot"]), 4),
                "grasp_error_cm": round(float(res["grasp_error_cm"]), 3) if res["grasp_error_cm"] is not None else None,
                "grasp_rot_err_rad": round(float(res["grasp_rot_err_rad"]), 4) if res["grasp_rot_err_rad"] is not None else None,
                "success": bool(res["success"] > 0.5),
            }
            all_trials.append(trial_record)

            if use_expert:
                expert_blocks.append(elev)
                expert_traj_ids.append(traj_id)
            else:
                natural_blocks.append(elev)
                natural_traj_ids.append(traj_id)

            if "transitions" in res:
                all_transitions.extend(res["transitions"])
            if "trajectory" in res:
                all_trajectories.append(res["trajectory"])
            if not use_expert:
                natural_results.append(res)

        # Performance breakdown for each trial group / elevation value
        elevation_breakdown: Dict[str, Dict[str, Any]] = {}
        for mode_name, is_exp in [("natural", False), ("expert", True)]:
            for plane in ("Low", "Mid", "High"):
                matching = [t for t in all_trials if t["is_expert"] == is_exp and t["elevation_plane"] == plane]
                if matching:
                    key = f"{mode_name}_{plane.lower()}"
                    elevation_breakdown[key] = {
                        "count": len(matching),
                        "trajectory_ids": [t["trajectory_id"] for t in matching],
                        "avg_vel_err": round(float(np.mean([t["final_vel"] for t in matching])), 4),
                        "avg_rot_err": round(float(np.mean([t["final_rot"] for t in matching])), 4),
                        "avg_cartesian_err_cm": round(float(np.mean([t["mean_cartesian_err_cm"] for t in matching])), 3),
                        "avg_terminal_err_cm": round(float(np.mean([t["terminal_error_cm"] for t in matching])), 3),
                    }

        n = len(natural_results)
        mean_score = float(np.mean([r["score"] for r in natural_results])) if natural_results else 0.0
        success_rate = float(np.mean([r["success"] for r in natural_results])) if natural_results else 0.0
        self.last_natural_success_rate = success_rate
        mean_cart_err = float(np.mean([r["mean_cartesian_err_cm"] for r in natural_results])) if natural_results else 0.0
        mean_grasp_err = float(np.mean([r["grasp_error_cm"] for r in natural_results])) if natural_results else 0.0
        mean_grasp_rot = float(np.mean([r["grasp_rot_err_rad"] for r in natural_results])) if natural_results else 0.0
        mean_term_err = float(np.mean([r["terminal_error_cm"] for r in natural_results])) if natural_results else 0.0
        mean_steady_state_err = float(np.mean([r["steady_state_error_cm"] for r in natural_results])) if natural_results else 0.0
        mean_min_err = float(np.mean([r["min_error_cm"] for r in natural_results])) if natural_results else 0.0
        mean_vel = float(np.mean([r["final_vel"] for r in natural_results])) if natural_results else 0.0
        mean_rot = float(np.mean([r["final_rot"] for r in natural_results])) if natural_results else 0.0
        mean_pos_pen = float(np.mean([r["pos_penalty"] for r in natural_results])) if natural_results else 0.0
        mean_vel_pen = float(np.mean([r["vel_penalty"] for r in natural_results])) if natural_results else 0.0
        mean_rot_pen = float(np.mean([r["rot_penalty"] for r in natural_results])) if natural_results else 0.0

        return {
            "mppi_score": mean_score,
            "mppi_cost": mean_score,
            "mppi_success_rate": success_rate,
            "success_rate": success_rate,
            "mppi_mean_error_cm": mean_cart_err,
            "mean_cartesian_err_cm": mean_cart_err,
            "mppi_grasp_error_cm": mean_grasp_err,
            "grasp_error_cm": mean_grasp_err,
            "mppi_grasp_rot_err_rad": mean_grasp_rot,
            "grasp_rot_err_rad": mean_grasp_rot,
            "mppi_terminal_error_cm": mean_term_err,
            "mppi_steady_state_error_cm": mean_steady_state_err,
            "mppi_min_error_cm": mean_min_err,
            "mppi_final_vel": mean_vel,
            "mppi_final_rot": mean_rot,
            "mppi_rot_err_rad": mean_rot,
            "rotational_err_rad": mean_rot,
            "mppi_pos_penalty": mean_pos_pen,
            "mppi_vel_penalty": mean_vel_pen,
            "mppi_rot_penalty": mean_rot_pen,
            "mppi_evaluated_trajectories": float(n),
            "mppi_expert_trajectories": float(total_trajs - n),
            "mppi_expert_blocks": expert_blocks,
            "mppi_expert_elevation_counts": {
                "Low": expert_blocks.count("Low"),
                "Mid": expert_blocks.count("Mid"),
                "High": expert_blocks.count("High"),
            },
            "mppi_expert_trajectory_ids": expert_traj_ids,
            "mppi_natural_blocks": natural_blocks,
            "mppi_natural_trajectory_ids": natural_traj_ids,
            "mppi_trials": all_trials,
            "mppi_elevation_breakdown": elevation_breakdown,
            "transitions": all_transitions,
            "trajectories": all_trajectories,
        }


__all__ = [
    "LatentMPPIController",
    "MPPIPlanResult",
    "MPPIStructuralScorer",
    "classify_elevation",
    "compute_rot_error",
    "compute_structural_score",
]
