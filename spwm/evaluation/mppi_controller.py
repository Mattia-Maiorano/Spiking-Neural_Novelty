from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from spwm.contracts.config import MPPIConfig
from spwm.contracts.tensors import assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentCritic, LatentPredictor, PhysicalDecoder


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
        self.nominal_update_gamma = float(config.nominal_update_gamma)
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
                # Anisotropic covariance boost: Shoulder_Lift (idx 1), Elbow_Flex (idx 2)
                sigmas[1] = base_s * float(config.shoulder_sigma_multiplier)
                sigmas[2] = base_s * float(config.elbow_sigma_multiplier)

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
            gamma = self.nominal_update_gamma
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

        # 2. Rotational Cost: W_rot * ((1 - cos(theta_error)) + 0.5 * theta_error)
        if target_orientation is not None and R_ee is not None:
            r_rel = torch.matmul(R_ee.transpose(-1, -2), target_orientation.view(1, 1, 3, 3))
            trace_r = torch.diagonal(r_rel, dim1=-2, dim2=-1).sum(-1)  # [K, H]
            cos_theta = torch.clamp((trace_r - 1.0) * 0.5, -1.0, 1.0)
            theta_err = torch.acos(torch.clamp(cos_theta, -0.999999, 0.999999))
            rot_err_step = (1.0 - cos_theta) + 0.5 * theta_err
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
            c_progress = -progress_reward.sum(dim=-1)  # [K]
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
    ) -> None:
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
    def plan_step(
        self,
        z_grounded: torch.Tensor,
        target_pos: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> MPPIPlanResult:
        start_time = time.perf_counter()
        z_init = z_grounded[0] if z_grounded.dim() == 2 else z_grounded

        target_orientation: Optional[torch.Tensor] = None
        if target_pos.shape == (4, 4) or (target_pos.dim() == 3 and target_pos.shape[-2:] == (4, 4)):
            t_mat = target_pos[0] if target_pos.dim() == 3 else target_pos
            target_coords = t_mat[:3, 3].to(device=self.device, dtype=torch.float32)
            target_orientation = t_mat[:3, :3].to(device=self.device, dtype=torch.float32)
        else:
            target_pos_3d = target_pos[0] if target_pos.dim() == 2 else target_pos
            target_coords = target_pos_3d.to(device=self.device, dtype=torch.float32)
            if target_rot is not None:
                t_rot = target_rot[0] if target_rot.dim() == 3 else target_rot
                target_orientation = t_rot.to(device=self.device, dtype=torch.float32)

        u_prev = prev_action.to(self.device) if prev_action is not None else self.prev_action
        K, H, D_u = self.num_samples, self.horizon, self.action_dim
        beta, decay = self.noise_beta, math.sqrt(max(0.0, 1.0 - self.noise_beta ** 2))

        curr_nominal = self.nominal_actions.clone()
        gen = generator if generator is not None else self.generator

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
                rot_err = (1.0 - torch.sum(R_ee[..., :, 2] * t_fwd, dim=-1)) + (
                    1.0 - torch.sum(R_ee[..., :, 1] * t_up, dim=-1)
                )
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
            step_costs = F.softplus(torch.matmul(hidden, self.W2) + self.b2).squeeze(-1)  # [K, H]

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

            curr_nominal = (1.0 - self.nominal_update_gamma) * curr_nominal + self.nominal_update_gamma * torch.sum(
                norm_weights.view(K, 1, 1) * candidates, dim=0
            )
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


__all__ = [
    "LatentMPPIController",
    "MPPIPlanResult",
    "NeuralCostMPPIController",
]

