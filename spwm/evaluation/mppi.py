from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from spwm.contracts.tensors import J, assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentPredictor, PhysicalDecoder


@dataclass(frozen=True)
class MPPIPlanResult:
    """Diagnostic outputs from a single MPPI planning step."""

    action: torch.Tensor
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
         across K=256 candidate trajectories over planning horizon H=20.
      3. Decodes kinematics via PhysicalDecoder and computes end-effector 3D position
         via DifferentiableForwardKinematics.
      4. Minimizes Cartesian goal distance, velocity damping, and action smoothness costs.
    """

    def __init__(
        self,
        predictor: LatentPredictor,
        physical_decoder: PhysicalDecoder,
        forward_kinematics: Optional[DifferentiableForwardKinematics] = None,
        num_samples: int = 256,
        horizon: int = 30,
        action_dim: int = J,
        action_min: float = -1.0,
        action_max: float = 1.0,
        noise_sigma: Union[float, Sequence[float], torch.Tensor] = 0.28,
        noise_beta: float = 0.75,
        temperature: float = 0.15,
        num_iterations: int = 3,
        w_pos: float = 1.0,
        w_rot: float = 0.5,
        w_vel: float = 0.08,
        w_act: float = 0.005,
        w_term_pos: float = 5.0,
        w_term_rot: float = 2.0,
        terminal_steps: int = 4,
        w_altitude: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        if terminal_steps <= 0 or terminal_steps > horizon:
            raise ValueError(f"terminal_steps must be in 1..{horizon}, got {terminal_steps}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0.0, got {temperature}")
        if not (0.0 <= noise_beta < 1.0):
            raise ValueError(f"noise_beta must be in [0.0, 1.0), got {noise_beta}")

        #HARDCODED FOR DEBUG

        self.predictor = predictor
        self.physical_decoder = physical_decoder
        self.fk = forward_kinematics if forward_kinematics is not None else DifferentiableForwardKinematics()

        self.num_samples = num_samples
        self.horizon = horizon
        self.terminal_steps = int(terminal_steps)
        self.action_dim = action_dim
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        self.noise_beta = float(noise_beta)
        self.temperature = float(temperature)
        self.num_iterations = num_iterations

        self.w_pos = float(w_pos)
        self.w_rot = float(w_rot)
        self.w_vel = float(w_vel)
        self.w_act = float(w_act)
        self.w_term_pos = float(w_term_pos)
        self.w_term_rot = float(w_term_rot)
        self.w_altitude = float(w_altitude)

        self.pos_gamma = 1.1336
        self.pos_norm_p = 1
        self.vel_radius = 0.2324
        self.vel_gamma = 3.6588
        self.act_gamma = 2.8109

        dev = device if device is not None else next(predictor.parameters()).device
        self.device = dev

        # Covariance Sigma: boost variance on Shoulder_Lift (idx 1) and Elbow_Flex (idx 2)
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

    def reset(self, initial_action_bias: Optional[torch.Tensor] = None) -> None:
        """Resets nominal action sequence and previous action buffer."""
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

        for _ in range(self.num_iterations):
            # 1. Sample temporally correlated action perturbations: AR(1) process
            eta = torch.randn(K, H, D_u, device=self.device) * self.sigma_tensor.view(1, 1, D_u)
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

            # 3. Decode generalized joint positions and velocities
            decoded = self.physical_decoder(z_trajectories)  # [K, H, 2 * joint_count]
            q_hat = decoded[..., : self.physical_decoder.joint_count]  # [K, H, 6]
            p_hat = decoded[..., self.physical_decoder.joint_count :]  # [K, H, 6]

            # 4. Compute end-effector Cartesian positions and poses via Differentiable FK
            if target_orientation is not None:
                T_ee = self.fk.forward_homogeneous(q_hat)  # [K, H, 4, 4]
                ee_pos = T_ee[..., :3, 3]                   # [K, H, 3]
                R_ee = T_ee[..., :3, :3]                    # [K, H, 3, 3]
            else:
                ee_pos = self.fk(q_hat)                     # [K, H, 3]
                R_ee = None

            # 5. Evaluate Trajectory Cost
            term_slice = slice(-self.terminal_steps, None)

            # A. Anisotropic L1 Position Cost with learned gamma exponent
            delta_pos = ee_pos - target_coords.view(1, 1, 3)
            axis_weights = torch.tensor([1.0, 2.5, 1.0], device=self.device)
            raw_pos_err = torch.norm(delta_pos * axis_weights.view(1, 1, 3), p=self.pos_norm_p, dim=-1)
            pos_err_weighted = raw_pos_err ** self.pos_gamma
            
            running_pos_cost = self.w_pos * pos_err_weighted.sum(dim=-1)
            terminal_pos_cost = self.w_term_pos * pos_err_weighted[:, term_slice].mean(dim=-1)

            # B. SE(3) Orientation Cost (unchanged)
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

            # C. Steep-Braking Velocity Penalty with expanded 23.2 cm radius
            vel_norm = torch.norm(p_hat, p=2, dim=-1)
            true_dist = torch.norm(delta_pos, p=2, dim=-1)
            
            dist_scaling = torch.clamp(true_dist / self.vel_radius, min=0.0, max=1.0)
            engagement_curve = (1.0 - dist_scaling) ** self.vel_gamma
            vel_cost = self.w_vel * torch.sum(vel_norm * engagement_curve, dim=-1)

            # D. Super-Quadratic Action Smoothness Penalty (act_gamma = 2.81)
            delta_u_0 = candidates[:, 0] - u_prev.unsqueeze(0)
            delta_u_rest = candidates[:, 1:] - candidates[:, :-1]
            delta_u = torch.cat([delta_u_0.unsqueeze(1), delta_u_rest], dim=1)
            delta_u_norm = torch.norm(delta_u, p=2, dim=-1)
            act_cost = self.w_act * torch.sum(delta_u_norm ** self.act_gamma, dim=-1)

            # E. Information-Theoretic MPPI Control Penalty
            variance = self.sigma_tensor ** 2
            control_penalty = self.temperature * torch.sum(
                (curr_nominal.unsqueeze(0) * noise) / variance.view(1, 1, D_u), dim=(1, 2)
            )

            # Total cost per candidate trajectory
            total_costs = (
                running_pos_cost + 
                terminal_pos_cost + 
                running_rot_cost + 
                terminal_rot_cost + 
                vel_cost + 
                act_cost + 
                control_penalty
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

        # Selected immediate action to execute
        action_to_execute = curr_nominal[0].clone()

        self.nominal_actions = curr_nominal.clone()
        self.prev_action = action_to_execute.clone()


        latency = (time.perf_counter() - start_time) * 1000.0

        return MPPIPlanResult(
            action=action_to_execute,
            nominal_action_sequence=curr_nominal,
            predicted_latent_trajectory=best_z_traj,
            predicted_ee_trajectory=best_ee_traj,
            min_cost=min_cost_val,
            mean_cost=mean_cost_val,
            latency_ms=latency,
            candidate_ee_trajectories=candidate_ee,
            candidate_weights=candidate_w,
        )


__all__ = [
    "LatentMPPIController",
    "MPPIPlanResult",
]
