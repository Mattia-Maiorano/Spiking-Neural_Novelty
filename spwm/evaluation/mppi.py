from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from spwm.contracts.protocols import Tier2EvaluatorProtocol
from spwm.contracts.tensors import J, assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentPredictor, PhysicalDecoder
from spwm.models.rollout import ground_initial_state, step_grounded


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
        seed: Optional[int] = None,
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


def compute_rot_error(
    fk: DifferentiableForwardKinematics,
    q: torch.Tensor,
    target_pose_mat: torch.Tensor,
) -> float:
    """Computes SE(3) rotation error matching the MPPI cost function formulation."""
    t_input = q.unsqueeze(0) if q.dim() == 1 else q
    T_ee = fk.forward_homogeneous(t_input)[0]
    R_ee = T_ee[:3, :3]
    target_rot = target_pose_mat[:3, :3]
    ee_fwd = R_ee[:, 2]
    ee_up = R_ee[:, 1]
    t_fwd = target_rot[:, 2]
    t_up = target_rot[:, 1]
    fwd_err = 1.0 - torch.sum(ee_fwd * t_fwd, dim=-1)
    up_err = 1.0 - torch.sum(ee_up * t_up, dim=-1)
    return float((fwd_err + up_err).item())


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
    success_bonus: float = 2.0,
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
    score = raw_score - (success_bonus if is_success else 0.0)
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
        horizon: int = 30,
        num_iterations: int = 3,
        fk: Optional[DifferentiableForwardKinematics] = None,
        approach_window: int = 20,
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

    @classmethod
    def from_dataset(
        cls,
        val_dataset: Any,
        device: torch.device,
        frequency: int = 10,
        num_trajectories: int = 6,
        base_seed: int = 42000,
        num_samples: int = 256,
        horizon: int = 30,
        num_iterations: int = 3,
        fk: Optional[DifferentiableForwardKinematics] = None,
        approach_window: int = 20,
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
        )

    def evaluate_trajectory(
        self,
        model: nn.Module,
        controller: LatentMPPIController,
        batch: Dict[str, Any],
        seed: int,
    ) -> Dict[str, float]:
        """Evaluates MPPI closed-loop control on a single trajectory with strictly seeded noise generation."""
        events = batch["events"]
        q_gt = batch["joint_positions"][0]
        phase_flags = batch["phase_flags"][0]
        actions = batch["actions"]

        p1_indices = (phase_flags == 1).nonzero()
        t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
        t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
        t_start = max(5, t_c - self.approach_window)

        target_pose_mat = self.fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
        target_pos_t = target_pose_mat[:3, 3]

        z_start, dyn_state, _ = ground_initial_state(model, events, t_start=t_start, actions=actions)
        dec_start = model.physical_decoder(z_start)
        initial_bias = dec_start[0, :6] / 3.14

        # Zero-Stochasticity Guarantee: seed controller explicitly before trajectory unrolling
        controller.reset(initial_action_bias=initial_bias, seed=seed)
        prev_u = initial_bias.clone()

        step_errors: List[float] = []
        final_vel = 0.0
        final_rot = 0.0

        for t in range(t_start, t_grasp + 1):
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
            ee_t = self.fk(dec_t[:, :6])[0].detach()
            err_cm = float((torch.norm(ee_t - target_pos_t) * 100.0).item())
            step_errors.append(err_cm)

            if t == t_grasp:
                final_vel = float(torch.norm(dec_t[0, 6:12]).item())
                final_rot = compute_rot_error(self.fk, dec_t[0, :6], target_pose_mat)

        score, is_success = compute_structural_score(step_errors, final_vel, final_rot)
        window_errors = step_errors[-3:] if len(step_errors) >= 3 else step_errors
        steady_state_error = float(np.mean(window_errors))

        return {
            "score": score,
            "success": 1.0 if is_success else 0.0,
            "min_error_cm": min(step_errors),
            "terminal_error_cm": step_errors[-1],
            "steady_state_error_cm": steady_state_error,
            "final_vel": final_vel,
            "final_rot": final_rot,
            "pos_penalty": (steady_state_error / 3.0) ** 2,
            "vel_penalty": (final_vel / 1.5) ** 2,
            "rot_penalty": (final_rot / 0.05) ** 2,
        }

    @torch.no_grad()
    def __call__(self, model: nn.Module, epoch: int) -> Dict[str, float]:
        """Executes Tier 2 closed-loop MPPI scoring on the fixed validation subset."""
        model.eval()

        controller = LatentMPPIController(
            predictor=model.m1,
            physical_decoder=model.physical_decoder,
            forward_kinematics=self.fk,
            num_samples=self.num_samples,
            horizon=self.horizon,
            num_iterations=self.num_iterations,
            device=self.device,
        )

        all_results: List[Dict[str, float]] = []
        for batch in self.fixed_batches:
            traj_id = int(batch["trajectory_id"])
            traj_seed = self.base_seed + traj_id
            res = self.evaluate_trajectory(model, controller, batch, seed=traj_seed)
            all_results.append(res)

        n = len(all_results)
        mean_score = float(np.mean([r["score"] for r in all_results]))
        success_rate = float(np.mean([r["success"] for r in all_results]))
        mean_min_err = float(np.mean([r["min_error_cm"] for r in all_results]))
        mean_term_err = float(np.mean([r["terminal_error_cm"] for r in all_results]))
        mean_vel = float(np.mean([r["final_vel"] for r in all_results]))
        mean_rot = float(np.mean([r["final_rot"] for r in all_results]))
        mean_pos_pen = float(np.mean([r["pos_penalty"] for r in all_results]))
        mean_vel_pen = float(np.mean([r["vel_penalty"] for r in all_results]))
        mean_rot_pen = float(np.mean([r["rot_penalty"] for r in all_results]))

        return {
            "mppi_score": mean_score,
            "mppi_success_rate": success_rate,
            "mppi_mean_error_cm": mean_min_err,
            "mppi_terminal_error_cm": mean_term_err,
            "mppi_final_vel": mean_vel,
            "mppi_final_rot": mean_rot,
            "mppi_pos_penalty": mean_pos_pen,
            "mppi_vel_penalty": mean_vel_pen,
            "mppi_rot_penalty": mean_rot_pen,
            "mppi_evaluated_trajectories": float(n),
        }


__all__ = [
    "LatentMPPIController",
    "MPPIPlanResult",
    "MPPIStructuralScorer",
    "classify_elevation",
    "compute_rot_error",
    "compute_structural_score",
]
