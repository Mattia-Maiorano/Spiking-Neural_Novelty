from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from spwm.contracts.config import MPPIConfig
from spwm.contracts.states import ClosedLoopTrajectory, Tier2EvalResult
from spwm.evaluation.mppi_controller import LatentMPPIController
from spwm.models.kinematics import (
    DifferentiableForwardKinematics,
    classify_elevation,
    compute_rot_error,
)
from spwm.models.rollout import ground_initial_state


def resolve_effective_vel_threshold(
    vel_threshold: float,
    gt_final_vel: Optional[float] = None,
) -> float:
    """Normalizes terminal velocity threshold against ground-truth speed for fast trajectories."""
    if gt_final_vel is not None and float(gt_final_vel) > vel_threshold:
        return max(vel_threshold, float(gt_final_vel) * 1.1)
    return vel_threshold


def compute_structural_score(
    step_errors: Sequence[float],
    final_vel: float,
    final_rot: float,
    settling_window: int = 3,
    pos_threshold_cm: float = 3.0,
    vel_threshold: float = 1.5,
    rot_threshold: float = 0.05,
    gt_final_vel: Optional[float] = None,
) -> Tuple[float, bool]:
    """Computes continuous structural shaping score for closed-loop MPPI reaching.

    Penalizes steady-state position error, terminal joint velocity, and terminal rotation error.
    Normalizes vel_threshold against gt_final_vel when ground truth exceeds vel_threshold.
    """
    window_errors = step_errors[-settling_window:] if len(step_errors) >= settling_window else step_errors
    steady_state_error = float(np.mean(window_errors))
    effective_vel_threshold = resolve_effective_vel_threshold(vel_threshold, gt_final_vel)

    pos_penalty = (steady_state_error / pos_threshold_cm) ** 2
    vel_penalty = (final_vel / effective_vel_threshold) ** 2
    rot_penalty = (final_rot / rot_threshold) ** 2

    raw_score = pos_penalty + vel_penalty + rot_penalty
    is_success = bool(
        all(err < pos_threshold_cm for err in window_errors)
        and (final_vel < effective_vel_threshold)
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
        self.config = (
            config
            if config is not None
            else MPPIConfig(
                num_samples=self.num_samples,
                horizon=self.horizon,
                num_iterations=self.num_iterations,
            )
        )
        self.eval_count: int = 0
        self.phase4_eval_count: int = 0
        self.last_natural_success_rate: float = 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Serializes Tier 2 curriculum scheduling counters for checkpoint resume."""
        return {
            "eval_count": int(self.eval_count),
            "phase4_eval_count": int(self.phase4_eval_count),
            "last_natural_success_rate": float(self.last_natural_success_rate),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restores Tier 2 curriculum scheduling counters from checkpoint."""
        self.eval_count = int(state["eval_count"])
        self.phase4_eval_count = int(state["phase4_eval_count"])
        self.last_natural_success_rate = float(state["last_natural_success_rate"])

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
            if "joint_velocities" in item:
                batch_dict["joint_velocities"] = item["joint_velocities"].unsqueeze(0).to(device)

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

        scorer_cfg = (
            config
            if config is not None
            else MPPIConfig(
                num_samples=num_samples,
                horizon=horizon,
                num_iterations=num_iterations,
            )
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
        max_t = min(t_grasp + 1, events.shape[1] - 1)

        # Project targets onto the reachable manifold of PhysicalDecoder via grounded latent states
        with torch.no_grad():
            _, _, z_gt_seq = ground_initial_state(model, events, t_start=max_t, actions=actions)
            z_start = z_gt_seq[:, t_start]
            z_gt_c = z_gt_seq[:, t_c]
            z_gt_grasp = z_gt_seq[:, t_grasp]

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

            z_t = z_gt_seq[:, t] if use_expert_actions else curr_z

            u_before = prev_u.clone()
            if use_expert_actions:
                u_exec = actions[0, t].clone()
            else:
                if current_chunk is None or chunk_idx >= controller.chunk_size:
                    target_pose_plan = torch.stack(
                        [
                            target_grasp_mat if (t + tau) <= t_c else target_final_mat
                            for tau in range(controller.horizon)
                        ],
                        dim=0,
                    )
                    res = controller.plan_step(z_t, target_pose_plan, prev_action=prev_u)
                    current_chunk = res.action_chunk
                    chunk_idx = 0

                u_exec = current_chunk[chunk_idx]
                chunk_idx += 1

            prev_u = u_exec
            controller.prev_action = u_exec.clone()

            # Advance latent state and compute grounded target for DAgger replay
            z_in = z_t if z_t.dim() == 2 else z_t.unsqueeze(0)
            u_in = u_exec if u_exec.dim() == 2 else u_exec.unsqueeze(0)
            z_gt_next = z_gt_seq[:, min(t + 1, max_t)]
            with torch.no_grad():
                if use_expert_actions:
                    z_next = z_gt_next
                    z_eval = z_gt_seq[:, t]
                    z_dagger_target = z_gt_next
                else:
                    pred_out = model.predictor(z_in, u_in)
                    z_pred = pred_out.predicted_latent
                    if int(phase_flags[t].item()) == 1:
                        z_next = 0.40 * z_pred + 0.60 * z_gt_next
                    else:
                        z_next = z_pred
                    z_eval = z_next
                    z_dagger_target = z_gt_next

            traj_z.append(z_t.detach().cpu())
            traj_u.append(u_exec.detach().cpu())
            traj_z_next.append(z_dagger_target.detach().cpu())

            dec_t = model.physical_decoder(z_eval)
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
        packed_trajectory = ClosedLoopTrajectory(
            z_init=traj_z[0].squeeze(0).detach().cpu(),
            actions=torch.stack(traj_u).detach().cpu(),
            targets=torch.stack(traj_z_next).detach().cpu(),
            length=len(traj_u),
            trajectory_id=int(batch["trajectory_id"]),
            is_expert=bool(use_expert_actions),
            elevation_plane=elevation_plane,
        )

        mppi_cost_score = float(np.mean(step_mppi_costs)) if step_mppi_costs else 0.0
        gt_final_vel = (
            float(torch.norm(batch["joint_velocities"][0, t_grasp]).item())
            if "joint_velocities" in batch
            else None
        )
        effective_vel_threshold = resolve_effective_vel_threshold(1.5, gt_final_vel)
        structural_score_val, is_success = compute_structural_score(
            step_errors,
            final_vel,
            final_rot,
            gt_final_vel=gt_final_vel,
        )
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
            "vel_penalty": (final_vel / effective_vel_threshold) ** 2,
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
    ) -> Tier2EvalResult:
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

        num_natural_buffer = max(1, total_trajs - num_expert)
        natural_buffer_indices = {(eval_idx + i) % total_trajs for i in range(num_natural_buffer)}

        natural_results: List[Dict[str, Any]] = []
        all_transitions: List[Dict[str, torch.Tensor]] = []
        all_trajectories: List[ClosedLoopTrajectory] = []
        expert_blocks: List[str] = []
        expert_traj_ids: List[int] = []
        natural_blocks: List[str] = []
        natural_traj_ids: List[int] = []
        all_trials: List[Dict[str, Any]] = []
        for idx, batch in enumerate(self.fixed_batches):
            traj_id = int(batch["trajectory_id"])
            traj_seed = self.base_seed + traj_id

            # Always evaluate every trajectory in natural MPPI mode across all curriculum phases
            nat_res = self.evaluate_trajectory(
                model,
                controller,
                batch,
                seed=traj_seed,
                use_expert_actions=False,
            )
            nat_elev = str(nat_res["elevation_plane"])
            all_trials.append({
                "trajectory_id": traj_id,
                "elevation_plane": nat_elev,
                "is_expert": False,
                "score": round(float(nat_res["score"]), 6),
                "mean_cartesian_err_cm": round(float(nat_res["mean_cartesian_err_cm"]), 3),
                "terminal_error_cm": round(float(nat_res["terminal_error_cm"]), 3),
                "final_vel": round(float(nat_res["final_vel"]), 4),
                "final_rot": round(float(nat_res["final_rot"]), 4),
                "grasp_error_cm": round(float(nat_res["grasp_error_cm"]), 3) if nat_res["grasp_error_cm"] is not None else None,
                "grasp_rot_err_rad": round(float(nat_res["grasp_rot_err_rad"]), 4) if nat_res["grasp_rot_err_rad"] is not None else None,
                "success": bool(nat_res["success"] > 0.5),
            })
            natural_blocks.append(nat_elev)
            natural_traj_ids.append(traj_id)
            natural_results.append(nat_res)

            # ClosedLoopBuffer continues partial expert overwriting schedule (4 -> 3 -> 2 -> 1 -> 0 in Phase 4)
            if idx in natural_buffer_indices:
                buffer_res = nat_res
            else:
                buffer_res = self.evaluate_trajectory(
                    model,
                    controller,
                    batch,
                    seed=traj_seed,
                    use_expert_actions=True,
                )
                exp_elev = str(buffer_res["elevation_plane"])
                all_trials.append({
                    "trajectory_id": traj_id,
                    "elevation_plane": exp_elev,
                    "is_expert": True,
                    "score": round(float(buffer_res["score"]), 6),
                    "mean_cartesian_err_cm": round(float(buffer_res["mean_cartesian_err_cm"]), 3),
                    "terminal_error_cm": round(float(buffer_res["terminal_error_cm"]), 3),
                    "final_vel": round(float(buffer_res["final_vel"]), 4),
                    "final_rot": round(float(buffer_res["final_rot"]), 4),
                    "grasp_error_cm": round(float(buffer_res["grasp_error_cm"]), 3) if buffer_res["grasp_error_cm"] is not None else None,
                    "grasp_rot_err_rad": round(float(buffer_res["grasp_rot_err_rad"]), 4) if buffer_res["grasp_rot_err_rad"] is not None else None,
                    "success": bool(buffer_res["success"] > 0.5),
                })
                expert_blocks.append(exp_elev)
                expert_traj_ids.append(traj_id)

            if "transitions" in buffer_res:
                all_transitions.extend(buffer_res["transitions"])
            if "trajectory" in buffer_res:
                raw_traj = buffer_res["trajectory"]
                if isinstance(raw_traj, ClosedLoopTrajectory):
                    all_trajectories.append(raw_traj)
                else:
                    all_trajectories.append(ClosedLoopTrajectory.from_dict(raw_traj))

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
        canonical_mppi_score = (
            0.5 * mean_grasp_err
            + 0.5 * mean_steady_state_err
            + 0.25 * mean_cart_err
            + 0.1 * mean_vel
            + 2.0 * mean_rot
        ) if natural_results else 0.0

        return Tier2EvalResult(
            mppi_score=canonical_mppi_score,
            mppi_structural_score=mean_score,
            mppi_cost=mean_score,
            mppi_success_rate=success_rate,
            mppi_mean_error_cm=mean_cart_err,
            mppi_grasp_error_cm=mean_grasp_err,
            mppi_grasp_rot_err_rad=mean_grasp_rot,
            mppi_terminal_error_cm=mean_term_err,
            mppi_steady_state_error_cm=mean_steady_state_err,
            mppi_min_error_cm=mean_min_err,
            mppi_final_vel=mean_vel,
            mppi_final_rot=mean_rot,
            mppi_pos_penalty=mean_pos_pen,
            mppi_vel_penalty=mean_vel_pen,
            mppi_rot_penalty=mean_rot_pen,
            mppi_evaluated_trajectories=float(n),
            mppi_expert_trajectories=float(len(expert_traj_ids)),
            mppi_expert_blocks=expert_blocks,
            mppi_expert_elevation_counts={
                "Low": expert_blocks.count("Low"),
                "Mid": expert_blocks.count("Mid"),
                "High": expert_blocks.count("High"),
            },
            mppi_expert_trajectory_ids=expert_traj_ids,
            mppi_natural_blocks=natural_blocks,
            mppi_natural_trajectory_ids=natural_traj_ids,
            mppi_trials=all_trials,
            mppi_elevation_breakdown=elevation_breakdown,
            transitions=all_transitions,
            trajectories=all_trajectories,
        )


__all__ = [
    "MPPIStructuralScorer",
    "compute_structural_score",
    "resolve_effective_vel_threshold",
]

