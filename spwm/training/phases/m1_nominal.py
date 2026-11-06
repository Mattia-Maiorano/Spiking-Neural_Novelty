"""Phase 1 Training Strategy: Action-Conditioned Latent Attractor Dynamics (`train_m1`)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from spwm.contracts.protocols import PhaseStrategy
from spwm.contracts.states import DynamicsState
from spwm.contracts.tensors import assert_phase_space_contract, assert_tensor_contract
from spwm.models.rollout import compute_rollout_schedule, ground_initial_state, rollout_latent, step_grounded
from spwm.training.metrics import position_error, velocity_error


class M1PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 1 (`train_m1`): Phasic M1 Unrolling with Active Perception Injection."""

    def setup(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.predictor.parameters():
            p.requires_grad = True
        self.predictor_optimizer = torch.optim.Adam(
            self.model.predictor.parameters(), lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay
        )

    def verify_gradient_isolation(self) -> None:
        authorized = ("predictor", "m1")
        for name, p in self.model.named_parameters():
            if not any(k in name for k in authorized) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m1 mode.")

    @torch.no_grad()
    def _step_grounded(self, event_frame: torch.Tensor, dyn_state: DynamicsState) -> Tuple[torch.Tensor, DynamicsState]:
        return step_grounded(self.model, event_frame, dyn_state)

    def _compute_m1_objectives(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        flat_kin: Optional[torch.Tensor] = None,
        phase_flags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T = events.shape[0], events.shape[1]
        assert_tensor_contract(events, (B, T, self.model.in_channels, self.model.height, self.model.width), "M1.events")
        assert_phase_space_contract(
            latent_dim=self.model.latent_dim, q_dim=self.model.q_dim,
            p_dim=self.model.p_dim, action_dim=self.model.action_dim, actions=actions,
        )
        flags = phase_flags.to(device=events.device, dtype=torch.long) if phase_flags is not None else torch.zeros(B, T, dtype=torch.long, device=events.device)
        t_start, horizon = compute_rollout_schedule(flags, sequence_length=T, phase_horizon=self.cfg.training.phase_horizon)

        with torch.no_grad():
            z_star, _ = self.model.forward_representation(events[:, : t_start + 1 + horizon], actions=actions[:, : t_start + 1 + horizon])
            z_target = z_star.detach()

        curr_z, dyn_state, z_prefix = ground_initial_state(self.model, events, t_start, actions=actions)
        roll_res = rollout_latent(
            model=self.model, z_init=curr_z, horizon=horizon,
            actions=actions[:, t_start : t_start + horizon],
            events=events[:, t_start + 1 : t_start + 1 + horizon],
            phase_flags=flags[:, t_start + 1 : t_start + 1 + horizon],
            initial_dynamics_state=dyn_state, with_grad=True, decode_kinematics=False,
            fixed_predictor=self.model.m1,
        )
        m1_traj = roll_res.trajectory
        target_latents = z_target[:, t_start + 1 : t_start + 1 + horizon]

        q_dim = self.model.q_dim
        step_weights = torch.ones_like(m1_traj[..., :1])
        vel_step_weights = torch.ones_like(m1_traj[..., :1])
        flags_rollout = flags[:, t_start + 1 : t_start + 1 + horizon]
        for b in range(B):
            heft_idx = (flags_rollout[b] == 1).nonzero(as_tuple=False)
            if len(heft_idx) > 0:
                h0 = int(heft_idx[0, 0].item())
                h_end = int(heft_idx[-1, 0].item()) if len(heft_idx) > 1 else h0
                step_weights[b, max(0, h0 - 5) : min(horizon, h_end + 10)] = 2.0
                # Passive inertial resistance weighting: emphasize velocity deceleration during and post contact
                vel_step_weights[b, max(0, h0) : min(horizon, h_end + 12)] = 3.0

        pos_diff_sq = (m1_traj[..., :q_dim] - target_latents[..., :q_dim]) ** 2
        vel_diff_sq = (m1_traj[..., q_dim:] - target_latents[..., q_dim:]) ** 2
        loss_pos = (step_weights * pos_diff_sq).mean()
        loss_vel = (vel_step_weights * vel_diff_sq).mean()
        loss_m1 = float(self.cfg.loss.lambda_position) * loss_pos + float(self.cfg.loss.lambda_velocity) * loss_vel
        total_loss = float(self.cfg.loss.lambda_latent) * loss_m1

        pos_err, vel_err = 0.0, 0.0
        if flat_kin is not None and hasattr(self.model, "physical_decoder"):
            with torch.no_grad():
                dec_st = self.model.physical_decoder.warm_up_filter(z_prefix)
                decoded_kin = self.model.physical_decoder(m1_traj, initial_filter_state=dec_st)
                target_kin = flat_kin[:, t_start + 1 : t_start + 1 + horizon]
                jc = self.cfg.environment.joint_count
                pos_err = position_error(decoded_kin, target_kin, joint_count=jc)
                vel_err = velocity_error(decoded_kin, target_kin, joint_count=jc)

        metrics = {
            "loss": float(total_loss.item()), "loss_m1": float(loss_m1.item()),
            "loss_latent": float(loss_m1.item()), "loss_position": float(loss_pos.item()),
            "loss_velocity": float(loss_vel.item()), "pos_err": pos_err, "vel_err": vel_err,
        }
        return total_loss, metrics

    def _extract_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        events = batch["events"].to(self.device)
        actions = batch["actions"].to(self.device) if batch.get("actions") is not None else torch.zeros(events.shape[0], events.shape[1], self.model.action_dim, device=self.device, dtype=events.dtype)
        flat_kin = batch["flat_kinematics"].to(self.device) if batch.get("flat_kinematics") is not None else None
        phase_flags = batch["phase_flags"].to(self.device) if batch.get("phase_flags") is not None else None
        return events, actions, flat_kin, phase_flags

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.model.m1.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()
        total_loss, m = self._compute_m1_objectives(*self._extract_batch(batch))
        total_loss.backward()
        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.training.grad_clip_norm)
        self.predictor_optimizer.step()
        return {f"train_{k}": v for k, v in m.items()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        _, m = self._compute_m1_objectives(*self._extract_batch(batch))
        return {f"val_{k}": v for k, v in m.items()}
