"""Phase 1 Training Strategy: Action-Conditioned Latent Attractor Dynamics (`train_m1`)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from spwm.contracts.protocols import PhaseStrategy
from spwm.contracts.states import DynamicsState
from spwm.contracts.tensors import assert_phase_space_contract, assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import compute_rollout_schedule, ground_initial_state, rollout_latent, step_grounded
from spwm.training.losses import SPWMLoss
from spwm.training.metrics import position_error, velocity_error
from spwm.training.noise import compute_mppi_noise_alpha, sample_ar1_noise


class M1PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 1 (`train_m1`): Phasic M1 Unrolling with Active Perception Injection."""

    def setup(self) -> None:
        if isinstance(self.loss_fn, nn.Identity):
            self.loss_fn = SPWMLoss.from_config(self.cfg.loss).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        train_params = list(self.model.predictor.parameters()) + list(self.model.critic.parameters())
        for p in train_params:
            p.requires_grad = True
        self.fk = DifferentiableForwardKinematics().to(self.device)
        self.predictor_optimizer = torch.optim.Adam(train_params, lr=self.cfg.training.learning_rate, weight_decay=self.cfg.training.weight_decay)

    def verify_gradient_isolation(self) -> None:
        for name, p in self.model.named_parameters():
            if not any(k in name for k in ("predictor", "m1", "critic")) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m1 mode.")

    @torch.no_grad()
    def _step_grounded(self, event_frame: torch.Tensor, dyn_state: DynamicsState) -> Tuple[torch.Tensor, DynamicsState]:
        return step_grounded(self.model, event_frame, dyn_state)

    def _compute_m1_objectives(
        self, events: torch.Tensor, actions: torch.Tensor, flat_kin: Optional[torch.Tensor] = None,
        phase_flags: Optional[torch.Tensor] = None, epoch: Optional[int] = None,
        is_train: bool = True, closed_loop_batch: Optional[Dict[str, torch.Tensor]] = None,
        curriculum_alpha: Optional[float] = None, curriculum_w_t: float = 0.60, curriculum_phase: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T = events.shape[0], events.shape[1]
        assert_tensor_contract(events, (B, T, self.model.in_channels, self.model.height, self.model.width), "M1.events")
        assert_phase_space_contract(latent_dim=self.model.latent_dim, q_dim=self.model.q_dim, p_dim=self.model.p_dim, action_dim=self.model.action_dim, actions=actions)
        flags = phase_flags.to(device=events.device, dtype=torch.long) if phase_flags is not None else torch.zeros(B, T, dtype=torch.long, device=events.device)
        t_start, horizon = compute_rollout_schedule(flags, sequence_length=T, phase_horizon=self.cfg.training.phase_horizon)

        with torch.no_grad():
            z_star, _ = self.model.forward_representation(events[:, : t_start + 1 + horizon], actions=actions[:, : t_start + 1 + horizon])
            z_target = z_star.detach()

        curr_z, dyn_state, z_prefix = ground_initial_state(self.model, events, t_start, actions=actions)
        rollout_actions, tc = actions[:, t_start : t_start + horizon], self.cfg.training
        noise_alpha = (float(curriculum_alpha) if curriculum_alpha is not None else (compute_mppi_noise_alpha(epoch, tc.mppi_noise_warmup_epochs, tc.mppi_noise_ramp_epochs, tc.mppi_noise_target_alpha) if epoch is not None else 0.0)) if is_train else 0.0
        if is_train and noise_alpha > 0.0:
            rollout_actions = torch.clamp(rollout_actions + noise_alpha * sample_ar1_noise(B, horizon, self.model.action_dim, events.device, beta=self.cfg.mppi.noise_beta, base_sigma=self.cfg.mppi.noise_sigma, shoulder_multiplier=self.cfg.mppi.shoulder_sigma_multiplier, elbow_multiplier=self.cfg.mppi.elbow_sigma_multiplier), -1.0, 1.0)

        roll_res = rollout_latent(
            model=self.model, z_init=curr_z, horizon=horizon, actions=rollout_actions,
            events=events[:, t_start + 1 : t_start + 1 + horizon], phase_flags=flags[:, t_start + 1 : t_start + 1 + horizon],
            initial_dynamics_state=dyn_state, with_grad=True, decode_kinematics=False, fixed_predictor=self.model.m1,
            active_perception_w_max=float(curriculum_w_t) if is_train else 0.60,
        )
        m1_traj, target_latents, q_dim = roll_res.trajectory, z_target[:, t_start + 1 : t_start + 1 + horizon], self.model.q_dim
        flags_rollout = flags[:, t_start + 1 : t_start + 1 + horizon]

        cl_pred, cl_target, cl_size = None, None, 0.0
        if closed_loop_batch is not None:
            if "actions" in closed_loop_batch and closed_loop_batch["actions"].numel() > 0:
                curr_z = (closed_loop_batch["z_init"] if "z_init" in closed_loop_batch else closed_loop_batch["z"]).to(events.device)
                cl_acts = closed_loop_batch["actions"].to(events.device)
                preds = [(curr_z := self.model.m1(curr_z, cl_acts[:, h]).predicted_latent) for h in range(cl_acts.shape[1])]
                cl_pred, cl_target = torch.stack(preds, dim=1), closed_loop_batch["targets"].to(events.device)
            elif "u" in closed_loop_batch and closed_loop_batch["u"].numel() > 0:
                cl_pred = self.model.m1(closed_loop_batch["z"].to(events.device), closed_loop_batch["u"].to(events.device)).predicted_latent.unsqueeze(1)
                cl_target = closed_loop_batch["z_next"].to(events.device).unsqueeze(1)
            if cl_pred is not None and cl_pred.numel() > 0:
                cl_size = float(cl_pred.shape[0])

        jc = self.cfg.environment.joint_count
        target_kin = flat_kin[:, t_start + 1 : t_start + 1 + horizon] if (flat_kin is not None and self.model.physical_decoder is not None) else None
        losses = self.loss_fn.compute_m1_loss(
            m1_traj=m1_traj, target_latents=target_latents, q_dim=q_dim,
            critic=self.model.critic, physical_decoder=self.model.physical_decoder, fk=self.fk,
            flags_rollout=flags_rollout, cl_pred=cl_pred, cl_target=cl_target,
            target_kin=target_kin, joint_count=jc, dt=float(self.cfg.environment.dt), curriculum_phase=curriculum_phase, gamma=0.95,
        )
        pos_err, vel_err = 0.0, 0.0
        if target_kin is not None:
            with torch.no_grad():
                decoded_kin = self.model.physical_decoder(m1_traj, initial_filter_state=self.model.physical_decoder.warm_up_filter(z_prefix))
                pos_err, vel_err = position_error(decoded_kin, target_kin, joint_count=jc), velocity_error(decoded_kin, target_kin, joint_count=jc)

        metrics = {
            "loss": float(losses["loss_m1"].item()), "loss_total": float(losses["total_loss"].item()),
            "loss_m1": float(losses["loss_m1"].item()), "loss_m1_latent": float(losses["loss_m1_latent"].item()),
            "loss_latent": float(losses["loss_m1_latent"].item()), "loss_kin_m1": float(losses["loss_kin_m1"].item()),
            "loss_critic": float(losses["loss_critic"].item()), "loss_position": float(losses["loss_position"].item()),
            "loss_velocity": float(losses["loss_velocity"].item()), "pos_err": pos_err, "vel_err": vel_err,
            "mppi_noise_alpha": float(noise_alpha), "active_perception_w_t": float(curriculum_w_t),
        }
        if closed_loop_batch is not None:
            metrics["dagger_batch_size"] = cl_size
        return losses["total_loss"], metrics

    def _extract_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        events = batch["events"].to(self.device)
        actions = batch["actions"].to(self.device) if batch.get("actions") is not None else torch.zeros(events.shape[0], events.shape[1], self.model.action_dim, device=self.device, dtype=events.dtype)
        return events, actions, (batch["flat_kinematics"].to(self.device) if batch.get("flat_kinematics") is not None else None), (batch["phase_flags"].to(self.device) if batch.get("phase_flags") is not None else None)

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval(); self.model.m1.train(); self.model.critic.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()
        cl_b, c_a = batch.get("closed_loop"), (float(batch["curriculum_alpha"]) if "curriculum_alpha" in batch else None)
        c_wt, c_p = (float(batch["curriculum_w_t"]) if "curriculum_w_t" in batch else 0.60), (int(batch["curriculum_phase"]) if "curriculum_phase" in batch else 1)
        total_loss, m = self._compute_m1_objectives(*self._extract_batch(batch), epoch=epoch, is_train=True, closed_loop_batch=cl_b, curriculum_alpha=c_a, curriculum_w_t=c_wt, curriculum_phase=c_p)
        total_loss.backward()
        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.training.grad_clip_norm)
        self.predictor_optimizer.step()
        return {f"train_{k}": v for k, v in m.items()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        return {f"val_{k}": v for k, v in self._compute_m1_objectives(*self._extract_batch(batch), epoch=None, is_train=False)[1].items()}
