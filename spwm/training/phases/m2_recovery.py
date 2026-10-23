"""
Phase 2 Training Strategy: M2 Recovery Predictor with Convergence Funnel (`train_m2`).

Trains M2 exclusively on a single recovery window (t = H -> 2H) starting from the dirty M1
anchor state after frozen M1 computes 0 -> H, guided by the interpolated Convergence Funnel target.
All other modules are strictly frozen.
"""

from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from spwm.contracts.protocols import PhaseStrategy
from spwm.models.rollout import rollout_latent
from spwm.training.metrics import position_error, velocity_error


class M2PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 2 (`train_m2`): Convergence Funnel recovery (H -> 2H)."""

    def setup(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.m2.parameters():
            p.requires_grad = True

        self.predictor_optimizer = torch.optim.Adam(
            self.model.m2.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

    def verify_gradient_isolation(self) -> None:
        for name, p in self.model.named_parameters():
            if "m2" not in name and p.requires_grad:
                raise AssertionError(
                    f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m2 mode."
                )

    def _compute_m2_objectives(
        self, events: torch.Tensor, flat_kin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Slice sequence to only the steps needed for M2 recovery: M1 computes 0->H and M2 computes H->2H (requires 2*H + 1 steps)
        H_req = 2 * self.cfg.training.phase_horizon + 1
        if events.shape[1] > H_req:
            events = events[:, :H_req]
            flat_kin = flat_kin[:, :H_req]

        with torch.no_grad():
            enc_latents = self.model(events).latent_states

        t_total = enc_latents.shape[1]
        H = self.cfg.training.phase_horizon
        phase1_h = min(H, max(0, t_total - 1 - H))
        if phase1_h <= 0:
            raise ValueError(f"Sequence length {t_total} too short for M2 phase_horizon {H}")

        with torch.no_grad():
            m1_res = rollout_latent(
                self.model, enc_latents[:, 0], horizon=H, with_grad=False,
                decode_kinematics=False, fixed_predictor=self.model.m1,
            )
            m1_traj = m1_res.trajectory
            z_anchor1 = m1_traj[:, -1].detach()
            z_prev1 = m1_traj[:, -2].detach() if H >= 2 else enc_latents[:, 0].detach()

        m2_res1 = rollout_latent(
            self.model, z_anchor1, horizon=phase1_h, with_grad=torch.is_grad_enabled(),
            decode_kinematics=False, fixed_predictor=self.model.m2,
        )
        m2_traj_1 = m2_res1.trajectory
        gt_latent_1 = enc_latents[:, H + 1 : H + 1 + phase1_h]
        f1 = self.loss_fn.compute_convergence_funnel_loss(
            predicted_latent=m2_traj_1,
            z_anchor=z_anchor1,
            z_prev=z_prev1,
            ground_truth_latent=gt_latent_1,
            t_anchor=H,
            phase_horizon=H,
            q_dim=self.model.q_dim,
            lambda_position=self.cfg.loss.lambda_position,
            lambda_velocity=self.cfg.loss.lambda_velocity,
        )
        total_loss = f1["total_loss"]
        loss_pos_latent = f1["loss_pos_latent"]
        loss_vel_latent = f1["loss_vel_latent"]

        # Decode combined M1 (0 -> H) + M2 (H -> 2H) trajectory for unified kinematic metrics
        num_obj = self.cfg.environment.num_objects
        combined_latent = torch.cat([m1_traj.detach(), m2_traj_1.detach()], dim=1)
        r0 = self.model.physical_decoder.warm_up_filter(enc_latents[:, :1].detach())
        decoded_combined = self.model.physical_decoder(combined_latent, initial_filter_state=r0)
        target_kin_combined = flat_kin[:, 1 : 1 + H + phase1_h]

        metrics = {
            "loss": float(total_loss.item()),
            "loss_encoder": 0.0,
            "loss_decoder": 0.0,
            "loss_sensory": 0.0,
            "loss_position": float(loss_pos_latent.item()),
            "loss_velocity": float(loss_vel_latent.item()),
            "loss_latent": float(total_loss.item()),
            "pos_err": position_error(decoded_combined, target_kin_combined, num_objects=num_obj),
            "vel_err": velocity_error(decoded_combined, target_kin_combined, num_objects=num_obj),
        }
        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.model.m2.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()

        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        total_loss, m = self._compute_m2_objectives(events, flat_kin)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.m2.parameters(), self.cfg.training.grad_clip_norm)
        self.predictor_optimizer.step()
        return {f"train_{k}": v for k, v in m.items()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        _, m = self._compute_m2_objectives(events, flat_kin)
        return {f"val_{k}": v for k, v in m.items()}
