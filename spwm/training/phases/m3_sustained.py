"""
Phase 3 Training Strategy: Joint M2/M3 Sustained Relay (`train_m3`).

Initializes M3 from M1 weights (`strict=True`), freezes M1, and trains M2 and M3
jointly over long-horizon alternating cycles (`t = H -> max_rollout_horizon`).
"""

from __future__ import annotations

from typing import Dict, Tuple
import chronicle
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.protocols import PhaseStrategy
from spwm.models.rollout import rollout_latent
from spwm.training.losses import decompose_latent_loss
from spwm.training.metrics import position_error, velocity_error



class M3PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 3 (`train_m3`): Sustained M2 <-> M3 cyclic rollout."""

    def setup(self) -> None:
        self.model.relay.init_m3_from_m1()
        chronicle.log_info("Initialized M3 weights as an exact copy of M1 (strict=True).")

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.m2.parameters():
            p.requires_grad = True
        for p in self.model.m3.parameters():
            p.requires_grad = True

        pred_params = list(self.model.m2.parameters()) + list(self.model.m3.parameters())
        self.predictor_optimizer = torch.optim.Adam(
            pred_params,
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )

    def verify_gradient_isolation(self) -> None:
        for name, p in self.model.named_parameters():
            if "m2" not in name and "m3" not in name and p.requires_grad:
                raise AssertionError(
                    f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m3 mode."
                )

    def _compute_m3_objectives(
        self, events: torch.Tensor, flat_kin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Slice sequence to only the steps needed for M3 sustained relay: up to max_rollout_horizon + 1 steps
        H_req = max(2 * self.cfg.training.phase_horizon + 1, self.cfg.training.max_rollout_horizon + 1)
        if events.shape[1] > H_req:
            events = events[:, :H_req]
            flat_kin = flat_kin[:, :H_req]

        with torch.no_grad():
            enc_latents = self.model(events).latent_states

        t_total = enc_latents.shape[1]
        H = self.cfg.training.phase_horizon
        max_h = min(self.cfg.training.max_rollout_horizon, t_total - 1)
        if max_h <= H:
            max_h = min(t_total - 1, H + 1)

        with torch.no_grad():
            m1_res = rollout_latent(
                self.model,
                enc_latents[:, 0],
                horizon=H,
                with_grad=False,
                decode_kinematics=False,
                fixed_predictor=self.model.m1,
            )
            m1_traj = m1_res.trajectory
            z_anchor = m1_traj[:, -1].detach()

        rem_horizon = max_h - H
        relay_res = rollout_latent(
            model=self.model,
            z_init=z_anchor,
            horizon=rem_horizon,
            with_grad=torch.is_grad_enabled(),
            decode_kinematics=False,
            start_step=H,
        )
        rollout_traj = relay_res.trajectory

        gt_latent = enc_latents[:, H + 1 : max_h + 1]
        decomp = decompose_latent_loss(
            predicted_latent=rollout_traj,
            target_latent=gt_latent,
            q_dim=self.model.q_dim,
            lambda_position=self.cfg.loss.lambda_position,
            lambda_velocity=self.cfg.loss.lambda_velocity,
        )
        total_loss = decomp["loss_latent"]
        loss_pos_latent = decomp["loss_pos_latent"]
        loss_vel_latent = decomp["loss_vel_latent"]

        # Decode combined M1 (0 -> H) + Relay M2/M3 (H -> max_h) trajectory
        combined_latent = torch.cat([m1_traj.detach(), rollout_traj.detach()], dim=1)
        r0 = self.model.physical_decoder.warm_up_filter(enc_latents[:, :1].detach())
        decoded_combined = self.model.physical_decoder(combined_latent, initial_filter_state=r0)
        target_kin_combined = flat_kin[:, 1 : max_h + 1]
        num_obj = self.cfg.environment.num_objects
        pos_err = position_error(decoded_combined, target_kin_combined, num_objects=num_obj)
        vel_err = velocity_error(decoded_combined, target_kin_combined, num_objects=num_obj)

        metrics = {
            "loss": float(total_loss.item()),
            "loss_encoder": 0.0,
            "loss_decoder": 0.0,
            "loss_sensory": 0.0,
            "loss_position": float(loss_pos_latent.item()),
            "loss_velocity": float(loss_vel_latent.item()),
            "loss_latent": float(total_loss.item()),
            "pos_err": pos_err,
            "vel_err": vel_err,
        }
        return total_loss, metrics


    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.model.m2.train()
        self.model.m3.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()

        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        total_loss, m = self._compute_m3_objectives(events, flat_kin)
        total_loss.backward()
        active_params = list(self.model.m2.parameters()) + list(self.model.m3.parameters())
        nn.utils.clip_grad_norm_(active_params, self.cfg.training.grad_clip_norm)
        self.predictor_optimizer.step()
        return {f"train_{k}": v for k, v in m.items()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        _, m = self._compute_m3_objectives(events, flat_kin)
        return {f"val_{k}": v for k, v in m.items()}
