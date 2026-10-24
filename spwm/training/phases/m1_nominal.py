"""Phase 1 Training Strategy: M1 Nominal Physics (`train_m1`)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.protocols import PhaseStrategy
from spwm.training.metrics import position_error, velocity_error


class M1PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 1 (`train_m1`): M1 Nominal Physics on Pristine Latents."""

    def setup(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.predictor.parameters():
            p.requires_grad = True

        lr, wd = self.cfg.training.learning_rate, self.cfg.training.weight_decay
        self.predictor_optimizer = torch.optim.Adam(
            self.model.predictor.parameters(), lr=lr, weight_decay=wd
        )

    def verify_gradient_isolation(self) -> None:
        authorized = ("predictor", "m1")
        for name, p in self.model.named_parameters():
            if not any(k in name for k in authorized) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m1 mode.")

    def _compute_m1_objectives(
        self, events: torch.Tensor, flat_kin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        H_req = self.cfg.training.phase_horizon + 1
        if events.shape[1] > H_req:
            events = events[:, :H_req]
            if flat_kin is not None:
                flat_kin = flat_kin[:, :H_req]

        # Generate ground-truth latent trajectory z^*_{0:T} using frozen Phase 0 modules
        with torch.no_grad():
            z_star, _ = self.model.forward_representation(events)
            z_target = z_star.detach()

        horizon = min(self.cfg.training.phase_horizon, z_target.shape[1] - 1)
        z_start = z_target[:, 0]

        # Run m1 autoregressively: \hat{z}_{t+1} = \hat{z}_t + m1(\hat{z}_t)
        predictions = []
        curr_z = z_start
        for _ in range(horizon):
            pred_out = self.model.m1(curr_z)
            curr_z = pred_out.predicted_latent
            predictions.append(curr_z)

        m1_traj = torch.stack(predictions, dim=1)
        target_latents = z_target[:, 1 : 1 + horizon]

        # Loss Function Constraint: L_m1 = MSE(\hat{z}_{t+1}, z_target_{t+1})
        loss_m1 = F.mse_loss(m1_traj, target_latents)
        total_loss = self.cfg.loss.lambda_latent * loss_m1

        q_dim = self.model.q_dim
        loss_pos_latent = F.mse_loss(m1_traj[..., :q_dim], target_latents[..., :q_dim])
        loss_vel_latent = F.mse_loss(m1_traj[..., q_dim:], target_latents[..., q_dim:])

        pos_err, vel_err = 0.0, 0.0
        if flat_kin is not None and hasattr(self.model, "physical_decoder"):
            with torch.no_grad():
                decoded_kin = self.model.physical_decoder(m1_traj)
                target_kin = flat_kin[:, 1 : 1 + horizon]
                num_obj = self.cfg.environment.num_objects
                pos_err = position_error(decoded_kin, target_kin, num_objects=num_obj)
                vel_err = velocity_error(decoded_kin, target_kin, num_objects=num_obj)

        metrics = {
            "loss": float(total_loss.item()),
            "loss_m1": float(loss_m1.item()),
            "loss_latent": float(loss_m1.item()),
            "loss_position": float(loss_pos_latent.item()),
            "loss_velocity": float(loss_vel_latent.item()),
            "pos_err": pos_err,
            "vel_err": vel_err,
        }
        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.model.m1.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()

        events = batch["events"].to(self.device)
        flat_kin = batch.get("flat_kinematics")
        if flat_kin is not None:
            flat_kin = flat_kin.to(self.device)
        total_loss, m = self._compute_m1_objectives(events, flat_kin)
        total_loss.backward()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(trainable, self.cfg.training.grad_clip_norm)
        self.predictor_optimizer.step()
        return {f"train_{k}": v for k, v in m.items()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch.get("flat_kinematics")
        if flat_kin is not None:
            flat_kin = flat_kin.to(self.device)
        _, m = self._compute_m1_objectives(events, flat_kin)
        return {f"val_{k}": v for k, v in m.items()}
