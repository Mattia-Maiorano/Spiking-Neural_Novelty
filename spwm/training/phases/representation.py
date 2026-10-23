"""Phase 0 Training Strategy: Grounded Representation Pre-training (`train_representation`)."""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.protocols import PhaseStrategy
from spwm.training.metrics import position_error, velocity_error


class RepresentationPhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 0 (`train_representation`): Grounded State Estimator."""

    def setup(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        train_mods = (
            self.model.encoder,
            self.model.dynamics,
            self.model.sensory_predictor,
            self.model.physical_decoder,
        )
        for mod in train_mods:
            for p in mod.parameters():
                p.requires_grad = True

        lr, wd = self.cfg.training.learning_rate, self.cfg.training.weight_decay
        p_lr, p_wd = self.cfg.training.probe_lr, self.cfg.training.probe_weight_decay
        self.predictor_optimizer = torch.optim.Adam([
            {"params": list(self.model.encoder.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.dynamics.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.sensory_predictor.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.physical_decoder.parameters()), "lr": p_lr, "weight_decay": p_wd},
        ])

    def verify_gradient_isolation(self) -> None:
        authorized = ("encoder", "dynamics", "sensory_predictor", "physical_decoder")
        for name, p in self.model.named_parameters():
            if not any(k in name for k in authorized) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in representation mode.")

    def _compute_representation_objectives(
        self, events: torch.Tensor, flat_kin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        latents, error_neurons = self.model.forward_representation(events)
        q_dim = self.model.q_dim
        q = latents[..., :q_dim]
        p = latents[..., q_dim:]

        num_obj = self.model.num_objects
        pos_true = flat_kin[..., : 2 * num_obj]
        vel_true = flat_kin[..., 2 * num_obj : 4 * num_obj]

        # 1. L_kinematic: Supervised loss against ground-truth physics
        pos_pred = self.model.physical_decoder.decode_pos(q)
        vel_pred = self.model.physical_decoder.decode_vel(p)
        l_kinematic = F.mse_loss(pos_pred, pos_true) + F.mse_loss(vel_pred, vel_true)

        # 2. L_derivative: Strict phase-space factorization constraint (raw mapping, no clamp/tanh)
        dt = float(self.cfg.environment.dt)
        delta_q = q[:, 1:] - q[:, :-1]
        p_t = p[:, :-1]
        p_mapped = p_t[..., :q_dim] if p_t.shape[-1] != q_dim else p_t
        l_derivative = F.mse_loss(delta_q, dt * p_mapped)

        # 3. L_smoothness: High-frequency noise suppression on ALIF outputs
        l_smoothness = F.mse_loss(p[:, 1:], p[:, :-1])

        # 4. L_sensory: Existing sensory reconstruction loss
        l_sensory = F.mse_loss(error_neurons, torch.zeros_like(error_neurons))

        lambda_1 = float(self.cfg.loss.lambda_kinematic)
        lambda_2 = float(self.cfg.loss.lambda_derivative)
        lambda_3 = float(self.cfg.loss.lambda_smoothness)
        lambda_4 = float(self.cfg.loss.lambda_sensory)

        total_loss = (
            lambda_1 * l_kinematic
            + lambda_2 * l_derivative
            + lambda_3 * l_smoothness
            + lambda_4 * l_sensory
        )

        decoded_all = torch.cat([pos_pred, vel_pred], dim=-1)
        pos_err = position_error(decoded_all, flat_kin[..., : 4 * num_obj], num_objects=num_obj)
        vel_err = velocity_error(decoded_all, flat_kin[..., : 4 * num_obj], num_objects=num_obj)

        metrics = {
            "loss": float(total_loss.item()),
            "loss_kinematic": float(l_kinematic.item()),
            "loss_derivative": float(l_derivative.item()),
            "loss_smoothness": float(l_smoothness.item()),
            "loss_sensory": float(l_sensory.item()),
            "pos_err": pos_err,
            "vel_err": vel_err,
        }
        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        train_mods = (
            self.model.encoder,
            self.model.dynamics,
            self.model.sensory_predictor,
            self.model.physical_decoder,
        )
        for mod in train_mods:
            mod.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()

        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        total_loss, m = self._compute_representation_objectives(events, flat_kin)
        total_loss.backward()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(trainable, self.cfg.training.grad_clip_norm)
        self.predictor_optimizer.step()
        return {f"train_{k}": v for k, v in m.items()}

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
        _, m = self._compute_representation_objectives(events, flat_kin)
        return {f"val_{k}": v for k, v in m.items()}
