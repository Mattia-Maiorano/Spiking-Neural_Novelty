"""Phase 0 Training Strategy: Grounded Articulated State Estimation (`train_representation`)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.protocols import PhaseStrategy
from spwm.contracts.tensors import assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.training.metrics import position_error, velocity_error


class RepresentationPhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 0 (`train_representation`): Grounded Articulated Joint State Estimator."""

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
        self.fk = DifferentiableForwardKinematics().to(self.device)

        lr, wd = self.cfg.training.learning_rate, self.cfg.training.weight_decay
        p_lr, p_wd = self.cfg.training.probe_lr, self.cfg.training.probe_weight_decay
        dyn_decay = [p for n, p in self.model.dynamics.named_parameters() if "beta_" in n]
        dyn_other = [p for n, p in self.model.dynamics.named_parameters() if "beta_" not in n]
        self.predictor_optimizer = torch.optim.Adam([
            {"params": list(self.model.encoder.parameters()), "lr": lr, "weight_decay": wd},
            {"params": dyn_other, "lr": lr, "weight_decay": wd},
            {"params": dyn_decay, "lr": lr * 0.5, "weight_decay": 0.0},
            {"params": list(self.model.sensory_predictor.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.physical_decoder.parameters()), "lr": p_lr, "weight_decay": p_wd},
        ])

    def verify_gradient_isolation(self) -> None:
        authorized = ("encoder", "dynamics", "sensory_predictor", "physical_decoder")
        for name, p in self.model.named_parameters():
            if not any(k in name for k in authorized) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in representation mode.")

    def _compute_representation_objectives(
        self, events: torch.Tensor, flat_kin: torch.Tensor, actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        joint_count = self.model.joint_count
        assert_tensor_contract(flat_kin, (None, None, 2 * joint_count), "RepresentationPhaseStrategy.flat_kin")
        latents, error_neurons = self.model.forward_representation(events, actions=actions)
        q_dim = self.model.q_dim
        q, p = latents[..., :q_dim], latents[..., q_dim:]
        joint_pos_true, joint_vel_true = flat_kin[..., :joint_count], flat_kin[..., joint_count : 2 * joint_count]

        # 1. L_kinematic: Supervised loss decoding complex joint angles q*, angular velocities p*, and Cartesian EE pose
        pos_pred = self.model.physical_decoder.decode_pos(q)
        vel_pred = self.model.physical_decoder.decode_vel(p)
        loss_pos_kin = F.mse_loss(pos_pred, joint_pos_true, reduction="none").sum(dim=-1).mean()
        vel_scale = max(1e-3, float(self.cfg.loss.velocity_loss_scale))
        loss_vel_kin = F.mse_loss(vel_pred, joint_vel_true, reduction="none").sum(dim=-1).mean() / (vel_scale ** 2)
        T_pred = self.fk.forward_homogeneous(pos_pred)
        T_true = self.fk.forward_homogeneous(joint_pos_true)
        loss_ee_pos = F.mse_loss(T_pred[..., :3, 3], T_true[..., :3, 3], reduction="none").sum(dim=-1).mean()
        loss_ee_rot = F.mse_loss(T_pred[..., :3, :3], T_true[..., :3, :3], reduction="none").sum(dim=(-2, -1)).mean()
        loss_cartesian = loss_ee_pos + 0.1 * loss_ee_rot
        l_kinematic = loss_pos_kin + loss_vel_kin + float(self.cfg.loss.cartesian_weight) * loss_cartesian

        # 2. L_derivative: Strict phase-space factorization constraint
        dt = float(self.cfg.environment.dt)
        delta_q, p_t = q[:, 1:] - q[:, :-1], p[:, :-1]
        p_mapped = p_t[..., :q_dim] if p_t.shape[-1] != q_dim else p_t
        l_derivative = F.mse_loss(delta_q, dt * p_mapped, reduction="none").sum(dim=-1).mean()

        # 3. L_smoothness & 4. L_sensory
        l_smoothness = F.smooth_l1_loss(p[:, 1:], p[:, :-1], beta=0.1, reduction="none").sum(dim=-1).mean()
        l_sensory = F.mse_loss(error_neurons, torch.zeros_like(error_neurons), reduction="none").sum(dim=-1).mean()

        total_loss = (
            float(self.cfg.loss.lambda_kinematic) * l_kinematic
            + float(self.cfg.loss.lambda_derivative) * l_derivative
            + float(self.cfg.loss.lambda_smoothness) * l_smoothness
            + float(self.cfg.loss.lambda_sensory) * l_sensory
        )
        decoded_all = torch.cat([pos_pred, vel_pred], dim=-1)
        pos_err = position_error(decoded_all, flat_kin[..., : 2 * joint_count], joint_count=joint_count)
        vel_err = velocity_error(decoded_all, flat_kin[..., : 2 * joint_count], joint_count=joint_count)

        metrics = {
            "loss": float(total_loss.item()), "loss_kinematic": float(l_kinematic.item()),
            "loss_position": float(loss_pos_kin.item()), "loss_velocity": float(loss_vel_kin.item()),
            "loss_cartesian": float(loss_cartesian.item()), "loss_derivative": float(l_derivative.item()),
            "loss_smoothness": float(l_smoothness.item()), "loss_sensory": float(l_sensory.item()),
            "pos_err": pos_err, "vel_err": vel_err,
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
        actions = batch["actions"].to(self.device) if "actions" in batch and batch["actions"] is not None else None
        total_loss, m = self._compute_representation_objectives(events, flat_kin, actions=actions)
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
        actions = batch["actions"].to(self.device) if "actions" in batch and batch["actions"] is not None else None
        _, m = self._compute_representation_objectives(events, flat_kin, actions=actions)
        return {f"val_{k}": v for k, v in m.items()}
