"""Phase 0 Training Strategy: Grounded Articulated State Estimation (`train_representation`)."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from spwm.contracts.protocols import PhaseStrategy
from spwm.contracts.tensors import assert_tensor_contract
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.training.losses import SPWMLoss
from spwm.training.metrics import position_error, velocity_error


class RepresentationPhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 0 (`train_representation`): Grounded Articulated Joint State Estimator."""

    def setup(self) -> None:
        if isinstance(self.loss_fn, nn.Identity):
            self.loss_fn = SPWMLoss.from_config(self.cfg.loss).to(self.device)
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
        pos_pred = self.model.physical_decoder.decode_pos(q)
        vel_pred = self.model.physical_decoder.decode_vel(p)

        losses = self.loss_fn.compute_representation_loss(
            q=q, p=p, pos_pred=pos_pred, vel_pred=vel_pred,
            error_neurons=error_neurons, flat_kin=flat_kin, fk=self.fk,
            joint_count=joint_count, dt=float(self.cfg.environment.dt),
        )
        total_loss = losses["total_loss"]
        decoded_all = torch.cat([pos_pred, vel_pred], dim=-1)
        pos_err = position_error(decoded_all, flat_kin[..., : 2 * joint_count], joint_count=joint_count)
        vel_err = velocity_error(decoded_all, flat_kin[..., : 2 * joint_count], joint_count=joint_count)

        metrics = {
            "loss": float(total_loss.item()),
            "loss_kinematic": float(losses["loss_kinematic"].item()),
            "loss_position": float(losses["loss_position"].item()),
            "loss_velocity": float(losses["loss_velocity"].item()),
            "loss_cartesian": float(losses["loss_cartesian"].item()),
            "loss_derivative": float(losses["loss_derivative"].item()),
            "loss_smoothness": float(losses["loss_smoothness"].item()),
            "loss_sensory": float(losses["loss_sensory"].item()),
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
