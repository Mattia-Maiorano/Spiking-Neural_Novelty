"""Phase 1 Training Strategy: M1 Nominal Physics & Representation (`train_m1`)."""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.protocols import PhaseStrategy
from spwm.models.rollout import rollout_latent
from spwm.training.losses import decompose_latent_loss
from spwm.training.metrics import position_error, velocity_error


class M1PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 1 (`train_m1`): Representation + M1 nominal physics."""

    def setup(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        train_mods = (

            self.model.m1, self.model.encoder, self.model.dynamics,
            self.model.sensory_predictor, self.model.physical_decoder,
        )
        for mod in train_mods:
            for p in mod.parameters():
                p.requires_grad = True

        lr, wd = self.cfg.training.learning_rate, self.cfg.training.weight_decay
        p_lr, p_wd = self.cfg.training.probe_lr, self.cfg.training.probe_weight_decay
        self.predictor_optimizer = torch.optim.Adam([
            {"params": list(self.model.m1.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.encoder.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.dynamics.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.sensory_predictor.parameters()), "lr": lr, "weight_decay": wd},
            {"params": list(self.model.physical_decoder.parameters()), "lr": p_lr, "weight_decay": p_wd},
        ])

    def verify_gradient_isolation(self) -> None:
        authorized = ("m1", "encoder", "dynamics", "sensory_predictor", "physical_decoder")
        for name, p in self.model.named_parameters():
            if not any(k in name for k in authorized) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m1 mode.")

    def _compute_m1_objectives(
        self, events: torch.Tensor, flat_kin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        H_req = self.cfg.training.phase_horizon + 1
        if events.shape[1] > H_req:
            events, flat_kin = events[:, :H_req], flat_kin[:, :H_req]

        seq_out = self.model(events)
        enc_latents, keypoints = seq_out.latent_states, seq_out.keypoints
        if keypoints is None or seq_out.decoded_kinematics is None:
            raise RuntimeError("SPWM forward pass must produce keypoints and decoded_kinematics.")

        horizon = min(self.cfg.training.phase_horizon, enc_latents.shape[1] - 1)
        z_start = enc_latents[:, 0].detach()

        decoder_state = self.model.physical_decoder.warm_up_filter(enc_latents[:, :1].detach())

        rollout_res = rollout_latent(
            model=self.model,
            z_init=z_start,
            horizon=horizon,
            with_grad=torch.is_grad_enabled(),
            decode_kinematics=True,
            initial_decoder_state=decoder_state,
            fixed_predictor=self.model.m1,
        )
        m1_traj = rollout_res.trajectory
        decoded_kin = rollout_res.decoded_kinematics
        target_latents = enc_latents[:, 1 : 1 + horizon].detach()
        target_kin = flat_kin[:, 1 : 1 + horizon]

        num_obj = self.cfg.environment.num_objects
        k_num = keypoints.shape[-1] // 2
        kp_reshaped = keypoints.view(*keypoints.shape[:-1], k_num, 2)
        if num_obj > 1:
            all_obj_xy = flat_kin[..., : 2 * num_obj].view(*flat_kin.shape[:-1], num_obj, 2)
            dists = torch.sum((kp_reshaped.unsqueeze(-2) - all_obj_xy.unsqueeze(-3)) ** 2, dim=-1)
            loss_encoder = dists.min(dim=-1).values.mean()
        else:
            gt_xy = flat_kin[..., :2]
            loss_encoder = F.mse_loss(kp_reshaped, gt_xy.unsqueeze(-2).expand_as(kp_reshaped))

        immediate_decoded = seq_out.decoded_kinematics
        loss_decoder = F.mse_loss(immediate_decoded, flat_kin[..., : immediate_decoded.shape[-1]])

        decomp = decompose_latent_loss(
            predicted_latent=m1_traj, target_latent=target_latents, q_dim=self.model.q_dim,
            lambda_position=self.cfg.loss.lambda_position, lambda_velocity=self.cfg.loss.lambda_velocity,
        )
        latent_loss = decomp["loss_latent"]
        loss_pos_latent, loss_vel_latent = decomp["loss_pos_latent"], decomp["loss_vel_latent"]
        loss_sensory = F.mse_loss(seq_out.error_neurons, torch.zeros_like(seq_out.error_neurons))

        pos_err, vel_err = 0.0, 0.0
        if decoded_kin is not None:
            pos_err = position_error(decoded_kin, target_kin, num_objects=num_obj)
            vel_err = velocity_error(decoded_kin, target_kin, num_objects=num_obj)

        total_loss = (
            self.cfg.loss.lambda_latent * latent_loss
            + self.cfg.loss.lambda_encoder * loss_encoder
            + self.cfg.loss.lambda_decoder * loss_decoder
            + self.cfg.loss.lambda_sensory * loss_sensory
        )
        metrics = {
            "loss": float(total_loss.item()),
            "loss_encoder": float(loss_encoder.item()),
            "loss_decoder": float(loss_decoder.item()),
            "loss_sensory": float(loss_sensory.item()),
            "loss_position": float(loss_pos_latent.item()),
            "loss_velocity": float(loss_vel_latent.item()),
            "loss_latent": float(latent_loss.item()),
            "pos_err": pos_err,
            "vel_err": vel_err,
        }
        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        train_mods = (
            self.model.m1, self.model.encoder, self.model.dynamics,
            self.model.sensory_predictor, self.model.physical_decoder,
        )
        for mod in train_mods:
            mod.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()


        events = batch["events"].to(self.device)
        flat_kin = batch["flat_kinematics"].to(self.device)
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
        flat_kin = batch["flat_kinematics"].to(self.device)
        _, m = self._compute_m1_objectives(events, flat_kin)
        return {f"val_{k}": v for k, v in m.items()}
