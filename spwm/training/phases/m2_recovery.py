"""Phase 2 Training Strategy: Stateless Manifold Projector / Geometric Denoiser M2 (`train_m2`)."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.protocols import PhaseStrategy
from spwm.training.metrics import position_error, velocity_error


class M2PhaseStrategy(PhaseStrategy):
    """Isolated strategy for Phase 2 (`train_m2`): Stateless Manifold Projector (Geometric Denoiser)."""

    def setup(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.m2.parameters():
            p.requires_grad = True

        lr, wd = self.cfg.training.learning_rate, self.cfg.training.weight_decay
        self.predictor_optimizer = torch.optim.Adam(
            self.model.m2.parameters(), lr=lr, weight_decay=wd
        )

    def verify_gradient_isolation(self) -> None:
        authorized = ("m2",)
        for name, p in self.model.named_parameters():
            if not any(k in name for k in authorized) and p.requires_grad:
                raise AssertionError(f"Gradient leak! Parameter '{name}' has requires_grad=True in train_m2 mode.")

    def _compute_m2_objectives(
        self, events: torch.Tensor, flat_kin: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Obtain pristine latent trajectory segments z^*_{0:T} from frozen representation
        with torch.no_grad():
            z_star, _ = self.model.forward_representation(events)
            z_target = z_star.detach()

        B, T, _ = z_target.shape
        if T < 2:
            raise ValueError(f"Sequence length {T} too short for M2 data generation (requires >= 2)")

        # Candidate step horizons k for autoregressive drift
        k_candidates: List[int] = [k_val for k_val in (5, 15, 25) if k_val < T]
        k = k_candidates[int(torch.randint(0, len(k_candidates), (1,)).item())] if k_candidates else max(1, T - 1)

        # 2. Construct z_noisy: Regime 1 (Autoregressive M1) & Regime 2 (Gaussian noise injection)
        b_half = B // 2
        rollout_target = z_target if b_half == 0 else z_target[:b_half]
        gauss_target = z_target if b_half == 0 else z_target[b_half:]

        with torch.no_grad():
            curr_z = rollout_target[:, 0]
            for _ in range(k):
                curr_z = self.model.m1(curr_z).predicted_latent
            z_noisy_rollout, z_clean_rollout = curr_z, rollout_target[:, k]

            z_clean_gauss = gauss_target[:, k]
            sigma = float(torch.empty(1).uniform_(0.01, 0.05).item())
            z_noisy_gauss = z_clean_gauss + torch.randn_like(z_clean_gauss) * sigma

            z_noisy = torch.cat([z_noisy_rollout, z_noisy_gauss], dim=0)
            z_clean_gt = torch.cat([z_clean_rollout, z_clean_gauss], dim=0)

        # 3. M2 stateless manifold projection: z_clean = M2(z_noisy)
        z_projected = self.model.m2(z_noisy)

        # 4. Decomposed Smooth-L1 loss against instantaneous ground truth (q^*_k, p^*_k)
        q_dim = self.model.q_dim
        q_projected = z_projected[..., :q_dim].contiguous()
        p_projected = z_projected[..., q_dim:].contiguous()
        q_gt = z_clean_gt[..., :q_dim].contiguous()
        p_gt = z_clean_gt[..., q_dim:].contiguous()

        lambda_q = float(self.cfg.loss.lambda_position)
        lambda_p = float(self.cfg.loss.lambda_velocity)
        loss_q = F.smooth_l1_loss(q_projected, q_gt)
        loss_p = F.smooth_l1_loss(p_projected, p_gt)
        total_loss = lambda_q * loss_q + lambda_p * loss_p

        pos_err, vel_err = 0.0, 0.0
        if flat_kin is not None and hasattr(self.model, "physical_decoder"):
            with torch.no_grad():
                decoded_kin = self.model.physical_decoder(z_projected)
                target_kin = flat_kin[:, k] if b_half == 0 else torch.cat([flat_kin[:b_half, k], flat_kin[b_half:, k]], dim=0)
                if b_half == 0:
                    target_kin = torch.cat([target_kin, target_kin], dim=0)
                num_obj = self.cfg.environment.num_objects
                pos_err = position_error(decoded_kin, target_kin, num_objects=num_obj)
                vel_err = velocity_error(decoded_kin, target_kin, num_objects=num_obj)

        metrics = {
            "loss": float(total_loss.item()),
            "loss_m2": float(total_loss.item()),
            "loss_q": float(loss_q.item()),
            "loss_p": float(loss_p.item()),
            "loss_position": float(loss_q.item()),
            "loss_velocity": float(loss_p.item()),
            "k_step": float(k),
            "pos_err": pos_err,
            "vel_err": vel_err,
        }
        return total_loss, metrics

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.model.m2.train()
        assert self.predictor_optimizer is not None
        self.predictor_optimizer.zero_grad()

        events = batch["events"].to(self.device)
        flat_kin = batch.get("flat_kinematics")
        if flat_kin is not None:
            flat_kin = flat_kin.to(self.device)

        total_loss, m = self._compute_m2_objectives(events, flat_kin)
        total_loss.backward()
        trainable = [p for p in self.model.m2.parameters() if p.requires_grad]
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
        _, m = self._compute_m2_objectives(events, flat_kin)
        return {f"val_{k}": v for k, v in m.items()}


Phase2DenoiserStrategy = M2PhaseStrategy

__all__ = ["M2PhaseStrategy", "Phase2DenoiserStrategy"]
