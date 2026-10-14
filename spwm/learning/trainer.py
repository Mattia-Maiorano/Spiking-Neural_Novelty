"""
Forward-Only Continuous Online Trainer for SPWM.
Maintains O(1) memory footprint scaling across long sequence horizons.

v10 Changes:
  - Integrated Neuromodulatory Gain Field Corrector support.
  - Added corrector_burn_in_steps to delay the Gain Mask application during rollouts.
  - Removed Isometry and Quiescence losses (Gain Mask natively prevents null-space explosion).
"""

from __future__ import annotations
from torch.distributed import distributed_c10d
import collections
import contextlib
import csv
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Union, Sequence

import chronicle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from spwm.learning.losses import LossOutput, SPWMLoss, _frozen
from spwm.learning.metrics import (
    position_error,
    velocity_error,
)


@contextlib.contextmanager
def freeze_parameters(module: nn.Module):
    prev_states = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad = False
        yield
    finally:
        for p, state in zip(module.parameters(), prev_states):
            p.requires_grad = state


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[SPWMLoss] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        learning_algorithm: str = "online_eprop",
        device: Optional[Union[str, torch.device]] = None,
        save_dir: str = "results/default_run",
        probe_lr: float = 5e-4,
        probe_weight_decay: float = 1e-2,
        tensorboard_logging: bool = False,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        best_val_pos_err: float = float("inf"),
        best_val_vel_err: float = float("inf"),          
        best_combined_score: float = float("inf"),
        best_rollout_mae: float = float("inf"),
        best_drift_ratio: float = float("inf"),
        best_gate_passed: bool = False,
        best_epoch: Optional[int] = None,   
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict] = None,
        curriculum_multi_step: bool = False,
        smooth_horizon_sampling: bool = False,
        k_max: int = 50,
        corrector_lr: Optional[float] = None,
        corrector_horizon: int = 100,
        corrector_burn_in_steps: int = 15,
        lambda_corrector_asymptotic: float = 1.0,
        perturb_corrector_states: bool = False,
        corrector_perturbation_std: float = 0.00,
        curriculum_thresholds: Optional[Dict[str, Any]] = None,
        encoder_warmup_epochs: int = 60,
        max_drift_ratio: float = 5.0,
        rollout_horizons: Optional[Sequence[int]] = None,
        num_objects: int = 1,
        mode: str = "train_predictor",
    ) -> None:
        self.mode = mode.lower()
        if self.mode not in ("train_predictor", "train_corrector", "joint"):
            raise ValueError(f"Unknown training mode '{mode}'.")

        self.best_epoch: Optional[int] = best_epoch
        self.best_val_pos_err: float = best_val_pos_err
        self.best_val_vel_err: float = best_val_vel_err
        self.best_combined_score: float = best_combined_score
        self.best_rollout_mae: float = best_rollout_mae
        self.best_drift_ratio: float = best_drift_ratio
        self.best_gate_passed: bool = best_gate_passed
        self.max_drift_ratio: float = float(max_drift_ratio)
        self.rollout_horizons: List[int] = list(rollout_horizons) if rollout_horizons is not None else [1, 5, 10, 25, 50, 100]
        self.num_objects: int = num_objects

        self.learning_algorithm = learning_algorithm.lower()
        self.learning_rate = learning_rate
        self.curriculum_multi_step = curriculum_multi_step
        self.smooth_horizon_sampling = smooth_horizon_sampling
        self.k_max = k_max
        self.corrector_horizon = corrector_horizon
        self.corrector_burn_in_steps = corrector_burn_in_steps
        self.lambda_corrector_asymptotic = lambda_corrector_asymptotic
        self.perturb_corrector_states = perturb_corrector_states
        self.corrector_perturbation_std = corrector_perturbation_std
        self.encoder_warmup_epochs = encoder_warmup_epochs if self.mode != "train_corrector" else 0
        self.curriculum_thresholds = curriculum_thresholds or {
            "phase_1_horizon": 3, "phase_2_horizon": 6, "phase_2_threshold": 0.20,
            "phase_3_horizon": 10, "phase_3_threshold": 0.12,
        }

        if device is None:
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.recent_grad_norms = collections.deque(maxlen=10)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = (loss_fn or SPWMLoss()).to(self.device)
        self.grad_clip_norm = grad_clip_norm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        from spwm.evaluation.rollout import RolloutEvaluator
        self.rollout_evaluator = RolloutEvaluator(
            model=self.model, device=self.device, horizons=self.rollout_horizons, num_objects=self.num_objects,
        )

        # ------------------------------------------------------------------
        # Optimizers Setup
        # ------------------------------------------------------------------
        if self.mode == "train_predictor":
            if hasattr(self.model, "corrector") and self.model.corrector is not None:
                for p in self.model.corrector.parameters():
                    p.requires_grad = False

            encoder_params = list(self.model.encoder.parameters()) if hasattr(self.model, "encoder") else []
            self.encoder_optimizer = torch.optim.AdamW(encoder_params, lr=2e-4, weight_decay=1e-4) if encoder_params else None

            predictor_params = [
                p for n, p in self.model.named_parameters()
                if "predictor" in n and "sensory_predictor" not in n and "corrector" not in n
            ]
            if hasattr(self.model, "dynamics"):
                dynamics_p_params = [
                    p for n, p in self.model.dynamics.named_parameters()
                    if "fuse_" in n or "recurrent_proj" in n or "norm" in n
                ]
                predictor_params.extend(dynamics_p_params)

            self.predictor_optimizer = torch.optim.AdamW(predictor_params, lr=learning_rate, weight_decay=weight_decay) if predictor_params else None

            probe_params = [
                p for n, p in self.model.named_parameters()
                if ("decoder" in n or "probe" in n) and "corrector" not in n
            ]
            self.probe_optimizer = torch.optim.AdamW(probe_params, lr=probe_lr, weight_decay=probe_weight_decay) if probe_params else None
            self.corrector_optimizer = None

        elif self.mode == "train_corrector":
            for n, p in self.model.named_parameters():
                if "corrector" not in n: p.requires_grad = False
                else: p.requires_grad = True

            if hasattr(self.model, "encoder"): self.model.encoder.eval()
            if hasattr(self.model, "dynamics"): self.model.dynamics.eval()
            if hasattr(self.model, "predictor"): self.model.predictor.eval()
            if hasattr(self.model, "sensory_predictor"): self.model.sensory_predictor.eval()
            if hasattr(self.model, "physical_decoder"): self.model.physical_decoder.eval()

            self.encoder_optimizer = None
            self.predictor_optimizer = None
            self.probe_optimizer = None

            corrector_params = list(self.model.corrector.parameters()) if hasattr(self.model, "corrector") and self.model.corrector is not None else []
            self.corrector_optimizer = torch.optim.AdamW(
                corrector_params,
                lr=(corrector_lr if corrector_lr is not None else learning_rate),
                weight_decay=weight_decay,
            ) if corrector_params else None

        self.verify_gradient_isolation()
        self.decoder_scheduler = None
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb")) if tensorboard_logging else None

        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.history = history if history is not None else []
        
        if self.history:
            def _selection_key(rec):
                return self.compute_selection_score(rec)[0]

            best_entry = min(self.history, key=_selection_key)
            best_hist_score = self.compute_selection_score(best_entry)[0]
            if self.best_combined_score == float("inf") or best_hist_score < self.best_combined_score or self.best_epoch is None:
                self.best_gate_passed = bool(best_entry.get("val_gate_passed", False))
                self.best_rollout_mae = best_entry.get("val_rollout_mae", float("inf"))
                self.best_drift_ratio = best_entry.get("val_drift_ratio", float("inf"))
                self.best_combined_score = best_hist_score
                self.best_val_pos_err = best_entry.get("val_pos_err", self.best_val_pos_err)
                self.best_val_vel_err = best_entry.get("val_vel_err", self.best_val_vel_err)
                self.best_epoch = best_entry.get("epoch", self.best_epoch)

        if optimizer_state:
            if self.encoder_optimizer and "encoder_optimizer" in optimizer_state and optimizer_state["encoder_optimizer"]:
                self.encoder_optimizer.load_state_dict(optimizer_state["encoder_optimizer"])
            if self.probe_optimizer and "probe_optimizer" in optimizer_state and optimizer_state["probe_optimizer"]:
                self.probe_optimizer.load_state_dict(optimizer_state["probe_optimizer"])
            if self.predictor_optimizer and "predictor_optimizer" in optimizer_state and optimizer_state["predictor_optimizer"]:
                self.predictor_optimizer.load_state_dict(optimizer_state["predictor_optimizer"])
            if self.corrector_optimizer and "corrector_optimizer" in optimizer_state and optimizer_state["corrector_optimizer"]:
                self.corrector_optimizer.load_state_dict(optimizer_state["corrector_optimizer"])

    def verify_gradient_isolation(self) -> None:
        if self.mode == "train_predictor":
            if hasattr(self.model, "corrector") and self.model.corrector is not None:
                for name, param in self.model.corrector.named_parameters():
                    assert not param.requires_grad, f"Gradient leak! Corrector param '{name}' must be frozen."
            assert self.corrector_optimizer is None
        elif self.mode == "train_corrector":
            for name, param in self.model.named_parameters():
                if "corrector" not in name:
                    assert not param.requires_grad, f"Gradient leak! Param '{name}' must be frozen."
                else:
                    assert param.requires_grad
            assert self.encoder_optimizer is None
            assert self.predictor_optimizer is None
            assert self.probe_optimizer is None
            assert self.corrector_optimizer is not None

    def compute_selection_score(self, val_metrics: Dict[str, float]) -> Tuple[float, str]:
        if self.mode == "train_predictor":
            h1_pos = val_metrics.get("val_pos_err_h1", val_metrics.get("val_pos_err", 0.0))
            h1_vel = val_metrics.get("val_vel_err_h1", val_metrics.get("val_vel_err", 0.0))
            h5_pos = val_metrics.get("val_pos_err_h5", h1_pos)
            score = float(h1_pos + 0.5 * h1_vel + 0.25 * h5_pos)
            desc = f"Local Kinematic Error (H1 Pos: {h1_pos:.4f}, H1 Vel: {h1_vel:.4f}, H5 Pos: {h5_pos:.4f})"
            return score, desc
        else:
            val_loss = float(val_metrics.get("val_corr_loss", val_metrics.get("val_total_loss", val_metrics.get("val_rollout_mae", 0.0))))
            mae = float(val_metrics.get("val_rollout_mae", 0.0))
            drift_ratio = float(val_metrics.get("val_drift_ratio", 1.0))
            desc = f"Corrector Val Loss: {val_loss:.5f} (MAE: {mae:.4f}, Drift: {drift_ratio:.2f}x)"
            return val_loss, desc

    def _print_training_header(self, total_epochs: int) -> None:
        chronicle.log_application_title("SPWM CONTINUOUS ONLINE TRAINER")
        chronicle.log_detail("Training Mode", self.mode.upper())
        chronicle.log_detail("Epochs", f"{self.start_epoch} -> {self.start_epoch + total_epochs - 1}")
        chronicle.log_detail("Corrector Burn-in Steps", f"{self.corrector_burn_in_steps}")
        chronicle.log_newline()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        if self.mode == "train_corrector":
            self.model.eval()
            if hasattr(self.model, "corrector") and self.model.corrector is not None:
                self.model.corrector.train()
        else:
            self.model.train()

        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0, "l_pred": 0.0, "l_multi": 0.0, "l_var": 0.0,
            "l_vel": 0.0, "l_sparse": 0.0, "l_probe": 0.0, "l_coord": 0.0,
            "l_corr": 0.0, "l_corr_asymptotic": 0.0,
            "spike_rate": 0.0, "grad_norm": 0.0,
        }
        num_batches = 0
        total_grad_norm_accum = 0.0
        is_warmup = (epoch <= self.encoder_warmup_epochs) if self.mode != "train_corrector" else False
        accum_eprop = (self.mode != "train_corrector")

        for batch in self.train_loader:
            if self.smooth_horizon_sampling and hasattr(self.loss_fn, "multi_step_horizon"):
                self.loss_fn.multi_step_horizon = 1 if is_warmup else random.randint(1, self.k_max)

            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            with torch.no_grad():
                out = self.model(
                    events,
                    accumulate_local_updates=accum_eprop,
                    learning_rate=self.learning_rate,
                    target_kinematics=true_kin,
                )

            if accum_eprop and hasattr(self.model, "apply_accumulated_updates"):
                self.model.apply_accumulated_updates(learning_rate=self.learning_rate)

            # --- Step 1: Encoder Update ---
            enc_loss_val = 0.0
            if self.encoder_optimizer is not None:
                self.encoder_optimizer.zero_grad()
                with torch.enable_grad():
                    enc_seq, kp_seq = self.model.encoder(events, return_keypoints=True)
                    lambda_coord = getattr(self.loss_fn, "lambda_coord", 0.15)
                    loss_enc = torch.tensor(0.0, device=self.device)
                    if lambda_coord > 0.0 and true_kin is not None:
                        l_coord = self.loss_fn.coordinate_loss(kp_seq, true_kin)
                        loss_enc = loss_enc + lambda_coord * l_coord
                        epoch_losses["l_coord"] += l_coord.item()
                    if loss_enc.requires_grad:
                        loss_enc.backward()
                        if self.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.grad_clip_norm)
                        self.encoder_optimizer.step()
                        enc_loss_val = loss_enc.item()

            # --- Step 2: Predictor Update ---
            pred_loss_val = 0.0
            if self.predictor_optimizer is not None:
                self.predictor_optimizer.zero_grad()
                with torch.enable_grad():
                    latents = out.latent_states.detach()
                    pred_res = self.model.predictor(latents[:, :-1])
                    pred_z = pred_res.predicted_latent

                    all_spikes = (
                        torch.cat([out.fast_spikes, out.slow_spikes], dim=-1)
                        if getattr(out, "fast_spikes", None) is not None and getattr(out, "slow_spikes", None) is not None
                        else getattr(out, "fast_spikes", None)
                    )

                    pred_loss_dict = self.loss_fn.compute_predictor_loss(
                        predicted_latent=pred_z,
                        target_latents=latents[:, 1:],
                        all_latents=latents,
                        model_predictor=self.model.predictor,
                        physical_decoder=getattr(self.model, "physical_decoder", None),
                        true_kinematics=true_kin,
                        spikes=all_spikes,
                        is_warmup=is_warmup,
                    )
                    total_pred_loss = pred_loss_dict["total_loss"]
                    total_pred_loss.backward()

                    norm_val = 0.0
                    if self.grad_clip_norm > 0:
                        norm_tensor = torch.nn.utils.clip_grad_norm_(self.predictor_optimizer.param_groups[0]["params"], self.grad_clip_norm)
                        norm_val = norm_tensor.item() if hasattr(norm_tensor, "item") else float(norm_tensor)
                    self.predictor_optimizer.step()
                    self.recent_grad_norms.append(norm_val)
                    total_grad_norm_accum += norm_val
                    pred_loss_val = total_pred_loss.item()
                    epoch_losses["l_pred"] += pred_loss_dict["l_1step"].item()
                    epoch_losses["l_multi"] += pred_loss_dict["l_multi"].item()
                    epoch_losses["l_vel"] += pred_loss_dict["l_vel"].item()
                    epoch_losses["l_sparse"] += pred_loss_dict["l_sparse"].item()
                    epoch_losses["l_var"] += pred_loss_dict["l_var"].item()

            # --- Step 3: Probe Update ---
            probe_loss_val = 0.0
            if self.probe_optimizer is not None and true_kin is not None:
                self.probe_optimizer.zero_grad()
                with torch.enable_grad():
                    decoded = self.model.physical_decoder(out.latent_states.detach())
                    if hasattr(self.loss_fn, "kinematic_loss"):
                        probe_loss = self.loss_fn.kinematic_loss(decoded, true_kin)
                    else:
                        probe_loss = nn.functional.mse_loss(decoded, true_kin)
                    probe_loss.backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.probe_optimizer.param_groups[0]["params"], self.grad_clip_norm)
                    self.probe_optimizer.step()
                    probe_loss_val = probe_loss.item()
                epoch_losses["l_probe"] += probe_loss_val

            # --- Step 4: Protected Corrector Optimization (Gain Mask Tuning) ---
            corr_loss_val = 0.0
            if (
                self.corrector_optimizer is not None
                and not is_warmup
                and hasattr(self.model, "corrector")
                and self.model.corrector is not None
                and hasattr(self.model, "predict_rollout")
            ):
                self.corrector_optimizer.zero_grad()
                with torch.enable_grad():
                    with _frozen(self.model.predictor), _frozen(self.model.physical_decoder):
                        latents = out.latent_states.detach()
                        B_lat, T_tot, D_lat = latents.shape
                        H_corr = min(self.corrector_horizon, T_tot - 2)

                        if H_corr >= 1 and true_kin is not None:
                            max_start = max(1, T_tot - H_corr - 1)
                            t0 = random.randint(0, max_start - 1) if max_start > 1 else 0
                            z_init = latents[:, t0].detach()
                            
                            if self.perturb_corrector_states and self.corrector_perturbation_std > 0.0:
                                z_init = z_init + torch.randn_like(z_init) * self.corrector_perturbation_std

                            b_idx = min(self.corrector_burn_in_steps, max(0, H_corr - 1))

                            pred_rollout, _ = self.model.predict_rollout(
                                initial_latent=z_init,
                                horizon=H_corr,
                                use_corrector=True,
                                return_fast_trajectory=False,
                                burn_in_steps=b_idx,
                            )

                            decoded_rollout = self.model.physical_decoder(pred_rollout)
                            target_kin = true_kin[:, t0 + 1 : t0 + 1 + H_corr]

                            corr_loss_dict = self.loss_fn.compute_corrector_rollout_loss(
                                decoded_rollout=decoded_rollout,
                                target_kinematics=target_kin,
                                burn_in_steps=b_idx,
                                num_objects=self.num_objects,
                                lambda_corrector_asymptotic=self.lambda_corrector_asymptotic,
                            )
                            loss_corr = corr_loss_dict["loss_corr"]

                            if loss_corr.requires_grad:
                                loss_corr.backward()
                                if self.grad_clip_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(self.corrector_optimizer.param_groups[0]["params"], self.grad_clip_norm)
                                self.corrector_optimizer.step()
                                corr_loss_val = loss_corr.item()
                                epoch_losses["l_corr_asymptotic"] += corr_loss_dict["l_kin"].item()
                epoch_losses["l_corr"] += corr_loss_val

            if hasattr(out, "mean_spike_rate"):
                epoch_losses["spike_rate"] += out.mean_spike_rate.item()

            lambda_probe = getattr(self.loss_fn, "lambda_probe", 2.0)
            epoch_losses["total_loss"] += pred_loss_val + enc_loss_val + (lambda_probe * probe_loss_val) + corr_loss_val
            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
        epoch_losses["grad_norm"] = total_grad_norm_accum / max(1, num_batches)

        return epoch_losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses: Dict[str, float] = collections.defaultdict(float)
        num_batches = 0

        for batch in self.val_loader:
            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None: true_kin = true_kin.to(self.device)

            out = self.model(events, accumulate_local_updates=False)

            if hasattr(self.model, "predictor") and out.latent_states.shape[1] > 1:
                pred_res_val = self.model.predictor(out.latent_states[:, :-1])
                val_l_pred = nn.functional.mse_loss(pred_res_val.predicted_latent, out.latent_states[:, 1:]).item()
            else:
                val_l_pred = 0.0
                
            val_losses["val_total_loss"] += val_l_pred
            val_losses["val_l_pred"] += val_l_pred

            if out.decoded_kinematics is not None and true_kin is not None:
                if hasattr(self.loss_fn, "kinematic_loss"):
                    probe_loss = self.loss_fn.kinematic_loss(out.decoded_kinematics, true_kin).item()
                else:
                    probe_loss = nn.functional.mse_loss(out.decoded_kinematics, true_kin).item()
                val_losses["val_l_probe"] += probe_loss
                val_losses["val_pos_err"] += position_error(out.decoded_kinematics, true_kin)
                val_losses["val_vel_err"] += velocity_error(out.decoded_kinematics, true_kin)

            # Compute validation corrector loss if in train_corrector mode
            # Evaluated with 0 perturbation starting from t=0, perfectly aligned with compute_corrector_rollout_loss
            if (
                self.mode == "train_corrector"
                and hasattr(self.model, "corrector")
                and self.model.corrector is not None
                and hasattr(self.model, "predict_rollout")
                and hasattr(self.model, "physical_decoder")
                and true_kin is not None
            ):
                latents = out.latent_states
                B_lat, T_tot, D_lat = latents.shape
                H_corr = min(self.corrector_horizon, T_tot - 2)
                b_idx = min(self.corrector_burn_in_steps, max(0, H_corr - 1))
                active_len = H_corr - b_idx

                if H_corr >= 1 and active_len > 0:
                    z_init = latents[:, 0]
                    val_pred_rollout, _ = self.model.predict_rollout(
                        initial_latent=z_init,
                        horizon=H_corr,
                        use_corrector=True,
                        return_fast_trajectory=False,
                        burn_in_steps=b_idx,
                    )
                    val_decoded_rollout = self.model.physical_decoder(val_pred_rollout)
                    val_target_kin = true_kin[:, 1 : 1 + H_corr]

                    val_corr_dict = self.loss_fn.compute_corrector_rollout_loss(
                        decoded_rollout=val_decoded_rollout,
                        target_kinematics=val_target_kin,
                        burn_in_steps=b_idx,
                        num_objects=self.num_objects,
                        lambda_corrector_asymptotic=self.lambda_corrector_asymptotic,
                    )
                    val_losses["val_corr_loss"] += val_corr_dict["loss_corr"].item()

            num_batches += 1

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)

        rollout_res = self.rollout_evaluator.evaluate_dataset(self.val_loader)
        val_losses["val_rollout_mae"] = rollout_res.rollout_mae
        val_losses["val_drift_ratio"] = rollout_res.drift_ratio
        val_losses["val_gate_passed"] = 1.0 if rollout_res.is_gate_passed(self.max_drift_ratio) else 0.0
        val_losses["val_dynamic_score"] = rollout_res.dynamic_score

        for h, pos_h in rollout_res.position_error_per_horizon.items():
            val_losses[f"val_pos_err_h{h}"] = pos_h
        for h, vel_h in rollout_res.velocity_error_per_horizon.items():
            val_losses[f"val_vel_err_h{h}"] = vel_h

        return val_losses

    def fit(self, epochs: int = 50, print_every: int = 1) -> List[Dict[str, float]]:
        """Ciclo principale ripristinato interamente."""
        start_time = time.time()
        base_elapsed = self.history[-1].get("elapsed_time", 0.0) if self.history else 0.0
        end_epoch = self.start_epoch + epochs - 1
        self._print_training_header(epochs)

        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_epoch_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate()
                epoch_duration = time.time() - t_epoch_start
                total_elapsed = base_elapsed + (time.time() - start_time)

                recent_norm = sum(self.recent_grad_norms) / max(1, len(self.recent_grad_norms))

                if (self.curriculum_multi_step and not self.smooth_horizon_sampling and hasattr(self.loss_fn, "multi_step_horizon")):
                    if epoch > self.encoder_warmup_epochs:
                        p1_k = self.curriculum_thresholds.get("phase_1_horizon", 3)
                        p2_k = self.curriculum_thresholds.get("phase_2_horizon", 6)
                        p2_th = self.curriculum_thresholds.get("phase_2_threshold", 0.20)
                        p3_k = self.curriculum_thresholds.get("phase_3_horizon", 10)
                        p3_th = self.curriculum_thresholds.get("phase_3_threshold", 0.12)
                        curr_val_pos = val_metrics.get("val_pos_err", float("inf"))
                        if curr_val_pos < p3_th: target_k = p3_k
                        elif curr_val_pos < p2_th: target_k = p2_k
                        else: target_k = p1_k

                        if target_k != self.loss_fn.multi_step_horizon:
                            old_k = self.loss_fn.multi_step_horizon
                            self.loss_fn.multi_step_horizon = target_k
                            chronicle.log_info(f"[Curriculum Horizon Shift] val_pos_err={curr_val_pos:.4f}: K transitioned from {old_k} to {target_k}")
                    else:
                        if getattr(self.loss_fn, "multi_step_horizon", 1) != 1:
                            self.loss_fn.multi_step_horizon = 1

                curr_score, score_description = self.compute_selection_score(val_metrics)
                curr_drift_ratio = val_metrics.get("val_drift_ratio", 1.0)
                curr_rollout_mae = val_metrics.get("val_rollout_mae", 0.0)
                curr_gate_passed = bool(val_metrics.get("val_gate_passed", 0.0) > 0.5)

                current_pos_h1 = val_metrics.get("val_pos_err_h1", val_metrics.get("val_pos_err", 0.0))
                current_vel_h1 = val_metrics.get("val_vel_err_h1", val_metrics.get("val_vel_err", 0.0))

                is_best = False
                if curr_score < self.best_combined_score:
                    is_best = True

                if is_best:
                    self.best_gate_passed = curr_gate_passed
                    self.best_rollout_mae = curr_rollout_mae
                    self.best_drift_ratio = curr_drift_ratio
                    self.best_combined_score = curr_score
                    self.best_epoch = epoch
                    self.best_val_pos_err = current_pos_h1
                    self.best_val_vel_err = current_vel_h1
                    
                    torch.save(self.model.state_dict(), self.save_dir / "model.pt")
                    if self.mode == "train_corrector":
                        torch.save(self.model.state_dict(), self.save_dir / "model_corrector.pt")
                        if hasattr(self.model, "corrector") and self.model.corrector is not None:
                            torch.save(self.model.corrector.state_dict(), self.save_dir / "corrector_weights.pt")
                    elif self.mode == "train_predictor":
                        torch.save(self.model.state_dict(), self.save_dir / "model_predictor.pt")

                record = {
                    "epoch": epoch,
                    "elapsed_time": total_elapsed,
                    **train_metrics,
                    **val_metrics,
                    "grad_norm_recent": recent_norm,
                    "selection_score": curr_score,
                    "combined_score": curr_score,
                }
                self.history.append(record)

                if self.writer is not None:
                    for k, v in record.items():
                        if k != "epoch":
                            self.writer.add_scalar(k, v, epoch)

                if self.decoder_scheduler is not None:
                    self.decoder_scheduler.step()

                checkpoint_path = self.save_dir / "checkpoint.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_val_loss": self.best_val_loss,
                        "best_val_pos_err": self.best_val_pos_err,
                        "best_val_vel_err": self.best_val_vel_err,
                        "best_combined_score": self.best_combined_score,
                        "best_rollout_mae": self.best_rollout_mae,
                        "best_drift_ratio": self.best_drift_ratio,
                        "best_gate_passed": self.best_gate_passed,
                        "best_epoch": self.best_epoch,
                        "history": self.history,
                        "optimizer_state": {
                            "encoder_optimizer": self.encoder_optimizer.state_dict() if self.encoder_optimizer else None,
                            "probe_optimizer": self.probe_optimizer.state_dict() if self.probe_optimizer else None,
                            "predictor_optimizer": self.predictor_optimizer.state_dict() if self.predictor_optimizer else None,
                            "corrector_optimizer": self.corrector_optimizer.state_dict() if self.corrector_optimizer else None,
                            "best_rollout_mae": self.best_rollout_mae,
                            "best_drift_ratio": self.best_drift_ratio,
                            "best_gate_passed": self.best_gate_passed,
                            "best_epoch": self.best_epoch,
                        },
                    },
                    checkpoint_path,
                )

                if epoch % print_every == 0 or epoch == end_epoch:
                    best_tag = " * [BEST]" if is_best else ""
                    current_k = f"~U(1,{self.k_max})" if (self.smooth_horizon_sampling and epoch > self.encoder_warmup_epochs) else getattr(self.loss_fn, "multi_step_horizon", 1)
                    warmup_tag = " [WARMUP]" if epoch <= self.encoder_warmup_epochs else ""
                    
                    if self.mode == "train_predictor":
                        header = f"Epoch [{epoch:03d}/{end_epoch:03d}] [PREDICTOR-PHASE] (K_fast={current_k}){warmup_tag}  Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag} | Remaining: {epoch_duration * (end_epoch - epoch) / 60:4.1f}m"
                        chronicle.log_section_header(header)
                        chronicle.log_detail("⚡ Fast Predictor Optimization", f"L_1step: {train_metrics.get('l_pred', 0.0):.5f} | L_multi: {train_metrics.get('l_multi', 0.0):.5f} | L_vel: {train_metrics.get('l_vel', 0.0):.5f} | L_sparse: {train_metrics.get('l_sparse', 0.0):.5f}", indent_level=1)
                        chronicle.log_detail("🎯 Representation & Probe", f"L_coord: {train_metrics.get('l_coord', 0.0):.5f} | L_probe: {train_metrics.get('l_probe', 0.0):.5f}", indent_level=1)
                        h5_pos = val_metrics.get('val_pos_err_h5', val_metrics.get('val_pos_err', 0.0))
                        chronicle.log_detail("📊 Validation Kinematics", f"Val Loss: {val_metrics.get('val_total_loss', 0.0):.5f} | Pos H1: {current_pos_h1:.4f} | Pos H5: {h5_pos:.4f} | Vel H1: {current_vel_h1:.4f}", indent_level=1)
                        chronicle.log_detail("🎯 Model Selection", f"Score: {curr_score:.5f} | Best: {self.best_combined_score:.5f} @ Ep {self.best_epoch}", indent_level=1)

                    elif self.mode == "train_corrector":
                        alpha_gain = getattr(self.model.corrector, 'alpha_gain', 0.5) if hasattr(self.model, "corrector") and self.model.corrector else 0.5
                        header = f"Epoch [{epoch:03d}/{end_epoch:03d}] [CORRECTOR-PHASE] (H={self.corrector_horizon}, burn-in={self.corrector_burn_in_steps}, α={alpha_gain:.2f})  Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag} | Remaining: {epoch_duration * (end_epoch - epoch) / 60:4.1f}m"
                        chronicle.log_section_header(header)
                        chronicle.log_detail("🔒 Predictor Core", "FROZEN (eval mode).", indent_level=1)
                        # L_kin = L_pos + beta_v * L_vel  (the raw kinematic rollout loss)
                        # L_corr = lambda_asymptotic * L_kin  (the optimized loss; equal to L_kin when lambda=1.0)
                        l_kin_val = train_metrics.get('l_corr_asymptotic', 0.0)
                        l_corr_val = train_metrics.get('l_corr', 0.0)
                        lambda_a = self.lambda_corrector_asymptotic
                        beta_v_val = getattr(self.loss_fn, 'beta_v', 2.0)
                        if abs(lambda_a - 1.0) < 1e-6:
                            corr_loss_str = f"L_kin (pos + {beta_v_val:.1f}·vel): {l_kin_val:.5f}"
                        else:
                            corr_loss_str = f"L_kin: {l_kin_val:.5f} | L_corr (λ={lambda_a:.2f}·L_kin): {l_corr_val:.5f}"
                        chronicle.log_detail("🧠 Gain Field Corrector", corr_loss_str, indent_level=1)
                        h1_pos = val_metrics.get('val_pos_err_h1', 0.0)
                        h25_pos = val_metrics.get('val_pos_err_h25', 0.0)
                        h100_pos = val_metrics.get('val_pos_err_h100', val_metrics.get('val_pos_err_h50', 0.0))
                        corr_spike_rate = val_metrics.get('mean_spike_rate', 0.0)
                        chronicle.log_detail("📈 Autonomous Rollout", f"MAE: {curr_rollout_mae:.5f} | H1: {h1_pos:.4f} | H25: {h25_pos:.4f} | H100: {h100_pos:.4f} | Drift: {curr_drift_ratio:.2f}x | Spikes: {corr_spike_rate:.3f}", indent_level=1)
                        chronicle.log_detail("🎯 Model Selection", f"Score: {curr_score:.5f} | Best: {self.best_combined_score:.5f} @ Ep {self.best_epoch}", indent_level=1)
                    else:
                        chronicle.log_section_header(f"Epoch [{epoch:03d}/{end_epoch:03d}]")
                        chronicle.log_detail("📊 Metrics", f"Train Loss: {train_metrics.get('total_loss', 0.0):.5f} | Val Loss: {val_metrics.get('val_total_loss', 0.0):.5f}", indent_level=1)
                    chronicle.log_newline()

        except KeyboardInterrupt:
            chronicle.log_warning("[INTERRUPT] Training interrupted manually.")
            
        self.save_training_log()
        chronicle.log_success(f"Session complete. Logs saved in '{self.save_dir}'. Best Selection Score: {self.best_combined_score:.5f} @ Epoch {self.best_epoch}")
        return self.history

    def save_training_log(self) -> None:
        if not self.history:
            return
        json_path = self.save_dir / "training_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        csv_path = self.save_dir / "training_log.csv"
        fieldnames = []
        for rec in self.history:
            for k in rec.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)