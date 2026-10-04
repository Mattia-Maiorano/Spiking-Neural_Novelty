"""
Forward-Only Continuous Online Trainer for SPWM (v7.1).
Maintains O(1) memory footprint scaling across long sequence horizons.
Executes online e-prop plasticity with online mini-batch updates.

Key features:
  - Pure Geometric Encoder Update: Ground-truth supervision via L_coord only,
    avoiding predictive confirmation bias.
  - Frozen Probe Velocity Supervision: L_vel differentiates through physical_decoder
    with its parameters frozen so gradients flow exclusively into predictor/dynamics p.
  - Decoupled Optimizers: Dynamics fusion layers (fuse_*, recurrent_proj, norm) are
    trained via predictor_optimizer. probe_optimizer only updates probe weights.
  - Reliable Joint Model Selection: Uses normalized combined validation metric (position + velocity)
    avoiding non-stationary historical Z-score traps.
"""

from __future__ import annotations
import collections
import contextlib
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import chronicle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from spwm.learning.losses import LossOutput, SPWMLoss
from spwm.learning.metrics import (
    position_error,
    velocity_error,
)


@contextlib.contextmanager
def freeze_parameters(module: nn.Module):
    """Context manager to temporarily freeze parameters of a module."""
    prev_states = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad = False
        yield
    finally:
        for p, state in zip(module.parameters(), prev_states):
            p.requires_grad = state


class Trainer:
    """
    Continuous Online Trainer (SPWM).
    Streams sequence batches frame-by-frame with O(1) memory footprint and
    executes online forward-only e-prop plasticity.
    """

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
        best_epoch: Optional[int] = None,   
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict] = None,
        curriculum_multi_step: bool = False,
        curriculum_thresholds: Optional[Dict[str, Any]] = None,
        encoder_warmup_epochs: int = 60,
    ) -> None:
        self.best_epoch: Optional[int] = None
        self.best_val_pos_err: float = best_val_pos_err
        self.best_val_vel_err: float = float("inf")
        self.best_combined_score: float = float("inf")
        self.learning_algorithm = learning_algorithm.lower()
        self.learning_rate = learning_rate
        self.curriculum_multi_step = curriculum_multi_step
        self.encoder_warmup_epochs = encoder_warmup_epochs
        self.curriculum_thresholds = curriculum_thresholds or {
            "phase_1_horizon": 3,
            "phase_2_horizon": 6,
            "phase_2_threshold": 0.20,
            "phase_3_horizon": 10,
            "phase_3_threshold": 0.12,
        }

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
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

        # ------------------------------------------------------------------
        # Configurazione Ottimizzatori
        # ------------------------------------------------------------------

        # Step-1: Ottimizzatore Encoder (SpatialSoftmax + front-end convoluzionale)
        encoder_params = (
            list(self.model.encoder.parameters())
            if hasattr(self.model, "encoder")
            else []
        )
        self.encoder_optimizer = (
            torch.optim.AdamW(encoder_params, lr=2e-4, weight_decay=1e-4)
            if encoder_params
            else None
        )

        # Step-2: Ottimizzatore Predictor + Dinamica del Momento Latente p
        predictor_params = [
            p
            for n, p in self.model.named_parameters()
            if "predictor" in n and "sensory_predictor" not in n
        ]
        if hasattr(self.model, "dynamics"):
            dynamics_p_params = [
                p
                for n, p in self.model.dynamics.named_parameters()
                if "fuse_" in n or "recurrent_proj" in n or "norm" in n
            ]
            predictor_params.extend(dynamics_p_params)

        self.predictor_optimizer = (
            torch.optim.AdamW(predictor_params, lr=learning_rate, weight_decay=weight_decay)
            if predictor_params
            else None
        )

        # Step-3: Ottimizzatore Probe Cinematico (SOLO decoder / probe)
        probe_params = [
            p
            for n, p in self.model.named_parameters()
            if "decoder" in n or "probe" in n
        ]
        self.probe_optimizer = (
            torch.optim.AdamW(probe_params, lr=probe_lr, weight_decay=probe_weight_decay)
            if probe_params
            else None
        )

        self.decoder_scheduler = None

        self.writer = (
            SummaryWriter(log_dir=str(self.save_dir / "tb"))
            if tensorboard_logging
            else None
        )

        # Ripristino stato Checkpoint
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.history = history if history is not None else []
        
        # Fallback automatico: se c'è una cronologia precedente ma best_combined_score è infinito
        if self.history and self.best_combined_score == float("inf"):
            best_entry = min(
                self.history, 
                key=lambda x: x.get("combined_score", float("inf"))
            )
            self.best_combined_score = best_entry.get("combined_score", float("inf"))
            self.best_val_pos_err = best_entry.get("val_pos_err", self.best_val_pos_err)
            self.best_val_vel_err = best_entry.get("val_vel_err", self.best_val_vel_err)
            self.best_epoch = best_entry.get("epoch", None)

        if optimizer_state:
            if self.encoder_optimizer and "encoder_optimizer" in optimizer_state:
                self.encoder_optimizer.load_state_dict(optimizer_state["encoder_optimizer"])
            if self.probe_optimizer and "probe_optimizer" in optimizer_state:
                self.probe_optimizer.load_state_dict(optimizer_state["probe_optimizer"])
            if self.predictor_optimizer and "predictor_optimizer" in optimizer_state:
                self.predictor_optimizer.load_state_dict(optimizer_state["predictor_optimizer"])
            if "best_combined_score" in optimizer_state:
                self.best_combined_score = optimizer_state["best_combined_score"]
            if "best_val_vel_err" in optimizer_state:
                self.best_val_vel_err = optimizer_state["best_val_vel_err"]
            if "best_val_pos_err" in optimizer_state:
                self.best_val_pos_err = optimizer_state["best_val_pos_err"]
            if "best_epoch" in optimizer_state:
                self.best_epoch = optimizer_state["best_epoch"]


    def _print_training_header(self, total_epochs: int) -> None:
        """Visualizza i parametri principali prima dell'avvio."""
        chronicle.log_application_title("SPWM CONTINUOUS ONLINE TRAINER")
        chronicle.log_detail("Device", self.device)
        chronicle.log_detail(
            "Epochs",
            f"{self.start_epoch} -> {self.start_epoch + total_epochs - 1}  (Total: {total_epochs})",
        )
        chronicle.log_detail("Learning Rate", f"{self.learning_rate:.2e}")
        chronicle.log_detail("Algorithm", self.learning_algorithm)
        chronicle.log_detail("Checkpoint Dir", str(self.save_dir))
        chronicle.log_detail("Encoder Warmup Epochs", f"{self.encoder_warmup_epochs}")
        chronicle.log_detail("Encoder Optimizer", "Enabled" if self.encoder_optimizer else "Disabled")
        chronicle.log_detail("Probe Optimizer", "Enabled" if self.probe_optimizer else "Disabled")
        chronicle.log_detail("Predictor Optimizer", "Enabled" if self.predictor_optimizer else "Disabled")
        chronicle.log_newline()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Esegue una singola epoca di training su tutti i batch dello stream."""
        self.model.train()
        epoch_losses: Dict[str, float] = {
            "total_loss": 0.0,
            "l_pred": 0.0,
            "l_multi": 0.0,
            "l_var": 0.0,
            "l_vel": 0.0,
            "l_sparse": 0.0,
            "l_probe": 0.0,
            "l_coord": 0.0,
            "spike_rate": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0
        total_grad_norm_accum = 0.0
        is_warmup = epoch <= self.encoder_warmup_epochs

        for batch in self.train_loader:
            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            # Passaggio neuromorfo forward-only con accumulo locale e-prop
            with torch.no_grad():
                out = self.model(
                    events,
                    accumulate_local_updates=True,
                    learning_rate=self.learning_rate,
                    target_kinematics=true_kin,
                )

            if hasattr(self.model, "apply_accumulated_updates"):
                self.model.apply_accumulated_updates(learning_rate=self.learning_rate)

            # ----------------------------------------------------------
            # Step 1: Encoder Update (Supervisione geometrica pura)
            # ----------------------------------------------------------
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
                            torch.nn.utils.clip_grad_norm_(
                                self.model.encoder.parameters(), self.grad_clip_norm
                            )
                        self.encoder_optimizer.step()
                        enc_loss_val = loss_enc.item()

            # ----------------------------------------------------------
            # Step 2: Predictor & Dynamics p Update
            # ----------------------------------------------------------
            pred_loss_val = 0.0
            if self.predictor_optimizer is not None:
                self.predictor_optimizer.zero_grad()
                with torch.enable_grad():
                    latents = out.latent_states.detach()
                    z_in = latents[:, :-1]
                    z_target = latents[:, 1:]

                    pred_res = self.model.predictor(z_in)
                    pred_z = pred_res.predicted_latent
                    l_1step = nn.functional.mse_loss(pred_z, z_target)

                    # Multi-step autoregressive rollout
                    l_multi = torch.tensor(0.0, device=self.device)
                    lambda_multi = getattr(self.loss_fn, "lambda_multi", 0.5) if not is_warmup else 0.0
                    if lambda_multi > 0.0:
                        l_multi = self.loss_fn.multi_step_rollout_loss(
                            latents, self.model.predictor
                        )

                    # Supervisione della velocità multi-passo: probe congelato temporaneamente
                    l_vel = torch.tensor(0.0, device=self.device)
                    lambda_vel = getattr(self.loss_fn, "lambda_vel", 0.5)
                    if (
                        lambda_vel > 0.0
                        and true_kin is not None
                        and hasattr(self.model, "physical_decoder")
                    ):
                        # Chiama la nuova versione strided (k=3, dt=0.01) definita in losses.py
                        l_vel = self.loss_fn.velocity_loss_frozen_probe(
                            pred_z=pred_z,
                            true_kinematics=true_kin,
                            physical_decoder=self.model.physical_decoder,
                            stride_k=3,
                            dt=0.01,
                        )


                    # Sparsità differenziabile sugli spike del predittore
                    l_sparse = torch.tensor(0.0, device=self.device)
                    lambda_sparse = getattr(self.loss_fn, "lambda_sparse", 0.5)

                    # 1. Recupera i due tensori differenziabili da SPWMSequenceOutput
                    fast_spk = getattr(out, "fast_spikes", None)
                    slow_spk = getattr(out, "slow_spikes", None)

                    if lambda_sparse > 0.0 and fast_spk is not None and slow_spk is not None:
                        # Concatena l'intera popolazione ALIF: [B, T, dim_fast + dim_slow]
                        all_spikes = torch.cat([fast_spk, slow_spk], dim=-1)
                        
                        target_sr = getattr(self.loss_fn, "target_spike_rate", 0.10)
                        # Calcolo L1 loss differenziabile verso il target (es. 10%)
                        l_sparse = torch.abs(all_spikes.mean() - target_sr)

                    # Regolarizzazione di varianza anti-collasso (VICReg style)
                    l_var = torch.tensor(0.0, device=self.device)
                    lambda_var = getattr(self.loss_fn, "lambda_var", 0.1)
                    if lambda_var > 0.0:
                        l_var = self.loss_fn.variance_loss(pred_z)

                    total_pred_loss = (
                        getattr(self.loss_fn, "lambda_pred", 1.0) * l_1step
                        + lambda_multi * l_multi
                        + lambda_vel * l_vel
                        + lambda_sparse * l_sparse
                        + lambda_var * l_var
                    )
                    total_pred_loss.backward()

                    norm_val = 0.0
                    if self.grad_clip_norm > 0:
                        norm_tensor = torch.nn.utils.clip_grad_norm_(
                            self.predictor_optimizer.param_groups[0]["params"],
                            self.grad_clip_norm,
                        )
                        norm_val = (
                            norm_tensor.item()
                            if hasattr(norm_tensor, "item")
                            else float(norm_tensor)
                        )
                    self.predictor_optimizer.step()

                    self.recent_grad_norms.append(norm_val)
                    total_grad_norm_accum += norm_val
                    pred_loss_val = total_pred_loss.item()

                    epoch_losses["l_pred"] += l_1step.item()
                    epoch_losses["l_multi"] += l_multi.item()
                    epoch_losses["l_var"] += l_var.item()
                    epoch_losses["l_vel"] += l_vel.item()
                    epoch_losses["l_sparse"] += l_sparse.item()

            # ----------------------------------------------------------
            # Step 3: Probe Update (su stati latenti z staccati dal grafo)
            # ----------------------------------------------------------
            probe_loss_val = 0.0
            if self.probe_optimizer is not None and true_kin is not None:
                self.probe_optimizer.zero_grad()
                with torch.enable_grad():
                    z_states = out.latent_states.detach()
                    decoded = self.model.physical_decoder(z_states)
                    probe_loss = nn.functional.mse_loss(decoded, true_kin)
                    probe_loss.backward()
                    if self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.probe_optimizer.param_groups[0]["params"],
                            self.grad_clip_norm,
                        )
                    self.probe_optimizer.step()
                    probe_loss_val = probe_loss.item()

                epoch_losses["l_probe"] += probe_loss_val

            if hasattr(out, "mean_spike_rate"):
                epoch_losses["spike_rate"] += out.mean_spike_rate.item()

            # Aggregazione Loss complessiva
            lambda_probe = getattr(self.loss_fn, "lambda_probe", 2.0)
            epoch_losses["total_loss"] += (
                pred_loss_val + enc_loss_val + (lambda_probe * probe_loss_val)
            )
            num_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
        epoch_losses["grad_norm"] = total_grad_norm_accum / max(1, num_batches)

        return epoch_losses

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Valuta il modello sul dataset di validazione in sola lettura."""
        self.model.eval()
        val_losses: Dict[str, float] = {
            "val_total_loss": 0.0,
            "val_l_pred": 0.0,
            "val_l_multi": 0.0,
            "val_l_probe": 0.0,
            "val_l_coord": 0.0,
            "val_pos_err": 0.0,
            "val_vel_err": 0.0,
            "val_spike_rate": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            out = self.model(events, accumulate_local_updates=False)

            if hasattr(self.model, "predictor") and out.latent_states.shape[1] > 1:
                z_in_val = out.latent_states[:, :-1]
                z_tgt_val = out.latent_states[:, 1:]
                pred_res_val = self.model.predictor(z_in_val)
                val_l_pred = nn.functional.mse_loss(pred_res_val.predicted_latent, z_tgt_val).item()
            else:
                val_l_pred = 0.0
            val_losses["val_total_loss"] += val_l_pred
            val_losses["val_l_pred"] += val_l_pred

            # Multi-step rollout su validation set
            lambda_multi = getattr(self.loss_fn, "lambda_multi", 0.5)
            if lambda_multi > 0.0 and hasattr(self.model, "predictor"):
                val_l_multi = self.loss_fn.multi_step_rollout_loss(
                    out.latent_states, self.model.predictor
                ).item()
                val_losses["val_l_multi"] += val_l_multi
                val_losses["val_total_loss"] += lambda_multi * val_l_multi

            # Decodifica cinematica ed errori fisici
            if out.decoded_kinematics is not None and true_kin is not None:
                probe_loss = nn.functional.mse_loss(out.decoded_kinematics, true_kin)
                val_losses["val_l_probe"] += probe_loss.item()
                lambda_probe = getattr(self.loss_fn, "lambda_probe", 2.0)
                val_losses["val_total_loss"] += lambda_probe * probe_loss.item()

                val_losses["val_pos_err"] += position_error(
                    out.decoded_kinematics, true_kin
                )
                val_losses["val_vel_err"] += velocity_error(
                    out.decoded_kinematics, true_kin
                )

            # Controllo diagnostico coordinate geometriche
            lambda_coord = getattr(self.loss_fn, "lambda_coord", 0.0)
            if lambda_coord > 0.0 and true_kin is not None and hasattr(self.model, "encoder"):
                _, kp_val = self.model.encoder(events, return_keypoints=True)
                l_coord_val = self.loss_fn.coordinate_loss(kp_val, true_kin)
                val_losses["val_l_coord"] += l_coord_val.item()

            if hasattr(out, "mean_spike_rate"):
                val_losses["val_spike_rate"] += out.mean_spike_rate.item()

            num_batches += 1

        for k in val_losses:
            val_losses[k] /= max(1, num_batches)

        return val_losses

    def fit(
        self,
        epochs: int = 50,
        print_every: int = 1,
    ) -> List[Dict[str, float]]:
        """Esegue il ciclo completo di addestramento su più epoche."""
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

                recent_norm = (
                    sum(self.recent_grad_norms) / max(1, len(self.recent_grad_norms))
                )

                # ------------------------------------------------------
                # Curriculum Adattivo su Orizzonte Multi-Step K
                # ------------------------------------------------------
                if self.curriculum_multi_step and hasattr(self.loss_fn, "multi_step_horizon"):
                    if epoch > self.encoder_warmup_epochs:
                        p1_k = self.curriculum_thresholds.get("phase_1_horizon", 3)
                        p2_k = self.curriculum_thresholds.get("phase_2_horizon", 6)
                        p2_th = self.curriculum_thresholds.get("phase_2_threshold", 0.20)
                        p3_k = self.curriculum_thresholds.get("phase_3_horizon", 10)
                        p3_th = self.curriculum_thresholds.get("phase_3_threshold", 0.12)

                        curr_val_pos = val_metrics["val_pos_err"]
                        if curr_val_pos < p3_th:
                            target_k = p3_k
                        elif curr_val_pos < p2_th:
                            target_k = p2_k
                        else:
                            target_k = p1_k

                        if target_k != self.loss_fn.multi_step_horizon:
                            old_k = self.loss_fn.multi_step_horizon
                            self.loss_fn.multi_step_horizon = target_k
                            chronicle.log_info(
                                f"[Curriculum Horizon Shift] val_pos_err={curr_val_pos:.4f}: "
                                f"K transitioned from {old_k} to {target_k}"
                            )
                    else:
                        if self.loss_fn.multi_step_horizon != 1:
                            self.loss_fn.multi_step_horizon = 1

                # ------------------------------------------------------
                # Selezione del Miglior Modello (Posizione + Velocità)
                # ------------------------------------------------------
                current_pos = val_metrics["val_pos_err"]
                current_vel = val_metrics["val_vel_err"]

                # Punteggio normalizzato basato su ordini tipici di scala
                # (pos ~ 0.10, vel ~ 0.60: peso 0.5 per bilanciare l'impatto)
                combined_score = current_pos + (0.5 * current_vel)

                is_best = combined_score < self.best_combined_score
                if is_best:
                    self.best_combined_score = combined_score
                    self.best_epoch = epoch
                    self.best_val_pos_err = current_pos
                    self.best_val_vel_err = current_vel
                    torch.save(self.model.state_dict(), self.save_dir / "model.pt")

                record = {
                    "epoch": epoch,
                    "elapsed_time": total_elapsed,
                    **train_metrics,
                    **val_metrics,
                    "grad_norm_recent": recent_norm,
                    "combined_score": combined_score,
                }
                self.history.append(record)

                if self.writer is not None:
                    for k, v in record.items():
                        if k != "epoch":
                            self.writer.add_scalar(k, v, epoch)

                if self.decoder_scheduler is not None:
                    self.decoder_scheduler.step()

                # Snapshot checkpoint per ripresa
                checkpoint_path = self.save_dir / "checkpoint.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "best_val_loss": self.best_val_loss,
                        "best_val_pos_err": self.best_val_pos_err,
                        "best_val_vel_err": self.best_val_vel_err,
                        "best_combined_score": self.best_combined_score,
                        "history": self.history,
                        "optimizer_state": {
                            "encoder_optimizer": self.encoder_optimizer.state_dict()
                            if self.encoder_optimizer
                            else None,
                            "probe_optimizer": self.probe_optimizer.state_dict()
                            if self.probe_optimizer
                            else None,
                            "predictor_optimizer": self.predictor_optimizer.state_dict()
                            if self.predictor_optimizer
                            else None,
                        },
                    },
                    checkpoint_path,
                )

                if epoch % print_every == 0 or epoch == end_epoch:
                    best_tag = " * [BEST]" if is_best else ""
                    current_k = getattr(self.loss_fn, "multi_step_horizon", 1)
                    warmup_tag = " [WARMUP]" if epoch <= self.encoder_warmup_epochs else ""
                    header = (
                        f"Epoch [{epoch:03d}/{end_epoch:03d}]"
                        f" (K={current_k}){warmup_tag}"
                        f"  Time: {epoch_duration:5.1f}s"
                        f"  Total: {total_elapsed / 60:4.1f}m"
                        f"{best_tag}"
                    )
                    chronicle.log_section_header(header)
                    chronicle.log_detail(
                        "Train Loss",
                        (
                            f"{train_metrics['total_loss']:.5f}"
                            f"  (Pred: {train_metrics['l_pred']:.5f},"
                            f" Multi: {train_metrics.get('l_multi', 0.0):.5f},"
                            f" Var: {train_metrics.get('l_var', 0.0):.5f},"
                            f" Probe: {train_metrics['l_probe']:.5f})"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Val Loss",
                        (
                            f"{val_metrics['val_total_loss']:.5f}"
                            f"  (Pred: {val_metrics['val_l_pred']:.5f},"
                            f" Multi: {val_metrics.get('val_l_multi', 0.0):.5f},"
                            f" Probe: {val_metrics.get('val_l_probe', 0.0):.5f})"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Pos Err / Vel Err",
                        f"Pos: {current_pos:.5f} | Vel: {current_vel:.5f}",
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "L_coord",
                        (
                            f"Train {train_metrics['l_coord']:.5f}"
                            f" / Val {val_metrics.get('val_l_coord', 0.0):.5f}"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "L_vel / L_sparse",
                        (
                            f"Vel {train_metrics.get('l_vel', 0.0):.5f}"
                            f" / Sparse {train_metrics.get('l_sparse', 0.0):.5f}"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Spike Rate",
                        (
                            f"Train {train_metrics['spike_rate']:.3f}"
                            f" / Val {val_metrics['val_spike_rate']:.3f}"
                        ),
                        indent_level=1,
                    )
                    chronicle.log_detail(
                        "Score (Comb)",
                        f"{combined_score:.5f} (Best: {self.best_combined_score:.5f} @ Ep {self.best_epoch})",
                        indent_level=1,
                    )
                    chronicle.log_newline()

        except KeyboardInterrupt:
            chronicle.log_warning(
                "[INTERRUPT] Training interrotto manualmente dall'utente."
            )
            try:
                chronicle.log_info("Valutazione dello stato attuale in corso...")
                val_metrics = self.evaluate()
                c_pos = val_metrics.get("val_pos_err", float("inf"))
                c_vel = val_metrics.get("val_vel_err", float("inf"))
                c_score = c_pos + (0.5 * c_vel)

                if c_score < self.best_combined_score:
                    self.best_combined_score = c_score
                    self.best_val_pos_err = c_pos
                    self.best_val_vel_err = c_vel
                    model_path = self.save_dir / "model.pt"
                    torch.save(self.model.state_dict(), model_path)
                    chronicle.log_success(
                        f"Nuovo miglior modello salvato su interrupt in: {model_path}"
                    )
            except Exception as e:
                chronicle.log_error(f"Errore post-interrupt: {e}")
                model_path = self.save_dir / "model.pt"
                torch.save(self.model.state_dict(), model_path)

        self.save_training_log()
        chronicle.log_success(
            f"Session complete. Logs saved in '{self.save_dir}'. "
            f"Best Val Pos Err: {self.best_val_pos_err:.5f} | "
            f"Best Val Vel Err: {self.best_val_vel_err:.5f} | "
            f"Best Score: {self.best_combined_score:.5f} (Epoch {self.best_epoch})"
        )
        return self.history

    def save_training_log(self) -> None:
        """Salva i log di addestramento nei formati JSON, CSV e TXT."""
        if not self.history:
            return

        json_path = self.save_dir / "training_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        csv_path = self.save_dir / "training_log.csv"
        keys = list(self.history[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

        txt_path = self.save_dir / "training_log.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for rec in self.history:
                f.write(
                    f"Epoch {rec.get('epoch'):03d} | TrainLoss {rec.get('total_loss'):.4f} | "
                    f"ValLoss {rec.get('val_total_loss'):.4f} | SpikeRate {rec.get('val_spike_rate'):.3f} | "
                    f"Time {rec.get('elapsed_time'):.1f}s\n"
                )
                loss_items = [
                    f"total_loss={rec.get('total_loss'):.4f}",
                    f"l_pred={rec.get('l_pred'):.4f}",
                    f"l_multi={rec.get('l_multi', 0.0):.4f}",
                    f"l_var={rec.get('l_var', 0.0):.4f}",
                    f"l_vel={rec.get('l_vel', 0.0):.4f}",
                    f"l_sparse={rec.get('l_sparse', 0.0):.4f}",
                    f"l_probe={rec.get('l_probe', 0.0):.4f}",
                    f"l_coord={rec.get('l_coord', 0.0):.4f}",
                    f"spike_rate={rec.get('spike_rate'):.3f}",
                    f"val_total_loss={rec.get('val_total_loss'):.4f}",
                    f"val_l_pred={rec.get('val_l_pred'):.4f}",
                    f"val_l_multi={rec.get('val_l_multi', 0.0):.4f}",
                    f"val_l_probe={rec.get('val_l_probe', 0.0):.4f}",
                    f"val_pos_err={rec.get('val_pos_err'):.4f}",
                    f"val_vel_err={rec.get('val_vel_err'):.4f}",
                    f"val_spike_rate={rec.get('val_spike_rate'):.3f}",
                    f"combined_score={rec.get('combined_score'):.4f}",
                ]
                f.write("  Losses: " + ", ".join(loss_items) + "\n")
                f.write(
                    f"  GradNormOverall={rec.get('grad_norm'):.4f}, "
                    f"GradNormRecent10={rec.get('grad_norm_recent'):.4f}\n\n"
                )
