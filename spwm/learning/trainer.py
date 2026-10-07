"""
Forward-Only Continuous Online Trainer for SPWM (v7.1).
Maintains O(1) memory footprint scaling across long sequence horizons.
Executes online e-prop plasticity with online mini-batch updates.

Key features:
  - Pure Geometric Encoder Update: Ground-truth supervision via L_coord only,
    avoiding predictive confirmation bias.
  - Frozen Probe Velocity Supervision: L_vel differentiates through physical_decoder
    with its parameters frozen so gradients flow exclusively into predictor/dynamics p.
  - Two-Scale Velocity Target: Convex combination of k=1 and k=3 velocity targets.
  - Smooth Horizon Sampling: Uniform continuous sampling K_t ~ U(1, K_max) to prevent asymptotic shock.
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
import random
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

from spwm.learning.losses import LossOutput, SPWMLoss, _frozen
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
        corrector_horizon: int = 25,
        lambda_corrector_asymptotic: float = 1.0,
        lambda_corrector_quiescence: float = 0.5,
        corrector_quiescence_margin: float = 0.15,
        corrector_quiescence_cap: float = 0.50,
        curriculum_thresholds: Optional[Dict[str, Any]] = None,
        encoder_warmup_epochs: int = 60,
        max_drift_ratio: float = 5.0,
        rollout_horizons: Optional[Sequence[int]] = None,
        save_best_metric: str = "pareto_rollout",
        num_objects: int = 1,
        mode: str = "train_predictor",
    ) -> None:
        self.mode = mode.lower()
        if self.mode not in ("train_predictor", "train_corrector", "joint"):
            raise ValueError(
                f"Unknown training mode '{mode}'. Must be one of: 'train_predictor', 'train_corrector', 'joint'."
            )

        self.best_epoch: Optional[int] = best_epoch
        self.best_val_pos_err: float = best_val_pos_err
        self.best_val_vel_err: float = best_val_vel_err
        self.best_combined_score: float = best_combined_score
        self.best_rollout_mae: float = best_rollout_mae
        self.best_drift_ratio: float = best_drift_ratio
        self.best_gate_passed: bool = best_gate_passed
        self.max_drift_ratio: float = float(max_drift_ratio)
        self.rollout_horizons: List[int] = list(rollout_horizons) if rollout_horizons is not None else [1, 5, 10, 25, 50]
        self.save_best_metric: str = save_best_metric
        self.num_objects: int = num_objects

        self.learning_algorithm = learning_algorithm.lower()
        self.learning_rate = learning_rate
        self.curriculum_multi_step = curriculum_multi_step
        self.smooth_horizon_sampling = smooth_horizon_sampling
        self.k_max = k_max
        self.corrector_horizon = corrector_horizon
        self.lambda_corrector_asymptotic = lambda_corrector_asymptotic
        self.lambda_corrector_quiescence = lambda_corrector_quiescence
        self.corrector_quiescence_margin = corrector_quiescence_margin
        self.corrector_quiescence_cap = corrector_quiescence_cap
        self.encoder_warmup_epochs = encoder_warmup_epochs if self.mode != "train_corrector" else 0
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

        from spwm.evaluation.rollout import RolloutEvaluator

        self.rollout_evaluator = RolloutEvaluator(
            model=self.model,
            device=self.device,
            horizons=self.rollout_horizons,
            num_objects=self.num_objects,
        )

        # ------------------------------------------------------------------
        # Configurazione Ottimizzatori e Disaccoppiamento Gradienti
        # ------------------------------------------------------------------

        if self.mode == "train_predictor":
            # MODALITÀ 1: train_predictor (Fase Nominale)
            # Allena solo Encoder, Predictor/Dynamics p e Probe decoder.
            # Il correttore è disattivato / congelato (nessun gradiente, nessun ottimizzatore).
            if hasattr(self.model, "corrector") and self.model.corrector is not None:
                for p in self.model.corrector.parameters():
                    p.requires_grad = False

            # Step-1: Ottimizzatore Encoder
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
                if "predictor" in n and "sensory_predictor" not in n and "corrector" not in n
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

            # Step-3: Ottimizzatore Probe Cinematico
            probe_params = [
                p
                for n, p in self.model.named_parameters()
                if ("decoder" in n or "probe" in n) and "corrector" not in n
            ]
            self.probe_optimizer = (
                torch.optim.AdamW(probe_params, lr=probe_lr, weight_decay=probe_weight_decay)
                if probe_params
                else None
            )

            # Modulo correttore disattivato
            self.corrector_optimizer = None

        elif self.mode == "train_corrector":
            # MODALITÀ 2: train_corrector (Fase di Stabilizzazione & Drift)
            # Modello Predittore/World Model rigorosamente congelato (requires_grad = False).
            # Allena esclusivamente i parametri del correttore.
            for n, p in self.model.named_parameters():
                if "corrector" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

            # Imposta i sottomoduli del predittore in modalità eval per stabilità
            if hasattr(self.model, "encoder"):
                self.model.encoder.eval()
            if hasattr(self.model, "dynamics"):
                self.model.dynamics.eval()
            if hasattr(self.model, "predictor"):
                self.model.predictor.eval()
            if hasattr(self.model, "sensory_predictor"):
                self.model.sensory_predictor.eval()
            if hasattr(self.model, "physical_decoder"):
                self.model.physical_decoder.eval()

            self.encoder_optimizer = None
            self.predictor_optimizer = None
            self.probe_optimizer = None

            corrector_params = (
                list(self.model.corrector.parameters())
                if hasattr(self.model, "corrector") and self.model.corrector is not None
                else []
            )
            self.corrector_optimizer = (
                torch.optim.AdamW(
                    corrector_params,
                    lr=(corrector_lr if corrector_lr is not None else learning_rate),
                    weight_decay=weight_decay,
                )
                if corrector_params
                else None
            )

        else:  # mode == "joint" (Legacy)
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

            corrector_params = (
                list(self.model.corrector.parameters())
                if hasattr(self.model, "corrector") and self.model.corrector is not None
                else []
            )
            self.corrector_optimizer = (
                torch.optim.AdamW(
                    corrector_params,
                    lr=(corrector_lr if corrector_lr is not None else learning_rate),
                    weight_decay=weight_decay,
                )
                if corrector_params
                else None
            )

        # Verifica di isolamento gradienti a inizio training
        self.verify_gradient_isolation()

        self.decoder_scheduler = None

        self.writer = (
            SummaryWriter(log_dir=str(self.save_dir / "tb"))
            if tensorboard_logging
            else None
        )

        # Ripristino stato Checkpoint & Pareto Tracker
        self.start_epoch = start_epoch
        self.best_val_loss = best_val_loss
        self.history = history if history is not None else []
        
        # Fallback automatico da cronologia con ordinamento specifico per modalità
        if self.history and (self.best_combined_score == float("inf") or self.best_rollout_mae == float("inf")):
            def _selection_key(rec):
                if self.mode == "train_predictor":
                    pos_h1 = rec.get("val_pos_err_h1", rec.get("val_pos_err", float("inf")))
                    vel_h1 = rec.get("val_vel_err_h1", rec.get("val_vel_err", 0.0))
                    pos_h5 = rec.get("val_pos_err_h5", pos_h1)
                    beta_v = getattr(self.loss_fn, "beta_v", 2.0)
                    return pos_h1 + beta_v * vel_h1 + 0.5 * pos_h5
                else:
                    return rec.get("val_dynamic_score", rec.get("combined_score", rec.get("val_rollout_mae", float("inf"))))

            best_entry = min(self.history, key=_selection_key)
            self.best_gate_passed = bool(best_entry.get("val_gate_passed", False))
            self.best_rollout_mae = best_entry.get("val_rollout_mae", float("inf"))
            self.best_drift_ratio = best_entry.get("val_drift_ratio", float("inf"))
            if self.mode == "train_predictor":
                pos_h1 = best_entry.get("val_pos_err_h1", best_entry.get("val_pos_err", float("inf")))
                vel_h1 = best_entry.get("val_vel_err_h1", best_entry.get("val_vel_err", 0.0))
                pos_h5 = best_entry.get("val_pos_err_h5", pos_h1)
                beta_v = getattr(self.loss_fn, "beta_v", 2.0)
                self.best_combined_score = pos_h1 + beta_v * vel_h1 + 0.5 * pos_h5
            else:
                self.best_combined_score = best_entry.get("val_dynamic_score", best_entry.get("combined_score", self.best_rollout_mae))
            self.best_val_pos_err = best_entry.get("val_pos_err", self.best_val_pos_err)
            self.best_val_vel_err = best_entry.get("val_vel_err", self.best_val_vel_err)
            self.best_epoch = best_entry.get("epoch", self.best_epoch)

        if optimizer_state:
            if self.encoder_optimizer and "encoder_optimizer" in optimizer_state:
                self.encoder_optimizer.load_state_dict(optimizer_state["encoder_optimizer"])
            if self.probe_optimizer and "probe_optimizer" in optimizer_state:
                self.probe_optimizer.load_state_dict(optimizer_state["probe_optimizer"])
            if self.predictor_optimizer and "predictor_optimizer" in optimizer_state:
                self.predictor_optimizer.load_state_dict(optimizer_state["predictor_optimizer"])
            if self.corrector_optimizer and "corrector_optimizer" in optimizer_state:
                if optimizer_state["corrector_optimizer"] is not None:
                    self.corrector_optimizer.load_state_dict(optimizer_state["corrector_optimizer"])
            if "best_rollout_mae" in optimizer_state:
                self.best_rollout_mae = optimizer_state["best_rollout_mae"]
            if "best_drift_ratio" in optimizer_state:
                self.best_drift_ratio = optimizer_state["best_drift_ratio"]
            if "best_gate_passed" in optimizer_state:
                self.best_gate_passed = optimizer_state["best_gate_passed"]
            if "best_combined_score" in optimizer_state:
                self.best_combined_score = optimizer_state["best_combined_score"]
            if "best_val_vel_err" in optimizer_state:
                self.best_val_vel_err = optimizer_state["best_val_vel_err"]
            if "best_val_pos_err" in optimizer_state:
                self.best_val_pos_err = optimizer_state["best_val_pos_err"]
            if "best_epoch" in optimizer_state:
                self.best_epoch = optimizer_state["best_epoch"]

    def verify_gradient_isolation(self) -> None:
        """
        Verifies via assertions that gradients of non-involved modules are strictly deactivated.
        """
        if self.mode == "train_predictor":
            # 1. Corrector must be inactive or have no trainable parameters
            if hasattr(self.model, "corrector") and self.model.corrector is not None:
                for name, param in self.model.corrector.named_parameters():
                    assert not param.requires_grad, (
                        f"[train_predictor mode] Gradient leak detected! "
                        f"Corrector parameter '{name}' has requires_grad=True, but should be frozen."
                    )
            assert self.corrector_optimizer is None, (
                "[train_predictor mode] corrector_optimizer must be None."
            )
            # 2. Predictor / World Model must have trainable parameters
            trainable_predictor_params = [
                n for n, p in self.model.named_parameters()
                if "corrector" not in n and p.requires_grad
            ]
            assert len(trainable_predictor_params) > 0, (
                "[train_predictor mode] No trainable parameters found for predictor / world model!"
            )

        elif self.mode == "train_corrector":
            # 1. All non-corrector parameters must be frozen
            for name, param in self.model.named_parameters():
                if "corrector" not in name:
                    assert not param.requires_grad, (
                        f"[train_corrector mode] Gradient leak detected! "
                        f"Non-corrector parameter '{name}' has requires_grad=True, but must be frozen."
                    )
                else:
                    assert param.requires_grad, (
                        f"[train_corrector mode] Corrector parameter '{name}' has requires_grad=False, but must be trainable."
                    )
            # 2. Optimizers for predictor, encoder, probe must be None
            assert self.encoder_optimizer is None, (
                "[train_corrector mode] encoder_optimizer must be None."
            )
            assert self.predictor_optimizer is None, (
                "[train_corrector mode] predictor_optimizer must be None."
            )
            assert self.probe_optimizer is None, (
                "[train_corrector mode] probe_optimizer must be None."
            )
            assert self.corrector_optimizer is not None, (
                "[train_corrector mode] corrector_optimizer must not be None."
            )
            assert hasattr(self.model, "corrector") and self.model.corrector is not None, (
                "[train_corrector mode] Model must have an active corrector module."
            )

    def compute_selection_score(self, val_metrics: Dict[str, float]) -> Tuple[float, str]:
        """Calcola la metrica di selezione del miglior checkpoint in modo specifico per la modalità attiva.
        
        - Modalità train_predictor (Fase Nominale):
          L'obiettivo è l'accuratezza cinematica locale e la fedeltà fisica a breve termine.
          Score = val_pos_err_h1 + 0.5 * val_vel_err_h1 + 0.25 * val_pos_err_h5
        
        - Modalità train_corrector (Fase di Drift & Stabilizzazione):
          L'obiettivo è la soppressione del drift asintotico su orizzonti estesi (H>=25, 50).
          Score = Dynamic Score (Rollout MAE pesato con drift ratio H50/H1)
        """
        if self.mode == "train_predictor":
            h1_pos = val_metrics.get("val_pos_err_h1", val_metrics.get("val_pos_err", 0.0))
            h1_vel = val_metrics.get("val_vel_err_h1", val_metrics.get("val_vel_err", 0.0))
            h5_pos = val_metrics.get("val_pos_err_h5", h1_pos)
            # Local Kinematic Acuity Score
            score = float(h1_pos + 0.5 * h1_vel + 0.25 * h5_pos)
            desc = f"Local Kinematic Error (H1 Pos: {h1_pos:.4f}, H1 Vel: {h1_vel:.4f}, H5 Pos: {h5_pos:.4f})"
            return score, desc
        else:
            # train_corrector: Valutazione asintotica e soppressione del drift a lungo raggio
            mae = float(val_metrics.get("val_rollout_mae", 0.0))
            drift_ratio = float(val_metrics.get("val_drift_ratio", 1.0))
            h50_pos = float(val_metrics.get("val_pos_err_h50", mae))
            # Score dedicato al correttore: MAE complessivo + penalità diretta su H50
            score = float(0.5 * mae + 0.5 * h50_pos)
            desc = f"Corrector Rollout Score (Mean MAE: {mae:.4f}, H50 Pos: {h50_pos:.4f}, Drift: {drift_ratio:.2f}x)"
            return score, desc

    def _print_training_header(self, total_epochs: int) -> None:
        """Visualizza i parametri principali prima dell'avvio."""
        chronicle.log_application_title("SPWM CONTINUOUS ONLINE TRAINER")
        chronicle.log_detail("Device", self.device)
        chronicle.log_detail("Training Mode", self.mode.upper())
        if self.mode == "train_predictor":
            chronicle.log_detail(
                "Phase Objective",
                "⚡ FAST PREDICTOR NOMINAL TRAINING (Local Kinematics & Physical Acuity)",
            )
            chronicle.log_detail(
                "Model Selection Metric",
                "Local Kinematic Score: H1_Pos + 0.5*H1_Vel + 0.25*H5_Pos",
            )
        elif self.mode == "train_corrector":
            chronicle.log_detail(
                "Phase Objective",
                "🐢 SLOW NEUROMORPHIC CORRECTOR (Drift Mitigation & Long-Horizon Stabilization)",
            )
            chronicle.log_detail(
                "Model Selection Metric",
                f"Dynamic Drift Score (Rollout MAE x Drift Ratio, Max Drift Gate: {self.max_drift_ratio:.1f}x)",
            )
        chronicle.log_detail(
            "Epochs",
            f"{self.start_epoch} -> {self.start_epoch + total_epochs - 1}  (Total: {total_epochs})",
        )
        chronicle.log_detail("Learning Rate", f"{self.learning_rate:.2e}")
        chronicle.log_detail("Algorithm", self.learning_algorithm)
        chronicle.log_detail("Checkpoint Dir", str(self.save_dir))
        chronicle.log_detail("Encoder Warmup Epochs", f"{self.encoder_warmup_epochs}")
        chronicle.log_detail("Encoder Optimizer", "Enabled" if self.encoder_optimizer else "Disabled (Frozen)")
        chronicle.log_detail("Probe Optimizer", "Enabled" if self.probe_optimizer else "Disabled (Frozen)")
        chronicle.log_detail("Predictor Optimizer", "Enabled" if self.predictor_optimizer else "Disabled (Frozen)")
        chronicle.log_detail("Corrector Optimizer", "Enabled" if self.corrector_optimizer else "Disabled (Bypassed)")
        sampling_mode = f"Smooth U(1, {self.k_max})" if self.smooth_horizon_sampling else "Curriculum Thresholds"
        chronicle.log_detail("Horizon Mode", sampling_mode)
        chronicle.log_newline()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Esegue una singola epoca di training su tutti i batch dello stream."""
        if self.mode == "train_corrector":
            self.model.eval()
            if hasattr(self.model, "corrector") and self.model.corrector is not None:
                self.model.corrector.train()
        else:
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
            "l_corr": 0.0,
            "l_corr_asymptotic": 0.0,
            "l_corr_quiescence": 0.0,
            "spike_rate": 0.0,
            "grad_norm": 0.0,
        }
        num_batches = 0
        total_grad_norm_accum = 0.0
        is_warmup = (epoch <= self.encoder_warmup_epochs) if self.mode != "train_corrector" else False
        accum_eprop = (self.mode != "train_corrector")

        for batch in self.train_loader:
            # Scheduled sampling continuo: campionamento uniforme stocastico K_t ~ U(1, K_max)
            if self.smooth_horizon_sampling and hasattr(self.loss_fn, "multi_step_horizon"):
                if is_warmup:
                    self.loss_fn.multi_step_horizon = 1
                else:
                    self.loss_fn.multi_step_horizon = random.randint(1, self.k_max)

            events = batch["events"].to(self.device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is not None:
                true_kin = true_kin.to(self.device)

            # Passaggio neuromorfo forward-only con accumulo locale e-prop (solo se non in train_corrector)
            with torch.no_grad():
                out = self.model(
                    events,
                    accumulate_local_updates=accum_eprop,
                    learning_rate=self.learning_rate,
                    target_kinematics=true_kin,
                )

            if accum_eprop and hasattr(self.model, "apply_accumulated_updates"):
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

                    # Two-Scale Velocity Supervision: combinazione convessa (0.5 * k=1 + 0.5 * k=3)
                    l_vel = torch.tensor(0.0, device=self.device)
                    lambda_vel = getattr(self.loss_fn, "lambda_vel", 0.5)
                    if (
                        lambda_vel > 0.0
                        and true_kin is not None
                        and hasattr(self.model, "physical_decoder")
                    ):
                        l_vel_k1 = self.loss_fn.velocity_loss_frozen_probe(
                            pred_z=pred_z,
                            true_kinematics=true_kin,
                            physical_decoder=self.model.physical_decoder,
                            stride_k=1,
                            dt=0.01,
                        )
                        l_vel_k3 = self.loss_fn.velocity_loss_frozen_probe(
                            pred_z=pred_z,
                            true_kinematics=true_kin,
                            physical_decoder=self.model.physical_decoder,
                            stride_k=3,
                            dt=0.01,
                        )
                        l_vel = 0.5 * l_vel_k1 + 0.5 * l_vel_k3

                    # Sparsità differenziabile sugli spike del predittore
                    l_sparse = torch.tensor(0.0, device=self.device)
                    lambda_sparse = getattr(self.loss_fn, "lambda_sparse", 0.5)

                    fast_spk = getattr(out, "fast_spikes", None)
                    slow_spk = getattr(out, "slow_spikes", None)

                    if lambda_sparse > 0.0 and fast_spk is not None and slow_spk is not None:
                        all_spikes = torch.cat([fast_spk, slow_spk], dim=-1)
                        target_sr = getattr(self.loss_fn, "target_spike_rate", 0.10)
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
                    if hasattr(self.loss_fn, "kinematic_loss"):
                        probe_loss = self.loss_fn.kinematic_loss(decoded, true_kin)
                    else:
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

            # ----------------------------------------------------------
            # Step 4: Protected Corrector Optimization (SPWM-v8)
            # ----------------------------------------------------------
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
                    # Congela rigorosamente il modulo predittore rapido per isolare
                    # le sue capacità locali a breve raggio (H <= 10) dai gradienti a lungo termine (H >= 25)
                    with _frozen(self.model.predictor):
                        latents = out.latent_states.detach()
                        B_lat, T_tot, D_lat = latents.shape
                        H_corr = min(self.corrector_horizon, T_tot - 2)

                        if H_corr >= 5:
                            max_start = max(1, T_tot - H_corr - 1)
                            t0 = random.randint(0, max_start - 1) if max_start > 1 else 0

                            z_init = latents[:, t0].detach()
                            target_rollout = latents[:, t0 + 1 : t0 + 1 + H_corr].detach()

                            pred_rollout, corr_spikes = self.model.predict_rollout(
                                initial_latent=z_init,
                                horizon=H_corr,
                                use_corrector=True,
                            )

                            # 1. Asymptotic Trajectory Stabilization Loss (Latent MSE + Kinematic Position Divergence)
                            l_asymptotic_latent = nn.functional.mse_loss(pred_rollout, target_rollout)
                            l_asymptotic_kin = torch.tensor(0.0, device=self.device)
                            if true_kin is not None and hasattr(self.model, "physical_decoder"):
                                decoded_rollout = self.model.physical_decoder(pred_rollout)
                                target_kin_rollout = true_kin[:, t0 + 1 : t0 + 1 + H_corr]
                                # Penalizza direttamente l'errore di posizione decodificato su orizzonti lunghi
                                l_asymptotic_kin = nn.functional.mse_loss(decoded_rollout, target_kin_rollout)

                            l_asymptotic = l_asymptotic_latent + 2.0 * l_asymptotic_kin

                            # 2. Quiescent Sparsity Penalty on Corrector Spikes (Hinge / Dead-Zone Margin with Upper Capping)
                            # Zero penalty for corrective activity below margin (e.g. 15%), and capped upper bound against chattering
                            if corr_spikes is not None:
                                spike_activity = corr_spikes.mean()
                                l_quiescence = torch.clamp(
                                    nn.functional.relu(spike_activity - self.corrector_quiescence_margin),
                                    max=self.corrector_quiescence_cap,
                                    min=0.0,
                                )
                            else:
                                l_quiescence = torch.tensor(0.0, device=self.device)

                            loss_corr = (
                                self.lambda_corrector_asymptotic * l_asymptotic
                                + self.lambda_corrector_quiescence * l_quiescence
                            )

                            loss_corr.backward()

                            if self.grad_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.corrector_optimizer.param_groups[0]["params"],
                                    self.grad_clip_norm,
                                )
                            self.corrector_optimizer.step()
                            corr_loss_val = loss_corr.item()
                            epoch_losses["l_corr_asymptotic"] += l_asymptotic.item()
                            epoch_losses["l_corr_quiescence"] += l_quiescence.item()

                epoch_losses["l_corr"] += corr_loss_val

            if hasattr(out, "mean_spike_rate"):
                epoch_losses["spike_rate"] += out.mean_spike_rate.item()

            # Aggregazione Loss complessiva
            lambda_probe = getattr(self.loss_fn, "lambda_probe", 2.0)
            epoch_losses["total_loss"] += (
                pred_loss_val + enc_loss_val + (lambda_probe * probe_loss_val) + corr_loss_val
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
                if hasattr(self.loss_fn, "kinematic_loss"):
                    probe_loss = self.loss_fn.kinematic_loss(out.decoded_kinematics, true_kin)
                else:
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

        # ------------------------------------------------------------------
        # Autonomous Multi-Step Rollout Evaluation on Validation Set
        # ------------------------------------------------------------------
        rollout_res = self.rollout_evaluator.evaluate_dataset(self.val_loader)
        val_losses["val_rollout_mae"] = rollout_res.rollout_mae
        val_losses["val_rollout_vel_mae"] = rollout_res.rollout_vel_mae
        val_losses["val_drift_ratio"] = rollout_res.drift_ratio
        val_losses["val_gate_passed"] = 1.0 if rollout_res.is_gate_passed(self.max_drift_ratio) else 0.0
        val_losses["val_dynamic_score"] = rollout_res.dynamic_score

        for h, pos_h in rollout_res.position_error_per_horizon.items():
            val_losses[f"val_pos_err_h{h}"] = pos_h
        for h, vel_h in rollout_res.velocity_error_per_horizon.items():
            val_losses[f"val_vel_err_h{h}"] = vel_h

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
                # Curriculum Adattivo a Soglie (se disattivato lo smooth sampling)
                # ------------------------------------------------------
                if (
                    self.curriculum_multi_step
                    and not self.smooth_horizon_sampling
                    and hasattr(self.loss_fn, "multi_step_horizon")
                ):
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
                # Selezione del Miglior Modello: Specifico per Modalità
                # ------------------------------------------------------
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
                    # Salva il checkpoint del modello completo e del modulo target per la modalità
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
                        "best_rollout_mae": self.best_rollout_mae,
                        "best_drift_ratio": self.best_drift_ratio,
                        "best_gate_passed": self.best_gate_passed,
                        "best_epoch": self.best_epoch,
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
                            "corrector_optimizer": self.corrector_optimizer.state_dict()
                            if self.corrector_optimizer
                            else None,
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
                    current_k = (
                        f"~U(1,{self.k_max})"
                        if (self.smooth_horizon_sampling and epoch > self.encoder_warmup_epochs)
                        else getattr(self.loss_fn, "multi_step_horizon", 1)
                    )
                    warmup_tag = " [WARMUP]" if epoch <= self.encoder_warmup_epochs else ""
                    
                    if self.mode == "train_predictor":
                        header = (
                            f"Epoch [{epoch:03d}/{end_epoch:03d}] [PREDICTOR-PHASE] (K_fast={current_k}){warmup_tag}"
                            f"  Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag}"
                        )
                        chronicle.log_section_header(header)

                        # FAST PREDICTOR & REPRESENTATION
                        chronicle.log_detail(
                            "⚡ Fast Predictor Optimization (Nominal)",
                            (
                                f"L_1step(k=1): {train_metrics['l_pred']:.5f} | "
                                f"L_multi(k={current_k}): {train_metrics.get('l_multi', 0.0):.5f} | "
                                f"L_vel(k=1,3): {train_metrics.get('l_vel', 0.0):.5f} | "
                                f"L_var: {train_metrics.get('l_var', 0.0):.5f} | "
                                f"L_sparse: {train_metrics.get('l_sparse', 0.0):.5f}"
                            ),
                            indent_level=1,
                        )
                        chronicle.log_detail(
                            "🎯 Representation & Probe",
                            (
                                f"L_coord(SpatialSoftmax): {train_metrics['l_coord']:.5f} | "
                                f"L_probe(Kinematics): {train_metrics['l_probe']:.5f}"
                            ),
                            indent_level=1,
                        )
                        chronicle.log_detail(
                            "📊 Validation Kinematics & Rollout",
                            (
                                f"Val Loss: {val_metrics['val_total_loss']:.5f} (Pred: {val_metrics['val_l_pred']:.5f}, Multi: {val_metrics.get('val_l_multi', 0.0):.5f}) | "
                                f"TF Pos: {current_pos_h1:.4f} | TF Vel: {current_vel_h1:.4f}\n"
                                f"    Rollout Horizons -> H1: {val_metrics.get('val_pos_err_h1', 0.0):.4f} | "
                                f"H5: {val_metrics.get('val_pos_err_h5', 0.0):.4f} | "
                                f"H10: {val_metrics.get('val_pos_err_h10', 0.0):.4f}"
                            ),
                            indent_level=1,
                        )
                        chronicle.log_detail(
                            "🎯 Model Selection (Kinematic Acuity)",
                            (
                                f"Current Score: {curr_score:.5f} ({score_description})\n"
                                f"    Best Selection: Score={self.best_combined_score:.5f} (H1_Pos={self.best_val_pos_err:.4f}, H1_Vel={self.best_val_vel_err:.4f} @ Ep {self.best_epoch})"
                            ),
                            indent_level=1,
                        )

                    elif self.mode == "train_corrector":
                        corr_cad = getattr(self.model.corrector, 'cadence', 5) if hasattr(self.model, "corrector") and self.model.corrector else 5
                        header = (
                            f"Epoch [{epoch:03d}/{end_epoch:03d}] [CORRECTOR-PHASE] (H_slow={self.corrector_horizon}, Δ={corr_cad})"
                            f"  Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag}"
                        )
                        chronicle.log_section_header(header)

                        chronicle.log_detail(
                            "🔒 Predictor Core & Representation",
                            "FROZEN (requires_grad=False, eval mode). No kinematic gradient interference.",
                            indent_level=1,
                        )
                        chronicle.log_detail(
                            "🐢 Slow Neuromorphic Corrector (ALIF)",
                            (
                                f"Total L_corr: {train_metrics.get('l_corr', 0.0):.5f} | "
                                f"L_asymptotic(H={self.corrector_horizon}, Δ={corr_cad}): {train_metrics.get('l_corr_asymptotic', 0.0):.5f} | "
                                f"L_quiescence: {train_metrics.get('l_corr_quiescence', 0.0):.5f}"
                            ),
                            indent_level=1,
                        )
                        h_pos_str = " | ".join([f"H{h}: {val_metrics.get(f'val_pos_err_h{h}', 0.0):.4f}" for h in self.rollout_horizons])
                        chronicle.log_detail(
                            "📈 Autonomous Rollout Trajectory (Drift Stabilization)",
                            (
                                f"Rollout Mean MAE: {curr_rollout_mae:.5f} | Drift Ratio (H50/H1): {curr_drift_ratio:.2f}x | "
                                f"Spike Rate: {val_metrics.get('val_spike_rate', 0.0):.3f}\n"
                                f"    Horizon Trajectory -> {h_pos_str}"
                            ),
                            indent_level=1,
                        )
                        chronicle.log_detail(
                            "🎯 Model Selection (Long-Horizon Dynamic Drift Score)",
                            (
                                f"Current Score: {curr_score:.5f} ({score_description})\n"
                                f"    Best Selection: Score={self.best_combined_score:.5f} (Drift Ratio={self.best_drift_ratio:.2f}x, Mean MAE={self.best_rollout_mae:.4f} @ Ep {self.best_epoch})"
                            ),
                            indent_level=1,
                        )
                    else:
                        # Fallback generic logging
                        header = (
                            f"Epoch [{epoch:03d}/{end_epoch:03d}]"
                            f"  Time: {epoch_duration:5.1f}s | Total: {total_elapsed / 60:4.1f}m{best_tag}"
                        )
                        chronicle.log_section_header(header)
                        chronicle.log_detail(
                            "📊 Losses & Metrics",
                            f"Train Loss: {train_metrics.get('total_loss', 0.0):.5f} | Val Loss: {val_metrics.get('val_total_loss', 0.0):.5f}",
                            indent_level=1,
                        )

                    # --- DEBUG REMINDER (EVERY 10 EPOCHS) ---
                    if epoch % 10 == 0:
                        if self.mode == "train_predictor":
                            debug_reminders = (
                                "[SPWM Predictor Training Guide Reminder]\n"
                                "  • L_1step: Next-latent MSE (z_t -> z_t+1) guaranteeing clean local transition dynamics.\n"
                                f"  • L_multi (k={current_k}): Autoregressive rollout loss over horizon k for multi-step temporal consistency.\n"
                                "  • L_vel (k=1, k=3): Differentiable velocity supervision via frozen probe grounding latent momentum p.\n"
                                "  • L_var & L_sparse: Representation variance protection and neuromorphic sparsity.\n"
                                "  • Selection Criterion: Local kinematic acuity (H1_Pos + 0.5*H1_Vel + 0.25*H5_Pos) on nominal physics."
                            )
                        else:
                            debug_reminders = (
                                "[SPWM Corrector Training Guide Reminder]\n"
                                f"  • L_asymptotic (H={self.corrector_horizon}, Δ={getattr(self.model.corrector, 'cadence', 5) if hasattr(self.model, 'corrector') and self.model.corrector else 'N/A'}): "
                                "Optimizes slow ALIF population on long rollout horizons with frozen predictor core to suppress drift.\n"
                                "  • L_quiescence: Sparsity penalty keeping corrector silent on nominal trajectories.\n"
                                "  • Selection Criterion: Dynamic drift score penalizing error growth over long horizons (H25, H50, H100)."
                            )
                        if hasattr(chronicle, "log_debug"):
                            chronicle.log_debug(debug_reminders)
                        else:
                            chronicle.log_info(f"[DEBUG] {debug_reminders}")

                    chronicle.log_newline()

        except KeyboardInterrupt:
            chronicle.log_warning(
                "[INTERRUPT] Training interrotto manualmente dall'utente."
            )
            try:
                chronicle.log_info("Valutazione dello stato attuale in corso...")
                val_metrics = self.evaluate()
                c_score, _ = self.compute_selection_score(val_metrics)
                c_drift = val_metrics.get("val_drift_ratio", 1.0)
                c_mae = val_metrics.get("val_rollout_mae", 0.0)
                c_gate = bool(val_metrics.get("val_gate_passed", 0.0) > 0.5)

                is_best_interrupt = False
                if c_score < self.best_combined_score:
                    is_best_interrupt = True

                if is_best_interrupt:
                    self.best_gate_passed = c_gate
                    self.best_rollout_mae = c_mae
                    self.best_drift_ratio = c_drift
                    self.best_combined_score = c_score
                    self.best_val_pos_err = val_metrics.get("val_pos_err_h1", val_metrics.get("val_pos_err", float("inf")))
                    self.best_val_vel_err = val_metrics.get("val_vel_err_h1", val_metrics.get("val_vel_err", float("inf")))
                    model_path = self.save_dir / "model.pt"
                    torch.save(self.model.state_dict(), model_path)
                    if self.mode == "train_corrector":
                        torch.save(self.model.state_dict(), self.save_dir / "model_corrector.pt")
                        if hasattr(self.model, "corrector") and self.model.corrector is not None:
                            torch.save(self.model.corrector.state_dict(), self.save_dir / "corrector_weights.pt")
                    elif self.mode == "train_predictor":
                        torch.save(self.model.state_dict(), self.save_dir / "model_predictor.pt")
                    chronicle.log_success(
                        f"Nuovo miglior modello salvato su interrupt in: {model_path}"
                    )
            except Exception as e:
                chronicle.log_error(f"Errore post-interrupt: {e}")
                model_path = self.save_dir / "model.pt"
                torch.save(self.model.state_dict(), model_path)
                if self.mode == "train_corrector":
                    torch.save(self.model.state_dict(), self.save_dir / "model_corrector.pt")
                    if hasattr(self.model, "corrector") and self.model.corrector is not None:
                        torch.save(self.model.corrector.state_dict(), self.save_dir / "corrector_weights.pt")
                elif self.mode == "train_predictor":
                    torch.save(self.model.state_dict(), self.save_dir / "model_predictor.pt")

        self.save_training_log()
        chronicle.log_success(
            f"Session complete. Logs saved in '{self.save_dir}'. "
            f"Best Selection Score: {self.best_combined_score:.5f} @ Epoch {self.best_epoch}"
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
        fieldnames = []
        for rec in self.history:
            for k in rec.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
                    f"val_rollout_mae={rec.get('val_rollout_mae', 0.0):.4f}",
                    f"val_drift_ratio={rec.get('val_drift_ratio', 0.0):.2f}",
                    f"val_gate_passed={rec.get('val_gate_passed', 0.0)}",
                    f"val_dynamic_score={rec.get('val_dynamic_score', 0.0):.4f}",
                    f"combined_score={rec.get('combined_score'):.4f}",
                ]
                f.write("  Losses: " + ", ".join(loss_items) + "\n")
                f.write(
                    f"  GradNormOverall={rec.get('grad_norm'):.4f}, "
                    f"GradNormRecent10={rec.get('grad_norm_recent'):.4f}\n\n"
                )
