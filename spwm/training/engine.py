"""
BaseTrainer & Trainer Engine for SPWM (Layer 2 — Training).

Single source of truth for:
- Epoch lifecycle orchestration (`fit`, `train_epoch`, `evaluate`)
- Scoped `strict=True` checkpoint management (`state_dict`, `save_checkpoint`, `load_checkpoint`)
- Gradient isolation verification (`verify_gradient_isolation`)

Checkpoint resolution helpers and replay/EMA utilities live in
`spwm/training/checkpoints.py` and `spwm/training/replay_buffer.py`, and are
re-exported here for backward compatibility.

Zero upward imports into `spwm.evaluation` and zero silent `getattr` or `strict=False` fallbacks.
"""

from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import chronicle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.contracts.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    LatentNeuronConfig,
    LossConfig,
    MemoryConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from spwm.contracts.protocols import (
    PhaseStrategy,
    StatefulTier2EvaluatorProtocol,
    Tier2EvaluatorProtocol,
)
from spwm.contracts.states import Tier2EvalResult
from spwm.training.checkpoints import (
    _extract_submodel_state,
    enforce_phase_prerequisites,
    load_representation_checkpoint,
    resolve_closed_loop_buffer,
    resolve_version_checkpoint,
)
from spwm.training.losses import SPWMLoss
from spwm.training.noise import LatchCurriculumState, compute_dagger_ratio
from spwm.training.phases import PredictorPhaseHandler
from spwm.training.replay_buffer import ClosedLoopBuffer, EMATracker


def _sanitize_checkpoint_value(obj: Any, path: str = "checkpoint") -> Any:
    """Recursively ensure checkpoint payloads contain strictly plain primitives and torch.Tensor."""
    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.requires_grad else obj
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        clean_dict: Dict[Any, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, (str, int, float, bool)):
                raise TypeError(
                    f"Checkpoint key of unsupported type {type(k).__name__} at '{path}'. "
                    "Only plain primitive keys are allowed."
                )
            clean_dict[k] = _sanitize_checkpoint_value(v, f"{path}.{k}")
        return clean_dict
    if isinstance(obj, list):
        return [_sanitize_checkpoint_value(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(_sanitize_checkpoint_value(v, f"{path}[{i}]") for i, v in enumerate(obj))
    raise TypeError(
        f"Non-serializable object of type {type(obj).__name__} at '{path}'. "
        "Checkpoints may only contain plain Python primitives and torch.Tensor objects."
    )


class BaseTrainer:
    """Standardized, modular training engine delegating phase mechanics to `PhaseStrategy`."""

    def __init__(
        self,
        model: nn.Module,
        cfg: ExperimentConfig,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[SPWMLoss] = None,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Union[str, Path] = "results/default_run",
        start_epoch: int = 1,
        best_score: float = float("inf"),
        best_epoch: Optional[int] = None,
        history: Optional[List[Dict[str, float]]] = None,
        auto_enforce_checkpoints: bool = False,
        results_root: Union[str, Path] = "results",
        tier2_evaluator: Optional[Tier2EvaluatorProtocol] = None,
        closed_loop_buffer: Optional[Union[str, Path]] = None,
        artifact_callback: Optional[Callable[["BaseTrainer"], None]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.results_root = Path(results_root)
        self.tier2_evaluator = tier2_evaluator
        self.artifact_callback = artifact_callback
        self.tier2_frequency = cfg.training.tier2_frequency
        self.tier2_candidate_selection = cfg.training.tier2_candidate_selection
        self.tier2_patience = cfg.training.tier2_patience
        self.tier2_min_delta = cfg.training.tier2_min_delta
        self.metrics_storage = cfg.training.metrics_storage
        self.classic_patience = cfg.training.classic_patience
        self.tier2_run_count = 0
        self.ema_decay = cfg.training.ema_decay

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.ema_tracker: Optional[EMATracker] = (
            EMATracker(model=self.model, decay=self.ema_decay, device=self.device)
            if self.tier2_candidate_selection == "ema"
            else None
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if loss_fn is None:
            self.loss_fn = SPWMLoss.from_config(cfg.loss).to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)

        self.mode = cfg.training.mode.lower()
        if auto_enforce_checkpoints and self.mode == "train_m1":
            enforce_phase_prerequisites(
                self.model, self.cfg, device=self.device, results_root=self.results_root
            )

        self.handler: PhaseStrategy = PredictorPhaseHandler(
            model=self.model, cfg=self.cfg, loss_fn=self.loss_fn, device=self.device
        )

        self.replay_buffer: Optional[ClosedLoopBuffer] = (
            ClosedLoopBuffer(
                capacity=50_000,
                max_eval_generations=self.cfg.training.max_eval_generations,
            )
            if self.mode in ("train_m1", "m1")
            else None
        )
        if self.replay_buffer is not None and closed_loop_buffer is not None:
            resolved_buffer = resolve_closed_loop_buffer(
                version_name=self.cfg.project.name,
                phase_mode=self.mode,
                explicit_path=closed_loop_buffer,
                save_dir=self.save_dir,
                results_root=self.results_root,
                is_resuming=False,
            )
            if resolved_buffer is not None:
                self.replay_buffer.load(resolved_buffer)
                chronicle.log_info(
                    f"Loaded explicit ClosedLoopBuffer with {len(self.replay_buffer)} transitions "
                    f"({len(self.replay_buffer.trajectories)} trajectories) from {resolved_buffer}"
                )

        self.start_epoch = start_epoch
        self.best_score = best_score
        self.best_epoch = best_epoch
        self.history: List[Dict[str, float]] = history or []
        self.curriculum = LatchCurriculumState(
            tier2_frequency=self.tier2_frequency,
            noise_ramp_epochs=self.cfg.training.mppi_noise_ramp_epochs,
            target_alpha=self.cfg.training.mppi_noise_target_alpha,
            phase1_loss_threshold=self.cfg.training.phase1_loss_threshold,
            phase2_loss_threshold=self.cfg.training.phase2_loss_threshold,
        )
        self.current_curriculum_phase: int = 1

    @property
    def predictor_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.handler.predictor_optimizer

    @property
    def representation_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return None

    @property
    def encoder_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return None

    @property
    def probe_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return None

    def verify_gradient_isolation(self) -> None:
        """Delegates gradient isolation assertion to the active PhaseStrategy."""
        self.handler.verify_gradient_isolation()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Executes a single epoch over train_loader using the 4-Phase Latch Curriculum and active PhaseStrategy."""
        if self.train_loader is None:
            raise RuntimeError("Cannot call train_epoch when train_loader is None.")
        train_accum: Dict[str, float] = {}
        batch_count = 0
        min_buffer_samples = 1
        cl_trajs_epoch = 0
        cl_trans_epoch = 0

        if self.mode in ("train_m1", "m1"):
            active_phase, w_t, alpha, dagger_ratio = self.curriculum.begin_epoch()
            self.current_curriculum_phase = active_phase
            target_lr = 0.00005 if active_phase == 4 else self.cfg.training.learning_rate
            if self.handler.predictor_optimizer is not None:
                for pg in self.handler.predictor_optimizer.param_groups:
                    pg["lr"] = target_lr
            if active_phase == 4 and dagger_ratio > 0.0 and self.replay_buffer is not None and len(self.replay_buffer.trajectories) > 0:
                buf_nat = [t.trajectory_id for t in self.replay_buffer.trajectories if t.is_expert is False]
                buf_exp = [t.trajectory_id for t in self.replay_buffer.trajectories if t.is_expert is True]
                buf_untracked = [
                    t.trajectory_id
                    for t in self.replay_buffer.trajectories
                    if t.is_expert not in (True, False)
                ]
                untracked_desc = (
                    f" | Unannotated Legacy Trajectories (excluded from natural/expert breakdown): {buf_untracked}"
                    if buf_untracked
                    else ""
                )
                chronicle.log_info(
                    f"[DAgger Training] Epoch {epoch} training on buffer ({len(self.replay_buffer.trajectories)} trajs, ratio={dagger_ratio:.2f}) | "
                    f"Verified Natural IDs: {buf_nat} | Verified Expert IDs: {buf_exp}{untracked_desc}"
                )
        else:
            active_phase, w_t, alpha, dagger_ratio = 1, 0.60, 0.0, 0.0

        for raw_batch in self.train_loader:
            batch = dict(raw_batch)
            if self.mode in ("train_m1", "m1"):
                batch["curriculum_phase"] = active_phase
                batch["curriculum_w_t"] = w_t
                batch["curriculum_alpha"] = alpha

            if (
                active_phase == 4
                and dagger_ratio > 0.0
                and self.replay_buffer is not None
                and len(self.replay_buffer) >= min_buffer_samples
                and self.mode in ("train_m1", "m1")
            ):
                b_size = batch["events"].shape[0] if "events" in batch else 16
                dagger_size = max(1, int(round(b_size * dagger_ratio)))
                open_size = max(1, b_size - dagger_size)
                for key in ("events", "actions", "flat_kinematics", "phase_flags", "joint_positions", "joint_velocities", "trajectory_ids"):
                    if key in batch and isinstance(batch[key], torch.Tensor) and batch[key].shape[0] == b_size:
                        batch[key] = batch[key][:open_size]

                cl_batch = self.replay_buffer.sample_trajectories(
                    dagger_size, horizon=self.cfg.training.phase_horizon, device=self.device
                )
                batch["closed_loop"] = cl_batch
                n_t = cl_batch["actions"].shape[0]
                h_t = cl_batch["actions"].shape[1]
                cl_trajs_epoch += n_t
                cl_trans_epoch += (n_t * h_t)
            else:
                batch.pop("closed_loop", None)

            step_metrics = self.handler.train_step(batch, epoch=epoch)
            for k, v in step_metrics.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            batch_count += 1
            if self.ema_tracker is not None:
                self.ema_tracker.update(self.model)

        epoch_metrics = {k: v / max(1, batch_count) for k, v in train_accum.items()}
        if self.mode in ("train_m1", "m1"):
            prev_phase = self.current_curriculum_phase

            if "train_loss_m1_latent" in epoch_metrics:
                loss_m1_latent = float(epoch_metrics["train_loss_m1_latent"])
            else:
                loss_m1_latent = epoch_metrics.get("train_loss_position", float("inf")) + epoch_metrics.get("train_loss_velocity", float("inf"))
            next_phase = self.curriculum.end_epoch(loss_m1_latent)

            self.current_curriculum_phase = next_phase
            if next_phase > prev_phase:
                chronicle.log_info(
                    f"[Latch Curriculum] Irreversible transition triggered at Epoch {epoch}: "
                    f"Phase {prev_phase} -> Phase {next_phase} (loss_m1_latent={loss_m1_latent:.5f})."
                )
                self.save_checkpoint(epoch, is_best=False, phase_label=f"checkpoint_phase_{prev_phase}")

        epoch_metrics["curriculum_phase"] = float(active_phase)
        epoch_metrics["next_curriculum_phase"] = float(self.current_curriculum_phase)
        epoch_metrics["active_perception_w_t"] = float(w_t)
        epoch_metrics["train_mppi_noise_alpha"] = float(alpha)
        epoch_metrics["closed_loop_trajs_epoch"] = float(cl_trajs_epoch)
        epoch_metrics["closed_loop_trans_epoch"] = float(cl_trans_epoch)
        epoch_metrics["closed_loop_buffer_trajs"] = float(len(self.replay_buffer.trajectories)) if self.replay_buffer is not None else 0.0
        epoch_metrics["closed_loop_buffer_total"] = float(len(self.replay_buffer)) if self.replay_buffer is not None else 0.0
        epoch_metrics["dagger_ratio"] = float(dagger_ratio)
        return epoch_metrics

    def _fit_single_phase(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        total_epochs = epochs if epochs is not None else self.cfg.training.epochs
        end_epoch = self.start_epoch + total_epochs - 1
        max_no_improve = patience if patience is not None else self.cfg.training.early_stopping_patience

        # For Tier 2, early stopping is governed strictly by consecutive Tier 2 checks
        if self.tier2_evaluator is not None:
            if patience is not None:
                tier2_patience_limit = max(1, patience // self.tier2_frequency) if patience >= self.tier2_frequency else max(1, patience)
            else:
                tier2_patience_limit = self.tier2_patience
        else:
            tier2_patience_limit = max_no_improve

        chronicle.log_application_title(
            f"Starting Training: {self.cfg.project.name} [{self.mode.upper()}]"
        )
        chronicle.log_detail("Device", str(self.device))
        if self.tier2_evaluator is not None:
            chronicle.log_detail(
                "Epochs",
                f"{self.start_epoch} -> {end_epoch} (Tier 2 Patience: {tier2_patience_limit} checks / "
                f"{tier2_patience_limit * self.tier2_frequency} epochs, min_delta={self.tier2_min_delta})",
            )
            chronicle.log_detail("Tier 2 Frequency", f"Every {self.tier2_frequency} epochs")
            chronicle.log_detail("Candidate Selection", f"{self.tier2_candidate_selection} (EMA decay: {self.ema_decay})")
        else:
            chronicle.log_detail("Epochs", f"{self.start_epoch} -> {end_epoch} (Patience: {max_no_improve} epochs)")
        chronicle.log_detail("Save Directory", str(self.save_dir))

        epochs_without_improvement = 0
        tier2_checks_without_improvement = 0
        self.tier2_run_count = 0
        start_time = time.time()
        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_ep_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader)
                tier1_loss = self.handler.get_selection_score(val_metrics)

                is_tier2_epoch = (self.tier2_evaluator is not None) and (epoch % self.tier2_frequency == 0)
                if self.tier2_evaluator is not None:
                    if is_tier2_epoch:
                        current_epoch_state = None
                        if self.tier2_candidate_selection == "ema" and self.ema_tracker is not None:
                            current_epoch_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                            self.ema_tracker.copy_to(self.model)
                            chronicle.log_info(
                                f"Tier 2 Evaluating EMA model weights (Epoch {epoch}, decay={self.ema_decay:.3f})."
                            )
                        elif self.tier2_candidate_selection == "latest":
                            chronicle.log_info(
                                f"Tier 2 Evaluating latest model weights (Epoch {epoch})."
                            )

                        active_curric_phase = int(train_metrics["curriculum_phase"])
                        sig = inspect.signature(self.tier2_evaluator.__call__)
                        if "curriculum_phase" in sig.parameters or any(
                            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                        ):
                            raw_tier2 = self.tier2_evaluator(
                                self.model, epoch, curriculum_phase=active_curric_phase
                            )
                        else:
                            raw_tier2 = self.tier2_evaluator(self.model, epoch)

                        if isinstance(raw_tier2, Tier2EvalResult):
                            cl_trajectories = raw_tier2.trajectories
                            cl_transitions = raw_tier2.transitions
                            tier2_metrics = raw_tier2.to_metrics_dict()
                            num_mppi_eval = int(raw_tier2.mppi_evaluated_trajectories)
                            num_expert_eval = int(raw_tier2.mppi_expert_trajectories)
                            expert_blocks = list(raw_tier2.mppi_expert_blocks)
                            expert_ids = list(raw_tier2.mppi_expert_trajectory_ids)
                            natural_blocks = list(raw_tier2.mppi_natural_blocks)
                            natural_ids = list(raw_tier2.mppi_natural_trajectory_ids)
                        else:
                            tier2_metrics = dict(raw_tier2)
                            cl_trajectories = tier2_metrics.pop("trajectories", None)
                            cl_transitions = tier2_metrics.pop("transitions", None)
                            if "mppi_evaluated_trajectories" not in tier2_metrics:
                                raise KeyError(
                                    "Tier 2 evaluator output is missing contractually required 'mppi_evaluated_trajectories'. "
                                    "Cannot proceed with unverified trajectory counts."
                                )
                            if "mppi_score" not in tier2_metrics:
                                raise KeyError(
                                    "Tier 2 evaluator output is missing contractually required 'mppi_score'. "
                                    "Cannot proceed with unverified evaluation score."
                                )
                            num_mppi_eval = int(tier2_metrics["mppi_evaluated_trajectories"])
                            num_expert_eval = int(tier2_metrics["mppi_expert_trajectories"]) if "mppi_expert_trajectories" in tier2_metrics else 0
                            expert_blocks = list(tier2_metrics["mppi_expert_blocks"]) if "mppi_expert_blocks" in tier2_metrics else []
                            expert_ids = list(tier2_metrics["mppi_expert_trajectory_ids"]) if "mppi_expert_trajectory_ids" in tier2_metrics else []
                            natural_blocks = list(tier2_metrics["mppi_natural_blocks"]) if "mppi_natural_blocks" in tier2_metrics else []
                            natural_ids = list(tier2_metrics["mppi_natural_trajectory_ids"]) if "mppi_natural_trajectory_ids" in tier2_metrics else []

                        if self.replay_buffer is not None:
                            if cl_trajectories is not None:
                                self.replay_buffer.push_trajectories(cl_trajectories)
                            elif cl_transitions is not None:
                                self.replay_buffer.push(cl_transitions)
                            self.replay_buffer.save(self.save_dir / "closed_loop_buffer.pt")
                            chronicle.log_info(
                                f"[DAgger Buffer] Training set updated with {len(self.replay_buffer.trajectories)} trajectories "
                                f"({len(self.replay_buffer)} total transitions) | "
                                f"Natural [{num_mppi_eval}]: IDs={natural_ids} ({natural_blocks}) | "
                                f"Expert [{num_expert_eval}]: IDs={expert_ids} ({expert_blocks})."
                            )
                        val_metrics.update(tier2_metrics)
                        score = float(tier2_metrics["mppi_score"])
                        is_full_subset = (
                            num_mppi_eval >= len(cl_trajectories)
                            if cl_trajectories is not None
                            else num_expert_eval == 0
                        )
                        if is_full_subset:
                            is_best = score < (self.best_score - self.tier2_min_delta)
                            if is_best:
                                self.best_score = score
                                self.best_epoch = epoch
                                epochs_without_improvement = 0
                                tier2_checks_without_improvement = 0
                                self.save_checkpoint(epoch, is_best=True)
                            else:
                                tier2_checks_without_improvement += 1
                                epochs_without_improvement += self.tier2_frequency
                        else:
                            is_best = False
                            chronicle.log_info(
                                f"[Tier 2 Selection] Skipping best_model.pt comparison at Epoch {epoch}: "
                                f"partial natural subset ({num_mppi_eval}/{num_mppi_eval + num_expert_eval} trajectories; "
                                f"{num_expert_eval} in Expert mode)."
                            )

                        # Restore current training weights so optimization continues forward
                        if current_epoch_state is not None:
                            self.model.load_state_dict(current_epoch_state, strict=True)
                    else:
                        score = self.best_score
                        is_best = False
                else:
                    score = tier1_loss
                    is_best = score < self.best_score
                    if is_best:
                        self.best_score = score
                        self.best_epoch = epoch
                        epochs_without_improvement = 0
                        self.save_checkpoint(epoch, is_best=True)
                    else:
                        epochs_without_improvement += 1

                ep_duration = time.time() - t_ep_start
                total_elapsed = time.time() - start_time

                log_entry = {
                    "epoch": epoch,
                    "phase": self.mode,
                    **train_metrics,
                    **val_metrics,
                    "score": score,
                    "is_best": is_best,
                    "epochs_without_improvement": epochs_without_improvement,
                    "tier2_checks_without_improvement": tier2_checks_without_improvement if self.tier2_evaluator is not None else epochs_without_improvement,
                    "duration": ep_duration,
                }
                self.history.append(log_entry)
                self.save_checkpoint(epoch, is_best=False)

                self.handler.log_epoch(
                    epoch=epoch,
                    end_epoch=end_epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch_duration=ep_duration,
                    total_elapsed=total_elapsed,
                    is_best=is_best,
                    best_epoch=self.best_epoch,
                    best_score=self.best_score,
                )

                if is_tier2_epoch:
                    self.tier2_run_count += 1
                    if self.tier2_run_count % self.metrics_storage == 0:
                        chronicle.log_info(
                            f"[Metrics Storage] Triggering conclusion metrics and artifacts at epoch {epoch} "
                            f"(Tier 2 run #{self.tier2_run_count}, metrics_storage={self.metrics_storage})."
                        )
                        self.store_conclusion_metrics()

                # Early stopping / patience evaluation: strictly driven by the active validation tier
                if self.tier2_evaluator is not None:
                    if is_tier2_epoch and tier2_checks_without_improvement >= tier2_patience_limit:
                        if self.classic_patience:
                            chronicle.log_info(
                                f"Early stopping triggered in [{self.mode.upper()}] after "
                                f"{tier2_checks_without_improvement} consecutive Tier 2 evaluations ({epochs_without_improvement} epochs) "
                                f"without MPPI structural improvement (Best Score: {self.best_score:.5f} @ Epoch {self.best_epoch})."
                            )
                            break
                        else:
                            chronicle.log_info(
                                f"Tier 2 patience threshold reached ({tier2_checks_without_improvement} checks without improvement) "
                                f"in [{self.mode.upper()}]. Classic patience is deactivated; storing conclusion metrics and artifacts and continuing execution."
                            )
                            self.store_conclusion_metrics()
                else:
                    if epochs_without_improvement >= max_no_improve:
                        if self.classic_patience:
                            chronicle.log_info(
                                f"Early stopping triggered in [{self.mode.upper()}] after {epochs_without_improvement} "
                                f"epochs without validation improvement (Best: {self.best_score:.5f} @ Epoch {self.best_epoch})."
                            )
                            break
                        else:
                            chronicle.log_info(
                                f"Validation patience threshold reached ({epochs_without_improvement} epochs without improvement) "
                                f"in [{self.mode.upper()}]. Classic patience is deactivated; storing conclusion metrics and artifacts and continuing execution."
                            )
                            self.store_conclusion_metrics()
        except KeyboardInterrupt:
            chronicle.log_warning("[INTERRUPT] Training interrupted manually.")

        self.store_conclusion_metrics()
        return self.history

    def store_conclusion_metrics(self) -> None:
        """Store conclusion metrics and artifacts (phase checkpoint, training logs, and evaluation artifacts)."""
        best_model_path = self.save_dir / "model.pt"
        if (not best_model_path.is_file() or self.best_epoch is None) and self.history:
            self.save_checkpoint(self.history[-1]["epoch"], is_best=True)

        # Save final phase checkpoint
        if self.mode in ("train_m1", "m1"):
            last_epoch = self.history[-1]["epoch"] if self.history else self.start_epoch
            self.save_checkpoint(
                last_epoch,
                is_best=False,
                phase_label=f"checkpoint_phase_{self.current_curriculum_phase}",
            )

        self.save_training_log()
        chronicle.log_success(
            f"Session complete [{self.mode.upper()}]. Logs saved in '{self.save_dir}'. "
            f"Best Selection Score: {self.best_score:.5f} @ Epoch {self.best_epoch}"
        )
        self.cfg.save(self.save_dir / "config.yaml")

        if self.artifact_callback is not None:
            self.artifact_callback(self)

    def fit(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        return self._fit_single_phase(epochs=epochs, patience=patience)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        target_loader = loader if loader is not None else self.val_loader
        if target_loader is None:
            raise RuntimeError("Cannot call evaluate when val_loader is None.")
        val_accum: Dict[str, float] = {}
        batch_count = 0
        for batch in target_loader:
            step_metrics = self.handler.validate_step(batch)
            for k, v in step_metrics.items():
                val_accum[k] = val_accum.get(k, 0.0) + v
            batch_count += 1
        return {k: v / max(1, batch_count) for k, v in val_accum.items()}

    def state_dict(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Return a pure primitive/tensor checkpoint dictionary for the trainer."""
        resolved_epoch = (
            int(epoch)
            if epoch is not None
            else (int(self.history[-1]["epoch"]) if self.history else int(self.start_epoch))
        )
        ckpt: Dict[str, Any] = {
            "epoch": resolved_epoch,
            "best_score": float(self.best_score),
            "best_epoch": int(self.best_epoch) if self.best_epoch is not None else None,
            "model_state": self.model.state_dict(),
            "optimizers_state": self.handler.get_optimizers_state(),
            "config": self.cfg.to_dict(),
            "history": self.history,
            "curriculum_state": self.curriculum.state_dict(),
        }
        if self.ema_tracker is not None:
            ckpt["ema_state"] = self.ema_tracker.state_dict()
        if self.replay_buffer is not None:
            ckpt["replay_buffer"] = self.replay_buffer.state_dict()
        if isinstance(self.tier2_evaluator, StatefulTier2EvaluatorProtocol):
            ckpt["tier2_evaluator_state"] = self.tier2_evaluator.state_dict()
        return _sanitize_checkpoint_value(ckpt, "checkpoint")

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        path: Optional[Union[str, Path]] = None,
        phase_label: Optional[str] = None,
    ) -> None:
        ckpt = self.state_dict(epoch=epoch)
        if self.replay_buffer is not None:
            buffer_dest = self.save_dir / "closed_loop_buffer.pt"
            if len(self.replay_buffer) > 0 or not buffer_dest.is_file() or buffer_dest.stat().st_size <= 2500:
                self.replay_buffer.save(buffer_dest)
        if path is not None:
            torch.save(ckpt, Path(path))
            return
        if phase_label is not None:
            label_filename = f"{phase_label}.pt" if not phase_label.endswith(".pt") else phase_label
            torch.save(ckpt, self.save_dir / label_filename)
            chronicle.log_success(f"Saved phase checkpoint: {self.save_dir / label_filename}")
        if is_best:
            torch.save(ckpt, self.save_dir / "model.pt")
            torch.save(ckpt, self.save_dir / "best_model.pt")
        torch.save(ckpt, self.save_dir / "checkpoint.pt")

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load checkpoint strictly (`strict=True`). Raises RuntimeError on any key mismatch."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=True)
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_score = ckpt.get("best_score", float("inf"))
            self.best_epoch = ckpt.get("best_epoch", None)
            self.history = ckpt.get("history", []) or []
            if "optimizers_state" in ckpt:
                self.handler.load_optimizers_state(ckpt["optimizers_state"])
            if self.handler.predictor_optimizer is not None:
                for pg in self.handler.predictor_optimizer.param_groups:
                    pg["lr"] = self.cfg.training.learning_rate
            if "curriculum_state" in ckpt and isinstance(ckpt["curriculum_state"], dict):
                self.curriculum.load_state_dict(ckpt["curriculum_state"])
                self.current_curriculum_phase = self.curriculum.current_curriculum_phase
            else:
                m = re.search(r"checkpoint_phase_(\d+)", path.name)
                if m:
                    loaded_phase = int(m.group(1))
                    self.curriculum.current_curriculum_phase = loaded_phase
                    self.current_curriculum_phase = loaded_phase
                    chronicle.log_info(
                        f"Parsed curriculum phase {loaded_phase} from phase checkpoint filename '{path.name}'."
                    )
            if self.ema_tracker is not None and "ema_state" in ckpt:
                self.ema_tracker.load_state_dict(ckpt["ema_state"])
            if (
                isinstance(self.tier2_evaluator, StatefulTier2EvaluatorProtocol)
                and "tier2_evaluator_state" in ckpt
            ):
                self.tier2_evaluator.load_state_dict(ckpt["tier2_evaluator_state"])
            if self.replay_buffer is not None:
                if "replay_buffer" in ckpt:
                    self.replay_buffer.load_state_dict(ckpt["replay_buffer"])
                else:
                    ckpt_dir = Path(path).parent
                    resolved_buf = resolve_closed_loop_buffer(
                        version_name=self.cfg.project.name,
                        phase_mode=self.mode,
                        save_dir=ckpt_dir,
                        results_root=self.results_root,
                        is_resuming=True,
                    )
                    if resolved_buf is not None:
                        self.replay_buffer.load(resolved_buf)
        else:
            self.model.load_state_dict(ckpt, strict=True)

    def save_training_log(self) -> None:
        if not self.history:
            return

        json_path = self.save_dir / "training_log.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        csv_path = self.save_dir / "training_log.csv"
        fieldnames: List[str] = []
        for rec in self.history:
            for k in rec.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)


_DEFAULT_ENV_CFG = EnvironmentConfig()
_DEFAULT_TRAIN_CFG = TrainingConfig()


class Trainer(BaseTrainer):
    """Typed Trainer constructor building an immutable ExperimentConfig when not passed explicitly."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[ExperimentConfig] = None,
        loss_fn: Optional[SPWMLoss] = None,
        mode: str = _DEFAULT_TRAIN_CFG.mode,
        learning_rate: float = _DEFAULT_TRAIN_CFG.learning_rate,
        weight_decay: float = _DEFAULT_TRAIN_CFG.weight_decay,
        grad_clip_norm: float = _DEFAULT_TRAIN_CFG.grad_clip_norm,
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Union[str, Path] = "results/default_run",
        probe_lr: float = _DEFAULT_TRAIN_CFG.probe_lr,
        probe_weight_decay: float = _DEFAULT_TRAIN_CFG.probe_weight_decay,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        best_epoch: Optional[int] = None,
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        rollout_horizons: Optional[Sequence[int]] = None,
        joint_count: int = _DEFAULT_ENV_CFG.joint_count,
        action_dim: int = _DEFAULT_ENV_CFG.action_dim,
        phase_horizon: int = _DEFAULT_TRAIN_CFG.phase_horizon,
        early_stopping_patience: int = _DEFAULT_TRAIN_CFG.early_stopping_patience,
        metrics_storage: int = _DEFAULT_TRAIN_CFG.metrics_storage,
        classic_patience: bool = _DEFAULT_TRAIN_CFG.classic_patience,
        auto_enforce_checkpoints: bool = False,
        results_root: Union[str, Path] = "results",
        tier2_evaluator: Optional[Tier2EvaluatorProtocol] = None,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            latent_dim = model.latent_dim
            q_dim = model.q_dim
            p_dim = model.p_dim
            num_kp = model.num_keypoints
            timescale_dims = model.timescale_dims
            roll_h = tuple(rollout_horizons) if rollout_horizons is not None else (1, 5, 10, 25, 50, 100)

            seq_len = max(45, 2 * phase_horizon + 5)
            cfg = ExperimentConfig(
                project=ProjectConfig(name=Path(save_dir).stem),
                environment=EnvironmentConfig(
                    height=model.height,
                    width=model.width,
                    joint_count=joint_count,
                    action_dim=action_dim,
                ),
                data=DataConfig(
                    sequence_length=seq_len,
                    joint_count=joint_count,
                    action_dim=action_dim,
                ),
                model=ModelConfig(
                    in_channels=model.in_channels,
                    encoder_dim=model.encoder_dim,
                    latent_dim=latent_dim,
                    q_dim=q_dim,
                    p_dim=p_dim,
                    num_keypoints=num_kp,
                ),
                neuron=LatentNeuronConfig(),
                memory=MemoryConfig(timescale_dims=timescale_dims),
                training=TrainingConfig(
                    mode=mode,
                    early_stopping_patience=early_stopping_patience,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    grad_clip_norm=grad_clip_norm,
                    probe_lr=probe_lr,
                    probe_weight_decay=probe_weight_decay,
                    phase_horizon=phase_horizon,
                    metrics_storage=metrics_storage,
                    classic_patience=classic_patience,
                ),
                loss=LossConfig(),
                evaluation=EvaluationConfig(rollout_horizons=roll_h),
            )

        super().__init__(
            model=model,
            cfg=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            save_dir=save_dir,
            start_epoch=start_epoch,
            best_score=best_val_loss,
            best_epoch=best_epoch,
            history=history,
            auto_enforce_checkpoints=auto_enforce_checkpoints,
            results_root=results_root,
            tier2_evaluator=tier2_evaluator,
        )

        if optimizer_state is not None:
            self.handler.load_optimizers_state(optimizer_state)


__all__ = [
    "BaseTrainer",
    "ClosedLoopBuffer",
    "EMATracker",
    "Trainer",
    "_extract_submodel_state",
    "load_representation_checkpoint",
    "resolve_version_checkpoint",
    "resolve_closed_loop_buffer",
    "enforce_phase_prerequisites",
    "compute_dagger_ratio",
]
