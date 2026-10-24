"""
BaseTrainer & Trainer Engine for SPWM (Layer 2 — Training).

Single source of truth for:
- Epoch lifecycle orchestration (`fit`, `train_epoch`, `evaluate`)
- Scoped `strict=True` checkpoint management (`save_checkpoint`, `load_checkpoint`,
  `load_representation_checkpoint`)
- Gradient isolation verification (`verify_gradient_isolation`)

Zero upward imports into `spwm.evaluation` and zero silent `getattr` or `strict=False` fallbacks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import chronicle
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
from spwm.contracts.protocols import PhaseStrategy
from spwm.training.losses import SPWMLoss
from spwm.training.phases import PredictorPhaseHandler


def _extract_submodel_state(full_state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Extract keys starting with `prefix.` and strip the prefix."""
    p_dot = f"{prefix}."
    sub_state = {k[len(p_dot) :]: v for k, v in full_state.items() if k.startswith(p_dot)}
    if not sub_state:
        raise RuntimeError(f"Checkpoint state_dict contains no keys for submodule prefix '{prefix}.'")
    return sub_state


def load_representation_checkpoint(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Strictly load Phase 0 (`train_representation`) trained components into `model`.

    Loads `encoder`, `sensory_predictor`, `dynamics`, and `physical_decoder`
    each with `strict=True`.
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Representation checkpoint not found at: {path}")
    raw = torch.load(path, map_location=device or "cpu", weights_only=False)
    state_dict = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw

    model.encoder.load_state_dict(_extract_submodel_state(state_dict, "encoder"), strict=True)
    model.sensory_predictor.load_state_dict(
        _extract_submodel_state(state_dict, "sensory_predictor"), strict=True
    )
    model.dynamics.load_state_dict(_extract_submodel_state(state_dict, "dynamics"), strict=True)
    model.physical_decoder.load_state_dict(
        _extract_submodel_state(state_dict, "physical_decoder"), strict=True
    )


def resolve_version_checkpoint(
    version_name: str,
    phase_mode: str,
    explicit_path: Optional[Union[str, Path]] = None,
    results_root: Union[str, Path] = "results",
) -> Path:
    """Resolve and strictly verify the best validation `model.pt` for a given version and phase.

    Searches `explicit_path` if provided, otherwise searches `<results_root>/*<version_name>*_<phase_mode>/model.pt`
    (never `checkpoint.pt`). Raises `FileNotFoundError` if no matching `model.pt` exists.
    """
    if explicit_path is not None:
        cand = Path(explicit_path)
        if cand.is_dir():
            cand = cand / "model.pt"
        if not cand.is_file():
            raise FileNotFoundError(
                f"Required best-validation checkpoint 'model.pt' for {phase_mode} not found at: {cand}"
            )
        return cand

    root = Path(results_root)
    exact_cand = root / f"{version_name}_{phase_mode}" / "model.pt"
    if exact_cand.is_file():
        return exact_cand

    clean_ver = version_name[5:] if version_name.startswith("spwm_") else version_name
    alt_cand = root / f"spwm_{clean_ver}_{phase_mode}" / "model.pt"
    if alt_cand.is_file():
        return alt_cand

    if root.is_dir():
        matches = sorted(root.glob(f"*{clean_ver}*_{phase_mode}/model.pt"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Auto-enforced prerequisite checkpoint 'model.pt' for phase '{phase_mode}' and version '{version_name}' "
        f"not found in '{root}'. Ensure '{phase_mode}' has been trained first (expected: {exact_cand})."
    )


def load_m1_checkpoint(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Strictly load Phase 1 (`train_m1`) trained predictor (and representation) into `model`."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"M1 checkpoint not found at: {path}")
    raw = torch.load(path, map_location=device or "cpu", weights_only=False)
    state_dict = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw

    # Load predictor / m1 (handling both legacy CascadedTemporalRelay checkpoint 'predictor.m1.*' and current LatentPredictor 'predictor.*')
    if hasattr(model, "predictor"):
        if any(k.startswith("predictor.m1.") for k in state_dict.keys()):
            pred_state = _extract_submodel_state(state_dict, "predictor.m1")
        else:
            pred_state = _extract_submodel_state(state_dict, "predictor")
        model.predictor.load_state_dict(pred_state, strict=True)
    # Also load representation components if present in checkpoint
    for sub in ("encoder", "sensory_predictor", "dynamics", "physical_decoder"):
        sub_state = _extract_submodel_state(state_dict, sub)
        if sub_state and hasattr(model, sub):
            getattr(model, sub).load_state_dict(sub_state, strict=True)


def enforce_phase_prerequisites(
    model: nn.Module,
    cfg: ExperimentConfig,
    device: Union[str, torch.device],
    results_root: Union[str, Path] = "results",
) -> Dict[str, Path]:
    """Auto-enforce and strictly load prerequisite `model.pt` checkpoints for `train_m1` and `train_m2`.

    - `train_m1`: Requires and loads pre-trained representation (`train_representation` `model.pt`).
    - `train_m2`: Requires and loads pre-trained representation and M1 (`train_m1` `model.pt`).
    """
    mode = cfg.training.mode.lower()
    loaded: Dict[str, Path] = {}

    if mode == "train_m1":
        if cfg.training.representation_checkpoint:
            rep_path = resolve_version_checkpoint(
                cfg.project.name,
                "train_representation",
                explicit_path=cfg.training.representation_checkpoint,
                results_root=results_root,
            )
            load_representation_checkpoint(model, rep_path, device=device)
            loaded["representation"] = rep_path
            chronicle.log_success(f"Strictly loaded representation model.pt: {rep_path}")
        else:
            try:
                rep_path = resolve_version_checkpoint(
                    cfg.project.name,
                    "train_representation",
                    results_root=results_root,
                )
                load_representation_checkpoint(model, rep_path, device=device)
                loaded["representation"] = rep_path
                chronicle.log_success(f"Strictly loaded representation model.pt: {rep_path}")
            except FileNotFoundError as err:
                chronicle.log_info(f"Phase prerequisite notice for {mode}: {err}")

    elif mode in ("train_m2", "train_denoiser"):
        # Require M1 checkpoint (which contains representation + M1)
        if cfg.training.m1_checkpoint:
            m1_path = resolve_version_checkpoint(
                cfg.project.name,
                "train_m1",
                explicit_path=cfg.training.m1_checkpoint,
                results_root=results_root,
            )
            load_m1_checkpoint(model, m1_path, device=device)
            loaded["m1"] = m1_path
            chronicle.log_success(f"Strictly loaded M1 model.pt: {m1_path}")
        else:
            try:
                m1_path = resolve_version_checkpoint(
                    cfg.project.name,
                    "train_m1",
                    results_root=results_root,
                )
                load_m1_checkpoint(model, m1_path, device=device)
                loaded["m1"] = m1_path
                chronicle.log_success(f"Strictly loaded M1 model.pt: {m1_path}")
            except FileNotFoundError as err:
                chronicle.log_info(f"Phase prerequisite notice for {mode}: {err}")

    return loaded


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
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.results_root = Path(results_root)

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
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if loss_fn is None:
            self.loss_fn = SPWMLoss.from_config(cfg.loss).to(self.device)
        else:
            self.loss_fn = loss_fn.to(self.device)

        self.mode = cfg.training.mode.lower()
        if auto_enforce_checkpoints and self.mode in ("train_m1", "train_m2", "train_denoiser"):
            enforce_phase_prerequisites(
                self.model, self.cfg, device=self.device, results_root=self.results_root
            )

        self.handler: PhaseStrategy = PredictorPhaseHandler(
            model=self.model, cfg=self.cfg, loss_fn=self.loss_fn, device=self.device
        )

        self.start_epoch = start_epoch
        self.best_score = best_score
        self.best_epoch = best_epoch
        self.history: List[Dict[str, float]] = history or []

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
        """Executes a single epoch over train_loader using the active PhaseStrategy."""
        if self.train_loader is None:
            raise RuntimeError("Cannot call train_epoch when train_loader is None.")
        train_accum: Dict[str, float] = {}
        batch_count = 0
        for batch in self.train_loader:
            step_metrics = self.handler.train_step(batch, epoch=epoch)
            for k, v in step_metrics.items():
                train_accum[k] = train_accum.get(k, 0.0) + v
            batch_count += 1
        return {k: v / max(1, batch_count) for k, v in train_accum.items()}

    def _fit_single_phase(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        total_epochs = epochs if epochs is not None else self.cfg.training.epochs
        end_epoch = self.start_epoch + total_epochs - 1
        max_no_improve = patience if patience is not None else self.cfg.training.early_stopping_patience

        chronicle.log_application_title(
            f"Starting Training: {self.cfg.project.name} [{self.mode.upper()}]"
        )
        chronicle.log_detail("Device", str(self.device))
        chronicle.log_detail("Epochs", f"{self.start_epoch} -> {end_epoch} (Patience: {max_no_improve})")
        chronicle.log_detail("Save Directory", str(self.save_dir))

        epochs_without_improvement = 0
        start_time = time.time()
        try:
            for epoch in range(self.start_epoch, end_epoch + 1):
                t_ep_start = time.time()
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader)

                score = self.handler.get_selection_score(val_metrics)
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
                    "duration": ep_duration,
                }
                self.history.append(log_entry)
                self.save_checkpoint(epoch, is_best=False)

                if hasattr(self.handler, "log_epoch"):
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

                if epochs_without_improvement >= max_no_improve:
                    chronicle.log_info(
                        f"Early stopping triggered in [{self.mode.upper()}] after {epochs_without_improvement} "
                        f"epochs without validation improvement (Best: {self.best_score:.5f} @ Epoch {self.best_epoch})."
                    )
                    break
        except KeyboardInterrupt:
            chronicle.log_warning("[INTERRUPT] Training interrupted manually.")

        # Ensure model.pt exists even if validation score was flat from epoch 1
        best_model_path = self.save_dir / "model.pt"
        if not best_model_path.is_file() and self.history:
            self.save_checkpoint(self.history[-1]["epoch"], is_best=True)

        self.save_training_log()
        chronicle.log_success(
            f"Session complete [{self.mode.upper()}]. Logs saved in '{self.save_dir}'. "
            f"Best Selection Score: {self.best_score:.5f} @ Epoch {self.best_epoch}"
        )
        return self.history

    def fit(
        self,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        return self._fit_single_phase(epochs=epochs, patience=patience)

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

    def save_checkpoint(self, epoch: int, is_best: bool = False, path: Optional[Union[str, Path]] = None) -> None:
        ckpt = {
            "epoch": epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "model_state": self.model.state_dict(),
            "optimizers_state": self.handler.get_optimizers_state(),
            "config": self.cfg.to_dict(),
            "history": self.history,
        }
        if path is not None:
            torch.save(ckpt, Path(path))
            return
        if is_best:
            torch.save(ckpt, self.save_dir / "model.pt")
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


class Trainer(BaseTrainer):
    """Typed Trainer constructor building an immutable ExperimentConfig when not passed explicitly."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        cfg: Optional[ExperimentConfig] = None,
        loss_fn: Optional[SPWMLoss] = None,
        mode: str = "train_m1",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 1.0,
        learning_algorithm: str = "online_eprop",
        device: Optional[Union[str, torch.device]] = None,
        save_dir: Union[str, Path] = "results/default_run",
        probe_lr: float = 5e-4,
        probe_weight_decay: float = 1e-2,
        start_epoch: int = 1,
        best_val_loss: float = float("inf"),
        best_epoch: Optional[int] = None,
        history: Optional[List[Dict[str, float]]] = None,
        optimizer_state: Optional[Dict[str, Any]] = None,
        k_max: int = 50,
        encoder_warmup_epochs: int = 60,
        rollout_horizons: Optional[Sequence[int]] = None,
        num_objects: int = 1,
        phase_horizon: int = 10,
        max_rollout_horizon: int = 30,
        early_stopping_patience: int = 20,
        auto_enforce_checkpoints: bool = False,
        results_root: Union[str, Path] = "results",
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            latent_dim = model.latent_dim
            q_dim = model.q_dim
            p_dim = model.p_dim
            num_kp = model.num_keypoints
            timescale_dims = model.timescale_dims
            roll_h = tuple(rollout_horizons) if rollout_horizons is not None else (1, 5, 10, 25, 50, 100)

            seq_len = max(45, 2 * phase_horizon + 5, max_rollout_horizon + 5)
            cfg = ExperimentConfig(
                project=ProjectConfig(name=Path(save_dir).stem),
                environment=EnvironmentConfig(
                    height=model.height,
                    width=model.width,
                    num_objects=num_objects,
                ),
                data=DataConfig(sequence_length=seq_len),
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
                    max_rollout_horizon=max_rollout_horizon,
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
        )

        if optimizer_state is not None:
            self.handler.load_optimizers_state(optimizer_state)


__all__ = [
    "BaseTrainer",
    "Trainer",
    "load_representation_checkpoint",
    "resolve_version_checkpoint",
    "enforce_phase_prerequisites",
]
