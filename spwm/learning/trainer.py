"""
Trainer interface for SPWM.
Provides typed instantiation around BaseTrainer.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spwm.config.schema import (
    ExperimentConfig,
    ProjectConfig,
    EnvironmentConfig,
    DataConfig,
    ModelConfig,
    LatentNeuronConfig,
    MemoryConfig,
    CorrectorConfig,
    TrainingConfig,
    LossConfig,
    EvaluationConfig,
)
from spwm.learning.engine import BaseTrainer
from spwm.learning.losses import SPWMLoss


class Trainer(BaseTrainer):
    """
    Standard Trainer for SPWM.
    Constructs an ExperimentConfig if not provided directly, then delegates to BaseTrainer.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Optional[ExperimentConfig] = None,
        loss_fn: Optional[SPWMLoss] = None,
        mode: str = "train_predictor",
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
        corrector_burn_in_steps: int = 0,
        perturb_corrector_states: bool = False,
        corrector_perturbation_std: float = 0.00,
        encoder_warmup_epochs: int = 60,
        rollout_horizons: Optional[Sequence[int]] = None,
        num_objects: int = 1,
        **kwargs: Any,
    ) -> None:
        if cfg is None:
            # Build strictly from model and arguments
            latent_dim = getattr(model, "latent_dim", 128)
            q_dim = getattr(model, "q_dim", 32)
            p_dim = getattr(model, "p_dim", 96)
            num_kp = getattr(model, "num_keypoints", q_dim // 2)

            corr_lr_val = corrector_lr if corrector_lr is not None else 0.0004
            roll_h = tuple(rollout_horizons) if rollout_horizons is not None else (1, 5, 10, 25, 50, 100)

            cfg = ExperimentConfig(
                project=ProjectConfig(name=Path(save_dir).stem),
                environment=EnvironmentConfig(num_objects=num_objects),
                data=DataConfig(),
                model=ModelConfig(
                    latent_dim=latent_dim,
                    q_dim=q_dim,
                    p_dim=p_dim,
                    num_keypoints=num_kp,
                ),
                neuron=LatentNeuronConfig(),
                memory=MemoryConfig(
                    timescale_dims=tuple(getattr(model, "timescale_dims", (latent_dim // 2, latent_dim - latent_dim // 2)))
                ),
                corrector=CorrectorConfig(
                    enabled=hasattr(model, "corrector") and model.corrector is not None,
                ),
                training=TrainingConfig(
                    mode=mode,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    grad_clip_norm=grad_clip_norm,
                    probe_lr=probe_lr,
                    probe_weight_decay=probe_weight_decay,
                    corrector_lr=corr_lr_val,
                    corrector_horizon=corrector_horizon,
                    corrector_burn_in_steps=corrector_burn_in_steps,
                    perturb_corrector_states=perturb_corrector_states,
                    corrector_perturbation_std=corrector_perturbation_std,
                ),
                loss=LossConfig(),
                evaluation=EvaluationConfig(
                    rollout_horizons=roll_h,
                ),
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
            best_score=best_combined_score,
            best_epoch=best_epoch,
            history=history,
        )
        self.best_val_pos_err = best_val_pos_err
        if optimizer_state is not None:
            self.handler.load_optimizers_state(optimizer_state)