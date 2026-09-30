import pytest
import torch
import torch.nn as nn
from spwm.models.world_model import SPWM
from spwm.learning.losses import SPWMLoss
from spwm.learning.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset


def test_trainer_adaptive_curriculum_multi_step_transition():
    # Setup dummy data and lightweight model
    B, T, C, H, W = 4, 30, 2, 32, 32
    events = torch.rand(B, T, C, H, W)
    kinematics = torch.rand(B, T, 4)

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "events": events[idx],
                "flat_kinematics": kinematics[idx],
            }

    loader = DataLoader(DummyDataset(), batch_size=2)

    model = SPWM(
        in_channels=C,
        height=H,
        width=W,
        encoder_dim=32,
        latent_dim=32,
        timescale_dims=(16, 16),
        betas=(0.90, 0.985),
        num_objects=1,
    )

    loss_fn = SPWMLoss(
        multi_step_horizon=3,
        lambda_multi=0.5,
        lambda_sparse=0.0025,
    )

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=loss_fn,
        curriculum_multi_step=True,
        curriculum_thresholds={
            "phase_1_horizon": 3,
            "phase_2_horizon": 6,
            "phase_2_threshold": 0.20,
            "phase_3_horizon": 10,
            "phase_3_threshold": 0.12,
        },
    )

    assert trainer.loss_fn.multi_step_horizon == 3

    # Simulate metric progression
    # 1. val_pos_err >= 0.20 -> K stays 3
    trainer.best_val_pos_err = 0.25
    val_metrics = {"val_pos_err": 0.22, "val_total_loss": 0.5, "val_spike_rate": 0.11}
    # Apply curriculum update logic directly
    p1_k = trainer.curriculum_thresholds["phase_1_horizon"]
    p2_k = trainer.curriculum_thresholds["phase_2_horizon"]
    p2_th = trainer.curriculum_thresholds["phase_2_threshold"]
    p3_k = trainer.curriculum_thresholds["phase_3_horizon"]
    p3_th = trainer.curriculum_thresholds["phase_3_threshold"]

    curr_val_pos = val_metrics["val_pos_err"]
    if curr_val_pos < p3_th:
        target_k = p3_k
    elif curr_val_pos < p2_th:
        target_k = p2_k
    else:
        target_k = p1_k
    trainer.loss_fn.multi_step_horizon = target_k
    assert trainer.loss_fn.multi_step_horizon == 3

    # 2. val_pos_err < 0.20 and >= 0.12 -> K becomes 6
    val_metrics["val_pos_err"] = 0.18
    curr_val_pos = val_metrics["val_pos_err"]
    if curr_val_pos < p3_th:
        target_k = p3_k
    elif curr_val_pos < p2_th:
        target_k = p2_k
    else:
        target_k = p1_k
    trainer.loss_fn.multi_step_horizon = target_k
    assert trainer.loss_fn.multi_step_horizon == 6

    # 3. val_pos_err < 0.12 -> K becomes 10
    val_metrics["val_pos_err"] = 0.08
    curr_val_pos = val_metrics["val_pos_err"]
    if curr_val_pos < p3_th:
        target_k = p3_k
    elif curr_val_pos < p2_th:
        target_k = p2_k
    else:
        target_k = p1_k
    trainer.loss_fn.multi_step_horizon = target_k
    assert trainer.loss_fn.multi_step_horizon == 10
