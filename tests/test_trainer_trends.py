import tempfile
from pathlib import Path
import torch
import pytest

from spwm.models.world_model import SPWM
from spwm.learning.trainer import Trainer


def test_trainer_trend_logging_and_file_exports():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Build tiny mock model & dataset
        model = SPWM(
            in_channels=2,
            height=32,
            width=32,
            encoder_conv_channels=(8, 16),
            encoder_dim=16,
            latent_dim=16,
            timescale_dims=(8, 8),
            predictor_hidden_dim=32,
            num_objects=1,
        )

        dummy_batch = {
            "events": torch.zeros(2, 5, 2, 32, 32),
            "flat_kinematics": torch.zeros(2, 5, 4),
            "kinematics": torch.zeros(2, 5, 1, 4),
        }
        train_loader = [dummy_batch]
        val_loader = [dummy_batch]

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=str(tmp_path),
            tensorboard_logging=False,
        )

        # Run 3 epochs
        history = trainer.fit(epochs=3, print_every=1)
        assert len(history) == 3

        # Check trends in history record
        rec3 = history[-1]
        assert "val_pos_err_rate_10" in rec3
        assert "val_pos_err_rel_imp_10" in rec3
        assert "val_total_loss_rate_run" in rec3
        assert "val_total_loss_rel_imp_run" in rec3

        # Check exported files
        assert (tmp_path / "training_log.json").exists()
        assert (tmp_path / "training_log.csv").exists()
        assert (tmp_path / "training_log.txt").exists()

        txt_content = (tmp_path / "training_log.txt").read_text(encoding="utf-8")
        assert "Trends ValPosErr:" in txt_content
        assert "Trends ValTotalLoss:" in txt_content
