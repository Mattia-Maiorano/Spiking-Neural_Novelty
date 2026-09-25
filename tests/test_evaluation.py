import tempfile
from pathlib import Path
import torch
import numpy as np
import pytest

from spwm.data.synthetic_world import MovingObjectsWorld
from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from torch.utils.data import DataLoader
from spwm.models.world_model import SPWM
from spwm.evaluation.rollout import RolloutEvaluator
from spwm.evaluation.plots import (
    plot_architecture_diagram,
    plot_training_curves,
    plot_multi_step_degradation,
    plot_spike_raster,
    plot_fast_vs_slow_memory,
    plot_baseline_comparison,
    plot_ablation_comparison,
)


def test_rollout_evaluator():
    world = MovingObjectsWorld()
    sim = EventCameraSimulator(height=32, width=32)
    ds = EventWorldDataset(
        trajectory_indices=[0, 1],
        world=world,
        event_simulator=sim,
        sequence_length=15,
        num_objects=1,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_event_batches)

    model = SPWM(latent_dim=32, timescale_dims=(16, 16), encoder_dim=32)
    evaluator = RolloutEvaluator(model=model, horizons=[1, 3, 5], num_objects=1)
    res = evaluator.evaluate_dataset(loader)

    assert 1 in res.latent_mse_per_horizon
    assert 3 in res.latent_mse_per_horizon
    assert 5 in res.latent_mse_per_horizon
    assert res.teacher_forcing_mse >= 0.0


def test_plotting_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Architecture diagram
        plot_architecture_diagram(tmp_path / "fig1.png")
        assert (tmp_path / "fig1.png").is_file()

        # 2. Training curves
        mock_history = [
            {"epoch": 1, "total_loss": 1.0, "val_total_loss": 0.9, "l_pred": 0.5, "val_l_pred": 0.45, "val_pos_err": 0.3, "val_vel_err": 0.4, "val_spike_rate": 0.05},
            {"epoch": 2, "total_loss": 0.8, "val_total_loss": 0.75, "l_pred": 0.4, "val_l_pred": 0.38, "val_pos_err": 0.25, "val_vel_err": 0.35, "val_spike_rate": 0.04},
        ]
        plot_training_curves(mock_history, tmp_path / "fig2.png")
        assert (tmp_path / "fig2.png").is_file()

        # 3. Multi-step degradation
        mock_curves = {
            "SPWM-v1": {1: 0.1, 5: 0.2, 10: 0.35},
            "GRU": {1: 0.08, 5: 0.22, 10: 0.40},
        }
        plot_multi_step_degradation(mock_curves, tmp_path / "fig3.png")
        assert (tmp_path / "fig3.png").is_file()

        # 4. Spike raster
        fast_spk = (np.random.rand(20, 16) > 0.9).astype(np.float32)
        slow_spk = (np.random.rand(20, 16) > 0.95).astype(np.float32)
        plot_spike_raster(fast_spk, slow_spk, tmp_path / "fig4.png")
        assert (tmp_path / "fig4.png").is_file()

        # 5. Fast vs slow memory
        fast_mem = np.random.randn(20, 16).astype(np.float32)
        slow_mem = np.random.randn(20, 16).astype(np.float32)
        plot_fast_vs_slow_memory(fast_mem, slow_mem, tmp_path / "fig5.png")
        assert (tmp_path / "fig5.png").is_file()

        # 6. Baseline comparison
        mock_metrics = {
            "SPWM-v1": {"one_step_mse": 0.05, "rollout_mse_h10": 0.2, "spike_rate": 0.04},
            "GRU": {"one_step_mse": 0.04, "rollout_mse_h10": 0.25, "spike_rate": 0.0},
        }
        plot_baseline_comparison(mock_metrics, tmp_path / "fig6.png")
        assert (tmp_path / "fig6.png").is_file()

        # 7. Ablation comparison
        mock_ablations = {"Full SPWM": 0.2, "No Slow Memory": 0.45, "Single Timescale": 0.38}
        plot_ablation_comparison(mock_ablations, tmp_path / "fig7.png")
        assert (tmp_path / "fig7.png").is_file()
