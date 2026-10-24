import tempfile
from pathlib import Path
import torch
import numpy as np
import pytest

from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from torch.utils.data import DataLoader
from spwm.models.world_model import SPWM
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import (
    plot_ablation_comparison,
    plot_architecture_diagram,
    plot_baseline_comparison,
    plot_fast_vs_slow_memory,
    plot_latent_rollout_degradation,
    plot_multi_step_degradation,
    plot_physical_rollout_degradation,
    plot_spike_raster,
    plot_training_curves,
)


def test_rollout_evaluator_long_horizon():
    sim = EventCameraSimulator(height=32, width=32)
    ds = EventWorldDataset(
        trajectory_indices=[0, 1],
        event_simulator=sim,
        sequence_length=150,
        joint_count=8,
        action_dim=8,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_event_batches)

    model = SPWM(latent_dim=32, timescale_dims=(16, 16), encoder_dim=32, joint_count=8, action_dim=8)
    evaluator = RolloutEvaluator(model=model, horizons=[1, 5, 10, 25, 50], joint_count=8)
    res = evaluator.evaluate_dataset(loader)

    for h in [1, 5, 10, 25, 50]:
        assert h in res.latent_mse_per_horizon
        assert res.latent_mse_per_horizon[h] >= 0.0
        if res.q_mse_per_horizon is not None:
            assert h in res.q_mse_per_horizon
            assert res.q_mse_per_horizon[h] >= 0.0
        if res.p_mse_per_horizon is not None:
            assert h in res.p_mse_per_horizon
            assert res.p_mse_per_horizon[h] >= 0.0
    assert res.teacher_forcing_mse >= 0.0
    assert res.dynamic_score >= 0.0
    assert res.rollout_mae >= 0.0



def test_plotting_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Architecture diagram
        plot_architecture_diagram(tmp_path / "fig1.png")
        assert (tmp_path / "fig1.png").is_file()

        # 2. Training curves
        mock_history = [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "train_pos_err": 0.3, "train_vel_err": 0.4, "val_pos_err": 0.3, "val_vel_err": 0.4},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.75, "train_pos_err": 0.25, "train_vel_err": 0.35, "val_pos_err": 0.25, "val_vel_err": 0.35},
        ]
        plot_training_curves(mock_history, tmp_path / "fig2.png")
        assert (tmp_path / "fig2.png").is_file()

        # 3. Multi-step degradation
        mock_curves = {
            "SPWM-v3": {1: 0.1, 5: 0.2, 10: 0.35, 25: 0.45, 50: 0.55},
            "GRU": {1: 0.08, 5: 0.22, 10: 0.40, 25: 0.60, 50: 0.80},
        }
        plot_multi_step_degradation(mock_curves, tmp_path / "fig3.png")
        assert (tmp_path / "fig3.png").is_file()
        plot_multi_step_degradation(mock_curves, save_path=tmp_path / "fig3_kw.png")
        assert (tmp_path / "fig3_kw.png").is_file()

        # Multi-metric degradation (pos, vel, latent)
        horizons = [1, 5, 10, 25]
        pos_errs = {1: 0.1, 5: 0.2, 10: 0.35, 25: 0.45}
        vel_errs = {1: 0.05, 5: 0.15, 10: 0.25, 25: 0.35}
        latent_errs = {1: 0.02, 5: 0.08, 10: 0.18, 25: 0.28}
        plot_multi_step_degradation(
            horizons=horizons,
            pos_errors=pos_errs,
            vel_errors=vel_errs,
            latent_errors=latent_errs,
            save_path=tmp_path / "fig3_all_metrics.png",
        )
        assert (tmp_path / "fig3_all_metrics.png").is_file()

        plot_physical_rollout_degradation(
            horizons=horizons,
            pos_errors=pos_errs,
            vel_errors=vel_errs,
            save_path=tmp_path / "physical_rollout_degradation.png",
            relative=True,
            pos_span=2.0,
            vel_span=2.0,
        )
        assert (tmp_path / "physical_rollout_degradation.png").is_file()

        q_errs = {1: 0.01, 5: 0.04, 10: 0.09, 25: 0.14}
        p_errs = {1: 0.01, 5: 0.04, 10: 0.09, 25: 0.14}
        plot_latent_rollout_degradation(
            horizons=horizons,
            latent_errors=latent_errs,
            q_errors=q_errs,
            p_errors=p_errs,
            save_path=tmp_path / "latent_rollout_degradation.png",
            relative=True,
        )
        assert (tmp_path / "latent_rollout_degradation.png").is_file()

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
            "SPWM-v3": {"one_step_mse": 0.05, "rollout_mse_h10": 0.2, "spike_rate": 0.12},
            "GRU": {"one_step_mse": 0.04, "rollout_mse_h10": 0.25, "spike_rate": 0.0},
        }
        plot_baseline_comparison(mock_metrics, tmp_path / "fig6.png")
        assert (tmp_path / "fig6.png").is_file()

        # 7. Ablation comparison
        mock_ablations = {"Full SPWM-v3": 0.2, "No Slow Memory": 0.45, "Single Timescale": 0.38}
        plot_ablation_comparison(mock_ablations, tmp_path / "fig7.png")
        assert (tmp_path / "fig7.png").is_file()


def test_generate_metrics_resume():
    import json
    from spwm.evaluation.metrics_resume import generate_metrics_resume, save_metrics_resume

    mock_history = [
        {"epoch": i + 1, "train_loss": 1.0 - 0.01 * i, "val_loss": 1.2 - 0.005 * i, "score": 1.2 - 0.005 * i, "train_pos_err": 0.5 - 0.002 * i}
        for i in range(50)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "metrics_test.json"
        from spwm.evaluation.evaluator import RolloutEvaluationResult
        mock_eval = RolloutEvaluationResult(
            horizons=[1, 5, 10],
            latent_mse_per_horizon={1: 0.1, 5: 0.2, 10: 0.3},
            position_error_per_horizon={1: 0.05, 5: 0.1, 10: 0.15},
            velocity_error_per_horizon={1: 0.02, 5: 0.04, 10: 0.06},
            teacher_forcing_mse=0.01,
            mean_spike_rate=0.08,
        )
        save_metrics_resume(
            target_path=json_path,
            history=mock_history,
            evaluation_result=mock_eval,
            parameters=123456,
            num_segments=10,
            model_name="test_model",
            version="v0_1_0",
            evaluation_result_no_corrector=mock_eval,
        )
        assert json_path.is_file()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["version"] == "v0_1_0"
        assert data["total_epochs"] == 50
        assert data["parameters"] == 123456
        assert "evaluation" in data
        assert "evaluation_no_corrector" in data
        assert "rollout_score" in data["evaluation"]
        assert "dynamic_score" in data["evaluation"]
        assert "rollout_score" in data["evaluation_no_corrector"]
        assert "gate_passed" in data["evaluation"]
        assert len(data["training_progression"]["sampled_segments"]) == 10
        assert data["training_progression"]["sampled_segments"][0]["epoch"] == 1
        assert data["training_progression"]["sampled_segments"][-1]["epoch"] == 50
        assert "train_loss" in data["training_progression"]["trends"]
        assert "val_loss" in data["training_progression"]["trends"]
        assert "physical_rollout_degradation" in data["evaluation"]
        assert "latent_rollout_degradation" in data["evaluation"]
        assert "position_relative" in data["evaluation"]["physical_rollout_degradation"]
        assert "velocity_relative" in data["evaluation"]["physical_rollout_degradation"]
        assert "position_normalized" in data["evaluation"]["physical_rollout_degradation"]
        assert "velocity_normalized" in data["evaluation"]["physical_rollout_degradation"]
        assert data["evaluation"]["physical_rollout_degradation"]["position_relative"]["1"] == 1.0
        assert data["evaluation"]["physical_rollout_degradation"]["velocity_relative"]["1"] == 1.0
        assert "latent_relative" in data["evaluation"]["latent_rollout_degradation"]
        assert data["evaluation"]["latent_rollout_degradation"]["latent_relative"]["1"] == 1.0


def test_get_canonical_metrics_filename():
    from spwm.evaluation.metrics_resume import get_canonical_metrics_filename
    assert get_canonical_metrics_filename("spwm_v0_1_0") == "metrics_v0_1_0.json"
    assert get_canonical_metrics_filename("v0_1_0") == "metrics_v0_1_0.json"
    assert get_canonical_metrics_filename("0_1_0") == "metrics_v0_1_0.json"
    assert get_canonical_metrics_filename("spwm_v8_3") == "metrics_v8_3.json"







