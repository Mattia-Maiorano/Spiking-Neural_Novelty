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
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.visualizer import (
    plot_architecture_diagram,
    plot_training_curves,
    plot_multi_step_degradation,
    plot_spike_raster,
    plot_fast_vs_slow_memory,
    plot_baseline_comparison,
    plot_ablation_comparison,
)



def test_rollout_evaluator_long_horizon():
    world = MovingObjectsWorld()
    sim = EventCameraSimulator(height=32, width=32)
    ds = EventWorldDataset(
        trajectory_indices=[0, 1],
        world=world,
        event_simulator=sim,
        sequence_length=150,
        num_objects=1,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_event_batches)

    model = SPWM(latent_dim=32, timescale_dims=(16, 16), encoder_dim=32)
    evaluator = RolloutEvaluator(model=model, horizons=[1, 5, 10, 25, 50], num_objects=1)
    res = evaluator.evaluate_dataset(loader)

    for h in [1, 5, 10, 25, 50]:
        assert h in res.latent_mse_per_horizon
        assert res.latent_mse_per_horizon[h] >= 0.0
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


def test_simplified_loss_forward():
    from spwm.learning.losses import SPWMLoss

    loss_fn = SPWMLoss(lambda_encoder=1.0, lambda_decoder=1.0, lambda_position=1.0, lambda_velocity=1.0)
    B, T, D = 4, 10, 4

    # Test train_representation mode
    keypoints = torch.randn(B, T, 8, requires_grad=True)
    decoded = torch.randn(B, T, D, requires_grad=True)
    gt = torch.randn(B, T, D)

    repr_loss = loss_fn(
        keypoints=keypoints,
        decoded_kinematics=decoded,
        target_kinematics=gt,
        mode="train_representation",
    )
    assert "total_loss" in repr_loss
    assert "loss_encoder" in repr_loss
    assert "loss_decoder" in repr_loss
    assert "loss_position" in repr_loss
    assert "loss_velocity" in repr_loss
    assert repr_loss["total_loss"].item() > 0.0
    assert repr_loss["loss_position"].item() == 0.0
    repr_loss["total_loss"].backward()
    assert keypoints.grad is not None
    assert decoded.grad is not None

    # Test train_predictor mode
    pos = torch.randn(B, T, 2, requires_grad=True)
    vel = torch.randn(B, T, 2, requires_grad=True)
    pred_loss = loss_fn(
        rollout_positions=pos,
        rollout_velocities=vel,
        target_kinematics=gt,
        mode="train_predictor",
    )
    assert "total_loss" in pred_loss
    assert "loss_encoder" in pred_loss
    assert "loss_decoder" in pred_loss
    assert "loss_position" in pred_loss
    assert "loss_velocity" in pred_loss
    assert pred_loss["total_loss"].item() > 0.0
    assert pred_loss["loss_encoder"].item() == 0.0
    pred_loss["total_loss"].backward()
    assert pos.grad is not None
    assert vel.grad is not None

    # Test train_joint mode
    kp_j = torch.randn(B, T, 8, requires_grad=True)
    dec_j = torch.randn(B, T, D, requires_grad=True)
    pos_j = torch.randn(B, T, 2, requires_grad=True)
    vel_j = torch.randn(B, T, 2, requires_grad=True)
    joint_loss = loss_fn(
        keypoints=kp_j,
        decoded_kinematics=dec_j,
        rollout_positions=pos_j,
        rollout_velocities=vel_j,
        target_kinematics=gt,
        mode="train_joint",
    )
    assert "total_loss" in joint_loss
    assert joint_loss["loss_encoder"].item() > 0.0
    assert joint_loss["loss_decoder"].item() > 0.0
    assert joint_loss["loss_position"].item() > 0.0
    assert joint_loss["loss_velocity"].item() > 0.0
    joint_loss["total_loss"].backward()
    assert kp_j.grad is not None
    assert dec_j.grad is not None
    assert pos_j.grad is not None
    assert vel_j.grad is not None



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


def test_get_canonical_metrics_filename():
    from spwm.evaluation.metrics_resume import get_canonical_metrics_filename
    assert get_canonical_metrics_filename("spwm_v0_1_0") == "metrics_v0_1_0.json"
    assert get_canonical_metrics_filename("v0_1_0") == "metrics_v0_1_0.json"
    assert get_canonical_metrics_filename("0_1_0") == "metrics_v0_1_0.json"
    assert get_canonical_metrics_filename("spwm_v8_3") == "metrics_v8_3.json"







