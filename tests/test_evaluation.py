import tempfile
from pathlib import Path
import torch
import numpy as np

from spwm.data.event_camera import EventCameraSimulator
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from torch.utils.data import DataLoader
from spwm.models.world_model import SPWM
from spwm.evaluation.evaluator import RolloutEvaluator


def test_rollout_evaluator_long_horizon():
    sim = EventCameraSimulator(height=32, width=32)
    ds = EventWorldDataset(
        trajectory_indices=[0, 1],
        event_simulator=sim,
        sequence_length=150,
        joint_count=6,
        action_dim=6,
        strict_dataset=False,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_event_batches)

    model = SPWM(latent_dim=32, timescale_dims=(16, 16), encoder_dim=32, joint_count=6, action_dim=6)
    evaluator = RolloutEvaluator(model=model, horizons=[1, 5, 10, 25, 50], joint_count=6)
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


def test_ghost_visualizer_exporter_contract():
    import json
    from spwm.evaluation.visualizer import GhostVisualizerExporter

    sim = EventCameraSimulator(height=32, width=32)
    ds = EventWorldDataset(
        trajectory_indices=[0, 1],
        event_simulator=sim,
        sequence_length=40,
        joint_count=6,
        action_dim=6,
        strict_dataset=False,
    )
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_event_batches)
    model = SPWM(latent_dim=32, timescale_dims=(16, 16), encoder_dim=32, joint_count=6, action_dim=6)
    evaluator = RolloutEvaluator(model=model, horizons=[30], joint_count=6, temporal_burn_in=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = GhostVisualizerExporter(evaluator, unity_project_dir=tmpdir, horizon=30, dt=0.05, model_version="spwm_v0_4_0")
        out = exporter.export(loader, batch_index=0, sample_index=1)
        assert out == Path(tmpdir) / "Assets" / "StreamingAssets" / "trajectory_ghost.json"
        payload = json.loads(out.read_text())

    meta, frames = payload["metadata"], payload["frames"]
    assert meta["horizon"] == 30 and meta["dt"] == 0.05 and meta["model_version"] == "spwm_v0_4_0"
    assert meta["trajectory_id"] == 1 and meta["t_start"] == 2
    assert len(frames) == 30
    for f in frames:
        assert len(f["ground_truth_q"]) == 6 and len(f["predicted_q"]) == 6
        m = f["instantaneous_metrics"]
        assert m["position_abs_error"] >= 0.0 and m["velocity_relative_degradation"] >= 0.0
    assert abs(frames[0]["instantaneous_metrics"]["velocity_relative_degradation"] - 1.0) < 1e-6
    # Ground truth must be the true kinematics aligned at t_start + t.
    true_q = ds[1]["joint_positions"].numpy()
    assert np.allclose(frames[0]["ground_truth_q"], true_q[3], atol=1e-5)


def test_evaluator_active_perception_soft_fade_handover():
    """Verify RolloutEvaluator active perception softly blends continuous M1 during sensory steps."""
    B, T = 2, 12
    model = SPWM(latent_dim=32, timescale_dims=(16, 16), encoder_dim=32, joint_count=6, action_dim=6)
    evaluator = RolloutEvaluator(model=model, horizons=[9], joint_count=6, temporal_burn_in=2)

    events = torch.rand(B, T, 4, 32, 32)
    actions = torch.randn(B, T, 6)
    flat_kin = torch.randn(B, T, 12)
    # Timesteps: 0..4 Phase 1 (0), 5..7 Phase 1.5 (1), 8..11 Phase 2 (2)
    phase_flags = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]] * B, dtype=torch.long)
    batch = {"events": events, "actions": actions, "flat_kinematics": flat_kin, "phase_flags": phase_flags}

    m1_call_count = 0
    orig_m1_forward = model.m1.forward

    def counted_m1_forward(*args, **kwargs):
        nonlocal m1_call_count
        m1_call_count += 1
        return orig_m1_forward(*args, **kwargs)

    model.m1.forward = counted_m1_forward  # type: ignore[method-assign]
    traj = evaluator.rollout_trajectory(batch, horizon=9)

    # In continuous soft-fade handover, M1 unrolls continuously across all horizon steps (9 steps)
    # without binary interruption cliff, softly blending with grounded dynamics during active perception.
    assert m1_call_count == 9
    assert traj.horizon == 9
    assert traj.t_start == 2
