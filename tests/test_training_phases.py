"""Phase Training Tests: Layer 2 Training Mechanics, Losses, Isolated Phase Strategies,
and Checkpoint Loading.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
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
from spwm.data.datasets import EventWorldDataset, collate_event_batches
from spwm.models.world_model import SPWM
from spwm.training import Trainer
from spwm.training.engine import BaseTrainer, enforce_phase_prerequisites, resolve_version_checkpoint
from spwm.training.losses import SPWMLoss
from spwm.training.phases import (
    M1PhaseStrategy,
    RepresentationPhaseStrategy,
)


def create_dummy_dataloader(sequence_length=35, batch_size=2, total_traj=4):
    dataset = EventWorldDataset(
        trajectory_indices=list(range(total_traj)),
        sequence_length=sequence_length,
        joint_count=4,
        action_dim=4,
        cache_data=True,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)


def _make_small_config(mode: str = "train_m1") -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="test_phase3"),
        environment=EnvironmentConfig(height=16, width=16, joint_count=4, action_dim=4),
        data=DataConfig(sequence_length=20, batch_size=2, total_trajectories=4, joint_count=4, action_dim=4),
        model=ModelConfig(
            in_channels=2,
            encoder_conv_channels=(8, 16),
            encoder_dim=16,
            latent_dim=16,
            q_dim=4,
            p_dim=12,
            num_keypoints=2,
            predictor_hidden_dim=16,
        ),
        neuron=LatentNeuronConfig(),
        memory=MemoryConfig(timescale_dims=(8, 8), betas=(0.90, 0.985)),
        training=TrainingConfig(
            mode=mode,
            epochs=1,
            phase_horizon=3,
        ),
        loss=LossConfig(lambda_encoder=1.0, lambda_decoder=1.0, lambda_position=1.0, lambda_velocity=1.0),
        evaluation=EvaluationConfig(rollout_horizons=(3, 6)),
    )


def test_phase_strategy_files_under_150_lines() -> None:
    """Each concrete phase strategy file must remain < 150 lines for small-LLM context isolation."""
    phases_dir = Path(__file__).resolve().parent.parent / "spwm" / "training" / "phases"
    for fname in ("representation.py", "m1_nominal.py"):
        fpath = phases_dir / fname
        assert fpath.exists(), f"Missing phase strategy file: {fpath}"
        line_count = len(fpath.read_text(encoding="utf-8").splitlines())
        assert line_count < 150, f"{fname} has {line_count} lines (must be < 150 lines)"


def test_representation_and_m1_isolated_phase_strategies() -> None:
    """Verify RepresentationPhaseStrategy and M1PhaseStrategy satisfy PhaseStrategy and isolate gradients."""
    B, T = 2, 20
    events = torch.rand(B, T, 2, 16, 16)
    flat_kin = torch.randn(B, T, 8)  # 2 * 4 joints
    actions = torch.randn(B, T, 4)
    batch = {"events": events, "flat_kinematics": flat_kin, "actions": actions}
    device = torch.device("cpu")

    # 0. Phase 0: Representation Phase Strategy (trains encoder, dynamics, sensory_predictor, physical_decoder)
    cfg_rep = _make_small_config("train_representation")
    model_rep = SPWM.from_config(cfg_rep)
    loss_fn = SPWMLoss.from_config(cfg_rep.loss)
    rep_strategy = RepresentationPhaseStrategy(model_rep, cfg_rep, loss_fn, device)
    assert isinstance(rep_strategy, PhaseStrategy)
    rep_strategy.verify_gradient_isolation()
    metrics_rep = rep_strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics_rep and "train_loss_kinematic" in metrics_rep
    assert "train_loss_derivative" in metrics_rep and "train_loss_smoothness" in metrics_rep
    assert "train_loss_sensory" in metrics_rep
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_rep.encoder.proj.parameters()
    ), "encoder.proj must receive non-zero gradients during train_representation!"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_rep.dynamics.parameters()
    ), "dynamics must receive non-zero gradients during train_representation!"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_rep.sensory_predictor.parameters()
    ), "sensory_predictor must receive non-zero gradients during train_representation!"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_rep.physical_decoder.parameters()
    ), "physical_decoder must receive non-zero gradients during train_representation!"
    # Predictor / M1 must be frozen in Phase 0
    assert all(p.grad is None for p in model_rep.predictor.parameters())

    # 1. Phase 1: M1 Phase Strategy (strictly trains M1 on detached latent targets with actions u_{0:H})
    cfg_m1 = _make_small_config("train_m1")
    model_m1 = SPWM.from_config(cfg_m1)
    m1_strategy = M1PhaseStrategy(model_m1, cfg_m1, loss_fn, device)
    assert isinstance(m1_strategy, PhaseStrategy)
    m1_strategy.verify_gradient_isolation()
    metrics_m1 = m1_strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics_m1 and "train_loss_m1" in metrics_m1
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_m1.m1.parameters()
    ), "m1 must receive non-zero gradients during train_m1!"
    # Representation modules must be frozen in Phase 1
    assert all(p.grad is None for p in model_m1.encoder.parameters())
    assert all(p.grad is None for p in model_m1.dynamics.parameters())
    assert all(p.grad is None for p in model_m1.sensory_predictor.parameters())
    assert all(p.grad is None for p in model_m1.physical_decoder.parameters())


def test_scoped_strict_checkpoint_loading(tmp_path: Path) -> None:
    """Verify strict checkpoint scoping and error handling for corrupted checkpoints."""
    cfg = _make_small_config("train_m1")
    model_src = SPWM.from_config(cfg)
    trainer_src = BaseTrainer(model_src, cfg, device="cpu", save_dir=tmp_path)

    ckpt_path = tmp_path / "model.pt"
    trainer_src.save_checkpoint(epoch=1, path=ckpt_path)

    model_dst = SPWM.from_config(cfg)
    with torch.no_grad():
        for p in model_dst.parameters():
            p.zero_()

    trainer_dst = BaseTrainer(model_dst, cfg, device="cpu", save_dir=tmp_path)
    trainer_dst.load_checkpoint(ckpt_path)

    for p_s, p_d in zip(model_src.parameters(), model_dst.parameters()):
        assert torch.allclose(p_s, p_d)

    # Corrupt checkpoint keys -> must raise RuntimeError due to strict=True
    corrupted = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    corrupted["model_state"]["unexpected_phantom_layer.weight"] = torch.ones(4, 4)
    bad_ckpt = tmp_path / "bad.pt"
    torch.save(corrupted, bad_ckpt)

    with pytest.raises(RuntimeError):
        trainer_src.load_checkpoint(bad_ckpt)


def test_enforce_phase_prerequisites_and_model_pt(tmp_path: Path) -> None:
    """Verifies that train_m1 enforces existence of train_representation model.pt."""
    model = SPWM(latent_dim=32, q_dim=8, p_dim=24, timescale_dims=(16, 16), phase_horizon=4)

    # Missing prerequisite model.pt must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="model.pt"):
        resolve_version_checkpoint("spwm_v_test", "train_representation", results_root=tmp_path)

    cfg_test = _make_small_config("train_m1").with_overrides(seed=42)
    # With project.name = "spwm_v_test"
    from dataclasses import replace
    cfg_test = replace(cfg_test, project=ProjectConfig(name="spwm_v_test"))

    cfg_missing = replace(
        cfg_test,
        training=replace(cfg_test.training, representation_checkpoint=str(tmp_path / "missing.pt")),
    )

    # Verify enforce_phase_prerequisites raises when explicit checkpoint does not exist
    with pytest.raises(FileNotFoundError):
        enforce_phase_prerequisites(
            model=model,
            cfg=cfg_missing,
            device="cpu",
            results_root=tmp_path,
        )

    # Create dummy representation checkpoint
    rep_dir = tmp_path / "spwm_v_test_train_representation"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_ckpt = rep_dir / "model.pt"
    torch.save(
        {
            "epoch": 10,
            "model_state": model.state_dict(),
            "config": _make_small_config("train_representation"),
        },
        rep_ckpt,
    )

    # Now enforce_phase_prerequisites should succeed and return the checkpoint paths dictionary
    loaded = enforce_phase_prerequisites(
        model=model,
        cfg=cfg_test,
        device="cpu",
        results_root=tmp_path,
    )
    assert loaded["representation"] == rep_ckpt
