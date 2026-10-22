"""
Phase 3 TDD Tests: Layer 2 Training Mechanics, Losses, Isolated Phase Strategies,
and Scoped Strict Checkpoint Loading.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

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
from spwm.models.world_model import SPWM
from spwm.training.engine import BaseTrainer, load_m1_checkpoint, load_m2_checkpoint
from spwm.training.losses import (
    ConvergenceFunnelLoss,
    SPWMLoss,
    compute_convergence_funnel_target,
    extrapolate_latent,
)
from spwm.training.phases import (
    M1PhaseStrategy,
    M2PhaseStrategy,
    M3PhaseStrategy,
)


def _make_small_config(mode: str = "train_m1") -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="test_phase3"),
        environment=EnvironmentConfig(height=16, width=16, num_objects=1),
        data=DataConfig(sequence_length=20, batch_size=2, total_trajectories=4),
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
            max_rollout_horizon=9,
        ),
        loss=LossConfig(lambda_encoder=1.0, lambda_decoder=1.0, lambda_position=1.0, lambda_velocity=1.0),
        evaluation=EvaluationConfig(rollout_horizons=(3, 6)),
    )


def test_phase_strategy_files_under_150_lines() -> None:
    """Each concrete phase strategy file must remain < 150 lines for small-LLM context isolation."""
    phases_dir = Path(__file__).resolve().parent.parent / "spwm" / "training" / "phases"
    for fname in ("m1_nominal.py", "m2_recovery.py", "m3_sustained.py"):
        fpath = phases_dir / fname
        assert fpath.exists(), f"Missing phase strategy file: {fpath}"
        line_count = len(fpath.read_text(encoding="utf-8").splitlines())
        assert line_count < 150, f"{fname} has {line_count} lines (must be < 150 lines)"


def test_convergence_funnel_exact_boundary_math() -> None:
    """Verify linear phase-space extrapolation and Convergence Funnel boundary conditions."""
    B, H, D = 2, 4, 8
    z_prev = torch.ones(B, D) * 1.0
    z_anchor = torch.ones(B, D) * 2.0  # delta = +1.0 per step
    z_true = torch.zeros(B, H, D)

    z_funnel = compute_convergence_funnel_target(
        z_anchor=z_anchor,
        z_prev=z_prev,
        z_ground_truth=z_true,
        t_anchor=10,
        phase_horizon=H,
    )
    assert z_funnel.shape == (B, H, D)

    # At idx=0 (t=11): tau = 1/4 = 0.25 -> (1 - 0.25) * Extrapolate(11) + 0.25 * 0
    expected_step0 = 0.75 * extrapolate_latent(z_anchor, t=11, t_anchor=10, z_prev=z_prev)
    assert torch.allclose(z_funnel[:, 0, :], expected_step0, atol=1e-6)

    # At idx=H-1 (t=14): tau = 4/4 = 1.0 -> purely ground-truth z_true[:, H-1, :]
    assert torch.allclose(z_funnel[:, H - 1, :], z_true[:, H - 1, :], atol=1e-6)

    loss_mod = ConvergenceFunnelLoss(t_anchor=10)
    out = loss_mod(z_funnel, z_anchor, z_prev, z_true, t_anchor=10, phase_horizon=H)
    assert torch.isclose(out["funnel_loss"], torch.tensor(0.0), atol=1e-6)


def test_m1_m2_m3_isolated_phase_strategies() -> None:
    """Verify M1PhaseStrategy, M2PhaseStrategy, and M3PhaseStrategy satisfy PhaseStrategy and isolate gradients."""
    B, T = 2, 20
    events = torch.rand(B, T, 2, 16, 16)
    flat_kin = torch.randn(B, T, 4)
    batch = {"events": events, "flat_kinematics": flat_kin}
    device = torch.device("cpu")

    # 1. M1 Phase Strategy (also verifies encoder.proj receives non-zero gradient!)
    cfg_m1 = _make_small_config("train_m1")
    model_m1 = SPWM.from_config(cfg_m1)
    loss_fn = SPWMLoss.from_config(cfg_m1.loss)
    m1_strategy = M1PhaseStrategy(model_m1, cfg_m1, loss_fn, device)
    assert isinstance(m1_strategy, PhaseStrategy)
    m1_strategy.verify_gradient_isolation()
    metrics_m1 = m1_strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics_m1 and "train_loss_encoder" in metrics_m1
    assert "train_loss_sensory" in metrics_m1
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_m1.encoder.proj.parameters()
    ), "encoder.proj must receive non-zero gradients during train_m1!"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_m1.dynamics.parameters()
    ), "dynamics must receive non-zero gradients during train_m1!"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_m1.sensory_predictor.parameters()
    ), "sensory_predictor must receive non-zero gradients during train_m1!"

    # 2. M2 Phase Strategy
    cfg_m2 = _make_small_config("train_m2")
    model_m2 = SPWM.from_config(cfg_m2)
    m2_strategy = M2PhaseStrategy(model_m2, cfg_m2, loss_fn, device)
    assert isinstance(m2_strategy, PhaseStrategy)
    m2_strategy.verify_gradient_isolation()
    metrics_m2 = m2_strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics_m2 and "train_loss_velocity" in metrics_m2

    # 3. M3 Phase Strategy
    cfg_m3 = _make_small_config("train_m3")
    model_m3 = SPWM.from_config(cfg_m3)
    m3_strategy = M3PhaseStrategy(model_m3, cfg_m3, loss_fn, device)
    assert isinstance(m3_strategy, PhaseStrategy)
    m3_strategy.verify_gradient_isolation()
    metrics_m3 = m3_strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics_m3 and "train_loss_position" in metrics_m3


def test_scoped_strict_checkpoint_loading(tmp_path: Path) -> None:
    """Verify strict checkpoint scoping for M1 -> M2 -> M3 transitions and full checkpoints."""
    cfg = _make_small_config("train_m1")
    model_src = SPWM.from_config(cfg)
    trainer_src = BaseTrainer(model_src, cfg, device="cpu", save_dir=tmp_path)

    m1_ckpt = tmp_path / "m1.pt"
    m2_ckpt = tmp_path / "m2.pt"
    trainer_src.save_checkpoint(epoch=1, path=m1_ckpt)
    trainer_src.save_checkpoint(epoch=1, path=m2_ckpt)

    model_dst = SPWM.from_config(cfg)
    with torch.no_grad():
        for p in model_dst.m1.parameters():
            p.zero_()
        for p in model_dst.m2.parameters():
            p.zero_()

    load_m1_checkpoint(model_dst, m1_ckpt, device=torch.device("cpu"))
    for p_s, p_d in zip(model_src.m1.parameters(), model_dst.m1.parameters()):
        assert torch.allclose(p_s, p_d)

    load_m2_checkpoint(model_dst, m2_ckpt, device=torch.device("cpu"))
    for p_s, p_d in zip(model_src.m2.parameters(), model_dst.m2.parameters()):
        assert torch.allclose(p_s, p_d)

    # Corrupt checkpoint keys -> must raise RuntimeError due to strict=True
    corrupted = torch.load(m1_ckpt, map_location="cpu", weights_only=False)
    corrupted["model_state"]["unexpected_phantom_layer.weight"] = torch.ones(4, 4)
    bad_ckpt = tmp_path / "bad.pt"
    torch.save(corrupted, bad_ckpt)

    with pytest.raises(RuntimeError):
        trainer_src.load_checkpoint(bad_ckpt)
