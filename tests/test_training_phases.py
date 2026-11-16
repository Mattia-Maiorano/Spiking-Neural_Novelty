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
        joint_count=6,
        action_dim=6,
        cache_data=True,
        strict_dataset=False,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)


def _make_small_config(mode: str = "train_m1") -> ExperimentConfig:
    return ExperimentConfig(
        project=ProjectConfig(name="test_phase3"),
        environment=EnvironmentConfig(height=16, width=16, joint_count=6, action_dim=6),
        data=DataConfig(sequence_length=20, batch_size=2, total_trajectories=4, joint_count=6, action_dim=6),
        model=ModelConfig(
            in_channels=4,
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
    events = torch.rand(B, T, 4, 16, 16)
    flat_kin = torch.randn(B, T, 12)  # 2 * 6 joints
    actions = torch.randn(B, T, 6)
    phase_flags = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]] * B, dtype=torch.long)
    batch = {"events": events, "flat_kinematics": flat_kin, "actions": actions, "phase_flags": phase_flags}
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
    # Predictor / M1 and Critic must be frozen in Phase 0
    assert all(p.grad is None for p in model_rep.predictor.parameters())
    assert all(p.grad is None for p in model_rep.critic.parameters())

    # 1. Phase 1: M1 Phase Strategy (strictly trains M1 + Critic with Active Perception Injection on phase_flags == 1)
    cfg_m1 = _make_small_config("train_m1")
    model_m1 = SPWM.from_config(cfg_m1)
    m1_strategy = M1PhaseStrategy(model_m1, cfg_m1, loss_fn, device)
    assert isinstance(m1_strategy, PhaseStrategy)
    m1_strategy.verify_gradient_isolation()
    metrics_m1 = m1_strategy.train_step(batch, epoch=1)
    assert "train_loss" in metrics_m1 and "train_loss_m1" in metrics_m1 and "train_loss_critic" in metrics_m1
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_m1.m1.parameters()
    ), "m1 must receive non-zero gradients during train_m1!"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model_m1.critic.parameters()
    ), "critic must receive non-zero gradients during train_m1!"
    # Representation modules must be frozen in Phase 1
    assert all(p.grad is None for p in model_m1.encoder.parameters())
    assert all(p.grad is None for p in model_m1.dynamics.parameters())
    assert all(p.grad is None for p in model_m1.sensory_predictor.parameters())
    assert all(p.grad is None for p in model_m1.physical_decoder.parameters())


def test_m1_active_perception_soft_fade_handover() -> None:
    """Verify M1PhaseStrategy continuously unrolls M1 while softly blending grounded dynamics during Phase 1.5."""
    B, T = 2, 12
    cfg = _make_small_config("train_m1")
    model = SPWM.from_config(cfg)
    loss_fn = SPWMLoss.from_config(cfg.loss)
    strategy = M1PhaseStrategy(model, cfg, loss_fn, torch.device("cpu"))

    events = torch.rand(B, T, 4, 16, 16)
    actions = torch.randn(B, T, 6)
    flat_kin = torch.randn(B, T, 12)
    # Timesteps: 0..1 setup, 2..4 Phase 1 (0), 5..7 Phase 1.5 (1), 8..11 Phase 2 (2)
    phase_flags = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]] * B, dtype=torch.long)

    m1_call_count = 0
    orig_m1_forward = model.m1.forward

    def counted_m1_forward(*args, **kwargs):
        nonlocal m1_call_count
        m1_call_count += 1
        return orig_m1_forward(*args, **kwargs)

    model.m1.forward = counted_m1_forward  # type: ignore[method-assign]
    loss, metrics = strategy._compute_m1_objectives(events, actions, flat_kin, phase_flags)
    # In continuous soft-fade handover, M1 unrolls continuously across all 9 steps
    # without stopping, maintaining motor inertia and continuous dynamics.
    assert m1_call_count == 9
    assert loss.requires_grad
    assert metrics["loss"] >= 0.0


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


def _assert_only_primitives_and_tensors(obj, path="root"):
    if isinstance(obj, torch.Tensor):
        return
    if obj is None or type(obj) in (bool, int, float, str):
        return
    if type(obj) is dict:
        for k, v in obj.items():
            assert type(k) in (bool, int, float, str), f"Non-primitive key {type(k)} at {path}"
            _assert_only_primitives_and_tensors(v, f"{path}.{k}")
        return
    if type(obj) in (list, tuple):
        for idx, item in enumerate(obj):
            _assert_only_primitives_and_tensors(item, f"{path}[{idx}]")
        return
    raise AssertionError(f"Non-primitive object {type(obj).__name__} found at {path}")


def test_trainer_checkpoint_roundtrip_strict_and_primitives_only(tmp_path: Path) -> None:
    """Verify BaseTrainer.save_checkpoint()/load_checkpoint() strict round-trip and zero pickled class instances."""
    import spwm.training as training_pkg
    import spwm.training.checkpoints as ckpts_mod
    import spwm.training.engine as engine_mod
    import spwm.training.replay_buffer as buf_mod

    # Backward-compatible re-exports in engine.py and spwm.training.__init__.py
    for sym in (
        "ClosedLoopBuffer",
        "EMATracker",
    ):
        assert getattr(engine_mod, sym) is getattr(buf_mod, sym)
        assert getattr(training_pkg, sym) is getattr(buf_mod, sym)
    for sym in (
        "load_representation_checkpoint",
        "resolve_version_checkpoint",
        "resolve_closed_loop_buffer",
        "enforce_phase_prerequisites",
    ):
        assert getattr(engine_mod, sym) is getattr(ckpts_mod, sym)
        assert getattr(training_pkg, sym) is getattr(ckpts_mod, sym)

    cfg = _make_small_config("train_m1")
    model_src = SPWM.from_config(cfg)
    trainer_src = BaseTrainer(model_src, cfg, device="cpu", save_dir=tmp_path)
    assert trainer_src.ema_tracker is not None
    assert trainer_src.replay_buffer is not None

    trainer_src.replay_buffer.push_trajectories(
        [
            {
                "z_init": torch.randn(cfg.model.latent_dim),
                "actions": torch.randn(5, cfg.environment.action_dim),
                "targets": torch.randn(5, cfg.model.latent_dim),
                "length": 5,
                "trajectory_id": 11,
                "is_expert": True,
                "elevation_plane": "High",
            }
        ]
    )
    trainer_src.best_score = 0.1234
    trainer_src.best_epoch = 3
    trainer_src.history = [{"epoch": 3, "train_loss": 0.2, "val_loss": 0.1234}]
    trainer_src.curriculum.current_curriculum_phase = 2

    sd = trainer_src.state_dict(epoch=3)
    _assert_only_primitives_and_tensors(sd, "trainer.state_dict")

    ckpt_path = tmp_path / "roundtrip_ckpt.pt"
    trainer_src.save_checkpoint(epoch=3, is_best=True, path=ckpt_path)

    # Must deserialize safely with weights_only=True (zero class instances pickled)
    raw_loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    _assert_only_primitives_and_tensors(raw_loaded, "roundtrip_ckpt.pt")

    model_dst = SPWM.from_config(cfg)
    with torch.no_grad():
        for p in model_dst.parameters():
            p.zero_()
    trainer_dst = BaseTrainer(model_dst, cfg, device="cpu", save_dir=tmp_path)
    trainer_dst.load_checkpoint(ckpt_path)

    for p_s, p_d in zip(model_src.parameters(), model_dst.parameters()):
        assert torch.allclose(p_s, p_d)
    assert trainer_dst.start_epoch == 4
    assert trainer_dst.best_score == pytest.approx(0.1234)
    assert trainer_dst.best_epoch == 3
    assert trainer_dst.history == [{"epoch": 3, "train_loss": 0.2, "val_loss": 0.1234}]
    assert trainer_dst.current_curriculum_phase == 2
    assert trainer_dst.replay_buffer is not None
    assert len(trainer_dst.replay_buffer.trajectories) == 1
    assert trainer_dst.replay_buffer.trajectories[0]["trajectory_id"] == 11
    assert trainer_dst.ema_tracker is not None
    for k, v in trainer_src.ema_tracker.state_dict().items():
        assert torch.allclose(v, trainer_dst.ema_tracker.state_dict()[k])

    # Non-primitive object in trainer state must fail loudly instead of pickling class instances
    trainer_src.history = [{"epoch": 3, "bad_obj": cfg}]  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Non-serializable object"):
        trainer_src.state_dict(epoch=3)


def test_phase0_homeostasis_se3_rot_and_m1_smooth_loss_contracts() -> None:
    """Verify v0.12.3 Phase 0 spike homeostasis + temporal kinematic smoothness + SE(3) losses and AR(1) noise scaling."""
    from spwm.models.kinematics import DifferentiableForwardKinematics
    from spwm.training.losses import compute_m1_loss, compute_representation_loss
    from spwm.training.noise import sample_ar1_noise

    cfg = _make_small_config("train_representation")
    model = SPWM.from_config(cfg)
    B, T, J_cnt = 2, 8, 6
    events = torch.rand(B, T, 4, 16, 16)
    actions = torch.zeros(B, T, 6)
    latents, err, spikes = model.forward_representation_with_spikes(events, actions=actions)
    assert latents.shape == (B, T, cfg.model.latent_dim)
    assert err.shape == (B, T, cfg.model.encoder_dim)
    assert spikes.shape == (B, T, sum(cfg.memory.timescale_dims))

    q_dim = model.q_dim
    q, p = latents[..., :q_dim], latents[..., q_dim:]
    pos_pred = model.physical_decoder.decode_pos(q)
    vel_pred = model.physical_decoder.decode_vel(p)
    flat_kin = torch.zeros(B, T, 2 * J_cnt)
    fk = DifferentiableForwardKinematics()

    out = compute_representation_loss(
        q=q, p=p, pos_pred=pos_pred, vel_pred=vel_pred, error_neurons=err,
        flat_kin=flat_kin, fk=fk, joint_count=J_cnt, dt=0.05, spikes=spikes,
    )
    assert out["loss_kinematic_smooth"].item() >= 0.0
    assert out["loss_spike_balance"].item() >= 0.0
    assert torch.isfinite(out["total_loss"])
    out["total_loss"].backward()
    assert any(
        param.grad is not None and param.grad.abs().sum() > 0 for param in model.dynamics.parameters()
    )

    m1_traj = torch.randn(B, T, cfg.model.latent_dim, requires_grad=True)
    target_latents = torch.zeros(B, T, cfg.model.latent_dim)
    out_m1_smooth = compute_m1_loss(
        m1_traj=m1_traj, target_latents=target_latents, q_dim=q_dim,
        critic=model.critic, physical_decoder=model.physical_decoder, fk=fk,
        target_kin=flat_kin, joint_count=J_cnt, dt=0.05, ee_rot_weight=0.5, lambda_kinematic_smooth=0.5,
    )
    out_m1_nosmooth = compute_m1_loss(
        m1_traj=m1_traj.detach(), target_latents=target_latents, q_dim=q_dim,
        critic=model.critic, physical_decoder=model.physical_decoder, fk=fk,
        target_kin=flat_kin, joint_count=J_cnt, dt=0.05, ee_rot_weight=0.0, lambda_kinematic_smooth=0.0,
    )
    assert out_m1_smooth["loss_kin_m1"].item() >= out_m1_nosmooth["loss_kin_m1"].item()

    noise = sample_ar1_noise(4, 16, 6, torch.device("cpu"), beta=0.90, base_sigma=0.08, shoulder_multiplier=1.6, elbow_multiplier=1.4)
    assert noise.shape == (4, 16, 6)
    assert torch.isfinite(noise).all()
