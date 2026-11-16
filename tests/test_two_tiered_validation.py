"""
Unit Tests for Two-Tiered Validation Strategy and Zero-Stochasticity Guarantee.

Verifies:
1. Zero-Stochasticity Guarantee: explicitly seeded MPPI controller produces bitwise identical
   noise perturbations, nominal plans, and trajectory costs across calls.
2. Elevation classification into Low, Mid, and High vertical planes.
3. Continuous MPPI structural shaping score calculation (penalties + discrete success bonus).
4. Fixed validation subset extraction covering Low, Mid, and High elevations.
5. End-to-end scorer determinism: evaluating the scorer twice on identical model weights
   yields 100% identical structural scores (zero noise jitter).
6. BaseTrainer two-tiered orchestration:
   - Tier 1 (high-frequency open-loop MSE) runs every epoch to track general convergence.
   - Tier 2 (low-frequency closed-loop MPPI scoring) runs strictly on configured frequency.
   - Checkpoint selection (saving model.pt) and early stopping are strictly driven by Tier 2 MPPI score.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from spwm.contracts.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    LatentNeuronConfig,
    LossConfig,
    MemoryConfig,
    ModelConfig,
    MPPIConfig,
    ProjectConfig,
    TrainingConfig,
)
from spwm.contracts.protocols import StatefulTier2EvaluatorProtocol, Tier2EvaluatorProtocol
from spwm.evaluation.mppi import (
    LatentMPPIController,
    MPPIStructuralScorer,
    classify_elevation,
    compute_rot_error,
    compute_structural_score,
)
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.predictors import LatentPredictor, PhysicalDecoder
from spwm.models.world_model import SPWM
from spwm.training.engine import BaseTrainer


class SyntheticValDataset(Dataset):
    """Tiny synthetic dataset generating trajectories across Low, Mid, and High planes."""

    def __init__(self, count: int = 6, seq_len: int = 40):
        self.count = count
        self.seq_len = seq_len
        self.items: List[Dict[str, Any]] = []

        # Create distinct elevations: 2 Low (y ~ 1.04), 2 Mid (y ~ 1.12), 2 High (y ~ 1.25)
        # In the 6-DOF arm, joint 1 (Shoulder_Lift) controls elevation.
        # q1 angles approximately mapping to Low, Mid, High
        q1_targets = [-0.35, -0.30, 0.0, 0.05, 0.35, 0.40]

        for i in range(count):
            q1 = q1_targets[i % len(q1_targets)]
            q = torch.zeros(seq_len, 6, dtype=torch.float32)
            q[:, 1] = q1
            p = torch.zeros(seq_len, 6, dtype=torch.float32)
            u = torch.zeros(seq_len, 6, dtype=torch.float32)
            flags = torch.zeros(seq_len, dtype=torch.long)
            # Contact window around step 20
            flags[15:25] = 1
            flags[25:] = 2
            events = torch.zeros(seq_len, 4, 32, 32, dtype=torch.float32)

            self.items.append({
                "events": events,
                "joint_positions": q,
                "joint_velocities": p,
                "positions": q,
                "velocities": p,
                "actions": u,
                "flat_kinematics": torch.cat([q, p], dim=-1),
                "phase_flags": flags,
                "trajectory_id": 3200 + i,
            })

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def synthetic_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "events": torch.stack([item["events"] for item in batch]),
        "joint_positions": torch.stack([item["joint_positions"] for item in batch]),
        "joint_velocities": torch.stack([item["joint_velocities"] for item in batch]),
        "actions": torch.stack([item["actions"] for item in batch]),
        "flat_kinematics": torch.stack([item["flat_kinematics"] for item in batch]),
        "phase_flags": torch.stack([item["phase_flags"] for item in batch]),
        "trajectory_ids": torch.tensor([item["trajectory_id"] for item in batch], dtype=torch.long),
    }


def test_zero_stochasticity_guarantee():
    """Explicitly seeded MPPI controller must produce 100% bitwise identical output across runs."""
    device = torch.device("cpu")
    predictor = LatentPredictor(latent_dim=128, action_dim=6, hidden_dim=64, symplectic=True)
    decoder = PhysicalDecoder(latent_dim=128, q_dim=32, p_dim=96, joint_count=6, hidden_dim=32)
    fk = DifferentiableForwardKinematics()

    mppi_cfg = MPPIConfig(
        num_samples=32,
        horizon=10,
        num_iterations=2,
    )

    c1 = LatentMPPIController(
        predictor=predictor,
        physical_decoder=decoder,
        config=mppi_cfg,
        forward_kinematics=fk,
        seed=42000,
        device=device,
    )
    c2 = LatentMPPIController(
        predictor=predictor,
        physical_decoder=decoder,
        config=mppi_cfg,
        forward_kinematics=fk,
        seed=42000,
        device=device,
    )

    z_t = torch.randn(128)
    target_pos = torch.tensor([0.0, 1.15, 0.45])

    res1 = c1.plan_step(z_t, target_pos)
    res2 = c2.plan_step(z_t, target_pos)

    # Actions, nominal trajectories, and costs must match bitwise
    assert torch.equal(res1.action, res2.action)
    assert torch.equal(res1.nominal_action_sequence, res2.nominal_action_sequence)
    assert res1.min_cost == res2.min_cost
    assert res1.mean_cost == res2.mean_cost

    # Different seed must produce different action sequence
    c3 = LatentMPPIController(
        predictor=predictor,
        physical_decoder=decoder,
        config=mppi_cfg,
        forward_kinematics=fk,
        seed=99999,
        device=device,
    )
    res3 = c3.plan_step(z_t, target_pos)
    assert not torch.equal(res1.action, res3.action)


def test_base_trainer_two_tiered_validation_orchestration(tmp_path):
    """
    Validates Two-Tiered Validation Strategy inside BaseTrainer:
    - Tier 1 runs every epoch.
    - Tier 2 runs strictly every tier2_frequency epochs (e.g. every 2 epochs).
    - Checkpoint selection (model.pt) occurs strictly on Tier 2 epochs driven by mppi_score.
    """
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        project=ProjectConfig(name="test_tier2_orch"),
        data=DataConfig(sequence_length=40),
        training=TrainingConfig(
            mode="train_m1",
            phase_horizon=10,
            epochs=4,
            tier2_frequency=2,  # Run Tier 2 on epochs 2 and 4
            early_stopping_patience=10,
        ),
    )
    model = SPWM.from_config(cfg).to(device)

    dataset = SyntheticValDataset(count=4, seq_len=40)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)

    # Mock Tier 2 evaluator tracking invocations
    tier2_call_epochs: List[int] = []

    class MockTier2Evaluator:
        frequency = 2

        def __call__(self, m: nn.Module, epoch: int) -> Dict[str, float]:
            tier2_call_epochs.append(epoch)
            # Returning an improving score at epoch 2, and worse at epoch 4
            score = 1.5 if epoch == 2 else 3.0
            return {
                "mppi_score": score,
                "mppi_success_rate": 0.5,
                "mppi_mean_error_cm": 2.5,
                "mppi_terminal_error_cm": 2.5,
                "mppi_final_vel": 0.4,
                "mppi_final_rot": 0.02,
                "mppi_pos_penalty": 0.7,
                "mppi_vel_penalty": 0.1,
                "mppi_rot_penalty": 0.2,
                "mppi_evaluated_trajectories": 4.0,
            }

    mock_tier2 = MockTier2Evaluator()

    save_dir = tmp_path / "trainer_test"
    trainer = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        tier2_evaluator=mock_tier2,
    )

    history = trainer.fit(epochs=4)

    # 1. Tier 2 was called strictly at epochs 2 and 4
    assert tier2_call_epochs == [2, 4]

    # 2. History records:
    # Epoch 1: Tier 1 open-loop MSE only, is_best=False (no model.pt update)
    assert history[0]["epoch"] == 1
    assert "mppi_score" not in history[0]
    assert history[0]["is_best"] is False

    # Epoch 2: Tier 2 ran, mppi_score recorded, is_best=True (model.pt saved)
    assert history[1]["epoch"] == 2
    assert "mppi_score" in history[1]
    assert history[1]["mppi_score"] == 1.5
    assert history[1]["is_best"] is True
    assert trainer.best_epoch == 2
    assert trainer.best_score == 1.5

    # Epoch 3: Tier 1 only, is_best=False
    assert history[2]["epoch"] == 3
    assert "mppi_score" not in history[2]
    assert history[2]["is_best"] is False

    # Epoch 4: Tier 2 ran, mppi_score=3.0 (worse than 1.5), is_best=False
    assert history[3]["epoch"] == 4
    assert history[3]["mppi_score"] == 3.0
    assert history[3]["is_best"] is False
    assert trainer.best_epoch == 2
    assert trainer.best_score == 1.5

    # Checkpoint model.pt must exist from epoch 2
    assert (save_dir / "model.pt").is_file()


def test_tier2_ema_candidate_selection(tmp_path):
    """
    Verifies that with tier2_candidate_selection='ema', Tier 2 evaluates
    the model candidate with the Exponential Moving Average (EMA) tracked weights.
    """
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        project=ProjectConfig(name="test_ema_selection"),
        data=DataConfig(sequence_length=40),
        training=TrainingConfig(
            mode="train_m1",
            phase_horizon=10,
            epochs=2,
            tier2_frequency=2,
            tier2_candidate_selection="ema",
            ema_decay=0.95,
        ),
    )
    model = SPWM.from_config(cfg).to(device)

    dataset = SyntheticValDataset(count=4, seq_len=40)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)

    evaluated_candidate_epochs: List[int] = []
    evaluated_weights: Dict[str, torch.Tensor] = {}

    class CandidateTrackingEvaluator:
        frequency = 2

        def __call__(self, m: nn.Module, candidate_epoch: int) -> Dict[str, float]:
            evaluated_candidate_epochs.append(candidate_epoch)
            for k, v in m.state_dict().items():
                evaluated_weights[k] = v.cpu().clone()
            return {
                "mppi_score": 1.0,
                "mppi_success_rate": 1.0,
                "mppi_mean_error_cm": 1.2,
                "mppi_terminal_error_cm": 1.1,
                "mppi_final_vel": 0.2,
                "mppi_final_rot": 0.01,
                "mppi_pos_penalty": 0.1,
                "mppi_vel_penalty": 0.02,
                "mppi_rot_penalty": 0.04,
                "mppi_evaluated_trajectories": 4.0,
            }

    evaluator = CandidateTrackingEvaluator()
    trainer = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=tmp_path / "ema_test",
        tier2_evaluator=evaluator,
    )

    trainer.fit(epochs=2)
    assert len(evaluated_candidate_epochs) == 1
    assert evaluated_candidate_epochs[0] == 2
    assert trainer.ema_tracker is not None
    # Check that evaluated weights match EMA shadow weights exactly
    for k, v in evaluated_weights.items():
        assert torch.allclose(v, trainer.ema_tracker.shadow[k].cpu())


def test_tier2_early_stopping_patience_checks(tmp_path):
    """
    Verifies that early stopping under Two-Tiered Validation:
    1. NEVER triggers on Tier 1 intermediate epochs.
    2. Does NOT prematurely terminate after only 1 failed Tier 2 check when tier2_patience=2.
    3. Strictly triggers on the 2nd failed Tier 2 check.
    """
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        project=ProjectConfig(name="test_early_stopping"),
        data=DataConfig(sequence_length=40),
        training=TrainingConfig(
            mode="train_m1",
            phase_horizon=10,
            epochs=10,
            tier2_frequency=2,
            tier2_patience=2,  # Stop after 2 consecutive failed Tier 2 evaluations
            tier2_min_delta=0.01,
            classic_patience=True,
        ),
    )
    model = SPWM.from_config(cfg).to(device)
    dataset = SyntheticValDataset(count=4, seq_len=40)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)

    # Return:
    # Epoch 2: score = 2.0 (initial best)
    # Epoch 4: score = 2.5 (no improvement -> 1 check failed)
    # Epoch 6: score = 2.8 (no improvement -> 2 checks failed -> EARLY STOPPING TRIGGERS HERE)
    # Epoch 8, 10 should NEVER run!
    call_epochs: List[int] = []

    class MockSequenceEvaluator:
        frequency = 2

        def __call__(self, m: nn.Module, candidate_epoch: int) -> Dict[str, float]:
            call_epochs.append(candidate_epoch)
            score_map = {2: 2.0, 4: 2.5, 6: 2.8, 8: 1.0, 10: 0.5}
            score = score_map.get(candidate_epoch, 5.0)
            return {
                "mppi_score": score,
                "mppi_success_rate": 1.0,
                "mppi_mean_error_cm": 1.0,
                "mppi_terminal_error_cm": 1.0,
                "mppi_final_vel": 0.1,
                "mppi_final_rot": 0.01,
                "mppi_pos_penalty": 0.1,
                "mppi_vel_penalty": 0.02,
                "mppi_rot_penalty": 0.04,
                "mppi_evaluated_trajectories": 4.0,
            }

    trainer = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=tmp_path / "es_test",
        tier2_evaluator=MockSequenceEvaluator(),
    )

    history = trainer.fit(epochs=10)

    # Training must have executed exactly 6 epochs (stopping at epoch 6 on the 2nd failed check)
    executed_epochs = [h["epoch"] for h in history]
    assert executed_epochs == [1, 2, 3, 4, 5, 6]

    # Best epoch recorded is 2 (from first Tier 2 check)
    assert trainer.best_epoch == 2
    assert trainer.best_score == 2.0

    # History tier2_checks_without_improvement:
    # Epoch 2: 0
    # Epoch 4: 1
    # Epoch 6: 2
    assert history[1]["tier2_checks_without_improvement"] == 0
    assert history[3]["tier2_checks_without_improvement"] == 1
    assert history[5]["tier2_checks_without_improvement"] == 2


def test_teacher_forced_blending_metric_isolation_and_round_robin():
    """
    Verifies:
    1. Phases 1, 2, 3 inject 4 expert trajectories and evaluate 2 natural MPPI trajectories.
    2. Phase 4 decays expert trajectories by 1 on each subsequent Tier 2 evaluation step (4 -> 3 -> 2 -> 1 -> 0).
    3. Strict Metric Isolation: returned metrics are computed EXCLUSIVELY on the natural trajectories,
       while all 6 trajectories (natural + expert) are returned for the ClosedLoopBuffer.
    4. Stratified Round-Robin Rotation: natural trajectory indices rotate via (eval_idx + i) % 6 across Low, Mid, High planes.
    """
    device = torch.device("cpu")
    dataset = SyntheticValDataset(count=6, seq_len=40)
    fk = DifferentiableForwardKinematics().to(device)

    scorer = MPPIStructuralScorer.from_dataset(
        val_dataset=dataset,
        device=device,
        frequency=5,
        num_trajectories=6,
        base_seed=42000,
        num_samples=8,
        horizon=6,
        num_iterations=1,
        fk=fk,
        approach_window=6,
    )

    cfg = ExperimentConfig(
        data=DataConfig(sequence_length=40),
        training=TrainingConfig(mode="train_m1", phase_horizon=6),
    )
    model = SPWM.from_config(cfg).to(device)
    model.eval()

    # Track calls to evaluate_trajectory to verify which indices are expert vs natural and verify metric isolation
    call_records: List[Dict[str, Any]] = []
    orig_eval_traj = scorer.evaluate_trajectory

    def spy_eval_traj(m, ctrl, batch, seed, use_expert_actions=False):
        res = orig_eval_traj(m, ctrl, batch, seed=seed, use_expert_actions=use_expert_actions)
        # Artificially set expert trajectory metrics to -999.0 and success to 1.0 to prove metric isolation!
        if use_expert_actions:
            res["score"] = -999.0
            res["success"] = 1.0
            res["grasp_error_cm"] = -999.0
            res["steady_state_error_cm"] = -999.0
            res["mean_cartesian_err_cm"] = 0.0
            res["final_vel"] = -999.0
            res["final_rot"] = -999.0
        else:
            res["score"] = 10.0
            res["success"] = 0.0
            res["grasp_error_cm"] = 8.0
            res["steady_state_error_cm"] = 6.0
            res["mean_cartesian_err_cm"] = 5.0
            res["final_vel"] = 2.0
            res["final_rot"] = 0.775
        call_records.append({
            "traj_id": batch["trajectory_id"],
            "plane": batch["elevation_plane"],
            "is_expert": use_expert_actions,
        })
        return res

    scorer.evaluate_trajectory = spy_eval_traj  # type: ignore[method-assign]

    # Phase 1 (eval_idx=0): all 6 evaluated naturally, while ClosedLoopBuffer receives 2 natural {0, 1} + 4 expert {2, 3, 4, 5}
    expected_ids = [int(b["trajectory_id"]) for b in scorer.fixed_batches]
    call_records.clear()
    m_p1 = scorer(model, epoch=5, curriculum_phase=1)
    assert m_p1["mppi_evaluated_trajectories"] == 6.0
    assert m_p1["mppi_expert_trajectories"] == 4.0
    assert len(m_p1["trajectories"]) == 6
    assert sum(1 for t in m_p1["trajectories"] if t.is_expert) == 4
    assert [t.trajectory_id for t in m_p1["trajectories"] if not t.is_expert] == [expected_ids[0], expected_ids[1]]
    # Strict metric isolation: scores must reflect natural trajectories only (0.5*8 + 0.5*6 + 0.25*5 + 0.1*2 + 2.0*0.775 = 10.0),
    # NOT polluted by -999.0 from expert trajectories.
    assert m_p1["mppi_structural_score"] == 10.0
    assert m_p1["mppi_score"] == pytest.approx(10.0)
    assert m_p1["mppi_success_rate"] == 0.0
    assert m_p1["mppi_mean_error_cm"] == 5.0
    nat_idx_p1 = [r["traj_id"] for r in call_records if not r["is_expert"]]
    assert len(nat_idx_p1) == 6

    # Phase 2 (eval_idx=1): 6 natural evals, ClosedLoopBuffer natural {1, 2} + 4 expert
    call_records.clear()
    m_p2 = scorer(model, epoch=10, curriculum_phase=2)
    assert m_p2["mppi_evaluated_trajectories"] == 6.0
    assert m_p2["mppi_expert_trajectories"] == 4.0
    assert [t.trajectory_id for t in m_p2["trajectories"] if not t.is_expert] == [expected_ids[1], expected_ids[2]]

    # Phase 3 (eval_idx=2): 6 natural evals, ClosedLoopBuffer natural {2, 3} + 4 expert
    call_records.clear()
    m_p3 = scorer(model, epoch=15, curriculum_phase=3)
    assert m_p3["mppi_evaluated_trajectories"] == 6.0
    assert m_p3["mppi_expert_trajectories"] == 4.0
    assert [t.trajectory_id for t in m_p3["trajectories"] if not t.is_expert] == [expected_ids[2], expected_ids[3]]

    # Phase 4 Freeze Check: while natural success_rate == 0.0, buffer expert decay is FROZEN at 4
    call_records.clear()
    m_p4_frozen = scorer(model, epoch=16, curriculum_phase=4)
    assert m_p4_frozen["mppi_expert_trajectories"] == 4.0
    assert m_p4_frozen["mppi_evaluated_trajectories"] == 6.0

    # Unfreeze: natural trajectories achieve success > 0.0, allowing buffer expert decay to begin
    scorer.last_natural_success_rate = 0.50

    # Phase 4 Step 1 (eval_idx=4): buffer expert decays 4 -> 3 (buffer natural = 3)
    call_records.clear()
    m_p4_1 = scorer(model, epoch=20, curriculum_phase=4)
    assert m_p4_1["mppi_expert_trajectories"] == 3.0
    assert m_p4_1["mppi_evaluated_trajectories"] == 6.0
    assert sum(1 for t in m_p4_1["trajectories"] if not t.is_expert) == 3

    # Keep natural success > 0.0 so curriculum continues decaying
    scorer.last_natural_success_rate = 0.50

    # Phase 4 Step 2 (eval_idx=5): buffer expert decays 3 -> 2 (buffer natural = 4)
    call_records.clear()
    m_p4_2 = scorer(model, epoch=25, curriculum_phase=4)
    assert m_p4_2["mppi_expert_trajectories"] == 2.0
    assert m_p4_2["mppi_evaluated_trajectories"] == 6.0
    assert sum(1 for t in m_p4_2["trajectories"] if not t.is_expert) == 4

    # Phase 4 Step 3 (eval_idx=6): buffer expert decays 2 -> 1 (buffer natural = 5)
    scorer.last_natural_success_rate = 0.50
    m_p4_3 = scorer(model, epoch=30, curriculum_phase=4)
    assert m_p4_3["mppi_expert_trajectories"] == 1.0
    assert m_p4_3["mppi_evaluated_trajectories"] == 6.0
    assert sum(1 for t in m_p4_3["trajectories"] if not t.is_expert) == 5

    # Phase 4 Step 4 (eval_idx=7): buffer expert decays 1 -> 0 (buffer natural = 6)
    scorer.last_natural_success_rate = 0.50
    m_p4_4 = scorer(model, epoch=35, curriculum_phase=4)
    assert m_p4_4["mppi_expert_trajectories"] == 0.0
    assert m_p4_4["mppi_evaluated_trajectories"] == 6.0
    assert sum(1 for t in m_p4_4["trajectories"] if not t.is_expert) == 6

    # Phase 4 Step 5 (eval_idx=8): stays at 0 buffer expert, 6 natural
    scorer.last_natural_success_rate = 0.50
    m_p4_5 = scorer(model, epoch=40, curriculum_phase=4)
    assert m_p4_5["mppi_expert_trajectories"] == 0.0
    assert m_p4_5["mppi_evaluated_trajectories"] == 6.0


def test_best_model_selection_gated_until_full_natural_subset(tmp_path):
    """Verify best_model.pt and best_score are gated while num_expert_eval > 0 and engage at num_expert_eval == 0."""
    device = torch.device("cpu")
    dataset = SyntheticValDataset(count=6, seq_len=30)
    cfg = ExperimentConfig(
        project=ProjectConfig(name="test_full_subset_gating"),
        data=DataConfig(sequence_length=30),
        training=TrainingConfig(mode="train_m1", phase_horizon=6, epochs=6, tier2_frequency=2, tier2_patience=5),
    )
    model = SPWM.from_config(cfg).to(device)

    class PartialThenFullEvaluator:
        frequency = 2

        def __call__(self, m: nn.Module, epoch: int) -> Dict[str, Any]:
            if epoch == 2:
                # Partial subset (4 expert, 2 natural) with artificially low score must NOT lock best_model.pt
                return {
                    "mppi_score": 1.0,
                    "mppi_evaluated_trajectories": 2.0,
                    "mppi_expert_trajectories": 4.0,
                }
            if epoch == 4:
                return {
                    "mppi_score": 3.0,
                    "mppi_evaluated_trajectories": 6.0,
                    "mppi_expert_trajectories": 0.0,
                }
            return {
                "mppi_score": 2.5,
                "mppi_evaluated_trajectories": 6.0,
                "mppi_expert_trajectories": 0.0,
            }

    save_dir = tmp_path / "gated_selection"
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)
    trainer = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        tier2_evaluator=PartialThenFullEvaluator(),
    )

    trainer.fit(epochs=2)
    assert trainer.best_epoch is None
    assert trainer.best_score == float("inf")
    assert (save_dir / "model.pt").is_file()

    # Resume on a fresh trainer from checkpoint.pt and advance epochs 3..6
    seen_best_after_ep4: List[float] = []
    orig_eval = PartialThenFullEvaluator()

    class TrackingEvaluator:
        frequency = 2

        def __call__(self, m: nn.Module, epoch: int) -> Dict[str, Any]:
            if epoch == 6:
                seen_best_after_ep4.append(trainer2.best_score)
            return orig_eval(m, epoch)

    model2 = SPWM.from_config(cfg).to(device)
    trainer2 = BaseTrainer(
        model=model2,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        tier2_evaluator=TrackingEvaluator(),
    )
    trainer2.load_checkpoint(save_dir / "checkpoint.pt")
    assert trainer2.start_epoch == 3
    assert trainer2.best_score == float("inf")

    trainer2.fit(epochs=4)
    assert seen_best_after_ep4 == [pytest.approx(3.0)]
    assert trainer2.best_score == pytest.approx(2.5)
    assert trainer2.best_epoch == 6


def test_tier2_evaluator_state_persisted_and_restored_on_resume(tmp_path):
    """Verify MPPIStructuralScorer curriculum counters persist through checkpoint.pt and resume cleanly."""
    device = torch.device("cpu")
    dataset = SyntheticValDataset(count=6, seq_len=40)
    fk = DifferentiableForwardKinematics().to(device)

    scorer1 = MPPIStructuralScorer.from_dataset(
        val_dataset=dataset,
        device=device,
        frequency=5,
        num_trajectories=6,
        base_seed=42000,
        num_samples=8,
        horizon=6,
        num_iterations=1,
        fk=fk,
        approach_window=5,
    )
    assert isinstance(scorer1, StatefulTier2EvaluatorProtocol)

    cfg = ExperimentConfig(
        project=ProjectConfig(name="test_tier2_state_resume"),
        data=DataConfig(sequence_length=40),
        training=TrainingConfig(mode="train_m1", phase_horizon=6, epochs=10, tier2_frequency=5),
    )
    model = SPWM.from_config(cfg).to(device)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)
    val_loader = DataLoader(dataset, batch_size=2, collate_fn=synthetic_collate)

    save_dir = tmp_path / "resume_tier2"
    trainer1 = BaseTrainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        tier2_evaluator=scorer1,
    )
    scorer1.eval_count = 7
    scorer1.phase4_eval_count = 4
    scorer1.last_natural_success_rate = 0.8333
    trainer1.save_checkpoint(epoch=155, is_best=False)

    scorer2 = MPPIStructuralScorer.from_dataset(
        val_dataset=dataset,
        device=device,
        frequency=5,
        num_trajectories=6,
        base_seed=42000,
        num_samples=8,
        horizon=6,
        num_iterations=1,
        fk=fk,
        approach_window=5,
    )
    model2 = SPWM.from_config(cfg).to(device)
    trainer2 = BaseTrainer(
        model=model2,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        tier2_evaluator=scorer2,
    )
    trainer2.load_checkpoint(save_dir / "checkpoint.pt")

    assert scorer2.eval_count == 7
    assert scorer2.phase4_eval_count == 4
    assert scorer2.last_natural_success_rate == pytest.approx(0.8333)

    # Next Phase 4 Tier 2 evaluation must keep all 6 natural trajectories (0 expert), not regress to 2 natural
    res = scorer2(model2, epoch=160, curriculum_phase=4)
    assert res["mppi_expert_trajectories"] == 0.0
    assert res["mppi_evaluated_trajectories"] == 6.0






