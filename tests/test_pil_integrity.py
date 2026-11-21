"""
Integrity tests for Physics-in-the-Loop (PIL) closed-loop evaluation:
1. Rigid payload attachment occurs at contact onset (t >= t_c), not t >= t_grasp.
2. All baselines (_BaselineModelWrapper for ann_rate, rssm, deep_gru) receive causal
   online sensory observations (ev_online) during phase_flags == 1 in PIL mode.
"""

from __future__ import annotations

import pytest
import torch

from experiments.benchmark_mppi_control import (
    _BaselineModelWrapper,
    _DeepGRUDecoderAdapter,
    _DeepGRUPredictorAdapter,
    _RSSMDecoderAdapter,
    _RSSMPredictorAdapter,
    _compute_residual_handover_weight,
    run_episode,
)
from spwm.contracts.config import ExperimentConfig, MPPIConfig
from spwm.data.sim_6dof import ArticulatedArmPhysicsStepper
from spwm.evaluation.mppi import LatentMPPIController, MPPIStructuralScorer
from spwm.models.ann_ablation import ANNWorldModel
from spwm.models.baselines import ContinuousRSSM, DeepGRUPredictor
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.world_model import SPWM


def _make_synthetic_episode(t_seq: int = 35, t_c: int = 22):
    events = torch.randint(0, 2, (1, t_seq, 4, 32, 32)).float()
    q_gt = torch.linspace(0.0, 0.4, t_seq).unsqueeze(-1).repeat(1, 6)
    p_gt = torch.ones(t_seq, 6) * 0.1
    actions = torch.zeros(1, t_seq, 6)
    flags = torch.zeros(t_seq, dtype=torch.long)
    flags[t_c - 4 : min(t_seq, t_c + 7)] = 1
    if t_c + 7 < t_seq:
        flags[t_c + 7 :] = 2
    return events, q_gt, p_gt, actions, flags


def test_pil_payload_attaches_at_contact_onset_t_c(monkeypatch):
    """Verify step_pd_target receives payload_attached=(t >= t_c) in both run_episode and mppi_scorer."""
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        mppi=MPPIConfig(num_samples=8, horizon=4, num_iterations=1),
    )
    model = SPWM.from_config(cfg).to(device)
    model.eval()
    fk = DifferentiableForwardKinematics().to(device)
    ctrl = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        config=cfg.mppi,
        forward_kinematics=fk,
        device=device,
    )

    recorded_flags = []
    orig_step = ArticulatedArmPhysicsStepper.step_pd_target

    def spy_step(self, q_des, dq_des, payload_attached, kp=18.0, kd=0.80):
        recorded_flags.append(bool(payload_attached))
        return orig_step(self, q_des, dq_des, payload_attached=payload_attached, kp=kp, kd=kd)

    monkeypatch.setattr(ArticulatedArmPhysicsStepper, "step_pd_target", spy_step)

    t_c = 22
    events, q_gt, p_gt, actions, flags = _make_synthetic_episode(t_seq=35, t_c=t_c)
    run_episode(
        model=model,
        controller=ctrl,
        fk=fk,
        events=events,
        q_gt=q_gt,
        phase_flags=flags,
        traj_id=0,
        payload_mass=3.0,
        payload_shape="Box",
        device=device,
        actions=actions,
        approach_window=10,
        p_gt=p_gt,
        physics_in_loop=True,
    )
    # t runs from t_start=12 to t_grasp=29 (18 steps): t < 22 -> False (10 steps), t >= 22 -> True (8 steps)
    assert len(recorded_flags) == 18
    assert recorded_flags[:10] == [False] * 10
    assert recorded_flags[10:] == [True] * 8

    # Also verify mppi_scorer in PIL mode
    recorded_flags.clear()
    scorer = MPPIStructuralScorer(
        fixed_batches=[
            {
                "trajectory_id": 0,
                "events": events,
                "joint_positions": q_gt.unsqueeze(0),
                "joint_velocities": p_gt.unsqueeze(0),
                "actions": actions,
                "phase_flags": flags.unsqueeze(0),
            }
        ],
        device=device,
        frequency=5,
        fk=fk,
        approach_window=10,
        config=cfg.mppi,
    )
    scorer.evaluate_trajectory(
        model=model,
        controller=ctrl,
        batch=scorer.fixed_batches[0],
        seed=42,
        physics_in_loop=True,
    )
    assert recorded_flags[:10] == [False] * 10
    assert recorded_flags[10:] == [True] * 8


def test_baselines_receive_online_observations_in_pil(monkeypatch):
    """Verify ANN-Rate, Continuous RSSM, and Deep GRU invoke online observation updates during phi_t == 1."""
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        mppi=MPPIConfig(num_samples=8, horizon=4, num_iterations=1),
    )
    fk = DifferentiableForwardKinematics().to(device)
    events, q_gt, p_gt, actions, flags = _make_synthetic_episode(t_seq=35, t_c=22)

    # 1. ANN-Rate
    ann = ANNWorldModel.from_config(cfg).to(device)
    ann.eval()
    wrapper_ann = _BaselineModelWrapper(ann, "ann_rate", ann.m1, ann.physical_decoder)
    calls_ann = 0
    orig_ann_step = wrapper_ann.step_online

    def spy_ann(ev_online, action, curr_z, dyn_state):
        nonlocal calls_ann
        calls_ann += 1
        return orig_ann_step(ev_online, action, curr_z, dyn_state)

    monkeypatch.setattr(wrapper_ann, "step_online", spy_ann)
    ctrl_ann = LatentMPPIController(ann.m1, ann.physical_decoder, cfg.mppi, fk, device=device)
    run_episode(
        model=wrapper_ann, controller=ctrl_ann, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=1, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt, physics_in_loop=True,
    )
    assert calls_ann == 18

    # 2. Continuous RSSM
    rssm = ContinuousRSSM().to(device)
    rssm.eval()
    p_rssm = _RSSMPredictorAdapter(rssm)
    d_rssm = _RSSMDecoderAdapter(rssm)
    wrapper_rssm = _BaselineModelWrapper(rssm, "rssm", p_rssm, d_rssm)
    calls_rssm = 0
    orig_rssm_step = wrapper_rssm.step_online

    def spy_rssm(ev_online, action, curr_z, dyn_state):
        nonlocal calls_rssm
        calls_rssm += 1
        return orig_rssm_step(ev_online, action, curr_z, dyn_state)

    monkeypatch.setattr(wrapper_rssm, "step_online", spy_rssm)
    ctrl_rssm = LatentMPPIController(p_rssm, d_rssm, cfg.mppi, fk, device=device)
    run_episode(
        model=wrapper_rssm, controller=ctrl_rssm, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=2, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt, physics_in_loop=True,
    )
    # p1_indices[0] == 18 (t_c in run_episode), t_grasp = 25 -> flags == 1 on [18..25] (8 steps)
    assert calls_rssm == 8

    # 3. Deep GRU
    gru = DeepGRUPredictor().to(device)
    gru.eval()
    p_gru = _DeepGRUPredictorAdapter(gru)
    d_gru = _DeepGRUDecoderAdapter(gru)
    wrapper_gru = _BaselineModelWrapper(gru, "deep_gru", p_gru, d_gru)
    calls_gru = 0
    orig_gru_step = wrapper_gru.step_online

    def spy_gru(ev_online, action, curr_z, dyn_state):
        nonlocal calls_gru
        calls_gru += 1
        return orig_gru_step(ev_online, action, curr_z, dyn_state)

    monkeypatch.setattr(wrapper_gru, "step_online", spy_gru)
    ctrl_gru = LatentMPPIController(p_gru, d_gru, cfg.mppi, fk, device=device)
    ep_gru = run_episode(
        model=wrapper_gru, controller=ctrl_gru, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=3, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt, physics_in_loop=True,
    )
    assert calls_gru == 8
    assert len(ep_gru.handover_weights_phi1) == 8

    # 4. Open-Loop PIL mode (use_handover=False) skips online sensory overwrite for RSSM & Deep GRU
    calls_rssm = 0
    ep_rssm_ol = run_episode(
        model=wrapper_rssm, controller=ctrl_rssm, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=2, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt,
        use_handover=False, physics_in_loop=True,
    )
    assert calls_rssm == 0
    assert ep_rssm_ol.handover_weights_phi1 == [0.0] * 8

    calls_gru = 0
    ep_gru_ol = run_episode(
        model=wrapper_gru, controller=ctrl_gru, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=3, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt,
        use_handover=False, physics_in_loop=True,
    )
    assert calls_gru == 0
    assert ep_gru_ol.handover_weights_phi1 == [0.0] * 8


def test_baselines_residual_gated_handover_in_pil(monkeypatch):
    """Verify RSSM and Deep GRU apply the identical FK residual gate vs ungated and open_loop modes."""
    device = torch.device("cpu")
    cfg = ExperimentConfig(
        mppi=MPPIConfig(num_samples=8, horizon=4, num_iterations=1),
    )
    fk = DifferentiableForwardKinematics().to(device)
    events, q_gt, p_gt, actions, flags = _make_synthetic_episode(t_seq=35, t_c=22)

    rssm = ContinuousRSSM().to(device)
    rssm.eval()
    p_rssm = _RSSMPredictorAdapter(rssm)
    d_rssm = _RSSMDecoderAdapter(rssm)
    wrapper_rssm = _BaselineModelWrapper(rssm, "rssm", p_rssm, d_rssm)
    ctrl_rssm = LatentMPPIController(p_rssm, d_rssm, cfg.mppi, fk, device=device)

    gru = DeepGRUPredictor().to(device)
    gru.eval()
    p_gru = _DeepGRUPredictorAdapter(gru)
    d_gru = _DeepGRUDecoderAdapter(gru)
    wrapper_gru = _BaselineModelWrapper(gru, "deep_gru", p_gru, d_gru)
    ctrl_gru = LatentMPPIController(p_gru, d_gru, cfg.mppi, fk, device=device)

    gate_calls = 0
    orig_gate = _compute_residual_handover_weight

    def spy_gate(fk_arg, decoder_arg, z_pred, z_obs, target_final_mat, t, t_c, dist_threshold=0.026):
        nonlocal gate_calls
        gate_calls += 1
        return orig_gate(fk_arg, decoder_arg, z_pred, z_obs, target_final_mat, t, t_c, dist_threshold=dist_threshold)

    monkeypatch.setattr("experiments.benchmark_mppi_control._compute_residual_handover_weight", spy_gate)

    ep_rssm_rg = run_episode(
        model=wrapper_rssm, controller=ctrl_rssm, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=2, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt,
        physics_in_loop=True, baseline_handover_mode="residual_gated",
    )
    assert gate_calls == 8
    assert all(0.0 <= w <= 0.85 for w in ep_rssm_rg.handover_weights_phi1)

    gate_calls = 0
    ep_gru_rg = run_episode(
        model=wrapper_gru, controller=ctrl_gru, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=3, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt,
        physics_in_loop=True, baseline_handover_mode="residual_gated",
    )
    assert gate_calls == 8
    assert all(0.0 <= w <= 0.85 for w in ep_gru_rg.handover_weights_phi1)

    # Ungated mode forces w_t = 1.0 without invoking the residual gate
    gate_calls = 0
    ep_rssm_ug = run_episode(
        model=wrapper_rssm, controller=ctrl_rssm, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=2, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt,
        physics_in_loop=True, baseline_handover_mode="ungated",
    )
    assert gate_calls == 0
    assert ep_rssm_ug.handover_weights_phi1 == [1.0] * 8

    # Open-loop baseline mode keeps w_t = 0.0
    ep_gru_ol_mode = run_episode(
        model=wrapper_gru, controller=ctrl_gru, fk=fk, events=events, q_gt=q_gt,
        phase_flags=flags, traj_id=3, payload_mass=2.5, payload_shape="Box",
        device=device, actions=actions, approach_window=10, p_gt=p_gt,
        physics_in_loop=True, baseline_handover_mode="open_loop",
    )
    assert ep_gru_ol_mode.handover_weights_phi1 == [0.0] * 8

    with pytest.raises(ValueError):
        run_episode(
            model=wrapper_rssm, controller=ctrl_rssm, fk=fk, events=events, q_gt=q_gt,
            phase_flags=flags, traj_id=2, payload_mass=2.5, payload_shape="Box",
            device=device, actions=actions, approach_window=10, p_gt=p_gt,
            physics_in_loop=True, baseline_handover_mode="invalid_mode",
        )
