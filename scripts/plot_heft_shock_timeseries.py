"""
Publication-Grade Time-Series Trajectory Comparison During +10 cm Contact Heft Shock.

Runs a closed-loop Physics-in-the-Loop (PIL) episode on canonical test trajectory 3622
for SPWM vs. Continuous RSSM vs. Ground Truth, recording:
  - Panel A: Shoulder Lift (q_1) & Elbow Flex (q_2) joint trajectories
  - Panel B: End-effector vertical lift (+10 cm) & Cartesian tracking error (cm)
  - Panel C: Joint velocity norm ||dq/dt||_2 & residual handover gate weight w_t
  - Panel D: Two-tier ALIF spike raster (Fast pool 0-127 vs. Slow pool 128-255)
Saves PDF + PNG to figures/heft_shock_timeseries.{pdf,png}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.benchmark_mppi_control import (
    _BaselineModelWrapper,
    _RSSMPredictorAdapter,
    _RSSMDecoderAdapter,
    load_trajectory_metadata,
)
from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import EventWorldDataset
from spwm.data.event_camera import EventCameraSimulator, OnlineEventCameraStream
from spwm.data.sim_6dof import ArticulatedArmPhysicsStepper
from spwm.evaluation.mppi import LatentMPPIController
from spwm.models.baselines import ContinuousRSSM
from spwm.models.kinematics import DifferentiableForwardKinematics, compute_rot_error
from spwm.models.rollout import ground_initial_state, step_grounded
from spwm.models.world_model import SPWM


def run_and_record_spwm(
    model: SPWM,
    ctrl: LatentMPPIController,
    fk: DifferentiableForwardKinematics,
    b: dict,
    mass: float,
    shape: str,
    device: torch.device,
) -> dict:
    events = b["events"].unsqueeze(0).to(device)
    q_gt = b["joint_positions"].to(device)
    p_gt = b["joint_velocities"].to(device)
    actions = b["actions"].unsqueeze(0).to(device)
    pf = b["phase_flags"].to(device)

    p1 = (pf == 1).nonzero()
    t_c = int(p1[0].item()) if len(p1) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
    t_start = max(5, t_c - 20)

    tg = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
    tf = fk.forward_homogeneous(q_gt[t_grasp : t_grasp + 1])[0].detach()

    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=t_start, actions=actions)
    dec_s = model.physical_decoder(z_start)
    init_u = dec_s[0, :6] / 3.14
    ctrl.reset(initial_action_bias=init_u, seed=42000 + int(b["trajectory_id"]))
    prev_u = init_u.clone()
    curr_z = z_start.clone()

    stepper = ArticulatedArmPhysicsStepper(q_gt[t_start].numpy(), p_gt[t_start].numpy(), mass, shape)
    ev_stream = OnlineEventCameraStream(EventCameraSimulator(32, 32, 0.08, in_channels=4), stepper.q)

    t_list, q_list, p_list, ee_y_list, err_list, w_list, spikes_list = [], [], [], [], [], [], []
    for t in range(t_start, t_grasp + 1):
        te = tg if t <= t_c else tf
        tp = torch.stack([tg if (t + tau) <= t_c else tf for tau in range(ctrl.horizon)], dim=0)
        res = ctrl.plan_step(curr_z, tp, prev_action=prev_u)
        u_exec = res.action
        prev_u = u_exec
        ctrl.prev_action = u_exec.clone()

        z_pred = model.predictor(curr_z, u_exec.unsqueeze(0)).predicted_latent
        dec_p = model.physical_decoder(z_pred)
        qn, pn, u_pd = stepper.step_pd_target(
            dec_p[0, :6].detach().numpy(),
            dec_p[0, 6:12].detach().numpy(),
            payload_attached=(t >= t_c),
        )
        ev_on = ev_stream.step(qn).to(device)
        z_obs, dyn_state = step_grounded(
            model, ev_on, dyn_state, in_sensory_phase=True, unconstrained=False, action=torch.from_numpy(u_pd).unsqueeze(0)
        )
        # Record ALIF spikes from dyn_state.memory_state[0] (fast/slow concatenated) or ema_spikes
        spk = dyn_state.ema_spikes[0].detach().cpu().numpy() if dyn_state.ema_spikes is not None else np.zeros(256)
        spikes_list.append(spk)

        w_t = 0.0
        if int(pf[t].item()) == 1:
            ee_p = fk(dec_p[:, :6])[0].detach()
            dp = float(torch.norm(ee_p - tf[:3, 3]).item())
            dec_o = model.physical_decoder(z_obs)
            do = float(torch.norm(fk(dec_o[:, :6])[0].detach() - tf[:3, 3]).item())
            ro = compute_rot_error(fk, dec_o[0, :6], tf)
            if dp > 0.026 and do < 0.025 and ro < 0.049:
                w_t = 0.85
            elif do < dp and t <= t_c + 2:
                w_t = 0.40 * max(0.0, min(1.0, (dp - 0.032) / 0.04))
            curr_z = (1.0 - w_t) * z_pred + w_t * z_obs
        else:
            curr_z = z_pred

        q_t = torch.from_numpy(qn)
        ee_t = fk(q_t.unsqueeze(0))[0].detach()
        err_cm = float((torch.norm(ee_t - te[:3, 3]) * 100.0).item())
        t_list.append(t)
        q_list.append(qn.copy())
        p_list.append(pn.copy())
        ee_y_list.append(float(ee_t[1].item()))
        err_list.append(err_cm)
        w_list.append(w_t)

    return {
        "t": np.array(t_list),
        "t_c": t_c,
        "t_grasp": t_grasp,
        "q": np.stack(q_list),
        "p": np.stack(p_list),
        "ee_y": np.array(ee_y_list),
        "err_cm": np.array(err_list),
        "w_t": np.array(w_list),
        "spikes": np.stack(spikes_list),
    }


def run_and_record_rssm(
    rssm: ContinuousRSSM,
    fk: DifferentiableForwardKinematics,
    b: dict,
    mass: float,
    shape: str,
    cfg: ExperimentConfig,
    device: torch.device,
) -> dict:
    pred_ad = _RSSMPredictorAdapter(rssm)
    dec_ad = _RSSMDecoderAdapter(rssm)
    wrapper = _BaselineModelWrapper(rssm, "rssm", pred_ad, dec_ad)
    ctrl = LatentMPPIController(
        predictor=pred_ad, physical_decoder=dec_ad, config=cfg.mppi, forward_kinematics=fk, device=device
    )

    events = b["events"].unsqueeze(0).to(device)
    q_gt = b["joint_positions"].to(device)
    p_gt = b["joint_velocities"].to(device)
    actions = b["actions"].unsqueeze(0).to(device)
    pf = b["phase_flags"].to(device)

    p1 = (pf == 1).nonzero()
    t_c = int(p1[0].item()) if len(p1) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
    t_start = max(5, t_c - 20)
    max_t = min(t_grasp + 1, events.shape[1] - 1)

    tg = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
    tf = fk.forward_homogeneous(q_gt[t_grasp : t_grasp + 1])[0].detach()

    z_gt_seq = wrapper.ground_sequence(events, max_t=max_t, actions=actions)
    curr_z = z_gt_seq[:, t_start].clone()
    init_u = dec_ad(curr_z)[0, :6] / 3.14
    ctrl.reset(initial_action_bias=init_u, seed=42000 + int(b["trajectory_id"]))
    prev_u = init_u.clone()

    stepper = ArticulatedArmPhysicsStepper(q_gt[t_start].numpy(), p_gt[t_start].numpy(), mass, shape)
    t_list, q_list, p_list, ee_y_list, err_list = [], [], [], [], []
    for t in range(t_start, t_grasp + 1):
        te = tg if t <= t_c else tf
        tp = torch.stack([tg if (t + tau) <= t_c else tf for tau in range(ctrl.horizon)], dim=0)
        res = ctrl.plan_step(curr_z, tp, prev_action=prev_u)
        u_exec = res.action
        prev_u = u_exec
        ctrl.prev_action = u_exec.clone()
        curr_z = pred_ad(curr_z, u_exec.unsqueeze(0)).predicted_latent
        dec_p = dec_ad(curr_z)
        qn, pn, _ = stepper.step_pd_target(
            dec_p[0, :6].detach().numpy(),
            dec_p[0, 6:12].detach().numpy(),
            payload_attached=(t >= t_c),
        )
        q_t = torch.from_numpy(qn)
        ee_t = fk(q_t.unsqueeze(0))[0].detach()
        err_cm = float((torch.norm(ee_t - te[:3, 3]) * 100.0).item())
        t_list.append(t)
        q_list.append(qn.copy())
        p_list.append(pn.copy())
        ee_y_list.append(float(ee_t[1].item()))
        err_list.append(err_cm)

    return {
        "t": np.array(t_list),
        "q": np.stack(q_list),
        "p": np.stack(p_list),
        "ee_y": np.array(ee_y_list),
        "err_cm": np.array(err_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-id", type=int, default=3622)
    args = parser.parse_args()

    cfg = ExperimentConfig.load("results/spwm_v0_12_4_train_m1/config.yaml")
    device = torch.device("cpu")
    ds = EventWorldDataset([args.traj_id], dataset_dir=Path("sim/ExportedTrajectories"))
    b = ds[0]
    mass, shape = load_trajectory_metadata(Path("sim/ExportedTrajectories"), args.traj_id)

    spwm = SPWM.from_config(cfg).to(device)
    spwm.load_state_dict(
        torch.load("results/spwm_v0_12_4_train_m1/model.pt", map_location=device, weights_only=False)["model_state"],
        strict=True,
    )
    spwm.eval()

    rssm = ContinuousRSSM(
        in_channels=cfg.model.in_channels,
        height=cfg.environment.height,
        width=cfg.environment.width,
        encoder_dim=cfg.model.encoder_dim,
        deterministic_dim=256,
        stochastic_dim=32,
        action_dim=cfg.environment.action_dim,
        joint_count=cfg.environment.joint_count,
    ).to(device)
    rssm.load_state_dict(
        torch.load("results/baselines/continuous_rssm/model.pt", map_location=device, weights_only=False)["model_state"],
        strict=True,
    )
    rssm.eval()

    fk = DifferentiableForwardKinematics(joint_count=6).to(device)
    ctrl_spwm = LatentMPPIController(
        predictor=spwm.m1, physical_decoder=spwm.physical_decoder, config=cfg.mppi, forward_kinematics=fk, device=device
    )

    with torch.no_grad():
        spwm_rec = run_and_record_spwm(spwm, ctrl_spwm, fk, b, mass, shape, device)
        rssm_rec = run_and_record_rssm(rssm, fk, b, mass, shape, cfg, device)

    t_arr = spwm_rec["t"]
    t_c = spwm_rec["t_c"]
    q_gt = b["joint_positions"][t_arr[0] : t_arr[-1] + 1].numpy()
    p_gt = b["joint_velocities"][t_arr[0] : t_arr[-1] + 1].numpy()
    ee_gt = fk(torch.from_numpy(q_gt)).detach().numpy()

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 10.0), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.15]})

    # Panel A: Joint 1 (Shoulder Lift) & Joint 2 (Elbow Flex)
    ax = axes[0]
    ax.plot(t_arr, q_gt[:, 1], "k--", lw=1.6, label="GT Shoulder Lift ($q_1$)")
    ax.plot(t_arr, spwm_rec["q"][:, 1], color="#0EA5E9", lw=2.0, label="SPWM ($q_1$)")
    ax.plot(t_arr, rssm_rec["q"][:, 1], color="#F97316", lw=1.6, alpha=0.85, label="RSSM ($q_1$)")
    ax.plot(t_arr, q_gt[:, 2], "k:", lw=1.4, label="GT Elbow Flex ($q_2$)")
    ax.plot(t_arr, spwm_rec["q"][:, 2], color="#10B981", lw=1.8, label="SPWM ($q_2$)")
    ax.axvline(t_c, color="#EF4444", ls="--", lw=1.3, label=f"Contact/Heft Onset ($t_c={t_c}$)")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title(f"Closed-Loop PIL +10 cm Contact Heft Shock (Traj {args.traj_id}, {mass:.2f} kg {shape})")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel B: Cartesian Tracking Error (cm) & End-Effector Lift (m)
    ax = axes[1]
    ax.plot(t_arr, spwm_rec["err_cm"], color="#0EA5E9", lw=2.2, label="SPWM Error (cm)")
    ax.plot(t_arr, rssm_rec["err_cm"], color="#F97316", lw=1.8, label="Continuous RSSM Error (cm)")
    ax.axhline(3.0, color="#64748B", ls=":", lw=1.2, label="3.0 cm Success Threshold")
    ax.axvline(t_c, color="#EF4444", ls="--", lw=1.3)
    ax.set_ylabel("EE Error (cm)")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel C: Joint Velocity Norm & Residual Handover Gate w_t
    ax = axes[2]
    ax.plot(t_arr, np.linalg.norm(p_gt, axis=-1), "k--", lw=1.4, label="GT $\\|\\dot{q}\\|_2$")
    ax.plot(t_arr, np.linalg.norm(spwm_rec["p"], axis=-1), color="#0EA5E9", lw=2.0, label="SPWM $\\|\\dot{q}\\|_2$")
    ax.plot(t_arr, np.linalg.norm(rssm_rec["p"], axis=-1), color="#F97316", lw=1.6, label="RSSM $\\|\\dot{q}\\|_2$")
    ax2 = ax.twinx()
    ax2.bar(t_arr, spwm_rec["w_t"], width=0.5, color="#8B5CF6", alpha=0.35, label="Handover Gate $w_t$")
    ax2.set_ylabel("Gate Weight $w_t$", color="#6D28D9")
    ax2.set_ylim(0.0, 1.0)
    ax.axvline(t_c, color="#EF4444", ls="--", lw=1.3)
    ax.set_ylabel("Vel Norm (rad/s)")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel D: Two-Tier ALIF Spike Activity Heatmap (Fast 0-127 vs Slow 128-255)
    ax = axes[3]
    spk_T = spwm_rec["spikes"].T  # [256, T_steps]
    im = ax.imshow(
        spk_T,
        aspect="auto",
        cmap="viridis",
        extent=[t_arr[0] - 0.5, t_arr[-1] + 0.5, 256, 0],
        interpolation="nearest",
    )
    ax.axhline(128, color="#F8FAFC", ls="-", lw=1.2)
    ax.axvline(t_c, color="#EF4444", ls="--", lw=1.5)
    ax.text(t_arr[0] + 0.5, 22, "Fast ALIF Pool ($\\tau=0.47$ s)", color="white", fontsize=8, fontweight="bold")
    ax.text(t_arr[0] + 0.5, 150, "Slow ALIF Pool ($\\tau=3.31$ s)", color="white", fontsize=8, fontweight="bold")
    ax.set_ylabel("ALIF Neuron Index")
    ax.set_xlabel(r"Simulation Step $t$ ($\Delta t = 0.05$ s)")
    plt.colorbar(im, ax=ax, label="EMA Spike Rate", pad=0.015)

    plt.tight_layout()
    Path("figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("figures/heft_shock_timeseries.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("figures/heft_shock_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved figures/heft_shock_timeseries.{pdf,png}")


if __name__ == "__main__":
    main()
