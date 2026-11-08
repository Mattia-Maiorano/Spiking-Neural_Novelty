"""
Figure 1: System Teaser (CoRL Publication Asset).

Generates:
  1. Panel (a) — Physical setup & latent MPPI predictive rollouts, rebuilt as two orthographic
     2D views (front X–Y, top X–Z) of the real 6-DOF kinematic chain (exact FK of the recorded
     joint angles), the head / wrist event cameras, the grasp target, and the K candidate
     end-effector rollouts of one genuine `LatentMPPIController.plan_step` call executed with the
     closed-loop benchmark protocol and hyper-parameters (`experiments/benchmark_mppi_control.py`).
  2. Panel (b) — Decoded joint-space position error ||q_hat_h - q*_h||_2 (rad) per horizon step
     h = 1..H on the full unseen test split, SPWM vs. parameter-matched continuous ANN-Rate,
     using the canonical rollout schedule of `experiments/train_ann_ablation.py`.
  3. Composite teaser `fig1_teaser` + standalone `fig1_mppi_setup`, `fig1_joint_error_horizon`.
  4. Latent phase-space views (`fig1_phase_space` 3D, `fig1_phase_space_2d` front view with the
     horizon step encoded as opacity) of a single trajectory, without drift annotations.

Zero AST Fallbacks strictly enforced.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from figure_style import PALETTE, save_publication_figure, set_paper_style
from spwm.contracts.config import ExperimentConfig
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import EventWorldDataset, collate_event_batches, partition_trajectories
from spwm.data.event_camera import EventCameraSimulator
from spwm.evaluation.mppi import LatentMPPIController
from spwm.models.ann_ablation import ANNWorldModel
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import compute_rollout_schedule, ground_initial_state, rollout_latent, step_grounded
from spwm.models.world_model import SPWM

# Head (global observer) camera pose, WORLD.md §1 (CinematicCameraTracker): world position and look target.
HEAD_CAMERA_POS: Tuple[float, float, float] = (-0.22, 1.60, -0.28)
HEAD_CAMERA_LOOK_AT: Tuple[float, float, float] = (-0.12, 1.06, 0.44)

# Closed-loop latent MPPI benchmark hyper-parameters (experiments/benchmark_mppi_control.py).
MPPI_NOISE_SIGMA: float = 0.20
MPPI_TEMPERATURE: float = 0.03
MPPI_APPROACH_WINDOW: int = 20
MPPI_ACTION_BIAS_SCALE: float = 3.14  # initial_bias = q_hat(z_start) / 3.14, as in the benchmark


# ==============================================================================
# Data / model helpers
# ==============================================================================
def load_world_model(model: nn.Module, ckpt_path: Path) -> nn.Module:
    """Strictly loads a checkpoint into `model` and switches to eval mode."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = raw["model_state"] if "model_state" in raw else raw["model_state_dict"]["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def build_test_loader(cfg: ExperimentConfig, trajectories_dir: Path, batch_size: int) -> DataLoader:
    """Unseen test split, identical partition and event simulator settings as `create_dataloaders`."""
    splits = partition_trajectories(cfg.data.total_trajectories, cfg.data.train_split, cfg.data.val_split)
    test_ids = list(range(splits["test"][0], splits["test"][1]))
    test_ds = EventWorldDataset(
        trajectory_indices=test_ids,
        dataset_dir=trajectories_dir,
        event_simulator=EventCameraSimulator(height=32, width=32, event_threshold=0.08, in_channels=4),
        sequence_length=cfg.data.sequence_length,
        joint_count=6,
        action_dim=6,
        velocity_range=(-1.0, 1.0),
        dt=0.05,
        cache_data=False,
        strict_dataset=True,
    )
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)


# ==============================================================================
# Panel (b): decoded joint-space error per horizon step
# ==============================================================================
@dataclass(frozen=True)
class JointErrorPerStep:
    """Per-trajectory, per-horizon-step decoded joint-space position errors."""

    errors: Dict[str, np.ndarray]   # model name -> [N, H] ||q_hat_h - q*_h||_2 (rad)
    heft_mask: np.ndarray           # [N, H] True where step h is a SensoryHeft step (real events injected)
    horizon: int


@torch.no_grad()
def compute_joint_error_per_step(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    horizon: int = 100,
    temporal_burn_in: int = 5,
    joint_count: int = 6,
) -> JointErrorPerStep:
    """
    Autonomous rollouts on the test split with the canonical schedule of
    `experiments/train_ann_ablation.py::evaluate_rollout`, but reporting the error at each
    individual horizon step h (not the running mean over steps 1..h).
    """
    errs: Dict[str, list] = {name: [] for name in models}
    heft: list = []
    for batch in test_loader:
        events = batch["events"]
        actions = batch["actions"]
        true_kin = batch["flat_kinematics"]
        phase_flags = batch["phase_flags"]
        t_start, max_eval_h = compute_rollout_schedule(
            phase_flags=phase_flags, sequence_length=events.shape[1], temporal_burn_in=temporal_burn_in,
        )
        if max_eval_h < horizon:
            raise ValueError(f"Rollout schedule allows only {max_eval_h} steps (< requested horizon {horizon})")
        target_q = true_kin[:, t_start + 1 : t_start + 1 + horizon, :joint_count]
        heft.append((phase_flags[:, t_start + 1 : t_start + 1 + horizon] == 1).numpy())
        for name, model in models.items():
            res = model.rollout(events=events, actions=actions, horizon=horizon, t_start=t_start, phase_flags=phase_flags)
            pred_q = res.decoded_kinematics[..., :joint_count]
            errs[name].append(torch.norm(pred_q - target_q, dim=-1).numpy().astype(np.float64))
    return JointErrorPerStep(
        errors={name: np.concatenate(v, axis=0) for name, v in errs.items()},
        heft_mask=np.concatenate(heft, axis=0),
        horizon=horizon,
    )


def draw_joint_error_panel(ax: plt.Axes, data: JointErrorPerStep, styles: Dict[str, Dict[str, object]]) -> None:
    """Mean ± inter-quartile band of per-step joint-space error vs. horizon step."""
    steps = np.arange(1, data.horizon + 1)

    # Sensory-heft background: opacity proportional to fraction of trajectories injecting real events at h
    heft_frac = data.heft_mask.mean(axis=0)
    for h, frac in zip(steps, heft_frac):
        if frac > 0.0:
            ax.axvspan(h - 0.5, h + 0.5, color=PALETTE["heft_border"], alpha=0.30 * float(frac), linewidth=0, zorder=0)

    for name, err in data.errors.items():
        st = styles[name]
        mean = err.mean(axis=0)
        q25, q75 = np.percentile(err, [25, 75], axis=0)
        ax.fill_between(steps, q25, q75, color=st["color"], alpha=0.16, linewidth=0, zorder=2)
        ax.plot(steps, mean, color=st["color"], linestyle=st["linestyle"], linewidth=2.2, label=st["label"], zorder=3)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=PALETTE["heft_border"], alpha=0.30))
    labels.append("Sensory heft (real events injected)")
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#64748B", alpha=0.16))
    labels.append("Inter-quartile range (25–75%)")
    ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.9, fontsize=7.4)

    ax.set_xlim(1, data.horizon)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"Horizon step $h$ (autonomous rollout, $\Delta t = 50\,\mathrm{ms}$)")
    ax.set_ylabel(r"Joint-space error $\|\hat{\mathbf{q}}_h - \mathbf{q}^*_h\|_2$ (rad)")


# ==============================================================================
# Panel (a): physical setup & latent MPPI rollouts
# ==============================================================================
def joint_chain_positions(fk: DifferentiableForwardKinematics, q: np.ndarray) -> np.ndarray:
    """
    World positions of every articulation pivot of the FK chain plus the end-effector tip.

    Mirrors `DifferentiableForwardKinematics.forward` exactly (Rodrigues joint rotations, local
    anchor translations) and asserts that the final tip matches `fk(q)`.

    Returns:
        [8, 3] = [base root, 6 joint pivots (Shoulder_Pan .. Wrist_Roll), tip].
    """
    if q.shape != (fk.joint_count,):
        raise ValueError(f"Expected q of shape ({fk.joint_count},), got {q.shape}")
    anchors = fk.anchors.numpy().astype(np.float64)
    axes = fk.axes.numpy().astype(np.float64)
    T = np.eye(4)
    T[:3, 3] = fk.base_origin.numpy() + anchors[0]
    points = [T[:3, 3].copy()]
    for j in range(fk.joint_count):
        a = axes[j]
        K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
        R = np.eye(3) + np.sin(q[j]) * K + (1.0 - np.cos(q[j])) * (K @ K)
        T_link = np.eye(4)
        T_link[:3, :3] = R
        T_link[:3, 3] = anchors[j + 1]
        T = T @ T_link
        points.append(T[:3, 3].copy())
    tip = T @ np.append(fk.tip_offset.numpy().astype(np.float64), 1.0)
    points.append(tip[:3])
    chain = np.stack(points, axis=0)

    tip_ref = fk(torch.tensor(q, dtype=torch.float32).unsqueeze(0))[0].numpy()
    if not np.allclose(chain[-1], tip_ref, atol=1e-4):
        raise RuntimeError(f"joint_chain_positions tip {chain[-1]} disagrees with FK tip {tip_ref}")
    return chain


@dataclass(frozen=True)
class MPPIScene:
    """Everything panel (a) draws, all in Unity world coordinates (X right, Y up, Z forward; metres)."""

    arm_chain: np.ndarray        # [8, 3] recorded arm pose at the planning step
    ee_recorded: np.ndarray      # [t_contact - t_plan + 1, 3] recorded EE path from planning step to contact
    ee_decoded_now: np.ndarray   # [3] EE position decoded from the grounded latent z_t
    candidates: np.ndarray       # [K, H, 3] candidate EE rollouts (final MPPI iteration)
    weights: np.ndarray          # [K] normalized path-integral weights
    best: np.ndarray             # [H, 3] lowest-cost candidate rollout
    target: np.ndarray           # [3] grasp target = FK(q*_{t_contact})
    t_plan: int
    t_contact: int
    payload_mass: float
    num_samples: int
    horizon: int


@torch.no_grad()
def compute_mppi_scene(
    model: SPWM,
    sample: Dict[str, torch.Tensor],
    fk: DifferentiableForwardKinematics,
    num_samples: int = 256,
    horizon: int = 20,
    num_iterations: int = 3,
    seed: int = 0,
) -> MPPIScene:
    """
    Reproduces the first control step of `benchmark_mppi_control.py::run_episode` on one trajectory:
    grounding up to t_plan = max(5, t_contact - 20), action-bias warm start, one grounded step,
    then one full `plan_step` (all iterations, full cost) toward the grasp target FK(q*_{t_contact}).
    """
    flags = sample["phase_flags"]
    heft_idx = (flags == 1).nonzero(as_tuple=False)
    if heft_idx.numel() == 0:
        raise ValueError("Trajectory has no SensoryHeft (phase flag 1) step; cannot define the grasp target")
    t_contact = int(heft_idx[0].item())
    t_plan = max(5, t_contact - MPPI_APPROACH_WINDOW)

    q_gt = sample["positions"]
    events = sample["events"].unsqueeze(0)
    target = fk(q_gt[t_contact : t_contact + 1])[0]

    controller = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        forward_kinematics=fk,
        num_samples=num_samples,
        horizon=horizon,
        num_iterations=num_iterations,
        noise_sigma=MPPI_NOISE_SIGMA,
        temperature=MPPI_TEMPERATURE,
        device=torch.device("cpu"),
    )

    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=t_plan)
    initial_bias = model.physical_decoder(z_start)[0, :6] / MPPI_ACTION_BIAS_SCALE
    controller.reset(initial_action_bias=initial_bias)
    z_t, dyn_state = step_grounded(
        model, events[:, t_plan], dyn_state, in_sensory_phase=True, unconstrained=True, action=initial_bias.unsqueeze(0),
    )

    torch.manual_seed(seed)
    res = controller.plan_step(z_t, target, prev_action=initial_bias)
    ee_now = fk(model.physical_decoder(z_t)[:, :6])[0]

    return MPPIScene(
        arm_chain=joint_chain_positions(fk, q_gt[t_plan].numpy().astype(np.float64)),
        ee_recorded=fk(q_gt[t_plan : t_contact + 1]).numpy(),
        ee_decoded_now=ee_now.numpy(),
        candidates=res.candidate_ee_trajectories.numpy(),
        weights=res.candidate_weights.numpy(),
        best=res.predicted_ee_trajectory.numpy(),
        target=target.numpy(),
        t_plan=t_plan,
        t_contact=t_contact,
        payload_mass=float("nan"),
        num_samples=num_samples,
        horizon=horizon,
    )


def _draw_orthographic_view(
    ax: plt.Axes,
    scene: MPPIScene,
    dims: Tuple[int, int],
    weight_norm: LogNorm,
    cmap: LinearSegmentedColormap,
    show_camera_frustum: bool,
) -> None:
    """Draws one orthographic projection (world axes `dims`) of the MPPI scene."""
    i, j = dims

    # 1. Candidate rollouts, low weight first so high-weight paths sit on top
    w_rel = scene.weights / float(np.max(scene.weights))
    order = np.argsort(w_rel)
    start = scene.ee_decoded_now[None, :]
    segs = []
    cols = []
    for k in order:
        path = np.concatenate([start, scene.candidates[k]], axis=0)
        segs.append(path[:, [i, j]])
        rgba = list(cmap(weight_norm(max(float(w_rel[k]), weight_norm.vmin))))
        rgba[3] = 0.25 + 0.60 * float(weight_norm(max(float(w_rel[k]), weight_norm.vmin)))
        cols.append(rgba)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.7, zorder=2))

    # 2. Lowest-cost rollout and recorded (ground-truth) EE approach path
    best = np.concatenate([start, scene.best], axis=0)
    ax.plot(best[:, i], best[:, j], color=PALETTE["spwm"], linewidth=2.6, zorder=6)
    ax.plot(scene.ee_recorded[:, i], scene.ee_recorded[:, j], color=PALETTE["ground_truth"],
            linewidth=1.6, linestyle=(0, (4, 2)), zorder=5)

    # 3. Arm skeleton (recorded pose at the planning step)
    chain = scene.arm_chain
    ax.plot(chain[:, i], chain[:, j], color="#334155", linewidth=5.0, solid_capstyle="round", zorder=7)
    ax.plot(chain[:, i], chain[:, j], color="#94A3B8", linewidth=2.6, solid_capstyle="round", zorder=8)
    ax.scatter(chain[1:-1, i], chain[1:-1, j], s=22, color="white", edgecolors="#1E293B", linewidths=1.1, zorder=9)
    ax.scatter([chain[0, i]], [chain[0, j]], s=70, marker="s", color="#1E293B", zorder=9)

    # 4. Decoded current EE, wrist camera, grasp target
    ax.scatter([scene.ee_decoded_now[i]], [scene.ee_decoded_now[j]], s=36, facecolors="white",
               edgecolors=PALETTE["spwm"], linewidths=1.5, zorder=11)
    ax.scatter([chain[-1, i]], [chain[-1, j]], s=55, marker="v", color="#D97706", edgecolors="#78350F", linewidths=0.8, zorder=10)
    ax.scatter([scene.target[i]], [scene.target[j]], s=110, marker="s", color=PALETTE["heft_region"],
               edgecolors=PALETTE["heft_border"], linewidths=1.8, zorder=10)
    ax.scatter([scene.target[i]], [scene.target[j]], s=14, marker="x", color="#92400E", linewidths=1.2, zorder=11)

    # 5. Head (global observer) camera with its optical axis
    cam = np.array(HEAD_CAMERA_POS)
    look = np.array(HEAD_CAMERA_LOOK_AT)
    ax.scatter([cam[i]], [cam[j]], s=70, marker="^", color="#059669", edgecolors="#064E3B", linewidths=0.8, zorder=10)
    if show_camera_frustum:
        ax.annotate("", xy=(look[i], look[j]), xytext=(cam[i], cam[j]),
                    arrowprops=dict(arrowstyle="->", color="#059669", linewidth=1.0, linestyle=":"), zorder=1)

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.4, alpha=0.6)


def draw_mppi_setup_panel(fig: plt.Figure, spec: gridspec.SubplotSpec, scene: MPPIScene) -> None:
    """Panel (a): front (X–Y) and top (X–Z) orthographic views + shared legend and weight colorbar."""
    sub = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=spec, height_ratios=[1.0, 0.06, 0.28],
                                           width_ratios=[1.0, 1.0], wspace=0.28, hspace=0.35)
    ax_front = fig.add_subplot(sub[0, 0])
    ax_top = fig.add_subplot(sub[0, 1])
    ax_cbar = fig.add_subplot(sub[1, :])
    ax_legend = fig.add_subplot(sub[2, :])
    ax_legend.axis("off")

    cmap = plt.cm.viridis
    weight_norm = LogNorm(vmin=1e-4, vmax=1.0, clip=True)

    _draw_orthographic_view(ax_front, scene, (0, 1), weight_norm, cmap, show_camera_frustum=True)
    ax_front.set_title("Front view (from behind the arm)", fontsize=9.0)
    ax_front.set_xlabel(r"$X$ lateral (m)")
    ax_front.set_ylabel(r"$Y$ elevation (m)")

    _draw_orthographic_view(ax_top, scene, (0, 2), weight_norm, cmap, show_camera_frustum=True)
    ax_top.set_title("Top view", fontsize=9.0)
    ax_top.set_xlabel(r"$X$ lateral (m)")
    ax_top.set_ylabel(r"$Z$ depth (m)")

    cb = fig.colorbar(ScalarMappable(norm=weight_norm, cmap=cmap), cax=ax_cbar, orientation="horizontal")
    cb.set_label(rf"Candidate MPPI weight $w_k / \max_k w_k$ ($K={scene.num_samples}$ rollouts, $H={scene.horizon}$ steps)", fontsize=8.0)
    cb.ax.tick_params(labelsize=7.5)

    handles = [
        Line2D([], [], color="#94A3B8", linewidth=4.0, label=rf"Arm pose at planning step $t={scene.t_plan}$"),
        Line2D([], [], color=PALETTE["spwm"], linewidth=2.6, label="Lowest-cost latent rollout"),
        Line2D([], [], color=PALETTE["ground_truth"], linewidth=1.6, linestyle=(0, (4, 2)),
               label=rf"Recorded EE path $t={scene.t_plan}\to{scene.t_contact}$"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=PALETTE["spwm"],
               markersize=6, label=r"EE decoded from $\mathbf{z}_t$ (rollout origin)"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor=PALETTE["heft_region"],
               markeredgecolor=PALETTE["heft_border"], markersize=8, label=rf"Grasp target (payload at contact, $t={scene.t_contact}$)"),
        Line2D([], [], marker="^", linestyle="none", color="#059669", markersize=7, label=r"Head event camera $C_{0:1}$"),
        Line2D([], [], marker="v", linestyle="none", color="#D97706", markersize=7, label=r"Wrist event camera $C_{2:3}$"),
    ]
    ax_legend.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=7.2,
                     columnspacing=1.2, handlelength=2.0)


# ==============================================================================
# Latent phase-space views (single trajectory, latent coordinates)
# ==============================================================================
def compute_phase_space_trajectories(
    spwm_model: SPWM,
    ann_model: ANNWorldModel,
    events: torch.Tensor,
    actions: torch.Tensor,
    horizon: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Autonomous latent rollouts of SPWM and ANN-Rate, both started from the SPWM-grounded latent z_0.

    Returns:
        (q_spwm, p_spwm, q_ann, p_ann): latent dimension 0 and L2 norm of latent dims 32:128
        (momentum block) at rollout steps 1..horizon.
    """
    with torch.no_grad():
        spwm_z0, _, _ = ground_initial_state(spwm_model, events, t_start=0)
        res_spwm = rollout_latent(spwm_model, spwm_z0, horizon=horizon, actions=actions[:, 1 : horizon + 1])
        res_ann = rollout_latent(ann_model, spwm_z0, horizon=horizon, actions=actions[:, 1 : horizon + 1])

    q_sp = res_spwm.trajectory[0, :, 0].numpy().astype(np.float64)
    p_sp = np.linalg.norm(res_spwm.trajectory[0, :, 32:].numpy(), axis=-1).astype(np.float64)
    q_an = res_ann.trajectory[0, :, 0].numpy().astype(np.float64)
    p_an = np.linalg.norm(res_ann.trajectory[0, :, 32:].numpy(), axis=-1).astype(np.float64)
    return q_sp, p_sp, q_an, p_an


def _alpha_ramp_segments(
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    alpha_min: float,
    alpha_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds per-segment line geometry and RGBA colors whose opacity encodes the horizon step.

    Segment i joins (x_i, y_i) -> (x_{i+1}, y_{i+1}) and is assigned alpha linearly
    interpolated from `alpha_min` (first step) to `alpha_max` (last step).
    """
    if x.shape != y.shape or x.ndim != 1 or x.shape[0] < 2:
        raise ValueError(f"Expected matching 1D arrays with >= 2 samples, got x={x.shape}, y={y.shape}")
    points = np.stack([x, y], axis=-1)[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    rgba = np.tile(np.array(to_rgba(color), dtype=np.float64), (segments.shape[0], 1))
    rgba[:, 3] = np.linspace(alpha_min, alpha_max, segments.shape[0])
    return segments, rgba


PHASE_LABEL_SPWM = "SPWM (spiking ALIF)"
PHASE_LABEL_ANN = "ANN-Rate (continuous, param.-matched)"
PHASE_LABEL_START = r"First rollout step $h=1$"
PHASE_X_LABEL = r"Latent coordinate $z_{h,0}$ (dim 0 of $\mathbf{q}$-block)"
PHASE_Y_LABEL = r"Latent momentum norm $\|\mathbf{z}_{h,32:128}\|_2$"


def draw_phase_space_front_2d(
    ax: plt.Axes,
    q_sp: np.ndarray,
    p_sp: np.ndarray,
    q_an: np.ndarray,
    p_an: np.ndarray,
    alpha_min: float = 0.10,
    alpha_max: float = 1.0,
) -> None:
    """Front (q, ||p||) projection of the latent phase space; horizon step encoded by segment opacity."""
    seg_an, rgba_an = _alpha_ramp_segments(q_an, p_an, PALETTE["ann_rate"], alpha_min, alpha_max)
    ax.add_collection(LineCollection(seg_an, colors=rgba_an, linewidths=2.0, linestyles="--", zorder=8))
    seg_sp, rgba_sp = _alpha_ramp_segments(q_sp, p_sp, PALETTE["spwm"], alpha_min, alpha_max)
    ax.add_collection(LineCollection(seg_sp, colors=rgba_sp, linewidths=2.4, zorder=9))

    ax.plot([], [], color=PALETTE["ann_rate"], linewidth=2.0, linestyle="--", label=PHASE_LABEL_ANN)
    ax.plot([], [], color=PALETTE["spwm"], linewidth=2.4, label=PHASE_LABEL_SPWM)
    ax.scatter([q_an[-1]], [p_an[-1]], color=PALETTE["ann_rate"], s=60, marker="X", edgecolors="#7F1D1D", zorder=12)
    ax.scatter([q_sp[-1]], [p_sp[-1]], color=PALETTE["spwm"], s=60, marker="o", edgecolors="#0F172A", zorder=12)
    ax.scatter([q_sp[0]], [p_sp[0]], color=PALETTE["spwm"], s=75, marker="*", edgecolors="#0F172A", zorder=15, label=PHASE_LABEL_START)
    ax.scatter([q_an[0]], [p_an[0]], color=PALETTE["ann_rate"], s=75, marker="*", edgecolors="#7F1D1D", zorder=15)

    all_q = np.concatenate([q_sp, q_an])
    all_p = np.concatenate([p_sp, p_an])
    q_pad = 0.06 * float(np.ptp(all_q))
    p_pad = 0.06 * float(np.ptp(all_p))
    ax.set_xlim(float(np.min(all_q)) - q_pad, float(np.max(all_q)) + q_pad)
    ax.set_ylim(float(np.min(all_p)) - p_pad, float(np.max(all_p)) + p_pad)
    ax.set_xlabel(PHASE_X_LABEL)
    ax.set_ylabel(PHASE_Y_LABEL)
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, fontsize=7.4)

    n_steps = q_sp.shape[0]
    alpha_cmap = LinearSegmentedColormap.from_list(
        "horizon_alpha", [to_rgba("#334155", alpha_min), to_rgba("#334155", alpha_max)]
    )
    cbar = plt.colorbar(ScalarMappable(norm=Normalize(vmin=1, vmax=n_steps), cmap=alpha_cmap), ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(rf"Horizon step $h \in [1, {n_steps}]$ (opacity)")


def draw_phase_space_3d(ax: plt.Axes, q_sp: np.ndarray, p_sp: np.ndarray, q_an: np.ndarray, p_an: np.ndarray) -> None:
    """3D (h, q, ||p||) latent phase-space view."""
    steps = np.arange(1, q_sp.shape[0] + 1)
    ax.plot(steps, q_an, p_an, color=PALETTE["ann_rate"], linewidth=2.0, linestyle="--", label=PHASE_LABEL_ANN)
    ax.plot(steps, q_sp, p_sp, color=PALETTE["spwm"], linewidth=2.4, label=PHASE_LABEL_SPWM)
    ax.scatter([1], [q_sp[0]], [p_sp[0]], color=PALETTE["spwm"], s=75, marker="*", edgecolors="#0F172A", label=PHASE_LABEL_START)
    ax.scatter([1], [q_an[0]], [p_an[0]], color=PALETTE["ann_rate"], s=75, marker="*", edgecolors="#7F1D1D")
    ax.set_xlabel(r"Horizon step $h$", labelpad=5)
    ax.set_ylabel(PHASE_X_LABEL, labelpad=5)
    ax.set_zlabel(PHASE_Y_LABEL, labelpad=5)
    ax.view_init(elev=24, azim=-62)
    ax.legend(loc="upper left", frameon=True, fontsize=7.6)


# ==============================================================================
# Figure assembly
# ==============================================================================
def render_figure_1(
    output_dir: Path,
    config_path: Path = GLOBAL_VERSION.m1_config(),
    spwm_ckpt: Path = GLOBAL_VERSION.m1_checkpoint(),
    ann_ckpt: Path = Path("results/ann_ablation/ann_rate_m1/model.pt"),
    trajectories_dir: Path = Path("sim/ExportedTrajectories"),
    scene_trajectory_id: int = 0,
    eval_batch_size: int = 32,
    mppi_seed: int = 0,
) -> None:
    """Renders Figure 1 and its standalone panels."""
    set_paper_style()

    print("[Fig 1] Loading models...")
    cfg = ExperimentConfig.load(config_path)
    spwm = load_world_model(SPWM.from_config(cfg), spwm_ckpt)
    ann = load_world_model(ANNWorldModel.from_config(cfg), ann_ckpt)
    fk = DifferentiableForwardKinematics()

    # Panel (a): one genuine closed-loop MPPI planning step
    print(f"[Fig 1] Computing latent MPPI plan on trajectory {scene_trajectory_id} (K=256, H=20, 3 iterations)...")
    ds = EventWorldDataset([scene_trajectory_id], dataset_dir=trajectories_dir)
    sample = ds[0]
    scene = compute_mppi_scene(spwm, sample, fk, num_samples=256, horizon=20, num_iterations=3, seed=mppi_seed)
    print(f"[Fig 1]   t_plan={scene.t_plan}, t_contact={scene.t_contact}, target={np.round(scene.target, 4).tolist()}, "
          f"best terminal dist={np.linalg.norm(scene.best[-1] - scene.target) * 100:.2f} cm, "
          f"effective sample size={1.0 / np.sum(scene.weights ** 2):.1f}")

    # Panel (b): decoded joint-space error per horizon step on the full test split
    print("[Fig 1] Evaluating per-step decoded joint-space error on the test split (H=100)...")
    test_loader = build_test_loader(cfg, trajectories_dir, eval_batch_size)
    joint_err = compute_joint_error_per_step({"spwm": spwm, "ann": ann}, test_loader, horizon=100)
    n_traj = joint_err.errors["spwm"].shape[0]
    for name, err in joint_err.errors.items():
        m = err.mean(axis=0)
        print(f"[Fig 1]   {name}: N={err.shape[0]}  mean err (rad) h=1:{m[0]:.4f} h=5:{m[4]:.4f} h=10:{m[9]:.4f} "
              f"h=25:{m[24]:.4f} h=50:{m[49]:.4f} h=100:{m[99]:.4f}  mean over h:{m.mean():.4f}")
    styles = {
        "spwm": {"color": PALETTE["spwm"], "linestyle": "-", "label": "SPWM (spiking ALIF)"},
        "ann": {"color": PALETTE["ann_rate"], "linestyle": "--", "label": "ANN-Rate (continuous, param.-matched)"},
    }

    # Composite teaser
    fig = plt.figure(figsize=(13.6, 5.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18, top=0.92, bottom=0.08)
    draw_mppi_setup_panel(fig, gs[0, 0], scene)
    fig.text(0.08, 0.965, "(a) Physical setup & latent MPPI predictive rollouts", fontsize=10.0, weight="bold", ha="left")
    ax_err = fig.add_subplot(gs[0, 1])
    draw_joint_error_panel(ax_err, joint_err, styles)
    ax_err.set_title(rf"(b) Decoded joint-space error vs. horizon ($N={n_traj}$ test trajectories)",
                     fontsize=10.0, weight="bold", loc="left", pad=10)
    save_publication_figure(fig, output_dir / "fig1_teaser")
    plt.close(fig)

    # Standalone panel (a)
    fig_a = plt.figure(figsize=(8.0, 5.4))
    draw_mppi_setup_panel(fig_a, gridspec.GridSpec(1, 1)[0, 0], scene)
    fig_a.suptitle("Physical setup & latent MPPI predictive rollouts", fontsize=10.5, weight="bold")
    save_publication_figure(fig_a, output_dir / "fig1_mppi_setup")
    plt.close(fig_a)

    # Standalone panel (b)
    fig_b, ax_b = plt.subplots(1, 1, figsize=(6.4, 4.4))
    draw_joint_error_panel(ax_b, joint_err, styles)
    ax_b.set_title(rf"Decoded joint-space error vs. horizon ($N={n_traj}$ test trajectories)", fontsize=10.0, weight="bold", pad=8)
    save_publication_figure(fig_b, output_dir / "fig1_joint_error_horizon")
    plt.close(fig_b)

    # Latent phase-space views (single trajectory, no drift annotations)
    print("[Fig 1] Computing latent phase-space rollouts (H=100)...")
    q_sp, p_sp, q_an, p_an = compute_phase_space_trajectories(
        spwm, ann, sample["events"].unsqueeze(0), sample["actions"].unsqueeze(0), horizon=100,
    )
    fig_ps = plt.figure(figsize=(7.2, 5.0))
    ax_ps = fig_ps.add_subplot(1, 1, 1, projection="3d")
    draw_phase_space_3d(ax_ps, q_sp, p_sp, q_an, p_an)
    ax_ps.set_title(f"Latent phase space, autonomous rollout (trajectory {scene_trajectory_id})", fontsize=10.0, weight="bold", pad=10)
    save_publication_figure(fig_ps, output_dir / "fig1_phase_space")
    plt.close(fig_ps)

    fig_ps2d, ax_ps2d = plt.subplots(1, 1, figsize=(6.6, 4.8))
    draw_phase_space_front_2d(ax_ps2d, q_sp, p_sp, q_an, p_an)
    ax_ps2d.set_title(f"Latent phase space, front view (trajectory {scene_trajectory_id})", fontsize=10.0, weight="bold", pad=10)
    save_publication_figure(fig_ps2d, output_dir / "fig1_phase_space_2d")
    plt.close(fig_ps2d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoRL Figure 1: System Teaser.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"), help="Directory for exported figures")
    parser.add_argument("--config", type=Path, default=GLOBAL_VERSION.m1_config())
    parser.add_argument("--spwm-ckpt", type=Path, default=GLOBAL_VERSION.m1_checkpoint())
    parser.add_argument("--ann-ckpt", type=Path, default=Path("results/ann_ablation/ann_rate_m1/model.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("sim/ExportedTrajectories"))
    parser.add_argument("--scene-trajectory", type=int, default=0, help="Trajectory id used for panel (a) and phase-space views")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--mppi-seed", type=int, default=0)
    args = parser.parse_args()

    render_figure_1(
        output_dir=args.output_dir,
        config_path=args.config,
        spwm_ckpt=args.spwm_ckpt,
        ann_ckpt=args.ann_ckpt,
        trajectories_dir=args.data_dir,
        scene_trajectory_id=args.scene_trajectory,
        eval_batch_size=args.eval_batch_size,
        mppi_seed=args.mppi_seed,
    )


if __name__ == "__main__":
    main()
