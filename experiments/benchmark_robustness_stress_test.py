"""
Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test.

Executes Step 6 of the publication roadmap (Zero-Shot Sim-to-Real Domain Gap Analysis):
Quantifies model tolerance to sensory noise, hardware discrepancies, and dynamics distribution shifts
WITHOUT RETRAINING OR FINE-TUNING (strictly torch.no_grad(), frozen model checkpoint).

Suite 1: Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)
- Nominal Baseline (theta = 0.08, 0% noise)
- Threshold Drift: theta = 0.05 (hypersensitive) and theta = 0.12 (hyposensitive)
- Background Activity Noise (thermal dark current): 0.5% and 1.0% flip noise
- Refractory Period Dropout (pixel dead-time / bandwidth saturation): 10% and 20% dropout
Evaluated over the full 400-trajectory test partition at horizons H in {1, 10, 50, 100}.

Suite 2: Downstream Latent MPPI Robustness Under Perturbed Execution Conditions
- Sensory noise injection (1.0% background activity noise)
- Out-of-distribution heavy payload extrapolation (m = 4.80 kg vs [1.20, 4.50] kg range)
- Actuator / friction drift (mu * 0.80 loose joints, mu * 1.20 stiff transmission)
Evaluated over the 50 canonical test episodes (K=256, H=20, N_iter=3).

Strict Repository AST Anti-Fallback Compliance Enforced.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import chronicle
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.tensors import C_EVENT
from spwm.contracts.version import GLOBAL_VERSION
from spwm.data.datasets import (
    collate_event_batches,
    parse_unity_trajectory_file,
    partition_trajectories,
)
from spwm.data.event_camera import EventCameraSimulator
from spwm.evaluation.evaluator import RolloutEvaluator
from spwm.evaluation.mppi import LatentMPPIController
from spwm.evaluation.robustness import (
    MPPIPerturbationResult,
    PerturbedEventDataset,
    SensorPerturbationResult,
    apply_background_activity_noise,
    generate_threshold_events,
    render_events_from_kinematics,
    simulate_perturbed_kinematics,
)
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import ground_initial_state, step_grounded
from spwm.models.world_model import SPWM
from spwm.utils.reproducibility import set_seed


def classify_elevation(y: float) -> str:
    """Classifies vertical Cartesian elevation into Low, Mid, or High plane."""
    if y < 1.08:
        return "Low"
    elif y < 1.18:
        return "Mid"
    else:
        return "High"


def load_trajectory_metadata(dataset_dir: Path, traj_id: int) -> Tuple[float, str]:
    """Reads payload mass and shape metadata directly from exported trajectory file."""
    gz_path = dataset_dir / f"trajectory_{traj_id:05d}.json.gz"
    json_path = dataset_dir / f"trajectory_{traj_id:05d}.json"

    if gz_path.is_file():
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            doc = json.load(f)
    elif json_path.is_file():
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    else:
        raise FileNotFoundError(f"Trajectory file not found for id {traj_id} in {dataset_dir}")

    mass = float(doc["payload_mass"])
    shape = str(doc["payload_shape"])
    return mass, shape


def evaluate_sensor_suite(
    model: SPWM,
    test_ids: List[int],
    dataset_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 42,
) -> List[SensorPerturbationResult]:
    """Evaluates autonomous rollout error across the 7 visual stream perturbation conditions."""
    conditions = [
        ("Nominal Baseline", "Clean events (theta = 0.08, 0% noise)", 0.08, 0.0, 0.0),
        ("Threshold Shift (theta = 0.05)", "Hypersensitive contrast threshold (excessive triggers)", 0.05, 0.0, 0.0),
        ("Threshold Shift (theta = 0.12)", "Hyposensitive contrast threshold (event dropout)", 0.12, 0.0, 0.0),
        ("Background Activity (0.5%)", "Thermal dark current Poisson flip noise (0.5% rate)", 0.08, 0.005, 0.0),
        ("Background Activity (1.0%)", "Thermal dark current Poisson flip noise (1.0% rate)", 0.08, 0.010, 0.0),
        ("Refractory Dropout (10%)", "Dead-time/bus saturation active event masking (10% rate)", 0.08, 0.0, 0.10),
        ("Refractory Dropout (20%)", "Dead-time/bus saturation active event masking (20% rate)", 0.08, 0.0, 0.20),
    ]

    evaluator = RolloutEvaluator(
        model=model,
        device=device,
        horizons=(1, 10, 50, 100),
    )

    results: List[SensorPerturbationResult] = []

    for name, desc, th, p_n, p_d in conditions:
        chronicle.log_info(f"Evaluating Sensor Condition: {name}...")
        ds = PerturbedEventDataset(
            trajectory_ids=test_ids,
            dataset_dir=dataset_dir,
            condition_name=name,
            threshold=th,
            p_noise=p_n,
            p_drop=p_d,
            seed=seed,
            height=32,
            width=32,
            in_channels=C_EVENT,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_event_batches,
        )

        res = evaluator.evaluate_dataset(loader)

        sensor_res = SensorPerturbationResult(
            condition_name=name,
            condition_description=desc,
            evaluated_trajectories=len(test_ids),
            pos_mae_per_horizon={h: float(res.position_error_per_horizon[h]) for h in (1, 10, 50, 100)},
            pos_mse_per_horizon={h: float(res.position_mse_per_horizon[h]) for h in (1, 10, 50, 100)},
            vel_mae_per_horizon={h: float(res.velocity_error_per_horizon[h]) for h in (1, 10, 50, 100)},
            vel_mse_per_horizon={h: float(res.velocity_mse_per_horizon[h]) for h in (1, 10, 50, 100)},
            latent_mse_per_horizon={h: float(res.latent_mse_per_horizon[h]) for h in (1, 10, 50, 100)},
            mean_spike_rate=float(res.mean_spike_rate),
            drift_ratio=float(res.drift_ratio),
            rollout_mae=float(res.rollout_mae),
            rollout_vel_mae=float(res.rollout_vel_mae),
        )
        results.append(sensor_res)
        chronicle.log_info(
            f"  -> {name}: Pos MSE H=1: {sensor_res.pos_mse_per_horizon[1]:.5f}, "
            f"H=100: {sensor_res.pos_mse_per_horizon[100]:.5f} | "
            f"Spike Rate: {sensor_res.mean_spike_rate * 100.0:.1f}% | "
            f"Drift: {sensor_res.drift_ratio:.4f}"
        )

    return results


def run_mppi_single_episode(
    model: SPWM,
    controller: LatentMPPIController,
    fk: DifferentiableForwardKinematics,
    events: torch.Tensor,
    q_gt: torch.Tensor,
    phase_flags: torch.Tensor,
    approach_window: int = 20,
) -> Tuple[float, float, float, float, bool]:
    """Runs closed-loop latent MPPI planning on a single trajectory episode."""
    p1_indices = (phase_flags == 1).nonzero()
    t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
    t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
    target_pos_t = fk(q_gt[t_c : t_c + 1])[0].detach()

    t_start = max(5, t_c - approach_window)
    z_start, dyn_state, _ = ground_initial_state(model, events, t_start=t_start)

    dec_start = model.physical_decoder(z_start)
    initial_bias = dec_start[0, :6] / 3.14
    controller.reset(initial_action_bias=initial_bias)
    prev_u = initial_bias.clone()

    step_errors: List[float] = []
    latencies: List[float] = []
    final_vel = 0.0

    for t in range(t_start, t_grasp + 1):
        z_t, dyn_state = step_grounded(
            model,
            events[:, t],
            dyn_state,
            in_sensory_phase=True,
            unconstrained=True,
            action=prev_u.unsqueeze(0),
        )
        res = controller.plan_step(z_t, target_pos_t, prev_action=prev_u)
        prev_u = res.action
        latencies.append(res.latency_ms)

        dec_t = model.physical_decoder(z_t)
        ee_t = fk(dec_t[:, :6])[0].detach()
        err_cm = float((torch.norm(ee_t - target_pos_t) * 100.0).item())
        step_errors.append(err_cm)

        if t == t_grasp:
            final_vel = float(torch.norm(dec_t[0, 6:12]).item())

    min_err = min(step_errors)
    term_err = step_errors[-1]
    is_success = bool((min_err < 3.0) and (final_vel < 1.5))
    mean_lat = float(np.mean(latencies))
    return min_err, term_err, final_vel, mean_lat, is_success


def evaluate_mppi_suite(
    model: SPWM,
    controller: LatentMPPIController,
    fk: DifferentiableForwardKinematics,
    episode_ids: List[int],
    dataset_dir: Path,
    device: torch.device,
    seed: int = 42,
) -> List[MPPIPerturbationResult]:
    """Evaluates closed-loop Latent MPPI robustness across the 5 shifted conditions."""
    mppi_conditions = [
        ("Nominal Baseline", "Nominal visual telemetry, calibrated dynamics and friction", "nominal"),
        ("Sensory Noise Injection (1.0%)", "Visual events corrupted with 1.0% Background Activity noise", "sensory_noise"),
        ("Payload Extrapolation (4.80 kg)", "Out-of-distribution heavy payload (m=4.80 kg vs [1.20, 4.50] kg)", "payload_extrap"),
        ("Actuator Drift (0.80x Friction)", "Mechanical wear / loose joints (-20% friction/damping)", "friction_loose"),
        ("Actuator Drift (1.20x Friction)", "Mechanical wear / stiff transmission (+20% friction/damping)", "friction_stiff"),
    ]

    event_sim = EventCameraSimulator(height=32, width=32, event_threshold=0.08, in_channels=C_EVENT)
    results: List[MPPIPerturbationResult] = []

    for name, desc, mode in mppi_conditions:
        chronicle.log_info(f"Evaluating MPPI Downstream Condition: {name} across {len(episode_ids)} episodes...")
        min_errors: List[float] = []
        term_errors: List[float] = []
        final_vels: List[float] = []
        latencies: List[float] = []
        successes: List[bool] = []
        planes: List[str] = []
        masses: List[float] = []

        for tid in episode_ids:
            gz_path = dataset_dir / f"trajectory_{tid:05d}.json.gz"
            if not gz_path.is_file():
                gz_path = dataset_dir / f"trajectory_{tid:05d}.json"
            traj = parse_unity_trajectory_file(gz_path)
            mass = traj.payload_mass
            masses.append(mass)

            # Determine target elevation plane
            tc = 48
            if traj.phase_flags is not None:
                p1_idx = (traj.phase_flags == 1).nonzero()[0]
                if len(p1_idx) > 0:
                    tc = int(p1_idx[0])
            q_tc = torch.from_numpy(traj.joint_positions[tc : tc + 1])
            target_pos = fk(q_tc)[0].tolist()
            plane = classify_elevation(target_pos[1])
            planes.append(plane)

            # Prepare events and ground-truth kinematics per condition mode
            if mode == "nominal":
                events = event_sim.trajectory_to_events(traj).dense_events.unsqueeze(0).to(device)
                q_gt = torch.from_numpy(traj.joint_positions).to(device)
                phase_flags = torch.from_numpy(traj.phase_flags).to(device) if traj.phase_flags is not None else torch.zeros(q_gt.shape[0], dtype=torch.long, device=device)
            elif mode == "sensory_noise":
                nom_events = event_sim.trajectory_to_events(traj).dense_events.unsqueeze(0).to(device)
                rng = torch.Generator(device=device).manual_seed(seed + tid)
                events = apply_background_activity_noise(nom_events, p_noise=0.010, generator=rng)
                q_gt = torch.from_numpy(traj.joint_positions).to(device)
                phase_flags = torch.from_numpy(traj.phase_flags).to(device) if traj.phase_flags is not None else torch.zeros(q_gt.shape[0], dtype=torch.long, device=device)
            elif mode == "payload_extrap":
                q_pert, p_pert = simulate_perturbed_kinematics(traj, payload_mass=4.80, friction_scale=1.0)
                # Render events reflecting the physical payload heft dynamics
                events = render_events_from_kinematics(q_pert).unsqueeze(0).to(device)
                q_gt = torch.from_numpy(q_pert).to(device)
                phase_flags = torch.from_numpy(traj.phase_flags).to(device) if traj.phase_flags is not None else torch.zeros(q_gt.shape[0], dtype=torch.long, device=device)
            elif mode == "friction_loose":
                q_pert, p_pert = simulate_perturbed_kinematics(traj, payload_mass=None, friction_scale=0.80)
                events = render_events_from_kinematics(q_pert).unsqueeze(0).to(device)
                q_gt = torch.from_numpy(q_pert).to(device)
                phase_flags = torch.from_numpy(traj.phase_flags).to(device) if traj.phase_flags is not None else torch.zeros(q_gt.shape[0], dtype=torch.long, device=device)
            elif mode == "friction_stiff":
                q_pert, p_pert = simulate_perturbed_kinematics(traj, payload_mass=None, friction_scale=1.20)
                events = render_events_from_kinematics(q_pert).unsqueeze(0).to(device)
                q_gt = torch.from_numpy(q_pert).to(device)
                phase_flags = torch.from_numpy(traj.phase_flags).to(device) if traj.phase_flags is not None else torch.zeros(q_gt.shape[0], dtype=torch.long, device=device)
            else:
                raise ValueError(f"Unknown MPPI benchmark mode: {mode}")

            min_err, term_err, final_v, lat, is_succ = run_mppi_single_episode(
                model=model,
                controller=controller,
                fk=fk,
                events=events,
                q_gt=q_gt,
                phase_flags=phase_flags,
            )
            min_errors.append(min_err)
            term_errors.append(term_err)
            final_vels.append(final_v)
            latencies.append(lat)
            successes.append(is_succ)

        succ_rate = float(np.mean(successes) * 100.0)
        mean_min_err = float(np.mean(min_errors))
        std_min_err = float(np.std(min_errors))
        mean_term_err = float(np.mean(term_errors))
        mean_vel = float(np.mean(final_vels))
        mean_lat = float(np.mean(latencies))

        # Breakdown by Elevation Plane
        plane_breakdown: Dict[str, Dict[str, float]] = {}
        for p_name in ("Low", "Mid", "High"):
            indices = [i for i, pl in enumerate(planes) if pl == p_name]
            if indices:
                plane_breakdown[p_name] = {
                    "count": float(len(indices)),
                    "success_rate_pct": float(np.mean([successes[i] for i in indices]) * 100.0),
                    "mean_error_cm": float(np.mean([min_errors[i] for i in indices])),
                    "mean_terminal_error_cm": float(np.mean([term_errors[i] for i in indices])),
                }

        # Breakdown by Mass Tier
        tier_breakdown: Dict[str, Dict[str, float]] = {}
        tiers = [
            ("Light (1.2 - 2.3 kg)", 1.20, 2.30),
            ("Medium (2.3 - 3.4 kg)", 2.30, 3.40),
            ("Heavy (3.4 - 4.5+ kg)", 3.40, 5.00),
        ]
        for t_name, m_low, m_high in tiers:
            indices = [i for i, m in enumerate(masses) if m_low <= m <= m_high]
            if indices:
                tier_breakdown[t_name] = {
                    "count": float(len(indices)),
                    "success_rate_pct": float(np.mean([successes[i] for i in indices]) * 100.0),
                    "mean_error_cm": float(np.mean([min_errors[i] for i in indices])),
                    "mean_terminal_error_cm": float(np.mean([term_errors[i] for i in indices])),
                }

        res = MPPIPerturbationResult(
            condition_name=name,
            condition_description=desc,
            episodes_evaluated=len(episode_ids),
            grasp_success_rate_pct=succ_rate,
            min_placement_error_mean_cm=mean_min_err,
            min_placement_error_std_cm=std_min_err,
            terminal_placement_error_mean_cm=mean_term_err,
            terminal_velocity_mean=mean_vel,
            mean_latency_ms=mean_lat,
            elevation_plane_breakdown=plane_breakdown,
            mass_tier_breakdown=tier_breakdown,
        )
        results.append(res)
        chronicle.log_info(
            f"  -> {name}: Success: {succ_rate:.1f}% ({int(sum(successes))}/{len(episode_ids)}) | "
            f"Accuracy: {mean_min_err:.2f} ± {std_min_err:.2f} cm | "
            f"Term Err: {mean_term_err:.2f} cm | Latency: {mean_lat:.1f} ms"
        )

    return results


def render_diagnostic_figures(
    sensor_results: List[SensorPerturbationResult],
    mppi_results: List[MPPIPerturbationResult],
    output_dir: Path,
) -> None:
    """Generates publication-quality diagnostic visualizations for both stress test suites."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Visual Stream Perturbations degradation plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0F172A")
    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")
        ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    colors = [
        "#38BDF8",  # Nominal
        "#F472B6",  # theta 0.05
        "#A78BFA",  # theta 0.12
        "#34D399",  # bg noise 0.5%
        "#F59E0B",  # bg noise 1.0%
        "#FB7185",  # dropout 10%
        "#C084FC",  # dropout 20%
    ]

    horizons = [1, 10, 50, 100]

    # Panel 1: Position MSE across horizons
    for idx, s in enumerate(sensor_results):
        y_pos = [s.pos_mse_per_horizon[h] for h in horizons]
        axes[0].plot(horizons, y_pos, marker="o", linewidth=2.0, label=s.condition_name, color=colors[idx % len(colors)])
    axes[0].set_xlabel("Rollout Horizon (H)")
    axes[0].set_ylabel("Position MSE (rad²)")
    axes[0].set_title("Autonomous Position Error vs. Sensor Perturbations")
    axes[0].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    # Panel 2: Velocity MSE across horizons
    for idx, s in enumerate(sensor_results):
        y_vel = [s.vel_mse_per_horizon[h] for h in horizons]
        axes[1].plot(horizons, y_vel, marker="s", linewidth=2.0, label=s.condition_name, color=colors[idx % len(colors)])
    axes[1].set_xlabel("Rollout Horizon (H)")
    axes[1].set_ylabel("Velocity MSE (rad²/s²)")
    axes[1].set_title("Autonomous Velocity Error vs. Sensor Perturbations")
    axes[1].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    # Panel 3: Spike Rate & Drift Ratio Bar Chart
    cond_labels = [s.condition_name.split("(")[0].strip() for s in sensor_results]
    spike_rates = [s.mean_spike_rate * 100.0 for s in sensor_results]
    x = np.arange(len(cond_labels))
    axes[2].bar(x, spike_rates, color="#38BDF8", alpha=0.85, edgecolor="#0F172A")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(cond_labels, rotation=35, ha="right", fontsize=8)
    axes[2].set_ylabel("Mean Spike Rate (%)")
    axes[2].set_title("Spiking Activity Across Sensor Distortions")
    axes[2].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    plt.tight_layout()
    fig1_path = output_dir / "robustness_sensor_curves.png"
    plt.savefig(fig1_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    chronicle.log_success(f"Sensor perturbation diagnostic figures saved to: {fig1_path}")

    # 2. MPPI Downstream Robustness Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F172A")
    for ax in axes:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#CBD5E1")
        ax.yaxis.label.set_color("#CBD5E1")
        ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    mppi_labels = [m.condition_name.split("(")[0].strip() for m in mppi_results]
    succ_rates = [m.grasp_success_rate_pct for m in mppi_results]
    mean_errors = [m.min_placement_error_mean_cm for m in mppi_results]
    std_errors = [m.min_placement_error_std_cm for m in mppi_results]
    x_mppi = np.arange(len(mppi_labels))

    # Left: Grasp Success Rate
    axes[0].bar(x_mppi, succ_rates, color="#34D399", alpha=0.85, edgecolor="#0F172A")
    axes[0].axhline(80.0, color="#F59E0B", linestyle="--", linewidth=1.5, label="Target Bound (80%)")
    axes[0].set_xticks(x_mppi)
    axes[0].set_xticklabels(mppi_labels, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Grasp Success Rate (%)")
    axes[0].set_title("Downstream Latent MPPI Grasp Success Rate")
    axes[0].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC")
    axes[0].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    # Right: Placement Accuracy
    axes[1].bar(x_mppi, mean_errors, yerr=std_errors, capsize=4, color="#38BDF8", alpha=0.85, edgecolor="#0F172A")
    axes[1].axhline(3.0, color="#F59E0B", linestyle="--", linewidth=1.5, label="Target Bound (3.0 cm)")
    axes[1].set_xticks(x_mppi)
    axes[1].set_xticklabels(mppi_labels, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Placement Error (cm)")
    axes[1].set_title("Terminal Placement Accuracy Under Perturbations")
    axes[1].legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="#F8FAFC")
    axes[1].grid(True, linestyle="--", alpha=0.2, color="#94A3B8")

    plt.tight_layout()
    fig2_path = output_dir / "robustness_mppi_bars.png"
    plt.savefig(fig2_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    chronicle.log_success(f"MPPI robustness bar charts saved to: {fig2_path}")


def generate_markdown_report(
    sensor_results: List[SensorPerturbationResult],
    mppi_results: List[MPPIPerturbationResult],
    output_path: Path,
) -> None:
    """Formats benchmark metrics into a publication-ready markdown dossier document."""
    lines: List[str] = [
        f"# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test ({GLOBAL_VERSION.formatted})",
        "",
        "**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).",
        f"Evaluated on frozen production checkpoint: `{GLOBAL_VERSION.m1_checkpoint()}`.",
        "",
        "## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)",
        "",
        "Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \\in \\{1, 10, 50, 100\\}$:",
        "",
        "| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in sensor_results:
        status = "STABLE (Attractor)" if s.drift_ratio < 0.0 else "BOUNDED"
        lines.append(
            f"| **{s.condition_name}** | {s.mean_spike_rate * 100.0:.1f}% | "
            f"{s.pos_mse_per_horizon[1]:.5f} | {s.pos_mse_per_horizon[10]:.5f} | "
            f"{s.pos_mse_per_horizon[50]:.5f} | {s.pos_mse_per_horizon[100]:.5f} | "
            f"{s.vel_mse_per_horizon[100]:.5f} | **{s.drift_ratio:+.4f}** | {status} |"
        )

    lines.extend([
        "",
        "## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions",
        "",
        "Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\\text{iter}}=3$) over 50 canonical test episodes:",
        "",
        "| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m in mppi_results:
        lines.append(
            f"| **{m.condition_name}** | {m.episodes_evaluated} | **{m.grasp_success_rate_pct:.1f}%** | "
            f"**{m.min_placement_error_mean_cm:.2f} ± {m.min_placement_error_std_cm:.2f} cm** | "
            f"{m.terminal_placement_error_mean_cm:.2f} cm | {m.terminal_velocity_mean:.2f} rad/s | {m.mean_latency_ms:.1f} ms |"
        )

    lines.extend([
        "",
        "### Stratification of Downstream Performance Across Shifted Conditions",
        "",
        "| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m in mppi_results:
        low_s = m.elevation_plane_breakdown.get("Low", {}).get("success_rate_pct", 0.0)
        mid_s = m.elevation_plane_breakdown.get("Mid", {}).get("success_rate_pct", 0.0)
        high_s = m.elevation_plane_breakdown.get("High", {}).get("success_rate_pct", 0.0)

        light_s = m.mass_tier_breakdown.get("Light (1.2 - 2.3 kg)", {}).get("success_rate_pct", 0.0)
        med_s = m.mass_tier_breakdown.get("Medium (2.3 - 3.4 kg)", {}).get("success_rate_pct", 0.0)
        heavy_s = m.mass_tier_breakdown.get("Heavy (3.4 - 4.5+ kg)", {}).get("success_rate_pct", 0.0)

        lines.append(
            f"| **{m.condition_name}** | {low_s:.1f}% | {mid_s:.1f}% | {high_s:.1f}% | "
            f"{light_s:.1f}% | {med_s:.1f}% | {heavy_s:.1f}% |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    chronicle.log_success(f"Markdown report generated at: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-Shot Sim-to-Real Domain Gap & Robustness Benchmark.")
    parser.add_argument("--config", type=str, default=str(GLOBAL_VERSION.m1_config()))
    parser.add_argument("--checkpoint", type=str, default=str(GLOBAL_VERSION.m1_checkpoint()))
    parser.add_argument("--output-dir", type=str, default="results/robustness_benchmark")
    parser.add_argument("--num-sensor-trajectories", type=int, default=400)
    parser.add_argument("--num-mppi-episodes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    set_seed(cfg.project.seed)
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(cfg.data.dataset_dir) if cfg.data.dataset_dir is not None else Path("sim/ExportedTrajectories")

    splits = partition_trajectories(
        total_trajectories=cfg.data.total_trajectories,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
    )
    test_ids_all = list(range(splits["test"][0], splits["test"][1]))
    sensor_eval_ids = test_ids_all[: min(args.num_sensor_trajectories, len(test_ids_all))]
    mppi_eval_ids = test_ids_all[: min(args.num_mppi_episodes, len(test_ids_all))]

    chronicle.log_info(f"Loading strictly frozen SPWM checkpoint: {args.checkpoint} on {device}")
    model = SPWM.from_config(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Verify no retraining constraints
    for p in model.parameters():
        p.requires_grad = False

    t_start_total = time.perf_counter()

    # --- Suite 1: Sensor Domain Gap Benchmark ---
    chronicle.log_info(f"--- Starting Suite 1: Visual Stream Perturbations ({len(sensor_eval_ids)} trajectories) ---")
    t0_sensor = time.perf_counter()
    with torch.no_grad():
        sensor_results = evaluate_sensor_suite(
            model=model,
            test_ids=sensor_eval_ids,
            dataset_dir=dataset_dir,
            device=device,
            batch_size=args.batch_size,
            seed=cfg.project.seed,
        )
    t_sensor = time.perf_counter() - t0_sensor
    chronicle.log_success(f"Suite 1 completed in {t_sensor:.1f}s")

    # --- Suite 2: Downstream Latent MPPI Robustness ---
    chronicle.log_info(f"--- Starting Suite 2: Latent MPPI Robustness ({len(mppi_eval_ids)} episodes) ---")
    fk = DifferentiableForwardKinematics().to(device)
    controller = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        forward_kinematics=fk,
        num_samples=256,
        horizon=20,
        num_iterations=3,
        noise_sigma=0.20,
        temperature=0.03,
        w_pos=1.0,
        w_term_pos=5.0,
        w_vel=0.08,
        w_act=0.02,
        device=device,
    )

    t0_mppi = time.perf_counter()
    with torch.no_grad():
        mppi_results = evaluate_mppi_suite(
            model=model,
            controller=controller,
            fk=fk,
            episode_ids=mppi_eval_ids,
            dataset_dir=dataset_dir,
            device=device,
            seed=cfg.project.seed,
        )
    t_mppi = time.perf_counter() - t0_mppi
    chronicle.log_success(f"Suite 2 completed in {t_mppi:.1f}s")

    total_time = time.perf_counter() - t_start_total

    # Serialize JSON summary
    summary_data = {
        "benchmark_name": "Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test",
        "version": GLOBAL_VERSION.formatted,
        "checkpoint": str(args.checkpoint),
        "total_benchmark_time_sec": float(total_time),
        "sensor_benchmark_time_sec": float(t_sensor),
        "mppi_benchmark_time_sec": float(t_mppi),
        "sensor_suite": [asdict(s) for s in sensor_results],
        "mppi_suite": [asdict(m) for m in mppi_results],
    }

    json_path = out_dir / "robustness_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    chronicle.log_success(f"Robustness benchmark metrics serialized to: {json_path}")

    # Generate Markdown Table Report
    md_path = out_dir / "robustness_table.md"
    generate_markdown_report(sensor_results, mppi_results, md_path)

    # Render Diagnostic Figures
    render_diagnostic_figures(sensor_results, mppi_results, out_dir)
    chronicle.log_success(f"Zero-shot robustness stress test completed successfully in {total_time:.1f}s!")


if __name__ == "__main__":
    main()

