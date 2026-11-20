"""
SPWM Inter-Seed Variance & Internal Module Convergence Audit (Seeds 42, 101, 2024).

Compares across Seeds 42, 101, and 2024 (both best_model.pt and final checkpoint.pt):
  1. Learned dynamics.action_proj.weight norms (spectral ||W_u||_2 and Frobenius ||W_u||_F)
  2. Learned dynamics.gate_scale distribution (raw and softplus)
  3. ALIF silent/hyperactive neuron fractions & mean spike rate on N=400 test split
  4. Online residual handover trigger frequency (w_t > 0 rate during phi_t = 1) in 50-episode PIL

Zero AST fallbacks strictly enforced.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmark_mppi_control import _compute_residual_handover_weight, load_trajectory_metadata
from spwm.contracts.config import ExperimentConfig
from spwm.data.datasets import EventWorldDataset
from spwm.data.event_camera import EventCameraSimulator, OnlineEventCameraStream
from spwm.data.sim_6dof import ArticulatedArmPhysicsStepper
from spwm.evaluation.mppi import LatentMPPIController
from spwm.models.kinematics import DifferentiableForwardKinematics
from spwm.models.rollout import ground_initial_state, step_grounded
from spwm.models.world_model import SPWM
from spwm.training.diagnostics import collect_spike_statistics
from spwm.utils.reproducibility import set_seed


@torch.no_grad()
def measure_handover_trigger_rate_pil(
    model: SPWM,
    cfg: ExperimentConfig,
    dataset_dir: Path,
    episode_ids: List[int],
    device: torch.device,
) -> Dict[str, float]:
    """Run 50-episode PIL rollout and record exact w_t > 0 trigger frequency during phi_t = 1."""
    set_seed(cfg.project.seed)
    fk = DifferentiableForwardKinematics().to(device)
    controller = LatentMPPIController(
        predictor=model.m1,
        physical_decoder=model.physical_decoder,
        config=cfg.mppi,
        forward_kinematics=fk,
        device=device,
    )
    ds = EventWorldDataset(episode_ids, dataset_dir=dataset_dir)

    total_p1_steps = 0
    triggered_p1_steps = 0
    w_values_all: List[float] = []
    w_values_nonzero: List[float] = []

    for idx, traj_id in enumerate(episode_ids):
        sample = ds[idx]
        payload_mass, payload_shape = load_trajectory_metadata(dataset_dir, traj_id)
        events = sample["events"].unsqueeze(0).to(device)
        q_gt = sample["joint_positions"][:, :6].to(device)
        p_gt = sample["joint_velocities"][:, :6].to(device)
        phase_flags = sample["phase_flags"].to(device)
        actions = sample["actions"].unsqueeze(0).to(device)

        p1_indices = (phase_flags == 1).nonzero()
        t_c = int(p1_indices[0].item()) if len(p1_indices) > 0 else 50
        t_grasp = min(t_c + 7, q_gt.shape[0] - 1)
        t_start = max(5, t_c - 20)

        target_grasp_mat = fk.forward_homogeneous(q_gt[t_c : t_c + 1])[0].detach()
        target_final_mat = fk.forward_homogeneous(q_gt[t_grasp : t_grasp + 1])[0].detach()

        z_start, dyn_state, _ = ground_initial_state(model, events, t_start=t_start, actions=actions)
        curr_z = z_start.clone()
        controller.reset()
        prev_u = None

        phys_stepper = ArticulatedArmPhysicsStepper(
            q_init=q_gt[t_start].detach().cpu().numpy(),
            p_init=p_gt[t_start].detach().cpu().numpy(),
            payload_mass=payload_mass,
            payload_shape=payload_shape,
            damping=0.15,
        )
        sim_cam = EventCameraSimulator(
            height=int(events.shape[-2]),
            width=int(events.shape[-1]),
            event_threshold=0.08,
            in_channels=int(events.shape[2]),
        )
        event_stream = OnlineEventCameraStream(sim_cam, phys_stepper.q)

        for t in range(t_start, t_grasp + 1):
            target_pose_plan = torch.stack(
                [target_grasp_mat if (t + tau) <= t_c else target_final_mat for tau in range(controller.horizon)],
                dim=0,
            )
            res = controller.plan_step(curr_z, target_pose_plan, prev_action=prev_u)
            u_exec = res.action
            prev_u = u_exec
            controller.prev_action = u_exec.clone()

            z_pred = model.predictor(curr_z, u_exec.unsqueeze(0)).predicted_latent
            dec_pred = model.physical_decoder(z_pred)
            q_des = dec_pred[0, :6].detach().cpu().numpy()
            dq_des = dec_pred[0, 6:12].detach().cpu().numpy()
            q_phys_next, _, u_pd = phys_stepper.step_pd_target(
                q_des=q_des,
                dq_des=dq_des,
                payload_attached=(t >= t_c),
            )
            ev_online = event_stream.step(q_phys_next).to(device)
            u_pd_t = torch.from_numpy(u_pd).to(device).unsqueeze(0)

            z_obs, dyn_state = step_grounded(
                model,
                ev_online,
                dyn_state,
                in_sensory_phase=True,
                unconstrained=False,
                action=u_pd_t,
            )
            if int(phase_flags[t].item()) == 1:
                w_t = _compute_residual_handover_weight(
                    fk, model.physical_decoder, z_pred, z_obs, target_final_mat, t, t_c
                )
                total_p1_steps += 1
                w_values_all.append(float(w_t))
                if w_t > 0.0:
                    triggered_p1_steps += 1
                    w_values_nonzero.append(float(w_t))
                curr_z = (1.0 - w_t) * z_pred + w_t * z_obs
            else:
                curr_z = z_pred

    return {
        "total_phi1_steps": int(total_p1_steps),
        "triggered_phi1_steps": int(triggered_p1_steps),
        "trigger_rate_pct": float(100.0 * triggered_p1_steps / max(1, total_p1_steps)),
        "mean_w_all_phi1": float(np.mean(w_values_all)) if w_values_all else 0.0,
        "mean_w_when_triggered": float(np.mean(w_values_nonzero)) if w_values_nonzero else 0.0,
    }


@torch.no_grad()
def audit_checkpoint(
    label: str,
    ckpt_path: Path,
    cfg: ExperimentConfig,
    test_loader: DataLoader,
    dataset_dir: Path,
    episode_ids: List[int],
    device: torch.device,
) -> Dict[str, Any]:
    """Audit internal module parameters, ALIF spike distribution, and PIL handover rate."""
    model = SPWM.from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    w_u = model.dynamics.action_proj.weight.detach().cpu().float()
    w_u_frob = float(torch.linalg.matrix_norm(w_u, ord="fro").item())
    w_u_spec = float(torch.linalg.matrix_norm(w_u, ord=2).item())

    gs_raw = model.dynamics.gate_scale.detach().cpu().float()
    gs_sp = F.softplus(gs_raw)

    all_spikes: List[torch.Tensor] = []
    for batch in test_loader:
        events = batch["events"].to(device)
        actions = batch["actions"].to(device)
        _, _, spk = model.forward_representation_with_spikes(events, actions=actions)
        all_spikes.append(spk.cpu())
    spikes_cat = torch.cat(all_spikes, dim=0)
    spike_stats = collect_spike_statistics(spikes_cat)

    handover_stats = measure_handover_trigger_rate_pil(model, cfg, dataset_dir, episode_ids, device)

    epoch_num = int(ckpt["epoch"]) if isinstance(ckpt, dict) and "epoch" in ckpt else -1
    best_epoch_num = int(ckpt["best_epoch"]) if isinstance(ckpt, dict) and "best_epoch" in ckpt and ckpt["best_epoch"] is not None else -1

    return {
        "label": label,
        "checkpoint": str(ckpt_path),
        "epoch": epoch_num,
        "best_epoch": best_epoch_num,
        "action_proj_spectral_norm": w_u_spec,
        "action_proj_frobenius_norm": w_u_frob,
        "gate_scale_raw": {
            "mean": float(gs_raw.mean().item()),
            "std": float(gs_raw.std().item()),
            "min": float(gs_raw.min().item()),
            "max": float(gs_raw.max().item()),
        },
        "gate_scale_softplus": {
            "mean": float(gs_sp.mean().item()),
            "std": float(gs_sp.std().item()),
            "min": float(gs_sp.min().item()),
            "max": float(gs_sp.max().item()),
        },
        "alif_spike_stats": spike_stats,
        "pil_handover_stats": handover_stats,
    }


def main() -> None:
    cfg = ExperimentConfig.load(REPO_ROOT / "results/spwm_v0_12_4_train_m1/config.yaml")
    device = torch.device("cpu")
    dataset_dir = REPO_ROOT / "sim/ExportedTrajectories"
    ds_test = EventWorldDataset(list(range(3600, 4000)), dataset_dir=dataset_dir)
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False)
    episode_ids = list(range(3600, 3650))

    targets = [
        ("Seed 42 (best_model.pt)", REPO_ROOT / "results/spwm_v0_12_4_train_m1/best_model.pt"),
        ("Seed 42 (final checkpoint.pt)", REPO_ROOT / "results/spwm_v0_12_4_train_m1/checkpoint.pt"),
        ("Seed 101 (best_model.pt)", REPO_ROOT / "results/multiseed/seed_101_m1/best_model.pt"),
        ("Seed 101 (final checkpoint.pt)", REPO_ROOT / "results/multiseed/seed_101_m1/checkpoint.pt"),
        ("Seed 2024 (best_model.pt)", REPO_ROOT / "results/multiseed/seed_2024_m1/best_model.pt"),
        ("Seed 2024 (final checkpoint.pt)", REPO_ROOT / "results/multiseed/seed_2024_m1/checkpoint.pt"),
    ]

    results: Dict[str, Any] = {}
    for label, ckpt_path in targets:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        res = audit_checkpoint(label, ckpt_path, cfg, test_loader, dataset_dir, episode_ids, device)
        results[label] = res
        sp = res["alif_spike_stats"]
        ho = res["pil_handover_stats"]
        print(
            f"{label:32s} | ep={res['epoch']:3d} | ||W_u||_2={res['action_proj_spectral_norm']:.4f} "
            f"(||W_u||_F={res['action_proj_frobenius_norm']:.4f}) | gate_sp={res['gate_scale_softplus']['mean']:.2f}±{res['gate_scale_softplus']['std']:.2f} "
            f"| spk={sp['mean_spike_rate']*100:.2f}% (silent={sp['fraction_silent_neurons']*100:.1f}%, hyper={sp['fraction_hyperactive_neurons']*100:.1f}%) "
            f"| w_t>0={ho['trigger_rate_pct']:.1f}% (mean_w={ho['mean_w_when_triggered']:.3f})"
        )

    out_path = REPO_ROOT / "results/diagnostics/spwm_seed_variance_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved seed variance audit to {out_path}")


if __name__ == "__main__":
    main()
