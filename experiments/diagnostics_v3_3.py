"""
SPWM-v3.3 Standalone Diagnostics
===================================
Runs three targeted analyses to characterise why Probe Loss stalled at ~0.306
(R^2 approx 0) and check whether the post-synaptic filter fix resolves the bottleneck.

Diagnostica 1 - Linear Probe R^2 on filtered latent space
Diagnostica 2 - Probe gradient norms (first 2 batches)
Diagnostica 3 - Latent distribution shift: Teacher-Forcing vs Autonomous Rollout (H=10)
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from spwm.utils.reproducibility import set_seed
from spwm.utils.config import load_config
from spwm.data.datasets import create_dataloaders
from spwm.models.world_model import SPWM

CONFIG_PATH = "configs/experiments/spwm_v3_3.yaml"
OUTPUT_DIR  = Path("results/spwm_v3_3_diagnostics")
SEED        = 42
ALPHA_FILTER = 0.85
ROLLOUT_H   = 10


def apply_ema_filter(z: torch.Tensor, alpha: float = 0.85) -> torch.Tensor:
    B, T, D = z.shape
    r = torch.zeros(B, D, device=z.device, dtype=z.dtype)
    out = []
    for t in range(T):
        r = alpha * r + (1.0 - alpha) * z[:, t]
        out.append(r)
    return torch.stack(out, dim=1)


def ridge_r2(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3) -> float:
    X = X.double().cpu()
    Y = Y.double().cpu()
    N, D = X.shape
    XtX = X.T @ X + lam * torch.eye(D, dtype=torch.float64)
    XtY = X.T @ Y
    try:
        W = torch.linalg.solve(XtX, XtY)
    except Exception:
        W, _, _, _ = torch.linalg.lstsq(X, Y)
    Y_hat = X @ W
    ss_res = ((Y - Y_hat) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    r2_per_dim = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2_per_dim.mean().item()


def diag1_linear_probe_r2(model, loader, device, max_batches=10):
    print("\n" + "=" * 60)
    print("  DIAGNOSTICA 1 -- Linear Probe R^2 on Latent Space")
    print("=" * 60)
    model.eval()
    all_z_raw, all_z_filt, all_kin = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            events   = batch["events"].to(device)
            true_kin = batch.get("flat_kinematics")
            if true_kin is None:
                print("  [SKIP] No flat_kinematics in batch."); break
            true_kin = true_kin.to(device)
            out = model(events, accumulate_local_updates=False)
            z = out.latent_states
            z_filt = apply_ema_filter(z, alpha=ALPHA_FILTER)
            B, T, D = z.shape
            all_z_raw.append(z.reshape(B * T, D).cpu())
            all_z_filt.append(z_filt.reshape(B * T, D).cpu())
            all_kin.append(true_kin.reshape(B * T, -1).cpu())

    if not all_z_raw:
        print("  [ERROR] No data."); return {}

    Z_raw  = torch.cat(all_z_raw,  dim=0)
    Z_filt = torch.cat(all_z_filt, dim=0)
    Y      = torch.cat(all_kin,    dim=0)

    print(f"  Samples: {Z_raw.shape[0]}  |  Latent dim: {Z_raw.shape[1]}  |  Kin dim: {Y.shape[1]}")
    trivial_mse = Y.var(dim=0).mean().item()
    print(f"\n  Target statistics:")
    print(f"    Mean : {Y.mean(dim=0).tolist()}")
    print(f"    Std  : {Y.std(dim=0).tolist()}")
    print(f"    Trivial MSE (variance): {trivial_mse:.4f}")

    r2_raw  = ridge_r2(Z_raw,  Y)
    r2_filt = ridge_r2(Z_filt, Y)
    print(f"\n  Ridge R^2 RAW spikes         : {r2_raw:+.4f}")
    print(f"  Ridge R^2 FILTERED traces    : {r2_filt:+.4f}")

    if r2_filt < 0.10:
        verdict = "WARNING: SNN is NOT encoding kinematics. Representational failure."
    elif r2_filt < 0.40:
        verdict = "WEAK: Filter helps, but representation needs improvement."
    elif r2_filt < 0.60:
        verdict = "MODERATE: Optimizer was bottleneck; filter fix should help."
    else:
        verdict = "STRONG: Information present; filter unlocks MLP convergence."
    print(f"\n  >> {verdict}")
    return {"r2_raw": r2_raw, "r2_filtered": r2_filt, "trivial_mse": trivial_mse}


def diag2_probe_grad_norms(model, loader, device, num_batches=2):
    print("\n" + "=" * 60)
    print("  DIAGNOSTICA 2 -- Probe Gradient Norm Inspection")
    print("=" * 60)
    probe_params = [p for n, p in model.named_parameters()
                    if ("decoder" in n or "probe" in n) and p.requires_grad]
    if not probe_params:
        print("  [ERROR] No probe parameters found."); return {}

    opt = torch.optim.AdamW(probe_params, lr=2e-3)
    model.train()
    losses, grad_norms = [], []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        events   = batch["events"].to(device)
        true_kin = batch.get("flat_kinematics")
        if true_kin is None:
            print("  [SKIP] No flat_kinematics."); break
        true_kin = true_kin.to(device)

        with torch.no_grad():
            out = model(events, accumulate_local_updates=False)

        opt.zero_grad()
        with torch.enable_grad():
            z_det   = out.latent_states.detach()
            decoded = model.physical_decoder(z_det)
            probe_loss = nn.functional.mse_loss(decoded, true_kin)
            probe_loss.backward()

        total_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in probe_params if p.grad is not None
        ) ** 0.5
        opt.step()
        losses.append(probe_loss.item())
        grad_norms.append(total_norm)

        print(f"  Batch {i+1}/{num_batches}")
        print(f"    Probe Loss    : {probe_loss.item():.5f}")
        print(f"    Gradient Norm : {total_norm:.6f}", end="")
        if total_norm < 1e-4:
            print("  [WARNING: VANISHING]")
        elif total_norm > 10.0:
            print("  [WARNING: EXPLODING]")
        else:
            print("  [OK]")

    return {"probe_losses": losses, "grad_norms": grad_norms}


def diag3_latent_distribution_shift(model, loader, device, rollout_horizon=ROLLOUT_H, num_batches=5):
    print("\n" + "=" * 60)
    print(f"  DIAGNOSTICA 3 -- TF vs Rollout Latent Distribution (H={rollout_horizon})")
    print("=" * 60)
    model.eval()
    tf_list, ro_list = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            events = batch["events"].to(device)
            out    = model(events, accumulate_local_updates=False)
            z_tf   = out.latent_states
            tf_list.append(z_tf.reshape(-1, z_tf.shape[-1]).cpu())
            z0  = z_tf[:, -1]
            z_ro = model.predict_future(z0, horizon=rollout_horizon)
            ro_list.append(z_ro.reshape(-1, z_ro.shape[-1]).cpu())

    if not tf_list:
        print("  [ERROR] No data."); return {}

    Z_tf = torch.cat(tf_list, dim=0)
    Z_ro = torch.cat(ro_list, dim=0)

    def stats(Z, label):
        print(f"\n  [{label}]")
        print(f"    Min  : {Z.min().item():+.4f}")
        print(f"    Max  : {Z.max().item():+.4f}")
        print(f"    Mean : {Z.mean().item():+.4f}")
        print(f"    Std  : {Z.std().item():.4f}")
        sp = (Z.abs() < 0.5).float().mean().item()
        print(f"    Sparsity (|z|<0.5): {sp:.3f}")
        return {"min": Z.min().item(), "max": Z.max().item(),
                "mean": Z.mean().item(), "std": Z.std().item(), "sparsity": sp}

    tf_d = stats(Z_tf, "Teacher-Forcing z_t")
    ro_d = stats(Z_ro, f"Rollout z_t (H={rollout_horizon})")

    ratio = Z_ro.abs().mean().item() / (Z_tf.abs().mean().item() + 1e-8)
    print(f"\n  Magnitude ratio Rollout/TF: {ratio:.3f}", end="")
    if ratio > 3.0:
        print("  [WARNING: LATENT EXPLOSION]")
    elif ratio > 1.5:
        print("  [MILD DRIFT]")
    else:
        print("  [STABLE]")

    return {"teacher_forcing": tf_d, "rollout": ro_d, "magnitude_ratio": ratio}


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n{'='*60}\n  SPWM-v3.3 PRE-TRAINING DIAGNOSTICS\n  Device: {device}\n{'='*60}")

    config    = load_config(CONFIG_PATH)
    data_cfg  = config.get("data", {})
    env_cfg   = config.get("environment", {})
    model_cfg = config.get("model", {})
    mem_cfg   = config.get("memory", {})
    neuron_cfg = config.get("neuron", {})

    dataloaders = create_dataloaders(
        total_trajectories=data_cfg.get("total_trajectories", 600),
        train_split=data_cfg.get("train_split", 0.8),
        val_split=data_cfg.get("val_split", 0.1),
        sequence_length=data_cfg.get("sequence_length", 150),
        batch_size=8,
        height=env_cfg.get("height", 32),
        width=env_cfg.get("width", 32),
        num_objects=env_cfg.get("num_objects", 1),
        num_workers=0,
        cache_data=True,
    )

    latent_dim = model_cfg.get("latent_dim", 128)
    model = SPWM(
        in_channels=model_cfg.get("in_channels", 2),
        height=env_cfg.get("height", 32),
        width=env_cfg.get("width", 32),
        encoder_dim=model_cfg.get("encoder_dim", 128),
        latent_dim=latent_dim,
        timescale_dims=tuple(mem_cfg.get("timescale_dims", (64, 64))),
        betas=tuple(mem_cfg.get("betas", (0.90, 0.985))),
        beta_mem=neuron_cfg.get("beta_mem", 0.80),
        threshold=neuron_cfg.get("threshold", 1.0),
        gamma=neuron_cfg.get("gamma", 0.18),
        surrogate_name=neuron_cfg.get("surrogate", "atan"),
        surrogate_alpha=neuron_cfg.get("surrogate_alpha", 2.0),
        predictor_hidden_dim=model_cfg.get("predictor_hidden_dim", 256),
        num_objects=env_cfg.get("num_objects", 1),
    ).to(device)

    for candidate in ["results/spwm_v3_2/model.pt", "results/spwm_v3_1/model.pt"]:
        p = Path(candidate)
        if p.is_file():
            try:
                ckpt = torch.load(p, map_location=device)
                state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
                model.load_state_dict(state, strict=False)
                print(f"\n  Loaded weights from: {p}")
            except Exception as e:
                print(f"\n  [WARN] Could not load {p}: {e}. Using random init.")
            break

    results = {}
    t0 = time.time()

    results["diagnostica_1_r2"] = diag1_linear_probe_r2(model, dataloaders["train"], device)
    results["diagnostica_2_grad_norms"] = diag2_probe_grad_norms(model, dataloaders["train"], device)
    results["diagnostica_3_latent_shift"] = diag3_latent_distribution_shift(model, dataloaders["val"], device)

    print(f"\n{'='*60}")
    print(f"  All diagnostics done in {time.time() - t0:.1f}s")
    print(f"{'='*60}\n")

    report = OUTPUT_DIR / "diagnostics_report.json"
    with open(report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Report saved: {report}\n")


if __name__ == "__main__":
    main()
