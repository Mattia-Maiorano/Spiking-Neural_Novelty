#!/usr/bin/env python3
"""
TASK: Layer-by-Layer R^2 Profiling (Isolamento del Bottleneck Spaziale).

Measures linear R^2 (via closed-form LSTSQ) with respect to ground-truth kinematics
(x, y, vx, vy) along each intermediate stage of SPWM:
- Level 0: Raw flattened event frames [B*T, 2048]
- Level 1: EventEncoder output: enc_out [B*T, 128]
- Level 2: Dynamics input projection: input_proj(enc_out) [B*T, 128]
- Level 3: ALIF membrane potentials: v_mem (fast & slow concatenated) [B*T, 128]
- Level 4: Final latent state: z_t [B*T, 128]

Also inspects:
- EventEncoder architecture (pooling, operations)
- Parameter training status / weight norms (untrained vs checkpointed)
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

from spwm.utils.config import load_config
from spwm.data.datasets import create_dataloaders
from spwm.models.world_model import SPWM


def compute_lstsq_r2(X: torch.Tensor, Y: torch.Tensor):
    """
    Computes closed-form LSTSQ linear regression and R^2 score.
    X: [N, D]
    Y: [N, 4]
    """
    N = X.shape[0]
    # Add bias term
    bias = torch.ones((N, 1), dtype=torch.float32, device=X.device)
    X_b = torch.cat([X, bias], dim=1)

    # Solve closed-form LSTSQ
    try:
        sol = torch.linalg.lstsq(X_b, Y).solution
        Y_hat = X_b @ sol
    except Exception as e:
        print(f"LSTSQ Error: {e}")
        return 0.0, [0.0, 0.0, 0.0, 0.0]

    # Total R^2
    ss_res = (Y - Y_hat).pow(2).sum()
    ss_tot = (Y - Y.mean(0, keepdim=True)).pow(2).sum()
    r2_total = (1 - ss_res / (ss_tot + 1e-8)).item()

    # Component R^2: [x, y, vx, vy]
    ss_res_comp = (Y - Y_hat).pow(2).sum(dim=0)
    ss_tot_comp = (Y - Y.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    r2_comp = (1 - ss_res_comp / (ss_tot_comp + 1e-8)).tolist()

    return r2_total, r2_comp


def profile_model(model: SPWM, batch: dict, desc: str = ""):
    print("\n" + "=" * 75)
    print(f"PROFILING RUN: {desc}")
    print("=" * 75)

    events = batch["events"]  # [B, T, 2, H, W]
    true_kin = batch["flat_kinematics"]  # [B, T, 4] (x, y, vx, vy)
    B, T, C, H, W = events.shape

    Y = true_kin.view(B * T, 4).float()

    # --- Livello 0: Frame grezzi appiattiti [B*T, 2048] ---
    X0 = events.view(B * T, -1).float()
    r2_tot_0, r2_comp_0 = compute_lstsq_r2(X0, Y)

    # Forward through model stages step-by-step
    model.eval()
    state = model.init_state(B, device=events.device)

    enc_out_list = []
    input_proj_list = []
    v_mem_list = []
    z_t_list = []

    with torch.no_grad():
        for t in range(T):
            frame_t = events[:, t]
            
            # Livello 1: Encoder
            sensory_x, new_enc_states = model.encoder.step(frame_t, state.encoder_states)
            enc_out_list.append(sensory_x)

            # Livello 2: Input Projection (somatic synaptic current from sensory input)
            inp_proj_t = model.dynamics.input_proj(sensory_x)
            input_proj_list.append(inp_proj_t)

            # Recurrent step
            soma_current = inp_proj_t + model.dynamics.recurrent_proj(state.dynamics_state.z_prev)
            spikes, new_mem_state = model.dynamics.memory(
                synaptic_inputs=soma_current,
                state=state.dynamics_state.memory_state,
            )

            # Livello 3: Concatenated v_mems of ALIF
            v_mem_t = new_mem_state.concatenated_mems
            v_mem_list.append(v_mem_t)

            # Livello 4: Latent state z_t
            mem_analog = torch.tanh(new_mem_state.concatenated_mems)
            z_t = model.dynamics.norm(model.dynamics.fuse_spikes(spikes) + model.dynamics.fuse_mems(mem_analog))
            z_t_list.append(z_t)

            # Update state container
            new_dyn_state = type(state.dynamics_state)(memory_state=new_mem_state, z_prev=z_t)
            state = type(state)(encoder_states=new_enc_states, dynamics_state=new_dyn_state)

    X1 = torch.stack(enc_out_list, dim=1).view(B * T, -1).float()
    X2 = torch.stack(input_proj_list, dim=1).view(B * T, -1).float()
    X3 = torch.stack(v_mem_list, dim=1).view(B * T, -1).float()
    X4 = torch.stack(z_t_list, dim=1).view(B * T, -1).float()

    r2_tot_1, r2_comp_1 = compute_lstsq_r2(X1, Y)
    r2_tot_2, r2_comp_2 = compute_lstsq_r2(X2, Y)
    r2_tot_3, r2_comp_3 = compute_lstsq_r2(X3, Y)
    r2_tot_4, r2_comp_4 = compute_lstsq_r2(X4, Y)

    results = [
        ("Livello 0: Frame grezzi", list(X0.shape), (r2_comp_0[0] + r2_comp_0[1])/2, (r2_comp_0[2] + r2_comp_0[3])/2, r2_tot_0),
        ("Livello 1: EventEncoder (enc_out)", list(X1.shape), (r2_comp_1[0] + r2_comp_1[1])/2, (r2_comp_1[2] + r2_comp_1[3])/2, r2_tot_1),
        ("Livello 2: Input Proj (input_proj)", list(X2.shape), (r2_comp_2[0] + r2_comp_2[1])/2, (r2_comp_2[2] + r2_comp_2[3])/2, r2_tot_2),
        ("Livello 3: Membrana ALIF (v_mem)", list(X3.shape), (r2_comp_3[0] + r2_comp_3[1])/2, (r2_comp_3[2] + r2_comp_3[3])/2, r2_tot_3),
        ("Livello 4: Stato Latente (z_t)", list(X4.shape), (r2_comp_4[0] + r2_comp_4[1])/2, (r2_comp_4[2] + r2_comp_4[3])/2, r2_tot_4),
    ]

    print(f"| {'Stadio':<35} | {'Shape':<15} | {'R^2(x, y)':<10} | {'R^2(vx, vy)':<12} | {'R^2 Totale':<10} |")
    print("|" + "-" * 37 + "|" + "-" * 17 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 12 + "|")
    for name, shape, r2_pos, r2_vel, r2_tot in results:
        shape_str = f"[{shape[0]}, {shape[1]}]"
        print(f"| {name:<35} | {shape_str:<15} | {r2_pos:<10.4f} | {r2_vel:<12.4f} | {r2_tot:<10.4f} |")

    # Detailed per-component breakdown
    print("\nDettaglio componenti [x, y, vx, vy]:")
    print(f" - L0 (Raw):      {[round(x, 4) for x in r2_comp_0]}")
    print(f" - L1 (Encoder):  {[round(x, 4) for x in r2_comp_1]}")
    print(f" - L2 (InputProj):{[round(x, 4) for x in r2_comp_2]}")
    print(f" - L3 (ALIF Mem): {[round(x, 4) for x in r2_comp_3]}")
    print(f" - L4 (Latent z): {[round(x, 4) for x in r2_comp_4]}")


def inspect_encoder(model: SPWM):
    print("\n" + "=" * 75)
    print("ISPEZIONE DETTAGLIATA EVENT ENCODER (SPWM-v3.7 Continuous Conv)")
    print("=" * 75)
    encoder = model.encoder
    print(encoder)

    print("\n1. Controllo architettura e moduli:")
    if hasattr(encoder, "net"):
        for i, layer in enumerate(encoder.net):
            print(f"   * Layer {i}: {layer}")

    print("\n2. Controllo Pesi e Training Status dell'Encoder:")
    for name, param in encoder.named_parameters():
        grad_status = param.requires_grad
        norm = param.data.norm().item()
        mean = param.data.mean().item()
        std = param.data.std().item()
        print(f"   * {name:<25}: shape={list(param.shape)}, requires_grad={grad_status}, norm={norm:.4f}, mean={mean:.5f}, std={std:.5f}")


def main():
    # Load dataset
    dataloaders = create_dataloaders(
        total_trajectories=100,
        sequence_length=150,
        batch_size=16,
        num_objects=1,
        num_workers=0,
        cache_data=True,
    )
    batch = next(iter(dataloaders["train"]))

    # Test 1: Untrained / Randomly Initialized Model (SPWM-v3.7)
    cfg_path = "configs/experiments/spwm_v3_7.yaml" if os.path.exists("configs/experiments/spwm_v3_7.yaml") else "configs/experiments/spwm_v3_6.yaml"
    cfg = load_config(cfg_path)
    untrained_model = SPWM(
        in_channels=2,
        height=32,
        width=32,
        encoder_conv_channels=(16, 32),
        encoder_dim=128,
        latent_dim=128,
        timescale_dims=(64, 64),
        betas=(0.90, 0.985),
        beta_mem=0.80,
        threshold=1.0,
        gamma=0.18,
    )
    inspect_encoder(untrained_model)
    profile_model(untrained_model, batch, desc="MODELLO NON ADDESTRATO v3.7 (Random Init)")

    # Test 2: Checkpoint Trained Model (spwm_v3_7 or spwm_v3_6) if available
    ckpt_path = "results/spwm_v3_7_test/checkpoint.pt" if os.path.exists("results/spwm_v3_7_test/checkpoint.pt") else "results/spwm_v3_6/checkpoint.pt"
    if os.path.exists(ckpt_path):
        trained_model = SPWM(
            in_channels=2,
            height=32,
            width=32,
            encoder_conv_channels=(16, 32),
            encoder_dim=128,
            latent_dim=128,
            timescale_dims=(64, 64),
            betas=(0.90, 0.985),
            beta_mem=0.80,
            threshold=1.0,
            gamma=0.18,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            trained_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "model_state" in ckpt:
            trained_model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            trained_model.load_state_dict(ckpt, strict=False)

        print("\n" + "=" * 75)
        print(f"VERIFICA DEI PESI ADDESTRATI (Checkpoint: {ckpt_path})")
        print("=" * 75)
        for n, p in trained_model.named_parameters():
            if "encoder" in n:
                print(f" - [TRAINED] {n:<30}: norm={p.data.norm().item():.4f}")

        profile_model(trained_model, batch, desc=f"MODELLO ADDESTRATO (Checkpoint: {ckpt_path})")
    else:
        print(f"\nCheckpoint {ckpt_path} non trovato, eseguito solo su modello base.")


if __name__ == "__main__":
    main()

