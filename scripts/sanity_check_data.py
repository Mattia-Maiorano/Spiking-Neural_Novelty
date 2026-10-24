#!/usr/bin/env python3
"""
Data Integrity & Direct Image Sanity Check (Bypass SNN).

Validates:
1. Ground truth kinematic alignment with generated events.
2. Center of mass correlation with (x, y).
3. Direct linear regression (lstsq) from raw event frames to flat kinematics.
"""

import sys
import torch
import numpy as np

from spwm.data.datasets import create_dataloaders


def main():
    print("=" * 70)
    print("TASK: Data Integrity & Direct Image Sanity Check (Bypass SNN)")
    print("=" * 70)

    # 1. Carica dataloaders e un batch reale dal train_loader
    print("\n[1] Loading dataset...")
    dataloaders = create_dataloaders(
        total_trajectories=100,
        sequence_length=150,
        batch_size=16,
        joint_count=8,
        action_dim=8,
        num_workers=0,
        cache_data=True,
    )
    train_loader = dataloaders["train"]
    batch = next(iter(train_loader))

    events = batch["events"]  # [B, T, 2, H, W]
    true_kin = batch["flat_kinematics"]  # [B, T, 2*J]
    positions = batch["positions"]  # [B, T, J]
    actions = batch["actions"]  # [B, T, D_u]

    B, T, C, H, W = events.shape
    kin_dim = true_kin.shape[-1]
    print(f"Batch shapes:")
    print(f" - Events: {events.shape} (B={B}, T={T}, C={C}, H={H}, W={W})")
    print(f" - Flat Kinematics: {true_kin.shape}")
    print(f" - Positions: {positions.shape}")
    print(f" - Actions: {actions.shape}")

    # 2. Calcola il centro di massa grezzo direttamente dai pixel degli eventi
    print("\n[2] Computing Raw Center of Mass (CoM) from events...")
    y_coords = torch.linspace(-1.0, 1.0, H)
    x_coords = torch.linspace(-1.0, 1.0, W)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H, W]

    com_list_x = []
    com_list_y = []
    gt_list_q0 = []
    gt_list_q1 = []

    for b in range(B):
        for t in range(1, T):  # t=0 has no events (delta_L=0)
            frame = events[b, t].sum(dim=0)  # [H, W] sum polarities
            total_mass = frame.sum()
            if total_mass > 0:
                u = (frame * grid_x).sum() / total_mass
                v = (frame * grid_y).sum() / total_mass
                com_list_x.append(u.item())
                com_list_y.append(v.item())
                gt_list_q0.append(true_kin[b, t, 0].item())
                gt_list_q1.append(true_kin[b, t, 1].item())

    corr_x = np.corrcoef(com_list_x, gt_list_q0)[0, 1]
    corr_y = np.corrcoef(com_list_y, gt_list_q1)[0, 1]
    print(f" - Correlation CoM_x vs True_q0: {corr_x:.4f}")
    print(f" - Correlation CoM_y vs True_q1: {corr_y:.4f}")

    # 3. Regressione lineare diretta frame-to-kinematics
    print("\n[3] Direct Frame-to-Kinematics Linear Regression (Closed-Form LSTSQ)...")
    X = events.view(B * T, -1).float()  # [B*T, 2 * H * W]
    bias = torch.ones((B * T, 1), dtype=torch.float32)
    X_with_bias = torch.cat([X, bias], dim=1)

    Y = true_kin.view(B * T, kin_dim).float()

    lstsq_res = torch.linalg.lstsq(X_with_bias, Y)
    W = lstsq_res.solution
    Y_hat = X_with_bias @ W

    ss_res = (Y - Y_hat).pow(2).sum()
    ss_tot = (Y - Y.mean(0, keepdim=True)).pow(2).sum()
    r2_total = (1 - ss_res / ss_tot).item()

    ss_res_comp = (Y - Y_hat).pow(2).sum(dim=0)
    ss_tot_comp = (Y - Y.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    r2_comp = (1 - ss_res_comp / ss_tot_comp).tolist()

    print(f" - Single Batch Total R^2 Score: {r2_total:.4f}")
    print(f" - Single Batch Component R^2: {[round(v, 4) for v in r2_comp]}")

    print("\n[3b] Evaluating across entire train set (all batches)...")
    all_X, all_Y = [], []
    for b_data in train_loader:
        ev = b_data["events"]
        kin = b_data["flat_kinematics"]
        b_B, b_T = ev.shape[0], ev.shape[1]
        all_X.append(ev.view(b_B * b_T, -1))
        all_Y.append(kin.view(b_B * b_T, kin_dim))

    all_X = torch.cat(all_X, dim=0).float()
    all_Y = torch.cat(all_Y, dim=0).float()
    all_bias = torch.ones((all_X.shape[0], 1), dtype=torch.float32)
    all_X_bias = torch.cat([all_X, all_bias], dim=1)

    all_W = torch.linalg.lstsq(all_X_bias, all_Y).solution
    all_Y_hat = all_X_bias @ all_W

    all_ss_res = (all_Y - all_Y_hat).pow(2).sum()
    all_ss_tot = (all_Y - all_Y.mean(0, keepdim=True)).pow(2).sum()
    all_r2_total = (1 - all_ss_res / all_ss_tot).item()

    all_ss_res_comp = (all_Y - all_Y_hat).pow(2).sum(dim=0)
    all_ss_tot_comp = (all_Y - all_Y.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)
    all_r2_comp = (1 - all_ss_res_comp / all_ss_tot_comp).tolist()

    print(f" - Full Train Set Total R^2 Score: {all_r2_total:.4f}")
    print(f" - Full Train Set Component R^2: {[round(v, 4) for v in all_r2_comp]}")

    # 4. Stampa l'esito
    print("\n" + "=" * 70)
    print("ESITO SANITY CHECK:")
    pos_r2_avg = (all_r2_comp[0] + all_r2_comp[1]) / 2.0
    print(f"Average Position R^2 (x, y): {pos_r2_avg:.4f}")
    if pos_r2_avg > 0.70 or all_r2_total > 0.70:
        print(">> VERDETTO: I dati sono CORRETTI e perfettamente allineati!")
        print(">> L'informazione cinematica e spaziale è presente con altissima fedeltà nei frame di eventi.")
        print(">> La perdita di informazione o il collasso di R^2 avviene all'interno della SNN o dell'Encoder.")
    elif all_r2_total < 0.10:
        print(">> VERDETTO: `flat_kinematics` ed `events` sono disaccoppiati/sfasati nel DataLoader.")
    else:
        print(f">> VERDETTO: R^2 intermedio ({all_r2_total:.4f}).")
    print("=" * 70)


if __name__ == "__main__":
    main()
