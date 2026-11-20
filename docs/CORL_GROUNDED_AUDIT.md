# CoRL Grounded Audit & Benchmark Report (`SPWM-v0.12.4`)

> [!NOTE]
> **Historical Nov 17 Audit Snapshot:** This document records the initial Nov 17 `SPWM-v0.12.4` leakage & PIL fidelity audit (`t >= t_grasp` vs `t >= t_c`). All final, multi-seed (`42, 101, 2024`), symmetric `t >= t_c` PIL benchmarks, disentangled Stage 4 ablations, $w_t$ activation telemetry, and Open-Loop vs. Online baseline comparisons are consolidated in [**docs/CORL_FINAL_BENCHMARK_SUMMARY.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/CORL_FINAL_BENCHMARK_SUMMARY.md).

**Audit Date:** 2026-11-17  
**Audited Checkpoint:** [`results/spwm_v0_12_4_train_m1/model.pt`](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/model.pt) (`392,141` total / `317,900` active inference parameters)  
**AST Anti-Fallback Gate:** Clean (`0` violations across `spwm/`, `experiments/benchmark_classical_and_bc.py`, `experiments/benchmark_robustness_stress_test.py`; `20/20` contract & anti-fallback tests passing).

---

## Executive Summary

1. **Test-Time Oracle Leakage Audit:**
   - **Physics-in-the-Loop (PIL)** closed-loop evaluation (`--physics-in-loop` in `benchmark_mppi_control.py` and `mppi_scorer.py`) has **zero test-time ground-truth state leakage** in state estimation and residual-gated sensory handover: `z_obs` is produced strictly causally from `OnlineEventCameraStream` observing `q_phys_next` and `u_pd_t`, and `dist_pred`, `dist_obs`, `rot_obs` compare decoded predictions against the two workspace SE(3) task goal poses (`target_grasp_mat`, `target_final_mat`).
   - **Latent-Grounded (non-PIL) mode** (`physics_in_loop=False`) **does** leak expert-demonstration latent states (`z_gt_next = z_gt_seq[:, t+1]`) during `phase_flags == 1` handover (`w_t <= 0.85`). Therefore, **PIL (`88.0%`, 44/50)** is the primary physically honest closed-loop benchmark, while Latent-Grounded (`96.0%`, 48/50) is an idealized upper bound.
2. **PIL vs. Unity PhysX 4 Fidelity Gap:**
   - Quantified exact kinematic, dynamic, and visual discrepancies between `ArticulatedArmPhysicsStepper` + `OnlineEventCameraStream` and Unity PhysX 4 (`ArticulatedRobotRig.cs`, `ProceduralObjectSpawner.cs`, `NeuromorphicEventBridge.cs`), including a discovered **FixedJoint attachment timing bug** (`payload_attached=(t >= t_grasp)` instead of `t >= t_c`, which accounts for a `2.0 pp` difference on SPWM PIL: `88.0%` vs `86.0%`).
3. **Complete 50-Episode PIL Ablations & Robustness:**
   - Filled all missing PIL entries (`---`) in the 4-stage SPWM ablation (`62.0% -> 62.0% -> 74.0% -> 88.0%`).
   - Re-ran the robustness stress test across **all 50 canonical episodes** under both Latent-Grounded (`96.0%` nominal) and PIL (`88.0%` nominal), eliminating the 20-episode subset and hardcoded `MPPIConfig` discrepancy.
4. **SSOT Synchronization:**
   - Purged all stale `v0.12.20` (`328.7k`, `12.81 nJ`, `0.43 nJ`) numbers from [`docs/DOSSIER.md`](file:///Users/Mattia/Desktop/Studies/Temp/docs/DOSSIER.md) and synchronized [`docs/FIGURE_METRICS.md`](file:///Users/Mattia/Desktop/Studies/Temp/docs/FIGURE_METRICS.md), [`results/mppi_benchmark/mppi_benchmark_metrics.json`](file:///Users/Mattia/Desktop/Studies/Temp/results/mppi_benchmark/mppi_benchmark_metrics.json), and [`results/robustness_benchmark/robustness_metrics.json`](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark/robustness_metrics.json).

---

## Phase 1: Adversarial Code & Data-Leakage Audit

### 1.1 Test-Time Oracle Leakage in Residual-Gated Handover & Target Lookahead
- **PIL Handover (`benchmark_mppi_control.py` L322–L335, `mppi_scorer.py` L381–L396):**
  - `z_pred = model.predictor(curr_z, u_exec).predicted_latent`
  - `z_obs, dyn_state = step_grounded(model, ev_online, dyn_state, in_sensory_phase=True, unconstrained=False, action=u_pd_t)` where `ev_online = event_stream.step(q_phys_next)`
  - $d_{\text{pred}} = \|\text{FK}(\text{PhysicalDecoder}(\mathbf{z}_{\text{pred}})_{:6}) - \mathbf{p}^*_{\text{final}}\|_2$
  - $d_{\text{obs}} = \|\text{FK}(\text{PhysicalDecoder}(\mathbf{z}_{\text{obs}})_{:6}) - \mathbf{p}^*_{\text{final}}\|_2$
  - $\theta_{\text{obs}} = \text{RotErr}(\text{FK}(\text{PhysicalDecoder}(\mathbf{z}_{\text{obs}})_{:6}), \mathbf{R}^*_{\text{final}})$
  - Handover weight $w_t = 0.85$ if ($d_{\text{pred}} > 0.026$, $d_{\text{obs}} < 0.025$, $\theta_{\text{obs}} < 0.049$); else $0.40 \cdot \text{clamp}((d_{\text{pred}} - 0.032)/0.04, 0, 1)$ if $d_{\text{obs}} < d_{\text{pred}}$ and $t \le t_c + 2$; else $0.0$.
  - **Verdict (PIL):** **Zero test-time state leakage.** Both $\mathbf{z}_{\text{pred}}$ and $\mathbf{z}_{\text{obs}}$ are strictly causal; only the task's two goal SE(3) poses (`target_grasp_mat`, `target_final_mat`) and transition step `t_c` are provided.
- **Non-PIL Handover (`benchmark_mppi_control.py` L344–L361, `mppi_scorer.py` L400–L412):**
  - Uses `z_gt_next = z_gt_seq[:, min(t + 1, max_t)]` pre-encoded from the **expert demonstration's future event frames and actions**, blending up to $w_t = 0.85$ of `z_gt_next` into `curr_z`.
  - **Verdict (Non-PIL):** **Demonstration-state leak** during $\phi_t = 1$. Must only be reported as an idealized latent-grounded upper bound alongside PIL.
- **Target Pose Construction (`benchmark_mppi_control.py` L225–L226 vs `mppi_scorer.py` L253–L264):**
  - `benchmark_mppi_control.py` constructs `target_grasp_mat` and `target_final_mat` directly from `fk.forward_homogeneous(q_gt[t_c])` and `fk.forward_homogeneous(q_gt[t_grasp])` (raw workspace SE(3) poses, NOT `PhysicalDecoder(z_gt)`).
  - In `mppi_scorer.py` (when `physics_in_loop=False` only), targets are projected through `PhysicalDecoder(z_gt)` (L260–L264); when `physics_in_loop=True`, raw `fk(q_gt)` is used (L253–L254).
  - Note: For `_BaselineModelWrapper` (`rssm`, `deep_gru`, `ann_rate`) in PIL mode (`benchmark_mppi_control.py` L311), `event_stream` is skipped (`curr_z = z_pred` open-loop), while `q_phys_next` is still stepped in `ArticulatedArmPhysicsStepper`.

### 1.2 Physics-in-the-Loop (PIL) Fidelity vs. Unity PhysX 4
| Dimension | Unity PhysX 4 (`sim/Assets/Scripts/*.cs`) | Python PIL (`sim_6dof.py`, `event_camera.py`) | Fidelity Gap Impact |
| :--- | :--- | :--- | :--- |
| **Joint Limits** | Asymmetric per-joint (`[-120°,120°]`, `[-90°,185°]`, `[-155°,15°]`, `[-95°,95°]`, `[-75°,75°]`, `[-180°,180°]`) | Symmetric `[-3.14, 3.14]` rad (`[-180°, 180°]`) on all 6 joints | Moderate (allows elbow/wrist angles outside Unity limits) |
| **Dynamics & Coupling** | Full 3D Featherstone `ArticulationBody` with skewed axes `(0,56,-16)` & off-diagonal $M(\mathbf{q}), C(\mathbf{q}, \dot{\mathbf{q}})$ | Decoupled diagonal `eff_inertia`, scalar `-eff_gravity * sin(q)`, no Coriolis coupling | Moderate (omits inter-link inertial coupling) |
| **Drive Gains & Limits** | Per-joint $K_p \in [150,320]$, $K_d \in [40,80]$, `MaxForceLimit` $\in [250,600]$ | Uniform `drive_stiffness=16.5`, `kp=18.0`, `kd=0.80`, `damping=0.15` | Moderate (normalized torque scale differs from physical Nm) |
| **FixedJoint Payload Lock** | Instant rigid `FixedJoint` at contact onset (`t_c`), full 3D offset inertia & ground friction (`0.90/0.80`) | `payload_attached=(t >= t_grasp)` in `benchmark_mppi_control.py` L309 (only active on final step `t_grasp`!) | **Direct test:** Fixing to `t >= t_c` yields **86.0% (43/50)** vs **88.0% (44/50)** (`-2.0 pp`, `0.86 ± 0.80 cm` min err) |
| **Event Camera Rendering** | 3D perspective `Camera` + `RenderTexture` + `EventCameraLogDiff.compute` with 3D occlusion & payload mesh | 2D Gaussian blobs (`render_articulated_frame`) + analytic trig wrist projection (`render_wrist_articulated_frame`), no 3D occlusion or payload mesh | High visual simplification (clean kinematics, zero occlusion clutter) |

### 1.3 Evaluation Horizon & Start-Step Alignment (`t_start` Discrepancy)
Reconciled the exact cause of the difference between `comparative_metrics_v0_12_4.json` and `metrics_spwm_v0_12_4.json`:
- **`comparative_metrics_v0_12_4.json` (`scripts/eval_table1_ssot.py`):** Evaluates all 5 models (`SPWM`, `ANN-Rate`, `Continuous RSSM`, `Deep GRU`, `Continuous ConvLSTM`) under **identical** `t_start = 4` (5 burn-in frames `0..4`), `horizon = 100`, pure open-loop (`phase_flags=None`), and reports **instantaneous** MSE at exact step `h`: `H=10` Pos MSE $= 0.00649\,\text{rad}^2$ (reflecting the contact/heft transient around `t=14`), settling to `H=100` Pos MSE $= 0.00246\,\text{rad}^2$.
- **`metrics_spwm_v0_12_4.json` (`spwm/evaluation/evaluator.py`):** Uses `compute_rollout_schedule` (`t_start = 2`), activates Active Perception sensory injection (`phase_flags == 1`), and reports **cumulative mean** MSE over `[1..h]`: `H=10` Pos MSE $= 0.00025\,\text{rad}^2$, `H=100` Pos MSE $= 0.00350\,\text{rad}^2$.

### 1.4 `ANN-Rate` 0.0% Closed-Loop Collapse Diagnosis
Measured step-by-step grounded estimation and $M_1$ rollout drift on test trajectories `[3600, 3609]` (`mean t_start = 16.9`):
- **Grounded State-Estimator Drift (`||q_dec(z*_t) - q_gt[t]||_2`):**
  - `SPWM`: `t=0: 0.0230`, `t=4: 0.0222`, `t=10: 0.0239`, `t=20: 0.0686`, `t=50: 0.0858`, `t=100: 0.1071 rad`
  - `ANN-Rate`: `t=0: 0.1020`, `t=4: 0.1671` (**7.5x worse**), `t=10: 0.1359`, `t=20: 0.2385` (**3.5x worse**), `t=50: 0.1522`, `t=100: 0.2008 rad`
- **$M_1$ Rollout Divergence from `t_start` (`h=1..20`):**
  - Latent drift `||z_hat - z*||_2`: `SPWM` `0.7541 -> 2.2288 -> 2.0325` (bounded) vs `ANN-Rate` `0.8987 -> 3.7803 -> 4.4605` (diverging).
  - Joint error `||q_hat - q_gt||_2`: `SPWM` `0.0564 -> 0.2252 -> 0.2122 rad` vs `ANN-Rate` `0.1412 -> 0.4415 -> 0.4978 rad`.
- **Root Cause:** Both severe grounded state-estimator error (`0.17–0.24 rad` prior to planning) and rapid $M_1$ rollout divergence (`0.50 rad` at `h=20`) place planned trajectories far outside the `3.0 cm` / `0.05 rad` success envelope.

### 1.5 Active vs. Dead Parameter Audit
| Model | Total Checkpoint Params | Active Inference Params | Inactive Params | Recurrent MACs / step |
| :--- | :---: | :---: | :---: | :---: |
| **SPWM-v0.12.4** | 392,141 | **317,900** | 74,241 (`critic.*`, `W_vel.weight`) | 67,072 (8,229 SOPs @ 12.27% spike rate) |
| **ANN-Rate** | 392,141 | **317,900** | 74,241 (`critic.*`, `W_vel.weight`) | 67,072 |
| **Continuous RSSM** | 380,460 | **380,460** | 0 | 365,056 |
| **Deep GRU** | 338,868 | **338,868** | 0 | 299,520 |
| **Continuous ConvLSTM** | 296,892 | **296,892** | 0 | 2,359,296 |

SPWM has **fewer** active inference parameters than `Continuous RSSM` (`-62.6k`) and `Deep GRU` (`-21.0k`), and exact 1:1 parity with `ANN-Rate`.

---

## Phase 2: Complete 50-Episode PIL & Robustness Benchmarks

### 2.1 Complete 50-Episode PIL Model & Ablation Matrix (`[3600, 3649]`)
| Model / Ablation Stage | Active (Total) Params | Latent-Grounded Succ | PIL Succ (50 ep) | PIL Min Err (cm) | PIL Term Err (cm) | PIL `|Δ²q|` (rad/step²) | PIL Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. SPWM-Base (Phase 1 Warmup, Unsmoothed)** | 317.9k (392.1k) | 70.0% | **62.0%** | 1.27 ± 1.14 | 2.50 ± 1.55 | 0.00713 ± 0.00171 | 24.2 |
| **2. + Smooth MPPI Controls** | 317.9k (392.1k) | 70.0% | **62.0%** | 1.09 ± 0.97 | 2.21 ± 1.69 | 0.00726 ± 0.00191 | 28.1 |
| **3. + Phase 1 Curriculum (AR(1)/DAgger)** | 317.9k (392.1k) | 76.0% | **74.0%** | 0.99 ± 0.97 | 1.82 ± 1.27 | 0.00597 ± 0.00166 | 27.5 |
| **4. + Residual Handover & Lookahead (SPWM)** | 317.9k (392.1k) | **96.0%** | **88.0%** *(86.0% @ `t>=t_c`)* | **0.88 ± 0.81** | **1.60 ± 1.00** | **0.00581 ± 0.00208** | **22.1** |
| **Continuous RSSM** | 380.5k | 46.0% | 48.0% | 1.32 ± 0.93 | 2.22 ± 1.33 | 0.01293 | 31.1 |
| **Deep GRU** | 338.9k | 14.0% | 0.0% | 12.67 | 29.83 | 0.00415 | 44.5 |
| **ANN-Rate** | 317.9k (392.1k) | 0.0% | 0.0% | 7.36 | 16.44 | 0.00462 | 30.4 |
| **Fixed-Gain Jacobian-IK + PD** *(Privileged Oracle `q_phys`, no vision)* | N/A | N/A | 100.0% | 0.012 ± 0.031 | 0.083 ± 0.101 | 0.00794 ± 0.00422 | 2.6 |
| **Reactive BC (EventEncoder + GRU)** | 322.2k | N/A | **0.0%** | 11.99 ± 6.38 | 14.39 ± 7.95 | 0.00208 ± 0.00080 | 0.96 |

### 2.2 Full 50-Episode Robustness Stress Test (`[3600, 3649]`)
| Perturbation Condition | Latent-Grounded Succ (50 ep) | Latent-Grounded Min Err | PIL Succ (50 ep) | PIL Min Err | PIL Term Err |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | **96.0%** (48/50) | 0.67 ± 0.46 cm | **88.0%** (44/50) | 0.88 ± 0.81 cm | 1.60 cm |
| **Sensory Noise Injection (1.0%)** | **82.0%** (41/50) | 1.06 ± 1.23 cm | **72.0%** (36/50) | 1.04 ± 0.87 cm | 1.89 cm |
| **Payload Extrapolation (4.80 kg)** | **74.0%** (37/50) | 1.57 ± 1.91 cm | **78.0%** (39/50) | 1.27 ± 1.12 cm | 1.80 cm |
| **Actuator Drift (0.80x Friction)** | **70.0%** (35/50) | 1.61 ± 2.07 cm | **70.0%** (35/50) | 1.35 ± 1.26 cm | 2.09 cm |
| **Actuator Drift (1.20x Friction)** | **80.0%** (40/50) | 1.53 ± 1.91 cm | **76.0%** (38/50) | 1.34 ± 1.23 cm | 1.98 cm |

### 2.3 Multi-Seed Statistical Rigor (`seeds = 42, 101, 2024`)

**Important Training-Budget Note:** `SPWM` Seed `42` (`results/spwm_v0_12_4_train_m1/model.pt`) was trained for the full production schedule (`160` Phase 0 + `240` Phase 1 epochs across all 4 Latch/DAgger curriculum stages). Under `train_multiseed.py`'s `30+30` epoch budget, `SPWM` Seeds `101` and `2024` only reached Curriculum Phase 1/4 (`DAgger = 0%`, `Noise alpha = 0.0`). `Continuous RSSM` was trained for the full `18` epochs on all 3 seeds.

| Model & Seed | Training Budget | H=100 Pos MSE (RolloutEval / Open-Loop) | H=100 Vel MSE | PIL Succ (50 ep) | PIL Min Err (cm) | PIL `|Δ²q|` (rad/step²) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPWM (Seed 42, Prod)** | 160 + 240 ep (4/4) | `0.00350` / `0.00246` | `0.01190` *(open)* | **88.0%** | 0.88 ± 0.81 | 0.00581 ± 0.00208 |
| **SPWM (Seed 101, 30+30)** | 30 + 30 ep (1/4) | `0.00716` | `0.03891` | 4.0% | 3.23 ± 1.58 | 0.00673 ± 0.00164 |
| **SPWM (Seed 2024, 30+30)** | 30 + 30 ep (1/4) | `0.00790` | `0.04574` | 24.0% | 2.14 ± 1.97 | 0.00636 ± 0.00102 |
| **SPWM Mean ± Std (N=3, mixed budget)** | — | `0.00619 ± 0.00235` | — | `38.7 ± 43.8%` | `2.08 ± 1.18` | `0.00630 ± 0.00046` |
| **Continuous RSSM (Seed 42)** | 18 ep | `0.00665` | `0.04421` | **48.0%** | 1.32 ± 0.93 | 0.01293 |
| **Continuous RSSM (Seed 101)** | 18 ep | `0.00339` | `0.03707` | **46.0%** | 1.35 ± 0.70 | 0.01242 ± 0.00331 |
| **Continuous RSSM (Seed 2024)** | 18 ep | `0.00347` | `0.03550` | **20.0%** | 1.26 ± 0.77 | 0.01261 ± 0.00332 |
| **RSSM Mean ± Std (N=3, full 18 ep)** | — | `0.00450 ± 0.00186` | `0.03893 ± 0.00464` | **38.0 ± 15.6%** | `1.31 ± 0.05` | `0.01265 ± 0.00026` |

**Key Empirical Finding:** Even at `30+30` epochs, `SPWM` maintains bounded `H=100` open-loop error (`0.00716–0.00790 rad²`) and `2x` smoother closed-loop acceleration (`0.00636–0.00673` vs RSSM `0.01265`), but **full Phase 0 convergence (`160` ep) and Phases 2–4 of the Latch/DAgger/AR(1) curriculum (`240` ep)** are required to reach `88.0%` PIL grasp success. Meanwhile, `Continuous RSSM` exhibits substantial seed-to-seed PIL variance (`20.0%–48.0%`, `38.0 ± 15.6%`) even at full `18`-epoch convergence.

---

## Publication-Grade Figures
- **Contact Heft Shock Time-Series:** [`figures/heft_shock_timeseries.pdf`](file:///Users/Mattia/Desktop/Studies/Temp/figures/heft_shock_timeseries.pdf) ([PNG](file:///Users/Mattia/Desktop/Studies/Temp/figures/heft_shock_timeseries.png)) — closed-loop PIL trajectory comparison on `traj 3622` (`2.45 kg ElongatedCylinder`) showing joint tracking, Cartesian error through `t_c = 38`, velocity norm + handover gate $w_t$, and two-tier ALIF spike activity.
