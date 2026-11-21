# 03 — Benchmarks, Verified Evidence & Replication Runbook

All closed-loop control metrics below were executed under causal **Physics-in-the-Loop (PIL)** (`ArticulatedArmPhysicsStepper`, rigid palm-latch `payload_attached = (t >= t_c)` where $t_c = \texttt{p1\_indices[0]} = t_{\text{contact}} - 7$ is the $\phi_t=1$ sensory-window onset and $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$, causal `OnlineEventCameraStream`, zero `z_gt_next` leakage) on `50` unseen test episodes (`[3600, 3649]`) with `K=256, H=10, N_iter=3`, slew clamp `0.10`. An episode counts as a grasp success iff end-effector error stays `< 3.0 cm` across the final 3-step settling window, terminal $\text{SO}(3)$ rotation error `< 0.05 rad`, and terminal joint velocity $\|\dot{\mathbf{q}}(t_{\text{grasp}})\|_2 < \tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)\,\text{rad/s}$.

---

## 0. Checkpoint Provenance & Equalized Training Budgets

`model.pt` and `best_model.pt` in `results/spwm_v0_12_4_train_m1/` are distinct files on disk (`sha256` prefixes `d358cccbd678` vs `f5cabfca3635` due to non-weight metadata), **but both store identical Epoch 200 EMA `model_state` tensors (`0` weight diffs)**; SPWM, ANN-Rate, and equalized baseline checkpoints are documented below:

| Model / Seed | Checkpoint File | Epochs (Best) | Wall-Clock (`MPS`) | Primary Consumers |
| :--- | :--- | :---: | :---: | :--- |
| `SPWM` (`42`) | [results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt) | `160+240` (`240`) | `10.40 h` | Table 2 (`N=25` Val-Sel.), Table 3 (Stages 3–4b), `forensic` |
| `SPWM` (`42`) | [results/spwm_v0_12_4_train_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/model.pt) / `best_model.pt` | `160+240` (`200`) | `10.40 h` | Table 1, Table 2 (`N=6`), Table 4, Table 5 |
| `SPWM` (`42`) | `checkpoint_phase_1.pt` | `181` | — | Table 3 (Stages 1–2) |
| `SPWM` (`101`) | [results/multiseed/seed_101_m1/checkpoint_phase_1.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_m1/checkpoint_phase_1.pt) | `160+240` (`96`) | `5.86 h` | Table 2 (`N=25` Val-Sel.), Table 3 (Stages 1, 2, 4a, 4b), `forensic` |
| `SPWM` (`101`) | [results/multiseed/seed_101_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_m1/model.pt) / `checkpoint_phase_4.pt` | `80` / `240` | `5.86 h` | Table 1, Table 2 (`N=6`), Table 3 (Stage 3), Tables 4–5 |
| `SPWM` (`2024`) | [results/multiseed/seed_2024_m1/checkpoint_phase_2.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_m1/checkpoint_phase_2.pt) | `160+260` (`102`) | `6.41 h` | Table 2 (`N=25` Val-Sel.), Table 3 (Stages 4a, 4b), `forensic` |
| `SPWM` (`2024`) | [results/multiseed/seed_2024_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_m1/model.pt) / `checkpoint_phase_1.pt` / `phase_4.pt` | `130` / `94` / `260` | `6.41 h` | Table 1, Table 2 (`N=6`), Table 3 (Stages 1–3), Tables 4–5 |
| `ANN-Rate` (`42/101/2024`) | `results/ann_ablation/seed_{42,101,2024}_m1/model.pt` | `160+240` | — | Tables 1, 2, 4, 5, `forensic` |
| `Continuous RSSM` (`42/101/2024`) | `results/baselines/continuous_rssm/model.pt`, `results/multiseed_rssm/seed_{101,2024}/continuous_rssm/model.pt` *(60-ep backups in `model_pre_dagger_60ep.pt`)* | `60+16=76` (`61`), `60+16=76` (`61`), `60+17=77` (`62`) | `46.0+13.0 m`, `44.0+12.9 m`, `44.0+13.8 m` | Tables 1, 2 (Open-Loop, Un-Gated, Residual-Gated), `forensic` |
| `Deep GRU` (`42/101/2024`) | `results/baselines/deep_gru/model.pt`, `results/multiseed_rssm/seed_{101,2024}/deep_gru/model.pt` *(backups in `model_pre_dagger_60ep.pt` / `model_pre_equalized_18ep.pt`)* | `59+40=99` (`90`), `80` (`68`), `80` (`80`) | `46.6+33.8 m`, `67.1 m`, `66.8 m` | Tables 1, 2 (Open-Loop, Un-Gated, Residual-Gated), `forensic` |
| `Continuous ConvLSTM` (`42`) | [results/baselines/continuous_convlstm/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/continuous_convlstm/model.pt) | `60` (`54`) | `86.9 m` | Table 1, Table 5 |

---

## 1. Table 1 — Open-Loop Rollout ($N=400$ Test `[3600, 3999]`, $t_{\text{start}}=4$)

**Evidence:** [comparative_metrics_v0_12_4.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json), [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json), [training_meta.json (GRU 42)](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/deep_gru/training_meta.json), [training_meta.json (GRU 101)](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed_rssm/seed_101/deep_gru/training_meta.json), [training_meta.json (GRU 2024)](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed_rssm/seed_2024/deep_gru/training_meta.json).

| Model Architecture | Params (Act.) | Pos H=1 | Pos H=10 | Pos H=50 | Pos H=100 | Vel H=1 | Vel H=10 | Vel H=50 | Vel H=100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Continuous ConvLSTM (Seed 42) | `296.9k` | **`0.00002`** | **`0.00012`** | **`0.00150`** | `0.00707` | `0.00147` | **`0.00553`** | **`0.01095`** | `0.33696` |
| Deep GRU (Equalized, 3 seeds) | `338.9k` | `0.00003 ± 0.00001` | `0.00147 ± 0.00044` | `0.00195 ± 0.00003` | `0.00469 ± 0.00048` | `0.00203 ± 0.00019` | `0.01852 ± 0.00237` | `0.01332 ± 0.00024` | `0.08716 ± 0.01445` |
| Continuous RSSM (Equalized, 3 seeds) | `380.5k` | `0.00007 ± 0.00004` | `0.00033 ± 0.00006` | `0.00174 ± 0.00004` | **`0.00152 ± 0.00014`** | `0.00153 ± 0.00186` | `0.01660 ± 0.00201` | `0.02256 ± 0.00088` | `0.02130 ± 0.00082` |
| ANN-Rate (Leaky, 3 seeds) | `392.1k (317.9k)` | `0.00028 ± 0.00012` | `0.00249 ± 0.00032` | `0.00341 ± 0.00264` | `0.00210 ± 0.00063` | `0.00193 ± 0.00069` | `0.03356 ± 0.00511` | `0.02440 ± 0.01698` | `0.01183 ± 0.00715` |
| **SPWM (3 seeds)** | **`392.1k (317.9k)`** | `0.00018 ± 0.00011` | `0.00348 ± 0.00265` | `0.00315 ± 0.00144` | `0.00240 ± 0.00037` | **`0.00139 ± 0.00076`** | `0.04507 ± 0.02601` | `0.02466 ± 0.01271` | **`0.00971 ± 0.00177`** |

* **Open-Loop vs. Unconditional Phasic Handover L2 Error (`SPWM` Seed 42, $H=100$, `12.27%` ALIF spike rate; [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json)):** Pure Open-Loop mean `0.1115 rad` vs. Unconditional Phasic Handover `0.1482 rad` (`h=10`: `0.1862 / 0.1865`, `h=50`: `0.1038 / 0.1496`, `h=80`: `0.0899 / 0.1827`, `h=100`: `0.1116 / 0.1420`).
* **Replication:** `.venv/bin/python scripts/eval_table1_ssot.py`

---

## 2. Table 2 — Closed-Loop PIL Benchmark (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$, 3-Regime Comparison)

**Evidence:** [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json) (`corl_final_verification.pil_3seed_comparison`), [corl_final_benchmark_summary.md](file:///Users/Mattia/Desktop/Studies/Temp/results/corl_final_benchmark_summary.md), [classical_bc_pil_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json).

| Method (`50` Test Episodes `[3600, 3649]`) | PIL Succ (%) | Min Err (cm) | Term Err (cm) | $\|\Delta^2 q\|$ | $\|\Delta^3 q\|$ | Flips | Latency (ms) | Source JSON(s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Privileged State-Oracle Jacobian-IK + PD ($\mathbf{q}_{\text{phys}}$, no vision, 2.85 kg cal.)* | *96.0* | *0.01 ± 0.03* | *0.43 ± 0.73* | *0.00771* | *0.00497* | *10.46* | *22.7* | [classical_bc_pil_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json) |
| Reactive BC (`322,210` params, Seed 42) | `0.0` | `13.13 ± 6.27` | `15.66 ± 8.15` | `0.00140` | `0.00135` | `13.98` | `1.9` | [classical_bc_pil_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json) |
| Continuous RSSM (Open-Loop PIL, $w_t=0$, 3 seeds) | `15.3 ± 7.0` | `1.23 ± 0.06` | `2.74 ± 0.07` | `0.02037` | `0.02139` | `36.86` | `46.8` | `rssm_open_loop/`, `seed_{101,2024}/mppi_pil_open_loop/` (`16% / 8% / 22%`) |
| Continuous RSSM (Un-Gated Online, $w_t=1.0$, 3 seeds) | `0.0 ± 0.0` | `1.43 ± 0.03` | `25.49 ± 1.31` | `0.03153` | `0.02956` | `36.49` | `46.9` | `rssm_ungated/`, `seed_{101,2024}/mppi_pil_ungated/` (`0% / 0% / 0%`) |
| Continuous RSSM (Residual-Gated Online, 3 seeds) | `15.3 ± 5.0` | `1.23 ± 0.05` | `2.72 ± 0.11` | `0.02049` | `0.02161` | `36.94` | `46.5` | `rssm_residual_gated/`, `seed_{101,2024}/mppi_pil_residual_gated/` (`16% / 10% / 20%`) |
| Deep GRU (Open-Loop PIL, $w_t=0$, 3 seeds) | `77.3 ± 4.2` | `1.14 ± 0.31` | `1.90 ± 0.18` | `0.00450` | `0.00268` | `10.79` | `57.8` | `deep_gru_open_loop/`, `seed_{101,2024}/deep_gru_mppi_pil_open_loop/` (`82% / 74% / 76%`) |
| Deep GRU (Un-Gated Online, $w_t=1.0$, 3 seeds) | `0.0 ± 0.0` | `1.47 ± 0.51` | `7.35 ± 1.34` | `0.00698` | `0.00483` | `15.08` | `57.1` | `deep_gru_ungated/`, `seed_{101,2024}/deep_gru_mppi_pil_ungated/` (`0% / 0% / 0%`) |
| Deep GRU (Residual-Gated Online, 3 seeds) | `76.7 ± 5.0` | `1.14 ± 0.31` | `1.90 ± 0.18` | `0.00449` | `0.00267` | `10.79` | `57.7` | `deep_gru_residual_gated/`, `seed_{101,2024}/deep_gru_mppi_pil_residual_gated/` (`82% / 72% / 76%`) |
| ANN-Rate (Leaky Rate, 3 seeds) | `37.3 ± 9.5` | `1.07 ± 0.28` | `2.98 ± 0.46` | `0.01148` | `0.00938` | `26.00` | `24.6` | `ann_ablation/seed_{42,101,2024}_mppi_pil/` (`34% / 30% / 48%`) |
| **SPWM ($N=25$ Val-Sel., 3 seeds)** | **`60.7 ± 2.3`** | **`0.88 ± 0.11`** | **`2.01 ± 0.12`** | **`0.00931`** | **`0.00663`** | **`21.14`** | **`26.6`** | `ablations/multiseed/stage4b_full_seed{42,101,2024}/` (`58% / 62% / 62%`) |
| SPWM ($N=6$ Tier-2, 3 seeds) | `58.0 ± 27.1` | `0.90 ± 0.04` | `2.23 ± 0.68` | `0.00860` | `0.00608` | `19.37` | `27.5` | `mppi_benchmark/`, `multiseed/seed_{101,2024}_mppi_pil/` (`86% / 56% / 32%`) |

* **Key Baseline Equalization & Trade-Off Takeaways:**
  1. Applying the identical causal FK residual gate eliminates the `0.0%` online collapse caused by un-gated domain-gapped 2D-vs-3D `OnlineEventCameraStream` frames (`w_t = 1.0`). Under residual gating, `Continuous RSSM` remains capped at `15.3 ± 5.0%` (`|Δ²q| = 0.02049`, `36.94` flips), confirming its failure is architectural.
  2. Once equalized across all 3 seeds (`99 / 80 / 80` epochs with AR(1)+DAgger) and gated identically, `Deep GRU` achieves `76.7 ± 5.0%` PIL success (`77.3 ± 4.2%` open-loop)—exceeding `SPWM` (`60.7 ± 2.3%`) in nominal grasp rate and smoothness (`|Δ²q| = 0.00449`), **but** violates the `50.0 ms` (`20 Hz`) real-time budget (`57.7 ms/step` vs `SPWM` `26.6 ms/step`), suffers `9.0×` higher $H=100$ open-loop velocity MSE (`0.08716` vs `0.00971 rad²/s²`), and consumes `22.1×` higher standby core energy / `1.59×` higher E2E step energy.
  3. Compared against parameter-matched `ANN-Rate` (`37.3 ± 9.5%`), `SPWM` retains a **`+23.4 pp`** 3-seed PIL gain (`60.7%` vs `37.3%`) isolating discrete ALIF spike thresholding.

---

## 3. Table 3 — Full 3-Seed 5-Stage SPWM PIL Ablation (`[3600, 3649]`, Seeds `42, 101, 2024`)

**Evidence:** [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json) (`corl_final_verification.disentangled_ablation_3seed`), `results/ablations/multiseed/stage*_seed{42,101,2024}/mppi_benchmark_metrics.json`.

| Ablation Stage (Seeds `42 / 101 / 2024`) | Per-Seed Succ (%) | 3-Seed PIL Succ (%) | Min Err (cm) | Term Err (cm) | $\|\Delta^2 q\|$ | $\|\Delta^3 q\|$ | Flips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. SPWM-Base (`checkpoint_phase_1.pt`, Unsmoothed) | `62 / 44 / 30` | `45.3 ± 16.0` | `1.03 ± 0.21` | `2.50 ± 0.11` | `0.00983` | `0.00765` | `23.03` |
| 2. + Smooth MPPI Controls (`checkpoint_phase_1.pt`) | `62 / 44 / 48` | `51.3 ± 9.5` | `1.02 ± 0.08` | `2.41 ± 0.13` | `0.00991` | `0.00705` | `21.99` |
| 3. + Phase 3/4 Curriculum (AR(1)+DAgger, `checkpoint_phase_4.pt`, No Lookahead/Handover) | `56 / 18 / 40` | `38.0 ± 19.1` | `1.53 ± 0.59` | `3.27 ± 1.13` | `0.00589` | `0.00354` | `11.17` |
| 4a. + 2-Stage SE(3) Lookahead ($N=25$ Val-Sel., $w_t \equiv 0$) | `58 / 62 / 62` | `60.7 ± 2.3` | `0.88 ± 0.11` | `2.01 ± 0.12` | `0.00931` | `0.00663` | `21.14` |
| **4b. + Residual Rejection Gate (Full SPWM, $N=25$ Val-Sel.)** | **`58 / 62 / 62`** | **`60.7 ± 2.3`** | **`0.88 ± 0.11`** | **`2.01 ± 0.12`** | **`0.00931`** | **`0.00663`** | **`21.14`** |

* **Transparent Stage 2 vs. Stage 3 DAgger Diagnostic:** Comparing Stage 2 (`51.3 ± 9.5%`) vs. Stage 3 (`38.0 ± 19.1%`) across all 3 seeds shows that Phase 3/4 AR(1)+DAgger replay cuts trajectory jitter (`|Δ²q|: 0.00991 -> 0.00589`, flips `21.99 -> 11.17`) **while degrading 3-seed mean PIL success by `-13.3 pp` and doubling inter-seed std (`9.5% -> 19.1%`)**. While DAgger improves validation peak on individual seeds (Seed 42 `phase_4`: `92.0%` val), it introduces multi-seed overfitting variance (`18.0%` test on Seed 101 `phase_4`), requiring **`N=25` validation-split (`[3200, 3224]`) checkpoint selection** (`42: phase_4`, `101: phase_1`, `2024: phase_2`) so 2-Stage SE(3) Lookahead delivers `60.7 ± 2.3%` (`11.7×` std reduction vs. `N=6` Tier-2).

---

## 4. Table 4 — 3-Seed PIL Robustness & Residual Handover Telemetry

**Evidence:** [robustness_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark/robustness_metrics.json), `results/robustness_benchmark_pil_spwm_seed{101,2024}/robustness_metrics.json`, `results/robustness_benchmark_pil_ann_rate_seed{42,101,2024}/robustness_metrics.json` ($N=6$ Tier-2 frozen checkpoints).

| Perturbation Condition (`50` Test Ep) | SPWM 3-Seed Mean (Succ / Min / Term) | SPWM Step $w_t>0$ (42 / 101 / 2024) | ANN-Rate 3-Seed Mean (Succ / Min / Term) |
| :--- | :---: | :---: | :---: |
| Nominal Baseline | **`58.0 ± 27.1%`** / `0.90` / `2.23 cm` | `0.0% / 0.0% / 0.0%` | `37.3 ± 9.5%` / `1.07` / `2.98 cm` |
| Sensory Noise (`1.0%`) | **`55.3 ± 19.4%`** (`-4.6% rel`) / `0.90` / `2.34 cm` | `0.0% / 0.0% / 0.0%` | `30.0 ± 3.5%` (`-19.6% rel`) / `1.04` / `3.36 cm` |
| Payload Extrapolation (`4.80 kg`) | **`39.3 ± 33.7%`** (`-32.2% rel`) / `1.18` / `2.50 cm` | `0.0% / 0.0% / 0.5%` (`4.0% ep, w=0.165`) | `26.7 ± 16.2%` (`-28.6% rel`) / `1.36` / `3.64 cm` |
| Actuator Drift (`0.80x` Friction) | **`40.7 ± 27.3%`** (`-29.9% rel`) / `1.19` / `2.61 cm` | `0.0% / 0.0% / 0.5%` (`4.0% ep, w=0.229`) | `30.7 ± 14.0%` (`-17.9% rel`) / `1.42` / `3.74 cm` |
| Actuator Drift (`1.20x` Friction) | **`44.0 ± 28.0%`** (`-24.1% rel`) / `1.22` / `2.53 cm` | `0.0% / 0.0% / 0.5%` (`4.0% ep, w=0.169`) | `33.3 ± 16.7%` (`-10.7% rel`) / `1.34` / `3.51 cm` |

* **Validation Handover Sweep ($\tau \in \{0.026, 0.020, 0.015\}\,\text{m}$, `[3200, 3249]`):** Across all 6 JSONs (`results/diagnostics/val_handover_sweep_{seed42,seed2024}_tau_{0026,0020,0015}/robustness_metrics.json`), **all 3 thresholds produce identical results with `0.0%` handovers** (`Seed 42`: `78.0 / 78.0 / 66.0 / 70.0 / 72.0%`; `Seed 2024`: `46.0 / 36.0 / 42.0 / 38.0 / 38.0%`).
* **Diagnostic Proof (`val` `[3200, 3214]`, `120` $\phi_t=1$ steps):** `Seed 42`: `dist_pred` median `0.0139 m`, `dist_obs` median `0.2155 m`, `rot_obs` median `0.6377 rad`, `obs < pred = 0.0%`; `Seed 2024`: `dist_pred` median `0.0174 m`, `dist_obs` median `0.1598 m`, `rot_obs` median `0.3850 rad`, `obs < pred = 0.0%`.

---

## 5. Table 5 — Standby Neuromorphic Core vs. End-to-End Closed-Loop Energy

**Evidence:** [energy_profiling_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/energy_profiling/energy_profiling_metrics.json) (`12.27%`, `8,229` SOPs, `12.34 nJ`, `4.89x`), [seed_101](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_energy/energy_profiling_metrics.json) (`12.26%`, `8,220` SOPs, `12.33 nJ`, `4.90x`), [seed_2024](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_energy/energy_profiling_metrics.json) (`11.93%`, `8,001` SOPs, `12.00 nJ`, `5.03x`), [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json).
*(Assumptions: Loihi-2 `1.5 pJ/SOP`, Jetson Orin GPU `0.9 pJ/MAC`, shared `EventEncoder` `882,688` MACs = `794.4 nJ`, $K=256 \times H=10 \times N_{\text{iter}}=3 = 7,680$ batched MPPI rollouts/step.)*

| Architecture | Core Ops / Step | Core Energy (nJ) | Core Power @ 20 Hz | Core vs. GPU | MPPI MACs | E2E Energy (mJ) | E2E Power @ 20 Hz |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPWM (Loihi-2 Standby Core + Orin, 3 seeds)** | **`8,150 ± 129 SOPs`** | **`12.22 ± 0.19`** | **`0.244 ± 0.004 mW`** | **`4.94 ± 0.08×`** | `1.219 G` | `1.0979` | `21.96 mW` |
| ANN-Rate (Orin GPU) | `67,072 MACs` | `60.36` | `1.207 mW` | `1.00×` | `1.219 G` | `1.0979` | `21.96 mW` |
| Deep GRU (Orin GPU) | `299,520 MACs` | `269.57` | `5.391 mW` | `1.00×` | `1.942 G` | `1.7490` | `34.98 mW` |
| Continuous RSSM (Orin GPU) | `365,056 MACs` | `328.55` | `6.571 mW` | `1.00×` | `2.358 G` | `2.1236` | `42.47 mW` |
| Continuous ConvLSTM (Orin GPU) | `2,359,296 MACs` | `2123.37` | `42.467 mW` | `1.00×` | `111.312 G` | `100.1833` | `2003.67 mW` |

* **Recalibrated Energy Framing:**
  1. `SPWM`'s primary advantage over `ANN-Rate` is **closed-loop dynamical stability and noise robustness** (`+23.4 pp` 3-seed PIL grasp success, `4.3×` better relative retention under `1.0%` sensory noise), **not** end-to-end energy savings: saving `48.14 nJ/step` on Loihi-2 (`60.36 -> 12.22 nJ/step`) atop `1.0971 mJ/step` (`1,097,100 nJ/step`) of dense GPU MPPI rollouts is a **`0.004%`** end-to-end difference (`1.0979 mJ/step`, `21.96 mW` @ `20 Hz` for both).
  2. The **`4.94×`** Loihi-2 recurrent core reduction (`12.22 ± 0.19 nJ/step`, `0.244 mW` @ `20 Hz`, computed from synaptic SOPs excluding passive membrane leak) is specifically an **always-on, sub-milliwatt standby state-estimation benefit** when batched MPPI planning is dormant.
  3. The **`1.93×`** E2E reduction vs. `Continuous RSSM` (`2.1236 mJ/step`) and `1.59×` reduction vs. `Deep GRU` (`1.7490 mJ/step`) come from **`M1`'s compact MLP rollout architecture** (`1.219 GMACs` vs. `2.358` and `1.942 GMACs`).
* **Analytical SOP & MAC Derivation ([profile_energy_sops.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/profile_energy_sops.py#L67-L146,L354-L388)):** `spwm_core_macs = (128+6)*256 + 256*128 = 67,072` MACs; multiplying by empirical ALIF spike rates (`12.27%`, `12.26%`, `11.93%`) yields `8,229`, `8,220`, and `8,001` SOPs/step (`1.5e-3 nJ/SOP`, passive membrane leak excluded).

---

## 6. Forensic Root-Cause Diagnostics & Validation Selection

### 6.1 Equalized 3-Seed Forensic Decomposition
**Evidence:** [forensic_rssm_gru_failure.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/forensic_rssm_gru_failure.py) $\to$ [forensic_rssm_gru_failure.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/forensic_rssm_gru_failure.json) (`150` episodes = `3 seeds × 50` per model):
* **Continuous RSSM Un-Gated (`0/150 = 0.0%`):** Reaches $\le 3\,\text{cm}$ on `144/150 (96.0%)`, then suffers `150/150` velocity violations and `143/150` post-$t_c$ `div+vel` (`2.75 ± 0.06 -> 25.49 ± 1.31 cm`, `7.09 ± 0.17 rad/s`).
* **Continuous RSSM Residual-Gated (`23/150 = 15.3 ± 5.0%`):** Rejecting corrupted bursts cuts velocity violations to `24/150` and terminal error to `2.72 ± 0.11 cm` (`0.90 ± 0.11 rad/s`), **yet 127/150 episodes still fail** (`38/150` post-$t_c$ spatial divergence + `75/150` settling/vel/rot fails + `12/150` `div+vel`) due to high-frequency MPPI jitter (`|Δ²q| = 0.02049`, `36.94` flips).
* **Deep GRU Un-Gated (`0/150 = 0.0%`):** Reaches $\le 3\,\text{cm}$ on `132/150 (88.0%)`, then collapses via `106/150` pure post-$t_c$ spatial divergences (`2.75 ± 0.45 -> 7.35 ± 1.34 cm`).
* **Deep GRU Residual-Gated (`115/150 = 76.7 ± 5.0%`):** `141/150` reach, `0/150` velocity violations, `0/150` `div+vel`, `9/150` post-$t_c$ spatial divergences (`2.75 ± 0.45 -> 1.90 ± 0.18 cm`, `0.30 ± 0.08 rad/s`), but runs at `57.7 ms/step` (`> 50 ms`).
* **ANN-Rate (`56/150 = 37.3 ± 9.5%`):** `146/150` reach, `2/150` velocity violations, `56/150` post-$t_c$ spatial divergences (`2.69 ± 0.21 -> 2.98 ± 0.46 cm`, `0.50 ± 0.11 rad/s`).
* **SPWM $N=25$ Val-Selected (`91/150 = 60.7 ± 2.3%`):** `148/150 (98.7%)` reach, `2/150 (1.3%)` velocity violations, `1/150` `div+vel`, `27/150` post-$t_c$ spatial divergences (`2.05 ± 0.34 -> 2.01 ± 0.12 cm`, `0.45 ± 0.07 rad/s`, `26.6 ms/step`).

### 6.2 SPWM `N=6` Tier-2 Variance Audit & $N=25$ Validation Selection
**Evidence:** [audit_spwm_seed_variance.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/audit_spwm_seed_variance.py) $\to$ [spwm_seed_variance_audit.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/spwm_seed_variance_audit.json) + `val25_checkpoint_selection` in [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json):
* **Internal Module Health Across All 3 Seeds:** `action_proj` spectral norm $\|W_u\|_2$ (`0.8594`, `1.0131`, `0.8189`), `gate_scale` (`~49.04–49.07`), ALIF mean spike rate (`12.27%`, `12.26%`, `11.93%`), silent fraction (`0.78%`, `0.78%`, `2.34%`), and hyperactive fraction (`0.0%`) are cleanly converged across all seeds.
* **$N=25$ Validation Selection (`[3200, 3224]` $\to$ `[3600, 3649]`):**
  * `Seed 42`: `best_model` (`200`): `80.0%`, `phase_1` (`181`): `60.0%`, `phase_2` (`188`): `52.0%`, `phase_3` (`200`): `88.0%`, **`phase_4` (`240`): `92.0%`** $\to$ Test **`58.0%`**.
  * `Seed 101`: `best_model` (`80`): `56.0%`, **`phase_1` (`96`): `68.0%`**, `phase_2` (`104`): `52.0%`, `phase_3` (`124`): `4.0%`, `phase_4` (`240`): `40.0%` $\to$ Test **`62.0%`**.
  * `Seed 2024`: `best_model` (`130`): `48.0%`, `phase_1` (`94`): `64.0%`, **`phase_2` (`102`): `80.0%`**, `phase_3` (`122`): `40.0%`, `phase_4` (`260`): `60.0%` $\to$ Test **`62.0%`**.
  * **Aggregate ($N=25$ Val-Selected):** **`60.7 ± 2.3%`** (`11.7×` reduction in inter-seed std vs. `N=6` Tier-2 `58.0 ± 27.1%`).

### 6.3 Standalone Phase 0 Representation Metrics (`results/spwm_v0_12_4_train_representation/`)
* **Training & Validation ([metrics_spwm_v0_12_4.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/metrics_spwm_v0_12_4.json)):** Seed 42 best checkpoint at **Ep 159** reaches `val_loss = 1.281731` (`-98.01%` from Ep 1 `65.4230`), `val_pos_err = 0.07430 rad`, `val_vel_err = 0.16678 rad/s`, `val_loss_cartesian = 0.002454` (Seed 101 Ep 160: `1.3251`; Seed 2024 Ep 146: `1.4362`).
* **Grounded FK Accuracy & Homeostasis Ablation ([diagnostics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/diagnostics.json)):** 50-episode grounded `forward_representation` at `t_grasp` achieves **`2.04 cm`** FK position MAE (`p90 = 3.33 cm`) and **`0.043 rad`** rotation error (`p90 = 0.082 rad`).

---

## 7. Master Verification & SSOT Aggregation Commands

```bash
# 1. Re-evaluate Table 1 (N=400 open-loop rollout across all seeds)
.venv/bin/python scripts/eval_table1_ssot.py

# 2. Regenerate forensic diagnostics & SSOT aggregation JSONs + summary
.venv/bin/python scripts/diagnostics/forensic_rssm_gru_failure.py
.venv/bin/python scripts/aggregate_corl_final_verification.py
.venv/bin/python scripts/diagnostics/audit_spwm_seed_variance.py

# 3. Run Zero-Fallback AST scan and full test suite
.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/
.venv/bin/pytest tests/ -q
```

---

## 8. Compute Hardware, Software Stack & Wall-Clock Training Times

* **Compute & Execution Devices:** Apple **M2** (`8` CPU cores, `8 GB` unified memory, `arm64`, **macOS 26.3.1**). All model training ran on Apple **`mps`**; PIL closed-loop benchmarks ran on **`cpu`** (`OMP_NUM_THREADS=2`).
* **Software Stack:** Python `3.11.12`, PyTorch `2.14.0` (`MPS built: True`), NumPy `2.4.6`, SciPy `1.17.1`, Matplotlib `3.11.2`, Pandas `3.0.6`, PyYAML `6.0.3` ([pyproject.toml](file:///Users/Mattia/Desktop/Studies/Temp/pyproject.toml#L13-L28)).
* **Wall-Clock Training Times:**
  * **SPWM (`mps`):** Seed 42 `2.58 h` (P0) + `7.82 h` (P1); Seed 101 `1.76 h` (P0) + `4.10 h` (P1); Seed 2024 `1.76 h` (P0) + `4.65 h` (P1).
  * **Equalized Baselines (`mps`):** `Continuous RSSM`: `46.0 + 13.0 min` (Seed 42), `44.0 + 12.9 min` (Seed 101), `44.0 + 13.8 min` (Seed 2024); `Deep GRU`: `46.6 + 33.8 min` (Seed 42, 99 ep), `67.1 min` (Seed 101, 80 ep scratch), `66.8 min` (Seed 2024, 80 ep scratch); `Continuous ConvLSTM`: `86.9 min` (Seed 42, 60 ep).
