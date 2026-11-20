# 03 — Benchmarks, Verified Evidence & Replication Runbook

All closed-loop control metrics below were executed under causal **Physics-in-the-Loop (PIL)** (`ArticulatedArmPhysicsStepper`, rigid palm-latch `payload_attached = (t >= t_c)` where $t_c = \texttt{p1\_indices[0]} = t_{\text{contact}} - 7$ is the $\phi_t=1$ sensory-window onset and $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$, causal `OnlineEventCameraStream`, zero `z_gt_next` leakage) on `50` unseen test episodes (`[3600, 3649]`) with `K=256, H=10, N_iter=3`, slew clamp `0.10`. An episode counts as a grasp success iff end-effector error stays `< 3.0 cm` across the final 3-step settling window, terminal $\text{SO}(3)$ rotation error `< 0.05 rad`, and terminal joint velocity $\|\dot{\mathbf{q}}(t_{\text{grasp}})\|_2 < \tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)\,\text{rad/s}$.

---

## 0. Checkpoint Provenance & SHA-256 Matrix

`model.pt` and `best_model.pt` in `results/spwm_v0_12_4_train_m1/` are distinct files on disk (`sha256` prefixes `d358cccbd678` vs `f5cabfca3635` due to non-weight metadata), **but both store identical Epoch 200 EMA `model_state` tensors (`0` weight diffs)**; benchmarks evaluated specific checkpoint files as documented per table:

| Seed | Checkpoint File | Epoch | SHA-256 Prefix | Primary Consumers |
| :--- | :--- | :---: | :---: | :--- |
| `42` | [results/spwm_v0_12_4_train_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/model.pt) | `200` | `d358cccbd678` | Table 1, Table 2 (`Canonical`/`N=6`), Table 3 (Stages 3 & 4b), Table 5, Figs 1, 2, 4 |
| `42` | [results/spwm_v0_12_4_train_m1/best_model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/best_model.pt) | `200` | `f5cabfca3635` | Table 3 (Stage 4a), Table 4 Robustness, Fig 3, `task-161.log` |
| `42` | `checkpoint_phase_1.pt` / `phase_4.pt` / `checkpoint.pt` | `181` / `240` / `240` | `1ab31a4e45cb` / `01a395e7523b` / `977f4aa168f6` | Table 3 (Stages 1–2), Sec 6.3 (`N=25` Val-Sel.), Sec 6.2 (`ep240`) |
| `101` | [results/multiseed/seed_101_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_m1/model.pt) / `best_model.pt` | `80` / `80` | `dad7ce160a64` / `1553a77688bc` | Tables 1, 2 (`N=6`), 4, 5 |
| `101` | `results/multiseed/seed_101_m1/checkpoint_phase_1.pt` / `checkpoint.pt` | `96` / `240` | — | Sec 6.3 (`N=25` Val-Sel.), Sec 6.2 (`ep240`) |
| `2024` | [results/multiseed/seed_2024_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_m1/model.pt) / `best_model.pt` | `130` / `130` | `b7d783df3b6a` / `8888d7e83e3a` | Tables 1, 2 (`N=6`), 4, 5 |
| `2024` | `results/multiseed/seed_2024_m1/checkpoint_phase_2.pt` / `checkpoint.pt` | `102` / `260` | — | Sec 6.3 (`N=25` Val-Sel.), Sec 6.2 (`pil_final_seed2024`) |

---

## 1. Table 1 — Open-Loop Rollout ($N=400$ Test `[3600, 3999]`, $t_{\text{start}}=4$) & Fig 2

**Evidence:** [comparative_metrics_v0_12_4.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json), [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json), [fig2_rollout_open_vs_handover.png](file:///Users/Mattia/Desktop/Studies/Temp/figures/fig2_rollout_open_vs_handover.png), [task-352.log](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/tasks/task-352.log), [task-524.log](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/tasks/task-524.log), [task-127.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-127.log).
*Actual Training Epochs Observed in Checkpoints/Logs:* `SPWM`: `160+240` (Seeds 42, 101) and `160+260` (Seed 2024, `best_model.pt` saved at Ep 130); `ANN-Rate`: `160+240` (all 3 seeds); `Continuous RSSM`: `60` (all 3 seeds); `Deep GRU`: `59` (Seed 42 — ran full `60/60` equalized `SE(3)` + 80-step rollout training with best `ValRoll` at Ep 59), `18` (Seeds 101, 2024 — trained in pre-equalization `train_baselines.py` with `--epochs 18 --patience 6 --lr 2e-4` single-step loss; improved monotonically through Ep 18 without early stopping, and only `Continuous RSSM` was re-trained under the 60-epoch equalized script); `Continuous ConvLSTM`: `54` (Seed 42 — ran `60/60` equalized epochs with best `ValRoll` at Ep 54).

| Model Architecture | Params (Act.) | Pos H=1 | Pos H=10 | Pos H=50 | Pos H=100 | Vel H=1 | Vel H=10 | Vel H=50 | Vel H=100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Continuous ConvLSTM (Seed 42) | `296.9k` | `0.00002` | **`0.00012`** | **`0.00150`** | `0.00707` | `0.00147` | **`0.00553`** | **`0.01095`** | `0.33696` |
| Deep GRU (Seed 42) | `338.9k` | `0.00003` | `0.00029` | `0.00173` | `0.00650` | `0.00097` | `0.00643` | `0.01605` | `0.11191` |
| Continuous RSSM (3 seeds) | `380.5k` | **`0.00001 ± 0.00001`** | `0.00029 ± 0.00006` | `0.00166 ± 0.00002` | **`0.00145 ± 0.00009`** | **`0.00033 ± 0.00017`** | `0.01659 ± 0.00244` | `0.02262 ± 0.00088` | `0.02101 ± 0.00044` |
| ANN-Rate (3 seeds) | `392.1k (317.9k)` | `0.00028 ± 0.00012` | `0.00249 ± 0.00032` | `0.00341 ± 0.00264` | `0.00210 ± 0.00063` | `0.00193 ± 0.00069` | `0.03356 ± 0.00511` | `0.02440 ± 0.01698` | `0.01183 ± 0.00715` |
| **SPWM (3 seeds)** | **`392.1k (317.9k)`** | `0.00018 ± 0.00011` | `0.00348 ± 0.00265` | `0.00315 ± 0.00144` | `0.00240 ± 0.00037` | `0.00139 ± 0.00076` | `0.04507 ± 0.02601` | `0.02466 ± 0.01271` | **`0.00971 ± 0.00177`** |

* **Figure 2 Horizon L2 Error (`SPWM` Seed 42, $H=100$, `12.27%` ALIF spike rate; [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json)):** Pure Open-Loop mean `0.1115 rad` vs. Phasic Handover `0.1482 rad` (note `fig2` also overlays `ANN-Rate` `0.0998 rad` and `RSSM` `0.0607 rad`):
  * `h=1`: `0.0298 ± 0.0074` / `0.0298 ± 0.0074` | `h=5`: `0.1109 ± 0.0131` / `0.1109 ± 0.0131` | `h=10`: `0.1862 ± 0.0652` / `0.1865 ± 0.0663` | `h=25`: `0.1160 ± 0.0413` / `0.1242 ± 0.0499` | `h=50`: `0.1038 ± 0.0710` / `0.1496 ± 0.0662` | `h=80`: `0.0899 ± 0.0500` / `0.1827 ± 0.0833` | `h=100`: `0.1116 ± 0.0478` / `0.1420 ± 0.0641`.
* **Replication:** `.venv/bin/python scripts/eval_table1_ssot.py`

---

## 2. Table 2 — Closed-Loop PIL Benchmark (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$) & Fig 1

**Evidence:** [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json) (`corl_final_verification.pil_3seed_comparison`), [classical_bc_pil_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json), [mppi_benchmark_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/mppi_benchmark/mppi_benchmark_metrics.json), [fig1_mppi_benchmark.png](file:///Users/Mattia/Desktop/Studies/Temp/figures/fig1_mppi_benchmark.png), task logs [task-352.log](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/tasks/task-352.log), [task-524.log](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/tasks/task-524.log), [task-111.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-111.log), [task-118.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-118.log), [task-152.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-152.log), [task-153.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-153.log).

| Method (`50` Test Episodes `[3600, 3649]`) | PIL Succ (%) | Min Err (cm) | Term Err (cm) | $\|\Delta^2 q\|$ | $\|\Delta^3 q\|$ | Flips | Latency (ms) | Source JSON(s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Privileged State-Oracle Jacobian-IK + PD ($\mathbf{q}_{\text{phys}}$, no vision, 2.85 kg cal.)* | *96.0* | *0.01 ± 0.03* | *0.43 ± 0.73* | *0.00771* | *0.00497* | *10.46* | *22.7$^*$* | [classical_bc_pil_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json) |
| Reactive BC (`322,210` params, Seed 42) | `0.0` | `13.13 ± 6.27` | `15.66 ± 8.15` | `0.00140` | `0.00135` | `13.98` | `1.9*` | [classical_bc_pil_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json) |
| Deep GRU (Open-Loop PIL, 3 seeds) | `1.3 ± 2.3` | `10.57 ± 7.83` | `16.72 ± 11.31` | `0.00777` | `0.00622` | `14.67` | `67.4` | `physics_in_loop/deep_gru_open_loop/`, `multiseed_rssm/seed_{101,2024}/deep_gru_mppi_pil_open_loop/` |
| Deep GRU (Online Obs., 3 seeds) | `0.0 ± 0.0` | `6.03 ± 4.00` | `10.87 ± 3.97` | `0.01654` | `0.01350` | `19.67` | `60.3†` | `physics_in_loop/deep_gru/`, `multiseed_rssm/seed_{101,2024}/deep_gru_mppi_pil/` |
| Continuous RSSM (Open-Loop PIL, 3 seeds) | `15.3 ± 4.6` | `1.31 ± 0.02` | `2.88 ± 0.10` | `0.01998` | `0.02059` | `36.23` | `52.4` | `physics_in_loop/rssm_open_loop/`, `multiseed_rssm/seed_{101,2024}/mppi_pil_open_loop/` |
| Continuous RSSM (Online Post., 3 seeds) | `0.0 ± 0.0` | `1.53 ± 0.06` | `22.43 ± 0.58` | `0.03119` | `0.02875` | `36.13` | `35.1*` | `physics_in_loop/rssm/`, `multiseed_rssm/seed_{101,2024}/mppi_pil/` |
| ANN-Rate (3 seeds) | `37.3 ± 9.5` | `1.07 ± 0.28` | `2.98 ± 0.46` | `0.01148` | `0.00938` | `26.00` | `24.6*` | `ann_ablation/seed_{42,101,2024}_mppi_pil/` (`34% / 30% / 48%`) |
| **SPWM ($N=6$ Tier-2, 3 seeds)** | **`58.0 ± 27.1`** | **`0.90 ± 0.04`** | **`2.23 ± 0.68`** | **`0.00860`** | **`0.00608`** | **`19.37`** | **`27.5 (23.1*)`** | `mppi_benchmark/`, `multiseed/seed_{101,2024}_mppi_pil/` (`86% / 56% / 32%`) |
| **SPWM ($N=25$ Val-Sel., 3 seeds)** | **`60.7 ± 2.3`** | **`0.88 ± 0.11`** | **`2.01 ± 0.12`** | **`0.00931`** | **`0.00663`** | **`21.14`** | **`35.5`** | `diagnostics/test50_val25sel_seed{42,101,2024}/` (`58% / 62% / 62%`) |
| **SPWM (Canonical Seed 42)** | **`86.0`** | **`0.86 ± 0.80`** | **`1.57 ± 1.00`** | **`0.00587`** | **`0.00354`** | **`14.02`** | **`36.7 (23.3*)`** | [mppi_benchmark_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/mppi_benchmark/mppi_benchmark_metrics.json) (`43/50`) |

* **Latency Provenance Note:** `$*$` denotes unthrottled CPU latency; primary values for SPWM, Open-Loop RSSM, and Open-Loop Deep GRU ran under `OMP_NUM_THREADS=2`, while `$^\dagger$` marks `Deep GRU (Online Obs., 3 seeds)` mixing unthrottled Seed 42 (`39.41 ms`) with `OMP_NUM_THREADS=2` Seeds 101 & 2024 (`65.38 ms` & `76.14 ms`). Specifically, [results/mppi_benchmark/mppi_benchmark_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/mppi_benchmark/mppi_benchmark_metrics.json) (`36.65 ms`) ran under `OMP_NUM_THREADS=2` (`task-111.log` in `1ca9af8a`), producing `27.51 ± 7.92 ms` in `ssot_eval_metrics.json`, whereas an unthrottled run on `model.pt` ([physics_in_loop/spwm/mppi_benchmark_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/mppi_benchmark/physics_in_loop/spwm/mppi_benchmark_metrics.json), Agent 3 `13:01`) yielded identical accuracy (`86.0%`, `0.855 cm`, `1.567 cm`, `0.00587`) at **`23.33 ms/step`** (`23.07 ± 0.23 ms` in [results/multiseed/multiseed_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/multiseed_metrics.json)).
* **Fig 1 Stratification (Seed 42, `mppi_benchmark_metrics.json`):** Low (`22` ep): `86.4%`, `0.86 cm` | Mid (`18` ep): `77.8%`, `1.05 cm` | High (`10` ep): `100.0%`, `0.50 cm` | Light (`17` ep): `82.4%`, `0.68 cm` | Medium (`17` ep): `100.0%`, `0.77 cm` | Heavy (`16` ep): `75.0%`, `1.14 cm`.
* **Replication:**
  ```bash
  .venv/bin/python experiments/benchmark_classical_and_bc.py --config results/spwm_v0_12_4_train_m1/config.yaml --output-dir results/baselines/classical_and_bc_pil --num-episodes 50 --device cpu
  OMP_NUM_THREADS=2 .venv/bin/python experiments/benchmark_mppi_control.py --config results/spwm_v0_12_4_train_m1/config.yaml --checkpoint results/spwm_v0_12_4_train_m1/model.pt --output-dir results/mppi_benchmark --num-episodes 50 --physics-in-loop --device cpu
  ```

---

## 3. Table 3 — Disentangled 5-Stage PIL Ablation (`[3600, 3649]`)

**Evidence:** [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json) (`corl_final_verification.disentangled_ablation`), [task-524.log](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/tasks/task-524.log), [task-111.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-111.log).

| Ablation Stage | Checkpoint (Epoch) | PIL Succ (%) | Min Err (cm) | Term Err (cm) | $\|\Delta^2 q\|$ | $\|\Delta^3 q\|$ | Flips | JSON |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1. SPWM-Base (Phase 1 Warmup, Unsmoothed) | `checkpoint_phase_1.pt` (`181`) | `62.0` | `1.27 ± 1.14` | `2.51 ± 1.59` | `0.00710` | `0.00499` | `16.04` | [stage1](file:///Users/Mattia/Desktop/Studies/Temp/results/ablations/stage1_base_pil/mppi_benchmark_metrics.json) |
| 2. + Smooth MPPI Controls | `checkpoint_phase_1.pt` (`181`) | `62.0` | `1.09 ± 0.98` | `2.26 ± 1.79` | `0.00729` | `0.00451` | `15.34` | [stage2](file:///Users/Mattia/Desktop/Studies/Temp/results/ablations/stage2_smooth_mppi_pil/mppi_benchmark_metrics.json) |
| 3. + Phase 1 Curriculum (No Handover/Lookahead) | `model.pt` (`200`) | `72.0` | `0.98 ± 0.97` | `1.85 ± 1.31` | `0.00600` | `0.00366` | `14.24` | [stage3](file:///Users/Mattia/Desktop/Studies/Temp/results/ablations/stage3_se3_homeostasis_pil/mppi_benchmark_metrics.json) |
| 4a. + 2-Stage Lookahead (`--disable-handover`, $w_t \equiv 0$) | `best_model.pt` (`200`, identical `model_state` to `model.pt`) | `86.0` | `0.86 ± 0.80` | `1.57 ± 1.00` | `0.00587` | `0.00354` | `14.02` | [stage4a](file:///Users/Mattia/Desktop/Studies/Temp/results/ablations/stage4a_lookahead_only_pil/mppi_benchmark_metrics.json) |
| **4b. + Residual Handover & Lookahead (Full)** | **`model.pt` (`200`)** | **`86.0`** | **`0.86 ± 0.80`** | **`1.57 ± 1.00`** | **`0.00587`** | **`0.00354`** | **`14.02`** | [stage4b](file:///Users/Mattia/Desktop/Studies/Temp/results/mppi_benchmark/mppi_benchmark_metrics.json) |

* **Replication (Stage 4a example):** `.venv/bin/python experiments/benchmark_mppi_control.py --config results/spwm_v0_12_4_train_m1/config.yaml --checkpoint results/spwm_v0_12_4_train_m1/best_model.pt --output-dir results/ablations/stage4a_lookahead_only_pil --num-episodes 50 --disable-handover --physics-in-loop --device cpu`

---

## 4. Table 4 — 3-Seed PIL Robustness & Residual Handover Telemetry (Fig 3)

**Evidence:** [robustness_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark/robustness_metrics.json) (Seed 42 `best_model.pt`), `results/robustness_benchmark_pil_spwm_seed{101,2024}/robustness_metrics.json`, `results/robustness_benchmark_pil_ann_rate_seed{42,101,2024}/robustness_metrics.json`, [fig3_robustness_domain.png](file:///Users/Mattia/Desktop/Studies/Temp/figures/fig3_robustness_domain.png), task logs [task-108..110.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-108.log), [task-58.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-58.log), [task-84.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-84.log).

| Perturbation Condition (`50` Test Ep) | SPWM Seed 42 (Succ / Term) | SPWM 3-Seed Mean (Succ / Term) | SPWM Step $w_t>0$ (42 / 101 / 2024) | ANN-Rate 3-Seed Mean (Succ / Min / Term) |
| :--- | :---: | :---: | :---: | :---: |
| Nominal Baseline | **`86.0%`** / `1.57 cm` | **`58.0 ± 27.1%`** / `2.23 ± 0.68` | `0.0% / 0.0% / 0.0%` | `37.3 ± 9.5%` / `1.07` / `2.98` |
| Sensory Noise (`1.0%`) | **`72.0%`** (`-16.3% rel`) / `1.86 cm` | **`55.3 ± 19.4%`** (`-4.6% rel`) / `2.34 ± 0.63` | `0.0% / 0.0% / 0.0%` | `30.0 ± 3.5%` (`-19.6% rel`) / `1.04` / `3.36` |
| Payload Extrapolation (`4.80 kg`) | **`78.0%`** (`-9.3% rel`) / `1.78 cm` | **`39.3 ± 33.7%`** (`-32.2% rel`) / `2.50 ± 0.62` | `0.0% / 0.0% / 0.5%` (`4.0% ep, w=0.165`) | `26.7 ± 16.2%` (`-28.6% rel`) / `1.36` / `3.64` |
| Actuator Drift (`0.80x` Friction) | **`70.0%`** (`-18.6% rel`) / `2.07 cm` | **`40.7 ± 27.3%`** (`-29.9% rel`) / `2.61 ± 0.54` | `0.0% / 0.0% / 0.5%` (`4.0% ep, w=0.229`) | `30.7 ± 14.0%` (`-17.9% rel`) / `1.42` / `3.74` |
| Actuator Drift (`1.20x` Friction) | **`76.0%`** (`-11.6% rel`) / `1.97 cm` | **`44.0 ± 28.0%`** (`-24.1% rel`) / `2.53 ± 0.56` | `0.0% / 0.0% / 0.5%` (`4.0% ep, w=0.169`) | `33.3 ± 16.7%` (`-10.7% rel`) / `1.34` / `3.51` |

* **Validation Handover Sweep ($\tau \in \{0.026, 0.020, 0.015\}\,\text{m}$, `[3200, 3249]`):** Across all 6 JSONs (`results/diagnostics/val_handover_sweep_{seed42,seed2024}_tau_{0026,0020,0015}/robustness_metrics.json`; [task-119.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-119.log), [task-138.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-138.log)), **all 3 thresholds produce identical results with `0.0%` handovers** (`Seed 42`: `78.0 / 78.0 / 66.0 / 70.0 / 72.0%`; `Seed 2024`: `46.0 / 36.0 / 42.0 / 38.0 / 38.0%`).
* **Diagnostic Proof ([task-161.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-161.log) on `val` `[3200, 3214]`, `120` $\phi_t=1$ steps):**
  * `Seed 42`: `dist_pred` median `0.0139 m` (`p90=0.0329`, `max=0.1335`), `dist_obs` median `0.2155 m` (`min=0.0613`), `rot_obs` median `0.6377 rad` (`min=0.3770`), `obs < pred = 0.0%`.
  * `Seed 2024`: `dist_pred` median `0.0174 m` (`p90=0.0415`, `max=0.1244`), `dist_obs` median `0.1598 m` (`min=0.0324`), `rot_obs` median `0.3850 rad` (`min=0.2435`), `obs < pred = 0.0%`.
  * *Conclusion:* `dist_obs` never satisfies `< 0.025 m` or `< dist_pred` on nominal/val trajectories, preventing sensory shock corruption.
* **Replication:**
  ```bash
  OMP_NUM_THREADS=2 .venv/bin/python experiments/benchmark_robustness_stress_test.py \
    --config results/spwm_v0_12_4_train_m1/config.yaml --checkpoint results/spwm_v0_12_4_train_m1/best_model.pt \
    --output-dir results/robustness_benchmark --num-mppi-episodes 50 --num-sensor-trajectories 400 --physics-in-loop --device cpu
  ```

---

## 5. Table 5 — Neuromorphic SOPs, Loihi-2 Core Energy & E2E Power (Fig 4)

**Evidence:** [energy_profiling_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/energy_profiling/energy_profiling_metrics.json) (Seed 42: `12.27%` spikes, `8,229` SOPs, `12.34 nJ`, `4.89x`), [seed_101](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_energy/energy_profiling_metrics.json) (`12.26%`, `8,220` SOPs, `12.33 nJ`, `4.90x`), [seed_2024](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_energy/energy_profiling_metrics.json) (`11.93%`, `8,001` SOPs, `12.00 nJ`, `5.03x`), [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json), [task-127.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-127.log), [fig4_energy_efficiency.png](file:///Users/Mattia/Desktop/Studies/Temp/figures/fig4_energy_efficiency.png).
*(Assumptions: Loihi-2 `1.5 pJ/SOP`, Jetson Orin GPU `0.9 pJ/MAC`, shared `EventEncoder` `882,688` MACs = `794.4 nJ`, $K=256 \times H=10 \times N_{\text{iter}}=3 = 7,680$ batched MPPI rollouts/step.)*

| Architecture | Core Ops / Step | Core Energy (nJ) | Core Power @ 20 Hz | Core vs. GPU | MPPI MACs | E2E Energy (mJ) | E2E Power @ 20 Hz |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SPWM (Loihi-2 Core + Orin, 3 seeds)** | **`8,150 ± 129 SOPs`** | **`12.22 ± 0.19`** | **`0.244 ± 0.004 mW`** | **`4.94 ± 0.08×`** | **`1.219 G`** | **`1.0979`** | **`21.96 mW`** |
| ANN-Rate (Orin GPU) | `67,072 MACs` | `60.36` | `1.207 mW` | `1.00×` | `1.219 G` | `1.0979` | `21.96 mW` |
| Deep GRU (Orin GPU) | `299,520 MACs` | `269.57` | `5.391 mW` | `1.00×` | `1.942 G` | `1.7490` | `34.98 mW` |
| Continuous RSSM (Orin GPU) | `365,056 MACs` | `328.55` | `6.571 mW` | `1.00×` | `2.358 G` | `2.1236` | `42.47 mW` |
| Continuous ConvLSTM (Orin GPU) | `2,359,296 MACs` | `2123.37` | `42.467 mW` | `1.00×` | `111.312 G` | `100.1833` | `2003.67 mW` |

* **Analytical SOP & MAC Derivation ([profile_energy_sops.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/profile_energy_sops.py#L67-L146,L354-L388)):**
  * **Recurrent Core (`67,072` MACs; `8,150 ± 129` 3-seed SOPs):** `run_energy_profiling()` computes `spwm_core_macs = (latent_dim + action_dim) * hidden_dim + hidden_dim * latent_dim = (128 + 6) * 256 + 256 * 128 = 34,304 + 32,768 = 67,072` MACs (`L362`). Multiplying by empirical ALIF spike rate (`SPWM.forward_representation_with_spikes` over 400 test trajectories) yields `8,229` (`12.27%`, Seed 42), `8,220` (`12.26%`, Seed 101), and `8,001` (`11.93%`, Seed 2024) SOPs/step. **ALIF membrane leak operations are excluded** (`compute_neuromorphic_energy_nj` is never called; `L365` uses `spwm_core_sops * 1.5e-3` nJ directly).
  * **Shared `EventEncoder` (`882,688` MACs = `794.42 nJ`):** Unified 4-ch stem (`4*16*9*16*16 + 16*32*9*8*8 + 32*16*1*8*8 = 475,136`) + Wrist 2-ch stem/fusion (`2*16*9*16*16 + 16*32*9*8*8 + 32*16*1*8*8 = 401,408`) + SpatialSoftmax (`16*8*8*2 = 2,048`) + Linear `proj` (`32*128 = 4,096`).
  * **Batched MPPI $M_1$ Rollouts (`1,218,969,600` $\approx 1.219\text{ GMACs}$):** Per rollout step (`158,720` MACs): `backbone[0]` (`134*256 = 34,304`) + `backbone[3]` (`256*256 = 65,536`) + `force_head` (`256*96 = 24,576`) + `dissipation_head` (`256*96 = 24,576`) + `restorative_q_head` (`256*32 = 8,192`) + simplified `PhysicalDecoder` (`2*128*6 = 1,536`), multiplied by $K \times H \times N_{\text{iter}} = 256 \times 10 \times 3 = 7,680$ steps.
* **Replication:** `.venv/bin/python experiments/profile_energy_sops.py --config results/spwm_v0_12_4_train_m1/config.yaml --checkpoint results/spwm_v0_12_4_train_m1/model.pt --output-dir results/energy_profiling --device cpu --total-trajectories 4000`

---

## 6. Forensic Root-Cause Diagnostics (Restored Agent 2 + Agent 1 Evidence)

### 6.1 Continuous RSSM, Deep GRU, ANN-Rate & SPWM Forensic Decomposition
**Evidence:** [forensic_rssm_gru_failure.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/forensic_rssm_gru_failure.py) $\to$ [forensic_rssm_gru_failure.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/forensic_rssm_gru_failure.json) (velocity violations measured against $\tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)\,\text{rad/s}$, overall reach $\min_t e \le 3.0\,\text{cm}$, and $t_c = t_{\text{contact}} - 7 \to t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$).
* **Continuous RSSM Online (`0/150 = 0.0%` across 3 seeds):** Overall reach $\le 3\,\text{cm}$ (all pre-$t_c$) on `49/50` (`98%` Seed 42), `44/50` (`88%` Seed 101), and `48/50` (`96%` Seed 2024) — `141/150 (94.0%)` total, with `1, 6, 2` never reaching — **then explodes immediately after `t >= t_c` payload engagement** (`141/150` fail simultaneously via post-$t_c$ impulse divergence + velocity violation; `150/150` total velocity violations; 3-seed aggregate `2.92 cm @ t_c -> 22.43 ± 0.58 cm @ t_grasp`, `7.17 ± 0.32 rad/s`; Seed 42: `3.13 cm -> 23.06 ± 10.19 cm`, `7.41 ± 1.42 rad/s`).
* **Deep GRU Online (`0/50`, Seed 42):** `43/50 (86.0%)` overall reach (`42` pre-$t_c$ + `1` post-$t_c$, `7/50` never reached), `50/50 (100%)` velocity violations, `42/50 (84.0%)` post-$t_c$ divergence + velocity violation (`5.99 cm @ t_c -> 14.96 ± 5.82 cm @ t_grasp`, `5.75 ± 1.71 rad/s`), and `1/50 (2.0%)` post-$t_c$ reach-then-diverge.
* **ANN-Rate (`17/50 = 34.0%`, Seed 42):** `50/50 (100%)` overall reach (`0/50` never reached), `2/50 (4.0%)` velocity violations, `0/50` Div+Vel, `19/50 (38.0%)` post-$t_c$ spatial-only divergence, and `14/50 (28.0%)` other fails (`7/50` post-$t_c$ reach + `1/50` vel-only + `6/50` rot-only; `2.61 -> 2.75 ± 1.65 cm`, `0.59 ± 0.32 rad/s`).
* **SPWM (`43/50 = 86.0%`, Seed 42):** `49/50 (98.0%)` overall reach (`1/50` never reached), `0/50 (0.0%)` velocity violations, `0/50` Div+Vel, `2/50 (4.0%)` post-$t_c$ spatial-only divergence, and `4/50 (8.0%)` post-$t_c$ reach-then-diverge (`1.69 -> 1.57 ± 1.00 cm`, `0.33 ± 0.25 rad/s`).

### 6.2 SPWM `N=6` Tier-2 Variance Audit & Late-Epoch (`ep240/260`) PIL Checkpoints
**Evidence:** [audit_spwm_seed_variance.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/audit_spwm_seed_variance.py) $\to$ [spwm_seed_variance_audit.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/spwm_seed_variance_audit.json) ([task-152.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-152.log)), plus 50-ep PIL evaluations in [task-60.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-60.log) and [task-107.log](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/tasks/task-107.log):
* **Internal Module Health Across All 3 Seeds:** `action_proj` spectral norm $\|W_u\|_2$ (`0.8594`, `1.0131`, `0.8189`), `gate_scale` (`~49.04–49.07`), ALIF mean spike rate (`12.27%`, `12.26%`, `11.93%`), silent fraction (`0.78%`, `0.78%`, `2.34%`), and hyperactive fraction (`0.0%`) are cleanly converged across all seeds.
* **Final `checkpoint.pt` PIL Evaluations:**
  * `Seed 42` (`ep 240`, [pil_ep240_seed42](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/pil_ep240_seed42/mppi_benchmark_metrics.json)): **`58.0%`** (`1.00 ± 0.72 cm` min, `2.12 ± 1.23 cm` term, `31.8 ms/step`).
  * `Seed 101` (`ep 240`, [pil_ep240_seed101](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/pil_ep240_seed101/mppi_benchmark_metrics.json)): **`24.0%`** (`1.02 ± 0.70 cm` min, `4.19 ± 2.35 cm` term, `31.7 ms/step`).
  * `Seed 2024` (`ep 260`, [pil_final_seed2024](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/pil_final_seed2024/mppi_benchmark_metrics.json)): **`52.0%`** (`2.01 ± 1.40 cm` min, `2.80 ± 1.63 cm` term, `24.8 ms/step`).

### 6.3 $N=25$ Validation Checkpoint Selection (`11.7×` Inter-Seed Variance Reduction)
**Evidence:** [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json) (`val25_checkpoint_selection`, `spwm_n25_val`), `results/diagnostics/val25_*`, `results/diagnostics/test50_val25sel_*`, [task-126.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-126.log), [task-152.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-152.log), [task-153.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-153.log):
Evaluating all 15 saved checkpoints strictly on held-out validation split `[3200, 3224]` proves the `27.1%` std under `N=6` Tier-2 (`86.0% / 56.0% / 32.0%`) was selection noise from a 6-trajectory validation probe:
* **Seed 42 (`[3200, 3224]`):** `best_model` (`200`, $\tau=0.990$ EMA weights): `80.0%`, `phase_1` (`181`): `60.0%`, `phase_2` (`188`): `52.0%`, `phase_3` (`200`, raw un-averaged weights at Latch Phase 3 entry): `88.0%`, **`phase_4` (`240`): `92.0%`** $\to$ Test `[3600, 3649]`: **`58.0%`** (`1.00 cm` min, `2.12 cm` term).
* **Seed 101 (`[3200, 3224]`):** `best_model` (`80`): `56.0%`, **`phase_1` (`96`): `68.0%`**, `phase_2` (`104`): `52.0%`, `phase_3` (`124`): `4.0%`, `phase_4` (`240`): `40.0%` $\to$ Test `[3600, 3649]`: **`62.0%`** (`0.85 cm` min, `2.02 cm` term).
* **Seed 2024 (`[3200, 3224]`):** `best_model` (`130`): `48.0%`, `phase_1` (`94`): `64.0%`, **`phase_2` (`102`): `80.0%`**, `phase_3` (`122`): `40.0%`, `phase_4` (`260`): `60.0%` $\to$ Test `[3600, 3649]`: **`62.0%`** (`0.79 cm` min, `1.89 cm` term).
* **Aggregate ($N=25$ Val-Selected):** **`60.7 ± 2.3%`** (`11.7×` reduction in inter-seed std from `27.06%` to `2.31%`).

### 6.4 `heft_shock_timeseries` Figure Provenance (`figures/heft_shock_timeseries.pdf`)
* **What It Plots ([plot_heft_shock_timeseries.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/plot_heft_shock_timeseries.py#L212-L318)):** A **live single-episode** closed-loop PIL rollout on canonical test trajectory `3622` (`2.45 kg ElongatedCylinder`)—not a 50-episode mean—comparing SPWM (Seed 42 `model.pt`) vs. Continuous RSSM (Seed 42 `model.pt`) vs. Ground Truth from `t_start = max(5, t_c - 20)` to `t_grasp = t_c + 7`: (a) `q_1` / `q_2` joint angles, (b) end-effector error (`err_cm`) with `3.0 cm` threshold, (c) joint velocity norm `||dq/dt||_2` and handover gate `w_t`, and (d) 256-neuron ALIF `ema_spikes` heatmap.
* **Data Source & Regeneration:** Loads raw trajectory [trajectory_03622.json.gz](file:///Users/Mattia/Desktop/Studies/Temp/sim/ExportedTrajectories/trajectory_03622.json.gz) (no summary JSON) and runs live PIL rollouts:
  ```bash
  .venv/bin/python scripts/plot_heft_shock_timeseries.py --traj-id 3622
  ```

### 6.5 Standalone Phase 0 Representation Metrics (`results/spwm_v0_12_4_train_representation/`)
* **Training & Validation ([metrics_spwm_v0_12_4.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/metrics_spwm_v0_12_4.json), [training_log.csv](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/training_log.csv)):** Seed 42 best checkpoint at **Ep 159** reaches `val_loss = 1.281731` (`-98.01%` from Ep 1 `65.4230`), `val_pos_err = 0.07430 rad`, `val_vel_err = 0.16678 rad/s`, `val_loss_cartesian = 0.002454` (Seed 101 Ep 160: `val_loss = 1.3251`, `val_pos_err = 0.07696`; Seed 2024 Ep 146: `val_loss = 1.4362`, `val_pos_err = 0.07738`).
* **Grounded FK Accuracy & Homeostasis Ablation ([docs/History.md](file:///Users/Mattia/Desktop/Studies/Temp/docs/History.md#L898), [diagnostics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/diagnostics.json)):** 50-episode grounded `forward_representation` at `t_grasp` achieves **`2.04 cm`** FK position MAE (`p90 = 3.33 cm`) and **`0.043 rad`** rotation error (`p90 = 0.082 rad`). On the single-batch Phase 0 diagnostic (`diagnostics.json`), homeostasis + SE(3) + smoothness (`v0.12.4` vs. `v0.12.20` without homeostasis) cut silent ALIF neurons (`< 0.01`) from `44.92% -> 19.14%` (`-57%`) and hyperactive neurons (`> 0.50`) from `5.86% -> 2.34%` (converging to `<= 2.34%` silent and `0.0%` hyperactive across the full $N=400$ test split in `spwm_seed_variance_audit.json`).

---

## 7. Master Verification & SSOT Plot Regeneration Commands

```bash
# 1. Regenerate SSOT aggregation JSONs & CoRL summary
.venv/bin/python scripts/aggregate_corl_final_verification.py

# 2. Regenerate all 4 figures (figures/fig1..4_*) without re-running benchmarks
.venv/bin/python scripts/run_ssot_benchmark_and_plot.py --skip-eval

# 3. Regenerate forensic diagnostics
.venv/bin/python scripts/diagnostics/forensic_rssm_gru_failure.py
.venv/bin/python scripts/diagnostics/audit_spwm_seed_variance.py

# 4. Run Zero-Fallback AST scan and full 101-test integrity gate
.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/
.venv/bin/pytest tests/test_leakage.py tests/test_pil_integrity.py -v
.venv/bin/pytest tests/ -q
```

---

## 8. Compute Hardware, Software Stack & Wall-Clock Training Times

* **Compute & Execution Devices:** Apple **M2** (`8` CPU cores, `8 GB` unified memory, `arm64`, **macOS 26.3.1**). All model training ran on Apple **`mps`**; PIL closed-loop benchmarks ran on **`cpu`** (with `OMP_NUM_THREADS=2` vs. unthrottled runs noted in Table 2).
* **Software Stack:** Python `3.11.12`, PyTorch `2.14.0` (`MPS built: True`), NumPy `2.4.6`, SciPy `1.17.1`, Matplotlib `3.11.2`, Pandas `3.0.6`, PyYAML `6.0.3` ([pyproject.toml](file:///Users/Mattia/Desktop/Studies/Temp/pyproject.toml#L13-L28)).
* **Wall-Clock Training Times (Sum of Per-Epoch `training_log.csv` / Task-Log Timings):**
  * **SPWM Seed 42:** Phase 0 (`160` ep) = **`2.58 h`** (`9,296 s`, `58.1 s/ep`); Phase 1 (`240` ep) = **`7.82 h`** (`28,166 s`, `117.4 s/ep`).
  * **SPWM Seed 101:** Phase 0 (`160` ep) = **`1.76 h`** (`6,321 s`, `39.5 s/ep`); Phase 1 (`240` ep) = **`4.10 h`** (`14,775 s`, `61.6 s/ep`).
  * **SPWM Seed 2024:** Phase 0 (`160` ep) = **`1.76 h`** (`6,323 s`, `39.5 s/ep`); Phase 1 (`260` ep) = **`4.65 h`** (`16,731 s`, `64.4 s/ep`).
  * **60-Epoch Equalized Baselines (`task-352.log`):** `Continuous RSSM` (`45.98 min` Seed 42, `44.04 min` Seed 101, `44.01 min` Seed 2024), `Deep GRU` (`46.60 min` Seed 42), `Continuous ConvLSTM` (`86.90 min` Seed 42).
