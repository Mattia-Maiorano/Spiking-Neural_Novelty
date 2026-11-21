# PAPER_CONTEXT — SPWM-v0.12.4 Complete SSOT Metrics, Architecture & Training Context
> Lossless, token-compacted compilation of all SSOT benchmark JSONs, configuration YAML, and architecture/training/world documentation.
> JSON/YAML files are compacted into schema-aligned Markdown tables and key-value blocks without dropping any fields or values.

## 1. SSOT Metrics & Benchmark Data

### `results/corl_final_benchmark_summary.md`
> CoRL final grounded 3-seed, 50-episode PIL, 5-stage ablation, equalized baselines, robustness, and energy summary + LaTeX tables

# SPWM-v0.12.4 — CoRL Final Grounded Benchmark Summary

All closed-loop control metrics below are evaluated under strict **Physics-in-the-Loop (PIL)** execution (`ArticulatedArmPhysicsStepper`, rigid payload attached at contact onset $t \ge t_c$, causal `OnlineEventCameraStream`, and **zero** ground-truth `z_gt_next` leakage) across `50` unseen test episodes (`[3600, 3649]`) under identical MPPI settings ($K=256, H=10, N_{\text{iter}}=3$, slew clamp $0.10$). All multi-seed evaluations aggregate Seeds `42, 101, 2024` as `mean ± std`. Both `Continuous RSSM` and `Deep GRU` are trained with equalized multi-step SE(3) rollout + AR(1) action perturbations + `ClosedLoopBuffer` DAgger replay across all 3 seeds, and evaluated under all three causal PIL regimes (**Open-Loop**, **Un-Gated Online $w_t=1.0$**, and **Residual-Gated Online** using the exact same FK rejection gate as SPWM). Headline SPWM results lead strictly with the **$N=25$ validation-selected (`[3200, 3224]`) 3-seed `mean ± std`**.

---

## 1. Equalized Training Budgets & Wall-Clock Provenance

| Model | Seed 42 (Ep / Best / Time) | Seed 101 (Ep / Best / Time) | Seed 2024 (Ep / Best / Time) | Regime |
| :--- | :---: | :---: | :---: | :--- |
| **SPWM-v0.12.4** | `160+240` / Ep `240` (`N=25` sel.) / `10.40 h` | `160+240` / Ep `96` (`N=25` sel.) / `5.86 h` | `160+260` / Ep `102` (`N=25` sel.) / `6.41 h` | 2-Phase / 4-Latch (`MPS`) |
| **ANN-Rate** | `160+240` / `best_model` / — | `160+240` / `best_model` / — | `160+240` / `best_model` / — | Identical to SPWM (`MPS`) |
| **Continuous RSSM** | `60+16=76` / Ep `61` / `13.0 m` (+60 ep base) | `60+16=76` / Ep `61` / `12.9 m` (+60 ep base) | `60+17=77` / Ep `62` / `13.8 m` (+60 ep base) | 80-step SE(3) + AR(1) + DAgger (`MPS`) |
| **Deep GRU** | `59+40=99` / Ep `90` / `33.8 m` (+59 ep base) | `80` (scratch) / Ep `68` / `67.1 m` | `80` (scratch) / Ep `80` / `66.8 m` | 80-step SE(3) + AR(1) + DAgger (`MPS`) |
| **Continuous ConvLSTM** | `60` / Ep `54` / `86.9 m` | — | — | 80-step SE(3) (`MPS`, open-loop only) |

---

## 2. Publication-Ready LaTeX Tables

### Table 1: Multi-Horizon Open-Loop Rollout ($N=400$, $H=100$, $t_{\text{start}}=4$; Equalized 3-Seed Mean $\pm$ Std)

```latex
\begin{table}[t]
\centering
\caption{\textbf{Multi-Horizon Open-Loop Rollout ($N=400$, $H=100$, $t_{\text{start}}=4$).} Symmetric Position and Velocity MSE horizons ($H \in \{1, 10, 50, 100\}$). Mean $\pm$ seed std across 3 seeds ($42, 101, 2024$) for SPWM, ANN-Rate, Continuous RSSM, and Equalized Deep GRU; single seed ($42$) for Continuous ConvLSTM.}
\label{tab:open_loop_rollout}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Model Architecture} & \textbf{Params (Act.)} & \textbf{Pos H=1} & \textbf{Pos H=10} & \textbf{Pos H=50} & \textbf{Pos H=100} & \textbf{Vel H=1} & \textbf{Vel H=10} & \textbf{Vel H=50} & \textbf{Vel H=100} \\
\midrule
Continuous ConvLSTM (Seed 42) & 296.9k & \textbf{0.00002} & \textbf{0.00012} & \textbf{0.00150} & 0.00707 & 0.00147 & \textbf{0.00553} & \textbf{0.01095} & 0.33696 \\
Deep GRU (Equalized, 3 seeds) & 338.9k & 0.00003 $\pm$ 0.00001 & 0.00147 $\pm$ 0.00044 & 0.00195 $\pm$ 0.00003 & 0.00469 $\pm$ 0.00048 & 0.00203 $\pm$ 0.00019 & 0.01852 $\pm$ 0.00237 & 0.01332 $\pm$ 0.00024 & 0.08716 $\pm$ 0.01445 \\
Continuous RSSM (Equalized, 3 seeds) & 380.5k & 0.00007 $\pm$ 0.00004 & 0.00033 $\pm$ 0.00006 & 0.00174 $\pm$ 0.00004 & \textbf{0.00152 $\pm$ 0.00014} & 0.00153 $\pm$ 0.00186 & 0.01660 $\pm$ 0.00201 & 0.02256 $\pm$ 0.00088 & 0.02130 $\pm$ 0.00082 \\
ANN-Rate (Leaky, 3 seeds) & 392.1k (317.9k) & 0.00028 $\pm$ 0.00012 & 0.00249 $\pm$ 0.00032 & 0.00341 $\pm$ 0.00264 & 0.00210 $\pm$ 0.00063 & 0.00193 $\pm$ 0.00069 & 0.03356 $\pm$ 0.00511 & 0.02440 $\pm$ 0.01698 & 0.01183 $\pm$ 0.00715 \\
\textbf{SPWM-v0.12.4 (3 seeds)} & \textbf{392.1k (317.9k)} & 0.00018 $\pm$ 0.00011 & 0.00348 $\pm$ 0.00265 & 0.00315 $\pm$ 0.00144 & 0.00240 $\pm$ 0.00037 & \textbf{0.00139 $\pm$ 0.00076} & 0.04507 $\pm$ 0.02601 & 0.02466 $\pm$ 0.01271 & \textbf{0.00971 $\pm$ 0.00177} \\
\bottomrule
\end{tabular}%
}
\end{table}
```

### Table 2: Closed-Loop Physics-in-the-Loop (PIL) Benchmark (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$, 3-Regime Comparison)

```latex
\begin{table}[t]
\centering
\caption{\textbf{Closed-Loop Physics-in-the-Loop (PIL) Manipulation Benchmark (`50` Test Episodes, Payload Attached at $t \ge t_c$).} Equalized Continuous RSSM and Deep GRU (3 seeds, AR(1)+DAgger) are reported side-by-side across all three regimes: Open-Loop PIL ($w_t=0$), Un-Gated Online Observation ($w_t=1.0$), and Residual-Gated Online Observation (identical FK rejection gate to SPWM). SPWM leads with $N=25$ validation-split (`[3200, 3224]`) checkpoint selection across all 3 seeds (with $N=6$ Tier-2 selection noted for variance comparison).}
\label{tab:pil_control}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Method} & \textbf{PIL Success (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} & $|\Delta^2 q|$ ($\text{rad/step}^2$) & $|\Delta^3 q|$ & \textbf{Flips} & \textbf{Latency (ms)} \\
\midrule
\textit{Privileged Jacobian-IK + PD (Oracle)} & \textit{96.0} & \textit{0.01 $\pm$ 0.03} & \textit{0.43 $\pm$ 0.73} & \textit{0.00771} & \textit{0.00497} & \textit{10.46} & \textit{22.7} \\
\midrule
Reactive BC (50 ep, Seed 42) & 0.0 & 13.13 $\pm$ 6.27 & 15.66 $\pm$ 8.15 & 0.00140 & 0.00135 & 13.98 & 1.9 \\
\midrule
Continuous RSSM (Open-Loop PIL, $w_t=0$) & 15.3 $\pm$ 7.0 & 1.23 $\pm$ 0.06 & 2.74 $\pm$ 0.07 & 0.02037 & 0.02139 & 36.86 & 46.8 \\
Continuous RSSM (Un-Gated Online, $w_t=1.0$) & 0.0 $\pm$ 0.0 & 1.43 $\pm$ 0.03 & 25.49 $\pm$ 1.31 & 0.03153 & 0.02956 & 36.49 & 46.9 \\
Continuous RSSM (Residual-Gated Online) & 15.3 $\pm$ 5.0 & 1.23 $\pm$ 0.05 & 2.72 $\pm$ 0.11 & 0.02049 & 0.02161 & 36.94 & 46.5 \\
\midrule
Deep GRU (Open-Loop PIL, $w_t=0$) & 77.3 $\pm$ 4.2 & 1.14 $\pm$ 0.31 & 1.90 $\pm$ 0.18 & 0.00450 & 0.00268 & 10.79 & 57.8 \\
Deep GRU (Un-Gated Online, $w_t=1.0$) & 0.0 $\pm$ 0.0 & 1.47 $\pm$ 0.51 & 7.35 $\pm$ 1.34 & 0.00698 & 0.00483 & 15.08 & 57.1 \\
Deep GRU (Residual-Gated Online) & 76.7 $\pm$ 5.0 & 1.14 $\pm$ 0.31 & 1.90 $\pm$ 0.18 & 0.00449 & 0.00267 & 10.79 & 57.7 \\
\midrule
ANN-Rate (Leaky Rate, 3 seeds) & 37.3 $\pm$ 9.5 & 1.07 $\pm$ 0.28 & 2.98 $\pm$ 0.46 & 0.01148 & 0.00938 & 26.00 & 24.6 \\
\textbf{SPWM-v0.12.4 ($N=25$ Val-Sel., 3 seeds)} & \textbf{60.7 $\pm$ 2.3} & \textbf{0.88 $\pm$ 0.11} & \textbf{2.01 $\pm$ 0.12} & \textbf{0.00931} & \textbf{0.00663} & \textbf{21.14} & \textbf{26.6} \\
SPWM-v0.12.4 ($N=6$ Tier-2, 3 seeds) & 58.0 $\pm$ 27.1 & 0.90 $\pm$ 0.04 & 2.23 $\pm$ 0.68 & 0.00860 & 0.00608 & 19.37 & 27.5 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

> **Critical Findings from Baseline Equalization & Shared FK Residual Gating:**
> 1. **Why un-gated online baselines collapsed (`0.0%`):** Forcing unconditional overwrites (`w_t = 1.0`) from Python's domain-gapped 2D `OnlineEventCameraStream` causes both `Continuous RSSM` (`0.0%`, term `25.49 ± 1.31 cm`) and `Deep GRU` (`0.0%`, term `7.35 ± 1.34 cm`) to fail. Applying the identical causal FK residual rejection gate restores `Continuous RSSM` to its open-loop ceiling (`15.3 ± 5.0%` vs `15.3 ± 7.0%`, term `2.72 ± 0.11 cm`), confirming RSSM's `15.3%` limit is architectural (MPPI jitter `|Δ²q| = 0.02049`, `36.94` flips). Conversely, equalized `Deep GRU` recovers to **`76.7 ± 5.0%`** (`82.0% / 72.0% / 76.0%`; `77.3 ± 4.2%` open-loop) with `0/150` velocity violations.
> 2. **Architectural Trade-Off vs. Equalized Deep GRU (Latency, Velocity & Mechanical OOD Collapse):** While equalized `Deep GRU` achieves higher nominal PIL grasp success (`76.7 ± 5.0%` vs. `60.7 ± 2.3%` for SPWM) and smoother nominal controls (`|Δ²q| = 0.00449` vs. `0.00931`), `Deep GRU` (i) exceeds the `50.0 ms` (`20 Hz`) real-time budget at **`57.7 ms/step`** (`2.17×` slower than SPWM's `26.6 ms/step` under identical `OMP_NUM_THREADS=2`), (ii) exhibits **`9.0×` higher $H=100$ open-loop velocity MSE** (`0.08716` vs. `0.00971 rad²/s²`), (iii) **collapses under zero-shot mechanical distribution shifts** (`23.3 ± 12.9%` at `4.80 kg` payload, `26.7 ± 23.9%` at `0.80×` friction, `30.0 ± 15.1%` at `1.20×` friction — trailing SPWM by **`+18.0 to +26.0 pp`** across all three physical perturbations in Table 4), and (iv) consumes **`22.1×` higher standby core energy** (`269.57 nJ/step` on Orin vs. `12.22 nJ/step` on Loihi-2) and **`1.59×` higher E2E step energy** (`1.7490 mJ` vs. `1.0979 mJ`).
> 3. **Controlled Spiking Isolation (`SPWM` vs. `ANN-Rate`):** Under identical parameter budget (`317.9k` active), `(q, p)` factorization, `160+240` training curriculum, and MPPI controller, SPWM's discrete ALIF thresholding outperforms `ANN-Rate` by **`+23.4 pp`** nominally (`60.7 ± 2.3%` vs. `37.3 ± 9.5%`), **`+24.7 pp`** under `1.0%` sensory noise (`54.7 ± 1.2%` vs. `30.0 ± 3.5%`, `2.0×` better relative retention `-9.9%` vs. `-19.6%`), and **`+14.7 to +22.0 pp`** across out-of-distribution payload and friction shifts.

### Table 3: Full 3-Seed 6-Row SPWM Architectural Ablation in PIL (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$, Seeds `42, 101, 2024`)

```latex
\begin{table}[t]
\centering
\caption{\textbf{3-Seed 6-Row SPWM Architectural Ablation in PIL (`50` Episodes `[3600, 3649]`, $t \ge t_c$, Seeds `42, 101, 2024`).} Stages 1--2 evaluate pre-DAgger Phase 1 warmup weights (`checkpoint\_phase\_1.pt`), Stage 3a evaluates Phase 3/4 AR(1)+DAgger curriculum weights (`checkpoint\_phase\_4.pt`) without lookahead/handover, Stage 3b evaluates the $N=25$ validation-selected checkpoints (`42: phase\_4`, `101: phase\_1`, `2024: phase\_2`) \emph{without} lookahead/handover to cleanly isolate checkpoint selection, and Stages 4a--4b apply 2-Stage $\text{SE}(3)$ Lookahead and Residual Handover to those same $N=25$ validation-selected checkpoints.}
\label{tab:spwm_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Ablation Stage} & \textbf{Seeds (42 / 101 / 2024)} & \textbf{PIL Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} & $|\Delta^2 q|$ & $|\Delta^3 q|$ & \textbf{Flips} \\
\midrule
1. SPWM-Base (Phase 1 Warmup, Unsmoothed) & 62 / 44 / 30 & 45.3 $\pm$ 16.0 & 1.03 $\pm$ 0.21 & 2.50 $\pm$ 0.11 & 0.00983 & 0.00765 & 23.03 \\
2. + Smooth MPPI Controls (Pre-DAgger) & 62 / 44 / 48 & 51.3 $\pm$ 9.5 & 1.02 $\pm$ 0.08 & 2.41 $\pm$ 0.13 & 0.00991 & 0.00705 & 21.99 \\
3a. + Phase 3/4 Curriculum (\texttt{phase\_4}, No Lookahead/Handover) & 56 / 18 / 40 & 38.0 $\pm$ 19.1 & 1.53 $\pm$ 0.59 & 3.27 $\pm$ 1.13 & 0.00589 & 0.00354 & 11.17 \\
3b. + $N=25$ Validation Checkpoint Selection (Val-Sel., No Lookahead/Handover) & 56 / 44 / 48 & 49.3 $\pm$ 6.1 & 1.00 $\pm$ 0.16 & 2.29 $\pm$ 0.23 & 0.00921 & 0.00659 & 21.18 \\
4a. + 2-Stage SE(3) Lookahead (Val-Sel., $w_t \equiv 0$) & 58 / 62 / 62 & 60.7 $\pm$ 2.3 & 0.88 $\pm$ 0.11 & 2.01 $\pm$ 0.12 & 0.00931 & 0.00663 & 21.14 \\
\textbf{4b. + Residual Rejection Gate (Full SPWM, Val-Sel.)} & 58 / 62 / 62 & \textbf{60.7 $\pm$ 2.3} & \textbf{0.88 $\pm$ 0.11} & \textbf{2.01 $\pm$ 0.12} & 0.00931 & 0.00663 & 21.14 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

> **Honest Disentangled Diagnostic (Stage 2 vs. Stage 3a vs. Stage 3b vs. Stage 4a):** Comparing Stage 2 (`51.3 ± 9.5%`) against Stage 3a (`38.0 ± 19.1%`) across all 3 seeds shows that while Phase 3/4 AR(1)+DAgger replay (`checkpoint_phase_4.pt`) smooths control trajectories (`|Δ²q|: 0.00991 -> 0.00589 rad/step²`, flips `21.99 -> 11.17`), it **degrades 3-seed mean PIL grasp success by `-13.3 pp`** (`56.0% / 18.0% / 40.0%`) and **doubles inter-seed standard deviation (`9.5% -> 19.1%`)**. Isolating **$N=25$ validation-split (`[3200, 3224]`) checkpoint selection** *without* Lookahead (`Stage 3b`, `42: phase_4`, `101: phase_1`, `2024: phase_2`) recovers 3-seed PIL success to **`49.3 ± 6.1%`** (`56.0% / 44.0% / 48.0%`, `+11.3 pp` from checkpoint selection alone). Applying **2-Stage SE(3) Lookahead** to those exact same $N=25$ validation-selected checkpoints (`Stage 4a/4b`) then contributes a further **`+11.3 pp`** gain (`49.3% -> 60.7 ± 2.3%`, `58.0% / 62.0% / 62.0%`), cleanly separating validation checkpoint selection from 2-Stage SE(3) Lookahead.

### Table 4: Symmetric 3-Seed Zero-Shot Sim-to-Real PIL Robustness Stress Test — SPWM ($N=25$ Val-Sel.) vs. ANN-Rate vs. Equalized Deep GRU (`50` Test Episodes, $t \ge t_c$)

```latex
\begin{table}[t]
\centering
\caption{\textbf{Zero-Shot Physical \& Sensory Robustness Stress Test in PIL (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$).} Symmetric 3-seed comparison ($42, 101, 2024$) of frozen SPWM-v0.12.4 ($N=25$ validation-selected checkpoints: `42: phase\_4`, `101: phase\_1`, `2024: phase\_2`), parameter-matched ANN-Rate, and Equalized Deep GRU (Residual-Gated Online). Relative success retention ($\Delta_{\text{rel}}$) is computed against each model's Nominal Baseline.}
\label{tab:pil_robustness}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
\textbf{Perturbation Condition} & \multicolumn{3}{c}{\textbf{SPWM-v0.12.4 ($N=25$ Val-Sel., 3 Seeds)}} & \multicolumn{3}{c}{\textbf{ANN-Rate (3 Seeds)}} & \multicolumn{3}{c}{\textbf{Equalized Deep GRU (Residual-Gated, 3 Seeds)}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
 & \textbf{Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} & \textbf{Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} & \textbf{Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} \\
\midrule
Nominal Baseline & \textbf{60.7 $\pm$ 2.3} & \textbf{0.88 $\pm$ 0.11} & \textbf{2.01 $\pm$ 0.12} & 37.3 $\pm$ 9.5 & 1.07 $\pm$ 0.28 & 2.98 $\pm$ 0.46 & 76.7 $\pm$ 5.0 & 1.14 $\pm$ 0.31 & 1.90 $\pm$ 0.18 \\
Sensory Noise Injection (1.0\%) & \textbf{54.7 $\pm$ 1.2 (-9.9\% rel)} & \textbf{0.92 $\pm$ 0.24} & \textbf{2.26 $\pm$ 0.45} & 30.0 $\pm$ 3.5 (-19.6\% rel) & 1.04 $\pm$ 0.18 & 3.36 $\pm$ 0.53 & 74.7 $\pm$ 3.1 (-2.6\% rel) & 1.52 $\pm$ 0.30 & 1.85 $\pm$ 0.23 \\
Payload Extrapolation (4.80 kg) & \textbf{44.7 $\pm$ 13.3 (-26.4\% rel)} & \textbf{1.31 $\pm$ 0.41} & \textbf{2.49 $\pm$ 0.29} & 26.7 $\pm$ 16.2 (-28.6\% rel) & 1.36 $\pm$ 0.20 & 3.64 $\pm$ 0.80 & 23.3 $\pm$ 12.9 (-69.6\% rel) & 3.21 $\pm$ 1.47 & 4.35 $\pm$ 1.06 \\
Actuator Drift (0.80$\times$ Friction) & \textbf{52.7 $\pm$ 13.3 (-13.2\% rel)} & \textbf{1.31 $\pm$ 0.35} & \textbf{2.39 $\pm$ 0.26} & 30.7 $\pm$ 14.0 (-17.9\% rel) & 1.42 $\pm$ 0.22 & 3.74 $\pm$ 0.83 & 26.7 $\pm$ 23.9 (-65.2\% rel) & 3.24 $\pm$ 1.45 & 4.38 $\pm$ 1.00 \\
Actuator Drift (1.20$\times$ Friction) & \textbf{48.0 $\pm$ 19.3 (-20.9\% rel)} & \textbf{1.29 $\pm$ 0.46} & \textbf{2.28 $\pm$ 0.29} & 33.3 $\pm$ 16.7 (-10.7\% rel) & 1.34 $\pm$ 0.21 & 3.51 $\pm$ 1.21 & 30.0 $\pm$ 15.1 (-60.9\% rel) & 3.10 $\pm$ 1.34 & 4.19 $\pm$ 1.03 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

### Table 5: Standby Recurrent Core vs. Full End-to-End Closed-Loop Step Energy Audit

```latex
\begin{table}[t]
\centering
\caption{\textbf{Standby Recurrent State-Estimator Core vs. Full End-to-End Closed-Loop Step Energy.} Panel (a) isolates the standby state-estimation core when dense GPU planning is dormant (Loihi-2 SOPs at $1.5\,\text{pJ/SOP}$ excluding passive membrane leak vs. Jetson Orin GPU MACs at $0.9\,\text{pJ/MAC}$; 3-seed SPWM spike rate: $12.15 \pm 0.19\%$, $8,150 \pm 129$ SOPs/step). Panel (b) reports full end-to-end closed-loop step energy including the shared Conv2D + SpatialSoftmax encoder ($882{,}688$ MACs, $794.4\,\text{nJ}$) and $K=256 \times H=10 \times N_{\text{iter}}=3 = 7{,}680$ batched GPU MPPI rollouts ($1.219\,\text{GMACs}$ / $1.0971\,\text{mJ}$ for SPWM and ANN-Rate).}
\label{tab:energy_audit}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Model Architecture} & \textbf{Core Ops / Step} & \textbf{Core Energy (nJ)} & \textbf{Core Power @ 20Hz} & \textbf{Core vs. GPU} & \textbf{MPPI Rollout MACs} & \textbf{E2E Step Energy (mJ)} & \textbf{E2E Power @ 20Hz} \\
\midrule
\textbf{SPWM-v0.12.4 (Loihi-2 Standby Core + Orin, 3 seeds)} & \textbf{8,150 $\pm$ 129 SOPs} & \textbf{12.22 $\pm$ 0.19} & \textbf{0.244 $\pm$ 0.004 mW} & \textbf{4.94 $\pm$ 0.08$\times$} & 1.219 G & 1.0979 & 21.96 mW \\
ANN-Rate (Orin GPU) & 67,072 MACs & 60.36 & 1.207 mW & 1.00$\times$ (Ref) & 1.219 G & 1.0979 & 21.96 mW \\
Deep GRU (Orin GPU) & 299,520 MACs & 269.57 & 5.391 mW & 1.00$\times$ & 1.942 G & 1.7490 & 34.98 mW \\
Continuous RSSM (Orin GPU) & 365,056 MACs & 328.55 & 6.571 mW & 1.00$\times$ & 2.358 G & 2.1236 & 42.47 mW \\
Continuous ConvLSTM (Orin GPU) & 2,359,296 MACs & 2123.37 & 42.467 mW & 1.00$\times$ & 111.312 G & 100.1833 & 2003.67 mW \\
\bottomrule
\end{tabular}%
}
\end{table}
```

> **Recalibrated Energy Interpretation:**
> - **Primary Advantage Over `ANN-Rate`:** `SPWM`'s advantage over `ANN-Rate` is **closed-loop dynamical stability and noise/dynamics robustness** (`+23.4 pp` nominal 3-seed PIL grasp success, `+24.7 pp` and `2.0×` better relative retention under `1.0%` sensory noise with $N=25$ validation-selected checkpoints), **not** end-to-end closed-loop energy savings: saving `48.14 nJ/step` on Loihi-2 (`60.36 -> 12.22 nJ/step`) atop `1.0971 mJ/step` of dense GPU MPPI rollouts is a **`0.004%`** end-to-end difference (`1.0979 mJ/step` / `21.96 mW` @ `20 Hz` for both).
> - **Always-On Standby Benefit & `M1` Rollout Compactness:** The `4.94×` Loihi-2 recurrent core reduction (`12.22 ± 0.19 nJ/step`, `0.244 mW` @ `20 Hz`, computed from synaptic SOPs excluding passive membrane leak) is specifically an **always-on, sub-milliwatt standby state-estimation benefit** when batched MPPI planning is dormant. The `1.93×` E2E energy reduction vs. `Continuous RSSM` (`2.1236 mJ/step`) and `1.59×` reduction vs. `Deep GRU` (`1.7490 mJ/step`) stem from `M1`'s compact MLP rollout architecture (`1.219 GMACs` vs. `2.358 GMACs` and `1.942 GMACs`).

---

## 3. Residual-Gated Handover ($w_t > 0$) Telemetry & Validation Threshold Sweep

| Perturbation Condition | Seed 42 (Ep% / Step% / Mean $w$) | Seed 101 (Ep% / Step% / Mean $w$) | Seed 2024 (Ep% / Step% / Mean $w$) | 3-Seed Mean Step $w_t > 0$ |
| :--- | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |
| **Sensory Noise Injection (1.0%)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |
| **Payload Extrapolation (4.80 kg)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |
| **Actuator Drift (0.80x Friction)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |
| **Actuator Drift (1.20x Friction)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |

| Seed & Threshold $\tau$ | Nominal (Succ / Ep $w_t>0$) | 1.0% Noise (Succ / Ep $w_t>0$) | 4.80 kg Payload (Succ / Ep $w_t>0$) | 0.80$\times$ Friction (Succ / Ep $w_t>0$) | 1.20$\times$ Friction (Succ / Ep $w_t>0$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **seed42 ($\tau=0.026\,\text{m}$)** | `78.0% / 0.0%` | `78.0% / 0.0%` | `66.0% / 0.0%` | `70.0% / 0.0%` | `72.0% / 0.0%` |
| **seed42 ($\tau=0.020\,\text{m}$)** | `78.0% / 0.0%` | `78.0% / 0.0%` | `66.0% / 0.0%` | `70.0% / 0.0%` | `72.0% / 0.0%` |
| **seed42 ($\tau=0.015\,\text{m}$)** | `78.0% / 0.0%` | `78.0% / 0.0%` | `66.0% / 0.0%` | `70.0% / 0.0%` | `72.0% / 0.0%` |
| **seed2024 ($\tau=0.026\,\text{m}$)** | `46.0% / 0.0%` | `36.0% / 0.0%` | `42.0% / 0.0%` | `38.0% / 0.0%` | `38.0% / 0.0%` |
| **seed2024 ($\tau=0.020\,\text{m}$)** | `46.0% / 0.0%` | `36.0% / 0.0%` | `42.0% / 0.0%` | `38.0% / 0.0%` | `38.0% / 0.0%` |
| **seed2024 ($\tau=0.015\,\text{m}$)** | `46.0% / 0.0%` | `36.0% / 0.0%` | `42.0% / 0.0%` | `38.0% / 0.0%` | `38.0% / 0.0%` |

---

## 4. Diagnostic Proof: $N=6$ Tier-2 vs. $N=25$ Validation Checkpoint Selection (`[3200, 3224]` $\to$ `[3600, 3649]`)

| Seed | Checkpoint | Epoch | Val $N=25$ Succ (%) | Test $N=50$ Succ (%) | Test Min Err (cm) | Test Term Err (cm) | Selection Policy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Seed 42 | `best_model.pt` | `200` | `80.0%` | **86.0%** | 0.86 | 1.57 | N=6 Tier-2 |
| Seed 42 | `checkpoint_phase_1.pt` | `181` | `60.0%` | — | — | — | Val-Only |
| Seed 42 | `checkpoint_phase_2.pt` | `188` | `52.0%` | — | — | — | Val-Only |
| Seed 42 | `checkpoint_phase_3.pt` | `200` | `88.0%` | — | — | — | Val-Only |
| Seed 42 | `checkpoint_phase_4.pt` | `240` | `92.0%` | **58.0%** | 1.00 | 2.12 | N=25 Val-Sel. |
| Seed 101 | `best_model.pt` | `80` | `56.0%` | **56.0%** | 0.90 | 2.19 | N=6 Tier-2 |
| Seed 101 | `checkpoint_phase_1.pt` | `96` | `68.0%` | **62.0%** | 0.85 | 2.02 | N=25 Val-Sel. |
| Seed 101 | `checkpoint_phase_2.pt` | `104` | `52.0%` | — | — | — | Val-Only |
| Seed 101 | `checkpoint_phase_3.pt` | `124` | `4.0%` | — | — | — | Val-Only |
| Seed 101 | `checkpoint_phase_4.pt` | `240` | `40.0%` | — | — | — | Val-Only |
| Seed 2024 | `best_model.pt` | `130` | `48.0%` | **32.0%** | 0.94 | 2.93 | N=6 Tier-2 |
| Seed 2024 | `checkpoint_phase_1.pt` | `94` | `64.0%` | — | — | — | Val-Only |
| Seed 2024 | `checkpoint_phase_2.pt` | `102` | `80.0%` | **62.0%** | 0.79 | 1.89 | N=25 Val-Sel. |
| Seed 2024 | `checkpoint_phase_3.pt` | `122` | `40.0%` | — | — | — | Val-Only |
| Seed 2024 | `checkpoint_phase_4.pt` | `260` | `60.0%` | — | — | — | Val-Only |

- **Original `N=6` Tier-2 Selection (`[3200..3205]`):** `58.0 ± 27.1%` (`86.0%`, `56.0%`, `32.0%`)
- **Expanded `N=25` Validation Selection (`[3200..3224]`):** **`60.7 ± 2.3%`** (`58.0%`, `62.0%`, `62.0%`) — an **`11.7×` reduction in inter-seed test std** (`27.1% → 2.3%`) with higher mean success and zero test-set peeking.

---

## 5. SSOT & Zero-Leakage Verification Sign-Off

- **Zero Test-Time State Leakage:** All PIL benchmarks executed strictly under `ArticulatedArmPhysicsStepper` (`payload_attached=(t >= t_c)`) and causal `OnlineEventCameraStream` with zero `z_gt_next` access (`tests/test_leakage.py` and `tests/test_pil_integrity.py` PASSED).
- **Strict Validation/Test Isolation:** Handover threshold sweeps and `N=25` checkpoint selection used exclusively validation split `[3200, 3599]` trajectories; test evaluation used `[3600, 3999]`.
- **Synchronized Artifacts:**
  - SSOT Master Metrics: [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json)
  - Multi-Seed Summary: [multiseed_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/multiseed_metrics.json)
  - Forensic Decomposition: [forensic_rssm_gru_failure.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/forensic_rssm_gru_failure.json)
  - PIL Robustness (Seeds 42, 101, 2024): [robustness_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark/robustness_metrics.json), [seed101](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark_pil_spwm_seed101/robustness_metrics.json), [seed2024](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark_pil_spwm_seed2024/robustness_metrics.json)

### `results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json`
> Multi-seed N=400 open-loop baseline comparison (SPWM, ANN-Rate, Continuous RSSM, and equalized Deep GRU across 3 seeds + ConvLSTM)

```yaml
SPWM-v0.12.4:
  n_test: 400
  seeds: [42,101,2024]
  parameter_count: 392141
  per_seed:
    42:
      n_test: 400
      position_mse_per_horizon: {1:0.00015687845007050782,5:0.0020792167633771896,10:0.006488359998911619,25:0.002526428084820509,50:0.002635047771036625,100:0.0024581069592386484}
      position_mse_std_per_horizon: {1:7.380892202490941e-05,5:0.00048825654084794223,10:0.005912567488849163,25:0.0019791650120168924,50:0.007415277883410454,100:0.002898908220231533}
      velocity_mse_per_horizon: {1:0.0013571062590926886,5:0.05235695838928223,10:0.07470371574163437,25:0.03928713873028755,50:0.01999685727059841,100:0.011747471988201141}
      velocity_mse_std_per_horizon: {1:0.0005559915443882346,5:0.028355369344353676,10:0.13030092418193817,25:0.07005384564399719,50:0.03499726951122284,100:0.0203139279037714}
      position_error_per_horizon: {1:0.029773293063044548,5:0.11091844737529755,10:0.18621589243412018,25:0.11596996337175369,50:0.10377092659473419,100:0.11163192987442017}
      position_error_std_per_horizon: {1:0.007404169999063015,5:0.01313008926808834,10:0.06522116810083389,25:0.04134652391076088,50:0.07100620120763779,100:0.04782211408019066}
      overall_l2_mean: 0.11145036667585373
      curve_l2_mean: [0.029773293063044548,0.050943732261657715,0.0694747120141983,0.08839104324579239,0.11091842502355576,0.13780221343040466,0.16399258375167847,0.1815030425786972,0.18704365193843842,0.1862158328294754,0.18566682934761047,0.18405292928218842,0.17932210862636566,0.17349916696548462,0.1672763228416443,0.1608705073595047,0.15358270704746246,0.14490999281406403,0.1363014280796051,0.12909653782844543,0.12362727522850037,0.11972808837890625,0.11753828823566437,0.1165211871266365,0.1159699335694313,0.11564317345619202,0.11539418995380402,0.11538257449865341,0.11532190442085266,0.11522750556468964,0.11492013931274414,0.11434490978717804,0.11361837387084961,0.11278533190488815,0.11203490197658539,0.11129018664360046,0.110580675303936,0.11005494743585587,0.10947441309690475,0.10889815539121628,0.10819628089666367,0.10736408829689026,0.10660341382026672,0.10608332604169846,0.10566526651382446,0.10527226328849792,0.1049216017127037,0.10457690060138702,0.10416756570339203,0.1037709042429924,0.10346436500549316,0.10337063670158386,0.10360246896743774,0.10405252128839493,0.10464460402727127,0.1052316352725029,0.10584589093923569,0.10640781372785568,0.10695965588092804,0.10726359486579895,0.10728471726179123,0.10700049251317978,0.10646244883537292,0.10588253289461136,0.10546524077653885,0.10542401671409607,0.10532966256141663,0.10512605309486389,0.104812853038311,0.10426167398691177,0.10346218943595886,0.10215451568365097,0.10054737329483032,0.09872634708881378,0.09687754511833191,0.09513633698225021,0.09349039196968079,0.09200796484947205,0.09077796339988708,0.08987804502248764,0.08930744230747223,0.08895017951726913,0.08870480209589005,0.08853450417518616,0.08850858360528946,0.08873715996742249,0.08926960825920105,0.09014730155467987,0.09130968153476715,0.09264545142650604,0.09404744952917099,0.0954543873667717,0.09681116789579391,0.0981278121471405,0.09945985674858093,0.10092268884181976,0.10273744910955429,0.10508394986391068,0.10807770490646362,0.11163196712732315]
      curve_l2_q25: [0.02418188378214836,0.04567704722285271,0.06356161087751389,0.08150790631771088,0.10323502123355865,0.1262536346912384,0.14727847278118134,0.1584264487028122,0.15536509454250336,0.14956873655319214,0.1456349790096283,0.14184843003749847,0.1292610913515091,0.11995430290699005,0.11374010890722275,0.10681220144033432,0.09909381717443466,0.09388910979032516,0.09162658452987671,0.09065213799476624,0.08755099773406982,0.08756190538406372,0.08479989320039749,0.08799444139003754,0.09005386382341385,0.09020231664180756,0.09062648564577103,0.08981894701719284,0.09002521634101868,0.0868510901927948,0.08429233729839325,0.0836835652589798,0.0824483335018158,0.08040601015090942,0.08030581474304199,0.07910536229610443,0.07994438707828522,0.07748156785964966,0.07809378951787949,0.07631292939186096,0.07542622089385986,0.07479224354028702,0.07388124614953995,0.07398474216461182,0.07215038686990738,0.07094414532184601,0.07133544981479645,0.07061903178691864,0.06940420717000961,0.07008630037307739,0.07016012072563171,0.06934629380702972,0.0711706280708313,0.0707891657948494,0.07164794206619263,0.07330648601055145,0.0729539692401886,0.07322145253419876,0.07386495172977448,0.07504188269376755,0.07543125748634338,0.07677724957466125,0.07613738626241684,0.07753583788871765,0.07773175090551376,0.07676748931407928,0.07657727599143982,0.07582847028970718,0.07529428601264954,0.07495700567960739,0.07451340556144714,0.07442523539066315,0.07374760508537292,0.0731559544801712,0.0720655620098114,0.07118524610996246,0.06973353028297424,0.06915608048439026,0.06889288127422333,0.06734035909175873,0.06694740056991577,0.06701298803091049,0.06639609485864639,0.0661369189620018,0.06590918451547623,0.06688971817493439,0.06725241243839264,0.06829902529716492,0.06985847651958466,0.07033969461917877,0.07075491547584534,0.07271713018417358,0.07260537892580032,0.07261520624160767,0.07316749542951584,0.07275650650262833,0.07283641397953033,0.07333727180957794,0.07595354318618774,0.07957030832767487]
      curve_l2_q75: [0.034978218376636505,0.056788939982652664,0.07592301815748215,0.09600119292736053,0.11982263624668121,0.148398295044899,0.17623081803321838,0.1945151388645172,0.1995457112789154,0.1977941244840622,0.19582737982273102,0.19316044449806213,0.18706178665161133,0.18558591604232788,0.19795067608356476,0.2016545534133911,0.1934921145439148,0.17840851843357086,0.16444751620292664,0.15170827507972717,0.14338824152946472,0.1383947879076004,0.13898776471614838,0.1367250382900238,0.13474488258361816,0.13561207056045532,0.13406044244766235,0.13343654572963715,0.13474959135055542,0.13010606169700623,0.13032756745815277,0.13053463399410248,0.13128319382667542,0.1298547089099884,0.12724623084068298,0.12781772017478943,0.1257426142692566,0.12586036324501038,0.1256319135427475,0.12389302253723145,0.12436303496360779,0.12179112434387207,0.12037302553653717,0.11990264058113098,0.11713168025016785,0.11632630228996277,0.1137358546257019,0.11422622203826904,0.10990928113460541,0.11078011989593506,0.10951422154903412,0.11053813993930817,0.11167839169502258,0.11137203127145767,0.1111529991030693,0.11056137830018997,0.11067964881658554,0.11162015050649643,0.11224442720413208,0.11193743348121643,0.10881629586219788,0.10633069276809692,0.10497108101844788,0.1049499660730362,0.10366843640804291,0.10430885851383209,0.10363071411848068,0.10382680594921112,0.1041797548532486,0.10438516736030579,0.10449538379907608,0.10305618494749069,0.10159837454557419,0.10002902895212173,0.0980919897556305,0.09744185954332352,0.09610909968614578,0.09600912034511566,0.09620500355958939,0.09556256979703903,0.0965682864189148,0.09527552872896194,0.0955197885632515,0.09612056612968445,0.09667248278856277,0.09796690940856934,0.09943855553865433,0.10062514990568161,0.10258600115776062,0.10414493083953857,0.10640185326337814,0.10827697813510895,0.11016464978456497,0.11279815435409546,0.11581233143806458,0.1171855628490448,0.12172946333885193,0.1247154027223587,0.12768083810806274,0.13074742257595062]
      parameter_count: 392141
      active_inference_params: 317900
    101:
      n_test: 400
      position_mse_per_horizon: {1:0.0003022114688064903,5:0.0007039911579340696,10:0.0014849373837932944,25:0.002334464807063341,50:0.004773399326950312,100:0.0027366611175239086}
      position_mse_std_per_horizon: {1:0.0001610179024282843,5:0.000287990813376382,10:0.002266738098114729,25:0.0029588211327791214,50:0.008498414419591427,100:0.002922333311289549}
      velocity_mse_per_horizon: {1:0.0006355720688588917,5:0.028187384828925133,10:0.026028921827673912,25:0.05330432578921318,50:0.039045095443725586,100:0.008709450252354145}
      velocity_mse_std_per_horizon: {1:0.0003107062366325408,5:0.05764496326446533,10:0.033079687505960464,25:0.10820354521274567,50:0.04812313988804817,100:0.018667807802557945}
      position_error_per_horizon: {1:0.040972862392663956,5:0.06406641751527786,10:0.0811016857624054,25:0.10541189461946487,50:0.14444534480571747,100:0.11924396455287933}
      position_error_std_per_horizon: {1:0.011597124859690666,5:0.010928923264145851,10:0.048292238265275955,25:0.05380633845925331,50:0.088181272149086,100:0.04691314697265625}
      overall_l2_mean: 0.11936638504266739
      curve_l2_mean: [0.04097284749150276,0.048630427569150925,0.047982364892959595,0.05449967086315155,0.06406639516353607,0.06887251138687134,0.07037930935621262,0.07690698653459549,0.07720477133989334,0.0811016857624054,0.08710882067680359,0.08103204518556595,0.08067528903484344,0.081471748650074,0.08317293971776962,0.08891712874174118,0.09533265233039856,0.09921182692050934,0.10266103595495224,0.10465338826179504,0.10506565868854523,0.1043521985411644,0.10338154435157776,0.10340525954961777,0.10541191697120667,0.10955735296010971,0.11634419113397598,0.12366588413715363,0.1292857676744461,0.13517162203788757,0.14016516506671906,0.14309115707874298,0.1450527459383011,0.14636534452438354,0.14682316780090332,0.14616264402866364,0.14492224156856537,0.14388532936573029,0.14283306896686554,0.14163337647914886,0.14056296646595,0.14011681079864502,0.14047645032405853,0.14106836915016174,0.1416243016719818,0.14223746955394745,0.14306943118572235,0.1438082605600357,0.14404012262821198,0.14444535970687866,0.1458127498626709,0.14824706315994263,0.15004044771194458,0.15100634098052979,0.1520134061574936,0.15341654419898987,0.15506994724273682,0.1563708335161209,0.15682068467140198,0.15706342458724976,0.1571761965751648,0.15640702843666077,0.1547890156507492,0.15295815467834473,0.15144295990467072,0.1494150310754776,0.14654117822647095,0.14326901733875275,0.13878370821475983,0.13434533774852753,0.1299455463886261,0.12516261637210846,0.12098479270935059,0.11662372946739197,0.11230256408452988,0.10847455263137817,0.10494914650917053,0.10225875675678253,0.10037616640329361,0.09967388212680817,0.09985426068305969,0.10092516243457794,0.10239679366350174,0.10379341244697571,0.10518789291381836,0.10690902918577194,0.10894813388586044,0.1110227033495903,0.11295216530561447,0.11485613137483597,0.11689567565917969,0.11895854026079178,0.12072498351335526,0.12196436524391174,0.12254275381565094,0.12243806570768356,0.12179262936115265,0.12080536037683487,0.11980734020471573,0.11924394965171814]
      curve_l2_q25: [0.03250741586089134,0.04055394232273102,0.04250919073820114,0.05018458887934685,0.05861694738268852,0.058719247579574585,0.05162852257490158,0.0486564002931118,0.04813934117555618,0.05084918066859245,0.055994730442762375,0.05155554786324501,0.04870648682117462,0.045392654836177826,0.04391692206263542,0.04567359387874603,0.048919662833213806,0.04941921681165695,0.051012810319662094,0.0533885732293129,0.05757906287908554,0.06001395359635353,0.06400541961193085,0.06539051979780197,0.06877566874027252,0.0690845176577568,0.07007907330989838,0.07373254746198654,0.07377094030380249,0.07501916587352753,0.07605437934398651,0.07542125135660172,0.07391267269849777,0.07442323863506317,0.07400914281606674,0.07340028882026672,0.07606038451194763,0.07740136235952377,0.07699435204267502,0.0757865458726883,0.07583902776241302,0.07517863065004349,0.07720182836055756,0.07578183710575104,0.07565755397081375,0.0759425163269043,0.07919011265039444,0.08013825118541718,0.07950405776500702,0.08397497981786728,0.08391553163528442,0.08594615757465363,0.08345894515514374,0.08315222710371017,0.08268439769744873,0.08317984640598297,0.08041946589946747,0.08407500386238098,0.08513862639665604,0.08405416458845139,0.08434665203094482,0.08475447446107864,0.0862656980752945,0.08799862861633301,0.09035150706768036,0.09111933410167694,0.09074477851390839,0.0890524834394455,0.08796101808547974,0.08477328717708588,0.08169376850128174,0.079465851187706,0.07512563467025757,0.07250513881444931,0.07005419582128525,0.06847880780696869,0.0664040744304657,0.06497754156589508,0.06402590125799179,0.0639352798461914,0.064520463347435,0.0654643103480339,0.06696335971355438,0.06855316460132599,0.070815809071064,0.07328331470489502,0.07625096291303635,0.07955586910247803,0.08203928172588348,0.08419544994831085,0.08756040036678314,0.0906951054930687,0.09385237097740173,0.09627972543239594,0.09668398648500443,0.0961935967206955,0.09505638480186462,0.09497466683387756,0.09328244626522064,0.09291389584541321]
      curve_l2_q75: [0.04972991719841957,0.055236607789993286,0.05272743105888367,0.05821100249886513,0.06604094803333282,0.06757736206054688,0.0681806206703186,0.08742452412843704,0.08808335661888123,0.09339259564876556,0.10288523882627487,0.0876614972949028,0.09538991004228592,0.10073699057102203,0.11356557905673981,0.12679846584796906,0.13077262043952942,0.13569065928459167,0.13583268225193024,0.13401539623737335,0.13294798135757446,0.12979772686958313,0.12725688517093658,0.12722760438919067,0.12823274731636047,0.12851397693157196,0.1339235007762909,0.1446610391139984,0.15282614529132843,0.16387127339839935,0.16639375686645508,0.17378027737140656,0.17729125916957855,0.18343006074428558,0.18618042767047882,0.18899351358413696,0.1872260868549347,0.1906617134809494,0.19132116436958313,0.18861909210681915,0.1857784390449524,0.18289516866207123,0.1823655366897583,0.18026268482208252,0.1835450977087021,0.1844027042388916,0.18729227781295776,0.18866221606731415,0.1862790286540985,0.18590377271175385,0.1875440925359726,0.19348514080047607,0.19832979142665863,0.19723553955554962,0.20282167196273804,0.20019854605197906,0.20026902854442596,0.20564404129981995,0.20788685977458954,0.20667684078216553,0.20630168914794922,0.2012256681919098,0.19432947039604187,0.18807673454284668,0.1848210245370865,0.17601263523101807,0.1737760752439499,0.17421954870224,0.1673009693622589,0.15903528034687042,0.15870288014411926,0.1522885113954544,0.1486351191997528,0.14621798694133759,0.13646458089351654,0.13204611837863922,0.12527677416801453,0.11945419758558273,0.11535979807376862,0.11605769395828247,0.1163351833820343,0.11956220865249634,0.12070093303918839,0.12392379343509674,0.12555302679538727,0.12459474802017212,0.1257094442844391,0.12704122066497803,0.12898148596286774,0.13082726299762726,0.13104265928268433,0.13057675957679749,0.13317380845546722,0.13443408906459808,0.13617682456970215,0.13559560477733612,0.13440343737602234,0.13353140652179718,0.13167308270931244,0.13232040405273438]
      parameter_count: 392141
      active_inference_params: 317900
    2024:
      n_test: 400
      position_mse_per_horizon: {1:8.75631085364148e-05,5:0.0006210815627127886,10:0.002461647614836693,25:0.0017300095641985536,50:0.002026821719482541,100:0.002009366638958454}
      position_mse_std_per_horizon: {1:7.439988985424861e-05,5:0.00025316045503132045,10:0.004423991311341524,25:0.0014237076975405216,50:0.00675568263977766,100:0.0028823448810726404}
      velocity_mse_per_horizon: {1:0.0021625349763780832,5:0.04693954437971115,10:0.034473713487386703,25:0.01722436025738716,50:0.01493775099515915,100:0.008667011745274067}
      velocity_mse_std_per_horizon: {1:0.0015680588549003005,5:0.02672167681157589,10:0.05292689800262451,25:0.06946571916341782,50:0.03426194563508034,100:0.018376773223280907}
      position_error_per_horizon: {1:0.021432053297758102,5:0.06007310748100281,10:0.098426952958107,25:0.09677165001630783,50:0.08652535080909729,100:0.09818258136510849}
      position_error_std_per_horizon: {1:0.008126846514642239,5:0.010849472135305405,10:0.07128828018903732,25:0.03186384588479996,50:0.0683688148856163,100:0.04915667697787285}
      overall_l2_mean: 0.08790452033281326
      curve_l2_mean: [0.021432064473628998,0.030940957367420197,0.04192980378866196,0.05101213976740837,0.0600731186568737,0.07445144653320312,0.09110533446073532,0.09926898032426834,0.09944499284029007,0.09842698276042938,0.09915661066770554,0.09976634383201599,0.09958755224943161,0.09879253059625626,0.09671840816736221,0.09401373565196991,0.09217559546232224,0.09151539206504822,0.09148741513490677,0.09196250885725021,0.09294717758893967,0.09422244876623154,0.09540285915136337,0.0962909385561943,0.09677167981863022,0.09695729613304138,0.09695394337177277,0.09702262282371521,0.09717518836259842,0.09744575619697571,0.09747783839702606,0.09711499512195587,0.09649709612131119,0.09585520625114441,0.09538117051124573,0.09500434994697571,0.09477697312831879,0.09461123496294022,0.09427622705698013,0.09383954852819443,0.09316999465227127,0.09233413636684418,0.09143620729446411,0.09080412238836288,0.09026128053665161,0.08971187472343445,0.08907637745141983,0.08834093809127808,0.08747363835573196,0.08652535080909729,0.0856902226805687,0.08500228077173233,0.08460228890180588,0.08424986153841019,0.08400610089302063,0.08378324657678604,0.08370611071586609,0.08365900814533234,0.08371409773826599,0.08366451412439346,0.08346258103847504,0.08313390612602234,0.08270569145679474,0.08238270878791809,0.08232539892196655,0.08274868875741959,0.08319231867790222,0.08351732045412064,0.08372201770544052,0.08374352753162384,0.08366478979587555,0.08329729735851288,0.08289235085248947,0.08256752789020538,0.08248627930879593,0.0827319547533989,0.08321595191955566,0.08391588181257248,0.08484233170747757,0.08598781377077103,0.08727657049894333,0.08855776488780975,0.08966568857431412,0.09047755599021912,0.09099116176366806,0.0912979394197464,0.0914657711982727,0.09160096943378448,0.0917380154132843,0.09187410026788712,0.09199133515357971,0.09213229268789291,0.09232395142316818,0.09258327633142471,0.09291087090969086,0.09332364052534103,0.09393181651830673,0.09485320001840591,0.09623980522155762,0.09818253666162491]
      curve_l2_q25: [0.015498905442655087,0.024717820808291435,0.03585927188396454,0.04572675749659538,0.055315688252449036,0.060900501906871796,0.06561881303787231,0.06548632681369781,0.061938464641571045,0.05969446897506714,0.060987070202827454,0.062407661229372025,0.06362974643707275,0.06525819003582001,0.06328015774488449,0.06395599246025085,0.06512613594532013,0.06651968508958817,0.06859006732702255,0.06941583752632141,0.0701713114976883,0.07145524024963379,0.07311172038316727,0.07417897135019302,0.0747128576040268,0.07593070715665817,0.0762840211391449,0.07594531029462814,0.07410571724176407,0.07332514226436615,0.07134845852851868,0.0698552280664444,0.06906865537166595,0.06497547030448914,0.06421426683664322,0.06580201536417007,0.06471775472164154,0.06498667597770691,0.06523842364549637,0.0650990903377533,0.06463169306516647,0.06219343841075897,0.05959281325340271,0.05814574286341667,0.05881882458925247,0.05782599374651909,0.05703023076057434,0.05717135965824127,0.055388133972883224,0.05501432716846466,0.05438612401485443,0.05238000303506851,0.05132746696472168,0.05125906318426132,0.04918723553419113,0.04834024980664253,0.04916540905833244,0.04817953333258629,0.04933788999915123,0.04914432018995285,0.04931003600358963,0.049756016582250595,0.05065782740712166,0.05023585259914398,0.050400298088788986,0.05074867606163025,0.051390696316957474,0.052359793335199356,0.05208117887377739,0.05227772146463394,0.05298853665590286,0.053046878427267075,0.05371962487697601,0.05401007831096649,0.05450211465358734,0.05580129474401474,0.05741836875677109,0.058268509805202484,0.060341499745845795,0.062326256185770035,0.06388892233371735,0.06532284617424011,0.06664977967739105,0.06723639369010925,0.06871815770864487,0.06945175677537918,0.07024890184402466,0.06999903917312622,0.07048308849334717,0.06962516903877258,0.06859228014945984,0.06802499294281006,0.06770095229148865,0.06731553375720978,0.06724008917808533,0.06667717546224594,0.06554042547941208,0.06491898000240326,0.06515618413686752,0.06684660911560059]
      curve_l2_q75: [0.025374628603458405,0.035816702991724014,0.046994227916002274,0.05611596256494522,0.06425561010837555,0.07375462353229523,0.0844850242137909,0.09341475367546082,0.0999329537153244,0.09425720572471619,0.1024552658200264,0.10926555842161179,0.11206367611885071,0.11415525525808334,0.11671680957078934,0.11369575560092926,0.11022305488586426,0.1099800169467926,0.10797414928674698,0.10985719412565231,0.11095203459262848,0.11338196694850922,0.11433461308479309,0.11581109464168549,0.11546257138252258,0.11540766060352325,0.11346930265426636,0.11370491236448288,0.11390945315361023,0.11412681639194489,0.11396092176437378,0.11360447853803635,0.11389864981174469,0.1147192120552063,0.11358556896448135,0.11375358700752258,0.11323516070842743,0.11264461278915405,0.11189301311969757,0.10911186039447784,0.10852266848087311,0.10598836839199066,0.10590509325265884,0.10370621085166931,0.1049627959728241,0.1033305749297142,0.10177259147167206,0.0998019203543663,0.10002736747264862,0.10047617554664612,0.0982658863067627,0.09711238741874695,0.09525200724601746,0.09489644318819046,0.0953739732503891,0.09288013726472855,0.09306342899799347,0.09274895489215851,0.09304951876401901,0.09309489279985428,0.09026522189378738,0.08927927911281586,0.08745555579662323,0.08787752687931061,0.08589904755353928,0.08505725115537643,0.08592118322849274,0.08730357885360718,0.08848626911640167,0.08881126344203949,0.08844342082738876,0.08931232243776321,0.08962893486022949,0.08977437019348145,0.09023341536521912,0.09099763631820679,0.09002768248319626,0.08994069695472717,0.09178996086120605,0.09326902776956558,0.09495523571968079,0.09548457711935043,0.09682148694992065,0.09828221052885056,0.09971140325069427,0.09999817609786987,0.09989237785339355,0.0994998961687088,0.09935472160577774,0.1019904613494873,0.10362684726715088,0.10517384856939316,0.10777708888053894,0.11229884624481201,0.11219097673892975,0.11277663707733154,0.11151605099439621,0.11315646767616272,0.11366723477840424,0.11610586196184158]
      parameter_count: 392141
      active_inference_params: 317900
  position_mse_per_horizon: {1:0.00018221767580447099,5:0.0011347631613413494,10:0.0034783149991805353,25:0.0021969674853608012,50:0.0031450896058231592,100:0.0024013782385736704}
  position_mse_std_per_horizon: {1:0.0001030755714358141,5:0.0003431359364185482,10:0.004201098966101806,25:0.0021205646141121783,50:0.007556458314259847,100:0.002901195470864574}
  position_mse_seed_std_per_horizon: {1:0.00010954467998152894,5:0.0008189706679996508,10:0.0026521253541566974,25:0.0004156317481905645,50:0.001442577257336728,100:0.000366950848819243}
  velocity_mse_per_horizon: {1:0.001385071101443221,5:0.04249462919930617,10:0.045068783685564995,25:0.036605274925629296,50:0.02465990123649438,100:0.009707977995276451}
  velocity_mse_std_per_horizon: {1:0.0008115855453070253,5:0.03757400314013163,10:0.07210250323017438,25:0.08257437000672023,50:0.03912745167811712,100:0.01911950297653675}
  velocity_mse_seed_std_per_horizon: {1:0.000763865468875661,5:0.012683060543225525,10:0.026009623700082144,25:0.01818887773598169,50:0.012712159468845796,100:0.0017663810650883683}
  position_error_per_horizon: {1:0.03072606958448887,5:0.07835265745719273,10:0.12191484371821086,25:0.10605116933584213,50:0.11158054073651631,100:0.10968615859746933}
  position_error_std_per_horizon: {1:0.009042713791131973,5:0.011636161555846533,10:0.06160056218504906,25:0.042338902751604714,50:0.07585209608078003,100:0.04796397934357325}
  position_error_seed_std_per_horizon: {1:0.00980518446260704,5:0.028273390953333897,10:0.056356098785534485,25:0.009615108582107753,50:0.029739266650756064,100:0.010664660559163815}
  active_inference_params: 317900
  overall_l2_mean: 0.10624042401711146
  overall_l2_seed_std: 0.016365202706835866
  curve_l2_mean: [0.029773293063044548,0.050943732261657715,0.0694747120141983,0.08839104324579239,0.11091842502355576,0.13780221343040466,0.16399258375167847,0.1815030425786972,0.18704365193843842,0.1862158328294754,0.18566682934761047,0.18405292928218842,0.17932210862636566,0.17349916696548462,0.1672763228416443,0.1608705073595047,0.15358270704746246,0.14490999281406403,0.1363014280796051,0.12909653782844543,0.12362727522850037,0.11972808837890625,0.11753828823566437,0.1165211871266365,0.1159699335694313,0.11564317345619202,0.11539418995380402,0.11538257449865341,0.11532190442085266,0.11522750556468964,0.11492013931274414,0.11434490978717804,0.11361837387084961,0.11278533190488815,0.11203490197658539,0.11129018664360046,0.110580675303936,0.11005494743585587,0.10947441309690475,0.10889815539121628,0.10819628089666367,0.10736408829689026,0.10660341382026672,0.10608332604169846,0.10566526651382446,0.10527226328849792,0.1049216017127037,0.10457690060138702,0.10416756570339203,0.1037709042429924,0.10346436500549316,0.10337063670158386,0.10360246896743774,0.10405252128839493,0.10464460402727127,0.1052316352725029,0.10584589093923569,0.10640781372785568,0.10695965588092804,0.10726359486579895,0.10728471726179123,0.10700049251317978,0.10646244883537292,0.10588253289461136,0.10546524077653885,0.10542401671409607,0.10532966256141663,0.10512605309486389,0.104812853038311,0.10426167398691177,0.10346218943595886,0.10215451568365097,0.10054737329483032,0.09872634708881378,0.09687754511833191,0.09513633698225021,0.09349039196968079,0.09200796484947205,0.09077796339988708,0.08987804502248764,0.08930744230747223,0.08895017951726913,0.08870480209589005,0.08853450417518616,0.08850858360528946,0.08873715996742249,0.08926960825920105,0.09014730155467987,0.09130968153476715,0.09264545142650604,0.09404744952917099,0.0954543873667717,0.09681116789579391,0.0981278121471405,0.09945985674858093,0.10092268884181976,0.10273744910955429,0.10508394986391068,0.10807770490646362,0.11163196712732315]
  curve_l2_q25: [0.02418188378214836,0.04567704722285271,0.06356161087751389,0.08150790631771088,0.10323502123355865,0.1262536346912384,0.14727847278118134,0.1584264487028122,0.15536509454250336,0.14956873655319214,0.1456349790096283,0.14184843003749847,0.1292610913515091,0.11995430290699005,0.11374010890722275,0.10681220144033432,0.09909381717443466,0.09388910979032516,0.09162658452987671,0.09065213799476624,0.08755099773406982,0.08756190538406372,0.08479989320039749,0.08799444139003754,0.09005386382341385,0.09020231664180756,0.09062648564577103,0.08981894701719284,0.09002521634101868,0.0868510901927948,0.08429233729839325,0.0836835652589798,0.0824483335018158,0.08040601015090942,0.08030581474304199,0.07910536229610443,0.07994438707828522,0.07748156785964966,0.07809378951787949,0.07631292939186096,0.07542622089385986,0.07479224354028702,0.07388124614953995,0.07398474216461182,0.07215038686990738,0.07094414532184601,0.07133544981479645,0.07061903178691864,0.06940420717000961,0.07008630037307739,0.07016012072563171,0.06934629380702972,0.0711706280708313,0.0707891657948494,0.07164794206619263,0.07330648601055145,0.0729539692401886,0.07322145253419876,0.07386495172977448,0.07504188269376755,0.07543125748634338,0.07677724957466125,0.07613738626241684,0.07753583788871765,0.07773175090551376,0.07676748931407928,0.07657727599143982,0.07582847028970718,0.07529428601264954,0.07495700567960739,0.07451340556144714,0.07442523539066315,0.07374760508537292,0.0731559544801712,0.0720655620098114,0.07118524610996246,0.06973353028297424,0.06915608048439026,0.06889288127422333,0.06734035909175873,0.06694740056991577,0.06701298803091049,0.06639609485864639,0.0661369189620018,0.06590918451547623,0.06688971817493439,0.06725241243839264,0.06829902529716492,0.06985847651958466,0.07033969461917877,0.07075491547584534,0.07271713018417358,0.07260537892580032,0.07261520624160767,0.07316749542951584,0.07275650650262833,0.07283641397953033,0.07333727180957794,0.07595354318618774,0.07957030832767487]
  curve_l2_q75: [0.034978218376636505,0.056788939982652664,0.07592301815748215,0.09600119292736053,0.11982263624668121,0.148398295044899,0.17623081803321838,0.1945151388645172,0.1995457112789154,0.1977941244840622,0.19582737982273102,0.19316044449806213,0.18706178665161133,0.18558591604232788,0.19795067608356476,0.2016545534133911,0.1934921145439148,0.17840851843357086,0.16444751620292664,0.15170827507972717,0.14338824152946472,0.1383947879076004,0.13898776471614838,0.1367250382900238,0.13474488258361816,0.13561207056045532,0.13406044244766235,0.13343654572963715,0.13474959135055542,0.13010606169700623,0.13032756745815277,0.13053463399410248,0.13128319382667542,0.1298547089099884,0.12724623084068298,0.12781772017478943,0.1257426142692566,0.12586036324501038,0.1256319135427475,0.12389302253723145,0.12436303496360779,0.12179112434387207,0.12037302553653717,0.11990264058113098,0.11713168025016785,0.11632630228996277,0.1137358546257019,0.11422622203826904,0.10990928113460541,0.11078011989593506,0.10951422154903412,0.11053813993930817,0.11167839169502258,0.11137203127145767,0.1111529991030693,0.11056137830018997,0.11067964881658554,0.11162015050649643,0.11224442720413208,0.11193743348121643,0.10881629586219788,0.10633069276809692,0.10497108101844788,0.1049499660730362,0.10366843640804291,0.10430885851383209,0.10363071411848068,0.10382680594921112,0.1041797548532486,0.10438516736030579,0.10449538379907608,0.10305618494749069,0.10159837454557419,0.10002902895212173,0.0980919897556305,0.09744185954332352,0.09610909968614578,0.09600912034511566,0.09620500355958939,0.09556256979703903,0.0965682864189148,0.09527552872896194,0.0955197885632515,0.09612056612968445,0.09667248278856277,0.09796690940856934,0.09943855553865433,0.10062514990568161,0.10258600115776062,0.10414493083953857,0.10640185326337814,0.10827697813510895,0.11016464978456497,0.11279815435409546,0.11581233143806458,0.1171855628490448,0.12172946333885193,0.1247154027223587,0.12768083810806274,0.13074742257595062]
ANN-Rate:
  n_test: 400
  seeds: [42,101,2024]
  parameter_count: 392141
  per_seed:
    42:
      n_test: 400
      position_mse_per_horizon: {1:0.00013674570072907954,5:0.0011123945005238056,10:0.002159136813133955,25:0.0011364539386704564,50:0.001849682186730206,100:0.001811641501262784}
      position_mse_std_per_horizon: {1:3.5051074519287795e-05,5:0.0004620733961928636,10:0.00108297448605299,25:0.001763601670973003,50:0.0071600633673369884,100:0.0028153385501354933}
      velocity_mse_per_horizon: {1:0.0024848710745573044,5:0.025722002610564232,10:0.02801252342760563,25:0.017590845003724098,50:0.014915362931787968,100:0.009254482574760914}
      velocity_mse_std_per_horizon: {1:0.0003200853243470192,5:0.06381455063819885,10:0.04584721103310585,25:0.06683516502380371,50:0.03246651217341423,100:0.01921427622437477}
      position_error_per_horizon: {1:0.028403006494045258,5:0.0790843591094017,10:0.10914812982082367,25:0.07186909765005112,50:0.07561712712049484,100:0.09217879921197891}
      position_error_std_per_horizon: {1:0.0037072135601192713,5:0.020494651049375534,10:0.03227239474654198,25:0.0406639501452446,50:0.07334945350885391,100:0.0487126000225544}
      overall_l2_mean: 0.07817552983760834
      curve_l2_mean: [0.028402993455529213,0.05409229174256325,0.06919767707586288,0.07416556030511856,0.0790843591094017,0.0878472700715065,0.09656296670436859,0.10067830234766006,0.1020779237151146,0.10914812982082367,0.11117352545261383,0.10601089149713516,0.10027242451906204,0.09265705198049545,0.08792710304260254,0.08330740034580231,0.07834500819444656,0.07513298094272614,0.07268580794334412,0.07121405750513077,0.07048943638801575,0.0699843242764473,0.07024625688791275,0.07103260606527328,0.07186911255121231,0.07283908873796463,0.07377299666404724,0.07496389746665955,0.07631167769432068,0.07760873436927795,0.07856328785419464,0.07912985235452652,0.07931079715490341,0.07924097031354904,0.07928299158811569,0.07933448255062103,0.07926910370588303,0.07924263179302216,0.07901707291603088,0.07876311987638474,0.07827514410018921,0.07765324413776398,0.07717940956354141,0.07693558931350708,0.07675628364086151,0.07660053670406342,0.07645636796951294,0.07628099620342255,0.07592596858739853,0.07561714947223663,0.07536737620830536,0.07527463883161545,0.07539671659469604,0.07562102377414703,0.07572361081838608,0.07570267468690872,0.07563478499650955,0.07550754398107529,0.07540243119001389,0.0750468298792839,0.07449661940336227,0.07387328147888184,0.07321488112211227,0.0727459192276001,0.07271791994571686,0.07326419651508331,0.0738634467124939,0.07443109154701233,0.07493837177753448,0.07527387142181396,0.07547730207443237,0.07534337788820267,0.07509634643793106,0.07475726306438446,0.07443652302026749,0.07422575354576111,0.07407048344612122,0.07402627915143967,0.07415244728326797,0.07442715018987656,0.07473479956388474,0.07499084621667862,0.07517151534557343,0.07530569285154343,0.075480155646801,0.0758022665977478,0.07630369812250137,0.07703898847103119,0.07799171656370163,0.0790792927145958,0.08018292486667633,0.08123725652694702,0.08216797560453415,0.08296094834804535,0.08365557342767715,0.08437445759773254,0.08537574857473373,0.08690201491117477,0.08917172253131866,0.09217877686023712]
      curve_l2_q25: [0.025688745081424713,0.0517529658973217,0.06582773476839066,0.06946137547492981,0.06982161849737167,0.07347680628299713,0.07758424431085587,0.07654827833175659,0.07615315169095993,0.0839645117521286,0.0856681764125824,0.08373668789863586,0.07903014123439789,0.06812075525522232,0.06539769470691681,0.060212716460227966,0.05562014505267143,0.05183684080839157,0.04982618987560272,0.04733149707317352,0.0451623797416687,0.04393559694290161,0.043220147490501404,0.04377090558409691,0.04441813752055168,0.045465223491191864,0.04610028490424156,0.04609829932451248,0.0469626858830452,0.047558873891830444,0.04737984389066696,0.04693424329161644,0.04714300483465195,0.047332197427749634,0.04776688665151596,0.04760952293872833,0.04753852263092995,0.047342609614133835,0.04805628955364227,0.047483399510383606,0.047179535031318665,0.046427953988313675,0.044934358447790146,0.044367603957653046,0.04315820336341858,0.04316725581884384,0.041933320462703705,0.04202969744801521,0.04313062131404877,0.04104733094573021,0.041153665632009506,0.04191537946462631,0.04235341399908066,0.042520519345998764,0.04089833050966263,0.04149852320551872,0.04167407751083374,0.041788697242736816,0.041652269661426544,0.04189186170697212,0.041234664618968964,0.04103300720453262,0.041546013206243515,0.04106965661048889,0.04186272248625755,0.04361831396818161,0.044556453824043274,0.044056959450244904,0.04459691420197487,0.04600435122847557,0.04677724093198776,0.04649851843714714,0.04706789553165436,0.04737219586968422,0.048247821629047394,0.04891939461231232,0.05002482980489731,0.05077916011214256,0.051292967051267624,0.053736940026283264,0.053761158138513565,0.05310306325554848,0.05243781954050064,0.05203860253095627,0.05201371759176254,0.051209457218647,0.05156080424785614,0.05117755010724068,0.05187275633215904,0.05312029644846916,0.054139476269483566,0.05515177547931671,0.05655680224299431,0.05725458264350891,0.05733560398221016,0.057014767080545425,0.05715153366327286,0.058652881532907486,0.05937575921416283,0.06200525537133217]
      curve_l2_q75: [0.030848084017634392,0.0568077489733696,0.07301976531744003,0.08181136846542358,0.09329673647880554,0.10683343559503555,0.11842299997806549,0.12626716494560242,0.1294131577014923,0.13687501847743988,0.13900557160377502,0.1323055922985077,0.12427413463592529,0.11480254679918289,0.10740269720554352,0.10009858012199402,0.09046373516321182,0.08680496364831924,0.0834229364991188,0.0830121785402298,0.08478143066167831,0.08581989258527756,0.08464576303958893,0.08663273602724075,0.08694306761026382,0.08552728593349457,0.08661364018917084,0.09050130844116211,0.09363279491662979,0.0964195504784584,0.09766033291816711,0.09766631573438644,0.09828822314739227,0.09566336870193481,0.09244328737258911,0.09387047588825226,0.08968652784824371,0.09100281447172165,0.09253418445587158,0.09094621241092682,0.0890679806470871,0.08840598165988922,0.08570457249879837,0.08603780716657639,0.08672039210796356,0.08579102903604507,0.08533994853496552,0.0850599855184555,0.08378507196903229,0.08354295045137405,0.08368957787752151,0.08443000912666321,0.08500698208808899,0.08575280010700226,0.08399265259504318,0.08330803364515305,0.08047395944595337,0.08058642596006393,0.08071506023406982,0.07909595966339111,0.07704782485961914,0.07479239255189896,0.07399509847164154,0.07332340627908707,0.07415328919887543,0.0736687034368515,0.07459498941898346,0.07657839357852936,0.07800287008285522,0.07916605472564697,0.07914764434099197,0.08006235212087631,0.08028841018676758,0.08072361350059509,0.07881306111812592,0.07789904624223709,0.07924868166446686,0.07942206412553787,0.07897322624921799,0.07970607280731201,0.08017571270465851,0.08156496286392212,0.08328864723443985,0.08187296986579895,0.08239028602838516,0.08291680365800858,0.08383144438266754,0.08661171793937683,0.08856142312288284,0.08987045288085938,0.09065719693899155,0.0910019725561142,0.0916810929775238,0.09279067814350128,0.0929693877696991,0.09498428553342819,0.09680746495723724,0.09994266927242279,0.10333787649869919,0.10428255796432495]
      parameter_count: 392141
      active_inference_params: 317900
    101:
      n_test: 400
      position_mse_per_horizon: {1:0.0003403051814530045,5:0.0033553973771631718,10:0.0025252627674490213,25:0.0046305786818265915,50:0.006457539740949869,100:0.002824246184900403}
      position_mse_std_per_horizon: {1:6.417715485440567e-05,5:0.0010157969081774354,10:0.0018206920940428972,25:0.0061861989088356495,50:0.009928937070071697,100:0.00308819068595767}
      velocity_mse_per_horizon: {1:0.002158749382942915,5:0.03554815426468849,10:0.03806614503264427,25:0.033490099012851715,50:0.043996404856443405,100:0.019914178177714348}
      velocity_mse_std_per_horizon: {1:0.00024094675609376281,5:0.03244440630078316,10:0.05103874206542969,25:0.08034590631723404,50:0.057198453694581985,100:0.022594958543777466}
      position_error_per_horizon: {1:0.044996242970228195,5:0.13943637907505035,10:0.11689078062772751,25:0.14385952055454254,50:0.1678421050310135,100:0.11790895462036133}
      position_error_std_per_horizon: {1:0.00414357939735055,5:0.02626550942659378,10:0.03857618197798729,25:0.08418971300125122,50:0.10283125936985016,100:0.05516299232840538}
      overall_l2_mean: 0.14079783856868744
      curve_l2_mean: [0.044996246695518494,0.09009227901697159,0.12582136690616608,0.1365877240896225,0.13943630456924438,0.14295902848243713,0.13923326134681702,0.13789193332195282,0.13515906035900116,0.11689074337482452,0.11509647965431213,0.12678176164627075,0.13809238374233246,0.1437377631664276,0.1405201554298401,0.1362462192773819,0.12744727730751038,0.12263883650302887,0.12350621819496155,0.12647409737110138,0.129033163189888,0.1320408582687378,0.13506437838077545,0.1386549025774002,0.14385947585105896,0.14965493977069855,0.1543615311384201,0.1580079048871994,0.16130952537059784,0.16489218175411224,0.1676536351442337,0.16934062540531158,0.17075413465499878,0.17130722105503082,0.17090953886508942,0.16977867484092712,0.16859449446201324,0.1681908816099167,0.1683841496706009,0.1684175729751587,0.16822969913482666,0.16813020408153534,0.16805244982242584,0.16835695505142212,0.16878440976142883,0.16887466609477997,0.16902346909046173,0.1691710650920868,0.16875997185707092,0.1678421050310135,0.1665489822626114,0.16499115526676178,0.16350874304771423,0.16195523738861084,0.15984441339969635,0.15752235054969788,0.15596981346607208,0.15531615912914276,0.15493786334991455,0.15434163808822632,0.1535956710577011,0.15354543924331665,0.15476709604263306,0.15664084255695343,0.15826231241226196,0.15964919328689575,0.16085542738437653,0.16188187897205353,0.16186374425888062,0.16063641011714935,0.15915024280548096,0.15728072822093964,0.15430542826652527,0.1503312736749649,0.14586743712425232,0.14097055792808533,0.1356242150068283,0.1305416077375412,0.12553788721561432,0.12070140242576599,0.11664393544197083,0.11332474648952484,0.1103086769580841,0.10747028142213821,0.10516456514596939,0.10330644249916077,0.10173887014389038,0.10055416822433472,0.09984520822763443,0.09961187094449997,0.09983135014772415,0.10055682808160782,0.10174404084682465,0.1032819151878357,0.10502516478300095,0.10689015686511993,0.10895559936761856,0.11135158687829971,0.11427972465753555,0.11790889501571655]
      curve_l2_q25: [0.042230941355228424,0.0886874794960022,0.12429475784301758,0.13392163813114166,0.13294613361358643,0.12569496035575867,0.11720659583806992,0.11218549311161041,0.10737130045890808,0.09014671295881271,0.08317360281944275,0.0862022414803505,0.09352557361125946,0.09582576900720596,0.08978646993637085,0.08756929636001587,0.0842319130897522,0.08360063284635544,0.08419688045978546,0.08526439964771271,0.08692747354507446,0.08910037577152252,0.09075013548135757,0.09159624576568604,0.09219774603843689,0.09146899729967117,0.08866826444864273,0.0891435295343399,0.0862768217921257,0.08424358069896698,0.08517336845397949,0.08703143894672394,0.08744355291128159,0.08504471182823181,0.0875864028930664,0.08921733498573303,0.09038302302360535,0.09120875597000122,0.09373091161251068,0.09592925012111664,0.09934984892606735,0.10187219083309174,0.1044459268450737,0.10392707586288452,0.10350161790847778,0.10442278534173965,0.10521921515464783,0.10194255411624908,0.10339448601007462,0.10275362432003021,0.1035756915807724,0.10156652331352234,0.10264202207326889,0.10169702768325806,0.10252727568149567,0.10072600841522217,0.10157205164432526,0.10351119935512543,0.10590684413909912,0.10867023468017578,0.10711356997489929,0.10826641321182251,0.1092955619096756,0.10959033668041229,0.10979266464710236,0.11063317954540253,0.10699643939733505,0.10734963417053223,0.10662618279457092,0.10208214819431305,0.0973905697464943,0.09295889735221863,0.08805093169212341,0.08149166405200958,0.07834170013666153,0.07615483552217484,0.0764310285449028,0.0760079100728035,0.07590123265981674,0.07574083656072617,0.07478676736354828,0.07515350729227066,0.07348845154047012,0.07164742052555084,0.06885208934545517,0.0664033517241478,0.06377128511667252,0.06173379719257355,0.05986718088388443,0.059725843369960785,0.060642339289188385,0.061406757682561874,0.06312713772058487,0.06378006935119629,0.06606774032115936,0.06893105804920197,0.07136708498001099,0.07273192703723907,0.07313640415668488,0.07593478262424469]
      curve_l2_q75: [0.047020237892866135,0.0915711373090744,0.12770390510559082,0.1431444138288498,0.1553322970867157,0.16837061941623688,0.16786505281925201,0.16659300029277802,0.1626996546983719,0.13770125806331635,0.12869304418563843,0.1363067924976349,0.14544884860515594,0.15537667274475098,0.16669976711273193,0.16534991562366486,0.1534145474433899,0.15135055780410767,0.15662603080272675,0.166617751121521,0.16381365060806274,0.16628974676132202,0.16140833497047424,0.16335584223270416,0.17289093136787415,0.18036551773548126,0.18433202803134918,0.18406794965267181,0.18375496566295624,0.18703076243400574,0.19894737005233765,0.22280815243721008,0.22483383119106293,0.2242210954427719,0.22950556874275208,0.23931755125522614,0.2368118315935135,0.23640330135822296,0.23013180494308472,0.22650833427906036,0.22564095258712769,0.22320044040679932,0.22118018567562103,0.2236270010471344,0.22103555500507355,0.2131897658109665,0.212509423494339,0.21154063940048218,0.20790481567382812,0.20203891396522522,0.20406590402126312,0.19799841940402985,0.18984685838222504,0.1889243870973587,0.17902100086212158,0.1751176118850708,0.1730288714170456,0.17132773995399475,0.16950349509716034,0.16852426528930664,0.16790179908275604,0.16505801677703857,0.16515175998210907,0.16889555752277374,0.17533011734485626,0.17974768579006195,0.18854305148124695,0.1921793669462204,0.19841712713241577,0.20263010263442993,0.2069268524646759,0.2116396725177765,0.2069082260131836,0.19567954540252686,0.18908703327178955,0.1835533082485199,0.17600788176059723,0.16597667336463928,0.15806829929351807,0.14572544395923615,0.13938766717910767,0.1284654587507248,0.1295188069343567,0.12618917226791382,0.12549929320812225,0.12358081340789795,0.12154737114906311,0.1206100657582283,0.12099342048168182,0.12597493827342987,0.1264778971672058,0.12777520716190338,0.12907320261001587,0.13262945413589478,0.13530884683132172,0.13732202351093292,0.1396312713623047,0.14232216775417328,0.1468854695558548,0.15164609253406525]
      parameter_count: 392141
      active_inference_params: 317900
    2024:
      n_test: 400
      position_mse_per_horizon: {1:0.0003630616411101073,5:0.0004629870527423918,10:0.002794234547764063,25:0.0016201403923332691,50:0.001924700103700161,100:0.0016555727925151587}
      position_mse_std_per_horizon: {1:0.00014821764489170164,5:0.000246340234298259,10:0.004740727134048939,25:0.0016990993171930313,50:0.007449111435562372,100:0.0027561141178011894}
      velocity_mse_per_horizon: {1:0.0011599470162764192,5:0.021739115938544273,10:0.03459526598453522,25:0.017695391550660133,50:0.014275645837187767,100:0.006328371819108725}
      velocity_mse_std_per_horizon: {1:0.00018507367349229753,5:0.050173837691545486,10:0.03210411220788956,25:0.0676126778125763,50:0.03091936558485031,100:0.018771804869174957}
      position_error_per_horizon: {1:0.04558508098125458,5:0.05123930051922798,10:0.10651859641075134,25:0.09089435636997223,50:0.0784277692437172,100:0.08420246839523315}
      position_error_std_per_horizon: {1:0.010018536821007729,5:0.012347341515123844,10:0.0736151784658432,25:0.03819761052727699,50:0.0734662190079689,100:0.053323354572057724}
      overall_l2_mean: 0.08049140870571136
      curve_l2_mean: [0.04558509215712547,0.046727899461984634,0.05108608305454254,0.05150367319583893,0.05123932287096977,0.05627255514264107,0.06741151958703995,0.08238615095615387,0.09770159423351288,0.10651856660842896,0.10616517812013626,0.10570859163999557,0.10516355186700821,0.1030050590634346,0.10053101181983948,0.09924838691949844,0.09695343673229218,0.09444159269332886,0.0926087498664856,0.09180374443531036,0.0912294089794159,0.09091147780418396,0.09110718965530396,0.09110216051340103,0.09089432656764984,0.09107436239719391,0.09148965030908585,0.0916314423084259,0.09141930937767029,0.09101584553718567,0.09033447504043579,0.08913739025592804,0.08780588209629059,0.08664378523826599,0.0859365463256836,0.08563630282878876,0.08539730310440063,0.08514612168073654,0.08477149158716202,0.08429870754480362,0.08377156406641006,0.08329521119594574,0.08285999298095703,0.08246038109064102,0.08211715519428253,0.08167550712823868,0.08099670708179474,0.08019383251667023,0.07929962873458862,0.0784277692437172,0.07777015119791031,0.07738103717565536,0.07718577980995178,0.077088862657547,0.07703642547130585,0.0768657997250557,0.0767262876033783,0.07672150433063507,0.07686689496040344,0.07702487707138062,0.07705602049827576,0.07687952369451523,0.07652352750301361,0.07611867785453796,0.07577702403068542,0.07572972774505615,0.07551100850105286,0.07511945068836212,0.07469571381807327,0.07419823855161667,0.07364340126514435,0.07278931885957718,0.07188624143600464,0.07101883739233017,0.07032987475395203,0.06992726773023605,0.06972682476043701,0.0697220116853714,0.0699535384774208,0.07042504101991653,0.07106899470090866,0.07181685417890549,0.07262520492076874,0.07346834987401962,0.07434048503637314,0.07526587694883347,0.07620806992053986,0.07717278599739075,0.07811351865530014,0.07896420359611511,0.0797041580080986,0.08036798238754272,0.08095376193523407,0.0814410150051117,0.08179677277803421,0.08201371878385544,0.08220022171735764,0.08248694241046906,0.08308754861354828,0.08420249819755554]
      curve_l2_q25: [0.0385672003030777,0.04008703678846359,0.04482712969183922,0.04541432112455368,0.043723177164793015,0.04048691689968109,0.039130762219429016,0.04343385249376297,0.056973740458488464,0.06634166836738586,0.06503923237323761,0.06433194130659103,0.06525729596614838,0.06380096077919006,0.06406402587890625,0.06410785019397736,0.06593109667301178,0.06608256697654724,0.06484154611825943,0.06370552629232407,0.06558939069509506,0.06486289948225021,0.06448284536600113,0.06348314881324768,0.061481840908527374,0.06156906485557556,0.06191134452819824,0.06254757940769196,0.0626317709684372,0.06155179440975189,0.06213000416755676,0.06048693135380745,0.05921904370188713,0.057032063603401184,0.05523332953453064,0.05505526438355446,0.05439531058073044,0.05473849177360535,0.05383038893342018,0.053433213382959366,0.052337102591991425,0.0508723109960556,0.05051840841770172,0.04961686581373215,0.04943898320198059,0.048887595534324646,0.048121701925992966,0.04817608743906021,0.04700346291065216,0.04681684821844101,0.04724687337875366,0.04727211594581604,0.04703887924551964,0.045801375061273575,0.04530475661158562,0.04453067481517792,0.04478707164525986,0.04423632472753525,0.04338372126221657,0.04311174899339676,0.04465007036924362,0.045441076159477234,0.045984528958797455,0.04580964520573616,0.045852527022361755,0.04522307962179184,0.045597758144140244,0.04476150870323181,0.044788409024477005,0.043976787477731705,0.04365428909659386,0.04430495202541351,0.04423728957772255,0.043967828154563904,0.04421411082148552,0.0441209115087986,0.043759092688560486,0.044110145419836044,0.04427691549062729,0.04480654001235962,0.04404401406645775,0.04428847134113312,0.044920727610588074,0.04482389986515045,0.045071013271808624,0.04601734131574631,0.04660153388977051,0.04735402390360832,0.04795714095234871,0.04829072952270508,0.04930131882429123,0.049101412296295166,0.04963154345750809,0.049724280834198,0.04916819930076599,0.048704348504543304,0.04779021814465523,0.04620051011443138,0.044817112386226654,0.045552097260951996]
      curve_l2_q75: [0.05385419353842735,0.05418040230870247,0.05736549571156502,0.05844096839427948,0.05740810185670853,0.05610096827149391,0.057677242904901505,0.06882047653198242,0.08761050552129745,0.10122160613536835,0.10661271959543228,0.114897221326828,0.1218862384557724,0.11916647851467133,0.12744739651679993,0.12675674259662628,0.12367430329322815,0.12075503170490265,0.11477090418338776,0.11464056372642517,0.11264678835868835,0.11039391905069351,0.11403142660856247,0.11391258239746094,0.11475257575511932,0.11511333286762238,0.1125544011592865,0.11428965628147125,0.11090659350156784,0.1117180734872818,0.10835850983858109,0.10661079734563828,0.10329154133796692,0.1022881492972374,0.10326588153839111,0.10350625216960907,0.10126980394124985,0.10009917616844177,0.10006764531135559,0.10050441324710846,0.09918425977230072,0.09697847068309784,0.09561675786972046,0.09603627026081085,0.09431266784667969,0.09277940541505814,0.09220002591609955,0.09152229130268097,0.08910305798053741,0.08580594509840012,0.08295691013336182,0.08410394936800003,0.08446596562862396,0.08340476453304291,0.0823148638010025,0.0822933092713356,0.08097124099731445,0.08078822493553162,0.08093845099210739,0.07962813973426819,0.07876890897750854,0.0776529461145401,0.07688891887664795,0.07610198855400085,0.07599785923957825,0.07421621680259705,0.07446284592151642,0.07484348118305206,0.07363546639680862,0.07377512753009796,0.0726773738861084,0.07339629530906677,0.07295563817024231,0.07403690367937088,0.074435755610466,0.07376805692911148,0.07452782988548279,0.0764060765504837,0.07752460986375809,0.07938003540039062,0.08100545406341553,0.08347000926733017,0.08511745929718018,0.08759275078773499,0.09032522886991501,0.09171367436647415,0.09206241369247437,0.09200254082679749,0.09279236197471619,0.09513454884290695,0.09823369234800339,0.1008501648902893,0.10153841227293015,0.10396142303943634,0.10607093572616577,0.10668398439884186,0.10856656730175018,0.11108756810426712,0.11221890151500702,0.11258179694414139]
      parameter_count: 392141
      active_inference_params: 317900
  position_mse_per_horizon: {1:0.0002800375077640638,5:0.0016435929768097897,10:0.0024928780427823463,25:0.0024623910042767725,50:0.0034106406771267452,100:0.0020971534928927817}
  position_mse_std_per_horizon: {1:8.248195808846503e-05,5:0.0005747368462228527,10:0.0025481312380482755,25:0.0032162999656672278,50:0.008179370624323687,100:0.002886547784631451}
  position_mse_seed_std_per_horizon: {1:0.0001246148891045717,5:0.001517609382255656,10:0.0003187849760055765,25:0.001893215914832058,50:0.0026389585731517837,100:0.0006344975945901646}
  velocity_mse_per_horizon: {1:0.0019345224912588794,5:0.027669757604599,10:0.033557978148261704,25:0.022925445189078648,50:0.02439580454180638,100:0.011832344190527996}
  velocity_mse_std_per_horizon: {1:0.0002487019179776932,5:0.04881093154350916,10:0.04299668843547503,25:0.07159791638453801,50:0.040194777150948845,100:0.020193679879109066}
  velocity_mse_seed_std_per_horizon: {1:0.0006903363051937698,5:0.00710758025004435,10:0.005106447041318032,25:0.009149407921130264,50:0.016977631136613044,100:0.007150353941303618}
  position_error_per_horizon: {1:0.039661443481842674,5:0.08992001290122668,10:0.11085250228643417,25:0.10220765819152196,50:0.10729566713174184,100:0.09809674074252446}
  position_error_std_per_horizon: {1:0.0059564432594925165,5:0.01970250066369772,10:0.04815458506345749,25:0.05435042455792427,50:0.08321564396222432,100:0.052399648974339165}
  position_error_seed_std_per_horizon: {1:0.009754536642350632,5:0.04508591477687565,10:0.005392051138840468,25:0.03730480325736876,50:0.052453582184670444,100:0.017615286752715828}
  active_inference_params: 317900
  overall_l2_mean: 0.09982159237066905
  overall_l2_seed_std: 0.035505357174564434
  curve_l2_mean: [0.028402993455529213,0.05409229174256325,0.06919767707586288,0.07416556030511856,0.0790843591094017,0.0878472700715065,0.09656296670436859,0.10067830234766006,0.1020779237151146,0.10914812982082367,0.11117352545261383,0.10601089149713516,0.10027242451906204,0.09265705198049545,0.08792710304260254,0.08330740034580231,0.07834500819444656,0.07513298094272614,0.07268580794334412,0.07121405750513077,0.07048943638801575,0.0699843242764473,0.07024625688791275,0.07103260606527328,0.07186911255121231,0.07283908873796463,0.07377299666404724,0.07496389746665955,0.07631167769432068,0.07760873436927795,0.07856328785419464,0.07912985235452652,0.07931079715490341,0.07924097031354904,0.07928299158811569,0.07933448255062103,0.07926910370588303,0.07924263179302216,0.07901707291603088,0.07876311987638474,0.07827514410018921,0.07765324413776398,0.07717940956354141,0.07693558931350708,0.07675628364086151,0.07660053670406342,0.07645636796951294,0.07628099620342255,0.07592596858739853,0.07561714947223663,0.07536737620830536,0.07527463883161545,0.07539671659469604,0.07562102377414703,0.07572361081838608,0.07570267468690872,0.07563478499650955,0.07550754398107529,0.07540243119001389,0.0750468298792839,0.07449661940336227,0.07387328147888184,0.07321488112211227,0.0727459192276001,0.07271791994571686,0.07326419651508331,0.0738634467124939,0.07443109154701233,0.07493837177753448,0.07527387142181396,0.07547730207443237,0.07534337788820267,0.07509634643793106,0.07475726306438446,0.07443652302026749,0.07422575354576111,0.07407048344612122,0.07402627915143967,0.07415244728326797,0.07442715018987656,0.07473479956388474,0.07499084621667862,0.07517151534557343,0.07530569285154343,0.075480155646801,0.0758022665977478,0.07630369812250137,0.07703898847103119,0.07799171656370163,0.0790792927145958,0.08018292486667633,0.08123725652694702,0.08216797560453415,0.08296094834804535,0.08365557342767715,0.08437445759773254,0.08537574857473373,0.08690201491117477,0.08917172253131866,0.09217877686023712]
  curve_l2_q25: [0.025688745081424713,0.0517529658973217,0.06582773476839066,0.06946137547492981,0.06982161849737167,0.07347680628299713,0.07758424431085587,0.07654827833175659,0.07615315169095993,0.0839645117521286,0.0856681764125824,0.08373668789863586,0.07903014123439789,0.06812075525522232,0.06539769470691681,0.060212716460227966,0.05562014505267143,0.05183684080839157,0.04982618987560272,0.04733149707317352,0.0451623797416687,0.04393559694290161,0.043220147490501404,0.04377090558409691,0.04441813752055168,0.045465223491191864,0.04610028490424156,0.04609829932451248,0.0469626858830452,0.047558873891830444,0.04737984389066696,0.04693424329161644,0.04714300483465195,0.047332197427749634,0.04776688665151596,0.04760952293872833,0.04753852263092995,0.047342609614133835,0.04805628955364227,0.047483399510383606,0.047179535031318665,0.046427953988313675,0.044934358447790146,0.044367603957653046,0.04315820336341858,0.04316725581884384,0.041933320462703705,0.04202969744801521,0.04313062131404877,0.04104733094573021,0.041153665632009506,0.04191537946462631,0.04235341399908066,0.042520519345998764,0.04089833050966263,0.04149852320551872,0.04167407751083374,0.041788697242736816,0.041652269661426544,0.04189186170697212,0.041234664618968964,0.04103300720453262,0.041546013206243515,0.04106965661048889,0.04186272248625755,0.04361831396818161,0.044556453824043274,0.044056959450244904,0.04459691420197487,0.04600435122847557,0.04677724093198776,0.04649851843714714,0.04706789553165436,0.04737219586968422,0.048247821629047394,0.04891939461231232,0.05002482980489731,0.05077916011214256,0.051292967051267624,0.053736940026283264,0.053761158138513565,0.05310306325554848,0.05243781954050064,0.05203860253095627,0.05201371759176254,0.051209457218647,0.05156080424785614,0.05117755010724068,0.05187275633215904,0.05312029644846916,0.054139476269483566,0.05515177547931671,0.05655680224299431,0.05725458264350891,0.05733560398221016,0.057014767080545425,0.05715153366327286,0.058652881532907486,0.05937575921416283,0.06200525537133217]
  curve_l2_q75: [0.030848084017634392,0.0568077489733696,0.07301976531744003,0.08181136846542358,0.09329673647880554,0.10683343559503555,0.11842299997806549,0.12626716494560242,0.1294131577014923,0.13687501847743988,0.13900557160377502,0.1323055922985077,0.12427413463592529,0.11480254679918289,0.10740269720554352,0.10009858012199402,0.09046373516321182,0.08680496364831924,0.0834229364991188,0.0830121785402298,0.08478143066167831,0.08581989258527756,0.08464576303958893,0.08663273602724075,0.08694306761026382,0.08552728593349457,0.08661364018917084,0.09050130844116211,0.09363279491662979,0.0964195504784584,0.09766033291816711,0.09766631573438644,0.09828822314739227,0.09566336870193481,0.09244328737258911,0.09387047588825226,0.08968652784824371,0.09100281447172165,0.09253418445587158,0.09094621241092682,0.0890679806470871,0.08840598165988922,0.08570457249879837,0.08603780716657639,0.08672039210796356,0.08579102903604507,0.08533994853496552,0.0850599855184555,0.08378507196903229,0.08354295045137405,0.08368957787752151,0.08443000912666321,0.08500698208808899,0.08575280010700226,0.08399265259504318,0.08330803364515305,0.08047395944595337,0.08058642596006393,0.08071506023406982,0.07909595966339111,0.07704782485961914,0.07479239255189896,0.07399509847164154,0.07332340627908707,0.07415328919887543,0.0736687034368515,0.07459498941898346,0.07657839357852936,0.07800287008285522,0.07916605472564697,0.07914764434099197,0.08006235212087631,0.08028841018676758,0.08072361350059509,0.07881306111812592,0.07789904624223709,0.07924868166446686,0.07942206412553787,0.07897322624921799,0.07970607280731201,0.08017571270465851,0.08156496286392212,0.08328864723443985,0.08187296986579895,0.08239028602838516,0.08291680365800858,0.08383144438266754,0.08661171793937683,0.08856142312288284,0.08987045288085938,0.09065719693899155,0.0910019725561142,0.0916810929775238,0.09279067814350128,0.0929693877696991,0.09498428553342819,0.09680746495723724,0.09994266927242279,0.10333787649869919,0.10428255796432495]
Continuous RSSM:
  n_test: 400
  seeds: [42,101,2024]
  parameter_count: 380460
  per_seed:
    42:
      n_test: 400
      position_mse_per_horizon: {1:2.2063661162974313e-05,5:9.420973947271705e-05,10:0.00031403437606059015,25:0.0008159067947417498,50:0.0016975541366264224,100:0.0013903399230912328}
      position_mse_std_per_horizon: {1:3.963471499446314e-06,5:9.16668213903904e-05,10:0.0003905495104845613,25:0.001349463826045394,50:0.006731225643306971,100:0.0027804546989500523}
      velocity_mse_per_horizon: {1:0.0005160506698302925,5:0.018466511741280556,10:0.01503689680248499,25:0.018678074702620506,50:0.021837402135133743,100:0.02143263630568981}
      velocity_mse_std_per_horizon: {1:0.00019272186909802258,5:0.03378969430923462,10:0.02326388843357563,25:0.06908168643712997,50:0.035098034888505936,100:0.023902960121631622}
      position_error_per_horizon: {1:0.011463386937975883,5:0.02229641005396843,10:0.0388997346162796,25:0.06143658980727196,50:0.07276146858930588,100:0.07697229087352753}
      position_error_std_per_horizon: {1:0.0009862579172477126,5:0.008254001848399639,10:0.019261786714196205,25:0.03348112851381302,50:0.0699363574385643,100:0.04916609078645706}
      overall_l2_mean: 0.06333083659410477
      curve_l2_mean: [0.011463391594588757,0.011260421015322208,0.012365542352199554,0.015929721295833588,0.022296421229839325,0.02928614430129528,0.03493636101484299,0.0381774939596653,0.03919664025306702,0.03889975696802139,0.03826487064361572,0.038217298686504364,0.03908471390604973,0.040584150701761246,0.042181506752967834,0.043829113245010376,0.04525160416960716,0.04684450104832649,0.04872138798236847,0.05078807845711708,0.0528855174779892,0.055083855986595154,0.05736937001347542,0.059585753828287125,0.06143657863140106,0.06319081038236618,0.06478103995323181,0.06640522927045822,0.06776407361030579,0.06893851608037949,0.0697292909026146,0.0700710192322731,0.07026242464780807,0.0704326182603836,0.07077030837535858,0.07112441211938858,0.0715433657169342,0.0720292255282402,0.07242216914892197,0.0727837085723877,0.07293185591697693,0.07291039824485779,0.07280029356479645,0.07287564128637314,0.07302694022655487,0.07300973683595657,0.07297173887491226,0.07294511049985886,0.07285936921834946,0.07276146113872528,0.07269163429737091,0.07282735407352448,0.07312853634357452,0.07339350134134293,0.07365009933710098,0.07371819764375687,0.07354281097650528,0.07310511916875839,0.07264538109302521,0.07210807502269745,0.0715232864022255,0.07096678018569946,0.07057050615549088,0.07042660564184189,0.07052962481975555,0.07093211263418198,0.07111003994941711,0.07108928263187408,0.07101858407258987,0.07082027196884155,0.07048171013593674,0.06975945085287094,0.06889816373586655,0.0680384710431099,0.06733889132738113,0.06684746593236923,0.06648658961057663,0.06629940867424011,0.0663762241601944,0.06675851345062256,0.06741182506084442,0.06827273219823837,0.0692458227276802,0.07023739069700241,0.07121509313583374,0.07219618558883667,0.0731360986828804,0.07401766628026962,0.07478515803813934,0.0753595232963562,0.07566819339990616,0.075731560587883,0.07558576762676239,0.07530222833156586,0.07497826218605042,0.07473771274089813,0.07476350665092468,0.07515984028577805,0.07594364881515503,0.07697230577468872]
      curve_l2_q25: [0.010706808418035507,0.010530298575758934,0.011923742480576038,0.014794009737670422,0.018410537391901016,0.02160072885453701,0.025295570492744446,0.026782436296343803,0.027612842619419098,0.02731100656092167,0.027227645739912987,0.027090301737189293,0.026997486129403114,0.026558347046375275,0.02642597258090973,0.02665811963379383,0.02687377855181694,0.02794378623366356,0.02921762876212597,0.030856337398290634,0.03270690143108368,0.03464135900139809,0.036643266677856445,0.03812652826309204,0.039326757192611694,0.04115968197584152,0.04286473989486694,0.04439994692802429,0.045362215489149094,0.0457846075296402,0.04602726921439171,0.045405659824609756,0.04494846612215042,0.04508977755904198,0.044802989810705185,0.04438367486000061,0.044127143919467926,0.04446050524711609,0.044314801692962646,0.04480468109250069,0.0432770699262619,0.04340970143675804,0.042841680347919464,0.04378228634595871,0.0435987263917923,0.04372811317443848,0.043370362371206284,0.043718092143535614,0.04397675022482872,0.04407918453216553,0.043996188789606094,0.04342690855264664,0.04383907467126846,0.04441108927130699,0.04456271603703499,0.0447361059486866,0.044320136308670044,0.04327201098203659,0.043457645922899246,0.04316595941781998,0.04305579513311386,0.04286554455757141,0.04281693696975708,0.04249044880270958,0.04264501482248306,0.042960528284311295,0.042576104402542114,0.04273608326911926,0.041710853576660156,0.04142852872610092,0.042193327099084854,0.04165973886847496,0.040922120213508606,0.0417061522603035,0.041548050940036774,0.040775712579488754,0.041922520846128464,0.04201730340719223,0.04230133071541786,0.042541950941085815,0.042059462517499924,0.04194073751568794,0.04261467233300209,0.04317449405789375,0.04361341893672943,0.044406741857528687,0.04507114738225937,0.04559807851910591,0.046338215470314026,0.04661919176578522,0.046180956065654755,0.04592975229024887,0.045918770134449005,0.04467355087399483,0.043579697608947754,0.04372145235538483,0.044027749449014664,0.04435843229293823,0.04609616845846176,0.047679442912340164]
      curve_l2_q75: [0.011992122046649456,0.01178180892020464,0.012974120676517487,0.017195865511894226,0.02397281862795353,0.03125434368848801,0.03698160871863365,0.040470924228429794,0.04172009974718094,0.04124011471867561,0.03949274122714996,0.03826536238193512,0.03927238658070564,0.04327080026268959,0.04681115970015526,0.05087456852197647,0.0542360320687294,0.056107811629772186,0.05801183730363846,0.059569310396909714,0.06241707503795624,0.06650331616401672,0.0691382884979248,0.07184644043445587,0.07205390185117722,0.07330650091171265,0.07602091133594513,0.07630808651447296,0.0776982307434082,0.07672049850225449,0.07903577387332916,0.07942059636116028,0.07993744313716888,0.0798814445734024,0.08178628981113434,0.08079956471920013,0.08033742010593414,0.08031443506479263,0.08063353598117828,0.07828755676746368,0.07717481255531311,0.07691731303930283,0.0771571546792984,0.07494708895683289,0.07509515434503555,0.077145054936409,0.07732116430997849,0.07698553055524826,0.07607056200504303,0.07530353963375092,0.07525460422039032,0.07487420737743378,0.07516314089298248,0.07507163286209106,0.07473614066839218,0.07450307160615921,0.0737198144197464,0.07267578691244125,0.07227933406829834,0.07168915867805481,0.07050883024930954,0.06985761970281601,0.06950785219669342,0.07013852894306183,0.06894098967313766,0.06966210156679153,0.07089312374591827,0.07062169909477234,0.07136419415473938,0.07154961675405502,0.07123735547065735,0.07114848494529724,0.0707588717341423,0.07079441845417023,0.07062837481498718,0.07119211554527283,0.0711594969034195,0.07017752528190613,0.07029874622821808,0.07153740525245667,0.0729413852095604,0.07456962019205093,0.07627198100090027,0.07925388962030411,0.08051666617393494,0.08113344013690948,0.08276278525590897,0.08404087275266647,0.085039883852005,0.08603381365537643,0.08776761591434479,0.08726959675550461,0.08880430459976196,0.08915401250123978,0.08879334479570389,0.08825865387916565,0.08846672624349594,0.08709301054477692,0.08778393268585205,0.08817002177238464]
      parameter_count: 380460
    101:
      n_test: 400
      position_mse_per_horizon: {1:8.215529669541866e-05,5:0.00011103424913017079,10:0.00028387978090904653,25:0.0008655569981783628,50:0.001734933815896511,100:0.0015106787905097008}
      position_mse_std_per_horizon: {1:8.075202458712738e-06,5:7.686603930778801e-05,10:0.0003137427265755832,25:0.0014106726739555597,50:0.006860995199531317,100:0.002764763543382287}
      velocity_mse_per_horizon: {1:0.000393129390431568,5:0.018857905641198158,10:0.018870759755373,25:0.021117882803082466,50:0.0235483068972826,100:0.022047914564609528}
      velocity_mse_std_per_horizon: {1:0.00017924248822964728,5:0.035433199256658554,10:0.03179612010717392,25:0.07052621245384216,50:0.0367245189845562,100:0.02574349381029606}
      position_error_per_horizon: {1:0.02217557467520237,5:0.024929767474532127,10:0.038022179156541824,25:0.06363818049430847,50:0.07386959344148636,100:0.0818834900856018}
      position_error_std_per_horizon: {1:0.0010842412011697888,5:0.006686719600111246,10:0.016049683094024658,25:0.03381603956222534,50:0.07037674635648727,100:0.04857127368450165}
      overall_l2_mean: 0.06463415175676346
      curve_l2_mean: [0.02217557467520237,0.022519782185554504,0.021268196403980255,0.02134619653224945,0.024929767474532127,0.030084123834967613,0.034351665526628494,0.036765679717063904,0.037748586386442184,0.03802219778299332,0.038079436868429184,0.03851339966058731,0.03968304768204689,0.04152459278702736,0.043563634157180786,0.045723069459199905,0.047672078013420105,0.04970746114850044,0.0518716536462307,0.054055824875831604,0.05613347515463829,0.05816299095749855,0.06018819659948349,0.062104735523462296,0.06363815069198608,0.06510812789201736,0.06647645682096481,0.06794299930334091,0.06917521357536316,0.07020923495292664,0.07086192816495895,0.07106232643127441,0.07115431129932404,0.07127255201339722,0.07158242911100388,0.07196565717458725,0.07246653735637665,0.07303754985332489,0.07351266592741013,0.0739436075091362,0.0741666778922081,0.07421586662530899,0.07418886572122574,0.07431687414646149,0.07448937743902206,0.07447195798158646,0.07440086454153061,0.07428263872861862,0.07408945262432098,0.07386959344148636,0.07367350906133652,0.07366431504487991,0.07379768043756485,0.07384556531906128,0.07389936596155167,0.07373137027025223,0.07335641980171204,0.07277841866016388,0.07230160385370255,0.07180947810411453,0.07129684090614319,0.07084783911705017,0.0705687552690506,0.07052921503782272,0.07072022557258606,0.07116913050413132,0.07137151807546616,0.07137823849916458,0.07135315239429474,0.07124015688896179,0.07103390991687775,0.0704914778470993,0.06984490156173706,0.06921077519655228,0.06872843950986862,0.0684373527765274,0.06825698167085648,0.06821110844612122,0.06834381073713303,0.0686885342001915,0.06921662390232086,0.06985123455524445,0.0704876258969307,0.07104960829019547,0.07152752578258514,0.07198023051023483,0.07241782546043396,0.07288189232349396,0.07335706800222397,0.0737975463271141,0.0741620659828186,0.07447237521409988,0.07477319240570068,0.07516074925661087,0.0757320374250412,0.07656165957450867,0.0777173787355423,0.07911773771047592,0.08058947324752808,0.0818834975361824]
      curve_l2_q25: [0.02142200991511345,0.021511469036340714,0.020121518522500992,0.020040104165673256,0.021903734654188156,0.02474929578602314,0.02743758074939251,0.029013726860284805,0.029862750321626663,0.029128404334187508,0.028099093586206436,0.02711939997971058,0.025925733149051666,0.02529917098581791,0.02559799700975418,0.0266769640147686,0.028127841651439667,0.030252857133746147,0.032310597598552704,0.03421163931488991,0.03593031316995621,0.03727969527244568,0.038615599274635315,0.040322571992874146,0.041514404118061066,0.04275676608085632,0.0439876914024353,0.04524334892630577,0.045638151466846466,0.04548228532075882,0.045405805110931396,0.045486241579055786,0.04555794596672058,0.04456173628568649,0.04518052190542221,0.04434571415185928,0.04518081247806549,0.04498053342103958,0.044735267758369446,0.04464980587363243,0.04439767450094223,0.045171067118644714,0.04503820836544037,0.04477827250957489,0.044994551688432693,0.04566600173711777,0.045860882848501205,0.0465383417904377,0.04523240774869919,0.044406577944755554,0.044439900666475296,0.04444543272256851,0.044733427464962006,0.04410438984632492,0.04425183683633804,0.044058505445718765,0.043054260313510895,0.042643602937459946,0.04260507598519325,0.042310260236263275,0.04214991256594658,0.04139463230967522,0.0415591225028038,0.0409664548933506,0.04050596430897713,0.040875207632780075,0.04116823896765709,0.04097748175263405,0.04094898700714111,0.0409875363111496,0.04144670069217682,0.04135585576295853,0.0411725714802742,0.04188651964068413,0.041769202798604965,0.04340367019176483,0.04299554601311684,0.042404089123010635,0.04281574487686157,0.04246832802891731,0.04165758937597275,0.04095113277435303,0.04108492285013199,0.041072335094213486,0.041324030607938766,0.04164813831448555,0.041815102100372314,0.04221172630786896,0.042777713388204575,0.04336986690759659,0.043623246252536774,0.04378043860197067,0.04455389827489853,0.04434685781598091,0.044630810618400574,0.04577956721186638,0.04692927002906799,0.04864256829023361,0.05057816207408905,0.052479371428489685]
      curve_l2_q75: [0.022851552814245224,0.023408476263284683,0.022298073396086693,0.022263208404183388,0.026056280359625816,0.03215770050883293,0.03728252276778221,0.04004466533660889,0.040243957191705704,0.038825441151857376,0.03693821281194687,0.038150765001773834,0.04140153527259827,0.04573214426636696,0.049856916069984436,0.05235422030091286,0.054701268672943115,0.05870203673839569,0.062114372849464417,0.06496500968933105,0.06659947335720062,0.07009994238615036,0.07240043580532074,0.07266846299171448,0.07349884510040283,0.07466162741184235,0.07708168029785156,0.07728710770606995,0.07966606318950653,0.0798368752002716,0.0819859653711319,0.08170922845602036,0.07989809662103653,0.08203247934579849,0.0825575441122055,0.08186204731464386,0.08020836859941483,0.08070351183414459,0.08052555471658707,0.07918630540370941,0.08026312291622162,0.07964617013931274,0.07930795848369598,0.07870477437973022,0.0781995952129364,0.07820981740951538,0.0785696730017662,0.0788830891251564,0.07707783579826355,0.07470253109931946,0.07340303808450699,0.07329452782869339,0.0743202343583107,0.0750918984413147,0.0746404230594635,0.07428595423698425,0.07436582446098328,0.07320328801870346,0.07231278717517853,0.07171672582626343,0.06905938684940338,0.06898946315050125,0.06990191340446472,0.06910283863544464,0.06965942680835724,0.07079338282346725,0.0712621659040451,0.07130321115255356,0.07107629626989365,0.0728839784860611,0.07237925380468369,0.07210999727249146,0.07190295308828354,0.07220416516065598,0.0719514787197113,0.07211747765541077,0.0715520828962326,0.07223184406757355,0.07253711670637131,0.07395287603139877,0.07666444778442383,0.07808207720518112,0.08079295605421066,0.08146101236343384,0.08211684972047806,0.08337569236755371,0.08411690592765808,0.08582539856433868,0.08816145360469818,0.08754836022853851,0.08796721696853638,0.08811815083026886,0.08888675272464752,0.08928724378347397,0.08987626433372498,0.0915708839893341,0.09079816192388535,0.09165550768375397,0.09356414526700974,0.09515539556741714]
      parameter_count: 380460
    2024:
      n_test: 400
      position_mse_per_horizon: {1:0.00010488614498171955,5:0.00013294491509441286,10:0.000395959650631994,25:0.0009463678579777479,50:0.001785421627573669,100:0.001666970900259912}
      position_mse_std_per_horizon: {1:1.6073785445769317e-05,5:0.00026270351372659206,10:0.0007350608939304948,25:0.0014478814555332065,50:0.0066442894749343395,100:0.0028185711707919836}
      velocity_mse_per_horizon: {1:0.003669383004307747,5:0.018771443516016006,10:0.015883563086390495,25:0.021422648802399635,50:0.022301437333226204,100:0.020432692021131516}
      velocity_mse_std_per_horizon: {1:0.0006319108651950955,5:0.043800048530101776,10:0.033365629613399506,25:0.07016664743423462,50:0.03567387908697128,100:0.023650331422686577}
      position_error_per_horizon: {1:0.025013625621795654,5:0.021677441895008087,10:0.037732381373643875,25:0.06671080738306046,50:0.07743774354457855,100:0.08730346709489822}
      position_error_std_per_horizon: {1:0.0019066949607804418,5:0.01810408942401409,10:0.030854905024170876,25:0.03504103794693947,50:0.06867258995771408,100:0.04878450185060501}
      overall_l2_mean: 0.06770292669534683
      curve_l2_mean: [0.025013618171215057,0.0258527509868145,0.024003420025110245,0.02181052416563034,0.021677443757653236,0.025605542585253716,0.03144880384206772,0.035616323351860046,0.03730694204568863,0.03773240000009537,0.038243114948272705,0.039483729749917984,0.0414951927959919,0.04387463256716728,0.04616905376315117,0.048462044447660446,0.050555456429719925,0.05273700878024101,0.055039893835783005,0.05733400955796242,0.05944924056529999,0.061460789293050766,0.06342564523220062,0.06524387001991272,0.06671081483364105,0.06813806295394897,0.06950914859771729,0.07098644226789474,0.07224526256322861,0.07332566380500793,0.07400134205818176,0.07429009675979614,0.07452792674303055,0.0748131275177002,0.07535705715417862,0.07597159594297409,0.07660792022943497,0.07727047055959702,0.07778457552194595,0.07822753489017487,0.07846102863550186,0.07851826399564743,0.0784294381737709,0.07846998423337936,0.07854724675416946,0.07841942459344864,0.07823078334331512,0.07802267372608185,0.0777459591627121,0.07743772119283676,0.0771661028265953,0.07712391018867493,0.07724764943122864,0.07733096927404404,0.07746310532093048,0.077341727912426,0.07696980983018875,0.07632386684417725,0.07575301826000214,0.07512931525707245,0.07447323203086853,0.07386676967144012,0.07340265065431595,0.07315363734960556,0.07309477031230927,0.07325299084186554,0.07315108180046082,0.072862409055233,0.07256502658128738,0.07221677899360657,0.07183174788951874,0.0711812674999237,0.07049546390771866,0.06988926976919174,0.06949136406183243,0.06933877617120743,0.06934013962745667,0.06952697783708572,0.06996291875839233,0.07071904838085175,0.07174072414636612,0.0728907510638237,0.0740627571940422,0.07516855001449585,0.07619459927082062,0.07716596126556396,0.07805836945772171,0.07888691127300262,0.07964495569467545,0.08030710369348526,0.08083856850862503,0.08125631511211395,0.08160852640867233,0.08197246491909027,0.08242229372262955,0.08302479982376099,0.08387483656406403,0.08495403081178665,0.08616068214178085,0.08730343729257584]
      curve_l2_q25: [0.023613179102540016,0.024737058207392693,0.022497860714793205,0.01707969605922699,0.011395536363124847,0.011302144266664982,0.014526615850627422,0.01705843023955822,0.019098982214927673,0.020203087478876114,0.02070114202797413,0.02099858783185482,0.021282296627759933,0.022078411653637886,0.023901931941509247,0.026447147130966187,0.029238473623991013,0.032093700021505356,0.03470585495233536,0.03659098222851753,0.03840098902583122,0.04015958681702614,0.041595593094825745,0.04308026656508446,0.0440780371427536,0.04516495019197464,0.04667951539158821,0.0480002760887146,0.0481906458735466,0.04901433363556862,0.04855842888355255,0.04877590760588646,0.048453617841005325,0.04743613675236702,0.04750804603099823,0.047861531376838684,0.04893076792359352,0.049902889877557755,0.05006103962659836,0.0484580434858799,0.04857167974114418,0.04943676292896271,0.04913056641817093,0.04921618103981018,0.050028398633003235,0.04974409192800522,0.04916638880968094,0.04962688684463501,0.049345023930072784,0.04872285574674606,0.04882058873772621,0.04923488199710846,0.04932820051908493,0.04963402822613716,0.05006438121199608,0.04898582398891449,0.048772022128105164,0.048508550971746445,0.0483330674469471,0.047945961356163025,0.047199200838804245,0.046558529138565063,0.04627493768930435,0.04679782688617706,0.046331144869327545,0.046559736132621765,0.04630878567695618,0.04644467309117317,0.04646804928779602,0.04604135826230049,0.04546992853283882,0.046017877757549286,0.045383915305137634,0.04486546665430069,0.04417514428496361,0.043601129204034805,0.043397821485996246,0.04380577802658081,0.04413178935647011,0.04388004168868065,0.044190965592861176,0.0448312871158123,0.04511936753988266,0.04532402381300926,0.045193836092948914,0.04489027336239815,0.045434076339006424,0.04552561044692993,0.04563136398792267,0.045720916241407394,0.04525633528828621,0.046642906963825226,0.04684959352016449,0.048286743462085724,0.04816820099949837,0.04966282844543457,0.05173406004905701,0.054013241082429886,0.05604812875390053,0.05775152146816254]
      curve_l2_q75: [0.026351969689130783,0.02699722722172737,0.024804510176181793,0.02214968577027321,0.022669440135359764,0.025024011731147766,0.029460005462169647,0.03410860523581505,0.039011623710393906,0.040608178824186325,0.04429183527827263,0.04666096344590187,0.04892049729824066,0.05365968123078346,0.05554701387882233,0.05819088965654373,0.06039087474346161,0.06393034011125565,0.06620930880308151,0.06723886728286743,0.07006188482046127,0.07433962821960449,0.07741791009902954,0.07668924331665039,0.07968840003013611,0.07955654710531235,0.08081738650798798,0.08224011212587357,0.08617272228002548,0.0853927880525589,0.08528177440166473,0.08490950614213943,0.08629712462425232,0.08653899282217026,0.08771806955337524,0.0889112651348114,0.08839239925146103,0.08797804266214371,0.08738331496715546,0.08865484595298767,0.08708669245243073,0.08587229996919632,0.08519189059734344,0.08399271219968796,0.08270064741373062,0.08331095427274704,0.08263412117958069,0.08279819786548615,0.08100994676351547,0.08022148907184601,0.08021578937768936,0.07907819747924805,0.07994373887777328,0.07789890468120575,0.07897844910621643,0.07939966768026352,0.07670139521360397,0.07614939659833908,0.07473192363977432,0.07351129502058029,0.07308050245046616,0.07221334427595139,0.07169175893068314,0.07142813503742218,0.07172993570566177,0.07152235507965088,0.0707235038280487,0.07058992981910706,0.07074785977602005,0.07072010636329651,0.07128963619470596,0.07144196331501007,0.07153022289276123,0.07225699722766876,0.07249279320240021,0.07377709448337555,0.07492520660161972,0.07690844684839249,0.07798361033201218,0.07968546450138092,0.08213973790407181,0.084759920835495,0.08830518275499344,0.08944187313318253,0.0908171534538269,0.09060566127300262,0.09275960177183151,0.09442757815122604,0.09526042640209198,0.09585133194923401,0.09737914800643921,0.10016489028930664,0.09979335218667984,0.09985164552927017,0.09895201027393341,0.10049264132976532,0.1002228632569313,0.10114771127700806,0.10098186880350113,0.102252297103405]
      parameter_count: 380460
  position_mse_per_horizon: {1:6.970170094670418e-05,5:0.0001127296345657669,10:0.0003312912692005436,25:0.0008759438836326202,50:0.001739303193365534,100:0.001522663204620282}
  position_mse_std_per_horizon: {1:9.370819801309457e-06,5:0.00014374545814159015,10:0.00047978437699687976,25:0.0014026726518447201,50:0.006745503439257543,100:0.0027879298043747744}
  position_mse_seed_std_per_horizon: {1:4.279263945798248e-05,5:1.942316159749303e-05,10:5.799848765192975e-05,25:6.584783821240204e-05,50:4.4096401054294876e-05,100:0.00013870434030161064}
  velocity_mse_per_horizon: {1:0.001526187688189869,5:0.01869862029949824,10:0.016597073214749496,25:0.02040620210270087,50:0.022562382121880848,100:0.02130441429714362}
  velocity_mse_std_per_horizon: {1:0.00033462507417425513,5:0.037674314031998314,10:0.029475212718049686,25:0.06992484877506892,50:0.03583214432001114,100:0.024432261784871418}
  velocity_mse_seed_std_per_horizon: {1:0.0018570788977597217,5:0.00020560815127609284,10:0.0020140627628837575,25:0.0015043399920783647,50:0.0008847982331189538,100:0.0008152095613071585}
  position_error_per_horizon: {1:0.01955086241165797,5:0.022967873141169548,10:0.0382180983821551,25:0.0639285258948803,50:0.07468960185845692,100:0.08205308268467586}
  position_error_std_per_horizon: {1:0.0013257313597326477,5:0.011014936957508326,10:0.022055458277463913,25:0.03411273534099261,50:0.06966189791758855,100:0.048840622107187905}
  position_error_seed_std_per_horizon: {1:0.007146263215619875,5:0.0017270067825320892,10:0.0006078376884471017,25:0.002649069288998305,50:0.002443603326494411,100:0.005167675663825986}
  overall_l2_mean: 0.06522263834873836
  overall_l2_seed_std: 0.002244667083343404
  curve_l2_mean: [0.011463391594588757,0.011260421015322208,0.012365542352199554,0.015929721295833588,0.022296421229839325,0.02928614430129528,0.03493636101484299,0.0381774939596653,0.03919664025306702,0.03889975696802139,0.03826487064361572,0.038217298686504364,0.03908471390604973,0.040584150701761246,0.042181506752967834,0.043829113245010376,0.04525160416960716,0.04684450104832649,0.04872138798236847,0.05078807845711708,0.0528855174779892,0.055083855986595154,0.05736937001347542,0.059585753828287125,0.06143657863140106,0.06319081038236618,0.06478103995323181,0.06640522927045822,0.06776407361030579,0.06893851608037949,0.0697292909026146,0.0700710192322731,0.07026242464780807,0.0704326182603836,0.07077030837535858,0.07112441211938858,0.0715433657169342,0.0720292255282402,0.07242216914892197,0.0727837085723877,0.07293185591697693,0.07291039824485779,0.07280029356479645,0.07287564128637314,0.07302694022655487,0.07300973683595657,0.07297173887491226,0.07294511049985886,0.07285936921834946,0.07276146113872528,0.07269163429737091,0.07282735407352448,0.07312853634357452,0.07339350134134293,0.07365009933710098,0.07371819764375687,0.07354281097650528,0.07310511916875839,0.07264538109302521,0.07210807502269745,0.0715232864022255,0.07096678018569946,0.07057050615549088,0.07042660564184189,0.07052962481975555,0.07093211263418198,0.07111003994941711,0.07108928263187408,0.07101858407258987,0.07082027196884155,0.07048171013593674,0.06975945085287094,0.06889816373586655,0.0680384710431099,0.06733889132738113,0.06684746593236923,0.06648658961057663,0.06629940867424011,0.0663762241601944,0.06675851345062256,0.06741182506084442,0.06827273219823837,0.0692458227276802,0.07023739069700241,0.07121509313583374,0.07219618558883667,0.0731360986828804,0.07401766628026962,0.07478515803813934,0.0753595232963562,0.07566819339990616,0.075731560587883,0.07558576762676239,0.07530222833156586,0.07497826218605042,0.07473771274089813,0.07476350665092468,0.07515984028577805,0.07594364881515503,0.07697230577468872]
  curve_l2_q25: [0.010706808418035507,0.010530298575758934,0.011923742480576038,0.014794009737670422,0.018410537391901016,0.02160072885453701,0.025295570492744446,0.026782436296343803,0.027612842619419098,0.02731100656092167,0.027227645739912987,0.027090301737189293,0.026997486129403114,0.026558347046375275,0.02642597258090973,0.02665811963379383,0.02687377855181694,0.02794378623366356,0.02921762876212597,0.030856337398290634,0.03270690143108368,0.03464135900139809,0.036643266677856445,0.03812652826309204,0.039326757192611694,0.04115968197584152,0.04286473989486694,0.04439994692802429,0.045362215489149094,0.0457846075296402,0.04602726921439171,0.045405659824609756,0.04494846612215042,0.04508977755904198,0.044802989810705185,0.04438367486000061,0.044127143919467926,0.04446050524711609,0.044314801692962646,0.04480468109250069,0.0432770699262619,0.04340970143675804,0.042841680347919464,0.04378228634595871,0.0435987263917923,0.04372811317443848,0.043370362371206284,0.043718092143535614,0.04397675022482872,0.04407918453216553,0.043996188789606094,0.04342690855264664,0.04383907467126846,0.04441108927130699,0.04456271603703499,0.0447361059486866,0.044320136308670044,0.04327201098203659,0.043457645922899246,0.04316595941781998,0.04305579513311386,0.04286554455757141,0.04281693696975708,0.04249044880270958,0.04264501482248306,0.042960528284311295,0.042576104402542114,0.04273608326911926,0.041710853576660156,0.04142852872610092,0.042193327099084854,0.04165973886847496,0.040922120213508606,0.0417061522603035,0.041548050940036774,0.040775712579488754,0.041922520846128464,0.04201730340719223,0.04230133071541786,0.042541950941085815,0.042059462517499924,0.04194073751568794,0.04261467233300209,0.04317449405789375,0.04361341893672943,0.044406741857528687,0.04507114738225937,0.04559807851910591,0.046338215470314026,0.04661919176578522,0.046180956065654755,0.04592975229024887,0.045918770134449005,0.04467355087399483,0.043579697608947754,0.04372145235538483,0.044027749449014664,0.04435843229293823,0.04609616845846176,0.047679442912340164]
  curve_l2_q75: [0.011992122046649456,0.01178180892020464,0.012974120676517487,0.017195865511894226,0.02397281862795353,0.03125434368848801,0.03698160871863365,0.040470924228429794,0.04172009974718094,0.04124011471867561,0.03949274122714996,0.03826536238193512,0.03927238658070564,0.04327080026268959,0.04681115970015526,0.05087456852197647,0.0542360320687294,0.056107811629772186,0.05801183730363846,0.059569310396909714,0.06241707503795624,0.06650331616401672,0.0691382884979248,0.07184644043445587,0.07205390185117722,0.07330650091171265,0.07602091133594513,0.07630808651447296,0.0776982307434082,0.07672049850225449,0.07903577387332916,0.07942059636116028,0.07993744313716888,0.0798814445734024,0.08178628981113434,0.08079956471920013,0.08033742010593414,0.08031443506479263,0.08063353598117828,0.07828755676746368,0.07717481255531311,0.07691731303930283,0.0771571546792984,0.07494708895683289,0.07509515434503555,0.077145054936409,0.07732116430997849,0.07698553055524826,0.07607056200504303,0.07530353963375092,0.07525460422039032,0.07487420737743378,0.07516314089298248,0.07507163286209106,0.07473614066839218,0.07450307160615921,0.0737198144197464,0.07267578691244125,0.07227933406829834,0.07168915867805481,0.07050883024930954,0.06985761970281601,0.06950785219669342,0.07013852894306183,0.06894098967313766,0.06966210156679153,0.07089312374591827,0.07062169909477234,0.07136419415473938,0.07154961675405502,0.07123735547065735,0.07114848494529724,0.0707588717341423,0.07079441845417023,0.07062837481498718,0.07119211554527283,0.0711594969034195,0.07017752528190613,0.07029874622821808,0.07153740525245667,0.0729413852095604,0.07456962019205093,0.07627198100090027,0.07925388962030411,0.08051666617393494,0.08113344013690948,0.08276278525590897,0.08404087275266647,0.085039883852005,0.08603381365537643,0.08776761591434479,0.08726959675550461,0.08880430459976196,0.08915401250123978,0.08879334479570389,0.08825865387916565,0.08846672624349594,0.08709301054477692,0.08778393268585205,0.08817002177238464]
Deep GRU:
  n_test: 400
  seeds: [42,101,2024]
  parameter_count: 338868
  per_seed:
    42:
      n_test: 400
      position_mse_per_horizon: {1:2.4542090613977052e-05,5:0.00021902367006987333,10:0.0019723328296095133,25:0.0013728151097893715,50:0.0019572121091187,100:0.004653798881918192}
      position_mse_std_per_horizon: {1:1.4353059668792412e-05,5:0.00042984579340554774,10:0.003479565493762493,25:0.0017829389544203877,50:0.006524883676320314,100:0.0041227745823562145}
      velocity_mse_per_horizon: {1:0.002094642724841833,5:0.021876106038689613,10:0.016457831487059593,25:0.01365212444216013,50:0.013402590528130531,100:0.09914136677980423}
      velocity_mse_std_per_horizon: {1:0.0016062709037214518,5:0.049819689244031906,10:0.01903422735631466,25:0.06862213462591171,50:0.030039796605706215,100:0.047812338918447495}
      position_error_per_horizon: {1:0.011710223741829395,5:0.02823401428759098,10:0.08657421171665192,25:0.08041425049304962,50:0.08125118911266327,100:0.15600261092185974}
      position_error_std_per_horizon: {1:0.003181696403771639,5:0.022737249732017517,10:0.06587035208940506,25:0.04207659140229225,50:0.07170437276363373,100:0.05988302081823349}
      overall_l2_mean: 0.08895710855722427
      curve_l2_mean: [0.011710227467119694,0.012936849147081375,0.015377569943666458,0.01958545856177807,0.028234023600816727,0.04267622530460358,0.059298738837242126,0.0732165053486824,0.08182910084724426,0.08657418191432953,0.08961499482393265,0.09140978753566742,0.09196840971708298,0.09147698432207108,0.09044458717107773,0.08945178240537643,0.08819460868835449,0.08673474192619324,0.08545838296413422,0.08448611199855804,0.083757184445858,0.0829358696937561,0.08213184028863907,0.08131557703018188,0.08041427284479141,0.07965745776891708,0.0790896788239479,0.07884010672569275,0.07868099957704544,0.07875554263591766,0.07873041182756424,0.0785391628742218,0.07835937291383743,0.07840542495250702,0.07889144122600555,0.079542376101017,0.08011084794998169,0.08068080991506577,0.0811728984117508,0.0816752016544342,0.0820375606417656,0.0822618305683136,0.08245564252138138,0.08274068683385849,0.08298899978399277,0.08301376551389694,0.08275522291660309,0.08226950466632843,0.08170945197343826,0.08125117421150208,0.08081015944480896,0.0803743377327919,0.07996569573879242,0.0794353261590004,0.07876868546009064,0.07803218066692352,0.07743567228317261,0.07697881758213043,0.07679542899131775,0.07671397924423218,0.07664503157138824,0.07668892294168472,0.0768790990114212,0.0772380456328392,0.07775942981243134,0.07860761135816574,0.07934582978487015,0.08000033348798752,0.08072824776172638,0.08145380765199661,0.08218353241682053,0.08273789286613464,0.08338665962219238,0.08413241058588028,0.08505832403898239,0.08625296503305435,0.08764424175024033,0.08928694576025009,0.09134231507778168,0.09401753544807434,0.09727959334850311,0.10104680061340332,0.10525057464838028,0.1097986027598381,0.11462610960006714,0.11968834698200226,0.12487402558326721,0.13008227944374084,0.13519816100597382,0.14006121456623077,0.14450667798519135,0.14839953184127808,0.15159574151039124,0.15400546789169312,0.1555834263563156,0.15635846555233002,0.15650928020477295,0.15628424286842346,0.15601080656051636,0.15600261092185974]
      curve_l2_q25: [0.009478351101279259,0.01017272099852562,0.011766701005399227,0.012813815847039223,0.014714471995830536,0.02105771005153656,0.02959192730486393,0.03570323437452316,0.04157891497015953,0.0459580272436142,0.0509035661816597,0.054830849170684814,0.05883076786994934,0.06322035938501358,0.06523919105529785,0.06519902497529984,0.06491078436374664,0.0634026825428009,0.06086549535393715,0.05879303067922592,0.05807976424694061,0.05619264021515846,0.0556686669588089,0.053877055644989014,0.05286724865436554,0.05095981806516647,0.050490446388721466,0.04976462200284004,0.04864826425909996,0.046937014907598495,0.046399347484111786,0.04606068134307861,0.04640299081802368,0.046606190502643585,0.046461787074804306,0.046999938786029816,0.0487990640103817,0.04794653132557869,0.04746611416339874,0.04885627329349518,0.048191871494054794,0.04911239817738533,0.05130492523312569,0.05055953562259674,0.05122510343790054,0.04985232651233673,0.05001647025346756,0.04875965416431427,0.04944450780749321,0.048916496336460114,0.047622427344322205,0.0455724373459816,0.044093143194913864,0.04299855977296829,0.04349008575081825,0.04440745711326599,0.0435575433075428,0.04304378107190132,0.04369741678237915,0.04344150424003601,0.042374007403850555,0.0422545000910759,0.04376477748155594,0.04338464140892029,0.04405137896537781,0.04493892565369606,0.04532107338309288,0.04553879052400589,0.04663769528269768,0.047985754907131195,0.048659153282642365,0.04936990886926651,0.0505673922598362,0.05063352733850479,0.0518377311527729,0.05440463125705719,0.056664399802684784,0.05902652442455292,0.06143798679113388,0.06467253714799881,0.06892421841621399,0.07348835468292236,0.07855933904647827,0.08242770284414291,0.0866125226020813,0.0902511477470398,0.09454748034477234,0.09872601181268692,0.10228566825389862,0.1060367003083229,0.11041051149368286,0.11355872452259064,0.1152787059545517,0.1163279116153717,0.11691883206367493,0.1171337142586708,0.11542949080467224,0.11400089412927628,0.11176669597625732,0.11026079952716827]
      curve_l2_q75: [0.013284844346344471,0.01524124201387167,0.018666885793209076,0.023900197818875313,0.03136380761861801,0.042843908071517944,0.05621318146586418,0.06756602227687836,0.08025629818439484,0.09607288241386414,0.1096610426902771,0.10977985709905624,0.11112401634454727,0.10960274934768677,0.1086910218000412,0.10805574059486389,0.10617788136005402,0.10511156916618347,0.1042959913611412,0.10445041209459305,0.10358647257089615,0.10140027105808258,0.09799876809120178,0.09696739166975021,0.09690972417593002,0.09774155914783478,0.09638267755508423,0.0979766696691513,0.09671938419342041,0.09487634897232056,0.09408587962388992,0.09302689135074615,0.09196697175502777,0.09100266546010971,0.09215012192726135,0.09111463278532028,0.09219571202993393,0.09213030338287354,0.09413235634565353,0.0922267884016037,0.08976743370294571,0.09047317504882812,0.08932207524776459,0.08870378136634827,0.0895615667104721,0.08990523964166641,0.08989108353853226,0.08835598826408386,0.08773985505104065,0.0888194665312767,0.08867103606462479,0.08900199830532074,0.0877770334482193,0.08821802586317062,0.08771126717329025,0.08656007051467896,0.0856020376086235,0.0844317227602005,0.08313164114952087,0.08299274742603302,0.08381546288728714,0.08391959965229034,0.0837290957570076,0.08385282754898071,0.08322902768850327,0.08326134085655212,0.0851469486951828,0.0870918557047844,0.08997106552124023,0.09159211069345474,0.09327829629182816,0.09528878331184387,0.09671689569950104,0.09797121584415436,0.09987909346818924,0.10112786293029785,0.10443564504384995,0.10539252310991287,0.10756756365299225,0.11068210005760193,0.11334045231342316,0.11855312436819077,0.12043701112270355,0.12440324574708939,0.1334417462348938,0.14018753170967102,0.14696422219276428,0.1512857973575592,0.15781660377979279,0.1628587543964386,0.1698557436466217,0.17386232316493988,0.17761224508285522,0.18128591775894165,0.18289829790592194,0.18311011791229248,0.1842355728149414,0.1833610087633133,0.18251270055770874,0.18482831120491028]
      parameter_count: 338868
    101:
      n_test: 400
      position_mse_per_horizon: {1:2.3439408323611133e-05,5:0.00017956715601030737,10:0.0012532277032732964,25:0.001081736758351326,50:0.0019159243674948812,100:0.00423949770629406}
      position_mse_std_per_horizon: {1:1.703918314888142e-05,5:0.0003713626938406378,10:0.0019962964579463005,25:0.0015464599709957838,50:0.006341207306832075,100:0.004080161452293396}
      velocity_mse_per_horizon: {1:0.0018174861324951053,5:0.02370626851916313,10:0.02111205644905567,25:0.014078030362725258,50:0.013050921261310577,100:0.0912385955452919}
      velocity_mse_std_per_horizon: {1:0.001223776489496231,5:0.045546598732471466,10:0.020694399252533913,25:0.06875663250684738,50:0.030163472518324852,100:0.052103668451309204}
      position_error_per_horizon: {1:0.01123877614736557,5:0.025826405733823776,10:0.07329756021499634,25:0.07052647322416306,50:0.08197269588708878,100:0.14699159562587738}
      position_error_std_per_horizon: {1:0.003785015782341361,5:0.020258326083421707,10:0.04633394628763199,25:0.03894146531820297,50:0.06910878419876099,100:0.06189066544175148}
      overall_l2_mean: 0.08914615958929062
      curve_l2_mean: [0.01123877428472042,0.011844051070511341,0.012667122296988964,0.016294652596116066,0.025826405733823776,0.040827080607414246,0.05653553456068039,0.06750869005918503,0.07230140268802643,0.07329759001731873,0.0728687196969986,0.07229311764240265,0.07199030369520187,0.07158941775560379,0.07093757390975952,0.07036393880844116,0.06972451508045197,0.06926092505455017,0.06914464384317398,0.06925182044506073,0.06939729303121567,0.06946372985839844,0.06967507302761078,0.07005839049816132,0.07052646577358246,0.071259506046772,0.07221019268035889,0.073382169008255,0.07438769191503525,0.07539388537406921,0.07615213096141815,0.07655409723520279,0.07678425312042236,0.07714854180812836,0.07779243588447571,0.07859160751104355,0.0793847069144249,0.08023303747177124,0.08097298443317413,0.08171214908361435,0.08230435103178024,0.08263632655143738,0.08279716223478317,0.08286480605602264,0.08292464911937714,0.08285543322563171,0.08268513530492783,0.08247075974941254,0.08221497386693954,0.08197271078824997,0.08170467615127563,0.08150961995124817,0.08145491778850555,0.08136109262704849,0.08120652288198471,0.08096244186162949,0.08086512237787247,0.08098075538873672,0.08133473247289658,0.08174627274274826,0.08217579871416092,0.08256710320711136,0.08294285088777542,0.08332400023937225,0.08372404426336288,0.08444345742464066,0.0852392315864563,0.08622612804174423,0.08752462267875671,0.08899804949760437,0.090575210750103,0.09199881553649902,0.09352265298366547,0.09511109441518784,0.09674500674009323,0.09845361113548279,0.10018011182546616,0.10203460603952408,0.10420738160610199,0.10692386329174042,0.1103467345237732,0.11449728906154633,0.11927357316017151,0.12446604669094086,0.1298629343509674,0.13525961339473724,0.14041416347026825,0.14513900876045227,0.14925985038280487,0.15260034799575806,0.15502075850963593,0.15649265050888062,0.15702416002750397,0.1566752940416336,0.1555449664592743,0.15379716455936432,0.15172357857227325,0.14965353906154633,0.14795561134815216,0.1469915509223938]
      curve_l2_q25: [0.008638028055429459,0.009165991097688675,0.009596358053386211,0.011016931384801865,0.015241501852869987,0.02296299859881401,0.0321667455136776,0.03882893919944763,0.04179370403289795,0.042216069996356964,0.041563138365745544,0.041510917246341705,0.04101485759019852,0.04095544293522835,0.04195144772529602,0.041504889726638794,0.041718240827322006,0.04205187410116196,0.041416607797145844,0.0421038419008255,0.04447555169463158,0.044814638793468475,0.044189341366291046,0.045703258365392685,0.04489276185631752,0.04595150426030159,0.04750441014766693,0.0490308478474617,0.048266682773828506,0.05006607994437218,0.05007074400782585,0.04986527934670448,0.05066395923495293,0.050395675003528595,0.0508139468729496,0.050888631492853165,0.050955306738615036,0.051571883261203766,0.05072538182139397,0.05043904483318329,0.050733186304569244,0.05174729600548744,0.052304498851299286,0.05135372653603554,0.05103050544857979,0.050969913601875305,0.0513087622821331,0.050652552396059036,0.05002020671963692,0.04878579080104828,0.048006363213062286,0.04936375096440315,0.049175698310136795,0.05024247616529465,0.048841867595911026,0.048371072858572006,0.04821430891752243,0.04798029363155365,0.04805177450180054,0.04744621366262436,0.04651441425085068,0.047181807458400726,0.047390926629304886,0.046395689249038696,0.046744346618652344,0.046564217656850815,0.04808192327618599,0.04890597611665726,0.04882551729679108,0.050793662667274475,0.05119464546442032,0.05225048586726189,0.05306253582239151,0.055359143763780594,0.05780249834060669,0.060561541467905045,0.06345884501934052,0.0663452222943306,0.06851711869239807,0.07197882235050201,0.07780885696411133,0.08237416297197342,0.08864971995353699,0.0941423624753952,0.1001323014497757,0.10664144903421402,0.11042776703834534,0.11301994323730469,0.11721226572990417,0.11870421469211578,0.12024298310279846,0.12040376663208008,0.12044231593608856,0.11920767277479172,0.1172320544719696,0.1171397864818573,0.1139373630285263,0.11264052987098694,0.11068189144134521,0.10750554502010345]
      curve_l2_q75: [0.01304228138178587,0.013721121475100517,0.015036608092486858,0.01761803962290287,0.025815576314926147,0.04150030016899109,0.059706952422857285,0.07530149817466736,0.08366412669420242,0.08705153316259384,0.09017851203680038,0.09231261909008026,0.09421632438898087,0.09456846863031387,0.09368759393692017,0.09145050495862961,0.09176283329725266,0.0911899134516716,0.0897754356265068,0.09085094928741455,0.08981397747993469,0.0878232792019844,0.0858825072646141,0.08458341658115387,0.08438457548618317,0.08360615372657776,0.08389055728912354,0.08404474705457687,0.08624939620494843,0.08908791840076447,0.08789053559303284,0.08890308439731598,0.08829624205827713,0.0900033712387085,0.0906226709485054,0.09173242747783661,0.09112173318862915,0.09150096774101257,0.09100955724716187,0.08903421461582184,0.09336140006780624,0.09367889910936356,0.09215410053730011,0.09119847416877747,0.09195853769779205,0.09056804329156876,0.09033956378698349,0.08920812606811523,0.08897276222705841,0.08765741437673569,0.08979298174381256,0.08908908069133759,0.089836485683918,0.08921618014574051,0.09092600643634796,0.09080357849597931,0.09032393991947174,0.0893627479672432,0.09002777189016342,0.0914711058139801,0.09120956063270569,0.09225961565971375,0.09276837855577469,0.0934581607580185,0.09542185068130493,0.09739552438259125,0.09794236719608307,0.09983742237091064,0.10361383855342865,0.10631503909826279,0.10914389789104462,0.11084230989217758,0.11226879805326462,0.11622411012649536,0.11717759072780609,0.11935977637767792,0.12145267426967621,0.12463758885860443,0.12601904571056366,0.12943412363529205,0.13164857029914856,0.13557055592536926,0.1398913860321045,0.14334514737129211,0.15007904171943665,0.15347690880298615,0.1597646325826645,0.16800488531589508,0.17321552336215973,0.17876674234867096,0.18233293294906616,0.18486237525939941,0.18289311230182648,0.18250657618045807,0.1812165081501007,0.1756214201450348,0.17329992353916168,0.1704956293106079,0.1681998074054718,0.1697857677936554]
      parameter_count: 338868
    2024:
      n_test: 400
      position_mse_per_horizon: {1:3.842426303890534e-05,5:0.00020339724142104387,10:0.001175960642285645,25:0.0010834869462996721,50:0.001962520182132721,100:0.0051902965642511845}
      position_mse_std_per_horizon: {1:3.0613755370723084e-05,5:0.00032832121360115707,10:0.0018107081996276975,25:0.001689327065832913,50:0.006606964394450188,100:0.004954804200679064}
      velocity_mse_per_horizon: {1:0.0021718856878578663,5:0.02230638451874256,10:0.01799813285470009,25:0.013865874148905277,50:0.013516302220523357,100:0.07110939174890518}
      velocity_mse_std_per_horizon: {1:0.0021509325597435236,5:0.04418591782450676,10:0.021508410573005676,25:0.06727899610996246,50:0.030661223456263542,100:0.037223707884550095}
      position_error_per_horizon: {1:0.013924850150942802,5:0.029514361172914505,10:0.0710960403084755,25:0.07031835615634918,50:0.08257610350847244,100:0.16136150062084198}
      position_error_std_per_horizon: {1:0.0060534393414855,5:0.018689194694161415,10:0.044733840972185135,25:0.03944934904575348,50:0.07040107250213623,100:0.07144398987293243}
      overall_l2_mean: 0.08897868543863297
      curve_l2_mean: [0.013924852944910526,0.014872239902615547,0.015689656138420105,0.019171901047229767,0.029514366760849953,0.045407820492982864,0.060022301971912384,0.06824436038732529,0.0707014873623848,0.07109605520963669,0.07113202661275864,0.07111858576536179,0.07105720043182373,0.07068359106779099,0.0699249804019928,0.0692322850227356,0.06855546683073044,0.06809072196483612,0.06798465549945831,0.06819100677967072,0.06862647831439972,0.06905266642570496,0.06944490224123001,0.06989064812660217,0.07031834870576859,0.07085787504911423,0.07151325047016144,0.07233712822198868,0.07299380004405975,0.07369774580001831,0.07419382780790329,0.07445386052131653,0.07468011975288391,0.07506773620843887,0.07580100744962692,0.07673685252666473,0.07765968143939972,0.07858816534280777,0.07941465079784393,0.08020975440740585,0.08076827228069305,0.08118198066949844,0.0815584659576416,0.08189767599105835,0.08216748386621475,0.08237769454717636,0.08251936733722687,0.0825929194688797,0.08258935809135437,0.08257613331079483,0.08255971968173981,0.08249306678771973,0.08249881118535995,0.08246167004108429,0.08244118094444275,0.08238495886325836,0.08246821165084839,0.08260126411914825,0.08292637765407562,0.08325991034507751,0.08355852961540222,0.0837797075510025,0.08392559736967087,0.08400676399469376,0.08410792052745819,0.08448299765586853,0.08480515331029892,0.08520239591598511,0.08586888015270233,0.08671726286411285,0.08772389590740204,0.08860934525728226,0.08953270316123962,0.0904858335852623,0.09153842180967331,0.09274248778820038,0.09401817619800568,0.09542778879404068,0.09709308296442032,0.09929551184177399,0.10205817967653275,0.10547208040952682,0.10968277603387833,0.11472178250551224,0.12050662189722061,0.12684588134288788,0.13345763087272644,0.14010319113731384,0.14654229581356049,0.15251344442367554,0.15777988731861115,0.16214703023433685,0.16543683409690857,0.16754913330078125,0.1684560775756836,0.16822515428066254,0.16707482933998108,0.16529013216495514,0.16324365139007568,0.1613614708185196]
      curve_l2_q25: [0.008477547205984592,0.00938932690769434,0.009773625060915947,0.012314432300627232,0.018852708861231804,0.028017157688736916,0.036750972270965576,0.04055752605199814,0.03922395408153534,0.04210568219423294,0.04323580116033554,0.04420129209756851,0.045895788818597794,0.045006826519966125,0.04511681944131851,0.04298758506774902,0.043237559497356415,0.04396779462695122,0.042808935046195984,0.04246344044804573,0.04342418164014816,0.04444987699389458,0.04473096877336502,0.04514193907380104,0.0452212393283844,0.0462275967001915,0.04555004835128784,0.045996878296136856,0.046029046177864075,0.04534747079014778,0.045499175786972046,0.04524621739983559,0.044775594025850296,0.04514714330434799,0.04641600325703621,0.046498045325279236,0.04627298563718796,0.047090619802474976,0.04721161723136902,0.04679461568593979,0.048983246088027954,0.04888589680194855,0.049465615302324295,0.049162231385707855,0.049555398523807526,0.048993632197380066,0.048204533755779266,0.049556758254766464,0.04861825332045555,0.04952693358063698,0.05054754763841629,0.050890013575553894,0.050220049917697906,0.05013694986701012,0.050312064588069916,0.04918643459677696,0.04928601533174515,0.05003434792160988,0.050747670233249664,0.05028873682022095,0.049098484218120575,0.04941931739449501,0.04934161901473999,0.04928651824593544,0.04784180223941803,0.047758445143699646,0.048179443925619125,0.04904688522219658,0.049359969794750214,0.049939896911382675,0.051069166511297226,0.05204012244939804,0.05418839678168297,0.056071795523166656,0.05742776021361351,0.05914097651839256,0.06103789061307907,0.0642847865819931,0.06530337780714035,0.0688333660364151,0.07136064767837524,0.0747344121336937,0.07914109528064728,0.08290889859199524,0.08775060623884201,0.09336122125387192,0.09772971272468567,0.10358566790819168,0.10812216997146606,0.11047311872243881,0.11359808593988419,0.11576518416404724,0.11604916304349899,0.11634388566017151,0.1162186935544014,0.11580854654312134,0.11510532349348068,0.11380229890346527,0.1117027997970581,0.1085132360458374]
      curve_l2_q75: [0.018680432811379433,0.01994846947491169,0.02028585597872734,0.023746374994516373,0.032998234033584595,0.04925020411610603,0.06601551920175552,0.07845433801412582,0.08407580852508545,0.0860472097992897,0.08615969121456146,0.0864633321762085,0.08785860240459442,0.0894412100315094,0.09032823890447617,0.09083650261163712,0.08793377876281738,0.08576411008834839,0.08533895015716553,0.08491832762956619,0.08383559435606003,0.08393016457557678,0.0843227282166481,0.08190329372882843,0.0818115621805191,0.08288028836250305,0.08313153684139252,0.08604252338409424,0.08699044585227966,0.08660098910331726,0.08635241538286209,0.08565466105937958,0.08664882183074951,0.08704563975334167,0.08960002660751343,0.0898253470659256,0.09047313779592514,0.0908408910036087,0.08992411196231842,0.0903966873884201,0.09066899120807648,0.09013424068689346,0.09182862192392349,0.09177367389202118,0.09267101436853409,0.09422723203897476,0.09338058531284332,0.09087806940078735,0.0909152626991272,0.0898841917514801,0.08991137146949768,0.09114443510770798,0.09163010120391846,0.09261927753686905,0.09284602850675583,0.09229244291782379,0.09194978326559067,0.0906514823436737,0.09159543365240097,0.09203959256410599,0.09193725883960724,0.09230310469865799,0.0927271917462349,0.09173080325126648,0.09206432104110718,0.09246170520782471,0.092209093272686,0.09480850398540497,0.09685541689395905,0.10046649724245071,0.10216927528381348,0.1050114557147026,0.10571903735399246,0.1076243445277214,0.10884363949298859,0.10995379090309143,0.11168262362480164,0.1136883944272995,0.11614967882633209,0.11851979047060013,0.12021775543689728,0.12551698088645935,0.12896084785461426,0.13426388800144196,0.14240388572216034,0.14954975247383118,0.1580910086631775,0.16638526320457458,0.17837825417518616,0.18739336729049683,0.195977121591568,0.20366749167442322,0.20741644501686096,0.20988234877586365,0.21371497213840485,0.2121884524822235,0.20864373445510864,0.2047862410545349,0.1995030641555786,0.19920778274536133]
      parameter_count: 338868
  position_mse_per_horizon: {1:2.8801920658831175e-05,5:0.00020066268916707486,10:0.0014671737250561516,25:0.0011793462714801233,50:0.0019452188862487674,100:0.004694531050821145}
  position_mse_std_per_horizon: {1:2.0668666062798973e-05,5:0.0003765099002824475,10:0.0024288567171121636,25:0.0016729086637496948,50:0.006491018459200859,100:0.004385913411776225}
  position_mse_seed_std_per_horizon: {1:8.35141195953715e-06,5:1.986988820943507e-05,10:0.0004391831512743664,25:0.00016755121407202987,50:2.550824438386319e-05,100:0.0004767063553811688}
  velocity_mse_per_horizon: {1:0.0020280048483982682,5:0.0226295863588651,10:0.01852267359693845,25:0.013865342984596888,50:0.013323271336654821,100:0.0871631180246671}
  velocity_mse_std_per_horizon: {1:0.0016603266509870689,5:0.046517401933670044,10:0.020412345727284748,25:0.06821925441424052,50:0.030288164193431538,100:0.045713238418102264}
  velocity_mse_seed_std_per_horizon: {1:0.00018636044683089688,5:0.0009569316842529526,10:0.0023710355842791204,25:0.00021295345710917497,50:0.00024261803292773183,100:0.014453547792811128}
  position_error_per_horizon: {1:0.01229128334671259,5:0.027858260398109753,10:0.07698927074670792,25:0.07375302662452062,50:0.0819333295027415,100:0.1547852357228597}
  position_error_std_per_horizon: {1:0.0043400505091995,5:0.02056159016986688,10:0.052312713116407394,25:0.0401558019220829,50:0.07040474315484364,100:0.0644058920443058}
  position_error_seed_std_per_horizon: {1:0.0014342144492695697,5:0.0018724708460413808,10:0.008373469563011809,25:0.005769727528096472,50:0.0006633338700971255,100:0.00726188984854717}
  overall_l2_mean: 0.08902731786171596
  overall_l2_seed_std: 0.0001034838519409168
  curve_l2_mean: [0.011710227467119694,0.012936849147081375,0.015377569943666458,0.01958545856177807,0.028234023600816727,0.04267622530460358,0.059298738837242126,0.0732165053486824,0.08182910084724426,0.08657418191432953,0.08961499482393265,0.09140978753566742,0.09196840971708298,0.09147698432207108,0.09044458717107773,0.08945178240537643,0.08819460868835449,0.08673474192619324,0.08545838296413422,0.08448611199855804,0.083757184445858,0.0829358696937561,0.08213184028863907,0.08131557703018188,0.08041427284479141,0.07965745776891708,0.0790896788239479,0.07884010672569275,0.07868099957704544,0.07875554263591766,0.07873041182756424,0.0785391628742218,0.07835937291383743,0.07840542495250702,0.07889144122600555,0.079542376101017,0.08011084794998169,0.08068080991506577,0.0811728984117508,0.0816752016544342,0.0820375606417656,0.0822618305683136,0.08245564252138138,0.08274068683385849,0.08298899978399277,0.08301376551389694,0.08275522291660309,0.08226950466632843,0.08170945197343826,0.08125117421150208,0.08081015944480896,0.0803743377327919,0.07996569573879242,0.0794353261590004,0.07876868546009064,0.07803218066692352,0.07743567228317261,0.07697881758213043,0.07679542899131775,0.07671397924423218,0.07664503157138824,0.07668892294168472,0.0768790990114212,0.0772380456328392,0.07775942981243134,0.07860761135816574,0.07934582978487015,0.08000033348798752,0.08072824776172638,0.08145380765199661,0.08218353241682053,0.08273789286613464,0.08338665962219238,0.08413241058588028,0.08505832403898239,0.08625296503305435,0.08764424175024033,0.08928694576025009,0.09134231507778168,0.09401753544807434,0.09727959334850311,0.10104680061340332,0.10525057464838028,0.1097986027598381,0.11462610960006714,0.11968834698200226,0.12487402558326721,0.13008227944374084,0.13519816100597382,0.14006121456623077,0.14450667798519135,0.14839953184127808,0.15159574151039124,0.15400546789169312,0.1555834263563156,0.15635846555233002,0.15650928020477295,0.15628424286842346,0.15601080656051636,0.15600261092185974]
  curve_l2_q25: [0.009478351101279259,0.01017272099852562,0.011766701005399227,0.012813815847039223,0.014714471995830536,0.02105771005153656,0.02959192730486393,0.03570323437452316,0.04157891497015953,0.0459580272436142,0.0509035661816597,0.054830849170684814,0.05883076786994934,0.06322035938501358,0.06523919105529785,0.06519902497529984,0.06491078436374664,0.0634026825428009,0.06086549535393715,0.05879303067922592,0.05807976424694061,0.05619264021515846,0.0556686669588089,0.053877055644989014,0.05286724865436554,0.05095981806516647,0.050490446388721466,0.04976462200284004,0.04864826425909996,0.046937014907598495,0.046399347484111786,0.04606068134307861,0.04640299081802368,0.046606190502643585,0.046461787074804306,0.046999938786029816,0.0487990640103817,0.04794653132557869,0.04746611416339874,0.04885627329349518,0.048191871494054794,0.04911239817738533,0.05130492523312569,0.05055953562259674,0.05122510343790054,0.04985232651233673,0.05001647025346756,0.04875965416431427,0.04944450780749321,0.048916496336460114,0.047622427344322205,0.0455724373459816,0.044093143194913864,0.04299855977296829,0.04349008575081825,0.04440745711326599,0.0435575433075428,0.04304378107190132,0.04369741678237915,0.04344150424003601,0.042374007403850555,0.0422545000910759,0.04376477748155594,0.04338464140892029,0.04405137896537781,0.04493892565369606,0.04532107338309288,0.04553879052400589,0.04663769528269768,0.047985754907131195,0.048659153282642365,0.04936990886926651,0.0505673922598362,0.05063352733850479,0.0518377311527729,0.05440463125705719,0.056664399802684784,0.05902652442455292,0.06143798679113388,0.06467253714799881,0.06892421841621399,0.07348835468292236,0.07855933904647827,0.08242770284414291,0.0866125226020813,0.0902511477470398,0.09454748034477234,0.09872601181268692,0.10228566825389862,0.1060367003083229,0.11041051149368286,0.11355872452259064,0.1152787059545517,0.1163279116153717,0.11691883206367493,0.1171337142586708,0.11542949080467224,0.11400089412927628,0.11176669597625732,0.11026079952716827]
  curve_l2_q75: [0.013284844346344471,0.01524124201387167,0.018666885793209076,0.023900197818875313,0.03136380761861801,0.042843908071517944,0.05621318146586418,0.06756602227687836,0.08025629818439484,0.09607288241386414,0.1096610426902771,0.10977985709905624,0.11112401634454727,0.10960274934768677,0.1086910218000412,0.10805574059486389,0.10617788136005402,0.10511156916618347,0.1042959913611412,0.10445041209459305,0.10358647257089615,0.10140027105808258,0.09799876809120178,0.09696739166975021,0.09690972417593002,0.09774155914783478,0.09638267755508423,0.0979766696691513,0.09671938419342041,0.09487634897232056,0.09408587962388992,0.09302689135074615,0.09196697175502777,0.09100266546010971,0.09215012192726135,0.09111463278532028,0.09219571202993393,0.09213030338287354,0.09413235634565353,0.0922267884016037,0.08976743370294571,0.09047317504882812,0.08932207524776459,0.08870378136634827,0.0895615667104721,0.08990523964166641,0.08989108353853226,0.08835598826408386,0.08773985505104065,0.0888194665312767,0.08867103606462479,0.08900199830532074,0.0877770334482193,0.08821802586317062,0.08771126717329025,0.08656007051467896,0.0856020376086235,0.0844317227602005,0.08313164114952087,0.08299274742603302,0.08381546288728714,0.08391959965229034,0.0837290957570076,0.08385282754898071,0.08322902768850327,0.08326134085655212,0.0851469486951828,0.0870918557047844,0.08997106552124023,0.09159211069345474,0.09327829629182816,0.09528878331184387,0.09671689569950104,0.09797121584415436,0.09987909346818924,0.10112786293029785,0.10443564504384995,0.10539252310991287,0.10756756365299225,0.11068210005760193,0.11334045231342316,0.11855312436819077,0.12043701112270355,0.12440324574708939,0.1334417462348938,0.14018753170967102,0.14696422219276428,0.1512857973575592,0.15781660377979279,0.1628587543964386,0.1698557436466217,0.17386232316493988,0.17761224508285522,0.18128591775894165,0.18289829790592194,0.18311011791229248,0.1842355728149414,0.1833610087633133,0.18251270055770874,0.18482831120491028]
Continuous ConvLSTM:
  n_test: 400
  position_mse_per_horizon: {1:2.1441708668135107e-05,5:7.792786345817149e-05,10:0.00011915400682482868,25:0.0005929369363002479,50:0.0014975055819377303,100:0.0070694806054234505}
  position_mse_std_per_horizon: {1:7.309350621653721e-06,5:6.898444553371519e-05,10:0.0001360190217383206,25:0.0013037155149504542,50:0.006751906592398882,100:0.004279767628759146}
  velocity_mse_per_horizon: {1:0.0014696833677589893,5:0.01790624111890793,10:0.005532163195312023,25:0.012077036313712597,50:0.010949527844786644,100:0.33696478605270386}
  velocity_mse_std_per_horizon: {1:0.00039774991455487907,5:0.03252099081873894,10:0.00845268089324236,25:0.06773145496845245,50:0.030475839972496033,100:0.2892436981201172}
  position_error_per_horizon: {1:0.011174989864230156,5:0.020498191937804222,10:0.024150032550096512,25:0.04663147032260895,50:0.06240079924464226,100:0.19661499559879303}
  position_error_std_per_horizon: {1:0.0019416138529777527,5:0.006884132046252489,10:0.011476059444248676,25:0.03719042241573334,50:0.07135245203971863,100:0.06131419911980629}
  overall_l2_mean: 0.06597094982862473
  curve_l2_mean: [0.011174994520843029,0.012042036280035973,0.01443552691489458,0.017500868067145348,0.020498182624578476,0.021310806274414062,0.020565588027238846,0.021205374971032143,0.022967524826526642,0.024150045588612556,0.024516183882951736,0.024949219077825546,0.026285680010914803,0.028364524245262146,0.030501186847686768,0.032273806631565094,0.03339214250445366,0.034676793962717056,0.03613998368382454,0.037844106554985046,0.03972066938877106,0.04154765605926514,0.043331246823072433,0.0450793020427227,0.04663148522377014,0.04822834953665733,0.04971938207745552,0.05125514045357704,0.05258331447839737,0.05388814955949783,0.05481310933828354,0.055385150015354156,0.055821023881435394,0.05629447102546692,0.05709781497716904,0.058017853647470474,0.05881661921739578,0.05966746434569359,0.06044695898890495,0.06126915290951729,0.06192292273044586,0.06230604648590088,0.06256699562072754,0.06291200965642929,0.06323806941509247,0.06333784759044647,0.06323898583650589,0.06305358558893204,0.06275155395269394,0.06240081787109375,0.062160566449165344,0.062108177691698074,0.0622650682926178,0.06249019131064415,0.06273999810218811,0.0628221333026886,0.0628344938158989,0.06280059367418289,0.062872976064682,0.06291487067937851,0.06296313554048538,0.06303594261407852,0.06318947672843933,0.06347059458494186,0.06378131359815598,0.06423491984605789,0.06440916657447815,0.06430461257696152,0.06415984034538269,0.0639825090765953,0.0637296736240387,0.06313662230968475,0.062484949827194214,0.061857547610998154,0.06139730289578438,0.06127776950597763,0.06141214817762375,0.0617896169424057,0.062487974762916565,0.06368053704500198,0.06556401401758194,0.06816024333238602,0.07155437767505646,0.07578963041305542,0.08099126070737839,0.08719971030950546,0.09429249912500381,0.10222319513559341,0.1108214482665062,0.11986123770475388,0.12912219762802124,0.13840486109256744,0.1475297212600708,0.15630219876766205,0.1645827293395996,0.17229579389095306,0.1794155091047287,0.18586070835590363,0.19157622754573822,0.1966150850057602]
  curve_l2_q25: [0.009833129122853279,0.010758273303508759,0.013283619657158852,0.01570872962474823,0.01714543253183365,0.016837768256664276,0.015808772295713425,0.015543938614428043,0.016716239973902702,0.017579250037670135,0.017777778208255768,0.016720665618777275,0.016725953668355942,0.017037972807884216,0.017331210896372795,0.01736113429069519,0.016791505739092827,0.01682026870548725,0.017166072502732277,0.017936432734131813,0.018573729321360588,0.01929551362991333,0.019927844405174255,0.020038869231939316,0.0211661234498024,0.022316675633192062,0.02395901083946228,0.024909893050789833,0.02557767927646637,0.02668934501707554,0.027532657608389854,0.027772005647420883,0.028711918741464615,0.029223356395959854,0.0289594866335392,0.029934661462903023,0.030316773802042007,0.031389959156513214,0.03160780668258667,0.031620435416698456,0.032050326466560364,0.03161490336060524,0.032776135951280594,0.03333061560988426,0.03366544842720032,0.03368096798658371,0.033055394887924194,0.03299317881464958,0.032334838062524796,0.03177057206630707,0.03161175549030304,0.03196222335100174,0.03209872171282768,0.03225778788328171,0.03272994980216026,0.03264808654785156,0.032329022884368896,0.03212696686387062,0.03229958191514015,0.03227371349930763,0.03268927335739136,0.033167362213134766,0.03393799439072609,0.03478199988603592,0.03520913049578667,0.03574232757091522,0.036027900874614716,0.03570816293358803,0.03667859733104706,0.038083355873823166,0.03855827450752258,0.03804157301783562,0.03743710741400719,0.03715921938419342,0.036280982196331024,0.036378610879182816,0.03630409017205238,0.03594471514225006,0.03614545986056328,0.03732176870107651,0.039269618690013885,0.041191279888153076,0.04373405873775482,0.048853978514671326,0.05364612489938736,0.0593610480427742,0.06577001512050629,0.07310430705547333,0.08149212598800659,0.09046413004398346,0.09762122482061386,0.10378028452396393,0.11048278957605362,0.11838661879301071,0.1260734349489212,0.13335299491882324,0.14146995544433594,0.14728763699531555,0.15219640731811523,0.158975750207901]
  curve_l2_q75: [0.012447676621377468,0.013363772071897984,0.015635773539543152,0.019114866852760315,0.022028185427188873,0.022444844245910645,0.021469227969646454,0.021697554737329483,0.02428925782442093,0.02647428959608078,0.02918153628706932,0.02993505820631981,0.029217882081866264,0.030611759051680565,0.03208623081445694,0.035104285925626755,0.03713863715529442,0.0406099297106266,0.04331076890230179,0.045834559947252274,0.04983978345990181,0.05376612767577171,0.05646521970629692,0.05892810970544815,0.060530971735715866,0.06101080775260925,0.06277523189783096,0.06375271081924438,0.06556684523820877,0.06629437953233719,0.06644044816493988,0.06751911342144012,0.06732741743326187,0.06795438379049301,0.06932401657104492,0.06846349686384201,0.069176584482193,0.06888626515865326,0.06911268830299377,0.07017932832241058,0.07026572525501251,0.06961970031261444,0.06910116970539093,0.0687602087855339,0.06860695034265518,0.06791669130325317,0.06730496138334274,0.06692463159561157,0.0668257623910904,0.06621240824460983,0.06483574211597443,0.06652463972568512,0.06753331422805786,0.06403781473636627,0.06433472782373428,0.06505073606967926,0.06519804149866104,0.06441941112279892,0.06383206695318222,0.06350776553153992,0.0652681291103363,0.06402671337127686,0.0635308101773262,0.0631193071603775,0.06232646852731705,0.06276586651802063,0.06329290568828583,0.06246981769800186,0.06200810894370079,0.06171587482094765,0.06155461072921753,0.06183331832289696,0.060780834406614304,0.06188236176967621,0.06334898620843887,0.0645117312669754,0.06501104682683945,0.066200390458107,0.0684833973646164,0.07127930223941803,0.0732201561331749,0.07558955997228622,0.07943040132522583,0.08390167355537415,0.09107397496700287,0.09991734474897385,0.11109901964664459,0.12169697135686874,0.1302212029695511,0.13809168338775635,0.1500256061553955,0.16216638684272766,0.17342667281627655,0.184728741645813,0.19388610124588013,0.20224247872829437,0.21095040440559387,0.21967104077339172,0.2265009731054306,0.23259736597537994]
  parameter_count: 296892
```

### `results/multiseed/multiseed_metrics.json`
> 3-seed (42, 101, 2024) SPWM, ANN-Rate, Continuous RSSM, and equalized Deep GRU open-loop, 50-ep PIL (open-loop, ungated, residual-gated), 5-stage ablation, and energy summary

```yaml
seeds: [42,101,2024]
open_loop_spwm:
  position_mse_per_horizon: {1:0.00018221767580447099,5:0.0011347631613413494,10:0.0034783149991805353,25:0.0021969674853608012,50:0.0031450896058231592,100:0.0024013782385736704}
  position_mse_seed_std_per_horizon: {1:0.00010954467998152894,5:0.0008189706679996508,10:0.0026521253541566974,25:0.0004156317481905645,50:0.001442577257336728,100:0.000366950848819243}
  velocity_mse_per_horizon: {1:0.001385071101443221,5:0.04249462919930617,10:0.045068783685564995,25:0.036605274925629296,50:0.02465990123649438,100:0.009707977995276451}
  velocity_mse_seed_std_per_horizon: {1:0.000763865468875661,5:0.012683060543225525,10:0.026009623700082144,25:0.01818887773598169,50:0.012712159468845796,100:0.0017663810650883683}
  per_seed:
    42:
      position_mse_per_horizon: {1:0.00015687845007050782,5:0.0020792167633771896,10:0.006488359998911619,25:0.002526428084820509,50:0.002635047771036625,100:0.0024581069592386484}
      velocity_mse_per_horizon: {1:0.0013571062590926886,5:0.05235695838928223,10:0.07470371574163437,25:0.03928713873028755,50:0.01999685727059841,100:0.011747471988201141}
    101:
      position_mse_per_horizon: {1:0.0003022114688064903,5:0.0007039911579340696,10:0.0014849373837932944,25:0.002334464807063341,50:0.004773399326950312,100:0.0027366611175239086}
      velocity_mse_per_horizon: {1:0.0006355720688588917,5:0.028187384828925133,10:0.026028921827673912,25:0.05330432578921318,50:0.039045095443725586,100:0.008709450252354145}
    2024:
      position_mse_per_horizon: {1:8.75631085364148e-05,5:0.0006210815627127886,10:0.002461647614836693,25:0.0017300095641985536,50:0.002026821719482541,100:0.002009366638958454}
      velocity_mse_per_horizon: {1:0.0021625349763780832,5:0.04693954437971115,10:0.034473713487386703,25:0.01722436025738716,50:0.01493775099515915,100:0.008667011745274067}
pil_50ep_mean_std:
  SPWM-v0.12.4 (N=25 Val-Selected):
    grasp_success_rate_pct: [60.666666666666664,2.3094010767585074]
    terminal_placement_accuracy_cm: [0.8798704931139946,0.10505539807183523]
    terminal_error_cm: [2.006662103931109,0.11581144347528279]
    mean_joint_accel_rad_s2: [0.009308865414932371,0.0034586698705177275]
    mean_joint_jerk_rad_s3: [0.006634432549277941,0.003263484407032791]
    mean_direction_flips: [21.14,8.549807015365904]
    mppi_inference_latency_ms: [26.649352693166346,2.271980219747186]
  SPWM-v0.12.4 (N=6 Tier-2):
    grasp_success_rate_pct: [58.0,27.055498516937366]
    terminal_placement_accuracy_cm: [0.8992845916748048,0.0417493623118023]
    terminal_error_cm: [2.231203550299009,0.6846823009200784]
    mean_joint_accel_rad_s2: [0.008598180722134809,0.003043671174468545]
    mean_joint_jerk_rad_s3: [0.00608342978792886,0.0029084716855952435]
    mean_direction_flips: [19.373333333333335,7.574545090850891]
    mppi_inference_latency_ms: [27.510263026675336,7.919433452921299]
  ANN-Rate:
    grasp_success_rate_pct: [37.333333333333336,9.451631252505218]
    terminal_placement_accuracy_cm: [1.0678065086901187,0.2828632725018437]
    terminal_error_cm: [2.9846295044819513,0.4621451582711122]
    mean_joint_accel_rad_s2: [0.011481044394895437,0.0008839583251549979]
    mean_joint_jerk_rad_s3: [0.009379258422801892,0.0010705614356774748]
    mean_direction_flips: [26.0,1.94648400969543]
    mppi_inference_latency_ms: [24.616702500178565,0.07683415875005363]
  Continuous RSSM (Open-Loop PIL):
    grasp_success_rate_pct: [15.333333333333334,7.023769168568492]
    terminal_placement_accuracy_cm: [1.2335922479629515,0.05552042269442134]
    terminal_error_cm: [2.738770661751429,0.06877343601692858]
    mean_joint_accel_rad_s2: [0.02037182800471783,0.000592333680216826]
    mean_joint_jerk_rad_s3: [0.02138510186225176,0.0004846155447845476]
    mean_direction_flips: [36.85999999999999,1.2150720143267217]
    mppi_inference_latency_ms: [46.81975616126076,0.1485665027581289]
  Continuous RSSM (Un-Gated Online):
    grasp_success_rate_pct: [0.0,0.0]
    terminal_placement_accuracy_cm: [1.425072744091352,0.03429525285538392]
    terminal_error_cm: [25.49343025525411,1.3050838203645958]
    mean_joint_accel_rad_s2: [0.03153110054632028,0.0006715274430858012]
    mean_joint_jerk_rad_s3: [0.029558277900020278,0.0004172700652986988]
    mean_direction_flips: [36.49333333333333,0.7757147241952662]
    mppi_inference_latency_ms: [46.936845150350116,0.33605994404428824]
  Continuous RSSM (Residual-Gated Online):
    grasp_success_rate_pct: [15.333333333333334,5.033222956847166]
    terminal_placement_accuracy_cm: [1.233093837896983,0.046829598001233756]
    terminal_error_cm: [2.7240475718180335,0.10990075377541059]
    mean_joint_accel_rad_s2: [0.020493890543778735,0.0006706944714435922]
    mean_joint_jerk_rad_s3: [0.021609529219567773,0.0005316353717258448]
    mean_direction_flips: [36.940000000000005,1.2257242756835658]
    mppi_inference_latency_ms: [46.472593615331895,0.1711331627137406]
  Deep GRU (Open-Loop PIL):
    grasp_success_rate_pct: [77.33333333333333,4.163331998932265]
    terminal_placement_accuracy_cm: [1.142497578263283,0.31404892460065076]
    terminal_error_cm: [1.8994051045179365,0.17642912388300128]
    mean_joint_accel_rad_s2: [0.004498507350217551,0.00012105405328595289]
    mean_joint_jerk_rad_s3: [0.002675323402121042,0.00018613435682786266]
    mean_direction_flips: [10.793333333333331,0.6331929669013493]
    mppi_inference_latency_ms: [57.845663712604754,0.2935253213907278]
  Deep GRU (Un-Gated Online):
    grasp_success_rate_pct: [0.0,0.0]
    terminal_placement_accuracy_cm: [1.4674344965815544,0.5106457690318531]
    terminal_error_cm: [7.351427071889241,1.3439473080339845]
    mean_joint_accel_rad_s2: [0.006980255008675158,0.00042445948743814776]
    mean_joint_jerk_rad_s3: [0.004832257371939098,0.00042073079965799826]
    mean_direction_flips: [15.079999999999998,0.647147587494537]
    mppi_inference_latency_ms: [57.07348911329685,0.5185482382876814]
  Deep GRU (Residual-Gated Online):
    grasp_success_rate_pct: [76.66666666666667,5.033222956847166]
    terminal_placement_accuracy_cm: [1.142546683549881,0.3141746410639535]
    terminal_error_cm: [1.9001078454653422,0.1792502651931432]
    mean_joint_accel_rad_s2: [0.004493320606804143,0.0001224338129939779]
    mean_joint_jerk_rad_s3: [0.0026687012903857978,0.0001826580482518807]
    mean_direction_flips: [10.786666666666667,0.6057502235520298]
    mppi_inference_latency_ms: [57.65735650555458,0.21359715146433306]
energy_spwm_mean_std:
  spike_rate_pct: [12.151117412707743,0.192861046198146]
  sops_per_step: [8150.0,129.11622671066561]
  energy_neuro_nj: [12.224996206577009,0.1940336413590309]
  speedup_vs_own_gpu: [4.938654535978057,0.07910719976272344]
robustness_pil_3seed:
  SPWM-v0.12.4 (N=25 Val-Selected) (5 items):
    - [0]
      condition_name: Nominal Baseline
      grasp_success_rate_pct: [60.666666666666664,2.3094010767585074]
      delta_rel_pct: 0.0
      min_placement_error_mean_cm: [0.8798704931139946,0.10505539807183523]
      terminal_placement_error_mean_cm: [2.006662103931109,0.11581144347528279]
      terminal_velocity_mean: [0.4501866717636585,0.07299750426577846]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 57.99999999999999
          min_placement_error_mean_cm: 0.9966417002677918
          terminal_placement_error_mean_cm: 2.117010636925697
          terminal_velocity_mean: 0.37095459580421447
          mean_latency_ms: 32.431992493769116
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 0.0
        101:
          grasp_success_rate_pct: 62.0
          min_placement_error_mean_cm: 0.8499453981220723
          terminal_placement_error_mean_cm: 2.0169071877002716
          terminal_velocity_mean: 0.5147094203531742
          mean_latency_ms: 39.79635441773567
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 0.0
        2024:
          grasp_success_rate_pct: 62.0
          min_placement_error_mean_cm: 0.7930243809521198
          terminal_placement_error_mean_cm: 1.8860684871673583
          terminal_velocity_mean: 0.46489599913358687
          mean_latency_ms: 39.787420788930085
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 0.0
    - [1]
      condition_name: Sensory Noise Injection (1.0%)
      grasp_success_rate_pct: [54.666666666666664,1.1547005383792557]
      delta_rel_pct: -9.89010989010989
      min_placement_error_mean_cm: [0.9180183924237887,0.24189672923787867]
      terminal_placement_error_mean_cm: [2.259123495916526,0.45390662193171594]
      terminal_velocity_mean: [0.40925723339120545,0.07120021870960654]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 54.0
          min_placement_error_mean_cm: 1.1959887701272964
          terminal_placement_error_mean_cm: 2.683878175020218
          terminal_velocity_mean: 0.33141832798719406
          mean_latency_ms: 32.060862347789936
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -6.89655172413792
        101:
          grasp_success_rate_pct: 54.0
          min_placement_error_mean_cm: 0.7552975878119469
          terminal_placement_error_mean_cm: 2.3126762568950654
          terminal_velocity_mean: 0.4710961046814919
          mean_latency_ms: 32.065856869886694
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -12.903225806451612
        2024:
          grasp_success_rate_pct: 56.00000000000001
          min_placement_error_mean_cm: 0.8027688193321229
          terminal_placement_error_mean_cm: 1.7808160558342934
          terminal_velocity_mean: 0.4252572675049305
          mean_latency_ms: 32.0065951976089
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -9.6774193548387
    - [2]
      condition_name: Payload Extrapolation (4.80 kg)
      grasp_success_rate_pct: [44.666666666666664,13.316656236958789]
      delta_rel_pct: -26.373626373626376
      min_placement_error_mean_cm: [1.3093093851208688,0.41128174441194415]
      terminal_placement_error_mean_cm: [2.4906945570309955,0.29389661766704617]
      terminal_velocity_mean: [0.6077169207235178,0.10828262151038422]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 56.00000000000001
          min_placement_error_mean_cm: 1.7644587180018425
          terminal_placement_error_mean_cm: 2.4560093522071837
          terminal_velocity_mean: 0.5556513021886349
          mean_latency_ms: 32.58270962906839
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -3.4482758620689413
        101:
          grasp_success_rate_pct: 30.0
          min_placement_error_mean_cm: 1.1991314259171486
          terminal_placement_error_mean_cm: 2.800394687652588
          terminal_velocity_mean: 0.7321977093815804
          mean_latency_ms: 32.619208271916534
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -51.61290322580645
        2024:
          grasp_success_rate_pct: 48.0
          min_placement_error_mean_cm: 0.964338011443615
          terminal_placement_error_mean_cm: 2.215679631233215
          terminal_velocity_mean: 0.535301750600338
          mean_latency_ms: 32.6236858331127
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -22.580645161290324
    - [3]
      condition_name: Actuator Drift (0.80x Friction)
      grasp_success_rate_pct: [52.666666666666664,13.316656236958785]
      delta_rel_pct: -13.186813186813188
      min_placement_error_mean_cm: [1.3143045140306153,0.353262980084955]
      terminal_placement_error_mean_cm: [2.387875195145607,0.255073983879998]
      terminal_velocity_mean: [0.5382303596039614,0.06750221070397516]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 68.0
          min_placement_error_mean_cm: 1.7202977061271667
          terminal_placement_error_mean_cm: 2.383537862300873
          terminal_velocity_mean: 0.4953418381512165
          mean_latency_ms: 33.133227099747344
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 17.24137931034484
        101:
          grasp_success_rate_pct: 44.0
          min_placement_error_mean_cm: 1.1455406373739243
          terminal_placement_error_mean_cm: 2.6450901865959167
          terminal_velocity_mean: 0.6160393211245537
          mean_latency_ms: 29.67846919087327
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -29.032258064516128
        2024:
          grasp_success_rate_pct: 46.0
          min_placement_error_mean_cm: 1.0770751985907554
          terminal_placement_error_mean_cm: 2.1349975365400313
          terminal_velocity_mean: 0.5033099195361137
          mean_latency_ms: 29.627431735741776
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -25.806451612903224
    - [4]
      condition_name: Actuator Drift (1.20x Friction)
      grasp_success_rate_pct: [48.0,19.28730152198591]
      delta_rel_pct: -20.879120879120876
      min_placement_error_mean_cm: [1.2900217257936797,0.4582222980166343]
      terminal_placement_error_mean_cm: [2.2831829558809598,0.2896295039858489]
      terminal_velocity_mean: [0.5080112110078335,0.040404712953968615]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 70.0
          min_placement_error_mean_cm: 1.787329178750515
          terminal_placement_error_mean_cm: 2.4599468809366227
          terminal_velocity_mean: 0.49150543451309203
          mean_latency_ms: 33.69387655897015
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 20.689655172413808
        101:
          grasp_success_rate_pct: 34.0
          min_placement_error_mean_cm: 1.197834121286869
          terminal_placement_error_mean_cm: 2.440669119954109
          terminal_velocity_mean: 0.554055769443512
          mean_latency_ms: 29.96104107358774
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -45.16129032258065
        2024:
          grasp_success_rate_pct: 40.0
          min_placement_error_mean_cm: 0.8849018773436547
          terminal_placement_error_mean_cm: 1.9489328667521477
          terminal_velocity_mean: 0.47847242906689647
          mean_latency_ms: 29.91935059930899
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -35.483870967741936
  ANN-Rate (5 items):
    - [0]
      condition_name: Nominal Baseline
      grasp_success_rate_pct: [37.333333333333336,9.451631252505218]
      delta_rel_pct: 0.0
      min_placement_error_mean_cm: [1.0678065086901187,0.2828632725018437]
      terminal_placement_error_mean_cm: [2.9846295044819513,0.4621451582711122]
      terminal_velocity_mean: [0.49538074051340425,0.11216522852158058]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 34.0
          min_placement_error_mean_cm: 0.8320508192479611
          terminal_placement_error_mean_cm: 2.7538758546113966
          terminal_velocity_mean: 0.5869238513708115
          mean_latency_ms: 29.43750379663485
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 0.0
        101:
          grasp_success_rate_pct: 30.0
          min_placement_error_mean_cm: 1.3814548110961915
          terminal_placement_error_mean_cm: 3.5167109155654908
          terminal_velocity_mean: 0.5289560982584953
          mean_latency_ms: 26.53065752230858
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 0.0
        2024:
          grasp_success_rate_pct: 48.0
          min_placement_error_mean_cm: 0.9899138957262039
          terminal_placement_error_mean_cm: 2.6833017432689665
          terminal_velocity_mean: 0.3702622719109058
          mean_latency_ms: 25.199068654014535
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 0.0
    - [1]
      condition_name: Sensory Noise Injection (1.0%)
      grasp_success_rate_pct: [30.0,3.4641016151377544]
      delta_rel_pct: -19.64285714285715
      min_placement_error_mean_cm: [1.0367476403713225,0.17609975464908534]
      terminal_placement_error_mean_cm: [3.362126860022545,0.529625351794463]
      terminal_velocity_mean: [0.5011851574977239,0.1012983853840122]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 32.0
          min_placement_error_mean_cm: 0.8365792346000671
          terminal_placement_error_mean_cm: 3.0554915750026703
          terminal_velocity_mean: 0.5730594700574875
          mean_latency_ms: 31.393095565214775
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -5.882352941176471
        101:
          grasp_success_rate_pct: 26.0
          min_placement_error_mean_cm: 1.1058386087417602
          terminal_placement_error_mean_cm: 3.9736847394704817
          terminal_velocity_mean: 0.5451662424206734
          mean_latency_ms: 26.004011596295314
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -13.333333333333334
        2024:
          grasp_success_rate_pct: 32.0
          min_placement_error_mean_cm: 1.1678250777721404
          terminal_placement_error_mean_cm: 3.0572042655944824
          terminal_velocity_mean: 0.38532976001501085
          mean_latency_ms: 25.061606960452693
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -33.333333333333336
    - [2]
      condition_name: Payload Extrapolation (4.80 kg)
      grasp_success_rate_pct: [26.666666666666668,16.165807537309522]
      delta_rel_pct: -28.571428571428573
      min_placement_error_mean_cm: [1.359116240243117,0.19601864522317206]
      terminal_placement_error_mean_cm: [3.6378836290041607,0.7955281178345174]
      terminal_velocity_mean: [0.6115619068592787,0.10662950140954822]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 24.0
          min_placement_error_mean_cm: 1.1392030370235444
          terminal_placement_error_mean_cm: 3.5584317278862
          terminal_velocity_mean: 0.6400154614448548
          mean_latency_ms: 29.94859264750354
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -29.41176470588235
        101:
          grasp_success_rate_pct: 12.0
          min_placement_error_mean_cm: 1.4226837638020515
          terminal_placement_error_mean_cm: 4.470156444311142
          terminal_velocity_mean: 0.7010783076286315
          mean_latency_ms: 25.1213509093544
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -60.0
        2024:
          grasp_success_rate_pct: 44.0
          min_placement_error_mean_cm: 1.5154619199037551
          terminal_placement_error_mean_cm: 2.88506271481514
          terminal_velocity_mean: 0.4935919515043497
          mean_latency_ms: 25.066311646045065
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -8.333333333333334
    - [3]
      condition_name: Actuator Drift (0.80x Friction)
      grasp_success_rate_pct: [30.666666666666668,14.047538337136984]
      delta_rel_pct: -17.857142857142858
      min_placement_error_mean_cm: [1.4180172796050707,0.21795292848457692]
      terminal_placement_error_mean_cm: [3.740554449160894,0.8331617947215637]
      terminal_velocity_mean: [0.5717242361605167,0.13817699362402994]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 32.0
          min_placement_error_mean_cm: 1.1665561926364898
          terminal_placement_error_mean_cm: 3.536238695383072
          terminal_velocity_mean: 0.643540482223034
          mean_latency_ms: 25.223359415711226
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -5.882352941176471
        101:
          grasp_success_rate_pct: 16.0
          min_placement_error_mean_cm: 1.5526343488693237
          terminal_placement_error_mean_cm: 4.656868257522583
          terminal_velocity_mean: 0.6592045310139656
          mean_latency_ms: 25.066783458762252
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -46.666666666666664
        2024:
          grasp_success_rate_pct: 44.0
          min_placement_error_mean_cm: 1.5348612973093987
          terminal_placement_error_mean_cm: 3.028556394577026
          terminal_velocity_mean: 0.4124276952445507
          mean_latency_ms: 24.790206348217293
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -8.333333333333334
    - [4]
      condition_name: Actuator Drift (1.20x Friction)
      grasp_success_rate_pct: [33.333333333333336,16.65332799572906]
      delta_rel_pct: -10.714285714285714
      min_placement_error_mean_cm: [1.3353092057506244,0.20851366158174045]
      terminal_placement_error_mean_cm: [3.510111181139946,1.2144949894390102]
      terminal_velocity_mean: [0.5205225745340188,0.10602974885015383]
      episodes_with_handover_pct: [0.0,0.0]
      phi1_steps_with_handover_pct: [0.0,0.0]
      mean_nonzero_handover_w_when_active: 0.0
      per_seed:
        42:
          grasp_success_rate_pct: 28.000000000000004
          min_placement_error_mean_cm: 1.101568892300129
          terminal_placement_error_mean_cm: 2.80165857732296
          terminal_velocity_mean: 0.6022750130295753
          mean_latency_ms: 25.216824718118666
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -17.647058823529402
        101:
          grasp_success_rate_pct: 20.0
          min_placement_error_mean_cm: 1.40215890198946
          terminal_placement_error_mean_cm: 4.912464030981064
          terminal_velocity_mean: 0.5585748782753944
          mean_latency_ms: 25.116685864205966
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: -33.333333333333336
        2024:
          grasp_success_rate_pct: 52.0
          min_placement_error_mean_cm: 1.5021998229622842
          terminal_placement_error_mean_cm: 2.816210935115814
          terminal_velocity_mean: 0.40071783229708674
          mean_latency_ms: 25.06705994109404
          episodes_with_handover_pct: 0.0
          phi1_steps_with_handover_pct: 0.0
          mean_nonzero_handover_w: 0.0
          delta_rel_pct: 8.333333333333334
  Deep GRU (Residual-Gated) (5 items):
    - [0]
      condition_name: Nominal Baseline
      grasp_success_rate_pct: [76.66666666666667,5.033222956847166]
      delta_rel_pct: 0.0
      min_placement_error_mean_cm: [1.142546683549881,0.3141746410639535]
      terminal_placement_error_mean_cm: [1.9001078454653422,0.1792502651931432]
      terminal_velocity_mean: [0.29516326268514,0.07536613976838809]
      episodes_with_handover_pct: [6.0,3.4641016151377544]
      phi1_steps_with_handover_pct: [1.0,0.8660254037844386]
      mean_nonzero_handover_w_when_active: 0.05455098082621892
      per_seed:
        42:
          grasp_success_rate_pct: 82.0
          min_placement_error_mean_cm: 0.8276583582162857
          terminal_placement_error_mean_cm: 1.7074048507213593
          terminal_velocity_mean: 0.29966184198856355
          mean_latency_ms: 57.95750650222223
          episodes_with_handover_pct: 10.0
          phi1_steps_with_handover_pct: 2.0
          mean_nonzero_handover_w: 0.09825952008366584
          delta_rel_pct: 0.0
        101:
          grasp_success_rate_pct: 72.0
          min_placement_error_mean_cm: 1.143978949189186
          terminal_placement_error_mean_cm: 2.06187970995903
          terminal_velocity_mean: 0.36817935094237325
          mean_latency_ms: 54.371179489554706
          episodes_with_handover_pct: 4.0
          phi1_steps_with_handover_pct: 0.5
          mean_nonzero_handover_w: 0.04547070369124412
          delta_rel_pct: 0.0
        2024:
          grasp_success_rate_pct: 76.0
          min_placement_error_mean_cm: 1.4560027432441711
          terminal_placement_error_mean_cm: 1.9310389757156372
          terminal_velocity_mean: 0.21764859512448312
          mean_latency_ms: 54.6402853776519
          episodes_with_handover_pct: 4.0
          phi1_steps_with_handover_pct: 0.5
          mean_nonzero_handover_w: 0.01992271870374679
          delta_rel_pct: 0.0
    - [1]
      condition_name: Sensory Noise Injection (1.0%)
      grasp_success_rate_pct: [74.66666666666667,3.055050463303893]
      delta_rel_pct: -2.608695652173913
      min_placement_error_mean_cm: [1.5151751696070033,0.30314335112481977]
      terminal_placement_error_mean_cm: [1.853916923602422,0.23174274041202594]
      terminal_velocity_mean: [0.2947493451088667,0.09277998260340467]
      episodes_with_handover_pct: [34.666666666666664,11.015141094572202]
      phi1_steps_with_handover_pct: [10.666666666666666,4.488411003165077]
      mean_nonzero_handover_w_when_active: 0.15266501222733564
      per_seed:
        42:
          grasp_success_rate_pct: 72.0
          min_placement_error_mean_cm: 1.8375933802127837
          terminal_placement_error_mean_cm: 2.056955361366272
          terminal_velocity_mean: 0.3729041929543018
          mean_latency_ms: 45.58725326396069
          episodes_with_handover_pct: 46.0
          phi1_steps_with_handover_pct: 15.75
          mean_nonzero_handover_w: 0.1645895887414614
          delta_rel_pct: -12.195121951219512
        101:
          grasp_success_rate_pct: 74.0
          min_placement_error_mean_cm: 1.4719947236776352
          terminal_placement_error_mean_cm: 1.9033488303422927
          terminal_velocity_mean: 0.31913049027323726
          mean_latency_ms: 54.78286882565354
          episodes_with_handover_pct: 34.0
          phi1_steps_with_handover_pct: 9.0
          mean_nonzero_handover_w: 0.20296063053111232
          delta_rel_pct: 2.7777777777777777
        2024:
          grasp_success_rate_pct: 78.0
          min_placement_error_mean_cm: 1.2359374049305916
          terminal_placement_error_mean_cm: 1.6014465790987016
          terminal_velocity_mean: 0.192213352099061
          mean_latency_ms: 55.27344265506174
          episodes_with_handover_pct: 24.0
          phi1_steps_with_handover_pct: 7.25
          mean_nonzero_handover_w: 0.09044481740943319
          delta_rel_pct: 2.6315789473684212
    - [2]
      condition_name: Payload Extrapolation (4.80 kg)
      grasp_success_rate_pct: [23.333333333333332,12.858201014657272]
      delta_rel_pct: -69.56521739130436
      min_placement_error_mean_cm: [3.2138332206010816,1.4700081785960324]
      terminal_placement_error_mean_cm: [4.3460664258400605,1.0614877655467412]
      terminal_velocity_mean: [0.49405148267745974,0.21990952766148325]
      episodes_with_handover_pct: [29.333333333333332,40.611985094714754]
      phi1_steps_with_handover_pct: [7.75,11.715374513859981]
      mean_nonzero_handover_w_when_active: 0.34301347039677516
      per_seed:
        42:
          grasp_success_rate_pct: 14.000000000000002
          min_placement_error_mean_cm: 2.678343471288681
          terminal_placement_error_mean_cm: 4.692428379058838
          terminal_velocity_mean: 0.3116999238729477
          mean_latency_ms: 39.307929548496304
          episodes_with_handover_pct: 10.0
          phi1_steps_with_handover_pct: 1.75
          mean_nonzero_handover_w: 0.3143822686587061
          delta_rel_pct: -82.92682926829268
        101:
          grasp_success_rate_pct: 38.0
          min_placement_error_mean_cm: 2.086636486649513
          terminal_placement_error_mean_cm: 3.154660854935646
          terminal_velocity_mean: 0.7382669043540955
          mean_latency_ms: 56.36546777295293
          episodes_with_handover_pct: 2.0
          phi1_steps_with_handover_pct: 0.25
          mean_nonzero_handover_w: 0.4
          delta_rel_pct: -47.22222222222222
        2024:
          grasp_success_rate_pct: 18.0
          min_placement_error_mean_cm: 4.876519703865052
          terminal_placement_error_mean_cm: 5.191110043525696
          terminal_velocity_mean: 0.432187619805336
          mean_latency_ms: 56.881797438130974
          episodes_with_handover_pct: 76.0
          phi1_steps_with_handover_pct: 21.25
          mean_nonzero_handover_w: 0.31465814253161933
          delta_rel_pct: -76.3157894736842
    - [3]
      condition_name: Actuator Drift (0.80x Friction)
      grasp_success_rate_pct: [26.666666666666668,23.86070689089771]
      delta_rel_pct: -65.21739130434783
      min_placement_error_mean_cm: [3.236167592306932,1.4473405546296252]
      terminal_placement_error_mean_cm: [4.381567585865656,1.0010487774447363]
      terminal_velocity_mean: [0.44115810414155326,0.20521995946808633]
      episodes_with_handover_pct: [28.666666666666668,39.46306289852998]
      phi1_steps_with_handover_pct: [8.083333333333334,11.877850535064555]
      mean_nonzero_handover_w_when_active: 0.3593107620300309
      per_seed:
        42:
          grasp_success_rate_pct: 10.0
          min_placement_error_mean_cm: 2.719559631347656
          terminal_placement_error_mean_cm: 4.764410569667816
          terminal_velocity_mean: 0.26115636616945265
          mean_latency_ms: 39.652253493452214
          episodes_with_handover_pct: 10.0
          phi1_steps_with_handover_pct: 2.25
          mean_nonzero_handover_w: 0.37809689356221093
          delta_rel_pct: -87.8048780487805
        101:
          grasp_success_rate_pct: 54.0
          min_placement_error_mean_cm: 2.1180153104662893
          terminal_placement_error_mean_cm: 3.2455974757671355
          terminal_velocity_mean: 0.6646309271454811
          mean_latency_ms: 57.60456265818991
          episodes_with_handover_pct: 2.0
          phi1_steps_with_handover_pct: 0.25
          mean_nonzero_handover_w: 0.4
          delta_rel_pct: -25.0
        2024:
          grasp_success_rate_pct: 16.0
          min_placement_error_mean_cm: 4.8709278351068495
          terminal_placement_error_mean_cm: 5.134694712162018
          terminal_velocity_mean: 0.3976870191097259
          mean_latency_ms: 57.76709282414901
          episodes_with_handover_pct: 74.0
          phi1_steps_with_handover_pct: 21.75
          mean_nonzero_handover_w: 0.29983539252788166
          delta_rel_pct: -78.94736842105263
    - [4]
      condition_name: Actuator Drift (1.20x Friction)
      grasp_success_rate_pct: [30.0,15.0996688705415]
      delta_rel_pct: -60.869565217391305
      min_placement_error_mean_cm: [3.104780209859212,1.3417177318534945]
      terminal_placement_error_mean_cm: [4.190640026132265,1.0252366530107015]
      terminal_velocity_mean: [0.42541863799095153,0.1914834173354981]
      episodes_with_handover_pct: [26.0,38.157568056677825]
      phi1_steps_with_handover_pct: [7.083333333333333,10.981044273352756]
      mean_nonzero_handover_w_when_active: 0.3567154332125237
      per_seed:
        42:
          grasp_success_rate_pct: 16.0
          min_placement_error_mean_cm: 2.6001962661743163
          terminal_placement_error_mean_cm: 4.637553496360779
          terminal_velocity_mean: 0.2670792493224144
          mean_latency_ms: 40.279339710981084
          episodes_with_handover_pct: 6.0
          phi1_steps_with_handover_pct: 1.25
          mean_nonzero_handover_w: 0.37100348895788193
          delta_rel_pct: -80.48780487804878
        101:
          grasp_success_rate_pct: 46.0
          min_placement_error_mean_cm: 2.0885089176893232
          terminal_placement_error_mean_cm: 3.0178090459108353
          terminal_velocity_mean: 0.6382385385036469
          mean_latency_ms: 57.86060653928422
          episodes_with_handover_pct: 2.0
          phi1_steps_with_handover_pct: 0.25
          mean_nonzero_handover_w: 0.4
          delta_rel_pct: -36.111111111111114
        2024:
          grasp_success_rate_pct: 28.000000000000004
          min_placement_error_mean_cm: 4.625635445713997
          terminal_placement_error_mean_cm: 4.916557536125183
          terminal_velocity_mean: 0.3709381261467934
          mean_latency_ms: 57.83468828605162
          episodes_with_handover_pct: 70.0
          phi1_steps_with_handover_pct: 19.75
          mean_nonzero_handover_w: 0.2991428106796892
          delta_rel_pct: -63.1578947368421
disentangled_ablation_3seed (6 rows):
  |label|mean_std|per_seed_success_pct|
  |---|---|---|
  |1. SPWM-Base (Phase 1 Warmup, Unsmoothed)|{grasp_success_rate_pct:[45.333333333333336,16.041612554021285],terminal_placement_accuracy_cm:[1.026371354262034,0.21267911327447728],terminal_error_cm:[2.502900831302007,0.10674231587914605],mean_joint_accel_rad_s2:[0.009831757796928286,0.002547908271341173],mean_joint_jerk_rad_s3:[0.007646870988731583,0.002496011682578312],mean_direction_flips:[23.03333333333333,6.83585644475755],mppi_inference_latency_ms:[26.607863672024582,0.31494921572578505]}|{42:62.0,101:44.0,2024:30.0}|
  |2. + Smooth MPPI Controls (Pre-DAgger)|{grasp_success_rate_pct:[51.333333333333336,9.451631252505216],terminal_placement_accuracy_cm:[1.0174584152301154,0.07739428773724741],terminal_error_cm:[2.413362322449684,0.131995777812547],mean_joint_accel_rad_s2:[0.009906536138926942,0.0024003560204176517],mean_joint_jerk_rad_s3:[0.007045794217847287,0.002429997855850162],mean_direction_flips:[21.993333333333336,6.660010010002488],mppi_inference_latency_ms:[27.186699187828037,0.15842402005752645]}|{42:62.0,101:44.0,2024:48.0}|
  |3a. + Phase 3/4 Curriculum (\texttt{phase\_4}, No Lookahead/Handover)|{grasp_success_rate_pct:[38.0,19.078784028338916],terminal_placement_accuracy_cm:[1.5255670695503554,0.5931443681847731],terminal_error_cm:[3.2744056106607116,1.1318006892038883],mean_joint_accel_rad_s2:[0.0058877142627413075,0.0008819400178644788],mean_joint_jerk_rad_s3:[0.0035392556875012814,0.0008280183162976543],mean_direction_flips:[11.166666666666666,0.6463229326995401],mppi_inference_latency_ms:[27.422173555313975,0.18268943009863092]}|{42:56.00000000000001,101:18.0,2024:40.0}|
  |3b. + $N=25$ Validation Checkpoint Selection (Val-Sel., No Lookahead/Handover)|{grasp_success_rate_pct:[49.333333333333336,6.110100926607791],terminal_placement_accuracy_cm:[1.0013195854425432,0.16452091586443013],terminal_error_cm:[2.2880257406830786,0.23124014692649303],mean_joint_accel_rad_s2:[0.009205199683395525,0.0033555613127640954],mean_joint_jerk_rad_s3:[0.006593972489548226,0.003202463912087665],mean_direction_flips:[21.18,8.536767538125892],mppi_inference_latency_ms:[30.746634633097653,5.982046411903032]}|{42:56.00000000000001,101:44.0,2024:48.0}|
  |4a. + 2-Stage SE(3) Lookahead (Val-Sel., $w_t \equiv 0$)|{grasp_success_rate_pct:[60.666666666666664,2.3094010767585074],terminal_placement_accuracy_cm:[0.8798704931139946,0.10505539807183523],terminal_error_cm:[2.006662103931109,0.11581144347528279],mean_joint_accel_rad_s2:[0.009308865414932371,0.0034586698705177275],mean_joint_jerk_rad_s3:[0.006634432549277941,0.003263484407032791],mean_direction_flips:[21.14,8.549807015365904],mppi_inference_latency_ms:[28.034092626839513,0.24046964122825348]}|{42:57.99999999999999,101:62.0,2024:62.0}|
  |4b. + Residual Rejection Gate (Full SPWM, Val-Sel.)|{grasp_success_rate_pct:[60.666666666666664,2.3094010767585074],terminal_placement_accuracy_cm:[0.8798704931139946,0.10505539807183523],terminal_error_cm:[2.006662103931109,0.11581144347528279],mean_joint_accel_rad_s2:[0.009308865414932371,0.0034586698705177275],mean_joint_jerk_rad_s3:[0.006634432549277941,0.003263484407032791],mean_direction_flips:[21.14,8.549807015365904],mppi_inference_latency_ms:[26.649352693166346,2.271980219747186]}|{42:57.99999999999999,101:62.0,2024:62.0}|
baseline_training_budgets:
  rssm:
    42:
      model_name: Continuous RSSM
      seed: 42
      device: mps
      enable_dagger_ar1: true
      base_epochs: 60
      ar1_dagger_epochs: 16
      total_epochs: 76
      best_epoch: 61
      best_val_loss: 0.4348871593291943
      wall_clock_sec: 778.0765965420287
      ar1_target_alpha: 0.25
      dagger_generations_collected: 3
    101:
      model_name: Continuous RSSM
      seed: 101
      device: mps
      enable_dagger_ar1: true
      base_epochs: 60
      ar1_dagger_epochs: 16
      total_epochs: 76
      best_epoch: 61
      best_val_loss: 0.44028351398614735
      wall_clock_sec: 775.8391683750087
      ar1_target_alpha: 0.25
      dagger_generations_collected: 3
    2024:
      model_name: Continuous RSSM
      seed: 2024
      device: mps
      enable_dagger_ar1: true
      base_epochs: 60
      ar1_dagger_epochs: 17
      total_epochs: 77
      best_epoch: 62
      best_val_loss: 0.4504756721166464
      wall_clock_sec: 826.2417404999724
      ar1_target_alpha: 0.25
      dagger_generations_collected: 3
  deep_gru:
    42:
      model_name: Deep GRU
      seed: 42
      device: mps
      enable_dagger_ar1: true
      base_epochs: 59
      ar1_dagger_epochs: 40
      total_epochs: 99
      best_epoch: 90
      best_val_loss: 0.5676361459952134
      wall_clock_sec: 2026.1408703329507
      ar1_target_alpha: 0.25
      dagger_generations_collected: 8
    101:
      model_name: Deep GRU
      seed: 101
      device: mps
      enable_dagger_ar1: true
      base_epochs: 0
      ar1_dagger_epochs: 80
      total_epochs: 80
      best_epoch: 68
      best_val_loss: 0.5387146770954132
      wall_clock_sec: 4026.366101166932
      ar1_target_alpha: 0.25
      dagger_generations_collected: 14
    2024:
      model_name: Deep GRU
      seed: 2024
      device: mps
      enable_dagger_ar1: true
      base_epochs: 0
      ar1_dagger_epochs: 80
      total_epochs: 80
      best_epoch: 80
      best_val_loss: 0.5893793037304511
      wall_clock_sec: 4009.184660000028
      ar1_target_alpha: 0.25
      dagger_generations_collected: 14
```

### `results/diagnostics/forensic_rssm_gru_failure.json`
> Forensic audit of RSSM vs. equalized Deep GRU across Open-Loop, Un-Gated Online (w_t=1.0), and Residual-Gated Online PIL regimes

```yaml
Continuous RSSM Un-Gated (Seed 42):
  label: Continuous RSSM Un-Gated (Seed 42)
  n_episodes: 50
  n_success: 0
  n_fail: 50
  success_rate_pct: 0.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:3,pct:6.0}
    b_velocity_failure_given_reach_le_3cm: {count:47,pct:94.0}
    b_velocity_failure_all_episodes: {count:50,pct:100.0}
    c_post_contact_impulse_divergence: {count:47,pct:94.0}
    settling_window_failure_given_reach_le_3cm: {count:47,pct:94.0}
    reached_only_after_tc_then_diverged: {count:0,pct:0.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:3,pct:6.0}
    2a_post_contact_div_plus_vel_violation: {count:47,pct:94.0}
    2b_post_contact_div_spatial_only: {count:0,pct:0.0}
    3_reached_post_tc_then_diverged: {count:0,pct:0.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:0,pct:0.0}
  summary_stats:
    min_err_cm_mean: 1.456915372610092
    min_err_cm_std: 0.888517029752088
    pre_tc_min_err_cm_mean: 1.4963469421863556
    tc_err_cm_mean: 2.7344481140375136
    tc_err_cm_std: 1.2793144083769883
    term_err_cm_mean: 24.710977087020876
    term_err_cm_std: 10.710275307660488
    err_jump_tc_to_grasp_cm_mean: 21.97652897298336
    final_vel_mean: 6.904007134437561
    final_vel_std: 1.3181468836809789
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.0310553040727973
    mean_joint_jerk_rad_s3: 0.02920039013028145
    mean_direction_flips: 36.16
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8060242354869844,std_cm:1.6607534974544462,median_cm:2.398180603981018,n:50}
    t_c-2: {mean_cm:2.7449924635887144,std_cm:1.6230154112791257,median_cm:2.4178829193115234,n:50}
    t_c-1: {mean_cm:2.758989669084549,std_cm:1.3719044387467887,median_cm:2.5782278776168823,n:50}
    t_c: {mean_cm:2.7344481140375136,std_cm:1.2793144083769883,median_cm:2.3618115186691284,n:50}
    t_c+1: {mean_cm:3.7436760783195497,std_cm:2.520346856036812,median_cm:2.9587130546569824,n:50}
    t_c+2: {mean_cm:4.858720359802246,std_cm:2.7714310760232097,median_cm:4.392287731170654,n:50}
    t_c+3: {mean_cm:7.781477720737457,std_cm:4.5042137924997245,median_cm:7.074056625366211,n:50}
    t_c+4: {mean_cm:11.416760206222534,std_cm:6.987858342337585,median_cm:9.924240112304688,n:50}
    t_c+5: {mean_cm:16.046059470176697,std_cm:8.649782729225509,median_cm:14.07004976272583,n:50}
    t_c+6: {mean_cm:20.710042076110838,std_cm:9.709148949863993,median_cm:21.032752990722656,n:50}
    t_c+7: {mean_cm:24.710977087020876,std_cm:10.710275307660488,median_cm:24.84556484222412,n:50}
Continuous RSSM Un-Gated (Seed 101):
  label: Continuous RSSM Un-Gated (Seed 101)
  n_episodes: 50
  n_success: 0
  n_fail: 50
  success_rate_pct: 0.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:1,pct:2.0}
    b_velocity_failure_given_reach_le_3cm: {count:49,pct:98.0}
    b_velocity_failure_all_episodes: {count:50,pct:100.0}
    c_post_contact_impulse_divergence: {count:49,pct:98.0}
    settling_window_failure_given_reach_le_3cm: {count:49,pct:98.0}
    reached_only_after_tc_then_diverged: {count:0,pct:0.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:1,pct:2.0}
    2a_post_contact_div_plus_vel_violation: {count:49,pct:98.0}
    2b_post_contact_div_spatial_only: {count:0,pct:0.0}
    3_reached_post_tc_then_diverged: {count:0,pct:0.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:0,pct:0.0}
  summary_stats:
    min_err_cm_mean: 1.3887627166509628
    min_err_cm_std: 0.7836427217228974
    pre_tc_min_err_cm_mean: 1.3975029104948045
    tc_err_cm_mean: 2.70519296169281
    tc_err_cm_std: 1.5419437782641576
    term_err_cm_mean: 27.000035276412962
    term_err_cm_std: 11.571441732882644
    err_jump_tc_to_grasp_cm_mean: 24.294842314720153
    final_vel_mean: 7.153379964828491
    final_vel_std: 1.4014774799430023
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.03229924626648426
    mean_joint_jerk_rad_s3: 0.030016598887741566
    mean_direction_flips: 35.94
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8696282362937926,std_cm:1.757145296973022,median_cm:2.4668288230895996,n:50}
    t_c-2: {mean_cm:2.738204095363617,std_cm:1.6470797666682986,median_cm:2.4108147621154785,n:50}
    t_c-1: {mean_cm:2.5767963498830797,std_cm:1.681966035582423,median_cm:2.3863481283187866,n:50}
    t_c: {mean_cm:2.70519296169281,std_cm:1.5419437782641576,median_cm:2.448335647583008,n:50}
    t_c+1: {mean_cm:3.9934270691871645,std_cm:2.892538698139626,median_cm:3.15087354183197,n:50}
    t_c+2: {mean_cm:5.022188692092896,std_cm:3.0241990062751327,median_cm:4.410186767578125,n:50}
    t_c+3: {mean_cm:8.315929174423218,std_cm:4.505515948778378,median_cm:6.820093870162964,n:50}
    t_c+4: {mean_cm:12.78562816619873,std_cm:6.7394078288254065,median_cm:12.054687023162842,n:50}
    t_c+5: {mean_cm:17.426408562660217,std_cm:9.056289972259131,median_cm:17.50439167022705,n:50}
    t_c+6: {mean_cm:22.43762668609619,std_cm:10.344942290544846,median_cm:21.58891487121582,n:50}
    t_c+7: {mean_cm:27.000035276412962,std_cm:11.571441732882644,median_cm:26.218161582946777,n:50}
Continuous RSSM Un-Gated (Seed 2024):
  label: Continuous RSSM Un-Gated (Seed 2024)
  n_episodes: 50
  n_success: 0
  n_fail: 50
  success_rate_pct: 0.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:2,pct:4.0}
    b_velocity_failure_given_reach_le_3cm: {count:48,pct:96.0}
    b_velocity_failure_all_episodes: {count:50,pct:100.0}
    c_post_contact_impulse_divergence: {count:47,pct:94.0}
    settling_window_failure_given_reach_le_3cm: {count:48,pct:96.0}
    reached_only_after_tc_then_diverged: {count:1,pct:2.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:2,pct:4.0}
    2a_post_contact_div_plus_vel_violation: {count:47,pct:94.0}
    2b_post_contact_div_spatial_only: {count:0,pct:0.0}
    3_reached_post_tc_then_diverged: {count:1,pct:2.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:0,pct:0.0}
  summary_stats:
    min_err_cm_mean: 1.4295401430130006
    min_err_cm_std: 0.8761971421556884
    pre_tc_min_err_cm_mean: 1.4613039577007294
    tc_err_cm_mean: 2.8204561990499495
    tc_err_cm_std: 1.4990684914725287
    term_err_cm_mean: 24.769278402328492
    term_err_cm_std: 10.309742604307631
    err_jump_tc_to_grasp_cm_mean: 21.948822203278542
    final_vel_mean: 7.222090845108032
    final_vel_std: 1.403482970648478
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.03123875129967928
    mean_joint_jerk_rad_s3: 0.02945784468203783
    mean_direction_flips: 37.38
  step_error_profile_around_tc:
    t_c-3: {mean_cm:3.0218841862678527,std_cm:1.6279105428250633,median_cm:2.544119119644165,n:50}
    t_c-2: {mean_cm:2.9575072264671327,std_cm:1.6575253626915922,median_cm:2.488058567047119,n:50}
    t_c-1: {mean_cm:2.8813152182102204,std_cm:1.5618026180524565,median_cm:2.486090898513794,n:50}
    t_c: {mean_cm:2.8204561990499495,std_cm:1.4990684914725287,median_cm:2.6535173654556274,n:50}
    t_c+1: {mean_cm:3.863834716081619,std_cm:2.6441825402640626,median_cm:3.332515239715576,n:50}
    t_c+2: {mean_cm:4.646144304275513,std_cm:2.5409079008700637,median_cm:4.337389230728149,n:50}
    t_c+3: {mean_cm:7.215016748905182,std_cm:3.82765948609738,median_cm:6.71310567855835,n:50}
    t_c+4: {mean_cm:11.144767661094665,std_cm:5.618361567548499,median_cm:10.410730361938477,n:50}
    t_c+5: {mean_cm:15.603601350784302,std_cm:7.579263539670666,median_cm:14.306604385375977,n:50}
    t_c+6: {mean_cm:20.22340539932251,std_cm:9.273408770894141,median_cm:20.142123222351074,n:50}
    t_c+7: {mean_cm:24.769278402328492,std_cm:10.309742604307631,median_cm:24.981929779052734,n:50}
Continuous RSSM Residual-Gated (Seed 42):
  label: Continuous RSSM Residual-Gated (Seed 42)
  n_episodes: 50
  n_success: 8
  n_fail: 42
  success_rate_pct: 16.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:1,pct:2.0}
    b_velocity_failure_given_reach_le_3cm: {count:13,pct:26.0}
    b_velocity_failure_all_episodes: {count:13,pct:26.0}
    c_post_contact_impulse_divergence: {count:19,pct:38.0}
    settling_window_failure_given_reach_le_3cm: {count:27,pct:54.0}
    reached_only_after_tc_then_diverged: {count:8,pct:16.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:1,pct:2.0}
    2a_post_contact_div_plus_vel_violation: {count:8,pct:16.0}
    2b_post_contact_div_spatial_only: {count:11,pct:22.0}
    3_reached_post_tc_then_diverged: {count:8,pct:16.0}
    4_settling_passed_vel_failed_only: {count:1,pct:2.0}
    5_settling_and_vel_passed_rot_failed_only: {count:13,pct:26.0}
  summary_stats:
    min_err_cm_mean: 1.2852557504177093
    min_err_cm_std: 0.6163579389948347
    pre_tc_min_err_cm_mean: 1.4963469421863556
    tc_err_cm_mean: 2.7344481140375136
    tc_err_cm_std: 1.2793144083769883
    term_err_cm_mean: 2.8026852571964262
    term_err_cm_std: 1.250732118205452
    err_jump_tc_to_grasp_cm_mean: 0.06823714315891266
    final_vel_mean: 0.9999081724137068
    final_vel_std: 0.6270123580932241
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.02125625854358077
    mean_joint_jerk_rad_s3: 0.02221311992034316
    mean_direction_flips: 36.42
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8060242354869844,std_cm:1.6607534974544462,median_cm:2.398180603981018,n:50}
    t_c-2: {mean_cm:2.7449924635887144,std_cm:1.6230154112791257,median_cm:2.4178829193115234,n:50}
    t_c-1: {mean_cm:2.758989669084549,std_cm:1.3719044387467887,median_cm:2.5782278776168823,n:50}
    t_c: {mean_cm:2.7344481140375136,std_cm:1.2793144083769883,median_cm:2.3618115186691284,n:50}
    t_c+1: {mean_cm:3.7004931473731997,std_cm:2.385862360417743,median_cm:3.013274669647217,n:50}
    t_c+2: {mean_cm:3.2856418287754057,std_cm:1.7684971965260652,median_cm:2.8707910776138306,n:50}
    t_c+3: {mean_cm:3.039464303255081,std_cm:1.535868137697539,median_cm:2.665216326713562,n:50}
    t_c+4: {mean_cm:3.065787924528122,std_cm:1.4495784566482008,median_cm:2.655968189239502,n:50}
    t_c+5: {mean_cm:3.0905008351802827,std_cm:1.3650752168204054,median_cm:2.9295456409454346,n:50}
    t_c+6: {mean_cm:3.0042423689365387,std_cm:1.3520194249543056,median_cm:2.615446925163269,n:50}
    t_c+7: {mean_cm:2.8026852571964262,std_cm:1.250732118205452,median_cm:2.3478347063064575,n:50}
Continuous RSSM Residual-Gated (Seed 101):
  label: Continuous RSSM Residual-Gated (Seed 101)
  n_episodes: 50
  n_success: 5
  n_fail: 45
  success_rate_pct: 10.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:1,pct:2.0}
    b_velocity_failure_given_reach_le_3cm: {count:5,pct:10.0}
    b_velocity_failure_all_episodes: {count:5,pct:10.0}
    c_post_contact_impulse_divergence: {count:16,pct:32.0}
    settling_window_failure_given_reach_le_3cm: {count:26,pct:52.0}
    reached_only_after_tc_then_diverged: {count:10,pct:20.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:1,pct:2.0}
    2a_post_contact_div_plus_vel_violation: {count:1,pct:2.0}
    2b_post_contact_div_spatial_only: {count:15,pct:30.0}
    3_reached_post_tc_then_diverged: {count:10,pct:20.0}
    4_settling_passed_vel_failed_only: {count:1,pct:2.0}
    5_settling_and_vel_passed_rot_failed_only: {count:17,pct:34.0}
  summary_stats:
    min_err_cm_mean: 1.2193563985824585
    min_err_cm_std: 0.6733557358778788
    pre_tc_min_err_cm_mean: 1.3975029104948045
    tc_err_cm_mean: 2.70519296169281
    tc_err_cm_std: 1.5419437782641576
    term_err_cm_mean: 2.770985679626465
    term_err_cm_std: 1.329820118983065
    err_jump_tc_to_grasp_cm_mean: 0.06579271793365478
    final_vel_mean: 0.7805915237218142
    final_vel_std: 0.4653217854539529
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.02023072073236108
    mean_joint_jerk_rad_s3: 0.021210806742310524
    mean_direction_flips: 36.06
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8696282362937926,std_cm:1.757145296973022,median_cm:2.4668288230895996,n:50}
    t_c-2: {mean_cm:2.738204095363617,std_cm:1.6470797666682986,median_cm:2.4108147621154785,n:50}
    t_c-1: {mean_cm:2.5767963498830797,std_cm:1.681966035582423,median_cm:2.3863481283187866,n:50}
    t_c: {mean_cm:2.70519296169281,std_cm:1.5419437782641576,median_cm:2.448335647583008,n:50}
    t_c+1: {mean_cm:3.951930454969406,std_cm:2.804953169671099,median_cm:3.078206777572632,n:50}
    t_c+2: {mean_cm:3.6224123001098634,std_cm:2.239429324482335,median_cm:3.0837604999542236,n:50}
    t_c+3: {mean_cm:3.2723434567451477,std_cm:1.7547916959524443,median_cm:3.129513740539551,n:50}
    t_c+4: {mean_cm:3.0971233302354815,std_cm:1.4996233767314215,median_cm:2.843792200088501,n:50}
    t_c+5: {mean_cm:2.9562839233875273,std_cm:1.2863846231919944,median_cm:2.575040340423584,n:50}
    t_c+6: {mean_cm:2.751187549829483,std_cm:1.3617498462434652,median_cm:2.5202354192733765,n:50}
    t_c+7: {mean_cm:2.770985679626465,std_cm:1.329820118983065,median_cm:2.494181513786316,n:50}
Continuous RSSM Residual-Gated (Seed 2024):
  label: Continuous RSSM Residual-Gated (Seed 2024)
  n_episodes: 50
  n_success: 10
  n_fail: 40
  success_rate_pct: 20.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:0,pct:0.0}
    b_velocity_failure_given_reach_le_3cm: {count:6,pct:12.0}
    b_velocity_failure_all_episodes: {count:6,pct:12.0}
    c_post_contact_impulse_divergence: {count:15,pct:30.0}
    settling_window_failure_given_reach_le_3cm: {count:19,pct:38.0}
    reached_only_after_tc_then_diverged: {count:4,pct:8.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:0,pct:0.0}
    2a_post_contact_div_plus_vel_violation: {count:3,pct:6.0}
    2b_post_contact_div_spatial_only: {count:12,pct:24.0}
    3_reached_post_tc_then_diverged: {count:4,pct:8.0}
    4_settling_passed_vel_failed_only: {count:1,pct:2.0}
    5_settling_and_vel_passed_rot_failed_only: {count:20,pct:40.0}
  summary_stats:
    min_err_cm_mean: 1.1946693646907807
    min_err_cm_std: 0.6255705025940155
    pre_tc_min_err_cm_mean: 1.4613039577007294
    tc_err_cm_mean: 2.8204561990499495
    tc_err_cm_std: 1.4990684914725287
    term_err_cm_mean: 2.59847177863121
    term_err_cm_std: 1.4417185983615668
    err_jump_tc_to_grasp_cm_mean: -0.2219844204187393
    final_vel_mean: 0.9102049902081489
    final_vel_std: 0.5446872851256144
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.019994692355394365
    mean_joint_jerk_rad_s3: 0.02140466099604964
    mean_direction_flips: 38.34
  step_error_profile_around_tc:
    t_c-3: {mean_cm:3.0218841862678527,std_cm:1.6279105428250633,median_cm:2.544119119644165,n:50}
    t_c-2: {mean_cm:2.9575072264671327,std_cm:1.6575253626915922,median_cm:2.488058567047119,n:50}
    t_c-1: {mean_cm:2.8813152182102204,std_cm:1.5618026180524565,median_cm:2.486090898513794,n:50}
    t_c: {mean_cm:2.8204561990499495,std_cm:1.4990684914725287,median_cm:2.6535173654556274,n:50}
    t_c+1: {mean_cm:3.8082215487957,std_cm:2.7305402251121134,median_cm:3.052035927772522,n:50}
    t_c+2: {mean_cm:3.3961012995243074,std_cm:2.132837995348024,median_cm:2.812078595161438,n:50}
    t_c+3: {mean_cm:2.935594902038574,std_cm:1.5694884663502995,median_cm:2.5725667476654053,n:50}
    t_c+4: {mean_cm:2.688289783000946,std_cm:1.5078044020264056,median_cm:2.5231261253356934,n:50}
    t_c+5: {mean_cm:2.6857805478572847,std_cm:1.6621445235568015,median_cm:2.3521034717559814,n:50}
    t_c+6: {mean_cm:2.6429535794258117,std_cm:1.6630230547801927,median_cm:2.3729885816574097,n:50}
    t_c+7: {mean_cm:2.59847177863121,std_cm:1.4417185983615668,median_cm:2.2666091918945312,n:50}
Deep GRU Un-Gated (Seed 42):
  label: Deep GRU Un-Gated (Seed 42)
  n_episodes: 50
  n_success: 0
  n_fail: 50
  success_rate_pct: 0.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:1,pct:2.0}
    b_velocity_failure_given_reach_le_3cm: {count:6,pct:12.0}
    b_velocity_failure_all_episodes: {count:7,pct:14.0}
    c_post_contact_impulse_divergence: {count:45,pct:90.0}
    settling_window_failure_given_reach_le_3cm: {count:45,pct:90.0}
    reached_only_after_tc_then_diverged: {count:0,pct:0.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:1,pct:2.0}
    2a_post_contact_div_plus_vel_violation: {count:6,pct:12.0}
    2b_post_contact_div_spatial_only: {count:39,pct:78.0}
    3_reached_post_tc_then_diverged: {count:0,pct:0.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:4,pct:8.0}
  summary_stats:
    min_err_cm_mean: 0.9433116026222705
    min_err_cm_std: 0.7225897143886602
    pre_tc_min_err_cm_mean: 1.0321084062755108
    tc_err_cm_mean: 2.2564656108617784
    tc_err_cm_std: 1.5565430332380916
    term_err_cm_mean: 6.630777020454406
    term_err_cm_std: 4.461640380623032
    err_jump_tc_to_grasp_cm_mean: 4.374311409592629
    final_vel_mean: 0.7467437729239463
    final_vel_std: 0.5887069250888312
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.0069884296925738455
    mean_joint_jerk_rad_s3: 0.004795440048910677
    mean_direction_flips: 15.54
  step_error_profile_around_tc:
    t_c-3: {mean_cm:1.8624016344547272,std_cm:1.234946277248152,median_cm:1.7487884759902954,n:50}
    t_c-2: {mean_cm:1.8769904211163522,std_cm:1.1807382550887493,median_cm:1.7688578367233276,n:50}
    t_c-1: {mean_cm:1.991890126466751,std_cm:1.3414012959618864,median_cm:1.505833387374878,n:50}
    t_c: {mean_cm:2.2564656108617784,std_cm:1.5565430332380916,median_cm:1.92295503616333,n:50}
    t_c+1: {mean_cm:2.082577033340931,std_cm:1.4932303563373885,median_cm:1.7682087421417236,n:50}
    t_c+2: {mean_cm:2.5753391909599306,std_cm:1.344338551801569,median_cm:2.24237060546875,n:50}
    t_c+3: {mean_cm:3.7418530905246734,std_cm:1.5823620969971155,median_cm:3.51343035697937,n:50}
    t_c+4: {mean_cm:4.952829649448395,std_cm:2.2408660940436667,median_cm:4.614519119262695,n:50}
    t_c+5: {mean_cm:5.839220027923584,std_cm:3.057588917725235,median_cm:5.166048288345337,n:50}
    t_c+6: {mean_cm:6.361779108047485,std_cm:3.9196748065533877,median_cm:5.486771106719971,n:50}
    t_c+7: {mean_cm:6.630777020454406,std_cm:4.461640380623032,median_cm:5.489295244216919,n:50}
Deep GRU Un-Gated (Seed 101):
  label: Deep GRU Un-Gated (Seed 101)
  n_episodes: 50
  n_success: 0
  n_fail: 50
  success_rate_pct: 0.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:6,pct:12.0}
    b_velocity_failure_given_reach_le_3cm: {count:9,pct:18.0}
    b_velocity_failure_all_episodes: {count:9,pct:18.0}
    c_post_contact_impulse_divergence: {count:43,pct:86.0}
    settling_window_failure_given_reach_le_3cm: {count:44,pct:88.0}
    reached_only_after_tc_then_diverged: {count:1,pct:2.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:6,pct:12.0}
    2a_post_contact_div_plus_vel_violation: {count:9,pct:18.0}
    2b_post_contact_div_spatial_only: {count:34,pct:68.0}
    3_reached_post_tc_then_diverged: {count:1,pct:2.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:0,pct:0.0}
  summary_stats:
    min_err_cm_mean: 1.4955504468083383
    min_err_cm_std: 0.974366769362752
    pre_tc_min_err_cm_mean: 1.5793519452214242
    tc_err_cm_mean: 2.842992330789566
    tc_err_cm_std: 1.902399889212926
    term_err_cm_mean: 8.90200080871582
    term_err_cm_std: 3.847520169336552
    err_jump_tc_to_grasp_cm_mean: 6.0590084779262545
    final_vel_mean: 1.0589640013873578
    final_vel_std: 1.217869594479841
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.0074005681113339964
    mean_joint_jerk_rad_s3: 0.005270186914131046
    mean_direction_flips: 15.36
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.7255666959285736,std_cm:1.554459728424148,median_cm:2.4002795219421387,n:50}
    t_c-2: {mean_cm:2.7539375793933867,std_cm:1.4624588501720588,median_cm:2.4477955102920532,n:50}
    t_c-1: {mean_cm:2.7441939234733583,std_cm:1.6520949327062857,median_cm:2.312147378921509,n:50}
    t_c: {mean_cm:2.842992330789566,std_cm:1.902399889212926,median_cm:2.207262396812439,n:50}
    t_c+1: {mean_cm:2.9963693618774414,std_cm:2.001091781110866,median_cm:2.375216484069824,n:50}
    t_c+2: {mean_cm:3.649802870750427,std_cm:1.9897841514475787,median_cm:3.0593615770339966,n:50}
    t_c+3: {mean_cm:4.718345837593079,std_cm:2.3077647367343213,median_cm:4.029115200042725,n:50}
    t_c+4: {mean_cm:5.778243823051453,std_cm:2.733888841110165,median_cm:4.814998388290405,n:50}
    t_c+5: {mean_cm:6.7685473442077635,std_cm:3.0037129018322726,median_cm:5.615656852722168,n:50}
    t_c+6: {mean_cm:7.85193067073822,std_cm:3.300754425937165,median_cm:7.246733665466309,n:50}
    t_c+7: {mean_cm:8.90200080871582,std_cm:3.847520169336552,median_cm:8.583430290222168,n:50}
Deep GRU Un-Gated (Seed 2024):
  label: Deep GRU Un-Gated (Seed 2024)
  n_episodes: 50
  n_success: 0
  n_fail: 50
  success_rate_pct: 0.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:11,pct:22.0}
    b_velocity_failure_given_reach_le_3cm: {count:2,pct:4.0}
    b_velocity_failure_all_episodes: {count:2,pct:4.0}
    c_post_contact_impulse_divergence: {count:35,pct:70.0}
    settling_window_failure_given_reach_le_3cm: {count:37,pct:74.0}
    reached_only_after_tc_then_diverged: {count:2,pct:4.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:11,pct:22.0}
    2a_post_contact_div_plus_vel_violation: {count:2,pct:4.0}
    2b_post_contact_div_spatial_only: {count:33,pct:66.0}
    3_reached_post_tc_then_diverged: {count:2,pct:4.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:2,pct:4.0}
  summary_stats:
    min_err_cm_mean: 1.9634414403140545
    min_err_cm_std: 1.1926076903006835
    pre_tc_min_err_cm_mean: 2.024408501982689
    tc_err_cm_mean: 3.1450187933444975
    tc_err_cm_std: 1.4082487197398263
    term_err_cm_mean: 6.521503386497497
    term_err_cm_std: 2.9093178808336364
    err_jump_tc_to_grasp_cm_mean: 3.376484593153
    final_vel_mean: 0.5038897021114827
    final_vel_std: 0.4166680326265456
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.0065517672221176325
    mean_joint_jerk_rad_s3: 0.004431145152775571
    mean_direction_flips: 14.34
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8084526360034943,std_cm:1.668798399075106,median_cm:2.350420832633972,n:50}
    t_c-2: {mean_cm:2.8601189959049225,std_cm:1.2646640637745379,median_cm:2.4336156845092773,n:50}
    t_c-1: {mean_cm:2.90585419178009,std_cm:1.2549937993404103,median_cm:2.4293744564056396,n:50}
    t_c: {mean_cm:3.1450187933444975,std_cm:1.4082487197398263,median_cm:2.827304720878601,n:50}
    t_c+1: {mean_cm:2.926797423809767,std_cm:1.5639518730076403,median_cm:2.4531939029693604,n:50}
    t_c+2: {mean_cm:3.5582837796211244,std_cm:1.546619993633832,median_cm:3.3048676252365112,n:50}
    t_c+3: {mean_cm:4.657134444713592,std_cm:1.7357791571274899,median_cm:4.509346961975098,n:50}
    t_c+4: {mean_cm:5.6222643327713016,std_cm:2.0179508046602996,median_cm:5.421851634979248,n:50}
    t_c+5: {mean_cm:6.210109462738037,std_cm:2.431919426597959,median_cm:5.955317735671997,n:50}
    t_c+6: {mean_cm:6.4707139253616335,std_cm:2.808902206263955,median_cm:6.108853578567505,n:50}
    t_c+7: {mean_cm:6.521503386497497,std_cm:2.9093178808336364,median_cm:6.057376146316528,n:50}
Deep GRU Residual-Gated (Seed 42):
  label: Deep GRU Residual-Gated (Seed 42)
  n_episodes: 50
  n_success: 41
  n_fail: 9
  success_rate_pct: 82.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:0,pct:0.0}
    b_velocity_failure_given_reach_le_3cm: {count:0,pct:0.0}
    b_velocity_failure_all_episodes: {count:0,pct:0.0}
    c_post_contact_impulse_divergence: {count:5,pct:10.0}
    settling_window_failure_given_reach_le_3cm: {count:6,pct:12.0}
    reached_only_after_tc_then_diverged: {count:1,pct:2.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:0,pct:0.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:5,pct:10.0}
    3_reached_post_tc_then_diverged: {count:1,pct:2.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:3,pct:6.0}
  summary_stats:
    min_err_cm_mean: 0.8276583582162857
    min_err_cm_std: 0.6198663417549084
    pre_tc_min_err_cm_mean: 1.0321084062755108
    tc_err_cm_mean: 2.2564656108617784
    tc_err_cm_std: 1.5565430332380916
    term_err_cm_mean: 1.7074048507213593
    term_err_cm_std: 0.8542619093890267
    err_jump_tc_to_grasp_cm_mean: -0.549060760140419
    final_vel_mean: 0.29966184198856355
    final_vel_std: 0.14138118620062723
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.004455102609936148
    mean_joint_jerk_rad_s3: 0.0027215255866758527
    mean_direction_flips: 11.48
  step_error_profile_around_tc:
    t_c-3: {mean_cm:1.8624016344547272,std_cm:1.234946277248152,median_cm:1.7487884759902954,n:50}
    t_c-2: {mean_cm:1.8769904211163522,std_cm:1.1807382550887493,median_cm:1.7688578367233276,n:50}
    t_c-1: {mean_cm:1.991890126466751,std_cm:1.3414012959618864,median_cm:1.505833387374878,n:50}
    t_c: {mean_cm:2.2564656108617784,std_cm:1.5565430332380916,median_cm:1.92295503616333,n:50}
    t_c+1: {mean_cm:1.96743328332901,std_cm:1.303326902045649,median_cm:1.6611997485160828,n:50}
    t_c+2: {mean_cm:1.6644573146104813,std_cm:0.9122110374003682,median_cm:1.5140461921691895,n:50}
    t_c+3: {mean_cm:1.4921204835176467,std_cm:0.7499674446678796,median_cm:1.4212709069252014,n:50}
    t_c+4: {mean_cm:1.4870184379816056,std_cm:0.7148408867748791,median_cm:1.4519299864768982,n:50}
    t_c+5: {mean_cm:1.5559572775661945,std_cm:0.7239483911182514,median_cm:1.4397103190422058,n:50}
    t_c+6: {mean_cm:1.636294121146202,std_cm:0.7708139191725581,median_cm:1.5500924587249756,n:50}
    t_c+7: {mean_cm:1.7074048507213593,std_cm:0.8542619093890267,median_cm:1.6311729550361633,n:50}
Deep GRU Residual-Gated (Seed 101):
  label: Deep GRU Residual-Gated (Seed 101)
  n_episodes: 50
  n_success: 36
  n_fail: 14
  success_rate_pct: 72.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:4,pct:8.0}
    b_velocity_failure_given_reach_le_3cm: {count:0,pct:0.0}
    b_velocity_failure_all_episodes: {count:0,pct:0.0}
    c_post_contact_impulse_divergence: {count:3,pct:6.0}
    settling_window_failure_given_reach_le_3cm: {count:8,pct:16.0}
    reached_only_after_tc_then_diverged: {count:5,pct:10.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:4,pct:8.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:3,pct:6.0}
    3_reached_post_tc_then_diverged: {count:5,pct:10.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:2,pct:4.0}
  summary_stats:
    min_err_cm_mean: 1.143978949189186
    min_err_cm_std: 1.0157706090458067
    pre_tc_min_err_cm_mean: 1.5793519452214242
    tc_err_cm_mean: 2.842992330789566
    tc_err_cm_std: 1.902399889212926
    term_err_cm_mean: 2.06187970995903
    term_err_cm_std: 2.30758448331492
    err_jump_tc_to_grasp_cm_mean: -0.7811126208305359
    final_vel_mean: 0.36817935094237325
    final_vel_std: 0.2529912876862468
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.004630304866004735
    mean_joint_jerk_rad_s3: 0.002819125670939684
    mean_direction_flips: 10.52
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.7255666959285736,std_cm:1.554459728424148,median_cm:2.4002795219421387,n:50}
    t_c-2: {mean_cm:2.7539375793933867,std_cm:1.4624588501720588,median_cm:2.4477955102920532,n:50}
    t_c-1: {mean_cm:2.7441939234733583,std_cm:1.6520949327062857,median_cm:2.312147378921509,n:50}
    t_c: {mean_cm:2.842992330789566,std_cm:1.902399889212926,median_cm:2.207262396812439,n:50}
    t_c+1: {mean_cm:2.623877874016762,std_cm:1.9605131690558608,median_cm:1.9665517807006836,n:50}
    t_c+2: {mean_cm:2.2239938607811927,std_cm:1.8999315553827745,median_cm:1.6640714406967163,n:50}
    t_c+3: {mean_cm:2.0926450949907305,std_cm:1.9941124350055748,median_cm:1.5632146000862122,n:50}
    t_c+4: {mean_cm:2.2232044506073,std_cm:2.089145514056964,median_cm:1.5273538827896118,n:50}
    t_c+5: {mean_cm:2.261555926203728,std_cm:2.2268998480629474,median_cm:1.2980888485908508,n:50}
    t_c+6: {mean_cm:2.17594662964344,std_cm:2.3177578099842266,median_cm:1.184807002544403,n:50}
    t_c+7: {mean_cm:2.06187970995903,std_cm:2.30758448331492,median_cm:1.2583984732627869,n:50}
Deep GRU Residual-Gated (Seed 2024):
  label: Deep GRU Residual-Gated (Seed 2024)
  n_episodes: 50
  n_success: 38
  n_fail: 12
  success_rate_pct: 76.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:5,pct:10.0}
    b_velocity_failure_given_reach_le_3cm: {count:0,pct:0.0}
    b_velocity_failure_all_episodes: {count:0,pct:0.0}
    c_post_contact_impulse_divergence: {count:1,pct:2.0}
    settling_window_failure_given_reach_le_3cm: {count:6,pct:12.0}
    reached_only_after_tc_then_diverged: {count:5,pct:10.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:5,pct:10.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:1,pct:2.0}
    3_reached_post_tc_then_diverged: {count:5,pct:10.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:1,pct:2.0}
  summary_stats:
    min_err_cm_mean: 1.4560027432441711
    min_err_cm_std: 1.0354441619455195
    pre_tc_min_err_cm_mean: 2.024408501982689
    tc_err_cm_mean: 3.1450187933444975
    tc_err_cm_std: 1.4082487197398263
    term_err_cm_mean: 1.9310389757156372
    term_err_cm_std: 1.4129652875940601
    err_jump_tc_to_grasp_cm_mean: -1.2139798176288605
    final_vel_mean: 0.21764859512448312
    final_vel_std: 0.11447028374291736
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.004394554344471544
    mean_joint_jerk_rad_s3: 0.0024654526135418566
    mean_direction_flips: 10.36
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8084526360034943,std_cm:1.668798399075106,median_cm:2.350420832633972,n:50}
    t_c-2: {mean_cm:2.8601189959049225,std_cm:1.2646640637745379,median_cm:2.4336156845092773,n:50}
    t_c-1: {mean_cm:2.90585419178009,std_cm:1.2549937993404103,median_cm:2.4293744564056396,n:50}
    t_c: {mean_cm:3.1450187933444975,std_cm:1.4082487197398263,median_cm:2.827304720878601,n:50}
    t_c+1: {mean_cm:2.6491046488285064,std_cm:1.4936212184858715,median_cm:2.35075581073761,n:50}
    t_c+2: {mean_cm:2.357303862571716,std_cm:1.3904488663154773,median_cm:2.172505736351013,n:50}
    t_c+3: {mean_cm:2.189952018857002,std_cm:1.4138888945985166,median_cm:1.989095687866211,n:50}
    t_c+4: {mean_cm:2.137187895476818,std_cm:1.3899572169274361,median_cm:1.8903083801269531,n:50}
    t_c+5: {mean_cm:2.079212930202484,std_cm:1.3826275838860635,median_cm:1.6671987175941467,n:50}
    t_c+6: {mean_cm:1.9995094066858292,std_cm:1.4009636454794605,median_cm:1.5755807757377625,n:50}
    t_c+7: {mean_cm:1.9310389757156372,std_cm:1.4129652875940601,median_cm:1.5323635935783386,n:50}
ANN-Rate (Seed 42):
  label: ANN-Rate (Seed 42)
  n_episodes: 50
  n_success: 17
  n_fail: 33
  success_rate_pct: 34.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:0,pct:0.0}
    b_velocity_failure_given_reach_le_3cm: {count:2,pct:4.0}
    b_velocity_failure_all_episodes: {count:2,pct:4.0}
    c_post_contact_impulse_divergence: {count:19,pct:38.0}
    settling_window_failure_given_reach_le_3cm: {count:26,pct:52.0}
    reached_only_after_tc_then_diverged: {count:7,pct:14.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:0,pct:0.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:19,pct:38.0}
    3_reached_post_tc_then_diverged: {count:7,pct:14.0}
    4_settling_passed_vel_failed_only: {count:1,pct:2.0}
    5_settling_and_vel_passed_rot_failed_only: {count:6,pct:12.0}
  summary_stats:
    min_err_cm_mean: 0.8320508192479611
    min_err_cm_std: 0.4655940325677702
    pre_tc_min_err_cm_mean: 1.164362844079733
    tc_err_cm_mean: 2.6134748697280883
    tc_err_cm_std: 1.4645071738342264
    term_err_cm_mean: 2.7538758546113966
    term_err_cm_std: 1.6488538132409942
    err_jump_tc_to_grasp_cm_mean: 0.1404009848833084
    final_vel_mean: 0.5869238513708115
    final_vel_std: 0.31809935150474355
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.012113787466660142
    mean_joint_jerk_rad_s3: 0.00986627260223031
    mean_direction_flips: 28.24
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.3941184145212175,std_cm:1.2282851130248458,median_cm:2.112388014793396,n:50}
    t_c-2: {mean_cm:2.53336696267128,std_cm:1.2756955653524797,median_cm:2.513612151145935,n:50}
    t_c-1: {mean_cm:2.5826906847953794,std_cm:1.3584516123404389,median_cm:2.456220507621765,n:50}
    t_c: {mean_cm:2.6134748697280883,std_cm:1.4645071738342264,median_cm:2.2219539880752563,n:50}
    t_c+1: {mean_cm:3.0151421278715134,std_cm:1.7541511653893591,median_cm:2.6719231605529785,n:50}
    t_c+2: {mean_cm:2.5468107771873476,std_cm:1.443726447976522,median_cm:2.211168050765991,n:50}
    t_c+3: {mean_cm:2.3111347663402557,std_cm:1.2396275942532466,median_cm:2.0264153480529785,n:50}
    t_c+4: {mean_cm:2.397536401748657,std_cm:1.2869941014388753,median_cm:2.2529900074005127,n:50}
    t_c+5: {mean_cm:2.5626229739189146,std_cm:1.4776500742017475,median_cm:2.2793211936950684,n:50}
    t_c+6: {mean_cm:2.6735214442014694,std_cm:1.6198485432416494,median_cm:2.496946930885315,n:50}
    t_c+7: {mean_cm:2.7538758546113966,std_cm:1.6488538132409942,median_cm:2.117562413215637,n:50}
ANN-Rate (Seed 101):
  label: ANN-Rate (Seed 101)
  n_episodes: 50
  n_success: 15
  n_fail: 35
  success_rate_pct: 30.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:3,pct:6.0}
    b_velocity_failure_given_reach_le_3cm: {count:0,pct:0.0}
    b_velocity_failure_all_episodes: {count:0,pct:0.0}
    c_post_contact_impulse_divergence: {count:21,pct:42.0}
    settling_window_failure_given_reach_le_3cm: {count:24,pct:48.0}
    reached_only_after_tc_then_diverged: {count:3,pct:6.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:3,pct:6.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:21,pct:42.0}
    3_reached_post_tc_then_diverged: {count:3,pct:6.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:8,pct:16.0}
  summary_stats:
    min_err_cm_mean: 1.3814548110961915
    min_err_cm_std: 1.0088033363029787
    pre_tc_min_err_cm_mean: 1.4610292547941208
    tc_err_cm_mean: 2.9168222725391386
    tc_err_cm_std: 1.6069247411732943
    term_err_cm_mean: 3.5167109155654908
    term_err_cm_std: 1.908053762203449
    err_jump_tc_to_grasp_cm_mean: 0.5998886430263519
    final_vel_mean: 0.5289560982584953
    final_vel_std: 0.23627259950728746
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.011858293414115905
    mean_joint_jerk_rad_s3: 0.010119730336591602
    mean_direction_flips: 25.04
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.8613218301534653,std_cm:1.4879508712309886,median_cm:2.841180205345154,n:50}
    t_c-2: {mean_cm:2.796315099000931,std_cm:1.551251035862099,median_cm:2.806033730506897,n:50}
    t_c-1: {mean_cm:2.7768456643819808,std_cm:1.5754521924477278,median_cm:2.692988634109497,n:50}
    t_c: {mean_cm:2.9168222725391386,std_cm:1.6069247411732943,median_cm:2.678290843963623,n:50}
    t_c+1: {mean_cm:4.091164743900299,std_cm:2.380557507605793,median_cm:3.846030116081238,n:50}
    t_c+2: {mean_cm:3.8094804775714874,std_cm:2.133758613040779,median_cm:3.2859740257263184,n:50}
    t_c+3: {mean_cm:3.491499652862549,std_cm:1.9280913492845544,median_cm:3.0375328063964844,n:50}
    t_c+4: {mean_cm:3.287111759185791,std_cm:1.8703497704524499,median_cm:3.0512077808380127,n:50}
    t_c+5: {mean_cm:3.2765079057216644,std_cm:1.8741169438468754,median_cm:3.09067964553833,n:50}
    t_c+6: {mean_cm:3.4137790858745576,std_cm:1.854704060273812,median_cm:3.0054166316986084,n:50}
    t_c+7: {mean_cm:3.5167109155654908,std_cm:1.908053762203449,median_cm:2.926141142845154,n:50}
ANN-Rate (Seed 2024):
  label: ANN-Rate (Seed 2024)
  n_episodes: 50
  n_success: 24
  n_fail: 26
  success_rate_pct: 48.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:1,pct:2.0}
    b_velocity_failure_given_reach_le_3cm: {count:0,pct:0.0}
    b_velocity_failure_all_episodes: {count:0,pct:0.0}
    c_post_contact_impulse_divergence: {count:16,pct:32.0}
    settling_window_failure_given_reach_le_3cm: {count:23,pct:46.0}
    reached_only_after_tc_then_diverged: {count:7,pct:14.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:1,pct:2.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:16,pct:32.0}
    3_reached_post_tc_then_diverged: {count:7,pct:14.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:2,pct:4.0}
  summary_stats:
    min_err_cm_mean: 0.9899138957262039
    min_err_cm_std: 0.6084400484165465
    pre_tc_min_err_cm_mean: 1.211065331697464
    tc_err_cm_mean: 2.5251586663722994
    tc_err_cm_std: 1.3941924101905212
    term_err_cm_mean: 2.6833017432689665
    term_err_cm_std: 1.3172601917506164
    err_jump_tc_to_grasp_cm_mean: 0.15814307689666748
    final_vel_mean: 0.3702622719109058
    final_vel_std: 0.2854822039155623
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.010471052303910256
    mean_joint_jerk_rad_s3: 0.008151772329583765
    mean_direction_flips: 24.72
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.0643702286481855,std_cm:1.2545159499072607,median_cm:1.6970002055168152,n:50}
    t_c-2: {mean_cm:2.2092204105854036,std_cm:1.3114178519116673,median_cm:1.8127614855766296,n:50}
    t_c-1: {mean_cm:2.365184108018875,std_cm:1.3616427372421689,median_cm:1.9652366638183594,n:50}
    t_c: {mean_cm:2.5251586663722994,std_cm:1.3941924101905212,median_cm:2.0829561948776245,n:50}
    t_c+1: {mean_cm:3.0815413963794707,std_cm:1.72407927359815,median_cm:2.6899569034576416,n:50}
    t_c+2: {mean_cm:2.6947919595241547,std_cm:1.2577469491773068,median_cm:2.5787729024887085,n:50}
    t_c+3: {mean_cm:2.5336720967292785,std_cm:1.2031166319767301,median_cm:2.319764256477356,n:50}
    t_c+4: {mean_cm:2.7471174240112304,std_cm:1.2616383554563007,median_cm:2.73140287399292,n:50}
    t_c+5: {mean_cm:2.8802491390705107,std_cm:1.3981469136345672,median_cm:2.9476654529571533,n:50}
    t_c+6: {mean_cm:2.8289798361063005,std_cm:1.4001974132431474,median_cm:2.821049213409424,n:50}
    t_c+7: {mean_cm:2.6833017432689665,std_cm:1.3172601917506164,median_cm:2.666401505470276,n:50}
SPWM N=25 Val-Sel (Seed 42):
  label: SPWM N=25 Val-Sel (Seed 42)
  n_episodes: 50
  n_success: 29
  n_fail: 21
  success_rate_pct: 58.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:2,pct:4.0}
    b_velocity_failure_given_reach_le_3cm: {count:0,pct:0.0}
    b_velocity_failure_all_episodes: {count:0,pct:0.0}
    c_post_contact_impulse_divergence: {count:9,pct:18.0}
    settling_window_failure_given_reach_le_3cm: {count:11,pct:22.0}
    reached_only_after_tc_then_diverged: {count:2,pct:4.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:2,pct:4.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:9,pct:18.0}
    3_reached_post_tc_then_diverged: {count:2,pct:4.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:8,pct:16.0}
  summary_stats:
    min_err_cm_mean: 0.9966417002677918
    min_err_cm_std: 0.7168366082198878
    pre_tc_min_err_cm_mean: 1.537956092953682
    tc_err_cm_mean: 2.233934363126755
    tc_err_cm_std: 1.4055678232228594
    term_err_cm_mean: 2.117010636925697
    term_err_cm_std: 1.2261713068281301
    err_jump_tc_to_grasp_cm_mean: -0.11692372620105744
    final_vel_mean: 0.37095459580421447
    final_vel_std: 0.2564984813973722
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.005460824174806475
    mean_joint_jerk_rad_s3: 0.003033191980794072
    mean_direction_flips: 11.82
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.570709978342056,std_cm:1.7423274454933606,median_cm:2.084847569465637,n:50}
    t_c-2: {mean_cm:2.339778987169266,std_cm:1.5107181017190565,median_cm:2.0399178862571716,n:50}
    t_c-1: {mean_cm:2.2121916997432707,std_cm:1.4194959177742037,median_cm:1.8959782719612122,n:50}
    t_c: {mean_cm:2.233934363126755,std_cm:1.4055678232228594,median_cm:1.890315294265747,n:50}
    t_c+1: {mean_cm:2.8417738568782807,std_cm:2.451966373256439,median_cm:2.03122878074646,n:50}
    t_c+2: {mean_cm:2.5271481239795683,std_cm:2.055432999759821,median_cm:1.9524443745613098,n:50}
    t_c+3: {mean_cm:2.2913997495174407,std_cm:1.698457269294526,median_cm:1.7559440732002258,n:50}
    t_c+4: {mean_cm:2.149617938995361,std_cm:1.4415171284106294,median_cm:1.7719542384147644,n:50}
    t_c+5: {mean_cm:2.083803117275238,std_cm:1.303564689006201,median_cm:1.669554352760315,n:50}
    t_c+6: {mean_cm:2.085492214560509,std_cm:1.230028941964752,median_cm:1.6684231758117676,n:50}
    t_c+7: {mean_cm:2.117010636925697,std_cm:1.2261713068281301,median_cm:1.6657440662384033,n:50}
SPWM N=25 Val-Sel (Seed 101):
  label: SPWM N=25 Val-Sel (Seed 101)
  n_episodes: 50
  n_success: 31
  n_fail: 19
  success_rate_pct: 62.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:0,pct:0.0}
    b_velocity_failure_given_reach_le_3cm: {count:1,pct:2.0}
    b_velocity_failure_all_episodes: {count:1,pct:2.0}
    c_post_contact_impulse_divergence: {count:9,pct:18.0}
    settling_window_failure_given_reach_le_3cm: {count:11,pct:22.0}
    reached_only_after_tc_then_diverged: {count:2,pct:4.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:0,pct:0.0}
    2a_post_contact_div_plus_vel_violation: {count:0,pct:0.0}
    2b_post_contact_div_spatial_only: {count:9,pct:18.0}
    3_reached_post_tc_then_diverged: {count:2,pct:4.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:8,pct:16.0}
  summary_stats:
    min_err_cm_mean: 0.8499453981220723
    min_err_cm_std: 0.5894733005655424
    pre_tc_min_err_cm_mean: 1.00250588580966
    tc_err_cm_mean: 2.2484344738721846
    tc_err_cm_std: 1.0884728057829296
    term_err_cm_mean: 2.0169071877002716
    term_err_cm_std: 1.1516123596745238
    err_jump_tc_to_grasp_cm_mean: -0.23152728617191315
    final_vel_mean: 0.5147094203531742
    final_vel_std: 0.3466832310637248
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.012158537823706865
    mean_joint_jerk_rad_s3: 0.00939610967412591
    mean_direction_flips: 28.62
  step_error_profile_around_tc:
    t_c-3: {mean_cm:2.127611813545227,std_cm:0.9468546353915807,median_cm:2.005717933177948,n:50}
    t_c-2: {mean_cm:2.1022756683826445,std_cm:0.9960624586989392,median_cm:1.9255191087722778,n:50}
    t_c-1: {mean_cm:2.113300464749336,std_cm:0.9904704048268995,median_cm:2.149985909461975,n:50}
    t_c: {mean_cm:2.2484344738721846,std_cm:1.0884728057829296,median_cm:2.377298951148987,n:50}
    t_c+1: {mean_cm:2.91045338511467,std_cm:2.276015684275159,median_cm:2.275682210922241,n:50}
    t_c+2: {mean_cm:2.4622361075878145,std_cm:1.8239405276415812,median_cm:1.8720372319221497,n:50}
    t_c+3: {mean_cm:2.0902341371774673,std_cm:1.3658186732217235,median_cm:1.6477518677711487,n:50}
    t_c+4: {mean_cm:1.9407165196537972,std_cm:1.100617004044868,median_cm:1.6595511436462402,n:50}
    t_c+5: {mean_cm:1.9036283481121063,std_cm:1.06899163268589,median_cm:1.7284755110740662,n:50}
    t_c+6: {mean_cm:1.9438227516412736,std_cm:1.047814852890829,median_cm:1.7264410853385925,n:50}
    t_c+7: {mean_cm:2.0169071877002716,std_cm:1.1516123596745238,median_cm:1.7109362483024597,n:50}
SPWM N=25 Val-Sel (Seed 2024):
  label: SPWM N=25 Val-Sel (Seed 2024)
  n_episodes: 50
  n_success: 31
  n_fail: 19
  success_rate_pct: 62.0
  failure_modes_non_exclusive:
    a_distance_failure_gt_3cm: {count:0,pct:0.0}
    b_velocity_failure_given_reach_le_3cm: {count:1,pct:2.0}
    b_velocity_failure_all_episodes: {count:1,pct:2.0}
    c_post_contact_impulse_divergence: {count:10,pct:20.0}
    settling_window_failure_given_reach_le_3cm: {count:12,pct:24.0}
    reached_only_after_tc_then_diverged: {count:2,pct:4.0}
  failure_modes_mutually_exclusive:
    1_distance_never_reached_3cm: {count:0,pct:0.0}
    2a_post_contact_div_plus_vel_violation: {count:1,pct:2.0}
    2b_post_contact_div_spatial_only: {count:9,pct:18.0}
    3_reached_post_tc_then_diverged: {count:2,pct:4.0}
    4_settling_passed_vel_failed_only: {count:0,pct:0.0}
    5_settling_and_vel_passed_rot_failed_only: {count:7,pct:14.0}
  summary_stats:
    min_err_cm_mean: 0.7930243809521198
    min_err_cm_std: 0.5496822026263848
    pre_tc_min_err_cm_mean: 1.041318565905094
    tc_err_cm_mean: 1.6540947642922401
    tc_err_cm_std: 0.9597142281560219
    term_err_cm_mean: 1.8860684871673583
    term_err_cm_std: 1.187425109721587
    err_jump_tc_to_grasp_cm_mean: 0.23197372287511825
    final_vel_mean: 0.46489599913358687
    final_vel_std: 0.2882770391033404
    effective_vel_threshold_mean: 1.5
    mean_joint_accel_rad_s2: 0.01030723424628377
    mean_joint_jerk_rad_s3: 0.007473995992913842
    mean_direction_flips: 22.98
  step_error_profile_around_tc:
    t_c-3: {mean_cm:1.7090584415197372,std_cm:1.0038515226335576,median_cm:1.5093061923980713,n:50}
    t_c-2: {mean_cm:1.6376670563220979,std_cm:0.950286313076338,median_cm:1.3231131434440613,n:50}
    t_c-1: {mean_cm:1.5959379148483277,std_cm:0.9457722896731929,median_cm:1.3848624229431152,n:50}
    t_c: {mean_cm:1.6540947642922401,std_cm:0.9597142281560219,median_cm:1.561939001083374,n:50}
    t_c+1: {mean_cm:2.8901937019824984,std_cm:2.869409902335871,median_cm:1.8953220844268799,n:50}
    t_c+2: {mean_cm:2.4392507207393646,std_cm:2.3982279019048067,median_cm:1.6679897904396057,n:50}
    t_c+3: {mean_cm:1.9912224166095256,std_cm:1.6911716131687784,median_cm:1.4298081398010254,n:50}
    t_c+4: {mean_cm:1.7378071856498718,std_cm:1.0693787755960384,median_cm:1.4254108667373657,n:50}
    t_c+5: {mean_cm:1.7148064762353896,std_cm:1.0352048752789338,median_cm:1.4202589392662048,n:50}
    t_c+6: {mean_cm:1.8477609047293664,std_cm:1.1551097605389506,median_cm:1.5609533786773682,n:50}
    t_c+7: {mean_cm:1.8860684871673583,std_cm:1.187425109721587,median_cm:1.6412156224250793,n:50}
Continuous RSSM Un-Gated (3-Seed Aggregate):
  total_episodes: 150
  success_total: 0
  reach_le_3cm_total: 144
  a_distance_failure_gt_3cm_total: 6
  b_velocity_failure_all_total: 150
  b_velocity_failure_given_reach_total: 144
  c_post_contact_impulse_divergence_total: 143
  excl_never_reached_total: 6
  excl_post_contact_div_plus_vel_total: 143
  excl_post_contact_div_spatial_only_total: 0
  excl_post_tc_reach_then_div_total: 1
  excl_vel_only_total: 0
  excl_rot_only_total: 0
Continuous RSSM Residual-Gated (3-Seed Aggregate):
  total_episodes: 150
  success_total: 23
  reach_le_3cm_total: 148
  a_distance_failure_gt_3cm_total: 2
  b_velocity_failure_all_total: 24
  b_velocity_failure_given_reach_total: 24
  c_post_contact_impulse_divergence_total: 50
  excl_never_reached_total: 2
  excl_post_contact_div_plus_vel_total: 12
  excl_post_contact_div_spatial_only_total: 38
  excl_post_tc_reach_then_div_total: 22
  excl_vel_only_total: 3
  excl_rot_only_total: 50
Deep GRU Un-Gated (3-Seed Aggregate):
  total_episodes: 150
  success_total: 0
  reach_le_3cm_total: 132
  a_distance_failure_gt_3cm_total: 18
  b_velocity_failure_all_total: 18
  b_velocity_failure_given_reach_total: 17
  c_post_contact_impulse_divergence_total: 123
  excl_never_reached_total: 18
  excl_post_contact_div_plus_vel_total: 17
  excl_post_contact_div_spatial_only_total: 106
  excl_post_tc_reach_then_div_total: 3
  excl_vel_only_total: 0
  excl_rot_only_total: 6
Deep GRU Residual-Gated (3-Seed Aggregate):
  total_episodes: 150
  success_total: 115
  reach_le_3cm_total: 141
  a_distance_failure_gt_3cm_total: 9
  b_velocity_failure_all_total: 0
  b_velocity_failure_given_reach_total: 0
  c_post_contact_impulse_divergence_total: 9
  excl_never_reached_total: 9
  excl_post_contact_div_plus_vel_total: 0
  excl_post_contact_div_spatial_only_total: 9
  excl_post_tc_reach_then_div_total: 11
  excl_vel_only_total: 0
  excl_rot_only_total: 6
ANN-Rate (3-Seed Aggregate):
  total_episodes: 150
  success_total: 56
  reach_le_3cm_total: 146
  a_distance_failure_gt_3cm_total: 4
  b_velocity_failure_all_total: 2
  b_velocity_failure_given_reach_total: 2
  c_post_contact_impulse_divergence_total: 56
  excl_never_reached_total: 4
  excl_post_contact_div_plus_vel_total: 0
  excl_post_contact_div_spatial_only_total: 56
  excl_post_tc_reach_then_div_total: 17
  excl_vel_only_total: 1
  excl_rot_only_total: 16
SPWM N=25 Val-Sel (3-Seed Aggregate):
  total_episodes: 150
  success_total: 91
  reach_le_3cm_total: 148
  a_distance_failure_gt_3cm_total: 2
  b_velocity_failure_all_total: 2
  b_velocity_failure_given_reach_total: 2
  c_post_contact_impulse_divergence_total: 28
  excl_never_reached_total: 2
  excl_post_contact_div_plus_vel_total: 1
  excl_post_contact_div_spatial_only_total: 27
  excl_post_tc_reach_then_div_total: 6
  excl_vel_only_total: 0
  excl_rot_only_total: 23
```

### `results/energy_profiling/energy_profiling_metrics.json`
> Neuromorphic SOPs (Loihi-2 standby recurrent core) vs. GPU MACs (Jetson Orin E2E MPPI) profiling across SPWM and baselines

```yaml
SPWM:
  paradigm: Hybrid SNN-ALIF + M1
  memory_type: ALIF Spiking (2-tier)
  params: 392141
  spike_rate: 0.12268746128449073
  sops_per_step: 8229
  recurrent_macs: 67072
  recurrent_flops: 134144
  energy_neuro_nj: 12.343340104910045
  energy_gpu_nj: 60.364799999999995
  energy_ratio_vs_gpu: 4.890475307893975
  power_20hz_mw: 0.2468668020982009
  cumulative_energy_h100_uj: 1.2343340104910043
  encoder_macs: 882688
  encoder_energy_nj: 794.4191999999999
  mppi_rollout_macs: 1218969600
  mppi_rollout_energy_nj: 1097072.64
  e2e_step_energy_nj: 1097879.4025401047
  e2e_step_energy_mj: 1.0978794025401046
  e2e_all_gpu_energy_mj: 1.0979274239999999
  e2e_power_20hz_mw: 21.957588050802094
ANN-Rate:
  paradigm: Continuous ANN
  memory_type: Continuous Leaky Rate
  params: 392141
  spike_rate: 0.0
  sops_per_step: 0
  recurrent_macs: 67072
  recurrent_flops: 134144
  energy_neuro_nj: null
  energy_gpu_nj: 60.364799999999995
  energy_ratio_vs_gpu: 1.0
  power_20hz_mw: 1.207296
  cumulative_energy_h100_uj: 6.036479999999999
  encoder_macs: 882688
  encoder_energy_nj: 794.4191999999999
  mppi_rollout_macs: 1218969600
  mppi_rollout_energy_nj: 1097072.64
  e2e_step_energy_nj: 1097927.4239999999
  e2e_step_energy_mj: 1.0979274239999999
  e2e_all_gpu_energy_mj: 1.0979274239999999
  e2e_power_20hz_mw: 21.958548479999997
Continuous RSSM:
  paradigm: Continuous ANN
  memory_type: GRU + Gaussian Latent
  params: 380460
  spike_rate: 0.0
  sops_per_step: 0
  recurrent_macs: 365056
  recurrent_flops: 730112
  energy_neuro_nj: null
  energy_gpu_nj: 328.55039999999997
  energy_ratio_vs_gpu: 1.0
  power_20hz_mw: 6.571008
  cumulative_energy_h100_uj: 32.855039999999995
  encoder_macs: 882688
  encoder_energy_nj: 794.4191999999999
  mppi_rollout_macs: 2358312960
  mppi_rollout_energy_nj: 2122481.664
  e2e_step_energy_nj: 2123604.6336
  e2e_step_energy_mj: 2.1236046336
  e2e_all_gpu_energy_mj: 2.1236046336
  e2e_power_20hz_mw: 42.472092671999995
Deep GRU:
  paradigm: Continuous ANN
  memory_type: 2-Layer Deep GRU
  params: 338868
  spike_rate: 0.0
  sops_per_step: 0
  recurrent_macs: 299520
  recurrent_flops: 599040
  energy_neuro_nj: null
  energy_gpu_nj: 269.568
  energy_ratio_vs_gpu: 1.0
  power_20hz_mw: 5.39136
  cumulative_energy_h100_uj: 26.956799999999998
  encoder_macs: 882688
  encoder_energy_nj: 794.4191999999999
  mppi_rollout_macs: 1942179840
  mppi_rollout_energy_nj: 1747961.856
  e2e_step_energy_nj: 1749025.8432
  e2e_step_energy_mj: 1.7490258432
  e2e_all_gpu_energy_mj: 1.7490258432
  e2e_power_20hz_mw: 34.980516864
Continuous ConvLSTM:
  paradigm: Continuous ANN
  memory_type: Spatial ConvLSTM (2D)
  params: 296892
  spike_rate: 0.0
  sops_per_step: 0
  recurrent_macs: 2359296
  recurrent_flops: 4718592
  energy_neuro_nj: null
  energy_gpu_nj: 2123.3664
  energy_ratio_vs_gpu: 1.0
  power_20hz_mw: 42.467327999999995
  cumulative_energy_h100_uj: 212.33664
  encoder_macs: 882688
  encoder_energy_nj: 794.4191999999999
  mppi_rollout_macs: 111311585280
  mppi_rollout_energy_nj: 100180426.752
  e2e_step_energy_nj: 100183344.53760001
  e2e_step_energy_mj: 100.18334453760001
  e2e_all_gpu_energy_mj: 100.18334453760001
  e2e_power_20hz_mw: 2003.6668907520004
```

### `results/baselines/classical_and_bc_pil/classical_bc_pil_metrics.json`
> 50-episode PIL benchmarks for Privileged Fixed-Gain PD and Reactive BC

```yaml
fixed_gain_pd_pil:
  model_label: fixed_pd
  episodes_evaluated: 50
  grasp_success_rate_pct: 96.0
  terminal_placement_accuracy_cm: 0.011748063395862119
  placement_accuracy_std_cm: 0.02977501137910883
  terminal_error_cm: 0.429177088290453
  terminal_error_std_cm: 0.7286152381204417
  mean_joint_accel_rad_s2: 0.007706445837393403
  std_joint_accel_rad_s2: 0.0038599028083092187
  mean_joint_jerk_rad_s3: 0.004971938964445144
  std_joint_jerk_rad_s3: 0.0029003430699155184
  mean_direction_flips: 10.46
  mppi_inference_latency_ms: 22.683568809901676
reactive_bc_pil:
  model_label: bc
  episodes_evaluated: 50
  grasp_success_rate_pct: 0.0
  terminal_placement_accuracy_cm: 13.129227232933044
  placement_accuracy_std_cm: 6.274945649634276
  terminal_error_cm: 15.657341589927674
  terminal_error_std_cm: 8.150904676156873
  mean_joint_accel_rad_s2: 0.0014032347162719816
  std_joint_accel_rad_s2: 0.0007124819846246704
  mean_joint_jerk_rad_s3: 0.001350532578653656
  std_joint_jerk_rad_s3: 0.0007350296402323977
  mean_direction_flips: 13.98
  mppi_inference_latency_ms: 1.8653048771781349
  parameter_count: 322210
```

### `results/spwm_v0_12_4_train_m1/metrics_spwm_v0_12_4.json`
> Phase 1 rollout, horizon metrics, training progression, and MPPI progression

```yaml
version: spwm_v0_12_4
model_name: spwm_v0_12_4_train_m1
total_epochs: 240
parameters: 392141
evaluation:
  teacher_forcing_mse: 0.00675271562515543
  mean_spike_rate: 0.16886643606882829
  rollout_mae: 0.05499629222620756
  rollout_score: 0.380692321453876
  dynamic_score: 0.380692321453876
  drift_ratio: 2.6913710582652475
  horizons: [1,5,10,25,50,100]
  latent_mse_per_horizon: {1:0.003632322885096073,5:0.016396312759472773,10:0.029200009141977016,25:0.05573204474953505,50:0.1523182770380607,100:0.47278271500880903}
  position_error_per_horizon: {1:0.02186665225487489,5:0.019092637472427808,10:0.028008795988101225,25:0.05885127501992079,50:0.0835205213381694,100:0.11863787128375126}
  velocity_error_per_horizon: {1:0.04313335882929655,5:0.0956632042160401,10:0.2146202681156305,25:0.25007535173342776,50:0.27648258209228516,100:0.3515469202628502}
  position_mse_per_horizon: {1:8.261288446821988e-05,5:6.561665549480285e-05,10:0.00025150043634661974,25:0.0010574773813669498,50:0.0017551812791050626,100:0.0035010092318631136}
  velocity_mse_per_horizon: {1:0.0003449322136405569,5:0.001802860750243641,10:0.016012800212662954,25:0.020002329005644873,50:0.023568693835001726,100:0.03643064301174421}
  q_mse_per_horizon: {1:0.00024637862113125337,5:0.002479826297181157,10:0.0106030679666079,25:0.04317507520318031,50:0.144227534532547,100:0.7817606238218454}
  p_mse_per_horizon: {1:0.004760970970472464,5:0.021035141239945706,10:0.03539899020240857,25:0.05991770298435138,50:0.15501519464529478,100:0.36979007491698634}
  physical_rollout_degradation:
    position_relative: {1:1.0,5:0.8731394842651943,10:1.280890904635712,25:2.6913710582652475,50:3.8195385541721216,100:5.425515982095635}
    velocity_relative: {1:1.0,5:2.2178473184672285,10:4.975737432482503,25:5.797724974841849,50:6.409947882484306,100:8.150232901038917}
    position_normalized: {1:0.010933326127437444,5:0.009546318736213904,10:0.014004397994050613,25:0.029425637509960394,50:0.0417602606690847,100:0.05931893564187563}
    velocity_normalized: {1:0.021566679414648276,5:0.04783160210802005,10:0.10731013405781525,25:0.12503767586671388,50:0.13824129104614258,100:0.1757734601314251}
  latent_rollout_degradation:
    latent_relative: {1:1.0,5:4.514001997660816,10:8.038935432141432,25:15.343361951166674,50:41.934123660383776,100:130.15988114622255}
    q_relative: {1:1.0,5:10.065103399779472,10:43.035665667433555,25:175.23872406193735,50:585.3898113006835,100:3173.0051099091825}
    p_relative: {1:1.0,5:4.418246061655411,10:7.435245965991613,25:12.585185533783108,50:32.55957568460275,100:77.6711467493551}
training_progression:
  total_epochs: 240
  num_segments: 10
  best_checkpoint: {epoch:200,score:2.866389}
  trends:
    train_loss: {initial:65.823114,final:0.568458,best:0.262027,best_epoch:179,change:-65.254656,pct_change:-99.14%}
    val_loss: {initial:22.597798,final:1.078581,best:0.248181,best_epoch:148,change:-21.519217,pct_change:-95.23%}
    train_pos_err: {initial:0.764842,final:0.108404,best:0.078147,best_epoch:182,change:-0.656438,pct_change:-85.83%}
    val_pos_err: {initial:0.619869,final:0.133633,best:0.07476,best_epoch:143,change:-0.486236,pct_change:-78.44%}
    train_vel_err: {initial:0.977212,final:0.26272,best:0.204905,best_epoch:149,change:-0.714492,pct_change:-73.12%}
    val_vel_err: {initial:0.976442,final:0.362843,best:0.201886,best_epoch:143,change:-0.613599,pct_change:-62.84%}
    score: {initial:6.067135,final:3.313515,best:2.866389,best_epoch:200,change:-2.75362,pct_change:-45.39%}
    mppi_score: {initial:6.067135,final:3.313515,best:2.866389,best_epoch:200,change:-2.75362,pct_change:-45.39%}
    mppi_cost: {initial:3.813159,final:0.165541,best:0.165541,best_epoch:240,change:-3.647618,pct_change:-95.66%}
    mppi_success_rate: {initial:0.0,final:1.0,best:1.0,best_epoch:150,change:1.0,pct_change:+0.00%}
    mppi_mean_error_cm: {initial:9.664292,final:6.66782,best:5.740013,best_epoch:200,change:-2.996472,pct_change:-31.01%}
    mppi_grasp_error_cm: {initial:5.680084,final:2.442919,best:1.472745,best_epoch:25,change:-3.237165,pct_change:-56.99%}
    mppi_grasp_rot_err_rad: {initial:0.199365,final:0.025015,best:0.01741,best_epoch:145,change:-0.17435,pct_change:-87.45%}
    mppi_terminal_error_cm: {initial:1.070882,final:0.577093,best:0.577093,best_epoch:240,change:-0.49379,pct_change:-46.11%}
    mppi_steady_state_error_cm: {initial:1.184397,final:0.757239,best:0.749547,best_epoch:215,change:-0.427159,pct_change:-36.07%}
    mppi_min_error_cm: {initial:1.001798,final:0.511224,best:0.420724,best_epoch:215,change:-0.490574,pct_change:-48.97%}
    mppi_rot_err_rad: {initial:0.094147,final:0.010447,best:0.010447,best_epoch:240,change:-0.0837,pct_change:-88.90%}
    mppi_final_vel: {initial:0.305288,final:0.255882,best:0.190651,best_epoch:15,change:-0.049406,pct_change:-16.18%}
    mppi_final_rot: {initial:0.094147,final:0.010447,best:0.010447,best_epoch:240,change:-0.0837,pct_change:-88.90%}
    mppi_pos_penalty: {initial:0.161571,final:0.074259,best:0.073816,best_epoch:205,change:-0.087312,pct_change:-54.04%}
    mppi_vel_penalty: {initial:0.043939,final:0.029689,best:0.017521,best_epoch:15,change:-0.01425,pct_change:-32.43%}
    mppi_rot_penalty: {initial:3.60765,final:0.061594,best:0.061594,best_epoch:240,change:-3.546056,pct_change:-98.29%}
    train_mppi_noise_alpha: {initial:0.0,final:0.25,best:0.0,best_epoch:1,change:0.25,pct_change:+0.00%}
    mppi_expert_trajectories: {initial:4.0,final:0.0,best:0.0,best_epoch:220,change:-4.0,pct_change:-100.00%}
  sampled_segments (10 rows):
    |segment|epoch|progress_pct|train_loss|val_loss|train_pos_err|val_pos_err|train_vel_err|val_vel_err|score|train_mppi_noise_alpha|mppi_score|mppi_cost|mppi_success_rate|mppi_mean_error_cm|mppi_grasp_error_cm|mppi_grasp_rot_err_rad|mppi_terminal_error_cm|mppi_steady_state_error_cm|mppi_min_error_cm|mppi_rot_err_rad|mppi_final_vel|mppi_final_rot|mppi_pos_penalty|mppi_vel_penalty|mppi_rot_penalty|mppi_expert_trajectories|mppi_expert_blocks|
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    |1|1|0.0%|65.823114|22.597798|0.764842|0.619869|0.977212|0.976442|null|0.0||||||||||||||||||
    |2|28|11.1%|1.012618|1.412534|0.164461|0.193228|0.401192|0.42097|3.523622|0.0||||||||||||||||||
    |3|54|22.2%|0.507682|0.584368|0.113526|0.105642|0.289188|0.30428|3.523622|0.0||||||||||||||||||
    |4|81|33.3%|0.410032|0.361417|0.100362|0.099617|0.245536|0.251311|3.523622|0.0||||||||||||||||||
    |5|107|44.4%|0.324666|0.347461|0.088973|0.087108|0.220055|0.226915|3.523622|0.0||||||||||||||||||
    |6|134|55.6%|0.317738|0.307969|0.089462|0.08472|0.220372|0.215222|3.523622|0.0||||||||||||||||||
    |7|160|66.7%|0.2995|0.274369|0.084628|0.084181|0.216568|0.207386|3.730349|0.0|3.730349|0.54552|1.0|5.941489|3.244716|0.023045|0.834735|1.080952|0.77975|0.027319|0.275049|0.027319|0.162586|0.035087|0.347847|4.0|[Low,Low,Mid,Mid]|
    |8|187|77.8%|0.429932|3.905435|0.096023|0.182986|0.277503|0.419829|3.472062|0.0||||||||||||||||||
    |9|213|88.9%|0.5119|1.045944|0.106881|0.130296|0.260242|0.373471|2.866389|0.25||||||||||||||||||
    |10|240|100.0%|0.568458|1.078581|0.108404|0.133633|0.26272|0.362843|3.313515|0.25|3.313515|0.165541|1.0|6.66782|2.442919|0.025015|0.577093|0.757239|0.511224|0.010447|0.255882|0.010447|0.074259|0.029689|0.061594|0.0|[]|
mppi_progression:
  total_evaluations: 48
  best_checkpoint: {epoch:200,score:2.866389,best_success_rate_pct:100.0}
  final_metrics:
    mppi_score: 3.313515
    success_rate_pct: 100.0
    mean_cartesian_error_cm: 6.668
    grasp_error_cm: 2.443
    grasp_rot_err_rad: 0.025
    terminal_error_cm: 0.577
    rotational_error_rad: 0.0104
    terminal_velocity_rad_s: 0.2559
  expert_execution:
    initial_expert_trajectories: 4
    final_expert_trajectories: 0
    curriculum_annealed_to_autonomous: true
    total_overwritten_trajectories: 166
    tracked_elevation_counts: {Low:55,Mid:84,High:27}
    final_natural_trajectory_ids: [3201,3202,3200,3204,3205,3203]
    final_natural_blocks: [Low,Mid,High,Low,Mid,Mid]
    final_expert_trajectory_ids: []
    final_elevation_breakdown:
      natural_low: {count:2,trajectory_ids:[3201,3204],avg_vel_err:0.229,avg_rot_err:0.0183,avg_cartesian_err_cm:4.745,avg_terminal_err_cm:0.934}
      natural_mid: {count:3,trajectory_ids:[3202,3205,3203],avg_vel_err:0.2794,avg_rot_err:0.0072,avg_cartesian_err_cm:7.069,avg_terminal_err_cm:0.331}
      natural_high: {count:1,trajectory_ids:[3200],avg_vel_err:0.2389,avg_rot_err:0.0046,avg_cartesian_err_cm:9.311,avg_terminal_err_cm:0.6}
    final_trials (6 rows):
      |trajectory_id|elevation_plane|is_expert|score|mean_cartesian_err_cm|terminal_error_cm|final_vel|final_rot|grasp_error_cm|grasp_rot_err_rad|success|
      |---|---|---|---|---|---|---|---|---|---|---|
      |3201|Low|false|0.481718|6.347|1.242|0.2743|0.0242|3.159|0.0518|true|
      |3202|Mid|false|0.115094|4.787|0.524|0.2703|0.0094|0.626|0.0149|true|
      |3200|High|false|0.065908|9.311|0.6|0.2389|0.0046|2.579|0.0132|true|
      |3204|Low|false|0.153745|3.142|0.627|0.1838|0.0124|0.492|0.0145|true|
      |3205|Mid|false|0.099074|8.453|0.313|0.2969|0.0046|5.117|0.0429|true|
      |3203|Mid|false|0.077709|7.968|0.157|0.2711|0.0076|2.685|0.0128|true|
  evaluations_history_summary (48 epochs):
    |epoch|mppi_score|success_rate_pct|mean_cartesian_error_cm|grasp_error_cm|terminal_error_cm|natural_trajectories_count|natural_blocks|natural_trajectory_ids|expert_trajectories_count|overwritten_blocks|overwritten_trajectory_ids|
    |---|---|---|---|---|---|---|---|---|---|---|---|
    |5|6.067135|0.0|9.664|5.68|1.071|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |10|5.995606|0.0|9.618|5.151|1.473|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |15|5.016235|0.0|9.327|3.324|1.488|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |20|4.008057|0.0|8.531|1.774|1.453|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |25|3.523622|0.0|7.516|1.473|1.302|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Low]|[3201,3202,3200,3204]|
    |30|3.587106|0.0|6.646|2.131|1.198|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Mid,High,Low,Mid]|[3202,3200,3204,3205]|
    |35|3.57019|50.0|5.832|2.519|1.204|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |40|4.707169|66.67|6.035|4.645|1.198|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |45|6.26773|50.0|6.572|7.367|1.257|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |50|6.269566|50.0|6.498|7.224|1.423|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |55|6.674172|50.0|6.748|8.015|1.387|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Low]|[3201,3202,3200,3204]|
    |60|6.699339|50.0|6.723|8.266|1.228|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Mid,High,Low,Mid]|[3202,3200,3204,3205]|
    |65|6.26075|83.33|6.548|7.508|1.16|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |70|6.715553|66.67|6.84|8.164|1.264|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |75|6.346787|66.67|6.655|7.679|1.165|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |80|6.20967|66.67|6.711|7.392|1.127|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |85|5.943709|83.33|6.451|7.064|1.124|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Low]|[3201,3202,3200,3204]|
    |90|5.865091|83.33|6.454|6.884|1.119|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Mid,High,Low,Mid]|[3202,3200,3204,3205]|
    |95|4.872372|83.33|5.981|5.311|1.002|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |100|4.994553|83.33|5.98|5.514|1.088|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |105|4.671514|83.33|5.883|4.996|1.031|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |110|4.652321|83.33|5.947|4.88|1.012|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |115|4.499358|83.33|5.852|4.624|1.021|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Low]|[3201,3202,3200,3204]|
    |120|4.460161|83.33|5.957|4.525|0.959|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Mid,High,Low,Mid]|[3202,3200,3204,3205]|
    |125|4.430693|83.33|5.923|4.486|0.99|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |130|4.096543|83.33|5.838|3.853|0.969|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |135|4.110275|83.33|5.879|3.902|0.949|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |140|3.971992|83.33|5.79|3.715|0.959|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |145|4.033666|83.33|5.861|3.798|0.933|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Low]|[3201,3202,3200,3204]|
    |150|3.772642|100.0|5.827|3.262|0.997|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Mid,High,Low,Mid]|[3202,3200,3204,3205]|
    |155|3.786047|100.0|5.964|3.347|0.853|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |160|3.730349|100.0|5.941|3.245|0.835|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |165|3.538581|100.0|5.885|2.888|0.871|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |170|3.718813|100.0|6.012|3.117|0.924|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |175|3.61896|100.0|6.014|3.121|0.77|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Low]|[3201,3202,3200,3204]|
    |180|3.653562|100.0|6.045|3.142|0.801|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Mid,High,Low,Mid]|[3202,3200,3204,3205]|
    |185|3.472062|100.0|5.947|2.868|0.765|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[High,Low,Mid,Mid]|[3200,3204,3205,3203]|
    |190|3.004768|100.0|5.846|2.16|0.618|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Low,Mid,Mid]|[3201,3204,3205,3203]|
    |195|3.027478|100.0|5.905|2.157|0.705|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,Mid,Mid]|[3201,3202,3205,3203]|
    |200|2.866389|100.0|5.74|2.0|0.636|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|4|[Low,Mid,High,Mid]|[3201,3202,3200,3203]|
    |205|2.988386|100.0|5.779|2.223|0.616|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|3|[Mid,High,Low]|[3202,3200,3204]|
    |210|3.009404|100.0|5.865|2.217|0.601|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|2|[Low,Mid]|[3204,3205]|
    |215|3.010731|100.0|5.98|2.185|0.591|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|1|[Mid]|[3203]|
    |220|3.046988|100.0|6.21|2.14|0.592|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|0|[]|[]|
    |225|3.143146|100.0|6.375|2.233|0.604|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|0|[]|[]|
    |230|3.196234|100.0|6.415|2.326|0.6|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|0|[]|[]|
    |235|3.245321|100.0|6.598|2.319|0.605|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|0|[]|[]|
    |240|3.313515|100.0|6.668|2.443|0.577|6|[Low,Mid,High,Low,Mid,Mid]|[3201,3202,3200,3204,3205,3203]|0|[]|[]|
  evaluations_history_elevation_breakdown (256 rows):
    |epoch|tier|count|trajectory_ids|avg_vel_err|avg_rot_err|avg_cartesian_err_cm|avg_terminal_err_cm|
    |---|---|---|---|---|---|---|---|
    |5|natural_low|2|[3201,3204]|0.2477|0.1026|8.215|1.172|
    |5|natural_mid|3|[3202,3205,3203]|0.3557|0.0911|9.62|1.029|
    |5|natural_high|1|[3200]|0.2691|0.0865|12.696|0.996|
    |5|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |5|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |5|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |10|natural_low|2|[3201,3204]|0.2295|0.1246|8.766|1.282|
    |10|natural_mid|3|[3202,3205,3203]|0.2037|0.1144|9.6|1.387|
    |10|natural_high|1|[3200]|0.1167|0.1108|11.377|2.114|
    |10|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |10|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |15|natural_low|2|[3201,3204]|0.2406|0.1169|8.479|1.304|
    |15|natural_mid|3|[3202,3205,3203]|0.1827|0.1043|9.359|1.42|
    |15|natural_high|1|[3200]|0.1148|0.098|10.924|2.059|
    |15|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |15|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |20|natural_low|2|[3201,3204]|0.2842|0.1015|7.208|1.254|
    |20|natural_mid|3|[3202,3205,3203]|0.222|0.0945|8.618|1.394|
    |20|natural_high|1|[3200]|0.1048|0.0901|10.913|2.029|
    |20|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |20|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |20|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |25|natural_low|2|[3201,3204]|0.3752|0.0793|6.0|1.187|
    |25|natural_mid|3|[3202,3205,3203]|0.3519|0.0762|7.605|1.236|
    |25|natural_high|1|[3200]|0.1998|0.0693|10.281|1.732|
    |25|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |25|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |25|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |30|natural_low|2|[3201,3204]|0.4527|0.0667|5.098|1.323|
    |30|natural_mid|3|[3202,3205,3203]|0.4346|0.0617|6.569|1.107|
    |30|natural_high|1|[3200]|0.3069|0.0521|9.972|1.221|
    |30|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |30|expert_mid|2|[3202,3205]|0.2586|0.0003|7.138|0.0|
    |30|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |35|natural_low|2|[3201,3204]|0.4844|0.0586|4.553|1.522|
    |35|natural_mid|3|[3202,3205,3203]|0.4757|0.0535|5.598|1.129|
    |35|natural_high|1|[3200]|0.3747|0.0378|9.091|0.789|
    |35|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |35|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |35|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |40|natural_low|2|[3201,3204]|0.4492|0.058|4.213|1.59|
    |40|natural_mid|3|[3202,3205,3203]|0.4466|0.0498|6.043|1.071|
    |40|natural_high|1|[3200]|0.3519|0.0397|9.657|0.796|
    |40|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |40|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |45|natural_low|2|[3201,3204]|0.4531|0.0559|4.439|1.617|
    |45|natural_mid|3|[3202,3205,3203]|0.4679|0.05|6.7|1.205|
    |45|natural_high|1|[3200]|0.4113|0.0357|10.45|0.693|
    |45|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |45|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |50|natural_low|2|[3201,3204]|0.4189|0.06|4.261|1.736|
    |50|natural_mid|3|[3202,3205,3203]|0.4538|0.0556|6.713|1.438|
    |50|natural_high|1|[3200]|0.4227|0.0319|10.324|0.749|
    |50|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |50|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |50|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |55|natural_low|2|[3201,3204]|0.3955|0.0583|4.106|1.784|
    |55|natural_mid|3|[3202,3205,3203]|0.4362|0.0519|7.307|1.379|
    |55|natural_high|1|[3200]|0.4202|0.0377|10.357|0.615|
    |55|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |55|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |55|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |60|natural_low|2|[3201,3204]|0.3639|0.0538|4.381|1.689|
    |60|natural_mid|3|[3202,3205,3203]|0.3664|0.0443|7.194|1.118|
    |60|natural_high|1|[3200]|0.3152|0.0287|9.996|0.636|
    |60|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |60|expert_mid|2|[3202,3205]|0.2586|0.0003|7.138|0.0|
    |60|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |65|natural_low|2|[3201,3204]|0.3496|0.053|4.161|1.69|
    |65|natural_mid|3|[3202,3205,3203]|0.3244|0.0401|6.909|0.993|
    |65|natural_high|1|[3200]|0.2383|0.0306|10.238|0.601|
    |65|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |65|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |65|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |70|natural_low|2|[3201,3204]|0.3372|0.0544|4.513|1.752|
    |70|natural_mid|3|[3202,3205,3203]|0.2964|0.0467|7.237|1.172|
    |70|natural_high|1|[3200]|0.2404|0.0299|10.305|0.561|
    |70|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |70|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |75|natural_low|2|[3201,3204]|0.354|0.0462|4.269|1.472|
    |75|natural_mid|3|[3202,3205,3203]|0.2986|0.0464|7.103|1.185|
    |75|natural_high|1|[3200]|0.2661|0.0312|10.077|0.494|
    |75|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |75|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |80|natural_low|2|[3201,3204]|0.3387|0.0496|4.603|1.594|
    |80|natural_mid|3|[3202,3205,3203]|0.2828|0.0408|7.005|1.03|
    |80|natural_high|1|[3200]|0.2401|0.0323|10.044|0.484|
    |80|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |80|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |80|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |85|natural_low|2|[3201,3204]|0.337|0.0491|4.075|1.599|
    |85|natural_mid|3|[3202,3205,3203]|0.2751|0.0418|6.948|1.022|
    |85|natural_high|1|[3200]|0.2234|0.0181|9.715|0.479|
    |85|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |85|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |85|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |90|natural_low|2|[3201,3204]|0.3328|0.0453|4.376|1.525|
    |90|natural_mid|3|[3202,3205,3203]|0.2546|0.0418|6.858|1.101|
    |90|natural_high|1|[3200]|0.2046|0.0254|9.399|0.36|
    |90|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |90|expert_mid|2|[3202,3205]|0.2586|0.0003|7.138|0.0|
    |90|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |95|natural_low|2|[3201,3204]|0.3273|0.0459|4.095|1.536|
    |95|natural_mid|3|[3202,3205,3203]|0.2567|0.0368|6.352|0.863|
    |95|natural_high|1|[3200]|0.1752|0.023|8.638|0.348|
    |95|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |95|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |95|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |100|natural_low|2|[3201,3204]|0.2931|0.0457|4.314|1.594|
    |100|natural_mid|3|[3202,3205,3203]|0.2429|0.039|6.314|0.986|
    |100|natural_high|1|[3200]|0.2024|0.0245|8.31|0.381|
    |100|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |100|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |105|natural_low|2|[3201,3204]|0.2857|0.0407|4.196|1.498|
    |105|natural_mid|3|[3202,3205,3203]|0.2382|0.0366|6.257|0.918|
    |105|natural_high|1|[3200]|0.1963|0.0205|8.139|0.435|
    |105|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |105|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |110|natural_low|2|[3201,3204]|0.2798|0.0465|4.296|1.654|
    |110|natural_mid|3|[3202,3205,3203]|0.2458|0.0337|6.158|0.792|
    |110|natural_high|1|[3200]|0.1923|0.0188|8.613|0.388|
    |110|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |110|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |110|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |115|natural_low|2|[3201,3204]|0.2864|0.0416|4.414|1.514|
    |115|natural_mid|3|[3202,3205,3203]|0.2442|0.0348|6.063|0.895|
    |115|natural_high|1|[3200]|0.1843|0.0162|8.094|0.413|
    |115|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |115|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |115|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |120|natural_low|2|[3201,3204]|0.2878|0.0405|4.266|1.458|
    |120|natural_mid|3|[3202,3205,3203]|0.2429|0.0305|6.174|0.835|
    |120|natural_high|1|[3200]|0.1941|0.0214|8.685|0.332|
    |120|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |120|expert_mid|2|[3202,3205]|0.2586|0.0003|7.138|0.0|
    |120|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |125|natural_low|2|[3201,3204]|0.2708|0.0403|4.268|1.525|
    |125|natural_mid|3|[3202,3205,3203]|0.2535|0.0336|6.214|0.853|
    |125|natural_high|1|[3200]|0.1789|0.0128|8.358|0.331|
    |125|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |125|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |125|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |130|natural_low|2|[3201,3204]|0.3176|0.0421|4.462|1.502|
    |130|natural_mid|3|[3202,3205,3203]|0.2569|0.0322|6.271|0.81|
    |130|natural_high|1|[3200]|0.2181|0.0196|7.292|0.384|
    |130|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |130|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |135|natural_low|2|[3201,3204]|0.3014|0.0399|4.482|1.445|
    |135|natural_mid|3|[3202,3205,3203]|0.2695|0.033|6.117|0.802|
    |135|natural_high|1|[3200]|0.1914|0.012|7.958|0.397|
    |135|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |135|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |140|natural_low|2|[3201,3204]|0.2963|0.0405|4.409|1.47|
    |140|natural_mid|3|[3202,3205,3203]|0.2749|0.0333|6.104|0.808|
    |140|natural_high|1|[3200]|0.2175|0.0152|7.609|0.39|
    |140|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |140|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |140|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |145|natural_low|2|[3201,3204]|0.2877|0.036|4.539|1.316|
    |145|natural_mid|3|[3202,3205,3203]|0.2737|0.0323|6.173|0.833|
    |145|natural_high|1|[3200]|0.1981|0.0122|7.569|0.469|
    |145|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |145|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |145|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |150|natural_low|2|[3201,3204]|0.298|0.0331|4.529|1.218|
    |150|natural_mid|3|[3202,3205,3203]|0.2802|0.039|6.224|1.062|
    |150|natural_high|1|[3200]|0.239|0.0171|7.235|0.36|
    |150|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |150|expert_mid|2|[3202,3205]|0.2586|0.0003|7.138|0.0|
    |150|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |155|natural_low|2|[3201,3204]|0.264|0.0343|4.755|1.35|
    |155|natural_mid|3|[3202,3205,3203]|0.2589|0.0275|6.208|0.696|
    |155|natural_high|1|[3200]|0.2426|0.0146|7.65|0.328|
    |155|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |155|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |155|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |160|natural_low|2|[3201,3204]|0.2797|0.0339|4.724|1.276|
    |160|natural_mid|3|[3202,3205,3203]|0.2805|0.0277|6.322|0.652|
    |160|natural_high|1|[3200]|0.2492|0.0131|7.232|0.498|
    |160|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |160|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |165|natural_low|2|[3201,3204]|0.2727|0.0318|4.506|1.187|
    |165|natural_mid|3|[3202,3205,3203]|0.2897|0.0328|6.28|0.844|
    |165|natural_high|1|[3200]|0.2256|0.0125|7.457|0.318|
    |165|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |165|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |170|natural_low|2|[3201,3204]|0.282|0.0376|4.776|1.416|
    |170|natural_mid|3|[3202,3205,3203]|0.2869|0.0327|6.201|0.81|
    |170|natural_high|1|[3200]|0.2354|0.0134|7.915|0.283|
    |170|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |170|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |170|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |175|natural_low|2|[3201,3204]|0.2366|0.0274|4.671|1.093|
    |175|natural_mid|3|[3202,3205,3203]|0.2793|0.0278|6.341|0.642|
    |175|natural_high|1|[3200]|0.2121|0.0014|7.719|0.506|
    |175|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |175|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |175|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |180|natural_low|2|[3201,3204]|0.2524|0.0296|4.75|1.236|
    |180|natural_mid|3|[3202,3205,3203]|0.2625|0.0235|6.304|0.618|
    |180|natural_high|1|[3200]|0.2397|0.008|7.855|0.479|
    |180|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |180|expert_mid|2|[3202,3205]|0.2586|0.0003|7.138|0.0|
    |180|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |185|natural_low|2|[3201,3204]|0.244|0.0312|4.577|1.202|
    |185|natural_mid|3|[3202,3205,3203]|0.28|0.0213|6.204|0.548|
    |185|natural_high|1|[3200]|0.2706|0.0075|7.917|0.54|
    |185|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |185|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |185|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |190|natural_low|2|[3201,3204]|0.2193|0.0221|4.441|1.037|
    |190|natural_mid|3|[3202,3205,3203]|0.2992|0.0166|6.199|0.399|
    |190|natural_high|1|[3200]|0.2604|0.0026|7.594|0.433|
    |190|expert_low|2|[3201,3204]|0.2314|0.0|6.077|0.0|
    |190|expert_mid|2|[3205,3203]|0.2605|0.0003|9.082|0.0|
    |195|natural_low|2|[3201,3204]|0.2089|0.0205|4.506|1.107|
    |195|natural_mid|3|[3202,3205,3203]|0.273|0.0134|6.156|0.557|
    |195|natural_high|1|[3200]|0.2769|0.0098|7.947|0.345|
    |195|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |195|expert_mid|3|[3202,3205,3203]|0.2436|0.0002|7.864|0.0|
    |200|natural_low|2|[3201,3204]|0.2157|0.0188|4.389|0.992|
    |200|natural_mid|3|[3202,3205,3203]|0.2816|0.0143|5.97|0.463|
    |200|natural_high|1|[3200]|0.245|0.0067|7.75|0.44|
    |200|expert_low|1|[3201]|0.3184|0.0|7.302|0.0|
    |200|expert_mid|2|[3202,3203]|0.2118|0.0|7.373|0.0|
    |200|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |205|natural_low|2|[3201,3204]|0.2493|0.0186|4.37|0.95|
    |205|natural_mid|3|[3202,3205,3203]|0.2993|0.0108|6.043|0.371|
    |205|natural_high|1|[3200]|0.2681|0.0089|7.807|0.685|
    |205|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |205|expert_mid|1|[3202]|0.2099|0.0|5.428|0.0|
    |205|expert_high|1|[3200]|0.1454|0.0|8.792|0.0|
    |210|natural_low|2|[3201,3204]|0.2294|0.0189|4.428|0.944|
    |210|natural_mid|3|[3202,3205,3203]|0.2877|0.0085|6.169|0.365|
    |210|natural_high|1|[3200]|0.2439|0.0083|7.828|0.625|
    |210|expert_low|1|[3204]|0.1444|0.0|4.852|0.0|
    |210|expert_mid|1|[3205]|0.3072|0.0005|8.848|0.0|
    |215|natural_low|2|[3201,3204]|0.2289|0.019|4.451|0.976|
    |215|natural_mid|3|[3202,3205,3203]|0.2914|0.0074|6.255|0.328|
    |215|natural_high|1|[3200]|0.267|0.006|8.212|0.608|
    |215|expert_mid|1|[3203]|0.2138|0.0|9.317|0.0|
    |220|natural_low|2|[3201,3204]|0.23|0.0178|4.566|0.936|
    |220|natural_mid|3|[3202,3205,3203]|0.2812|0.0073|6.578|0.349|
    |220|natural_high|1|[3200]|0.2531|0.0073|8.389|0.632|
    |225|natural_low|2|[3201,3204]|0.2247|0.018|4.63|0.94|
    |225|natural_mid|3|[3202,3205,3203]|0.284|0.0061|6.788|0.325|
    |225|natural_high|1|[3200]|0.2494|0.0092|8.629|0.769|
    |230|natural_low|2|[3201,3204]|0.2342|0.0186|4.754|0.971|
    |230|natural_mid|3|[3202,3205,3203]|0.2828|0.007|6.775|0.321|
    |230|natural_high|1|[3200]|0.248|0.0069|8.658|0.693|
    |235|natural_low|2|[3201,3204]|0.2289|0.0178|4.748|0.954|
    |235|natural_mid|3|[3202,3205,3203]|0.2923|0.0075|7.041|0.342|
    |235|natural_high|1|[3200]|0.2476|0.0072|8.967|0.697|
    |240|natural_low|2|[3201,3204]|0.229|0.0183|4.745|0.934|
    |240|natural_mid|3|[3202,3205,3203]|0.2794|0.0072|7.069|0.331|
    |240|natural_high|1|[3200]|0.2389|0.0046|9.311|0.6|
  evaluations_history_trials (454 rows):
    |epoch|trajectory_id|elevation_plane|is_expert|score|mean_cartesian_err_cm|terminal_error_cm|final_vel|final_rot|grasp_error_cm|grasp_rot_err_rad|success|
    |---|---|---|---|---|---|---|---|---|---|---|---|
    |5|3201|Low|false|5.888116|8.388|1.237|0.3088|0.1188|4.389|0.2914|false|
    |5|3202|Mid|false|4.01652|8.115|1.23|0.2763|0.097|2.697|0.136|false|
    |5|3200|High|false|3.121052|12.696|0.996|0.2691|0.0865|7.203|0.1345|false|
    |5|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |5|3204|Low|false|3.198733|8.042|1.107|0.1865|0.0864|8.602|0.1475|false|
    |5|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |5|3205|Mid|false|3.927205|10.377|1.045|0.3687|0.096|5.03|0.2833|false|
    |5|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |5|3203|Mid|false|2.727329|10.367|0.811|0.4222|0.0802|6.159|0.2035|false|
    |5|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |10|3201|Low|false|8.504559|8.566|1.449|0.2866|0.1437|3.486|0.119|false|
    |10|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |10|3202|Mid|false|6.061799|7.853|1.379|0.1491|0.1205|5.333|0.0465|false|
    |10|3200|High|false|5.375383|11.377|2.114|0.1167|0.1108|3.653|0.0169|false|
    |10|3204|Low|false|4.671666|8.966|1.114|0.1723|0.1056|9.14|0.1126|false|
    |10|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |10|3205|Mid|false|5.799897|10.508|1.203|0.2202|0.1185|5.016|0.0879|false|
    |10|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |10|3203|Mid|false|4.684153|10.439|1.579|0.2419|0.1041|4.278|0.0664|false|
    |10|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |15|3201|Low|false|7.227895|8.248|1.44|0.2912|0.132|2.042|0.0848|false|
    |15|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |15|3202|Mid|false|5.35468|7.76|1.389|0.1427|0.1128|3.573|0.0307|false|
    |15|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |15|3200|High|false|4.292769|10.924|2.059|0.1148|0.098|2.698|0.0395|false|
    |15|3204|Low|false|4.386253|8.711|1.167|0.1899|0.1017|5.673|0.0525|false|
    |15|3205|Mid|false|4.736094|10.204|1.214|0.202|0.1064|3.315|0.0177|false|
    |15|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |15|3203|Mid|false|3.887111|10.113|1.657|0.2034|0.0936|2.644|0.0229|false|
    |15|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |20|3201|Low|false|6.079354|7.412|1.441|0.3028|0.1204|0.735|0.0976|false|
    |20|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |20|3202|Mid|false|4.217758|6.908|1.357|0.2029|0.0994|0.812|0.046|false|
    |20|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |20|3200|High|false|3.67774|10.913|2.029|0.1048|0.0901|3.602|0.0352|false|
    |20|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |20|3204|Low|false|2.962892|7.005|1.067|0.2655|0.0826|1.21|0.0415|false|
    |20|3205|Mid|false|3.834127|9.34|1.198|0.2315|0.0954|1.661|0.0066|false|
    |20|3203|Mid|false|3.51929|9.607|1.628|0.2315|0.0886|2.624|0.0206|false|
    |20|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |25|3201|Low|false|3.925763|6.587|1.341|0.347|0.0955|1.572|0.059|false|
    |25|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |25|3202|Mid|false|2.625241|5.648|1.227|0.3604|0.0765|0.512|0.0436|false|
    |25|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |25|3200|High|false|2.250804|10.281|1.732|0.1998|0.0693|3.163|0.0412|false|
    |25|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |25|3204|Low|false|1.889736|5.413|1.032|0.4035|0.0631|0.718|0.0519|false|
    |25|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |25|3205|Mid|false|2.579435|8.321|1.053|0.3757|0.077|0.938|0.0305|false|
    |25|3203|Mid|false|2.583417|8.847|1.427|0.3197|0.0751|1.933|0.0335|false|
    |30|3201|Low|false|3.132775|5.984|1.475|0.406|0.0826|2.463|0.0503|false|
    |30|3202|Mid|false|2.324718|4.606|1.247|0.4698|0.0706|1.3|0.0373|false|
    |30|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |30|3200|High|false|1.285695|9.972|1.221|0.3069|0.0521|4.182|0.0364|false|
    |30|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |30|3204|Low|false|1.399891|4.212|1.171|0.4994|0.0508|0.908|0.0579|false|
    |30|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |30|3205|Mid|false|1.593054|7.167|0.949|0.4805|0.0578|2.098|0.0271|false|
    |30|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |30|3203|Mid|false|1.517884|7.934|1.126|0.3534|0.0566|1.837|0.0217|false|
    |35|3201|Low|false|2.563058|5.394|1.721|0.4369|0.0713|2.208|0.0515|false|
    |35|3202|Mid|false|1.915841|3.775|1.338|0.541|0.0615|1.279|0.0187|false|
    |35|3200|High|false|0.702039|9.091|0.789|0.3747|0.0378|5.376|0.0393|true|
    |35|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |35|3204|Low|false|1.262892|3.712|1.324|0.5319|0.0458|0.639|0.0412|true|
    |35|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |35|3205|Mid|false|1.362867|5.965|1.105|0.4957|0.0511|3.418|0.0286|false|
    |35|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |35|3203|Mid|false|1.108259|7.055|0.945|0.3904|0.0479|2.195|0.018|true|
    |35|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |40|3201|Low|false|2.737405|5.207|1.835|0.4177|0.0734|2.664|0.0508|false|
    |40|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |40|3202|Mid|false|1.785229|4.203|1.376|0.469|0.0582|2.628|0.0332|false|
    |40|3200|High|false|0.760839|9.657|0.796|0.3519|0.0397|8.912|0.0355|true|
    |40|3204|Low|false|1.152496|3.218|1.345|0.4806|0.0425|2.335|0.043|true|
    |40|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |40|3205|Mid|false|1.109162|7.004|0.972|0.4388|0.0446|7.348|0.0386|true|
    |40|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |40|3203|Mid|false|1.048531|6.921|0.865|0.4321|0.0466|3.986|0.0143|true|
    |40|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |45|3201|Low|false|2.745551|5.157|1.873|0.4597|0.0732|3.04|0.051|false|
    |45|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |45|3202|Mid|false|1.594963|4.577|1.326|0.4678|0.055|6.018|0.0404|false|
    |45|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |45|3200|High|false|0.646265|10.45|0.693|0.4113|0.0357|12.221|0.0627|true|
    |45|3204|Low|false|1.060128|3.721|1.361|0.4465|0.0387|5.601|0.058|true|
    |45|3205|Mid|false|2.01254|7.926|1.531|0.4686|0.0558|9.61|0.0264|false|
    |45|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |45|3203|Mid|false|0.7803|7.598|0.758|0.4673|0.0392|7.709|0.0223|true|
    |45|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |50|3201|Low|false|3.434348|5.325|2.1|0.456|0.0809|2.955|0.0484|false|
    |50|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |50|3202|Mid|false|1.592828|4.511|1.367|0.4434|0.0554|5.601|0.0203|false|
    |50|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |50|3200|High|false|0.542755|10.324|0.749|0.4227|0.0319|13.134|0.0319|true|
    |50|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |50|3204|Low|false|1.006461|3.198|1.372|0.3817|0.0391|3.819|0.0546|true|
    |50|3205|Mid|false|3.308748|8.143|2.194|0.4607|0.0712|9.78|0.0537|false|
    |50|3203|Mid|false|0.801804|7.485|0.754|0.4574|0.0402|8.054|0.0163|true|
    |50|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |55|3201|Low|false|3.360824|5.289|2.186|0.4619|0.0796|3.54|0.05|false|
    |55|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |55|3202|Mid|false|1.963287|5.733|1.779|0.4088|0.0614|8.638|0.0341|false|
    |55|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |55|3200|High|false|0.703463|10.357|0.615|0.4202|0.0377|13.047|0.0589|true|
    |55|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |55|3204|Low|false|0.957759|2.923|1.383|0.3292|0.0371|2.437|0.0627|true|
    |55|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |55|3205|Mid|false|1.74601|8.195|1.538|0.4313|0.0507|10.79|0.0347|false|
    |55|3203|Mid|false|0.91658|7.994|0.821|0.4684|0.0437|9.636|0.0119|true|
    |60|3201|Low|false|2.429744|5.061|1.808|0.4523|0.0677|3.574|0.0513|false|
    |60|3202|Mid|false|1.636898|6.039|1.54|0.326|0.0563|8.918|0.0264|false|
    |60|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |60|3200|High|false|0.428745|9.996|0.636|0.3152|0.0287|13.463|0.0769|true|
    |60|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |60|3204|Low|false|1.089526|3.701|1.57|0.2754|0.0399|3.87|0.0565|true|
    |60|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |60|3205|Mid|false|1.179219|8.09|1.193|0.3682|0.0412|10.501|0.0474|false|
    |60|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |60|3203|Mid|false|0.62087|7.453|0.62|0.4051|0.0355|9.271|0.0097|true|
    |65|3201|Low|false|2.87651|5.102|1.997|0.4688|0.0733|3.714|0.0453|false|
    |65|3202|Mid|false|1.190193|4.727|1.341|0.2958|0.0473|4.003|0.0093|true|
    |65|3200|High|false|0.460154|10.238|0.601|0.2383|0.0306|14.854|0.0743|true|
    |65|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |65|3204|Low|false|0.844753|3.22|1.383|0.2303|0.0327|1.701|0.0471|true|
    |65|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |65|3205|Mid|false|1.099002|8.197|1.011|0.322|0.0404|10.798|0.061|true|
    |65|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |65|3203|Mid|false|0.522669|7.804|0.628|0.3555|0.0326|9.979|0.0166|true|
    |65|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |70|3201|Low|false|3.079479|5.26|2.177|0.4549|0.0754|4.066|0.0466|false|
    |70|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |70|3202|Mid|false|1.269632|4.844|1.408|0.2601|0.0489|5.635|0.0111|true|
    |70|3200|High|false|0.439734|10.305|0.561|0.2404|0.0299|14.005|0.0797|true|
    |70|3204|Low|false|0.808934|3.767|1.327|0.2195|0.0335|3.032|0.0167|true|
    |70|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |70|3205|Mid|false|1.830138|8.496|1.434|0.2869|0.0536|10.945|0.0851|false|
    |70|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |70|3203|Mid|false|0.661558|8.37|0.675|0.3423|0.0377|11.304|0.0182|true|
    |70|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |75|3201|Low|false|2.318772|5.066|1.838|0.4994|0.0668|4.053|0.0459|false|
    |75|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |75|3202|Mid|false|1.112696|4.595|1.3|0.2574|0.0464|4.418|0.0106|true|
    |75|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |75|3200|High|false|0.470148|10.077|0.494|0.2661|0.0312|13.098|0.0472|true|
    |75|3204|Low|false|0.529926|3.473|1.106|0.2086|0.0256|2.497|0.0225|true|
    |75|3205|Mid|false|2.116383|8.63|1.661|0.2946|0.0575|11.623|0.0846|false|
    |75|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |75|3203|Mid|false|0.586677|8.085|0.594|0.3439|0.0353|10.388|0.0318|true|
    |75|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |80|3201|Low|false|2.503653|5.095|1.974|0.4559|0.0679|3.413|0.0305|false|
    |80|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |80|3202|Mid|false|0.870799|4.394|1.115|0.2337|0.0398|3.748|0.0175|true|
    |80|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |80|3200|High|false|0.485554|10.044|0.484|0.2401|0.0323|12.61|0.0475|true|
    |80|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |80|3204|Low|false|0.665734|4.111|1.213|0.2214|0.0313|3.562|0.0208|true|
    |80|3205|Mid|false|1.415851|8.906|1.311|0.2902|0.0451|12.515|0.1108|false|
    |80|3203|Mid|false|0.65561|7.714|0.664|0.3245|0.0375|8.504|0.0226|true|
    |80|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |85|3201|Low|false|2.574525|5.097|1.995|0.4632|0.0694|3.973|0.038|false|
    |85|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |85|3202|Mid|false|1.02564|4.05|1.216|0.2291|0.0441|3.535|0.0231|true|
    |85|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |85|3200|High|false|0.175033|9.715|0.479|0.2234|0.0181|11.027|0.0427|true|
    |85|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |85|3204|Low|false|0.67429|3.052|1.204|0.2108|0.0287|1.862|0.0156|true|
    |85|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |85|3205|Mid|false|0.827639|8.687|0.922|0.2706|0.0338|12.674|0.0928|true|
    |85|3203|Mid|false|1.003712|8.107|0.928|0.3256|0.0474|9.312|0.017|true|
    |90|3201|Low|false|2.021518|5.339|1.729|0.4775|0.0593|4.563|0.0378|false|
    |90|3202|Mid|false|0.910523|3.99|1.163|0.2072|0.0404|3.493|0.0235|true|
    |90|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |90|3200|High|false|0.29146|9.399|0.36|0.2046|0.0254|10.089|0.0246|true|
    |90|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |90|3204|Low|false|0.738064|3.413|1.321|0.1881|0.0313|2.332|0.013|true|
    |90|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |90|3205|Mid|false|1.336205|8.855|1.429|0.2583|0.045|13.002|0.0834|true|
    |90|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |90|3203|Mid|false|0.721372|7.729|0.712|0.2982|0.04|7.825|0.0141|true|
    |95|3201|Low|false|2.096055|5.122|1.768|0.4665|0.062|3.815|0.0398|false|
    |95|3202|Mid|false|1.00314|3.586|1.206|0.2052|0.0434|2.619|0.0232|true|
    |95|3200|High|false|0.250708|8.638|0.348|0.1752|0.023|7.569|0.0125|true|
    |95|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |95|3204|Low|false|0.706338|3.068|1.305|0.1881|0.0297|1.715|0.0052|true|
    |95|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |95|3205|Mid|false|0.490594|7.944|0.711|0.2706|0.0281|9.61|0.0814|true|
    |95|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |95|3203|Mid|false|0.685178|7.525|0.672|0.2943|0.039|6.54|0.0126|true|
    |95|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |100|3201|Low|false|2.113701|5.258|1.862|0.4186|0.0611|3.894|0.0352|false|
    |100|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |100|3202|Mid|false|0.901413|3.392|1.183|0.1958|0.0412|2.745|0.0095|true|
    |100|3200|High|false|0.272681|8.31|0.381|0.2024|0.0245|7.058|0.0242|true|
    |100|3204|Low|false|0.722278|3.371|1.325|0.1675|0.0303|3.017|0.0142|true|
    |100|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |100|3205|Mid|false|0.663168|8.054|0.949|0.2573|0.0335|9.787|0.0745|true|
    |100|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |100|3203|Mid|false|0.797501|7.496|0.827|0.2755|0.0422|6.583|0.0102|true|
    |100|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |105|3201|Low|false|1.956509|5.35|1.891|0.3896|0.0582|3.891|0.0365|false|
    |105|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |105|3202|Mid|false|0.707586|3.577|1.058|0.1812|0.0352|2.787|0.0122|true|
    |105|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |105|3200|High|false|0.201729|8.139|0.435|0.1963|0.0205|6.246|0.0121|true|
    |105|3204|Low|false|0.484384|3.041|1.106|0.1817|0.0232|2.21|0.0146|true|
    |105|3205|Mid|false|0.792854|8.045|1.06|0.2556|0.0378|9.157|0.0568|true|
    |105|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |105|3203|Mid|false|0.611926|7.149|0.635|0.2778|0.0369|5.682|0.017|true|
    |105|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |110|3201|Low|false|2.285872|5.423|2.001|0.3797|0.0628|3.923|0.0369|false|
    |110|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |110|3202|Mid|false|0.599518|3.509|0.906|0.1908|0.0326|2.586|0.0101|true|
    |110|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |110|3200|High|false|0.172268|8.613|0.388|0.1923|0.0188|6.926|0.0157|true|
    |110|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |110|3204|Low|false|0.727194|3.168|1.307|0.18|0.0302|2.301|0.0084|true|
    |110|3205|Mid|false|0.505865|7.89|0.773|0.254|0.0282|8.607|0.0492|true|
    |110|3203|Mid|false|0.733798|7.076|0.697|0.2927|0.0404|4.937|0.0091|true|
    |110|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |115|3201|Low|false|1.964274|5.496|1.862|0.3908|0.0584|3.481|0.0274|false|
    |115|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |115|3202|Mid|false|0.959773|3.428|1.227|0.1978|0.0418|2.431|0.0121|true|
    |115|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |115|3200|High|false|0.133475|8.094|0.413|0.1843|0.0162|6.193|0.0098|true|
    |115|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |115|3204|Low|false|0.544586|3.332|1.165|0.1821|0.0247|3.248|0.0193|true|
    |115|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |115|3205|Mid|false|0.833043|7.668|1.101|0.2536|0.0372|7.322|0.044|true|
    |115|3203|Mid|false|0.3202|7.092|0.358|0.2812|0.0255|5.067|0.0086|true|
    |120|3201|Low|false|1.823673|5.483|1.782|0.3851|0.0561|3.145|0.0281|false|
    |120|3202|Mid|false|0.955184|3.274|1.32|0.2022|0.0405|2.203|0.0144|true|
    |120|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |120|3200|High|false|0.222339|8.685|0.332|0.1941|0.0214|7.008|0.0127|true|
    |120|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |120|3204|Low|false|0.537269|3.05|1.135|0.1905|0.0249|1.839|0.009|true|
    |120|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |120|3205|Mid|false|0.45378|7.847|0.782|0.2474|0.0258|7.796|0.0423|true|
    |120|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |120|3203|Mid|false|0.313255|7.402|0.402|0.279|0.0251|5.158|0.0044|true|
    |125|3201|Low|false|1.880956|5.447|1.917|0.3525|0.0566|3.038|0.0267|false|
    |125|3202|Mid|false|1.110638|3.806|1.336|0.2262|0.0441|2.213|0.0141|true|
    |125|3200|High|false|0.089053|8.358|0.331|0.1789|0.0128|7.011|0.0099|true|
    |125|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |125|3204|Low|false|0.520526|3.09|1.133|0.1892|0.024|2.256|0.0189|true|
    |125|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |125|3205|Mid|false|0.446699|7.522|0.719|0.2555|0.0263|6.717|0.0396|true|
    |125|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |125|3203|Mid|false|0.435172|7.314|0.505|0.2788|0.0304|5.681|0.0039|true|
    |125|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |130|3201|Low|false|1.669139|5.755|1.634|0.4332|0.0524|2.984|0.0225|false|
    |130|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |130|3202|Mid|false|0.743711|3.598|1.078|0.212|0.0376|2.542|0.0154|true|
    |130|3200|High|false|0.190952|7.292|0.384|0.2181|0.0196|3.533|0.0106|true|
    |130|3204|Low|false|0.807545|3.168|1.37|0.2021|0.0317|2.122|0.0111|true|
    |130|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |130|3205|Mid|false|0.693901|7.583|0.928|0.2598|0.0337|6.182|0.0429|true|
    |130|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |130|3203|Mid|false|0.321186|7.631|0.423|0.299|0.0252|5.754|0.0095|true|
    |130|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |135|3201|Low|false|1.694019|5.663|1.716|0.4055|0.0533|2.927|0.0272|false|
    |135|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |135|3202|Mid|false|0.846904|3.398|1.208|0.2208|0.0375|2.1|0.0154|true|
    |135|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |135|3200|High|false|0.086156|7.958|0.397|0.1914|0.012|4.767|0.0115|true|
    |135|3204|Low|false|0.575281|3.302|1.173|0.1972|0.0265|2.058|0.0319|true|
    |135|3205|Mid|false|0.526753|7.502|0.701|0.2873|0.0291|6.011|0.0378|true|
    |135|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |135|3203|Mid|false|0.48164|7.451|0.498|0.3005|0.0323|5.551|0.0045|true|
    |135|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |140|3201|Low|false|1.716609|5.689|1.729|0.401|0.0547|3.029|0.0308|false|
    |140|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |140|3202|Mid|false|0.855131|3.573|1.174|0.2309|0.039|2.643|0.0115|true|
    |140|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |140|3200|High|false|0.130416|7.609|0.39|0.2175|0.0152|4.07|0.0104|true|
    |140|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |140|3204|Low|false|0.60694|3.128|1.211|0.1916|0.0263|1.813|0.0169|true|
    |140|3205|Mid|false|0.365072|7.229|0.596|0.2925|0.0247|4.978|0.0421|true|
    |140|3203|Mid|false|0.600043|7.509|0.655|0.3013|0.0363|5.76|0.0109|true|
    |140|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |145|3201|Low|false|1.640585|5.87|1.714|0.3945|0.0529|3.001|0.0365|false|
    |145|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |145|3202|Mid|false|0.723059|3.368|1.063|0.2579|0.0349|1.792|0.0051|true|
    |145|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |145|3200|High|false|0.104062|7.569|0.469|0.1981|0.0122|4.168|0.0084|true|
    |145|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |145|3204|Low|false|0.370535|3.208|0.918|0.1808|0.0192|1.609|0.0132|true|
    |145|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |145|3205|Mid|false|0.435209|7.567|0.668|0.2733|0.0254|6.17|0.0373|true|
    |145|3203|Mid|false|0.614176|7.584|0.769|0.29|0.0366|6.05|0.004|true|
    |150|3201|Low|false|1.473508|5.945|1.638|0.4026|0.0497|2.883|0.0251|true|
    |150|3202|Mid|false|0.869391|3.65|1.31|0.2614|0.0387|2.354|0.0106|true|
    |150|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |150|3200|High|false|0.15867|7.235|0.36|0.239|0.0171|2.311|0.0097|true|
    |150|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |150|3204|Low|false|0.304786|3.113|0.799|0.1934|0.0166|1.471|0.0135|true|
    |150|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |150|3205|Mid|false|0.870967|7.707|1.068|0.2769|0.0391|6.184|0.0429|true|
    |150|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |150|3203|Mid|false|0.693707|7.314|0.808|0.3024|0.0391|4.367|0.0045|true|
    |155|3201|Low|false|1.393972|6.045|1.609|0.3334|0.0472|2.837|0.0242|true|
    |155|3202|Mid|false|0.616101|3.726|1.064|0.2161|0.0327|1.523|0.0102|true|
    |155|3200|High|false|0.118961|7.65|0.328|0.2426|0.0146|3.483|0.0049|true|
    |155|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |155|3204|Low|false|0.433137|3.465|1.091|0.1946|0.0213|1.92|0.0213|true|
    |155|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |155|3205|Mid|false|0.41043|7.594|0.596|0.2878|0.0251|6.218|0.0487|true|
    |155|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |155|3203|Mid|false|0.302923|7.303|0.428|0.2728|0.0247|4.099|0.0113|true|
    |155|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |160|3201|Low|false|1.42444|6.107|1.599|0.368|0.0478|2.832|0.0279|true|
    |160|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |160|3202|Mid|false|0.416131|4.054|0.784|0.2386|0.025|2.212|0.0163|true|
    |160|3200|High|false|0.115385|7.232|0.498|0.2492|0.0131|2.077|0.0078|true|
    |160|3204|Low|false|0.402023|3.342|0.954|0.1915|0.0199|1.814|0.0221|true|
    |160|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |160|3205|Mid|false|0.620619|7.653|0.748|0.3208|0.0341|6.301|0.055|true|
    |160|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |160|3203|Mid|false|0.294523|7.26|0.425|0.282|0.024|4.233|0.0092|true|
    |160|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |165|3201|Low|false|1.329973|5.991|1.562|0.3553|0.047|3.085|0.0318|true|
    |165|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |165|3202|Mid|false|0.861853|4.04|1.277|0.2597|0.0387|1.975|0.0165|true|
    |165|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |165|3200|High|false|0.093985|7.457|0.318|0.2256|0.0125|2.331|0.0068|true|
    |165|3204|Low|false|0.306765|3.021|0.812|0.19|0.0166|1.216|0.0135|true|
    |165|3205|Mid|false|0.279501|7.562|0.454|0.295|0.0198|5.239|0.0439|true|
    |165|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |165|3203|Mid|false|0.740125|7.238|0.801|0.3143|0.04|3.483|0.0062|true|
    |165|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |170|3201|Low|false|1.39976|6.056|1.578|0.3362|0.0481|3.215|0.0362|true|
    |170|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |170|3202|Mid|false|0.546158|3.59|0.902|0.2677|0.0309|1.372|0.0118|true|
    |170|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |170|3200|High|false|0.103596|7.915|0.283|0.2354|0.0134|3.159|0.0099|true|
    |170|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |170|3204|Low|false|0.645716|3.496|1.254|0.2278|0.0272|2.05|0.012|true|
    |170|3205|Mid|false|0.948925|7.668|1.005|0.3214|0.0425|4.753|0.0372|true|
    |170|3203|Mid|false|0.305404|7.344|0.524|0.2716|0.0247|4.151|0.008|true|
    |170|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |175|3201|Low|false|1.151078|6.087|1.556|0.2928|0.0423|3.028|0.03|true|
    |175|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |175|3202|Mid|false|0.590624|4.142|0.933|0.2584|0.0321|2.508|0.0198|true|
    |175|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |175|3200|High|false|0.031953|7.719|0.506|0.2121|0.0014|2.98|0.0157|true|
    |175|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |175|3204|Low|false|0.2105|3.256|0.629|0.1805|0.0126|1.302|0.0106|true|
    |175|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |175|3205|Mid|false|0.282462|7.353|0.444|0.2812|0.0213|4.489|0.0446|true|
    |175|3203|Mid|false|0.43445|7.528|0.55|0.2984|0.03|4.419|0.0083|true|
    |180|3201|Low|false|1.117948|6.111|1.558|0.2979|0.0416|3.016|0.0273|true|
    |180|3202|Mid|false|0.568719|3.897|0.887|0.2649|0.0312|1.982|0.0236|true|
    |180|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |180|3200|High|false|0.066006|7.855|0.479|0.2397|0.008|3.347|0.0119|true|
    |180|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |180|3204|Low|false|0.32421|3.39|0.915|0.2068|0.0176|1.487|0.0234|true|
    |180|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |180|3205|Mid|false|0.214718|7.736|0.44|0.2644|0.016|5.274|0.0437|true|
    |180|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |180|3203|Mid|false|0.272141|7.28|0.527|0.2581|0.0233|3.744|0.0115|true|
    |185|3201|Low|false|1.16127|6.085|1.507|0.2684|0.0439|3.383|0.0326|true|
    |185|3202|Mid|false|0.209884|3.899|0.505|0.2545|0.0157|2.286|0.0136|true|
    |185|3200|High|false|0.071276|7.917|0.54|0.2706|0.0075|3.101|0.0064|true|
    |185|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |185|3204|Low|false|0.353212|3.068|0.897|0.2195|0.0185|1.004|0.0216|true|
    |185|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |185|3205|Mid|false|0.218391|7.223|0.378|0.2965|0.0164|3.996|0.0428|true|
    |185|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |185|3203|Mid|false|0.481841|7.49|0.761|0.2889|0.0318|3.441|0.0146|true|
    |185|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |190|3201|Low|false|0.686056|5.866|1.375|0.2338|0.0317|3.727|0.0505|true|
    |190|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |190|3202|Mid|false|0.244723|3.873|0.481|0.3021|0.0199|1.46|0.0118|true|
    |190|3200|High|false|0.042826|7.594|0.433|0.2604|0.0026|1.298|0.0047|true|
    |190|3204|Low|false|0.202019|3.015|0.699|0.2048|0.0126|0.8|0.02|true|
    |190|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |190|3205|Mid|false|0.241324|7.534|0.425|0.3092|0.016|3.908|0.0165|true|
    |190|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |190|3203|Mid|false|0.132397|7.191|0.292|0.2862|0.0138|1.765|0.0111|true|
    |190|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |195|3201|Low|false|0.538595|5.841|1.311|0.2261|0.0276|3.272|0.0535|true|
    |195|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |195|3202|Mid|false|0.17423|3.809|0.69|0.2474|0.0131|1.053|0.0136|true|
    |195|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |195|3200|High|false|0.079977|7.947|0.345|0.2769|0.0098|1.984|0.0055|true|
    |195|3204|Low|false|0.255193|3.171|0.903|0.1917|0.0134|0.855|0.0199|true|
    |195|3205|Mid|false|0.12149|7.458|0.38|0.2937|0.0053|3.472|0.0183|true|
    |195|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |195|3203|Mid|false|0.254825|7.2|0.602|0.278|0.0219|2.304|0.0061|true|
    |195|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |200|3201|Low|false|0.48947|5.844|1.244|0.2577|0.0267|2.972|0.0464|true|
    |200|3201|Low|true|0.108762|7.302|0.0|0.3184|0.0|0.0|0.0|true|
    |200|3202|Mid|false|0.167134|3.737|0.686|0.2696|0.0143|1.094|0.0175|true|
    |200|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |200|3200|High|false|0.055216|7.75|0.44|0.245|0.0067|1.857|0.0137|true|
    |200|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |200|3204|Low|false|0.187468|2.934|0.739|0.1738|0.0109|0.825|0.0178|true|
    |200|3205|Mid|false|0.232539|7.225|0.554|0.3053|0.0169|3.051|0.0253|true|
    |200|3203|Mid|false|0.101943|6.949|0.15|0.27|0.0116|2.202|0.0078|true|
    |200|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |205|3201|Low|false|0.512688|5.976|1.21|0.309|0.0266|3.032|0.0443|true|
    |205|3202|Mid|false|0.113813|3.765|0.532|0.2899|0.0099|1.542|0.0171|true|
    |205|3202|Mid|true|0.082396|5.428|0.0|0.2099|0.0|0.0|0.0|true|
    |205|3200|High|false|0.08738|7.807|0.685|0.2681|0.0089|1.958|0.0212|true|
    |205|3200|High|true|0.018876|8.792|0.0|0.1454|0.0|0.0|0.0|true|
    |205|3204|Low|false|0.162155|2.763|0.689|0.1896|0.0105|1.298|0.0161|true|
    |205|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |205|3205|Mid|false|0.156329|7.3|0.396|0.3157|0.0102|3.217|0.0276|true|
    |205|3203|Mid|false|0.121474|7.065|0.184|0.2922|0.0124|2.292|0.0087|true|
    |210|3201|Low|false|0.50073|6.02|1.191|0.2773|0.0261|3.172|0.0468|true|
    |210|3202|Mid|false|0.111315|3.78|0.551|0.2707|0.0089|1.199|0.0211|true|
    |210|3200|High|false|0.077265|7.828|0.625|0.2439|0.0083|1.6|0.0179|true|
    |210|3204|Low|false|0.17662|2.837|0.696|0.1816|0.0117|1.198|0.0176|true|
    |210|3204|Low|true|0.036414|4.852|0.0|0.1444|0.0|0.0|0.0007|true|
    |210|3205|Mid|false|0.131078|7.396|0.402|0.3054|0.006|3.369|0.0287|true|
    |210|3205|Mid|true|0.105459|8.848|0.0|0.3072|0.0005|0.0|0.0|true|
    |210|3203|Mid|false|0.098171|7.33|0.143|0.287|0.0107|2.767|0.0137|true|
    |215|3201|Low|false|0.497096|6.098|1.213|0.2754|0.0254|3.121|0.0466|true|
    |215|3202|Mid|false|0.090667|3.872|0.431|0.2898|0.0072|1.15|0.017|true|
    |215|3200|High|false|0.065994|8.212|0.608|0.267|0.006|1.914|0.0154|true|
    |215|3204|Low|false|0.18745|2.804|0.739|0.1824|0.0126|1.265|0.0169|true|
    |215|3205|Mid|false|0.138487|7.552|0.436|0.2978|0.0082|3.45|0.0299|true|
    |215|3203|Mid|false|0.07117|7.341|0.118|0.2866|0.0069|2.208|0.0115|true|
    |215|3203|Mid|true|0.059666|9.317|0.0|0.2138|0.0|0.0|0.0008|true|
    |220|3201|Low|false|0.455359|6.163|1.216|0.2755|0.0237|3.229|0.0466|true|
    |220|3202|Mid|false|0.110414|4.284|0.519|0.2626|0.0087|0.896|0.0173|true|
    |220|3200|High|false|0.079066|8.389|0.632|0.2531|0.0073|1.556|0.0158|true|
    |220|3204|Low|false|0.158569|2.969|0.656|0.1845|0.0119|0.746|0.0156|true|
    |220|3205|Mid|false|0.115097|8.02|0.41|0.2998|0.0046|4.409|0.0364|true|
    |220|3203|Mid|false|0.079634|7.431|0.118|0.2812|0.0087|2.002|0.0166|true|
    |225|3201|Low|false|0.484817|6.246|1.213|0.2656|0.0246|3.228|0.049|true|
    |225|3202|Mid|false|0.109329|4.427|0.484|0.2726|0.0086|0.761|0.0166|true|
    |225|3200|High|false|0.095285|8.629|0.769|0.2494|0.0092|1.527|0.0087|true|
    |225|3204|Low|false|0.163705|3.014|0.666|0.1838|0.0113|0.74|0.0151|true|
    |225|3205|Mid|false|0.10745|8.213|0.379|0.3025|0.0034|4.745|0.0364|true|
    |225|3203|Mid|false|0.065224|7.723|0.112|0.2769|0.0063|2.395|0.0111|true|
    |230|3201|Low|false|0.510855|6.315|1.247|0.281|0.0258|3.303|0.0481|true|
    |230|3202|Mid|false|0.092375|4.269|0.441|0.2744|0.0072|0.852|0.0144|true|
    |230|3200|High|false|0.074608|8.658|0.693|0.248|0.0069|1.842|0.0134|true|
    |230|3204|Low|false|0.164408|3.193|0.695|0.1873|0.0113|0.716|0.0126|true|
    |230|3205|Mid|false|0.11163|8.225|0.349|0.2921|0.0069|4.641|0.0385|true|
    |230|3203|Mid|false|0.079241|7.831|0.173|0.282|0.0069|2.603|0.011|true|
    |235|3201|Low|false|0.50056|6.361|1.25|0.2702|0.0248|3.297|0.0498|true|
    |235|3202|Mid|false|0.087526|4.682|0.434|0.2799|0.0067|0.482|0.0186|true|
    |235|3200|High|false|0.081283|8.967|0.697|0.2476|0.0072|1.972|0.0128|true|
    |235|3204|Low|false|0.150373|3.135|0.658|0.1876|0.0108|0.383|0.0148|true|
    |235|3205|Mid|false|0.10849|8.459|0.321|0.3145|0.0039|5.123|0.0426|true|
    |235|3203|Mid|false|0.123223|7.981|0.271|0.2824|0.012|2.658|0.012|true|
    |240|3201|Low|false|0.481718|6.347|1.242|0.2743|0.0242|3.159|0.0518|true|
    |240|3202|Mid|false|0.115094|4.787|0.524|0.2703|0.0094|0.626|0.0149|true|
    |240|3200|High|false|0.065908|9.311|0.6|0.2389|0.0046|2.579|0.0132|true|
    |240|3204|Low|false|0.153745|3.142|0.627|0.1838|0.0124|0.492|0.0145|true|
    |240|3205|Mid|false|0.099074|8.453|0.313|0.2969|0.0046|5.117|0.0429|true|
    |240|3203|Mid|false|0.077709|7.968|0.157|0.2711|0.0076|2.685|0.0128|true|
  spec_verification: {grasp_success_rate_ge_80pct:true,placement_accuracy_lt_3cm:true,rotational_alignment_lt_0_05rad:true,terminal_velocity_lt_1_5rad_s:true}
```

### `results/spwm_v0_12_4_train_m1/diagnostics.json`
> ALIF firing-rate histogram, silent/hyperactive fractions, and gradient diagnostics

```yaml
model_name: spwm_v0_12_4_train_m1
version: spwm_v0_12_4
mode: train_m1
spike_statistics:
  mean_spike_rate: 0.1684163361787796
  median_spike_rate: 0.16354167461395264
  min_spike_rate: 0.0
  max_spike_rate: 0.59375
  fraction_silent_neurons: 0.19140625
  fraction_hyperactive_neurons: 0.0234375
  histogram_counts: [93,61,63,19,14,6,0,0,0,0]
  histogram_bin_edges: [0.0,0.10000000149011612,0.20000000298023224,0.30000001192092896,0.4000000059604645,0.5,0.6000000238418579,0.699999988079071,0.800000011920929,0.8999999761581421,1.0]
gradient_diagnostics: {total_norm:0.7999999955500426,has_nan:false,has_inf:false,zero_gradient_ratio:0.2740574151195815}
```

### `results/ssot_eval_metrics.json`
> N=400 open-loop vs. phasic handover curves and analytical energy estimates

```yaml
n_test: 400
horizon: 100
mean_spike_rate: 0.12268746128449073
open_overall_mean: 0.11145036667585373
open_h100_mean: 0.11163192987442017
handover_overall_mean: 0.14822937548160553
open_curve_mean: [0.029773293063044548,0.050943732261657715,0.0694747120141983,0.08839104324579239,0.11091842502355576,0.13780221343040466,0.16399258375167847,0.1815030425786972,0.18704365193843842,0.1862158328294754,0.18566682934761047,0.18405292928218842,0.17932210862636566,0.17349916696548462,0.1672763228416443,0.1608705073595047,0.15358270704746246,0.14490999281406403,0.1363014280796051,0.12909653782844543,0.12362727522850037,0.11972808837890625,0.11753828823566437,0.1165211871266365,0.1159699335694313,0.11564317345619202,0.11539418995380402,0.11538257449865341,0.11532190442085266,0.11522750556468964,0.11492013931274414,0.11434490978717804,0.11361837387084961,0.11278533190488815,0.11203490197658539,0.11129018664360046,0.110580675303936,0.11005494743585587,0.10947441309690475,0.10889815539121628,0.10819628089666367,0.10736408829689026,0.10660341382026672,0.10608332604169846,0.10566526651382446,0.10527226328849792,0.1049216017127037,0.10457690060138702,0.10416756570339203,0.1037709042429924,0.10346436500549316,0.10337063670158386,0.10360246896743774,0.10405252128839493,0.10464460402727127,0.1052316352725029,0.10584589093923569,0.10640781372785568,0.10695965588092804,0.10726359486579895,0.10728471726179123,0.10700049251317978,0.10646244883537292,0.10588253289461136,0.10546524077653885,0.10542401671409607,0.10532966256141663,0.10512605309486389,0.104812853038311,0.10426167398691177,0.10346218943595886,0.10215451568365097,0.10054737329483032,0.09872634708881378,0.09687754511833191,0.09513633698225021,0.09349039196968079,0.09200796484947205,0.09077796339988708,0.08987804502248764,0.08930744230747223,0.08895017951726913,0.08870480209589005,0.08853450417518616,0.08850858360528946,0.08873715996742249,0.08926960825920105,0.09014730155467987,0.09130968153476715,0.09264545142650604,0.09404744952917099,0.0954543873667717,0.09681116789579391,0.0981278121471405,0.09945985674858093,0.10092268884181976,0.10273744910955429,0.10508394986391068,0.10807770490646362,0.11163196712732315]
open_curve_q25: [0.02418188378214836,0.04567704722285271,0.06356161087751389,0.08150790631771088,0.10323502123355865,0.1262536346912384,0.14727847278118134,0.1584264487028122,0.15536509454250336,0.14956873655319214,0.1456349790096283,0.14184843003749847,0.1292610913515091,0.11995430290699005,0.11374010890722275,0.10681220144033432,0.09909381717443466,0.09388910979032516,0.09162658452987671,0.09065213799476624,0.08755099773406982,0.08756190538406372,0.08479989320039749,0.08799444139003754,0.09005386382341385,0.09020231664180756,0.09062648564577103,0.08981894701719284,0.09002521634101868,0.0868510901927948,0.08429233729839325,0.0836835652589798,0.0824483335018158,0.08040601015090942,0.08030581474304199,0.07910536229610443,0.07994438707828522,0.07748156785964966,0.07809378951787949,0.07631292939186096,0.07542622089385986,0.07479224354028702,0.07388124614953995,0.07398474216461182,0.07215038686990738,0.07094414532184601,0.07133544981479645,0.07061903178691864,0.06940420717000961,0.07008630037307739,0.07016012072563171,0.06934629380702972,0.0711706280708313,0.0707891657948494,0.07164794206619263,0.07330648601055145,0.0729539692401886,0.07322145253419876,0.07386495172977448,0.07504188269376755,0.07543125748634338,0.07677724957466125,0.07613738626241684,0.07753583788871765,0.07773175090551376,0.07676748931407928,0.07657727599143982,0.07582847028970718,0.07529428601264954,0.07495700567960739,0.07451340556144714,0.07442523539066315,0.07374760508537292,0.0731559544801712,0.0720655620098114,0.07118524610996246,0.06973353028297424,0.06915608048439026,0.06889288127422333,0.06734035909175873,0.06694740056991577,0.06701298803091049,0.06639609485864639,0.0661369189620018,0.06590918451547623,0.06688971817493439,0.06725241243839264,0.06829902529716492,0.06985847651958466,0.07033969461917877,0.07075491547584534,0.07271713018417358,0.07260537892580032,0.07261520624160767,0.07316749542951584,0.07275650650262833,0.07283641397953033,0.07333727180957794,0.07595354318618774,0.07957030832767487]
open_curve_q75: [0.034978218376636505,0.056788939982652664,0.07592301815748215,0.09600119292736053,0.11982263624668121,0.148398295044899,0.17623081803321838,0.1945151388645172,0.1995457112789154,0.1977941244840622,0.19582737982273102,0.19316044449806213,0.18706178665161133,0.18558591604232788,0.19795067608356476,0.2016545534133911,0.1934921145439148,0.17840851843357086,0.16444751620292664,0.15170827507972717,0.14338824152946472,0.1383947879076004,0.13898776471614838,0.1367250382900238,0.13474488258361816,0.13561207056045532,0.13406044244766235,0.13343654572963715,0.13474959135055542,0.13010606169700623,0.13032756745815277,0.13053463399410248,0.13128319382667542,0.1298547089099884,0.12724623084068298,0.12781772017478943,0.1257426142692566,0.12586036324501038,0.1256319135427475,0.12389302253723145,0.12436303496360779,0.12179112434387207,0.12037302553653717,0.11990264058113098,0.11713168025016785,0.11632630228996277,0.1137358546257019,0.11422622203826904,0.10990928113460541,0.11078011989593506,0.10951422154903412,0.11053813993930817,0.11167839169502258,0.11137203127145767,0.1111529991030693,0.11056137830018997,0.11067964881658554,0.11162015050649643,0.11224442720413208,0.11193743348121643,0.10881629586219788,0.10633069276809692,0.10497108101844788,0.1049499660730362,0.10366843640804291,0.10430885851383209,0.10363071411848068,0.10382680594921112,0.1041797548532486,0.10438516736030579,0.10449538379907608,0.10305618494749069,0.10159837454557419,0.10002902895212173,0.0980919897556305,0.09744185954332352,0.09610909968614578,0.09600912034511566,0.09620500355958939,0.09556256979703903,0.0965682864189148,0.09527552872896194,0.0955197885632515,0.09612056612968445,0.09667248278856277,0.09796690940856934,0.09943855553865433,0.10062514990568161,0.10258600115776062,0.10414493083953857,0.10640185326337814,0.10827697813510895,0.11016464978456497,0.11279815435409546,0.11581233143806458,0.1171855628490448,0.12172946333885193,0.1247154027223587,0.12768083810806274,0.13074742257595062]
handover_curve_mean: [0.029773293063044548,0.050943732261657715,0.0694747120141983,0.08839104324579239,0.11091842502355576,0.13780221343040466,0.16399258375167847,0.1815030425786972,0.18711601197719574,0.1865493357181549,0.18650802969932556,0.1853022426366806,0.1807335913181305,0.1740194708108902,0.16652943193912506,0.15872937440872192,0.15016847848892212,0.14133521914482117,0.13379758596420288,0.12783282995224,0.12372341752052307,0.12156546860933304,0.1216108426451683,0.12290579825639725,0.12420029938220978,0.12491463869810104,0.12475544959306717,0.12454214692115784,0.12493330240249634,0.12585550546646118,0.12717634439468384,0.1287602186203003,0.1304422914981842,0.1318274438381195,0.13327164947986603,0.13474585115909576,0.13588467240333557,0.1372513622045517,0.13852530717849731,0.14000536501407623,0.14124196767807007,0.14216503500938416,0.14288842678070068,0.14370214939117432,0.1445225328207016,0.14544229209423065,0.14647279679775238,0.14748235046863556,0.1485184133052826,0.14960844814777374,0.15068279206752777,0.1518946886062622,0.15314893424510956,0.15417881309986115,0.15508046746253967,0.15590828657150269,0.1568310409784317,0.15773236751556396,0.15838617086410522,0.15874864161014557,0.15898774564266205,0.15903113782405853,0.1589553952217102,0.15910688042640686,0.15973971784114838,0.16099222004413605,0.16246603429317474,0.164189413189888,0.166151225566864,0.16811607778072357,0.16987861692905426,0.17103104293346405,0.17175953090190887,0.17214207351207733,0.17240771651268005,0.17284435033798218,0.17391212284564972,0.17599841952323914,0.17907680571079254,0.18273352086544037,0.1860274076461792,0.18805965781211853,0.18833787739276886,0.18662995100021362,0.1829143911600113,0.1774335354566574,0.17054075002670288,0.16272969543933868,0.15462777018547058,0.14686095714569092,0.13995322585105896,0.1343172788619995,0.13019315898418427,0.12766927480697632,0.1267102211713791,0.1272147297859192,0.12912783026695251,0.13235068321228027,0.13673529028892517,0.14202861487865448]
handover_curve_q25: [0.02418188378214836,0.04567704722285271,0.06356161087751389,0.08150790631771088,0.10323502123355865,0.1262536346912384,0.14727847278118134,0.1584264487028122,0.15536509454250336,0.14956873655319214,0.1456349790096283,0.14184843003749847,0.1292610913515091,0.11995430290699005,0.11374010890722275,0.10681220144033432,0.09909381717443466,0.09388910979032516,0.09162658452987671,0.0889335423707962,0.08685201406478882,0.08715690672397614,0.08743292093276978,0.09089215844869614,0.092357337474823,0.09309165179729462,0.09434390068054199,0.09267505258321762,0.09158483892679214,0.09401020407676697,0.09473395347595215,0.09599326550960541,0.09823679178953171,0.09881702065467834,0.09927522391080856,0.10124707967042923,0.10002836585044861,0.1001812145113945,0.10120449215173721,0.10226130485534668,0.10199446976184845,0.10150440782308578,0.10265687108039856,0.10444115847349167,0.10426055639982224,0.10424090921878815,0.10685345530509949,0.1066870167851448,0.1047300174832344,0.10603687167167664,0.10726380348205566,0.10767601430416107,0.10972559452056885,0.10918937623500824,0.10844255983829498,0.10804241895675659,0.1093723475933075,0.11152873933315277,0.11248204857110977,0.11146142333745956,0.10930593311786652,0.1112322136759758,0.11333563923835754,0.11527179926633835,0.11717584729194641,0.11977388709783554,0.1219034343957901,0.12375897914171219,0.12594640254974365,0.12658101320266724,0.12917889654636383,0.12999826669692993,0.1301594078540802,0.13084694743156433,0.12985849380493164,0.1280229091644287,0.12634795904159546,0.1252342164516449,0.12456217408180237,0.12294067442417145,0.11967344582080841,0.11572802066802979,0.11224879324436188,0.10945623368024826,0.10433883219957352,0.10201263427734375,0.10024400055408478,0.09812115877866745,0.09517122805118561,0.09254132211208344,0.08827183395624161,0.08498714864253998,0.08442756533622742,0.08073912560939789,0.08235818147659302,0.08320027589797974,0.08642669022083282,0.09044907987117767,0.09397381544113159,0.09960572421550751]
handover_curve_q75: [0.034978218376636505,0.056788939982652664,0.07592301815748215,0.09600119292736053,0.11982263624668121,0.148398295044899,0.17623081803321838,0.1945151388645172,0.1995457112789154,0.1977941244840622,0.19582737982273102,0.19316044449806213,0.18706178665161133,0.18558591604232788,0.19551125168800354,0.1977681666612625,0.17980054020881653,0.1675701141357422,0.16123896837234497,0.15432441234588623,0.14869529008865356,0.14888237416744232,0.14957216382026672,0.14846652746200562,0.14913947880268097,0.14683443307876587,0.1424354612827301,0.14332982897758484,0.1452476680278778,0.1468905508518219,0.14901050925254822,0.15288755297660828,0.15561281144618988,0.1587507426738739,0.16121651232242584,0.16120567917823792,0.16121523082256317,0.16311495006084442,0.1632762998342514,0.16982655227184296,0.17117498815059662,0.1693042665719986,0.16789492964744568,0.16850243508815765,0.170912966132164,0.17255598306655884,0.17400507628917694,0.1742665320634842,0.1742340326309204,0.178177610039711,0.17715469002723694,0.17915478348731995,0.18171408772468567,0.18318957090377808,0.18398776650428772,0.18668106198310852,0.18693403899669647,0.19325096905231476,0.19556614756584167,0.19353653490543365,0.1893535852432251,0.18858852982521057,0.18759498000144958,0.1859980970621109,0.18598005175590515,0.18370208144187927,0.18406309187412262,0.1852990686893463,0.18721792101860046,0.1904820054769516,0.19161991775035858,0.19496455788612366,0.19757159054279327,0.2023206502199173,0.2060345560312271,0.21145808696746826,0.21052353084087372,0.21011635661125183,0.21909022331237793,0.22912992537021637,0.24160674214363098,0.2509598731994629,0.24516946077346802,0.2321656048297882,0.21993917226791382,0.20301060378551483,0.19283846020698547,0.18495000898838043,0.1809331625699997,0.17212508618831635,0.16322381794452667,0.15971767902374268,0.15500305593013763,0.15285718441009521,0.1479397565126419,0.14677734673023224,0.14868751168251038,0.15340100228786469,0.1585966944694519,0.16291645169258118]
horizon_stats (7 rows):
  |h|open_mean|open_std|open_q25|open_q50|open_q75|handover_mean|handover_std|handover_q25|handover_q50|handover_q75|
  |---|---|---|---|---|---|---|---|---|---|---|
  |1|0.029773293063044548|0.007404169999063015|0.02418188378214836|0.03045070543885231|0.034978218376636505|0.029773293063044548|0.007404169999063015|0.02418188378214836|0.03045070543885231|0.034978218376636505|
  |5|0.11091844737529755|0.01313008926808834|0.10323502123355865|0.11103898286819458|0.11982263624668121|0.11091844737529755|0.01313008926808834|0.10323502123355865|0.11103898286819458|0.11982263624668121|
  |10|0.18621589243412018|0.06522116810083389|0.14956873655319214|0.17904238402843475|0.1977941244840622|0.1865493357181549|0.06625955551862717|0.14956873655319214|0.17904238402843475|0.1977941244840622|
  |25|0.11596996337175369|0.04134652391076088|0.09005386382341385|0.11042642593383789|0.13474488258361816|0.12420030683279037|0.049917254596948624|0.092357337474823|0.11520783603191376|0.14913947880268097|
  |50|0.10377092659473419|0.07100620120763779|0.07008630037307739|0.09022549539804459|0.11078011989593506|0.14960841834545135|0.06617525219917297|0.10603687167167664|0.13271647691726685|0.178177610039711|
  |80|0.08987805992364883|0.050000619143247604|0.06734035909175873|0.07978630065917969|0.09556256979703903|0.18273349106311798|0.08334916085004807|0.12294067442417145|0.1545676589012146|0.22912992537021637|
  |100|0.11163192987442017|0.04782211408019066|0.07957030832767487|0.10282396525144577|0.13074742257595062|0.14202861487865448|0.064083032310009|0.09960572421550751|0.13159942626953125|0.16291645169258118|
energy:
  spwm_neuro_nj: 12.343340104910045
  spwm_gpu_nj: 60.364799999999995
  ann_rate_gpu_nj: 60.364799999999995
  deep_gru_gpu_nj: 269.568
  rssm_gpu_nj: 328.55039999999997
  convlstm_gpu_nj: 2123.3664
  speedup_vs_own_gpu: 4.890475307893975
corl_final_verification:
  pil_3seed_comparison:
    spwm_n25_val:
      mean_std:
        grasp_success_rate_pct: [60.666666666666664,2.3094010767585074]
        terminal_placement_accuracy_cm: [0.8798704931139946,0.10505539807183523]
        terminal_error_cm: [2.006662103931109,0.11581144347528279]
        mean_joint_accel_rad_s2: [0.009308865414932371,0.0034586698705177275]
        mean_joint_jerk_rad_s3: [0.006634432549277941,0.003263484407032791]
        mean_direction_flips: [21.14,8.549807015365904]
        mppi_inference_latency_ms: [26.649352693166346,2.271980219747186]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 57.99999999999999
            terminal_placement_accuracy_cm: 0.9966417002677918
            placement_accuracy_std_cm: 0.7168366082198878
            terminal_error_cm: 2.117010636925697
            terminal_error_std_cm: 1.2261713068281301
            mppi_inference_latency_ms: 27.953717743547482
            mean_joint_accel_rad_s2: 0.005460824174806475
            std_joint_accel_rad_s2: 0.0014449158194058374
            mean_joint_jerk_rad_s3: 0.003033191980794072
            std_joint_jerk_rad_s3: 0.0006583887773716086
            mean_direction_flips: 11.82
            std_direction_flips: 6.098163658020339
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 0.8499453981220723
            placement_accuracy_std_cm: 0.5894733005655424
            terminal_error_cm: 2.0169071877002716
            terminal_error_std_cm: 1.1516123596745238
            mppi_inference_latency_ms: 27.968430673446996
            mean_joint_accel_rad_s2: 0.012158537823706865
            std_joint_accel_rad_s2: 0.0038326392576793562
            mean_joint_jerk_rad_s3: 0.00939610967412591
            std_joint_jerk_rad_s3: 0.0024506745164746524
            mean_direction_flips: 28.62
            std_direction_flips: 8.126229137798171
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_2.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 0.7930243809521198
            placement_accuracy_std_cm: 0.5496822026263848
            terminal_error_cm: 1.8860684871673583
            terminal_error_std_cm: 1.187425109721587
            mppi_inference_latency_ms: 24.025909662504564
            mean_joint_accel_rad_s2: 0.01030723424628377
            std_joint_accel_rad_s2: 0.003965905046145654
            mean_joint_jerk_rad_s3: 0.007473995992913842
            std_joint_jerk_rad_s3: 0.0027093614883279904
            mean_direction_flips: 22.98
            std_direction_flips: 5.540721974616666
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
    spwm_n6_tier2:
      mean_std:
        grasp_success_rate_pct: [58.0,27.055498516937366]
        terminal_placement_accuracy_cm: [0.8992845916748048,0.0417493623118023]
        terminal_error_cm: [2.231203550299009,0.6846823009200784]
        mean_joint_accel_rad_s2: [0.008598180722134809,0.003043671174468545]
        mean_joint_jerk_rad_s3: [0.00608342978792886,0.0029084716855952435]
        mean_direction_flips: [19.373333333333335,7.574545090850891]
        mppi_inference_latency_ms: [27.510263026675336,7.919433452921299]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: null
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 86.0
            terminal_placement_accuracy_cm: 0.8552020224928856
            placement_accuracy_std_cm: 0.7991095820631844
            terminal_error_cm: 1.5668222349882126
            terminal_error_std_cm: 1.004804437956763
            mppi_inference_latency_ms: 36.65447710343201
            mean_joint_accel_rad_s2: 0.0058662799885496495
            std_joint_accel_rad_s2: 0.0014419035578661795
            mean_joint_jerk_rad_s3: 0.003542280187830329
            std_joint_jerk_rad_s3: 0.000713988545542795
            mean_direction_flips: 14.02
            std_direction_flips: 7.643271550847844
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: null
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 56.00000000000001
            terminal_placement_accuracy_cm: 0.9044275161623955
            placement_accuracy_std_cm: 0.6471105496304267
            terminal_error_cm: 2.1922635650634765
            terminal_error_std_cm: 1.1926734087904072
            mppi_inference_latency_ms: 23.008426210175344
            mean_joint_accel_rad_s2: 0.011878934735432268
            std_joint_accel_rad_s2: 0.00374689697219965
            mean_joint_jerk_rad_s3: 0.009255616161972284
            std_joint_jerk_rad_s3: 0.0022864994051547106
            mean_direction_flips: 28.04
            std_direction_flips: 7.623542483648924
          handover_telemetry: null
        2024:
          checkpoint: results/multiseed/seed_2024_m1/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: null
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 32.0
            terminal_placement_accuracy_cm: 0.938224236369133
            placement_accuracy_std_cm: 0.6394334407237005
            terminal_error_cm: 2.934524850845337
            terminal_error_std_cm: 1.8091352151209616
            mppi_inference_latency_ms: 22.86788576641866
            mean_joint_accel_rad_s2: 0.008049327442422509
            std_joint_accel_rad_s2: 0.0032349260861896404
            mean_joint_jerk_rad_s3: 0.005452393013983965
            std_joint_jerk_rad_s3: 0.0021297406964725127
            mean_direction_flips: 16.06
            std_direction_flips: 4.6364210335128115
          handover_telemetry: null
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
    ann_rate:
      mean_std:
        grasp_success_rate_pct: [37.333333333333336,9.451631252505218]
        terminal_placement_accuracy_cm: [1.0678065086901187,0.2828632725018437]
        terminal_error_cm: [2.9846295044819513,0.4621451582711122]
        mean_joint_accel_rad_s2: [0.011481044394895437,0.0008839583251549979]
        mean_joint_jerk_rad_s3: [0.009379258422801892,0.0010705614356774748]
        mean_direction_flips: [26.0,1.94648400969543]
        mppi_inference_latency_ms: [24.616702500178565,0.07683415875005363]
      per_seed:
        42:
          checkpoint: results/ann_ablation/seed_42_m1/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: null
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 34.0
            terminal_placement_accuracy_cm: 0.8320508192479611
            placement_accuracy_std_cm: 0.4655940325677702
            terminal_error_cm: 2.7538758546113966
            terminal_error_std_cm: 1.6488538132409942
            mppi_inference_latency_ms: 24.528368778591304
            mean_joint_accel_rad_s2: 0.012113787466660142
            std_joint_accel_rad_s2: 0.004143865865535021
            mean_joint_jerk_rad_s3: 0.00986627260223031
            std_joint_jerk_rad_s3: 0.0032970702130067234
            mean_direction_flips: 28.24
            std_direction_flips: 7.061331319234355
          handover_telemetry: null
        101:
          checkpoint: results/ann_ablation/seed_101_m1/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: null
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 30.0
            terminal_placement_accuracy_cm: 1.3814548110961915
            placement_accuracy_std_cm: 1.0088033363029787
            terminal_error_cm: 3.5167109155654908
            terminal_error_std_cm: 1.908053762203449
            mppi_inference_latency_ms: 24.66803547384716
            mean_joint_accel_rad_s2: 0.011858293414115905
            std_joint_accel_rad_s2: 0.0038084789794641236
            mean_joint_jerk_rad_s3: 0.010119730336591602
            std_joint_jerk_rad_s3: 0.0028768772037011977
            mean_direction_flips: 25.04
            std_direction_flips: 6.483702645865247
          handover_telemetry: null
        2024:
          checkpoint: results/ann_ablation/seed_2024_m1/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: null
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 48.0
            terminal_placement_accuracy_cm: 0.9899138957262039
            placement_accuracy_std_cm: 0.6084400484165465
            terminal_error_cm: 2.6833017432689665
            terminal_error_std_cm: 1.3172601917506164
            mppi_inference_latency_ms: 24.65370324809723
            mean_joint_accel_rad_s2: 0.010471052303910256
            std_joint_accel_rad_s2: 0.004437141968405458
            mean_joint_jerk_rad_s3: 0.008151772329583765
            std_joint_jerk_rad_s3: 0.0029226110981672468
            mean_direction_flips: 24.72
            std_direction_flips: 6.318354216091402
          handover_telemetry: null
    rssm_open_loop:
      mean_std:
        grasp_success_rate_pct: [15.333333333333334,7.023769168568492]
        terminal_placement_accuracy_cm: [1.2335922479629515,0.05552042269442134]
        terminal_error_cm: [2.738770661751429,0.06877343601692858]
        mean_joint_accel_rad_s2: [0.02037182800471783,0.000592333680216826]
        mean_joint_jerk_rad_s3: [0.02138510186225176,0.0004846155447845476]
        mean_direction_flips: [36.85999999999999,1.2150720143267217]
        mppi_inference_latency_ms: [46.81975616126076,0.1485665027581289]
      per_seed:
        42:
          checkpoint: results/baselines/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 16.0
            terminal_placement_accuracy_cm: 1.2976850962638855
            placement_accuracy_std_cm: 0.6328096448883076
            terminal_error_cm: 2.7923242807388307
            terminal_error_std_cm: 1.2528453059283846
            mppi_inference_latency_ms: 46.82606889366672
            mean_joint_accel_rad_s2: 0.021038663629442452
            std_joint_accel_rad_s2: 0.004804290298606692
            mean_joint_jerk_rad_s3: 0.02193130200728774
            std_joint_jerk_rad_s3: 0.004306558567727457
            mean_direction_flips: 36.24
            std_direction_flips: 9.049994475136435
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed_rssm/seed_101/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 8.0
            terminal_placement_accuracy_cm: 1.2002819275856018
            placement_accuracy_std_cm: 0.5980547321361083
            terminal_error_cm: 2.7627756690979
            terminal_error_std_cm: 1.3512200627382058
            mppi_inference_latency_ms: 46.96506567598321
            mean_joint_accel_rad_s2: 0.020170155689120294
            std_joint_accel_rad_s2: 0.004630519878358685
            mean_joint_jerk_rad_s3: 0.021006639190018175
            std_joint_jerk_rad_s3: 0.0047702734702861245
            mean_direction_flips: 36.08
            std_direction_flips: 9.397531590795532
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed_rssm/seed_2024/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 22.0
            terminal_placement_accuracy_cm: 1.2028097200393677
            placement_accuracy_std_cm: 0.6313406366226356
            terminal_error_cm: 2.661212035417557
            terminal_error_std_cm: 1.5630930624192183
            mppi_inference_latency_ms: 46.66813391413238
            mean_joint_accel_rad_s2: 0.019906664695590735
            std_joint_accel_rad_s2: 0.005206586521354414
            mean_joint_jerk_rad_s3: 0.021217364389449357
            std_joint_jerk_rad_s3: 0.004626785958574825
            mean_direction_flips: 38.26
            std_direction_flips: 8.62278377323704
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
    rssm_ungated:
      mean_std:
        grasp_success_rate_pct: [0.0,0.0]
        terminal_placement_accuracy_cm: [1.425072744091352,0.03429525285538392]
        terminal_error_cm: [25.49343025525411,1.3050838203645958]
        mean_joint_accel_rad_s2: [0.03153110054632028,0.0006715274430858012]
        mean_joint_jerk_rad_s3: [0.029558277900020278,0.0004172700652986988]
        mean_direction_flips: [36.49333333333333,0.7757147241952662]
        mppi_inference_latency_ms: [46.936845150350116,0.33605994404428824]
      per_seed:
        42:
          checkpoint: results/baselines/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: ungated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 0.0
            terminal_placement_accuracy_cm: 1.456915372610092
            placement_accuracy_std_cm: 0.888517029752088
            terminal_error_cm: 24.710977087020876
            terminal_error_std_cm: 10.710275307660488
            mppi_inference_latency_ms: 46.55402094512669
            mean_joint_accel_rad_s2: 0.0310553040727973
            std_joint_accel_rad_s2: 0.006334347208369975
            mean_joint_jerk_rad_s3: 0.02920039013028145
            std_joint_jerk_rad_s3: 0.004958710804918712
            mean_direction_flips: 36.16
            std_direction_flips: 9.016340721157336
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 50
            episodes_with_w_gt_0_pct: 100.0
            total_phi1_steps: 400
            triggered_phi1_steps: 400
            phi1_steps_with_w_gt_0_pct: 100.0
            mean_nonzero_w: 1.0
        101:
          checkpoint: results/multiseed_rssm/seed_101/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: ungated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 0.0
            terminal_placement_accuracy_cm: 1.3887627166509628
            placement_accuracy_std_cm: 0.7836427217228974
            terminal_error_cm: 27.000035276412962
            terminal_error_std_cm: 11.571441732882644
            mppi_inference_latency_ms: 47.07329811439202
            mean_joint_accel_rad_s2: 0.03229924626648426
            std_joint_accel_rad_s2: 0.006498370430261997
            mean_joint_jerk_rad_s3: 0.030016598887741566
            std_joint_jerk_rad_s3: 0.005532750884270267
            mean_direction_flips: 35.94
            std_direction_flips: 8.951893654417482
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 50
            episodes_with_w_gt_0_pct: 100.0
            total_phi1_steps: 400
            triggered_phi1_steps: 400
            phi1_steps_with_w_gt_0_pct: 100.0
            mean_nonzero_w: 1.0
        2024:
          checkpoint: results/multiseed_rssm/seed_2024/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: ungated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 0.0
            terminal_placement_accuracy_cm: 1.4295401430130006
            placement_accuracy_std_cm: 0.8761971421556884
            terminal_error_cm: 24.769278402328492
            terminal_error_std_cm: 10.309742604307631
            mppi_inference_latency_ms: 47.18321639153165
            mean_joint_accel_rad_s2: 0.03123875129967928
            std_joint_accel_rad_s2: 0.007419252910420967
            mean_joint_jerk_rad_s3: 0.02945784468203783
            std_joint_jerk_rad_s3: 0.006067063371662604
            mean_direction_flips: 37.38
            std_direction_flips: 8.66
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 50
            episodes_with_w_gt_0_pct: 100.0
            total_phi1_steps: 400
            triggered_phi1_steps: 400
            phi1_steps_with_w_gt_0_pct: 100.0
            mean_nonzero_w: 1.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[100.0,0.0],phi1_steps_with_w_gt_0_pct:[100.0,0.0],mean_nonzero_w_when_active:1.0}
    rssm_residual_gated:
      mean_std:
        grasp_success_rate_pct: [15.333333333333334,5.033222956847166]
        terminal_placement_accuracy_cm: [1.233093837896983,0.046829598001233756]
        terminal_error_cm: [2.7240475718180335,0.10990075377541059]
        mean_joint_accel_rad_s2: [0.020493890543778735,0.0006706944714435922]
        mean_joint_jerk_rad_s3: [0.021609529219567773,0.0005316353717258448]
        mean_direction_flips: [36.940000000000005,1.2257242756835658]
        mppi_inference_latency_ms: [46.472593615331895,0.1711331627137406]
      per_seed:
        42:
          checkpoint: results/baselines/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 16.0
            terminal_placement_accuracy_cm: 1.2852557504177093
            placement_accuracy_std_cm: 0.6163579389948347
            terminal_error_cm: 2.8026852571964262
            terminal_error_std_cm: 1.250732118205452
            mppi_inference_latency_ms: 46.27922801228423
            mean_joint_accel_rad_s2: 0.02125625854358077
            std_joint_accel_rad_s2: 0.0052680983475184295
            mean_joint_jerk_rad_s3: 0.02221311992034316
            std_joint_jerk_rad_s3: 0.004852316653278585
            mean_direction_flips: 36.42
            std_direction_flips: 8.913113933973918
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 9
            episodes_with_w_gt_0_pct: 18.0
            total_phi1_steps: 400
            triggered_phi1_steps: 9
            phi1_steps_with_w_gt_0_pct: 2.25
            mean_nonzero_w: 0.18157258285416497
        101:
          checkpoint: results/multiseed_rssm/seed_101/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 10.0
            terminal_placement_accuracy_cm: 1.2193563985824585
            placement_accuracy_std_cm: 0.6733557358778788
            terminal_error_cm: 2.770985679626465
            terminal_error_std_cm: 1.329820118983065
            mppi_inference_latency_ms: 46.53400791747006
            mean_joint_accel_rad_s2: 0.02023072073236108
            std_joint_accel_rad_s2: 0.004711884963105292
            mean_joint_jerk_rad_s3: 0.021210806742310524
            std_joint_jerk_rad_s3: 0.00497133476627864
            mean_direction_flips: 36.06
            std_direction_flips: 9.491912346835068
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 20
            episodes_with_w_gt_0_pct: 40.0
            total_phi1_steps: 400
            triggered_phi1_steps: 27
            phi1_steps_with_w_gt_0_pct: 6.75
            mean_nonzero_w: 0.22636996290198078
        2024:
          checkpoint: results/multiseed_rssm/seed_2024/continuous_rssm/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 20.0
            terminal_placement_accuracy_cm: 1.1946693646907807
            placement_accuracy_std_cm: 0.6255705025940155
            terminal_error_cm: 2.59847177863121
            terminal_error_std_cm: 1.4417185983615668
            mppi_inference_latency_ms: 46.604544916241395
            mean_joint_accel_rad_s2: 0.019994692355394365
            std_joint_accel_rad_s2: 0.005334068808050576
            mean_joint_jerk_rad_s3: 0.02140466099604964
            std_joint_jerk_rad_s3: 0.0048616737052424785
            mean_direction_flips: 38.34
            std_direction_flips: 8.564134515524614
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 17
            episodes_with_w_gt_0_pct: 34.0
            total_phi1_steps: 400
            triggered_phi1_steps: 24
            phi1_steps_with_w_gt_0_pct: 6.0
            mean_nonzero_w: 0.21099277816712858
      handover_telemetry_3seed:
        episodes_with_w_gt_0_pct: [30.666666666666668,11.372481406154654]
        phi1_steps_with_w_gt_0_pct: [5.0,2.4109126902482387]
        mean_nonzero_w_when_active: 0.20631177464109143
    deep_gru_open_loop:
      mean_std:
        grasp_success_rate_pct: [77.33333333333333,4.163331998932265]
        terminal_placement_accuracy_cm: [1.142497578263283,0.31404892460065076]
        terminal_error_cm: [1.8994051045179365,0.17642912388300128]
        mean_joint_accel_rad_s2: [0.004498507350217551,0.00012105405328595289]
        mean_joint_jerk_rad_s3: [0.002675323402121042,0.00018613435682786266]
        mean_direction_flips: [10.793333333333331,0.6331929669013493]
        mppi_inference_latency_ms: [57.845663712604754,0.2935253213907278]
      per_seed:
        42:
          checkpoint: results/baselines/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 82.0
            terminal_placement_accuracy_cm: 0.8277105885744095
            placement_accuracy_std_cm: 0.619948305625996
            terminal_error_cm: 1.7091500055789948
            terminal_error_std_cm: 0.8538347073403084
            mppi_inference_latency_ms: 57.99607291789624
            mean_joint_accel_rad_s2: 0.004467548907268792
            std_joint_accel_rad_s2: 0.0025332651658058424
            mean_joint_jerk_rad_s3: 0.002733891133684665
            std_joint_jerk_rad_s3: 0.0015915138679954569
            mean_direction_flips: 11.52
            std_direction_flips: 4.00119982005398
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed_rssm/seed_101/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 74.0
            terminal_placement_accuracy_cm: 1.143978949189186
            placement_accuracy_std_cm: 1.0157706090458067
            terminal_error_cm: 2.0576158034801484
            terminal_error_std_cm: 2.307079695708016
            mppi_inference_latency_ms: 57.50741934581636
            mean_joint_accel_rad_s2: 0.004632034294772893
            std_joint_accel_rad_s2: 0.0036805739683259996
            mean_joint_jerk_rad_s3: 0.0028251299168914556
            std_joint_jerk_rad_s3: 0.002693590173884474
            mean_direction_flips: 10.5
            std_direction_flips: 5.99749947894954
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed_rssm/seed_2024/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 76.0
            terminal_placement_accuracy_cm: 1.4558031970262528
            placement_accuracy_std_cm: 1.0356276483254638
            terminal_error_cm: 1.9314495044946671
            terminal_error_std_cm: 1.4135605930042352
            mppi_inference_latency_ms: 58.03349887410164
            mean_joint_accel_rad_s2: 0.004395938848610967
            std_joint_accel_rad_s2: 0.0036332558167753008
            mean_joint_jerk_rad_s3: 0.002466949155787006
            std_joint_jerk_rad_s3: 0.002610848934958039
            mean_direction_flips: 10.36
            std_direction_flips: 4.353205715332093
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
    deep_gru_ungated:
      mean_std:
        grasp_success_rate_pct: [0.0,0.0]
        terminal_placement_accuracy_cm: [1.4674344965815544,0.5106457690318531]
        terminal_error_cm: [7.351427071889241,1.3439473080339845]
        mean_joint_accel_rad_s2: [0.006980255008675158,0.00042445948743814776]
        mean_joint_jerk_rad_s3: [0.004832257371939098,0.00042073079965799826]
        mean_direction_flips: [15.079999999999998,0.647147587494537]
        mppi_inference_latency_ms: [57.07348911329685,0.5185482382876814]
      per_seed:
        42:
          checkpoint: results/baselines/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: ungated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 0.0
            terminal_placement_accuracy_cm: 0.9433116026222705
            placement_accuracy_std_cm: 0.7225897143886602
            terminal_error_cm: 6.630777020454406
            terminal_error_std_cm: 4.461640380623032
            mppi_inference_latency_ms: 57.61593215447252
            mean_joint_accel_rad_s2: 0.0069884296925738455
            std_joint_accel_rad_s2: 0.004147339068371963
            mean_joint_jerk_rad_s3: 0.004795440048910677
            std_joint_jerk_rad_s3: 0.002581380012349003
            mean_direction_flips: 15.54
            std_direction_flips: 4.205757957847788
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 50
            episodes_with_w_gt_0_pct: 100.0
            total_phi1_steps: 400
            triggered_phi1_steps: 400
            phi1_steps_with_w_gt_0_pct: 100.0
            mean_nonzero_w: 1.0
        101:
          checkpoint: results/multiseed_rssm/seed_101/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: ungated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 0.0
            terminal_placement_accuracy_cm: 1.4955504468083383
            placement_accuracy_std_cm: 0.974366769362752
            terminal_error_cm: 8.90200080871582
            terminal_error_std_cm: 3.847520169336552
            mppi_inference_latency_ms: 56.582702259057896
            mean_joint_accel_rad_s2: 0.0074005681113339964
            std_joint_accel_rad_s2: 0.005858706618872085
            mean_joint_jerk_rad_s3: 0.005270186914131046
            std_joint_jerk_rad_s3: 0.004484711997640233
            mean_direction_flips: 15.36
            std_direction_flips: 5.172078885709305
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 50
            episodes_with_w_gt_0_pct: 100.0
            total_phi1_steps: 400
            triggered_phi1_steps: 400
            phi1_steps_with_w_gt_0_pct: 100.0
            mean_nonzero_w: 1.0
        2024:
          checkpoint: results/multiseed_rssm/seed_2024/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: ungated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 0.0
            terminal_placement_accuracy_cm: 1.9634414403140545
            placement_accuracy_std_cm: 1.1926076903006835
            terminal_error_cm: 6.521503386497497
            terminal_error_std_cm: 2.9093178808336364
            mppi_inference_latency_ms: 57.02183292636013
            mean_joint_accel_rad_s2: 0.0065517672221176325
            std_joint_accel_rad_s2: 0.005410815833356076
            mean_joint_jerk_rad_s3: 0.004431145152775571
            std_joint_jerk_rad_s3: 0.003884288999617533
            mean_direction_flips: 14.34
            std_direction_flips: 5.0541468122720765
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 50
            episodes_with_w_gt_0_pct: 100.0
            total_phi1_steps: 400
            triggered_phi1_steps: 400
            phi1_steps_with_w_gt_0_pct: 100.0
            mean_nonzero_w: 1.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[100.0,0.0],phi1_steps_with_w_gt_0_pct:[100.0,0.0],mean_nonzero_w_when_active:1.0}
    deep_gru_residual_gated:
      mean_std:
        grasp_success_rate_pct: [76.66666666666667,5.033222956847166]
        terminal_placement_accuracy_cm: [1.142546683549881,0.3141746410639535]
        terminal_error_cm: [1.9001078454653422,0.1792502651931432]
        mean_joint_accel_rad_s2: [0.004493320606804143,0.0001224338129939779]
        mean_joint_jerk_rad_s3: [0.0026687012903857978,0.0001826580482518807]
        mean_direction_flips: [10.786666666666667,0.6057502235520298]
        mppi_inference_latency_ms: [57.65735650555458,0.21359715146433306]
      per_seed:
        42:
          checkpoint: results/baselines/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 82.0
            terminal_placement_accuracy_cm: 0.8276583582162857
            placement_accuracy_std_cm: 0.6198663417549084
            terminal_error_cm: 1.7074048507213593
            terminal_error_std_cm: 0.8542619093890267
            mppi_inference_latency_ms: 57.71581469443712
            mean_joint_accel_rad_s2: 0.004455102609936148
            std_joint_accel_rad_s2: 0.002510365616416541
            mean_joint_jerk_rad_s3: 0.0027215255866758527
            std_joint_jerk_rad_s3: 0.0015617256914884875
            mean_direction_flips: 11.48
            std_direction_flips: 3.9255063367672713
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 5
            episodes_with_w_gt_0_pct: 10.0
            total_phi1_steps: 400
            triggered_phi1_steps: 8
            phi1_steps_with_w_gt_0_pct: 2.0
            mean_nonzero_w: 0.09825952008366584
        101:
          checkpoint: results/multiseed_rssm/seed_101/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 72.0
            terminal_placement_accuracy_cm: 1.143978949189186
            placement_accuracy_std_cm: 1.0157706090458067
            terminal_error_cm: 2.06187970995903
            terminal_error_std_cm: 2.30758448331492
            mppi_inference_latency_ms: 57.83563818969626
            mean_joint_accel_rad_s2: 0.004630304866004735
            std_joint_accel_rad_s2: 0.0036767455708153303
            mean_joint_jerk_rad_s3: 0.002819125670939684
            std_joint_jerk_rad_s3: 0.0026801469632155425
            mean_direction_flips: 10.52
            std_direction_flips: 6.067091560212356
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 2
            episodes_with_w_gt_0_pct: 4.0
            total_phi1_steps: 400
            triggered_phi1_steps: 2
            phi1_steps_with_w_gt_0_pct: 0.5
            mean_nonzero_w: 0.04547070369124412
        2024:
          checkpoint: results/multiseed_rssm/seed_2024/deep_gru/model.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 76.0
            terminal_placement_accuracy_cm: 1.4560027432441711
            placement_accuracy_std_cm: 1.0354441619455195
            terminal_error_cm: 1.9310389757156372
            terminal_error_std_cm: 1.4129652875940601
            mppi_inference_latency_ms: 57.42061663253034
            mean_joint_accel_rad_s2: 0.004394554344471544
            std_joint_accel_rad_s2: 0.003631740816377271
            mean_joint_jerk_rad_s3: 0.0024654526135418566
            std_joint_jerk_rad_s3: 0.0026099877867713536
            mean_direction_flips: 10.36
            std_direction_flips: 4.353205715332093
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 2
            episodes_with_w_gt_0_pct: 4.0
            total_phi1_steps: 400
            triggered_phi1_steps: 2
            phi1_steps_with_w_gt_0_pct: 0.5
            mean_nonzero_w: 0.01992271870374679
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[6.0,3.4641016151377544],phi1_steps_with_w_gt_0_pct:[1.0,0.8660254037844386],mean_nonzero_w_when_active:0.05455098082621892}
  disentangled_ablation_3seed (6 items):
    - [0]
      mean_std:
        grasp_success_rate_pct: [45.333333333333336,16.041612554021285]
        terminal_placement_accuracy_cm: [1.026371354262034,0.21267911327447728]
        terminal_error_cm: [2.502900831302007,0.10674231587914605]
        mean_joint_accel_rad_s2: [0.009831757796928286,0.002547908271341173]
        mean_joint_jerk_rad_s3: [0.007646870988731583,0.002496011682578312]
        mean_direction_flips: [23.03333333333333,6.83585644475755]
        mppi_inference_latency_ms: [26.607863672024582,0.31494921572578505]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 1.270518583357334
            placement_accuracy_std_cm: 1.1433059847844562
            terminal_error_cm: 2.51325437605381
            terminal_error_std_cm: 1.591850890790287
            mppi_inference_latency_ms: 26.800422591444175
            mean_joint_accel_rad_s2: 0.007099576396867633
            std_joint_accel_rad_s2: 0.001774520067574007
            mean_joint_jerk_rad_s3: 0.004990413328632712
            std_joint_jerk_rad_s3: 0.0013068799665881348
            mean_direction_flips: 16.04
            std_direction_flips: 8.304119459641703
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 44.0
            terminal_placement_accuracy_cm: 0.8813520839810371
            placement_accuracy_std_cm: 0.6318976317589341
            terminal_error_cm: 2.391359003186226
            terminal_error_std_cm: 1.2452452117898576
            mppi_inference_latency_ms: 26.778761691530867
            mean_joint_accel_rad_s2: 0.012142954394221306
            std_joint_accel_rad_s2: 0.004002151102285214
            mean_joint_jerk_rad_s3: 0.009943345058709384
            std_joint_jerk_rad_s3: 0.0026217121504472653
            mean_direction_flips: 29.7
            std_direction_flips: 7.1449282711585
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 30.0
            terminal_placement_accuracy_cm: 0.927243395447731
            placement_accuracy_std_cm: 0.6572480428206486
            terminal_error_cm: 2.6040891146659852
            terminal_error_std_cm: 1.6592857961072656
            mppi_inference_latency_ms: 26.244406733098714
            mean_joint_accel_rad_s2: 0.01025274259969592
            std_joint_accel_rad_s2: 0.003239719861131902
            mean_joint_jerk_rad_s3: 0.008006854578852654
            std_joint_jerk_rad_s3: 0.002396611082899919
            mean_direction_flips: 23.36
            std_direction_flips: 6.314301228164522
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
      label: 1. SPWM-Base (Phase 1 Warmup, Unsmoothed)
      bold: false
    - [1]
      mean_std:
        grasp_success_rate_pct: [51.333333333333336,9.451631252505216]
        terminal_placement_accuracy_cm: [1.0174584152301154,0.07739428773724741]
        terminal_error_cm: [2.413362322449684,0.131995777812547]
        mean_joint_accel_rad_s2: [0.009906536138926942,0.0024003560204176517]
        mean_joint_jerk_rad_s3: [0.007045794217847287,0.002429997855850162]
        mean_direction_flips: [21.993333333333336,6.660010010002488]
        mppi_inference_latency_ms: [27.186699187828037,0.15842402005752645]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 1.088925544321537
            placement_accuracy_std_cm: 0.9806278517950008
            terminal_error_cm: 2.2633657586574554
            terminal_error_std_cm: 1.7920521176335729
            mppi_inference_latency_ms: 27.004514395112366
            mean_joint_accel_rad_s2: 0.00728669355623424
            std_joint_accel_rad_s2: 0.001994464166249948
            mean_joint_jerk_rad_s3: 0.004507547407411039
            std_joint_jerk_rad_s3: 0.001292394719881764
            mean_direction_flips: 15.34
            std_direction_flips: 7.783598139678076
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 44.0
            terminal_placement_accuracy_cm: 0.9352575194835663
            placement_accuracy_std_cm: 0.6859384826701114
            terminal_error_cm: 2.464937143921852
            terminal_error_std_cm: 1.2342414829049078
            mppi_inference_latency_ms: 27.26348437940824
            mean_joint_accel_rad_s2: 0.012000056598335505
            std_joint_accel_rad_s2: 0.0037190684706323927
            mean_joint_jerk_rad_s3: 0.009350710520520807
            std_joint_jerk_rad_s3: 0.002442366979561876
            mean_direction_flips: 28.66
            std_direction_flips: 8.320120191439544
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 48.0
            terminal_placement_accuracy_cm: 1.0281921818852424
            placement_accuracy_std_cm: 0.6377036954563536
            terminal_error_cm: 2.5117840647697447
            terminal_error_std_cm: 1.6509070359593174
            mppi_inference_latency_ms: 27.2920987889635
            mean_joint_accel_rad_s2: 0.010432858262211084
            std_joint_accel_rad_s2: 0.003465216091106268
            mean_joint_jerk_rad_s3: 0.007279124725610018
            std_joint_jerk_rad_s3: 0.0022355287885564954
            mean_direction_flips: 21.98
            std_direction_flips: 5.948075318958226
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
      label: 2. + Smooth MPPI Controls (Pre-DAgger)
      bold: false
    - [2]
      mean_std:
        grasp_success_rate_pct: [38.0,19.078784028338916]
        terminal_placement_accuracy_cm: [1.5255670695503554,0.5931443681847731]
        terminal_error_cm: [3.2744056106607116,1.1318006892038883]
        mean_joint_accel_rad_s2: [0.0058877142627413075,0.0008819400178644788]
        mean_joint_jerk_rad_s3: [0.0035392556875012814,0.0008280183162976543]
        mean_direction_flips: [11.166666666666666,0.6463229326995401]
        mppi_inference_latency_ms: [27.422173555313975,0.18268943009863092]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 56.00000000000001
            terminal_placement_accuracy_cm: 1.1886036065220833
            placement_accuracy_std_cm: 1.0840099983177816
            terminal_error_cm: 2.372771575152874
            terminal_error_std_cm: 1.4912734211279743
            mppi_inference_latency_ms: 27.322396415769973
            mean_joint_accel_rad_s2: 0.005483685848303139
            std_joint_accel_rad_s2: 0.001604663932872117
            mean_joint_jerk_rad_s3: 0.0030811078241094947
            std_joint_jerk_rad_s3: 0.0007820425899183598
            mean_direction_flips: 11.88
            std_direction_flips: 5.891145898719535
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 18.0
            terminal_placement_accuracy_cm: 1.1776555788516998
            placement_accuracy_std_cm: 0.7053943270776382
            terminal_error_cm: 4.544530171155929
            terminal_error_std_cm: 2.1922424790505866
            mppi_inference_latency_ms: 27.311099956586432
            mean_joint_accel_rad_s2: 0.006899289446882903
            std_joint_accel_rad_s2: 0.00269182361761365
            mean_joint_jerk_rad_s3: 0.004495096248574555
            std_joint_jerk_rad_s3: 0.0013739281474203257
            mean_direction_flips: 11.0
            std_direction_flips: 3.7202150475476548
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 40.0
            terminal_placement_accuracy_cm: 2.210442023277283
            placement_accuracy_std_cm: 1.5042453738644626
            terminal_error_cm: 2.9059150856733322
            terminal_error_std_cm: 1.5601189634804675
            mppi_inference_latency_ms: 27.633024293585517
            mean_joint_accel_rad_s2: 0.005280167493037879
            std_joint_accel_rad_s2: 0.0019914505240249524
            mean_joint_jerk_rad_s3: 0.003041562989819795
            std_joint_jerk_rad_s3: 0.0010705033743340885
            mean_direction_flips: 10.62
            std_direction_flips: 3.6654604076432196
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
      label: 3a. + Phase 3/4 Curriculum (\texttt{phase\_4}, No Lookahead/Handover)
      bold: false
    - [3]
      mean_std:
        grasp_success_rate_pct: [49.333333333333336,6.110100926607791]
        terminal_placement_accuracy_cm: [1.0013195854425432,0.16452091586443013]
        terminal_error_cm: [2.2880257406830786,0.23124014692649303]
        mean_joint_accel_rad_s2: [0.009205199683395525,0.0033555613127640954]
        mean_joint_jerk_rad_s3: [0.006593972489548226,0.003202463912087665]
        mean_direction_flips: [21.18,8.536767538125892]
        mppi_inference_latency_ms: [30.746634633097653,5.982046411903032]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 56.00000000000001
            terminal_placement_accuracy_cm: 1.1886036065220833
            placement_accuracy_std_cm: 1.0840099983177816
            terminal_error_cm: 2.372771575152874
            terminal_error_std_cm: 1.4912734211279743
            mppi_inference_latency_ms: 27.322396415769973
            mean_joint_accel_rad_s2: 0.005483685848303139
            std_joint_accel_rad_s2: 0.001604663932872117
            mean_joint_jerk_rad_s3: 0.0030811078241094947
            std_joint_jerk_rad_s3: 0.0007820425899183598
            mean_direction_flips: 11.88
            std_direction_flips: 5.891145898719535
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 44.0
            terminal_placement_accuracy_cm: 0.9352575194835663
            placement_accuracy_std_cm: 0.6859384826701114
            terminal_error_cm: 2.464937143921852
            terminal_error_std_cm: 1.2342414829049078
            mppi_inference_latency_ms: 27.26348437940824
            mean_joint_accel_rad_s2: 0.012000056598335505
            std_joint_accel_rad_s2: 0.0037190684706323927
            mean_joint_jerk_rad_s3: 0.009350710520520807
            std_joint_jerk_rad_s3: 0.002442366979561876
            mean_direction_flips: 28.66
            std_direction_flips: 8.320120191439544
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_2.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 48.0
            terminal_placement_accuracy_cm: 0.8800976303219795
            placement_accuracy_std_cm: 0.5272036061850179
            terminal_error_cm: 2.02636850297451
            terminal_error_std_cm: 1.242492101039257
            mppi_inference_latency_ms: 37.65402310411474
            mean_joint_accel_rad_s2: 0.01013185660354793
            std_joint_accel_rad_s2: 0.0035494300180994444
            mean_joint_jerk_rad_s3: 0.0073500991240143775
            std_joint_jerk_rad_s3: 0.002394249901087773
            mean_direction_flips: 23.0
            std_direction_flips: 5.678027826631356
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
      label: 3b. + $N=25$ Validation Checkpoint Selection (Val-Sel., No Lookahead/Handover)
      bold: false
    - [4]
      mean_std:
        grasp_success_rate_pct: [60.666666666666664,2.3094010767585074]
        terminal_placement_accuracy_cm: [0.8798704931139946,0.10505539807183523]
        terminal_error_cm: [2.006662103931109,0.11581144347528279]
        mean_joint_accel_rad_s2: [0.009308865414932371,0.0034586698705177275]
        mean_joint_jerk_rad_s3: [0.006634432549277941,0.003263484407032791]
        mean_direction_flips: [21.14,8.549807015365904]
        mppi_inference_latency_ms: [28.034092626839513,0.24046964122825348]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 57.99999999999999
            terminal_placement_accuracy_cm: 0.9966417002677918
            placement_accuracy_std_cm: 0.7168366082198878
            terminal_error_cm: 2.117010636925697
            terminal_error_std_cm: 1.2261713068281301
            mppi_inference_latency_ms: 27.770463703774798
            mean_joint_accel_rad_s2: 0.005460824174806475
            std_joint_accel_rad_s2: 0.0014449158194058374
            mean_joint_jerk_rad_s3: 0.003033191980794072
            std_joint_jerk_rad_s3: 0.0006583887773716086
            mean_direction_flips: 11.82
            std_direction_flips: 6.098163658020339
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 0.8499453981220723
            placement_accuracy_std_cm: 0.5894733005655424
            terminal_error_cm: 2.0169071877002716
            terminal_error_std_cm: 1.1516123596745238
            mppi_inference_latency_ms: 28.09040548875141
            mean_joint_accel_rad_s2: 0.012158537823706865
            std_joint_accel_rad_s2: 0.0038326392576793562
            mean_joint_jerk_rad_s3: 0.00939610967412591
            std_joint_jerk_rad_s3: 0.0024506745164746524
            mean_direction_flips: 28.62
            std_direction_flips: 8.126229137798171
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_2.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: open_loop
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 0.7930243809521198
            placement_accuracy_std_cm: 0.5496822026263848
            terminal_error_cm: 1.8860684871673583
            terminal_error_std_cm: 1.187425109721587
            mppi_inference_latency_ms: 28.241408687992333
            mean_joint_accel_rad_s2: 0.01030723424628377
            std_joint_accel_rad_s2: 0.003965905046145654
            mean_joint_jerk_rad_s3: 0.007473995992913842
            std_joint_jerk_rad_s3: 0.0027093614883279904
            mean_direction_flips: 22.98
            std_direction_flips: 5.540721974616666
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
      label: 4a. + 2-Stage SE(3) Lookahead (Val-Sel., $w_t \equiv 0$)
      bold: false
    - [5]
      mean_std:
        grasp_success_rate_pct: [60.666666666666664,2.3094010767585074]
        terminal_placement_accuracy_cm: [0.8798704931139946,0.10505539807183523]
        terminal_error_cm: [2.006662103931109,0.11581144347528279]
        mean_joint_accel_rad_s2: [0.009308865414932371,0.0034586698705177275]
        mean_joint_jerk_rad_s3: [0.006634432549277941,0.003263484407032791]
        mean_direction_flips: [21.14,8.549807015365904]
        mppi_inference_latency_ms: [26.649352693166346,2.271980219747186]
      per_seed:
        42:
          checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 57.99999999999999
            terminal_placement_accuracy_cm: 0.9966417002677918
            placement_accuracy_std_cm: 0.7168366082198878
            terminal_error_cm: 2.117010636925697
            terminal_error_std_cm: 1.2261713068281301
            mppi_inference_latency_ms: 27.953717743547482
            mean_joint_accel_rad_s2: 0.005460824174806475
            std_joint_accel_rad_s2: 0.0014449158194058374
            mean_joint_jerk_rad_s3: 0.003033191980794072
            std_joint_jerk_rad_s3: 0.0006583887773716086
            mean_direction_flips: 11.82
            std_direction_flips: 6.098163658020339
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        101:
          checkpoint: results/multiseed/seed_101_m1/checkpoint_phase_1.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 0.8499453981220723
            placement_accuracy_std_cm: 0.5894733005655424
            terminal_error_cm: 2.0169071877002716
            terminal_error_std_cm: 1.1516123596745238
            mppi_inference_latency_ms: 27.968430673446996
            mean_joint_accel_rad_s2: 0.012158537823706865
            std_joint_accel_rad_s2: 0.0038326392576793562
            mean_joint_jerk_rad_s3: 0.00939610967412591
            std_joint_jerk_rad_s3: 0.0024506745164746524
            mean_direction_flips: 28.62
            std_direction_flips: 8.126229137798171
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
        2024:
          checkpoint: results/multiseed/seed_2024_m1/checkpoint_phase_2.pt
          evaluation_protocol: physics_in_loop
          baseline_handover_mode: residual_gated
          episodes_evaluated: 50
          overall_metrics:
            grasp_success_rate_pct: 62.0
            terminal_placement_accuracy_cm: 0.7930243809521198
            placement_accuracy_std_cm: 0.5496822026263848
            terminal_error_cm: 1.8860684871673583
            terminal_error_std_cm: 1.187425109721587
            mppi_inference_latency_ms: 24.025909662504564
            mean_joint_accel_rad_s2: 0.01030723424628377
            std_joint_accel_rad_s2: 0.003965905046145654
            mean_joint_jerk_rad_s3: 0.007473995992913842
            std_joint_jerk_rad_s3: 0.0027093614883279904
            mean_direction_flips: 22.98
            std_direction_flips: 5.540721974616666
          handover_telemetry:
            episodes_evaluated: 50
            triggered_episodes: 0
            episodes_with_w_gt_0_pct: 0.0
            total_phi1_steps: 400
            triggered_phi1_steps: 0
            phi1_steps_with_w_gt_0_pct: 0.0
            mean_nonzero_w: 0.0
      handover_telemetry_3seed: {episodes_with_w_gt_0_pct:[0.0,0.0],phi1_steps_with_w_gt_0_pct:[0.0,0.0],mean_nonzero_w_when_active:0.0}
      label: 4b. + Residual Rejection Gate (Full SPWM, Val-Sel.)
      bold: true
  baseline_training_budgets:
    rssm:
      42:
        model_name: Continuous RSSM
        seed: 42
        device: mps
        enable_dagger_ar1: true
        base_epochs: 60
        ar1_dagger_epochs: 16
        total_epochs: 76
        best_epoch: 61
        best_val_loss: 0.4348871593291943
        wall_clock_sec: 778.0765965420287
        ar1_target_alpha: 0.25
        dagger_generations_collected: 3
      101:
        model_name: Continuous RSSM
        seed: 101
        device: mps
        enable_dagger_ar1: true
        base_epochs: 60
        ar1_dagger_epochs: 16
        total_epochs: 76
        best_epoch: 61
        best_val_loss: 0.44028351398614735
        wall_clock_sec: 775.8391683750087
        ar1_target_alpha: 0.25
        dagger_generations_collected: 3
      2024:
        model_name: Continuous RSSM
        seed: 2024
        device: mps
        enable_dagger_ar1: true
        base_epochs: 60
        ar1_dagger_epochs: 17
        total_epochs: 77
        best_epoch: 62
        best_val_loss: 0.4504756721166464
        wall_clock_sec: 826.2417404999724
        ar1_target_alpha: 0.25
        dagger_generations_collected: 3
    deep_gru:
      42:
        model_name: Deep GRU
        seed: 42
        device: mps
        enable_dagger_ar1: true
        base_epochs: 59
        ar1_dagger_epochs: 40
        total_epochs: 99
        best_epoch: 90
        best_val_loss: 0.5676361459952134
        wall_clock_sec: 2026.1408703329507
        ar1_target_alpha: 0.25
        dagger_generations_collected: 8
      101:
        model_name: Deep GRU
        seed: 101
        device: mps
        enable_dagger_ar1: true
        base_epochs: 0
        ar1_dagger_epochs: 80
        total_epochs: 80
        best_epoch: 68
        best_val_loss: 0.5387146770954132
        wall_clock_sec: 4026.366101166932
        ar1_target_alpha: 0.25
        dagger_generations_collected: 14
      2024:
        model_name: Deep GRU
        seed: 2024
        device: mps
        enable_dagger_ar1: true
        base_epochs: 0
        ar1_dagger_epochs: 80
        total_epochs: 80
        best_epoch: 80
        best_val_loss: 0.5893793037304511
        wall_clock_sec: 4009.184660000028
        ar1_target_alpha: 0.25
        dagger_generations_collected: 14
  robustness_3seed_spwm:
    conditions (5 items):
      - [0]
        condition_name: Nominal Baseline
        grasp_success_rate_pct: [60.666666666666664,2.3094010767585074]
        delta_rel_pct: 0.0
        min_placement_error_mean_cm: [0.8798704931139946,0.10505539807183523]
        terminal_placement_error_mean_cm: [2.006662103931109,0.11581144347528279]
        terminal_velocity_mean: [0.4501866717636585,0.07299750426577846]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 57.99999999999999
            min_placement_error_mean_cm: 0.9966417002677918
            terminal_placement_error_mean_cm: 2.117010636925697
            terminal_velocity_mean: 0.37095459580421447
            mean_latency_ms: 32.431992493769116
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 0.0
          101:
            grasp_success_rate_pct: 62.0
            min_placement_error_mean_cm: 0.8499453981220723
            terminal_placement_error_mean_cm: 2.0169071877002716
            terminal_velocity_mean: 0.5147094203531742
            mean_latency_ms: 39.79635441773567
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 0.0
          2024:
            grasp_success_rate_pct: 62.0
            min_placement_error_mean_cm: 0.7930243809521198
            terminal_placement_error_mean_cm: 1.8860684871673583
            terminal_velocity_mean: 0.46489599913358687
            mean_latency_ms: 39.787420788930085
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 0.0
      - [1]
        condition_name: Sensory Noise Injection (1.0%)
        grasp_success_rate_pct: [54.666666666666664,1.1547005383792557]
        delta_rel_pct: -9.89010989010989
        min_placement_error_mean_cm: [0.9180183924237887,0.24189672923787867]
        terminal_placement_error_mean_cm: [2.259123495916526,0.45390662193171594]
        terminal_velocity_mean: [0.40925723339120545,0.07120021870960654]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 54.0
            min_placement_error_mean_cm: 1.1959887701272964
            terminal_placement_error_mean_cm: 2.683878175020218
            terminal_velocity_mean: 0.33141832798719406
            mean_latency_ms: 32.060862347789936
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -6.89655172413792
          101:
            grasp_success_rate_pct: 54.0
            min_placement_error_mean_cm: 0.7552975878119469
            terminal_placement_error_mean_cm: 2.3126762568950654
            terminal_velocity_mean: 0.4710961046814919
            mean_latency_ms: 32.065856869886694
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -12.903225806451612
          2024:
            grasp_success_rate_pct: 56.00000000000001
            min_placement_error_mean_cm: 0.8027688193321229
            terminal_placement_error_mean_cm: 1.7808160558342934
            terminal_velocity_mean: 0.4252572675049305
            mean_latency_ms: 32.0065951976089
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -9.6774193548387
      - [2]
        condition_name: Payload Extrapolation (4.80 kg)
        grasp_success_rate_pct: [44.666666666666664,13.316656236958789]
        delta_rel_pct: -26.373626373626376
        min_placement_error_mean_cm: [1.3093093851208688,0.41128174441194415]
        terminal_placement_error_mean_cm: [2.4906945570309955,0.29389661766704617]
        terminal_velocity_mean: [0.6077169207235178,0.10828262151038422]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 56.00000000000001
            min_placement_error_mean_cm: 1.7644587180018425
            terminal_placement_error_mean_cm: 2.4560093522071837
            terminal_velocity_mean: 0.5556513021886349
            mean_latency_ms: 32.58270962906839
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -3.4482758620689413
          101:
            grasp_success_rate_pct: 30.0
            min_placement_error_mean_cm: 1.1991314259171486
            terminal_placement_error_mean_cm: 2.800394687652588
            terminal_velocity_mean: 0.7321977093815804
            mean_latency_ms: 32.619208271916534
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -51.61290322580645
          2024:
            grasp_success_rate_pct: 48.0
            min_placement_error_mean_cm: 0.964338011443615
            terminal_placement_error_mean_cm: 2.215679631233215
            terminal_velocity_mean: 0.535301750600338
            mean_latency_ms: 32.6236858331127
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -22.580645161290324
      - [3]
        condition_name: Actuator Drift (0.80x Friction)
        grasp_success_rate_pct: [52.666666666666664,13.316656236958785]
        delta_rel_pct: -13.186813186813188
        min_placement_error_mean_cm: [1.3143045140306153,0.353262980084955]
        terminal_placement_error_mean_cm: [2.387875195145607,0.255073983879998]
        terminal_velocity_mean: [0.5382303596039614,0.06750221070397516]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 68.0
            min_placement_error_mean_cm: 1.7202977061271667
            terminal_placement_error_mean_cm: 2.383537862300873
            terminal_velocity_mean: 0.4953418381512165
            mean_latency_ms: 33.133227099747344
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 17.24137931034484
          101:
            grasp_success_rate_pct: 44.0
            min_placement_error_mean_cm: 1.1455406373739243
            terminal_placement_error_mean_cm: 2.6450901865959167
            terminal_velocity_mean: 0.6160393211245537
            mean_latency_ms: 29.67846919087327
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -29.032258064516128
          2024:
            grasp_success_rate_pct: 46.0
            min_placement_error_mean_cm: 1.0770751985907554
            terminal_placement_error_mean_cm: 2.1349975365400313
            terminal_velocity_mean: 0.5033099195361137
            mean_latency_ms: 29.627431735741776
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -25.806451612903224
      - [4]
        condition_name: Actuator Drift (1.20x Friction)
        grasp_success_rate_pct: [48.0,19.28730152198591]
        delta_rel_pct: -20.879120879120876
        min_placement_error_mean_cm: [1.2900217257936797,0.4582222980166343]
        terminal_placement_error_mean_cm: [2.2831829558809598,0.2896295039858489]
        terminal_velocity_mean: [0.5080112110078335,0.040404712953968615]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 70.0
            min_placement_error_mean_cm: 1.787329178750515
            terminal_placement_error_mean_cm: 2.4599468809366227
            terminal_velocity_mean: 0.49150543451309203
            mean_latency_ms: 33.69387655897015
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 20.689655172413808
          101:
            grasp_success_rate_pct: 34.0
            min_placement_error_mean_cm: 1.197834121286869
            terminal_placement_error_mean_cm: 2.440669119954109
            terminal_velocity_mean: 0.554055769443512
            mean_latency_ms: 29.96104107358774
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -45.16129032258065
          2024:
            grasp_success_rate_pct: 40.0
            min_placement_error_mean_cm: 0.8849018773436547
            terminal_placement_error_mean_cm: 1.9489328667521477
            terminal_velocity_mean: 0.47847242906689647
            mean_latency_ms: 29.91935059930899
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -35.483870967741936
  robustness_3seed_ann_rate:
    conditions (5 items):
      - [0]
        condition_name: Nominal Baseline
        grasp_success_rate_pct: [37.333333333333336,9.451631252505218]
        delta_rel_pct: 0.0
        min_placement_error_mean_cm: [1.0678065086901187,0.2828632725018437]
        terminal_placement_error_mean_cm: [2.9846295044819513,0.4621451582711122]
        terminal_velocity_mean: [0.49538074051340425,0.11216522852158058]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 34.0
            min_placement_error_mean_cm: 0.8320508192479611
            terminal_placement_error_mean_cm: 2.7538758546113966
            terminal_velocity_mean: 0.5869238513708115
            mean_latency_ms: 29.43750379663485
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 0.0
          101:
            grasp_success_rate_pct: 30.0
            min_placement_error_mean_cm: 1.3814548110961915
            terminal_placement_error_mean_cm: 3.5167109155654908
            terminal_velocity_mean: 0.5289560982584953
            mean_latency_ms: 26.53065752230858
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 0.0
          2024:
            grasp_success_rate_pct: 48.0
            min_placement_error_mean_cm: 0.9899138957262039
            terminal_placement_error_mean_cm: 2.6833017432689665
            terminal_velocity_mean: 0.3702622719109058
            mean_latency_ms: 25.199068654014535
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 0.0
      - [1]
        condition_name: Sensory Noise Injection (1.0%)
        grasp_success_rate_pct: [30.0,3.4641016151377544]
        delta_rel_pct: -19.64285714285715
        min_placement_error_mean_cm: [1.0367476403713225,0.17609975464908534]
        terminal_placement_error_mean_cm: [3.362126860022545,0.529625351794463]
        terminal_velocity_mean: [0.5011851574977239,0.1012983853840122]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 32.0
            min_placement_error_mean_cm: 0.8365792346000671
            terminal_placement_error_mean_cm: 3.0554915750026703
            terminal_velocity_mean: 0.5730594700574875
            mean_latency_ms: 31.393095565214775
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -5.882352941176471
          101:
            grasp_success_rate_pct: 26.0
            min_placement_error_mean_cm: 1.1058386087417602
            terminal_placement_error_mean_cm: 3.9736847394704817
            terminal_velocity_mean: 0.5451662424206734
            mean_latency_ms: 26.004011596295314
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -13.333333333333334
          2024:
            grasp_success_rate_pct: 32.0
            min_placement_error_mean_cm: 1.1678250777721404
            terminal_placement_error_mean_cm: 3.0572042655944824
            terminal_velocity_mean: 0.38532976001501085
            mean_latency_ms: 25.061606960452693
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -33.333333333333336
      - [2]
        condition_name: Payload Extrapolation (4.80 kg)
        grasp_success_rate_pct: [26.666666666666668,16.165807537309522]
        delta_rel_pct: -28.571428571428573
        min_placement_error_mean_cm: [1.359116240243117,0.19601864522317206]
        terminal_placement_error_mean_cm: [3.6378836290041607,0.7955281178345174]
        terminal_velocity_mean: [0.6115619068592787,0.10662950140954822]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 24.0
            min_placement_error_mean_cm: 1.1392030370235444
            terminal_placement_error_mean_cm: 3.5584317278862
            terminal_velocity_mean: 0.6400154614448548
            mean_latency_ms: 29.94859264750354
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -29.41176470588235
          101:
            grasp_success_rate_pct: 12.0
            min_placement_error_mean_cm: 1.4226837638020515
            terminal_placement_error_mean_cm: 4.470156444311142
            terminal_velocity_mean: 0.7010783076286315
            mean_latency_ms: 25.1213509093544
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -60.0
          2024:
            grasp_success_rate_pct: 44.0
            min_placement_error_mean_cm: 1.5154619199037551
            terminal_placement_error_mean_cm: 2.88506271481514
            terminal_velocity_mean: 0.4935919515043497
            mean_latency_ms: 25.066311646045065
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -8.333333333333334
      - [3]
        condition_name: Actuator Drift (0.80x Friction)
        grasp_success_rate_pct: [30.666666666666668,14.047538337136984]
        delta_rel_pct: -17.857142857142858
        min_placement_error_mean_cm: [1.4180172796050707,0.21795292848457692]
        terminal_placement_error_mean_cm: [3.740554449160894,0.8331617947215637]
        terminal_velocity_mean: [0.5717242361605167,0.13817699362402994]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 32.0
            min_placement_error_mean_cm: 1.1665561926364898
            terminal_placement_error_mean_cm: 3.536238695383072
            terminal_velocity_mean: 0.643540482223034
            mean_latency_ms: 25.223359415711226
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -5.882352941176471
          101:
            grasp_success_rate_pct: 16.0
            min_placement_error_mean_cm: 1.5526343488693237
            terminal_placement_error_mean_cm: 4.656868257522583
            terminal_velocity_mean: 0.6592045310139656
            mean_latency_ms: 25.066783458762252
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -46.666666666666664
          2024:
            grasp_success_rate_pct: 44.0
            min_placement_error_mean_cm: 1.5348612973093987
            terminal_placement_error_mean_cm: 3.028556394577026
            terminal_velocity_mean: 0.4124276952445507
            mean_latency_ms: 24.790206348217293
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -8.333333333333334
      - [4]
        condition_name: Actuator Drift (1.20x Friction)
        grasp_success_rate_pct: [33.333333333333336,16.65332799572906]
        delta_rel_pct: -10.714285714285714
        min_placement_error_mean_cm: [1.3353092057506244,0.20851366158174045]
        terminal_placement_error_mean_cm: [3.510111181139946,1.2144949894390102]
        terminal_velocity_mean: [0.5205225745340188,0.10602974885015383]
        episodes_with_handover_pct: [0.0,0.0]
        phi1_steps_with_handover_pct: [0.0,0.0]
        mean_nonzero_handover_w_when_active: 0.0
        per_seed:
          42:
            grasp_success_rate_pct: 28.000000000000004
            min_placement_error_mean_cm: 1.101568892300129
            terminal_placement_error_mean_cm: 2.80165857732296
            terminal_velocity_mean: 0.6022750130295753
            mean_latency_ms: 25.216824718118666
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -17.647058823529402
          101:
            grasp_success_rate_pct: 20.0
            min_placement_error_mean_cm: 1.40215890198946
            terminal_placement_error_mean_cm: 4.912464030981064
            terminal_velocity_mean: 0.5585748782753944
            mean_latency_ms: 25.116685864205966
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: -33.333333333333336
          2024:
            grasp_success_rate_pct: 52.0
            min_placement_error_mean_cm: 1.5021998229622842
            terminal_placement_error_mean_cm: 2.816210935115814
            terminal_velocity_mean: 0.40071783229708674
            mean_latency_ms: 25.06705994109404
            episodes_with_handover_pct: 0.0
            phi1_steps_with_handover_pct: 0.0
            mean_nonzero_handover_w: 0.0
            delta_rel_pct: 8.333333333333334
  robustness_3seed_deep_gru:
    conditions (5 items):
      - [0]
        condition_name: Nominal Baseline
        grasp_success_rate_pct: [76.66666666666667,5.033222956847166]
        delta_rel_pct: 0.0
        min_placement_error_mean_cm: [1.142546683549881,0.3141746410639535]
        terminal_placement_error_mean_cm: [1.9001078454653422,0.1792502651931432]
        terminal_velocity_mean: [0.29516326268514,0.07536613976838809]
        episodes_with_handover_pct: [6.0,3.4641016151377544]
        phi1_steps_with_handover_pct: [1.0,0.8660254037844386]
        mean_nonzero_handover_w_when_active: 0.05455098082621892
        per_seed:
          42:
            grasp_success_rate_pct: 82.0
            min_placement_error_mean_cm: 0.8276583582162857
            terminal_placement_error_mean_cm: 1.7074048507213593
            terminal_velocity_mean: 0.29966184198856355
            mean_latency_ms: 57.95750650222223
            episodes_with_handover_pct: 10.0
            phi1_steps_with_handover_pct: 2.0
            mean_nonzero_handover_w: 0.09825952008366584
            delta_rel_pct: 0.0
          101:
            grasp_success_rate_pct: 72.0
            min_placement_error_mean_cm: 1.143978949189186
            terminal_placement_error_mean_cm: 2.06187970995903
            terminal_velocity_mean: 0.36817935094237325
            mean_latency_ms: 54.371179489554706
            episodes_with_handover_pct: 4.0
            phi1_steps_with_handover_pct: 0.5
            mean_nonzero_handover_w: 0.04547070369124412
            delta_rel_pct: 0.0
          2024:
            grasp_success_rate_pct: 76.0
            min_placement_error_mean_cm: 1.4560027432441711
            terminal_placement_error_mean_cm: 1.9310389757156372
            terminal_velocity_mean: 0.21764859512448312
            mean_latency_ms: 54.6402853776519
            episodes_with_handover_pct: 4.0
            phi1_steps_with_handover_pct: 0.5
            mean_nonzero_handover_w: 0.01992271870374679
            delta_rel_pct: 0.0
      - [1]
        condition_name: Sensory Noise Injection (1.0%)
        grasp_success_rate_pct: [74.66666666666667,3.055050463303893]
        delta_rel_pct: -2.608695652173913
        min_placement_error_mean_cm: [1.5151751696070033,0.30314335112481977]
        terminal_placement_error_mean_cm: [1.853916923602422,0.23174274041202594]
        terminal_velocity_mean: [0.2947493451088667,0.09277998260340467]
        episodes_with_handover_pct: [34.666666666666664,11.015141094572202]
        phi1_steps_with_handover_pct: [10.666666666666666,4.488411003165077]
        mean_nonzero_handover_w_when_active: 0.15266501222733564
        per_seed:
          42:
            grasp_success_rate_pct: 72.0
            min_placement_error_mean_cm: 1.8375933802127837
            terminal_placement_error_mean_cm: 2.056955361366272
            terminal_velocity_mean: 0.3729041929543018
            mean_latency_ms: 45.58725326396069
            episodes_with_handover_pct: 46.0
            phi1_steps_with_handover_pct: 15.75
            mean_nonzero_handover_w: 0.1645895887414614
            delta_rel_pct: -12.195121951219512
          101:
            grasp_success_rate_pct: 74.0
            min_placement_error_mean_cm: 1.4719947236776352
            terminal_placement_error_mean_cm: 1.9033488303422927
            terminal_velocity_mean: 0.31913049027323726
            mean_latency_ms: 54.78286882565354
            episodes_with_handover_pct: 34.0
            phi1_steps_with_handover_pct: 9.0
            mean_nonzero_handover_w: 0.20296063053111232
            delta_rel_pct: 2.7777777777777777
          2024:
            grasp_success_rate_pct: 78.0
            min_placement_error_mean_cm: 1.2359374049305916
            terminal_placement_error_mean_cm: 1.6014465790987016
            terminal_velocity_mean: 0.192213352099061
            mean_latency_ms: 55.27344265506174
            episodes_with_handover_pct: 24.0
            phi1_steps_with_handover_pct: 7.25
            mean_nonzero_handover_w: 0.09044481740943319
            delta_rel_pct: 2.6315789473684212
      - [2]
        condition_name: Payload Extrapolation (4.80 kg)
        grasp_success_rate_pct: [23.333333333333332,12.858201014657272]
        delta_rel_pct: -69.56521739130436
        min_placement_error_mean_cm: [3.2138332206010816,1.4700081785960324]
        terminal_placement_error_mean_cm: [4.3460664258400605,1.0614877655467412]
        terminal_velocity_mean: [0.49405148267745974,0.21990952766148325]
        episodes_with_handover_pct: [29.333333333333332,40.611985094714754]
        phi1_steps_with_handover_pct: [7.75,11.715374513859981]
        mean_nonzero_handover_w_when_active: 0.34301347039677516
        per_seed:
          42:
            grasp_success_rate_pct: 14.000000000000002
            min_placement_error_mean_cm: 2.678343471288681
            terminal_placement_error_mean_cm: 4.692428379058838
            terminal_velocity_mean: 0.3116999238729477
            mean_latency_ms: 39.307929548496304
            episodes_with_handover_pct: 10.0
            phi1_steps_with_handover_pct: 1.75
            mean_nonzero_handover_w: 0.3143822686587061
            delta_rel_pct: -82.92682926829268
          101:
            grasp_success_rate_pct: 38.0
            min_placement_error_mean_cm: 2.086636486649513
            terminal_placement_error_mean_cm: 3.154660854935646
            terminal_velocity_mean: 0.7382669043540955
            mean_latency_ms: 56.36546777295293
            episodes_with_handover_pct: 2.0
            phi1_steps_with_handover_pct: 0.25
            mean_nonzero_handover_w: 0.4
            delta_rel_pct: -47.22222222222222
          2024:
            grasp_success_rate_pct: 18.0
            min_placement_error_mean_cm: 4.876519703865052
            terminal_placement_error_mean_cm: 5.191110043525696
            terminal_velocity_mean: 0.432187619805336
            mean_latency_ms: 56.881797438130974
            episodes_with_handover_pct: 76.0
            phi1_steps_with_handover_pct: 21.25
            mean_nonzero_handover_w: 0.31465814253161933
            delta_rel_pct: -76.3157894736842
      - [3]
        condition_name: Actuator Drift (0.80x Friction)
        grasp_success_rate_pct: [26.666666666666668,23.86070689089771]
        delta_rel_pct: -65.21739130434783
        min_placement_error_mean_cm: [3.236167592306932,1.4473405546296252]
        terminal_placement_error_mean_cm: [4.381567585865656,1.0010487774447363]
        terminal_velocity_mean: [0.44115810414155326,0.20521995946808633]
        episodes_with_handover_pct: [28.666666666666668,39.46306289852998]
        phi1_steps_with_handover_pct: [8.083333333333334,11.877850535064555]
        mean_nonzero_handover_w_when_active: 0.3593107620300309
        per_seed:
          42:
            grasp_success_rate_pct: 10.0
            min_placement_error_mean_cm: 2.719559631347656
            terminal_placement_error_mean_cm: 4.764410569667816
            terminal_velocity_mean: 0.26115636616945265
            mean_latency_ms: 39.652253493452214
            episodes_with_handover_pct: 10.0
            phi1_steps_with_handover_pct: 2.25
            mean_nonzero_handover_w: 0.37809689356221093
            delta_rel_pct: -87.8048780487805
          101:
            grasp_success_rate_pct: 54.0
            min_placement_error_mean_cm: 2.1180153104662893
            terminal_placement_error_mean_cm: 3.2455974757671355
            terminal_velocity_mean: 0.6646309271454811
            mean_latency_ms: 57.60456265818991
            episodes_with_handover_pct: 2.0
            phi1_steps_with_handover_pct: 0.25
            mean_nonzero_handover_w: 0.4
            delta_rel_pct: -25.0
          2024:
            grasp_success_rate_pct: 16.0
            min_placement_error_mean_cm: 4.8709278351068495
            terminal_placement_error_mean_cm: 5.134694712162018
            terminal_velocity_mean: 0.3976870191097259
            mean_latency_ms: 57.76709282414901
            episodes_with_handover_pct: 74.0
            phi1_steps_with_handover_pct: 21.75
            mean_nonzero_handover_w: 0.29983539252788166
            delta_rel_pct: -78.94736842105263
      - [4]
        condition_name: Actuator Drift (1.20x Friction)
        grasp_success_rate_pct: [30.0,15.0996688705415]
        delta_rel_pct: -60.869565217391305
        min_placement_error_mean_cm: [3.104780209859212,1.3417177318534945]
        terminal_placement_error_mean_cm: [4.190640026132265,1.0252366530107015]
        terminal_velocity_mean: [0.42541863799095153,0.1914834173354981]
        episodes_with_handover_pct: [26.0,38.157568056677825]
        phi1_steps_with_handover_pct: [7.083333333333333,10.981044273352756]
        mean_nonzero_handover_w_when_active: 0.3567154332125237
        per_seed:
          42:
            grasp_success_rate_pct: 16.0
            min_placement_error_mean_cm: 2.6001962661743163
            terminal_placement_error_mean_cm: 4.637553496360779
            terminal_velocity_mean: 0.2670792493224144
            mean_latency_ms: 40.279339710981084
            episodes_with_handover_pct: 6.0
            phi1_steps_with_handover_pct: 1.25
            mean_nonzero_handover_w: 0.37100348895788193
            delta_rel_pct: -80.48780487804878
          101:
            grasp_success_rate_pct: 46.0
            min_placement_error_mean_cm: 2.0885089176893232
            terminal_placement_error_mean_cm: 3.0178090459108353
            terminal_velocity_mean: 0.6382385385036469
            mean_latency_ms: 57.86060653928422
            episodes_with_handover_pct: 2.0
            phi1_steps_with_handover_pct: 0.25
            mean_nonzero_handover_w: 0.4
            delta_rel_pct: -36.111111111111114
          2024:
            grasp_success_rate_pct: 28.000000000000004
            min_placement_error_mean_cm: 4.625635445713997
            terminal_placement_error_mean_cm: 4.916557536125183
            terminal_velocity_mean: 0.3709381261467934
            mean_latency_ms: 57.83468828605162
            episodes_with_handover_pct: 70.0
            phi1_steps_with_handover_pct: 19.75
            mean_nonzero_handover_w: 0.2991428106796892
            delta_rel_pct: -63.1578947368421
  val_handover_threshold_sweep:
    seed42:
      0026 (5 rows):
        |condition_name|grasp_success_rate_pct|episodes_with_handover_pct|phi1_steps_with_handover_pct|mean_nonzero_handover_w|
        |---|---|---|---|---|
        |Nominal Baseline|78.0|0.0|0.0|0.0|
        |Sensory Noise Injection (1.0%)|78.0|0.0|0.0|0.0|
        |Payload Extrapolation (4.80 kg)|66.0|0.0|0.0|0.0|
        |Actuator Drift (0.80x Friction)|70.0|0.0|0.0|0.0|
        |Actuator Drift (1.20x Friction)|72.0|0.0|0.0|0.0|
      0020 (5 rows):
        |condition_name|grasp_success_rate_pct|episodes_with_handover_pct|phi1_steps_with_handover_pct|mean_nonzero_handover_w|
        |---|---|---|---|---|
        |Nominal Baseline|78.0|0.0|0.0|0.0|
        |Sensory Noise Injection (1.0%)|78.0|0.0|0.0|0.0|
        |Payload Extrapolation (4.80 kg)|66.0|0.0|0.0|0.0|
        |Actuator Drift (0.80x Friction)|70.0|0.0|0.0|0.0|
        |Actuator Drift (1.20x Friction)|72.0|0.0|0.0|0.0|
      0015 (5 rows):
        |condition_name|grasp_success_rate_pct|episodes_with_handover_pct|phi1_steps_with_handover_pct|mean_nonzero_handover_w|
        |---|---|---|---|---|
        |Nominal Baseline|78.0|0.0|0.0|0.0|
        |Sensory Noise Injection (1.0%)|78.0|0.0|0.0|0.0|
        |Payload Extrapolation (4.80 kg)|66.0|0.0|0.0|0.0|
        |Actuator Drift (0.80x Friction)|70.0|0.0|0.0|0.0|
        |Actuator Drift (1.20x Friction)|72.0|0.0|0.0|0.0|
    seed2024:
      0026 (5 rows):
        |condition_name|grasp_success_rate_pct|episodes_with_handover_pct|phi1_steps_with_handover_pct|mean_nonzero_handover_w|
        |---|---|---|---|---|
        |Nominal Baseline|46.0|0.0|0.0|0.0|
        |Sensory Noise Injection (1.0%)|36.0|0.0|0.0|0.0|
        |Payload Extrapolation (4.80 kg)|42.0|0.0|0.0|0.0|
        |Actuator Drift (0.80x Friction)|38.0|0.0|0.0|0.0|
        |Actuator Drift (1.20x Friction)|38.0|0.0|0.0|0.0|
      0020 (5 rows):
        |condition_name|grasp_success_rate_pct|episodes_with_handover_pct|phi1_steps_with_handover_pct|mean_nonzero_handover_w|
        |---|---|---|---|---|
        |Nominal Baseline|46.0|0.0|0.0|0.0|
        |Sensory Noise Injection (1.0%)|36.0|0.0|0.0|0.0|
        |Payload Extrapolation (4.80 kg)|42.0|0.0|0.0|0.0|
        |Actuator Drift (0.80x Friction)|38.0|0.0|0.0|0.0|
        |Actuator Drift (1.20x Friction)|38.0|0.0|0.0|0.0|
      0015 (5 rows):
        |condition_name|grasp_success_rate_pct|episodes_with_handover_pct|phi1_steps_with_handover_pct|mean_nonzero_handover_w|
        |---|---|---|---|---|
        |Nominal Baseline|46.0|0.0|0.0|0.0|
        |Sensory Noise Injection (1.0%)|36.0|0.0|0.0|0.0|
        |Payload Extrapolation (4.80 kg)|42.0|0.0|0.0|0.0|
        |Actuator Drift (0.80x Friction)|38.0|0.0|0.0|0.0|
        |Actuator Drift (1.20x Friction)|38.0|0.0|0.0|0.0|
  val25_checkpoint_selection:
    seed42:
      best_model:
        grasp_success_rate_pct: 80.0
        terminal_placement_accuracy_cm: 0.7863803243637085
        placement_accuracy_std_cm: 0.8316681269711571
        terminal_error_cm: 1.6063240730762482
        terminal_error_std_cm: 0.939099794510116
        mppi_inference_latency_ms: 37.17919258983962
        mean_joint_accel_rad_s2: 0.005947311874479055
        std_joint_accel_rad_s2: 0.0014173367373157876
        mean_joint_jerk_rad_s3: 0.0037400932516902683
        std_joint_jerk_rad_s3: 0.0007666562902440007
        mean_direction_flips: 13.48
        std_direction_flips: 7.099971830930035
        epoch: 200
      checkpoint_phase_1:
        grasp_success_rate_pct: 60.0
        terminal_placement_accuracy_cm: 0.8103122827410698
        placement_accuracy_std_cm: 0.6474393576335297
        terminal_error_cm: 1.9722674000263214
        terminal_error_std_cm: 1.4443806192871131
        mppi_inference_latency_ms: 37.00736762533041
        mean_joint_accel_rad_s2: 0.007161940094083548
        std_joint_accel_rad_s2: 0.0012864202551724185
        mean_joint_jerk_rad_s3: 0.004661122979596258
        std_joint_jerk_rad_s3: 0.0008925166603476878
        mean_direction_flips: 14.36
        std_direction_flips: 7.853050362757137
        epoch: 181
      checkpoint_phase_2:
        grasp_success_rate_pct: 52.0
        terminal_placement_accuracy_cm: 1.1208700883388518
        placement_accuracy_std_cm: 1.0542713101583618
        terminal_error_cm: 2.103588616847992
        terminal_error_std_cm: 1.37658387567322
        mppi_inference_latency_ms: 37.0669846446348
        mean_joint_accel_rad_s2: 0.007121144887059927
        std_joint_accel_rad_s2: 0.0012632364174318858
        mean_joint_jerk_rad_s3: 0.004813599018380046
        std_joint_jerk_rad_s3: 0.0008479405115867379
        mean_direction_flips: 14.88
        std_direction_flips: 8.223478582692362
        epoch: 188
      checkpoint_phase_3:
        grasp_success_rate_pct: 88.0
        terminal_placement_accuracy_cm: 0.8251915320754051
        placement_accuracy_std_cm: 1.061095902824546
        terminal_error_cm: 1.4272860133647918
        terminal_error_std_cm: 1.0950905816243908
        mppi_inference_latency_ms: 36.50109283002571
        mean_joint_accel_rad_s2: 0.005840645581483841
        std_joint_accel_rad_s2: 0.0012045624301631605
        mean_joint_jerk_rad_s3: 0.0036732837650924922
        std_joint_jerk_rad_s3: 0.0006777807353952663
        mean_direction_flips: 13.08
        std_direction_flips: 7.625850772208961
        epoch: 200
      checkpoint_phase_4:
        grasp_success_rate_pct: 92.0
        terminal_placement_accuracy_cm: 0.8263749217987061
        placement_accuracy_std_cm: 0.8131163291457509
        terminal_error_cm: 1.5978229093551635
        terminal_error_std_cm: 0.8771730168266677
        mppi_inference_latency_ms: 36.57112133071043
        mean_joint_accel_rad_s2: 0.005506738461554051
        std_joint_accel_rad_s2: 0.0014232916122690269
        mean_joint_jerk_rad_s3: 0.0031630545668303966
        std_joint_jerk_rad_s3: 0.0006048513470103441
        mean_direction_flips: 12.08
        std_direction_flips: 5.388283585707048
        epoch: 240
    seed101:
      best_model:
        grasp_success_rate_pct: 56.00000000000001
        terminal_placement_accuracy_cm: 0.8067825222015381
        placement_accuracy_std_cm: 0.5102949840049316
        terminal_error_cm: 1.919658842086792
        terminal_error_std_cm: 0.9328204365318986
        mppi_inference_latency_ms: 36.6830046373979
        mean_joint_accel_rad_s2: 0.012233468238264322
        std_joint_accel_rad_s2: 0.004098371811678969
        mean_joint_jerk_rad_s3: 0.00957370424643159
        std_joint_jerk_rad_s3: 0.002600662905870819
        mean_direction_flips: 29.24
        std_direction_flips: 7.941183790846299
        epoch: 80
      checkpoint_phase_1:
        grasp_success_rate_pct: 68.0
        terminal_placement_accuracy_cm: 0.7755509388446807
        placement_accuracy_std_cm: 0.46954426710336217
        terminal_error_cm: 1.9073844707012177
        terminal_error_std_cm: 1.2624602195323336
        mppi_inference_latency_ms: 35.32790372871258
        mean_joint_accel_rad_s2: 0.012579807974398137
        std_joint_accel_rad_s2: 0.0041718596464833295
        mean_joint_jerk_rad_s3: 0.009853317625820637
        std_joint_jerk_rad_s3: 0.0027360135113209676
        mean_direction_flips: 28.64
        std_direction_flips: 8.831217356627567
        epoch: 96
      checkpoint_phase_2:
        grasp_success_rate_pct: 52.0
        terminal_placement_accuracy_cm: 0.9253814625740051
        placement_accuracy_std_cm: 0.4784543164925958
        terminal_error_cm: 2.3393139266967773
        terminal_error_std_cm: 1.2209142752391906
        mppi_inference_latency_ms: 33.997791576217644
        mean_joint_accel_rad_s2: 0.012314827740192413
        std_joint_accel_rad_s2: 0.003613214220178058
        mean_joint_jerk_rad_s3: 0.009852458033710718
        std_joint_jerk_rad_s3: 0.0023019908291644284
        mean_direction_flips: 28.08
        std_direction_flips: 8.776878716263544
        epoch: 104
      checkpoint_phase_3:
        grasp_success_rate_pct: 4.0
        terminal_placement_accuracy_cm: 1.0723411238193512
        placement_accuracy_std_cm: 0.5488965121484285
        terminal_error_cm: 6.585751514434815
        terminal_error_std_cm: 2.462142148479034
        mppi_inference_latency_ms: 36.89454996307587
        mean_joint_accel_rad_s2: 0.010207082200795413
        std_joint_accel_rad_s2: 0.0032440182882328065
        mean_joint_jerk_rad_s3: 0.00770574914291501
        std_joint_jerk_rad_s3: 0.0024250018933556217
        mean_direction_flips: 15.8
        std_direction_flips: 4.791659420284375
        epoch: 124
      checkpoint_phase_4:
        grasp_success_rate_pct: 40.0
        terminal_placement_accuracy_cm: 0.8932908475399017
        placement_accuracy_std_cm: 0.6142532524831947
        terminal_error_cm: 3.61430414557457
        terminal_error_std_cm: 2.0803425569119773
        mppi_inference_latency_ms: 36.891264794957294
        mean_joint_accel_rad_s2: 0.00655104667879641
        std_joint_accel_rad_s2: 0.0019870385418092346
        mean_joint_jerk_rad_s3: 0.004338146522641182
        std_joint_jerk_rad_s3: 0.0009688836318665361
        mean_direction_flips: 9.6
        std_direction_flips: 3.370459909270544
        epoch: 240
    seed2024:
      best_model:
        grasp_success_rate_pct: 48.0
        terminal_placement_accuracy_cm: 0.8991707092523575
        placement_accuracy_std_cm: 0.5571952553535476
        terminal_error_cm: 2.5209722006320954
        terminal_error_std_cm: 1.6852678641416425
        mppi_inference_latency_ms: 37.86354131541487
        mean_joint_accel_rad_s2: 0.008363267537206412
        std_joint_accel_rad_s2: 0.0036055991833148448
        mean_joint_jerk_rad_s3: 0.005904661854729056
        std_joint_jerk_rad_s3: 0.0027313948333358177
        mean_direction_flips: 16.16
        std_direction_flips: 3.170867389217026
        epoch: 130
      checkpoint_phase_1:
        grasp_success_rate_pct: 64.0
        terminal_placement_accuracy_cm: 0.9167939400672913
        placement_accuracy_std_cm: 0.4842631085247366
        terminal_error_cm: 1.8658612310886382
        terminal_error_std_cm: 0.892232074215814
        mppi_inference_latency_ms: 41.55131654470088
        mean_joint_accel_rad_s2: 0.01057371573522687
        std_joint_accel_rad_s2: 0.003922804958063201
        mean_joint_jerk_rad_s3: 0.007624266240745783
        std_joint_jerk_rad_s3: 0.002724674283801329
        mean_direction_flips: 20.88
        std_direction_flips: 5.171614834846076
        epoch: 94
      checkpoint_phase_2:
        grasp_success_rate_pct: 80.0
        terminal_placement_accuracy_cm: 0.8254767823219299
        placement_accuracy_std_cm: 0.45833934757821987
        terminal_error_cm: 1.7428334522247315
        terminal_error_std_cm: 0.9631726326989476
        mppi_inference_latency_ms: 42.998756124646725
        mean_joint_accel_rad_s2: 0.010881818272173405
        std_joint_accel_rad_s2: 0.004106524851980133
        mean_joint_jerk_rad_s3: 0.008174818065017462
        std_joint_jerk_rad_s3: 0.0030611518004082313
        mean_direction_flips: 23.8
        std_direction_flips: 4.7833042972405595
        epoch: 102
      checkpoint_phase_3:
        grasp_success_rate_pct: 40.0
        terminal_placement_accuracy_cm: 0.9588495105504989
        placement_accuracy_std_cm: 0.622048862001958
        terminal_error_cm: 2.971398718953133
        terminal_error_std_cm: 1.8832630559498067
        mppi_inference_latency_ms: 31.357012376353577
        mean_joint_accel_rad_s2: 0.008617250081151724
        std_joint_accel_rad_s2: 0.003343341246914032
        mean_joint_jerk_rad_s3: 0.006131271691992879
        std_joint_jerk_rad_s3: 0.002452792632865182
        mean_direction_flips: 17.12
        std_direction_flips: 3.9427908897125143
        epoch: 122
      checkpoint_phase_4:
        grasp_success_rate_pct: 60.0
        terminal_placement_accuracy_cm: 1.9300097036361694
        placement_accuracy_std_cm: 1.1646329528416253
        terminal_error_cm: 2.6945787906646728
        terminal_error_std_cm: 1.8902841139496236
        mppi_inference_latency_ms: 30.234106229985173
        mean_joint_accel_rad_s2: 0.005349645232781768
        std_joint_accel_rad_s2: 0.0018365806990176805
        mean_joint_jerk_rad_s3: 0.003066285639069974
        std_joint_jerk_rad_s3: 0.0009587316322348077
        mean_direction_flips: 9.44
        std_direction_flips: 3.990789395595814
        epoch: 260
  energy_summary:
    framing_note: SPWM's primary advantage over ANN-Rate is closed-loop dynamical stability (+23.4 pp PIL grasp success) and 1.0% sensory noise retention (+24.7 pp absolute, 2.0x better relative retention on N=25 val-selected checkpoints), not E2E energy savings (1.0979 mJ/step matches ANN-Rate; 48.14 nJ Loihi-2 core reduction is 0.004% of E2E power). The 4.94x Loihi-2 recurrent core reduction (12.22 +- 0.19 nJ/step, 0.244 mW @ 20 Hz, SOPs excluding passive leak) is an always-on standby state-estimation benefit when batched MPPI rollouts are dormant; the 1.93x E2E reduction vs. Continuous RSSM (2.1236 mJ/step) comes from M1's compact MLP rollout architecture.
    spwm_3seed:
      spike_rate_pct: [12.151117412707743,0.192861046198146]
      sops_per_step: [8150.0,129.11622671066561]
      energy_neuro_nj: [12.224996206577009,0.1940336413590309]
      power_20hz_mw: [0.24449992413154012,0.0038806728271806117]
      speedup_vs_own_gpu: [4.938654535978057,0.07910719976272344]
      e2e_step_energy_mj: [1.0978792841962064,1.9403364130818727e-07]
      e2e_power_20hz_mw: [21.957585683924133,3.880672826405402e-06]
    baselines:
      ANN-Rate:
        paradigm: Continuous ANN
        memory_type: Continuous Leaky Rate
        params: 392141
        spike_rate: 0.0
        sops_per_step: 0
        recurrent_macs: 67072
        recurrent_flops: 134144
        energy_neuro_nj: null
        energy_gpu_nj: 60.364799999999995
        energy_ratio_vs_gpu: 1.0
        power_20hz_mw: 1.207296
        cumulative_energy_h100_uj: 6.036479999999999
        encoder_macs: 882688
        encoder_energy_nj: 794.4191999999999
        mppi_rollout_macs: 1218969600
        mppi_rollout_energy_nj: 1097072.64
        e2e_step_energy_nj: 1097927.4239999999
        e2e_step_energy_mj: 1.0979274239999999
        e2e_all_gpu_energy_mj: 1.0979274239999999
        e2e_power_20hz_mw: 21.958548479999997
      Deep GRU:
        paradigm: Continuous ANN
        memory_type: 2-Layer Deep GRU
        params: 338868
        spike_rate: 0.0
        sops_per_step: 0
        recurrent_macs: 299520
        recurrent_flops: 599040
        energy_neuro_nj: null
        energy_gpu_nj: 269.568
        energy_ratio_vs_gpu: 1.0
        power_20hz_mw: 5.39136
        cumulative_energy_h100_uj: 26.956799999999998
        encoder_macs: 882688
        encoder_energy_nj: 794.4191999999999
        mppi_rollout_macs: 1942179840
        mppi_rollout_energy_nj: 1747961.856
        e2e_step_energy_nj: 1749025.8432
        e2e_step_energy_mj: 1.7490258432
        e2e_all_gpu_energy_mj: 1.7490258432
        e2e_power_20hz_mw: 34.980516864
      Continuous RSSM:
        paradigm: Continuous ANN
        memory_type: GRU + Gaussian Latent
        params: 380460
        spike_rate: 0.0
        sops_per_step: 0
        recurrent_macs: 365056
        recurrent_flops: 730112
        energy_neuro_nj: null
        energy_gpu_nj: 328.55039999999997
        energy_ratio_vs_gpu: 1.0
        power_20hz_mw: 6.571008
        cumulative_energy_h100_uj: 32.855039999999995
        encoder_macs: 882688
        encoder_energy_nj: 794.4191999999999
        mppi_rollout_macs: 2358312960
        mppi_rollout_energy_nj: 2122481.664
        e2e_step_energy_nj: 2123604.6336
        e2e_step_energy_mj: 2.1236046336
        e2e_all_gpu_energy_mj: 2.1236046336
        e2e_power_20hz_mw: 42.472092671999995
      Continuous ConvLSTM:
        paradigm: Continuous ANN
        memory_type: Spatial ConvLSTM (2D)
        params: 296892
        spike_rate: 0.0
        sops_per_step: 0
        recurrent_macs: 2359296
        recurrent_flops: 4718592
        energy_neuro_nj: null
        energy_gpu_nj: 2123.3664
        energy_ratio_vs_gpu: 1.0
        power_20hz_mw: 42.467327999999995
        cumulative_energy_h100_uj: 212.33664
        encoder_macs: 882688
        encoder_energy_nj: 794.4191999999999
        mppi_rollout_macs: 111311585280
        mppi_rollout_energy_nj: 100180426.752
        e2e_step_energy_nj: 100183344.53760001
        e2e_step_energy_mj: 100.18334453760001
        e2e_all_gpu_energy_mj: 100.18334453760001
        e2e_power_20hz_mw: 2003.6668907520004
  forensic_summary_3seed:
    Continuous RSSM Un-Gated (3-Seed Aggregate):
      total_episodes: 150
      success_total: 0
      reach_le_3cm_total: 144
      a_distance_failure_gt_3cm_total: 6
      b_velocity_failure_all_total: 150
      b_velocity_failure_given_reach_total: 144
      c_post_contact_impulse_divergence_total: 143
      excl_never_reached_total: 6
      excl_post_contact_div_plus_vel_total: 143
      excl_post_contact_div_spatial_only_total: 0
      excl_post_tc_reach_then_div_total: 1
      excl_vel_only_total: 0
      excl_rot_only_total: 0
    Continuous RSSM Residual-Gated (3-Seed Aggregate):
      total_episodes: 150
      success_total: 23
      reach_le_3cm_total: 148
      a_distance_failure_gt_3cm_total: 2
      b_velocity_failure_all_total: 24
      b_velocity_failure_given_reach_total: 24
      c_post_contact_impulse_divergence_total: 50
      excl_never_reached_total: 2
      excl_post_contact_div_plus_vel_total: 12
      excl_post_contact_div_spatial_only_total: 38
      excl_post_tc_reach_then_div_total: 22
      excl_vel_only_total: 3
      excl_rot_only_total: 50
    Deep GRU Un-Gated (3-Seed Aggregate):
      total_episodes: 150
      success_total: 0
      reach_le_3cm_total: 132
      a_distance_failure_gt_3cm_total: 18
      b_velocity_failure_all_total: 18
      b_velocity_failure_given_reach_total: 17
      c_post_contact_impulse_divergence_total: 123
      excl_never_reached_total: 18
      excl_post_contact_div_plus_vel_total: 17
      excl_post_contact_div_spatial_only_total: 106
      excl_post_tc_reach_then_div_total: 3
      excl_vel_only_total: 0
      excl_rot_only_total: 6
    Deep GRU Residual-Gated (3-Seed Aggregate):
      total_episodes: 150
      success_total: 115
      reach_le_3cm_total: 141
      a_distance_failure_gt_3cm_total: 9
      b_velocity_failure_all_total: 0
      b_velocity_failure_given_reach_total: 0
      c_post_contact_impulse_divergence_total: 9
      excl_never_reached_total: 9
      excl_post_contact_div_plus_vel_total: 0
      excl_post_contact_div_spatial_only_total: 9
      excl_post_tc_reach_then_div_total: 11
      excl_vel_only_total: 0
      excl_rot_only_total: 6
    ANN-Rate (3-Seed Aggregate):
      total_episodes: 150
      success_total: 56
      reach_le_3cm_total: 146
      a_distance_failure_gt_3cm_total: 4
      b_velocity_failure_all_total: 2
      b_velocity_failure_given_reach_total: 2
      c_post_contact_impulse_divergence_total: 56
      excl_never_reached_total: 4
      excl_post_contact_div_plus_vel_total: 0
      excl_post_contact_div_spatial_only_total: 56
      excl_post_tc_reach_then_div_total: 17
      excl_vel_only_total: 1
      excl_rot_only_total: 16
    SPWM N=25 Val-Sel (3-Seed Aggregate):
      total_episodes: 150
      success_total: 91
      reach_le_3cm_total: 148
      a_distance_failure_gt_3cm_total: 2
      b_velocity_failure_all_total: 2
      b_velocity_failure_given_reach_total: 2
      c_post_contact_impulse_divergence_total: 28
      excl_never_reached_total: 2
      excl_post_contact_div_plus_vel_total: 1
      excl_post_contact_div_spatial_only_total: 27
      excl_post_tc_reach_then_div_total: 6
      excl_vel_only_total: 0
      excl_rot_only_total: 23
```

### `results/mppi_benchmark/mppi_benchmark_metrics.json`
> 50-episode closed-loop Physics-in-the-Loop (PIL, t >= t_c) MPPI benchmark breakdown

```yaml
benchmark_name: SPWM Closed-Loop Latent MPPI Control Benchmark
evaluation_protocol: physics_in_loop
split: test
handover_dist_threshold: 0.026
checkpoint: results/spwm_v0_12_4_train_m1/model.pt
episodes_evaluated: 50
total_benchmark_time_sec: 56.54453133302741
overall_metrics:
  grasp_success_rate_pct: 86.0
  terminal_placement_accuracy_cm: 0.8552020224928856
  placement_accuracy_std_cm: 0.7991095820631844
  terminal_error_cm: 1.5668222349882126
  terminal_error_std_cm: 1.004804437956763
  mppi_inference_latency_ms: 36.65447710343201
  mean_joint_accel_rad_s2: 0.0058662799885496495
  std_joint_accel_rad_s2: 0.0014419035578661795
  mean_joint_jerk_rad_s3: 0.003542280187830329
  std_joint_jerk_rad_s3: 0.000713988545542795
  mean_direction_flips: 14.02
  std_direction_flips: 7.643271550847844
handover_telemetry:
  episodes_evaluated: 50
  triggered_episodes: 0
  episodes_with_w_gt_0_pct: 0.0
  total_phi1_steps: 400
  triggered_phi1_steps: 0
  phi1_steps_with_w_gt_0_pct: 0.0
  mean_nonzero_w: 0.0
elevation_plane_breakdown:
  Low: {count:22,success_rate_pct:86.36363636363636,mean_error_cm:0.855392383580858,mean_terminal_error_cm:1.9869136796756224,mean_mass_kg:2.7719357826493005}
  Mid: {count:18,success_rate_pct:77.77777777777779,mean_error_cm:1.054786902334955,mean_terminal_error_cm:1.3530784381760492,mean_mass_kg:2.8496671517690024}
  High: {count:10,success_rate_pct:100.0,mean_error_cm:0.4955304443836212,mean_terminal_error_cm:1.0273598909378052,mean_mass_kg:2.9106760144233705}
mass_tier_breakdown:
  Light (1.20 - 2.30 kg): {count:17,success_rate_pct:82.35294117647058,mean_error_cm:0.6789520838681389,mean_terminal_error_cm:1.643607579609927}
  Medium (2.30 - 3.40 kg): {count:17,success_rate_pct:100.0,mean_error_cm:0.767013340311892,mean_terminal_error_cm:1.453135742860682}
  Heavy (3.40 - 4.50 kg): {count:16,success_rate_pct:75.0,mean_error_cm:1.1361680570989847,mean_terminal_error_cm:1.6060297042131424}
episodes (50 rows):
  |trajectory_id|payload_mass|payload_shape|elevation_plane|target_pos|final_ee_pos|min_placement_error_cm|terminal_placement_error_cm|terminal_angular_velocity|grasp_success|mean_latency_ms|step_errors_cm|grasp_error_cm|grasp_rot_error_rad|mean_joint_accel_rad_s2|mean_joint_jerk_rad_s3|direction_flips|handover_weights_phi1|
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  |3600|3.0923237800598145|ElongatedCylinder|Low|[-0.1601915806531906,1.0277425050735474,0.44486987590789795]|[-0.181523859500885,1.0407003164291382,0.4354255497455597]|0.9916855096817017|1.9336272478103638|0.09670901298522949|true|37.85080052745928|[11.573441505432129,9.13671588897705,6.814445495605469,5.375359058380127,4.415168285369873,3.781996726989746,3.6314282417297363,3.5600736141204834,3.1390655040740967,2.3105342388153076,1.5440272092819214,1.337774634361267,1.178063154220581,1.0694260597229004,1.0108526945114136,0.9916855096817017,1.249696969985962,1.4435142278671265,1.4471330642700195,1.390917181968689,1.3779929876327515,1.089792251586914,1.4277375936508179,1.8794653415679932,2.150221586227417,2.137094736099243,2.005171298980713,1.9336272478103638]|1.3779929876327515|0.024531586095690727|0.005452719982713461|0.004464099183678627|24|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3601|1.7130316495895386|Box|Mid|[-0.14557956159114838,1.1094894409179688,0.4503576457500458]|[-0.1422460377216339,1.1097484827041626,0.46741801500320435]|0.506766676902771|0.6558789610862732|0.2623443901538849|true|36.46088522509672|[18.863054275512695,18.71589469909668,18.000974655151367,16.560466766357422,14.5841646194458,12.38664722442627,10.208590507507324,8.222845077514648,6.613028049468994,5.488822937011719,4.7486748695373535,4.199735164642334,3.70710825920105,3.214980363845825,2.6828536987304688,2.1142473220825195,1.5646533966064453,1.1199415922164917,0.9260068535804749,1.024008870124817,1.2364610433578491,1.3526791334152222,0.9986738562583923,0.680321455001831,0.5190710425376892,0.506766676902771,0.5487858057022095,0.6558789610862732]|1.2364610433578491|0.015533333644270897|0.005324265919625759|0.0024402805138379335|8|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3602|3.633739471435547|Box|High|[-0.1951867938041687,1.2046148777008057,0.48281943798065186]|[-0.21047256886959076,1.2045191526412964,0.48794427514076233]|0.5780963897705078|1.0662696361541748|0.18477372825145721|true|41.74411605345085|[13.0217866897583,10.909154891967773,8.175790786743164,5.087525844573975,2.2599239349365234,0.8492842316627502,0.9008960723876953,0.757885754108429,0.5780963897705078,0.6291033029556274,0.6766077876091003,0.7128912210464478,0.8593243360519409,0.989650547504425,0.9515424966812134,0.7987130284309387,0.6454941034317017,0.7472003102302551,0.9919666647911072,1.2199548482894897,1.4382659196853638,1.04286527633667,1.2014124393463135,1.2570909261703491,1.1720829010009766,1.056650161743164,1.0197794437408447,1.0662696361541748]|1.4382659196853638|0.01839362643659115|0.004433663096278906|0.003265181090682745|28|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3603|2.2544474601745605|Box|Low|[-0.1755688488483429,1.0101819038391113,0.41953858733177185]|[-0.19853876531124115,1.0139968395233154,0.4144792854785919]|0.30120307207107544|1.1931208372116089|0.1762886792421341|true|37.52263701919998|[20.206327438354492,19.067882537841797,17.597280502319336,15.646058082580566,13.301840782165527,10.84429931640625,8.568504333496094,6.6685004234313965,5.259915828704834,4.385437965393066,3.9030771255493164,3.529649257659912,3.1092278957366943,2.640129804611206,2.145613193511963,1.6746375560760498,1.267038345336914,0.966471791267395,0.8053648471832275,0.7555561661720276,0.7555692195892334,0.7496841549873352,0.30120307207107544,0.5870426297187805,1.0924853086471558,1.3483059406280518,1.3546491861343384,1.1931208372116089]|0.7555692195892334|0.024149538949131966|0.005527789704501629|0.0028964304365217686|10|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3604|4.175155162811279|Box|Mid|[-0.11803130805492401,1.1047005653381348,0.43481385707855225]|[-0.12391683459281921,1.0984159708023071,0.4646947979927063]|0.7615321278572083|0.7615321278572083|0.6631839275360107|true|35.27139569632709|[16.419307708740234,16.47688865661621,15.590681076049805,13.62572193145752,10.981175422668457,8.17818832397461,5.680799961090088,3.8853890895843506,2.8723678588867188,2.208507537841797,1.5882532596588135,1.0557125806808472,0.8434239625930786,3.9065706729888916,3.397132396697998,2.748361349105835,2.064880847930908,1.4650567770004272,1.0028752088546753,0.7615321278572083]|0.8434239625930786|0.06260932981967926|0.00853119045495987|0.004648132715374231|6|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3605|2.795863151550293|Sphere|High|[-0.1699182093143463,1.1916710138320923,0.4913409948348999]|[-0.18167349696159363,1.1891207695007324,0.49705737829208374]|0.5579374432563782|0.6727336645126343|0.07888572663068771|true|35.35138541233859|[16.62736701965332,14.852818489074707,12.7636137008667,10.363534927368164,7.559988498687744,4.73409366607666,2.6688780784606934,1.5254493951797485,1.013270616531372,1.049547553062439,1.119684100151062,1.0351685285568237,0.9366130232810974,0.9422999620437622,0.792514979839325,0.5579374432563782,0.6398530602455139,0.8698168992996216,0.9959698915481567,0.9806293249130249,1.004894495010376,1.0207127332687378,0.9610164165496826,0.7606266140937805,0.5927325487136841,0.6301759481430054,0.7024286389350891,0.6727336645126343]|1.004894495010376|0.016024427488446236|0.004445232450962067|0.0036578206345438957|25|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3606|1.416571021080017|Sphere|Low|[-0.1465684324502945,1.022121548652649,0.42273831367492676]|[-0.18414996564388275,1.0300668478012085,0.42452389001846313]|0.9793704152107239|2.7034308910369873|0.3070491552352905|false|39.55251928917797|[22.074623107910156,21.98685073852539,21.067712783813477,19.142045974731445,16.49403953552246,13.516757011413574,10.508341789245605,7.695045471191406,5.307694435119629,3.533613681793213,2.4142346382141113,1.7754491567611694,1.3944028615951538,1.1556730270385742,1.021419644355774,0.9793704152107239,1.023956060409546,1.1146032810211182,1.2350940704345703,1.3911460638046265,1.6232742071151733,2.5722403526306152,2.5768916606903076,2.7226579189300537,2.9161627292633057,3.0095701217651367,2.924699306488037,2.7034308910369873]|1.6232742071151733|0.05123559385538101|0.006821285467594862|0.0029433222953230143|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3607|3.3372788429260254|Sphere|Low|[-0.18835283815860748,1.0335981845855713,0.4100155532360077]|[-0.1967482566833496,1.045570731163025,0.4280588924884796]|0.8437272310256958|2.5115978717803955|0.3293541669845581|true|36.09264730143228|[11.552645683288574,10.05626392364502,8.145291328430176,6.236244201660156,4.6283979415893555,3.410289764404297,2.563709020614624,1.993027925491333,1.5858583450317383,1.3744828701019287,1.4772422313690186,1.6358176469802856,1.671844720840454,1.593384861946106,1.4417979717254639,1.292467713356018,1.21205472946167,1.1391667127609253,1.0256832838058472,0.9042754769325256,0.8437272310256958,1.2209042310714722,1.323520541191101,1.4906668663024902,1.6714000701904297,1.8630757331848145,2.128577470779419,2.5115978717803955]|0.8437272310256958|0.024257903918623924|0.003289909800514579|0.002672412432730198|18|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3608|1.957986831665039|Sphere|Mid|[-0.17253130674362183,1.1075801849365234,0.4440624713897705]|[-0.1953900009393692,1.0996379852294922,0.4503025412559509]|0.5789094567298889|1.174257755279541|0.20306235551834106|true|35.81420834442334|[20.210655212402344,20.13730812072754,19.567161560058594,18.173227310180664,16.056732177734375,13.573919296264648,11.073359489440918,8.819870948791504,7.090324878692627,5.957820415496826,5.183243274688721,4.517940998077393,3.84212064743042,3.1134719848632812,2.3700008392333984,1.633023977279663,0.9585938453674316,0.5789094567298889,0.7325471639633179,0.9477258920669556,1.1634713411331177,1.8663829565048218,1.5532097816467285,1.2724469900131226,1.1728980541229248,1.275977611541748,1.3527140617370605,1.174257755279541]|1.1634713411331177|0.020162278786301613|0.005878619849681854|0.0031695489306002855|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3609|3.878694772720337|ElongatedCylinder|Mid|[-0.062913678586483,1.148561954498291,0.44581520557403564]|[-0.16867566108703613,1.1847368478775024,0.4695170521736145]|2.7950124740600586|2.7950124740600586|1.27104651927948|false|37.432494733366184|[11.491044998168945,11.323838233947754,10.423653602600098,8.758676528930664,6.829644680023193,5.491884231567383,5.2949981689453125,5.697611331939697,6.236111640930176,11.8062162399292,10.396753311157227,8.782577514648438,7.050666332244873,5.391898155212402,3.9452381134033203,2.7950124740600586]|6.236111640930176|0.12469201534986496|0.007096568122506142|0.004245960619300604|2|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3610|2.4994025230407715|ElongatedCylinder|Low|[-0.17838016152381897,1.0129873752593994,0.4401816427707672]|[-0.20216695964336395,1.0263774394989014,0.422979474067688]|1.877776861190796|2.783543348312378|0.07825074344873428|true|37.31123811615232|[18.435977935791016,16.922863006591797,15.10299015045166,12.972124099731445,10.614072799682617,8.201216697692871,5.934650897979736,4.027065277099609,2.677015781402588,1.9750359058380127,1.877776861190796,2.0940470695495605,2.284146547317505,2.3391668796539307,2.2694313526153564,2.2057278156280518,2.3101887702941895,2.5440399646759033,2.833667755126953,2.984696388244629,3.1289186477661133,3.0692648887634277,3.2259013652801514,3.1816940307617188,2.989564895629883,2.83784818649292,2.78542423248291,2.783543348312378]|3.1289186477661133|0.015867428854107857|0.005322796758264303|0.0031827620696276426|11|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3611|4.42011022567749|ElongatedCylinder|Mid|[-0.14028692245483398,1.1213961839675903,0.4443739652633667]|[-0.1954422891139984,1.1178123950958252,0.45919543504714966]|0.9093067646026611|1.2214659452438354|0.5523556470870972|true|34.9572480848545|[18.295475006103516,18.227298736572266,17.291290283203125,15.353883743286133,12.755929946899414,10.001691818237305,7.515650272369385,5.640676021575928,4.5070624351501465,3.7491605281829834,2.9990320205688477,2.255490779876709,1.6379972696304321,1.2188856601715088,0.9093067646026611,0.9962462782859802,3.5999107360839844,2.764124631881714,1.9871240854263306,1.4203698635101318,1.0682471990585327,0.964020848274231,1.2214659452438354]|0.9962462782859802|0.0162387415766716|0.008901944383978844|0.004968914669007063|9|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3612|3.040818452835083|Box|High|[-0.20262062549591064,1.2052944898605347,0.480135053396225]|[-0.21779294312000275,1.21334707736969,0.4896395802497864]|0.5043447613716125|1.4388676881790161|0.1696101725101471|true|36.59438390085207|[17.265993118286133,14.611112594604492,11.49609375,8.478445053100586,5.986507415771484,4.340770244598389,3.3893299102783203,2.8372721672058105,2.4031970500946045,2.125366687774658,1.9125895500183105,1.637894868850708,1.2815426588058472,0.9825573563575745,0.8239065408706665,0.7885252237319946,0.5043447613716125,0.5407300591468811,0.6841803193092346,0.7782511711120605,0.8721257448196411,1.0638645887374878,0.8714159727096558,0.8670576810836792,0.9904080033302307,1.1552531719207764,1.3246654272079468,1.4388676881790161]|0.8721257448196411|0.023443123325705528|0.004613562952727079|0.00382964126765728|23|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3613|1.6615263223648071|Box|Low|[-0.16457204520702362,1.0336177349090576,0.4418622851371765]|[-0.18708594143390656,1.0427496433258057,0.443037748336792]|1.3211021423339844|3.6936566829681396|0.21231871843338013|false|37.78940321561614|[22.635013580322266,21.666013717651367,20.228515625,18.30913734436035,15.96351432800293,13.332262992858887,10.606343269348145,7.95735502243042,5.548788070678711,3.6008517742156982,2.279406785964966,1.5705217123031616,1.3211021423339844,1.3404786586761475,1.4598009586334229,1.584273338317871,1.7173569202423096,1.8706848621368408,2.0681674480438232,2.2601828575134277,2.3963098526000977,3.829526424407959,3.7018563747406006,3.600928544998169,3.5688180923461914,3.5829193592071533,3.631002187728882,3.6936566829681396]|2.3963098526000977|0.032908886671066284|0.005970076657831669|0.0028146791737526655|11|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3614|3.5822341442108154|Box|Low|[-0.16669978201389313,1.0320783853530884,0.43108421564102173]|[-0.17755067348480225,1.0514410734176636,0.43648266792297363]|0.25326672196388245|2.251241683959961|0.3407454192638397|true|34.611700888490304|[8.943724632263184,6.457819938659668,4.258119106292725,2.7879974842071533,1.6090335845947266,0.657162606716156,0.25326672196388245,0.5613920092582703,1.0396243333816528,1.3477591276168823,1.2966938018798828,1.098821997642517,0.9724741578102112,0.9122208952903748,0.8851869702339172,0.8754488825798035,0.9920758008956909,1.122313141822815,1.0916647911071777,1.087362289428711,1.1220457553863525,1.0817625522613525,1.0170663595199585,0.9720361828804016,1.0801496505737305,1.369455337524414,1.8080843687057495,2.251241683959961]|1.1220457553863525|0.012118896469473839|0.004202086012810469|0.0032998882234096527|29|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3615|2.202942132949829|Box|Mid|[-0.12205535918474197,1.0982131958007812,0.4642801880836487]|[-0.11232805252075195,1.0891683101654053,0.48184800148010254]|0.8049491047859192|1.1089707612991333|0.14178039133548737|true|36.96852971084549|[15.237411499023438,14.689679145812988,13.532974243164062,11.798145294189453,9.836645126342773,8.019935607910156,6.638298988342285,5.810934543609619,5.392032623291016,5.103555679321289,4.753120422363281,4.272016525268555,3.674802541732788,2.97409725189209,2.1504015922546387,1.3033897876739502,0.8049491047859192,1.1342395544052124,1.6589303016662598,2.0108494758605957,2.185743570327759,2.380646228790283,1.8546696901321411,1.3684314489364624,1.1160258054733276,1.0192497968673706,1.0118829011917114,1.1089707612991333]|2.185743570327759|0.024030763655900955|0.005630700848996639|0.0027866153977811337|8|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3616|4.123650074005127|Sphere|Mid|[-0.10549936443567276,1.1719096899032593,0.4632115066051483]|[-0.15461504459381104,1.1856918334960938,0.4917733669281006]|1.5770885944366455|1.5770885944366455|0.9466993808746338|false|38.165223680583665|[16.050519943237305,15.976490020751953,15.13637638092041,13.470935821533203,11.436126708984375,9.652623176574707,8.465885162353516,7.745002269744873,7.039708614349365,6.082975387573242,4.872997283935547,3.6610443592071533,7.939085483551025,6.98851203918457,5.904697418212891,4.721248149871826,3.5598673820495605,2.4989805221557617,1.5770885944366455]|3.6610443592071533|0.08423466235399246|0.00812214333564043|0.004782377276569605|4|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3617|2.7443578243255615|Sphere|Low|[-0.167842298746109,1.0162646770477295,0.4412359595298767]|[-0.18875917792320251,1.0354089736938477,0.4323013424873352]|1.593156337738037|2.6905677318573|0.17327050864696503|true|35.567986708234194|[15.525907516479492,13.69143009185791,11.748913764953613,9.79043197631836,7.812884330749512,5.794358730316162,3.8684420585632324,2.4004430770874023,1.6821708679199219,1.593156337738037,1.7684998512268066,2.0129566192626953,2.1663718223571777,2.143958330154419,2.0554094314575195,2.0646297931671143,2.1620230674743652,2.282470703125,2.4374706745147705,2.5553109645843506,2.5426084995269775,2.327301025390625,2.236995220184326,2.28572416305542,2.4575271606445312,2.673576831817627,2.7628209590911865,2.6905677318573]|2.5426084995269775|0.0140927629545331|0.004870283883064985|0.003284703241661191|17|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3618|1.3650658130645752|Sphere|Mid|[-0.14363370835781097,1.0971513986587524,0.43820515275001526]|[-0.1776869148015976,1.083391785621643,0.4438779354095459]|0.2804223299026489|1.309301733970642|0.19441042840480804|true|35.93278040912432|[18.96586036682129,18.994182586669922,18.124658584594727,16.225196838378906,13.667381286621094,10.945236206054688,8.417349815368652,6.339962959289551,4.899565696716309,4.044349193572998,3.4539718627929688,2.899522542953491,2.314434051513672,1.7337180376052856,1.3081575632095337,1.001285433769226,0.61441570520401,0.2804223299026489,0.6859754323959351,1.9977960586547852,1.3402527570724487,0.9435201287269592,1.047229528427124,1.2624526023864746,1.3500157594680786,1.309301733970642]|0.6859754323959351|0.05123326554894447|0.007750516291707754|0.00392241170629859|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3619|3.285773515701294|Sphere|High|[-0.1843661069869995,1.2083699703216553,0.49158498644828796]|[-0.19406375288963318,1.2120311260223389,0.4977548122406006]|0.23154476284980774|1.0162675380706787|0.3529472351074219|true|37.60198823459047|[14.836817741394043,12.165616035461426,9.24910831451416,6.438962459564209,3.843275547027588,1.7284754514694214,0.726525068283081,1.2850594520568848,1.8167084455490112,2.115046501159668,2.2320969104766846,2.3749582767486572,2.3783395290374756,2.012259006500244,1.519439697265625,1.1090201139450073,0.715383768081665,0.6320114135742188,0.8577454090118408,0.6961470246315002,0.36497995257377625,0.23154476284980774,0.33703991770744324,0.2613683342933655,0.2697426974773407,0.46811261773109436,0.7332866787910461,1.0162675380706787]|0.36497995257377625|0.02564825303852558|0.004813095089048147|0.0038277339190244675|24|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3620|1.9064816236495972|ElongatedCylinder|Low|[-0.15938085317611694,1.0115280151367188,0.4201980531215668]|[-0.20352128148078918,1.029079556465149,0.43916359543800354]|0.4718458354473114|3.86948299407959|0.18680691719055176|false|36.389944937712116|[22.682846069335938,22.249004364013672,21.29085350036621,19.63762855529785,17.446985244750977,14.978410720825195,12.485459327697754,10.159170150756836,8.12783432006836,6.48287296295166,5.24565315246582,4.3107991218566895,3.539684295654297,2.8321807384490967,2.164886713027954,1.5560272932052612,1.0373705625534058,0.630649209022522,0.4718458354473114,0.7249923944473267,1.2667677402496338,1.4220088720321655,1.993300437927246,2.6451690196990967,3.194894313812256,3.546635150909424,3.748642921447754,3.86948299407959]|1.2667677402496338|0.0557076595723629|0.00595518946647644|0.002554785693064332|6|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3621|3.8271894454956055|ElongatedCylinder|Low|[-0.1774565875530243,1.0358524322509766,0.42764782905578613]|[-0.18887236714363098,1.050490140914917,0.4359287917613983]|0.3817315399646759|2.090648651123047|0.25309044122695923|true|36.97099707538395|[8.642985343933105,6.75567626953125,4.741387367248535,3.059702157974243,1.7649656534194946,0.8298937082290649,0.49204501509666443,0.509000301361084,0.561610758304596,0.6736430525779724,0.8548423647880554,1.0265390872955322,0.9770238995552063,0.7715215682983398,0.9006391167640686,0.9879025816917419,0.7482883334159851,0.49399620294570923,0.3817315399646759,0.4170243740081787,0.5321795344352722,0.8061192035675049,0.9784588813781738,1.2111084461212158,1.4901573657989502,1.730722427368164,1.8969652652740479,2.090648651123047]|0.5321795344352722|0.01997813768684864|0.00431095901876688|0.0032779478933662176|31|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3622|2.447897434234619|ElongatedCylinder|Mid|[-0.17179328203201294,1.1022673845291138,0.44678300619125366]|[-0.19449082016944885,1.1043334007263184,0.4525398015975952]|0.726320743560791|1.1849288940429688|0.2717391550540924|true|35.011696288295624|[16.82199478149414,15.541854858398438,13.890707015991211,11.913989067077637,9.72465705871582,7.542094707489014,5.628571033477783,4.102341175079346,2.9960925579071045,2.155912160873413,1.4741482734680176,1.0455098152160645,0.8131979703903198,0.726320743560791,0.8358361124992371,1.1714401245117188,1.55228590965271,1.8570122718811035,2.034912586212158,2.142489194869995,2.2720539569854736,2.5594186782836914,2.6303365230560303,2.579683303833008,2.2707765102386475,1.8033210039138794,1.4140961170196533,1.1849288940429688]|2.2720539569854736|0.027213485911488533|0.005534427706152201|0.0036903945729136467|11|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3623|4.368605136871338|ElongatedCylinder|High|[-0.1286347657442093,1.184536099433899,0.4736163020133972]|[-0.18811526894569397,1.202555775642395,0.4871509075164795]|0.24395236372947693|0.24395236372947693|0.24362404644489288|true|37.505466041197494|[18.77583122253418,18.723112106323242,17.885498046875,16.213924407958984,14.154230117797852,12.184904098510742,10.590035438537598,9.423843383789062,8.515920639038086,7.59041166305542,6.530089855194092,5.389125823974609,4.240482330322266,3.0955276489257812,2.0136680603027344,6.165989398956299,4.705014228820801,3.1967461109161377,1.8611623048782349,0.9207494258880615,0.44111916422843933,0.24395236372947693]|2.0136680603027344|0.010961941443383694|0.00837490800768137|0.004784216172993183|7|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3624|2.9893131256103516|Box|Low|[-0.11366361379623413,1.0136218070983887,0.43174314498901367]|[-0.10725440829992294,1.005520224571228,0.4294309616088867]|0.6164560317993164|0.6164560317993164|0.3290953040122986|true|36.89854754234797|[10.494338035583496,9.256561279296875,7.613646984100342,5.67695951461792,3.76519775390625,2.2274856567382812,1.480208396911621,1.484779715538025,1.4586421251296997,1.1572813987731934,0.766203761100769,0.6457382440567017,0.8564169406890869,1.0558832883834839,1.1180847883224487,1.0153495073318481,0.8240696787834167,0.7575821876525879,0.9162877798080444,1.120628833770752,1.2076910734176636,2.143428087234497,1.6191837787628174,1.0302115678787231,0.8530060052871704,0.9111985564231873,0.7762708067893982,0.6164560317993164]|1.2076910734176636|0.031411100178956985|0.004940117243677378|0.0033501291181892157|22|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3625|1.6100211143493652|Box|Mid|[-0.15163391828536987,1.1202882528305054,0.44957059621810913]|[-0.18571840226650238,1.1247386932373047,0.45394933223724365]|0.5093672871589661|1.1884957551956177|0.17524683475494385|true|36.16864304058254|[19.41160774230957,19.184282302856445,18.204917907714844,16.385005950927734,14.035934448242188,11.550073623657227,9.241286277770996,7.3553571701049805,6.083436012268066,5.2912139892578125,4.68248176574707,4.0826849937438965,3.4779255390167236,2.8884329795837402,2.3411431312561035,1.8529651165008545,1.4121062755584717,0.9527010917663574,0.5093672871589661,0.575826108455658,1.1319252252578735,1.320868730545044,0.9260712265968323,0.6734710931777954,0.6484938859939575,0.8177240490913391,1.0385916233062744,1.1884957551956177]|1.1319252252578735|0.043823856860399246|0.006040005944669247|0.002965521765872836|11|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3626|3.530728816986084|Box|High|[-0.1873343586921692,1.2099320888519287,0.4893725514411926]|[-0.19845989346504211,1.2147208452224731,0.4983998239040375]|0.5240110754966736|1.1512185335159302|0.23987576365470886|true|36.50462951294945|[14.32147216796875,12.501469612121582,10.044136047363281,7.205102443695068,4.707459926605225,3.048065185546875,2.06121563911438,1.341864824295044,0.560186505317688,0.5240110754966736,0.8561111092567444,1.1277230978012085,1.1553541421890259,1.0414628982543945,0.9279810786247253,0.7972808480262756,0.6980960965156555,0.6170682907104492,0.6240937113761902,0.645821213722229,0.6820462942123413,0.6744535565376282,0.6788455247879028,0.7289376258850098,0.7893009185791016,0.8636918067932129,0.9657697081565857,1.1512185335159302]|0.6820462942123413|0.03351915255188942|0.004448932595551014|0.003769020549952984|23|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3627|2.1514368057250977|Box|Low|[-0.17351315915584564,1.024309515953064,0.426189661026001]|[-0.20382976531982422,1.0255351066589355,0.4227962791919708]|0.27199679613113403|1.5214118957519531|0.24358859658241272|true|36.83903416718489|[21.4793643951416,20.674772262573242,19.59881019592285,17.957653045654297,15.753443717956543,13.286334037780762,10.895737648010254,8.78799819946289,7.059591293334961,5.740137100219727,4.7783098220825195,4.027419567108154,3.347012758255005,2.6467018127441406,1.9523911476135254,1.3258517980575562,0.79252028465271,0.3476771116256714,0.27199679613113403,0.6811124086380005,1.0963740348815918,1.0882114171981812,1.0155190229415894,1.0543897151947021,1.1822434663772583,1.342909812927246,1.4852806329727173,1.5214118957519531]|1.0963740348815918|0.03250057250261307|0.0059038507752120495|0.002812380902469158|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3628|4.072144985198975|Sphere|Low|[-0.10735169053077698,1.0578010082244873,0.41456565260887146]|[-0.1736275553703308,1.0184121131896973,0.41274935007095337]|0.4293580949306488|0.4293580949306488|0.6511563062667847|true|36.00941199369522|[16.863754272460938,16.969100952148438,16.19860076904297,14.320588111877441,11.617219924926758,8.539579391479492,5.49940824508667,2.8723387718200684,1.362320899963379,2.011624574661255,3.2131540775299072,3.8280110359191895,2.873960018157959,2.192607879638672,1.747725486755371,1.3480571508407593,0.8919717669487,0.4293580949306488]|3.2131540775299072|0.18120460212230682|0.007587549742311239|0.004643850959837437|6|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3629|2.692852735519409|Sphere|Mid|[-0.16786649823188782,1.1062926054000854,0.44768890738487244]|[-0.17676395177841187,1.1037216186523438,0.45157766342163086]|0.2780361771583557|0.5643303394317627|0.21355482935905457|true|35.70231097234812|[15.108315467834473,13.863889694213867,12.259894371032715,10.33352279663086,8.217118263244629,6.16634464263916,4.549056053161621,3.618645668029785,3.105990409851074,2.545013189315796,1.7088111639022827,0.8749033212661743,0.7857321500778198,1.1200131177902222,1.2223923206329346,1.1335296630859375,1.092045545578003,1.2129268646240234,1.2728813886642456,1.0829355716705322,0.8737503290176392,1.065178632736206,0.7113313674926758,0.42401254177093506,0.2780361771583557,0.3038209080696106,0.4257553815841675,0.5643303394317627]|0.8737503290176392|0.026878463104367256|0.005157529376447201|0.003597099334001541|18|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3630|1.3135606050491333|Sphere|High|[-0.14967475831508636,1.1902045011520386,0.4741608798503876]|[-0.18765167891979218,1.1900206804275513,0.48302626609802246]|0.7940020561218262|1.3787058591842651|0.32437649369239807|true|38.212206810712814|[20.57914924621582,20.554241180419922,19.798397064208984,18.202659606933594,16.102378845214844,13.914566040039062,11.919513702392578,10.236549377441406,8.882159233093262,7.760317802429199,6.665537357330322,5.523362159729004,4.399424076080322,3.349357843399048,2.385547161102295,1.48786199092865,0.7940020561218262,0.7993075251579285,2.8225419521331787,2.3018763065338135,1.8295098543167114,1.4806870222091675,1.3426263332366943,1.3508507013320923,1.3787058591842651]|0.7993075251579285|0.015231078490614891|0.007239998318254948|0.003844128455966711|8|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3631|3.2342684268951416|Sphere|Low|[-0.1634715497493744,1.038787603378296,0.4216647148132324]|[-0.16901221871376038,1.051634669303894,0.43485012650489807]|0.5522339940071106|2.192549705505371|0.2343723624944687|true|35.38596286255467|[12.116371154785156,11.2184419631958,9.626860618591309,7.558220863342285,5.372666358947754,3.385794162750244,2.0066590309143066,1.4076454639434814,1.2972474098205566,1.3060187101364136,1.3613369464874268,1.4632114171981812,1.4641536474227905,1.3204423189163208,1.1247540712356567,0.9767554998397827,0.8764401078224182,0.7808994650840759,0.6803053021430969,0.5949706435203552,0.5522339940071106,1.058268666267395,1.153839349746704,1.297049641609192,1.404298186302185,1.5512619018554688,1.837510585784912,2.192549705505371]|0.5522339940071106|0.032261237502098083|0.004817571956664324|0.003316606394946575|19|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3632|1.8549762964248657|ElongatedCylinder|Mid|[-0.14753121137619019,1.0955888032913208,0.451601505279541]|[-0.1812039464712143,1.1048529148101807,0.45963841676712036]|0.7968575954437256|1.3177038431167603|0.20399363338947296|true|35.66230360801065|[19.342063903808594,18.983379364013672,18.054981231689453,16.45620346069336,14.414267539978027,12.255228042602539,10.253486633300781,8.632150650024414,7.486773490905762,6.76416015625,6.307520866394043,5.925041675567627,5.509479522705078,4.997115135192871,4.3656792640686035,3.6312711238861084,2.85254168510437,2.1179962158203125,1.4718070030212402,0.8772793412208557,0.7968575954437256,1.6057676076889038,1.335113286972046,1.3509929180145264,1.3890559673309326,1.3941981792449951,1.3731639385223389,1.3177038431167603]|0.7968575954437256|0.031021516770124435|0.005859631113708019|0.0027889711782336235|9|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3633|3.775684118270874|ElongatedCylinder|High|[-0.18505986034870148,1.2041765451431274,0.4869106113910675]|[-0.19260895252227783,1.2084227800369263,0.4974644184112549]|0.7855739593505859|1.1281824111938477|0.16408735513687134|true|38.46556856712726|[12.669571876525879,10.255179405212402,7.147441864013672,3.8521337509155273,1.366694450378418,1.4371938705444336,2.0550882816314697,1.832348346710205,1.2951494455337524,1.1186015605926514,1.186187505722046,1.4207302331924438,1.9344004392623901,2.133354663848877,1.9762160778045654,1.6909351348876953,1.4376757144927979,1.2607914209365845,1.2375408411026,1.247198224067688,1.2034505605697632,0.8263468742370605,0.7855739593505859,0.7911210060119629,0.8397747874259949,0.9397642016410828,1.0418787002563477,1.1281824111938477]|1.2034505605697632|0.022108323872089386|0.004672179464250803|0.003283898113295436|33|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3634|2.3963921070098877|ElongatedCylinder|Low|[-0.1695619374513626,1.008594274520874,0.4257681965827942]|[-0.1913250833749771,1.0218191146850586,0.4128127098083496]|1.3192967176437378|2.388773202896118|0.21397584676742554|true|35.25742107220659|[19.755704879760742,18.304271697998047,16.583255767822266,14.52197551727295,12.138203620910645,9.580709457397461,7.1021952629089355,4.950356483459473,3.298713207244873,2.1986145973205566,1.5662553310394287,1.3192967176437378,1.4072436094284058,1.6220219135284424,1.7744224071502686,1.80570650100708,1.8297003507614136,1.9185863733291626,2.0446410179138184,2.182018280029297,2.3400394916534424,2.3438832759857178,2.4185314178466797,2.433094024658203,2.3778717517852783,2.3170011043548584,2.32934308052063,2.388773202896118]|2.3400394916534424|0.02148478850722313|0.005335341673344374|0.0030782262329012156|9|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3635|4.3171000480651855|ElongatedCylinder|Low|[-0.12416441738605499,1.0549348592758179,0.42791247367858887]|[-0.17528286576271057,1.0130916833877563,0.42876237630844116]|1.0889581441879272|1.0889581441879272|0.7931868433952332|true|35.85717918467708|[18.502355575561523,18.27083396911621,17.270139694213867,15.337618827819824,12.718391418457031,9.789525985717773,6.928038597106934,4.494130611419678,2.9154772758483887,2.4798803329467773,2.6673948764801025,2.933116912841797,3.2401137351989746,3.3816239833831787,2.6891512870788574,2.142812967300415,1.8799694776535034,1.7621996402740479,1.5325089693069458,1.0889581441879272]|3.2401137351989746|0.13765671849250793|0.007390642538666725|0.0040932828560471535|6|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3636|2.937808036804199|Box|Mid|[-0.1312476545572281,1.0861977338790894,0.45727676153182983]|[-0.14555317163467407,1.0684548616409302,0.45085036754608154]|0.2914676070213318|0.6492407321929932|0.2071605771780014|true|37.23876329188767|[10.31956958770752,8.802988052368164,6.940525054931641,5.064373016357422,3.359471321105957,1.8927263021469116,0.7804811000823975,0.2914676070213318,0.6138421297073364,0.8282315135002136,0.9533159732818604,1.0607719421386719,1.0664030313491821,0.9736469984054565,0.9433544874191284,0.9863813519477844,0.9851071238517761,0.8984163403511047,0.7281327843666077,0.5963709950447083,0.6774990558624268,2.8396880626678467,2.403714418411255,1.9324673414230347,1.4848918914794922,1.098495602607727,0.8102396726608276,0.6492407321929932]|0.6774990558624268|0.05991054326295853|0.004092748276889324|0.0029092789627611637|19|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3637|1.5585157871246338|Box|High|[-0.1817418932914734,1.191416621208191,0.4692378342151642]|[-0.23025259375572205,1.1904958486557007,0.4722398817539215]|0.40775224566459656|1.6980962753295898|0.2553197145462036|true|35.976718687119764|[23.194746017456055,23.1240234375,22.250764846801758,20.465267181396484,18.07330322265625,15.529826164245605,13.189214706420898,11.199926376342773,9.52406120300293,8.038626670837402,6.672653675079346,5.454441070556641,4.39469051361084,3.4986720085144043,2.740023374557495,1.9397814273834229,1.0336793661117554,0.40775224566459656,1.037522792816162,1.7057992219924927,2.211172580718994,1.4629265069961548,0.9825827479362488,0.4543554484844208,0.4402349293231964,0.9306989908218384,1.3699051141738892,1.6980962753295898]|2.211172580718994|0.017242833971977234|0.0060327318497002125|0.002987214829772711|10|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3638|3.4792237281799316|Box|Low|[-0.16325247287750244,1.0150082111358643,0.4373423457145691]|[-0.1809995174407959,1.0283489227294922,0.43002769351005554]|1.1486928462982178|2.1363041400909424|0.1614849865436554|true|35.30202989350073|[10.394128799438477,8.186164855957031,6.03852653503418,4.224868297576904,2.7496955394744873,1.653647780418396,1.1486928462982178,1.2467883825302124,1.5465056896209717,1.6987943649291992,1.764533281326294,1.8028564453125,1.690259337425232,1.4747930765151978,1.3749431371688843,1.4346140623092651,1.5223978757858276,1.6193041801452637,1.750245451927185,1.9481781721115112,2.1063692569732666,2.0408568382263184,1.9368743896484375,1.9465082883834839,2.0731067657470703,2.136549711227417,2.1330487728118896,2.1363041400909424]|2.1063692569732666|0.00534885935485363|0.00390033470466733|0.0029892371967434883|23|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3639|2.0999317169189453|Box|Mid|[-0.12953197956085205,1.1047871112823486,0.46129995584487915]|[-0.12611684203147888,1.1035633087158203,0.4736282229423523]|0.19728639721870422|0.5408960580825806|0.23843805491924286|true|37.302351298941566|[16.882518768310547,16.298480987548828,15.300130844116211,13.768312454223633,11.823846817016602,9.698659896850586,7.67355489730835,6.075736999511719,5.153343677520752,4.747296333312988,4.464205741882324,4.098345756530762,3.647346019744873,3.1023693084716797,2.4813685417175293,1.8571218252182007,1.3726853132247925,1.1176477670669556,1.0671550035476685,1.1756161451339722,1.3043283224105835,1.7461892366409302,1.2606699466705322,0.7163934707641602,0.3264453113079071,0.19728639721870422,0.28514009714126587,0.5408960580825806]|1.3043283224105835|0.018619094043970108|0.005742546170949936|0.003130862256512046|10|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3640|4.020639419555664|Sphere|Mid|[-0.07282336056232452,1.155226469039917,0.44457247853279114]|[-0.13655848801136017,1.1784952878952026,0.4775695204734802]|5.1237664222717285|5.457233428955078|0.9036175012588501|false|37.418770989461336|[12.106219291687012,11.887073516845703,10.927496910095215,9.245451927185059,7.400056838989258,6.055064678192139,5.411217212677002,5.1237664222717285,5.197731971740723,11.007996559143066,9.174333572387695,7.583587646484375,6.424367427825928,5.806436538696289,5.610735893249512,5.457233428955078]|5.197731971740723|0.12554411590099335|0.00803210586309433|0.0053236945532262325|5|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3641|2.6413474082946777|Sphere|Low|[-0.1712183803319931,1.0232043266296387,0.41830378770828247]|[-0.18267711997032166,1.0249611139297485,0.41108906269073486]|0.8129216432571411|1.002864956855774|0.1469576060771942|true|34.69635267642194|[17.45759391784668,16.137500762939453,14.335711479187012,12.083817481994629,9.625020980834961,7.259502410888672,5.193816661834717,3.6148576736450195,2.6363444328308105,2.071397304534912,1.5696130990982056,1.0513806343078613,0.8129216432571411,0.9578769207000732,1.0612804889678955,0.9755831956863403,0.8927322626113892,0.9791470766067505,1.0651450157165527,1.0607165098190308,1.0426815748214722,1.187910556793213,1.085572361946106,0.9842857718467712,0.9382396936416626,0.9692791700363159,0.9989485144615173,1.002864956855774]|1.0426815748214722|0.015525656752288342|0.005344632547348738|0.0030696671456098557|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3642|1.2620552778244019|Sphere|Low|[-0.1312602162361145,1.0292799472808838,0.4370562434196472]|[-0.19579005241394043,1.0340815782546997,0.44326087832450867]|1.872833013534546|2.9891819953918457|0.6108399033546448|true|37.089663036370084|[20.891340255737305,20.74936294555664,19.731258392333984,17.740068435668945,15.093345642089844,12.181805610656738,9.298479080200195,6.684168338775635,4.661325454711914,3.4724388122558594,2.974473714828491,2.787003517150879,2.7043299674987793,2.72312068939209,2.885225296020508,3.182893991470337,2.684072732925415,2.1566312313079834,1.872833013534546,1.88119375705719,2.0804104804992676,2.4414424896240234,2.9891819953918457]|3.182893991470337|0.05378839746117592|0.008433608338236809|0.004169710446149111|8|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3643|3.1827633380889893|ElongatedCylinder|Mid|[-0.18268278241157532,1.108489751815796,0.45304444432258606]|[-0.1941467523574829,1.107060194015503,0.4537445902824402]|0.4981037378311157|0.4981037378311157|0.16859650611877441|true|38.56947315424414|[13.283951759338379,11.755435943603516,9.870942115783691,7.9077534675598145,6.085778713226318,4.72605037689209,4.04270601272583,3.602937698364258,2.8331615924835205,1.762768268585205,0.9426671266555786,0.9300665855407715,1.2048301696777344,1.428088665008545,1.644145131111145,1.9059216976165771,2.148912191390991,2.236454963684082,2.1320278644561768,1.8367730379104614,1.6368309259414673,1.2173523902893066,1.1631443500518799,1.1730637550354004,1.086466908454895,0.9149377942085266,0.6963791251182556,0.4981037378311157]|1.6368309259414673|0.030634766444563866|0.004453351721167564|0.003326810197904706|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3644|1.8034710884094238|ElongatedCylinder|High|[-0.16410599648952484,1.1950924396514893,0.4804796874523163]|[-0.18799126148223877,1.197487235069275,0.49020323157310486]|0.3280893862247467|0.4793049395084381|0.31080228090286255|true|34.829306694778744|[22.113142013549805,21.60063934326172,20.48586082458496,18.77707290649414,16.721248626708984,14.585563659667969,12.573369026184082,10.891844749450684,9.56178092956543,8.41137409210205,7.34030818939209,6.3063859939575195,5.243131637573242,4.107929229736328,2.9775493144989014,1.9287328720092773,1.1422733068466187,0.850356936454773,0.9671266078948975,1.1499271392822266,1.219704270362854,1.268770694732666,1.0623815059661865,0.8515442609786987,0.5961471796035767,0.3547673225402832,0.3280893862247467,0.4793049395084381]|1.219704270362854|0.015121103264391422|0.005406329873949289|0.003100489266216755|9|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3645|3.7241790294647217|ElongatedCylinder|Low|[-0.17290806770324707,1.039229154586792,0.42825472354888916]|[-0.18130864202976227,1.0456393957138062,0.4320616126060486]|0.3600579798221588|1.0797256231307983|0.1928645521402359|true|36.69993467130033|[9.150331497192383,7.02732515335083,5.039035797119141,3.4176599979400635,2.1973917484283447,1.4488155841827393,1.1005501747131348,1.0656315088272095,1.1803799867630005,1.4522701501846313,1.7623915672302246,1.7683615684509277,1.4685189723968506,1.1425657272338867,0.824962854385376,0.5326120853424072,0.3600579798221588,0.40539079904556274,0.4263612926006317,0.4116189777851105,0.45109236240386963,0.6647157073020935,0.5826814770698547,0.46109431982040405,0.43979427218437195,0.6068703532218933,0.8567692041397095,1.0797256231307983]|0.45109236240386963|0.027112534269690514|0.0040748100727796555|0.0031210738234221935|20|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3646|2.3448870182037354|ElongatedCylinder|Mid|[-0.1649940311908722,1.1026322841644287,0.44981849193573]|[-0.19301730394363403,1.1013203859329224,0.45314034819602966]|1.1326873302459717|1.1326873302459717|0.21577148139476776|true|36.86733770882711|[18.053485870361328,17.192907333374023,15.874687194824219,14.030045509338379,11.866291999816895,9.676223754882812,7.797937870025635,6.696337699890137,6.3602118492126465,6.142547607421875,5.646338939666748,4.926209449768066,4.057616710662842,3.116490602493286,2.2719972133636475,1.694402813911438,1.5115152597427368,1.526049256324768,1.440524935722351,1.331712245941162,1.3502124547958374,1.7926610708236694,1.433056116104126,1.3598670959472656,1.3656342029571533,1.2598556280136108,1.153818130493164,1.1326873302459717]|1.3502124547958374|0.01200027298182249|0.0061018443666398525|0.0037790373899042606|12|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3647|4.265594959259033|Box|Mid|[-0.10839852690696716,1.1728864908218384,0.46367132663726807]|[-0.18966303765773773,1.1965923309326172,0.4795551598072052]|1.2182834148406982|1.2182834148406982|0.8430679440498352|false|35.962317978326034|[16.345417022705078,16.25582504272461,15.411358833312988,13.723749160766602,11.581009864807129,9.577980995178223,8.163518905639648,7.325316905975342,6.6397247314453125,5.7454352378845215,4.658514499664307,3.6466095447540283,9.12308120727539,8.072050094604492,6.637843132019043,4.977214336395264,3.4049465656280518,2.139024019241333,1.2182834148406982]|3.6466095447540283|0.06668733805418015|0.008798625320196152|0.00531770708039403|6|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3648|2.8863027095794678|Box|Low|[-0.15866422653198242,1.0276182889938354,0.4206014573574066]|[-0.15615913271903992,1.0297589302062988,0.4308958649635315]|0.2115298956632614|1.426167607307434|0.263416588306427|true|35.831544681319166|[13.931703567504883,12.361063957214355,10.612894058227539,8.721366882324219,6.746662616729736,4.788078784942627,2.9990289211273193,1.6014795303344727,0.7589187026023865,0.3922695219516754,0.2115298956632614,0.5550619959831238,1.110539436340332,1.3995920419692993,1.329877495765686,1.059828519821167,0.8175868391990662,0.7511590719223022,0.8653322458267212,0.8741737604141235,0.6955620646476746,1.1257160902023315,0.728600263595581,0.4560602903366089,0.4230881929397583,0.6361039280891418,1.03836989402771,1.426167607307434]|0.6955620646476746|0.04563148319721222|0.0048606665804982185|0.0033648773096501827|20|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
  |3649|1.507010579109192|Box|Low|[-0.13779330253601074,1.0398962497711182,0.4130282998085022]|[-0.16762113571166992,1.01058030128479,0.41067200899124146]|1.1194316148757935|1.1194316148757935|0.35556474328041077|true|37.50439418049959|[20.014331817626953,19.74274444580078,18.646495819091797,16.674230575561523,14.164755821228027,11.494549751281738,9.000923156738281,6.943305015563965,5.48641300201416,4.610429763793945,4.100951194763184,3.6810014247894287,3.220621109008789,2.718043088912964,2.167541980743408,1.6999106407165527,1.4585003852844238,1.550903081893921,1.8900858163833618,2.9828853607177734,2.679309129714966,2.357097864151001,1.999541163444519,1.6408946514129639,1.3293319940567017,1.1194316148757935]|1.8900858163833618|0.053485989570617676|0.007472377736121416|0.003570972941815853|11|[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]|
```

### `results/robustness_benchmark/robustness_metrics.json`
> 400-trajectory sensor sweep and 50-episode/condition PIL MPPI stress tests

```yaml
benchmark_name: Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test
version: SPWM-v0.12.3
checkpoint: results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt
split: test
handover_dist_threshold: 0.026
total_benchmark_time_sec: 540.10493016697
sensor_benchmark_time_sec: 290.17435620795004
mppi_benchmark_time_sec: 249.930085292086
sensor_suite (7 rows):
  |condition_name|condition_description|evaluated_trajectories|pos_mae_per_horizon|pos_mse_per_horizon|vel_mae_per_horizon|vel_mse_per_horizon|latent_mse_per_horizon|mean_spike_rate|drift_ratio|rollout_mae|rollout_vel_mae|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |Nominal Baseline|Clean events (theta = 0.08, 0% noise)|400|{1:0.023200019764212463,10:0.02914743970793027,50:0.09496219055010723,100:0.12599697594459242}|{1:9.270948458624144e-05,10:0.0002862775447563483,50:0.002346467746135134,100:0.0037753761101227542}|{1:0.03487798657554846,10:0.193737920660239,50:0.2915314229635092,100:0.3264090854388017}|{1:0.00023363830741100872,10:0.017213198069769602,50:0.02561490022792266,100:0.028923398456894435}|{1:0.0034456770473088208,10:0.03336573793337895,50:0.14944499616439527,100:0.4552898705005646}|0.16886679484294012|5.430899508928476|0.0683266564917106|0.21163910390952462|
  |Threshold Shift (theta = 0.05)|Hypersensitive contrast threshold (excessive triggers)|400|{1:0.024730355837024175,10:0.029505800885649826,50:0.09451624636466686,100:0.12587882234500006}|{1:0.00010471417548475214,10:0.00028009701838107925,50:0.0022852978257175824,100:0.0037466236748374426}|{1:0.03602095349476887,10:0.19536264813863313,50:0.29220786919960606,100:0.3215826107905461}|{1:0.0002907233182878162,10:0.017898809666243885,50:0.02550199284003331,100:0.028077146038413048}|{1:0.003464652905957057,10:0.03037899976166395,50:0.1060851405446346,100:0.3690155813327202}|0.16239805290332207|5.090053017213163|0.06865780635808523|0.21129352040588856|
  |Threshold Shift (theta = 0.12)|Hyposensitive contrast threshold (event dropout)|400|{1:0.023581712291790888,10:0.031107897082200416,50:0.10158508614851879,100:0.13018850409067595}|{1:9.633198687179874e-05,10:0.00029724174573157844,50:0.0028267552932867636,100:0.004083285967891033}|{1:0.03636597756009836,10:0.19551872519346383,50:0.2999404966831207,100:0.3458442504589374}|{1:0.0002451437097848751,10:0.017195390394100778,50:0.02750812370616656,100:0.032117279103169076}|{1:0.0034971129185018632,10:0.037674871201698594,50:0.21911612153053284,100:0.6815015031741216}|0.18293510606655708|5.520740075180899|0.0716157999032965|0.2194173624739051|
  |Background Activity (0.5%)|Thermal dark current Poisson flip noise (0.5% rate)|400|{1:0.31446244854193467,10:0.3130458914316617,50:0.24148339835497049,100:0.1990219675577604}|{1:0.017893086689022872,10:0.018401766195893288,50:0.012290612006416688,100:0.008773094723717524}|{1:0.2884048590293297,10:0.407500127187142,50:0.6605527217571552,100:0.5391311874756446}|{1:0.015800764569296286,10:0.06501392991496967,50:0.11117749374646407,100:0.0750805540726735}|{1:0.012719172841081252,10:0.04755970147939829,50:0.14018257879293883,100:0.40129072620318484}|0.16496412112162664|0.6328958146849134|0.2670034264715818|0.47389722386231786|
  |Background Activity (1.0%)|Thermal dark current Poisson flip noise (1.0% rate)|400|{1:0.40155811034716093,10:0.4219544736238626,50:0.30423943125284636,100:0.24146529344411996}|{1:0.028586954021683104,10:0.03298173343332914,50:0.01928400004712435,100:0.013036714723476997}|{1:0.42406835693579453,10:0.4751130457107837,50:0.7700750873639033,100:0.6146105619577261}|{1:0.0335168858560232,10:0.07526819837781099,50:0.1416193166604409,100:0.0936875704389352}|{1:0.014243993979807083,10:0.05295944815644851,50:0.13227384480146262,100:0.3583175264872037}|0.16082660624614128|0.6013209227311206|0.34230432716699744|0.570966762992052|
  |Refractory Dropout (10%)|Dead-time/bus saturation active event masking (10% rate)|400|{1:0.023716040098896392,10:0.029676143223276504,50:0.09631844552663657,100:0.1269878871165789}|{1:9.798971195078384e-05,10:0.00029030294144132105,50:0.002433990334303906,100:0.003845633276236745}|{1:0.03561046060461264,10:0.1951288626744197,50:0.2929766751252688,100:0.3297435549589304}|{1:0.00024922941520344466,10:0.017276655739316575,50:0.025928828578728896,100:0.02948701438995508}|{1:0.003493771906225727,10:0.034238147047849804,50:0.16500934729209313,100:0.49022589509303754}|0.17115286336495325|5.354514775107342|0.0691746289913471|0.21336488834080788|
  |Refractory Dropout (20%)|Dead-time/bus saturation active event masking (20% rate)|400|{1:0.024733260560494203,10:0.030587169986504775,50:0.0973021611571312,100:0.12772450882654923}|{1:0.0001075945547199808,10:0.00029786214313039987,50:0.0024925051615215265,100:0.003891697314639504}|{1:0.0367551466020254,10:0.19575042449511015,50:0.29445403126569897,100:0.3328194045103513}|{1:0.0002661194347848113,10:0.0173449946853977,50:0.02626550254913477,100:0.03003255292200125}|{1:0.0035658765476770126,10:0.035167103490004174,50:0.18052387122924513,100:0.5291586151489844}|0.1735653831408574|5.16407889344603|0.07008677513266985|0.21494475171829644|
mppi_suite (5 items):
  - [0]
    condition_name: Nominal Baseline
    condition_description: Nominal visual telemetry, calibrated dynamics and friction
    episodes_evaluated: 50
    grasp_success_rate_pct: 57.99999999999999
    min_placement_error_mean_cm: 0.9966417002677918
    min_placement_error_std_cm: 0.7168366082198878
    terminal_placement_error_mean_cm: 2.117010636925697
    terminal_velocity_mean: 0.37095459580421447
    mean_latency_ms: 32.431992493769116
    elevation_plane_breakdown:
      Low: {count:22.0,success_rate_pct:40.909090909090914,mean_error_cm:0.9283917966214094,mean_terminal_error_cm:2.744907847859643}
      Mid: {count:18.0,success_rate_pct:61.111111111111114,mean_error_cm:1.3163958506451712,mean_terminal_error_cm:1.8659200933244493}
      High: {count:10.0,success_rate_pct:90.0,mean_error_cm:0.5712340176105499,mean_terminal_error_cm:1.187599751353264}
    mass_tier_breakdown:
      Light (1.2 - 2.3 kg): {count:17.0,success_rate_pct:41.17647058823529,mean_error_cm:0.8797548360684339,mean_terminal_error_cm:2.0735019743442535}
      Medium (2.3 - 3.4 kg): {count:17.0,success_rate_pct:82.35294117647058,mean_error_cm:0.8255256133921006,mean_terminal_error_cm:2.0967908571748173}
      Heavy (3.4 - 4.5+ kg): {count:16.0,success_rate_pct:50.0,mean_error_cm:1.3026448357850313,mean_terminal_error_cm:2.1847221069037914}
    episodes_with_handover_pct: 0.0
    phi1_steps_with_handover_pct: 0.0
    mean_nonzero_handover_w: 0.0
    total_phi1_steps: 400
    triggered_phi1_steps: 0
    triggered_episodes: 0
  - [1]
    condition_name: Sensory Noise Injection (1.0%)
    condition_description: Visual events corrupted with 1.0% Background Activity noise
    episodes_evaluated: 50
    grasp_success_rate_pct: 54.0
    min_placement_error_mean_cm: 1.1959887701272964
    min_placement_error_std_cm: 0.891931935595883
    terminal_placement_error_mean_cm: 2.683878175020218
    terminal_velocity_mean: 0.33141832798719406
    mean_latency_ms: 32.060862347789936
    elevation_plane_breakdown:
      Low: {count:22.0,success_rate_pct:50.0,mean_error_cm:1.377844351259145,mean_terminal_error_cm:3.0191407068209215}
      Mid: {count:18.0,success_rate_pct:33.33333333333333,mean_error_cm:1.1520791616704729,mean_terminal_error_cm:2.761846797333823}
      High: {count:10.0,success_rate_pct:100.0,mean_error_cm:0.8749437868595124,mean_terminal_error_cm:1.8059570848941804}
    mass_tier_breakdown:
      Light (1.2 - 2.3 kg): {count:17.0,success_rate_pct:52.94117647058824,mean_error_cm:0.97267567059573,mean_terminal_error_cm:2.49932455665925}
      Medium (2.3 - 3.4 kg): {count:17.0,success_rate_pct:64.70588235294117,mean_error_cm:1.1479233976672678,mean_terminal_error_cm:2.8443444406285003}
      Heavy (3.4 - 4.5+ kg): {count:16.0,success_rate_pct:43.75,mean_error_cm:1.4843283966183662,mean_terminal_error_cm:2.7094709873199463}
    episodes_with_handover_pct: 0.0
    phi1_steps_with_handover_pct: 0.0
    mean_nonzero_handover_w: 0.0
    total_phi1_steps: 400
    triggered_phi1_steps: 0
    triggered_episodes: 0
  - [2]
    condition_name: Payload Extrapolation (4.80 kg)
    condition_description: Out-of-distribution heavy payload (m=4.80 kg vs [1.20, 4.50] kg)
    episodes_evaluated: 50
    grasp_success_rate_pct: 56.00000000000001
    min_placement_error_mean_cm: 1.7644587180018425
    min_placement_error_std_cm: 1.8775155414354365
    terminal_placement_error_mean_cm: 2.4560093522071837
    terminal_velocity_mean: 0.5556513021886349
    mean_latency_ms: 32.58270962906839
    elevation_plane_breakdown:
      Low: {count:22.0,success_rate_pct:63.63636363636363,mean_error_cm:1.4761234576051885,mean_terminal_error_cm:2.2185537300326605}
      Mid: {count:18.0,success_rate_pct:38.88888888888889,mean_error_cm:2.432745859854751,mean_terminal_error_cm:3.117100712325838}
      High: {count:10.0,success_rate_pct:70.0,mean_error_cm:1.1958794355392457,mean_terminal_error_cm:1.7884472727775573}
    mass_tier_breakdown:
      Light (1.2 - 2.3 kg): {count:17.0,success_rate_pct:58.82352941176471,mean_error_cm:1.0763100201592726,mean_terminal_error_cm:1.8543823571766125}
      Medium (2.3 - 3.4 kg): {count:17.0,success_rate_pct:70.58823529411765,mean_error_cm:1.6607786907869226,mean_terminal_error_cm:2.4388887181001553}
      Heavy (3.4 - 4.5+ kg): {count:16.0,success_rate_pct:37.5,mean_error_cm:2.6057767383754253,mean_terminal_error_cm:3.113428708165884}
    episodes_with_handover_pct: 0.0
    phi1_steps_with_handover_pct: 0.0
    mean_nonzero_handover_w: 0.0
    total_phi1_steps: 400
    triggered_phi1_steps: 0
    triggered_episodes: 0
  - [3]
    condition_name: Actuator Drift (0.80x Friction)
    condition_description: Mechanical wear / loose joints (-20% friction/damping)
    episodes_evaluated: 50
    grasp_success_rate_pct: 68.0
    min_placement_error_mean_cm: 1.7202977061271667
    min_placement_error_std_cm: 1.8452188046070126
    terminal_placement_error_mean_cm: 2.383537862300873
    terminal_velocity_mean: 0.4953418381512165
    mean_latency_ms: 33.133227099747344
    elevation_plane_breakdown:
      Low: {count:22.0,success_rate_pct:72.72727272727273,mean_error_cm:1.3773770914836363,mean_terminal_error_cm:2.070421573790637}
      Mid: {count:18.0,success_rate_pct:50.0,mean_error_cm:2.5278382500012717,mean_terminal_error_cm:3.2292134364446006}
      High: {count:10.0,success_rate_pct:90.0,mean_error_cm:1.021150079369545,mean_terminal_error_cm:1.550177663564682}
    mass_tier_breakdown:
      Light (1.2 - 2.3 kg): {count:17.0,success_rate_pct:88.23529411764706,mean_error_cm:1.0297587829477646,mean_terminal_error_cm:1.9232896531329435}
      Medium (2.3 - 3.4 kg): {count:17.0,success_rate_pct:76.47058823529412,mean_error_cm:1.6581512759713566,mean_terminal_error_cm:2.3134185040698334}
      Heavy (3.4 - 4.5+ kg): {count:16.0,success_rate_pct:37.5,mean_error_cm:2.5200258940458298,mean_terminal_error_cm:2.947053402662277}
    episodes_with_handover_pct: 0.0
    phi1_steps_with_handover_pct: 0.0
    mean_nonzero_handover_w: 0.0
    total_phi1_steps: 400
    triggered_phi1_steps: 0
    triggered_episodes: 0
  - [4]
    condition_name: Actuator Drift (1.20x Friction)
    condition_description: Mechanical wear / stiff transmission (+20% friction/damping)
    episodes_evaluated: 50
    grasp_success_rate_pct: 70.0
    min_placement_error_mean_cm: 1.787329178750515
    min_placement_error_std_cm: 1.8200874079923721
    terminal_placement_error_mean_cm: 2.4599468809366227
    terminal_velocity_mean: 0.49150543451309203
    mean_latency_ms: 33.69387655897015
    elevation_plane_breakdown:
      Low: {count:22.0,success_rate_pct:68.18181818181817,mean_error_cm:1.4700086469000035,mean_terminal_error_cm:2.166050752455538}
      Mid: {count:18.0,success_rate_pct:66.66666666666666,mean_error_cm:2.5528457396560245,mean_terminal_error_cm:3.327590372827318}
      High: {count:10.0,success_rate_pct:80.0,mean_error_cm:1.1075045391917229,mean_terminal_error_cm:1.5447600781917572}
    mass_tier_breakdown:
      Light (1.2 - 2.3 kg): {count:17.0,success_rate_pct:82.35294117647058,mean_error_cm:1.001258819418795,mean_terminal_error_cm:1.8067133040989147}
      Medium (2.3 - 3.4 kg): {count:17.0,success_rate_pct:82.35294117647058,mean_error_cm:1.7711794867235071,mean_terminal_error_cm:2.4717557535452}
      Heavy (3.4 - 4.5+ kg): {count:16.0,success_rate_pct:43.75,mean_error_cm:2.6396879833191633,mean_terminal_error_cm:3.1414606291800737}
    episodes_with_handover_pct: 0.0
    phi1_steps_with_handover_pct: 0.0
    mean_nonzero_handover_w: 0.0
    total_phi1_steps: 400
    triggered_phi1_steps: 0
    triggered_episodes: 0
```

## 2. Architecture, Training & Environment Context

### `docs_verified/README.md`
> Verified master overview, CoRL grounded PIL benchmarks, 3-seed validation, and documentation index

# SPWM Verified Documentation (`docs_verified/`)

> **Consolidated Single-Source-of-Truth Documentation** replacing legacy `docs/` with unified, directly verified evidence. Every claim, equation, hyperparameter, and benchmark value below is directly linked to source code, JSON/checkpoint artifacts ([ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json), [multiseed_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/multiseed_metrics.json), [corl_final_benchmark_summary.md](file:///Users/Mattia/Desktop/Studies/Temp/results/corl_final_benchmark_summary.md)), and reproducible CLI commands.

---

## 1. Executive Audit Verdict: Are the Documentation Values Correct and Backed Up?

**Yes for all core empirical results.** Every benchmark metric in the latest results pipeline ([ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json), [multiseed_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/multiseed_metrics.json), [comparative_metrics_v0_12_4.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json), and `results/`) matches actual command runs recorded across three sequential Antigravity sessions:
* **Agent 3 (Iteration 8, finished `2026-11-20 13:23 CET`):** [transcript.jsonl](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/logs/transcript.jsonl) (`task-346`, `352`, `361`, `402`, `524`, `557`, `586`).
* **Agent 2 (Iteration 9, `13:24–16:01 CET`):** [transcript.jsonl](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/logs/transcript.jsonl) (`task-29`, `58`, `60`, `84`, `107`, `127`, `141`, `152`).
* **Agent 1 (Iteration 10, `15:11–16:01 CET`):** [transcript.jsonl](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/logs/transcript.jsonl) (`task-34`, `108–111`, `118–119`, `126`, `138`, `152–153`, `161`, `206`, `213`, `249`).

### Progression of Testing Precision & Baseline Equalization
The reported SPWM and comparative baseline numbers evolved directly as evaluation rigor, baseline training equality, and gating fairness tightened:
1. **Non-PIL Latent-Grounded (`96.0%`) & Pre-Fix PIL (`88.0%`):** Early runs either blended ground-truth latents (`z_gt_next`) during $\phi_t = 1$ or delayed payload attachment to `t >= t_grasp` (historical Nov 17 snapshot).
2. **Causal Physics-in-the-Loop (`60.7 ± 2.3%` $N=25$ Validation-Selected 3-Seed Mean):** Enforced causal `payload_attached=(t >= t_c)` in [experiments/benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py#L390) and [spwm/evaluation/mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py#L369), where $t_c$ indexes the $\phi_t=1$ sensory-window onset (`p1_indices[0] = t_contact - 7`) and PIL evaluates the 7-step loaded transition through $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$, with zero `z_gt_next` leakage, causal `OnlineEventCameraStream`, and strict grasp success criteria (`< 3.0 cm` 3-step settling, `< 0.05 rad` $\text{SO}(3)$ error, $\tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)$) ([tests/test_leakage.py](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_leakage.py), [tests/test_pil_integrity.py](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_pil_integrity.py)).
3. **Baseline Equalization (AR(1)+DAgger) & Identical FK Residual Rejection Gate:** Fixed the truncated `18`-epoch `Deep GRU` runs on Seeds `101, 2024`, trained both `Continuous RSSM` and `Deep GRU` across all 3 seeds (`42, 101, 2024`) with multi-step SE(3) rollout + AR(1) MPPI-matched action noise + `ClosedLoopBuffer` DAgger replay, and evaluated all three PIL regimes (**Open-Loop $w_t=0$**, **Un-Gated Online $w_t=1.0$**, and **Residual-Gated Online** using the identical FK rejection gate as SPWM). Under residual gating, `Continuous RSSM` remains at `15.3 ± 5.0%` (`0.0%` un-gated, confirming its ceiling is architectural), while equalized `Deep GRU` recovers from `0.0%` (un-gated) to `76.7 ± 5.0%` (`77.3 ± 4.2%` open-loop)—achieving higher nominal PIL success than `SPWM` (`60.7 ± 2.3%`) while exceeding the `50 ms` (`20 Hz`) real-time budget (`57.7 ms/step` vs `26.6 ms/step`), trailing `SPWM` by `9.0×` on $H=100$ velocity MSE (`0.08716` vs `0.00971`), **collapsing under zero-shot mechanical perturbations** (`23.3 ± 12.9%` at `4.80 kg` payload, `26.7 ± 23.9%` at `0.80×` friction, `30.0 ± 15.1%` at `1.20×` friction vs. `SPWM` `44.7 ± 13.3% / 52.7 ± 13.3% / 48.0 ± 19.3%`), and using `22.1×` higher standby core energy / `1.59×` higher E2E step energy.
4. **Full 3-Seed 6-Row Disentangled Ablation (`45.3 ± 16.0% -> 51.3 ± 9.5% -> 38.0 ± 19.1% -> 49.3 ± 6.1% -> 60.7 ± 2.3% -> 60.7 ± 2.3%`):** Evaluated across all 3 seeds (`42, 101, 2024`) on `[3600, 3649]`. Transparently reveals that Phase 3/4 AR(1)+DAgger replay (`Stage 3a`, `checkpoint_phase_4.pt`) reduces trajectory jitter (`|Δ²q|: 0.00991 -> 0.00589`) while degrading 3-seed mean PIL success (`-13.3 pp`) and doubling seed std (`9.5% -> 19.1%`). Isolating **`N=25` validation checkpoint selection (`[3200, 3224]`) without Lookahead (`Stage 3b`: `42: phase_4`, `101: phase_1`, `2024: phase_2`)** recovers 3-seed PIL success to **`49.3 ± 6.1%`** (`56 / 44 / 48%`, `+11.3 pp` from checkpoint selection alone), and adding **2-Stage SE(3) Lookahead (`Stage 4a/4b`)** contributes a further **`+11.3 pp`** gain to **`60.7 ± 2.3%`** (`58 / 62 / 62%`).
5. **Standby Core vs. E2E Energy Recalibration:** Frames the `4.94×` Loihi-2 core reduction (`12.22 ± 0.19 nJ/step`, `0.244 mW` @ `20 Hz`, excluding passive membrane leak) strictly as an **always-on standby state-estimation benefit** when dense MPPI planning is dormant; E2E step energy (`1.0979 mJ/step`) matches `ANN-Rate` (`0.004%` diff), so `SPWM`'s primary advantage over `ANN-Rate` is dynamical stability (`+23.4 pp` nominal PIL success, `+24.7 pp` and `2.0×` better relative retention under `1.0%` sensory noise on `$N=25$` validation-selected checkpoints).

---

## 2. Corrections & Reconciliations Applied

| Historical Issue Resolved | Ground Truth Verified in Code / Artifacts | Where Resolved Here |
| :--- | :--- | :--- |
| Un-gated (`w_t=1.0`) online baseline collapse & truncated `Deep GRU` epochs (`59/18/18`) | Equalized `Continuous RSSM` (`60+16/16/17`) & `Deep GRU` (`59+40=99`, `80`, `80`) with AR(1)+DAgger; evaluated Open-Loop, Un-Gated, and Residual-Gated PIL regimes across all 3 seeds | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L22-L70) |
| Single-seed `"Canonical Seed 42"` (`86.0%`) headline & Seed-42-only ablation | Removed all `"Canonical Seed 42"` headline claims; lead strictly with $N=25$ val-selected (`60.7 ± 2.3%`) and 3-seed 6-row ablation isolating $N=25$ selection without Lookahead (`45.3 -> 51.3 -> 38.0 -> 49.3 -> 60.7 -> 60.7%`) | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L72-L90) |
| Contradictory `heft_shock_timeseries` & extra plots / scripts | Removed all non-architecture figures (`fig1..fig4`, `heft_shock_timeseries`) and plotting scripts (`figure_style.py`, `plot_heft_shock_timeseries.py`, `run_ssot_benchmark_and_plot.py`); retained [architecture.pdf](file:///Users/Mattia/Desktop/Studies/Temp/figures/architecture.pdf) | [01_SYSTEM_AND_ARCHITECTURE.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md#L16) |
| Overstated neuromorphic E2E energy framing (`48.14 nJ` atop `1.0971 mJ` MPPI) | Recalibrated Loihi-2 `12.22 nJ` (`4.94×`) as an always-on standby core benefit; E2E matches `ANN-Rate` (`1.0979 mJ`), while `1.93×` E2E gain vs `RSSM` comes from `M1`'s compact MLP | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L106-L130) |
| `GLOBAL_VERSION` vs YAML project name & legacy formula drift | [spwm/contracts/version.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/version.py#L29-L32) defines `0.12.3` (`spwm_v0_12_3`); YAML overrides `project.name: spwm_v0_12_4`. Equations verified directly from code | [01_SYSTEM_AND_ARCHITECTURE.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md#L28) |

---

## 3. Directory Map (`docs_verified/`)

| Document | Scope | Replaces Legacy `docs/` Files |
| :--- | :--- | :--- |
| [README.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/README.md) | Audit verdict, testing precision progression, and discrepancy reconciliations | `CORL_GROUNDED_AUDIT.md`, `History.md` (meta) |
| [01_SYSTEM_AND_ARCHITECTURE.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md) | Repo skeleton, 4-layer contracts, Unity `6000.5.7f1` / Python PIL physics, dataset splits, neural equations, exact parameter census, and design decisions / negative ablations | `REPO_SKELETON.md`, `WORKSPACE_AND_CONTRACTS.md`, `WORLD.md`, `Architecture.md`, `DOSSIER.md` (ablations) |
| [02_TRAINING_CURRICULUM_AND_MPPI.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md) | Phase 0 & Phase 1 training strategies, exact losses, 4-phase latch curriculum, active perception / residual handover, MPPI cost, Tier-2 validation, training CLIs, and baseline / online update specs | `TRAINING_AND_LOSSES.md`, `Architecture.md` (MPPI) |
| [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md) | Checkpoint provenance, Tables 1–5 (3-regime PIL, 3-seed ablation, standby vs. E2E energy), Phase 0 metrics, validation sweeps, forensic diagnostics, compute/wall-clock stack, and replication runbook | `CORL_FINAL_BENCHMARK_SUMMARY.md`, `FIGURE_METRICS.md`, `DOSSIER.md` (benchmarks), `History.md` (Iter 8–10) |

---

## 4. Quick Verification Gate

Run the repository AST anti-fallback check and full test suite:

```bash
.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/
.venv/bin/pytest tests/ -q
```

### `docs_verified/01_SYSTEM_AND_ARCHITECTURE.md`
> Verified system, Unity PhysX environment, and neural architecture specification

# 01 — System, Contracts, World Physics & Neural Architecture

---

## 1. Repository Layout & Layer Dependency Architecture

### 1.1 Verified Directory Skeleton
* **`spwm/` (46 `.py` files):** Core library organized into 4 strict dependency layers plus `utils/` and `version.py`:
  * **Layer 0 (`spwm/contracts/`):** [config.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py), [tensors.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py), [states.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py), [protocols.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py), [version.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/version.py), [\_\_init\_\_.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py)
  * **Layer 1 (`spwm/models/`):** [encoder.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py), [neurons.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/neurons.py), [memory.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/memory.py), [dynamics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py), [predictors.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py), [kinematics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/kinematics.py), [rollout.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/rollout.py), [world_model.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/world_model.py), [ann_ablation.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/ann_ablation.py), [baselines.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/baselines.py), [\_\_init\_\_.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py)
  * **Layer 1 (`spwm/data/`):** [event_camera.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/event_camera.py), [datasets.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/datasets.py), [io_unity.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/io_unity.py), [sim_6dof.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/sim_6dof.py), [unity_episode_meta.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/unity_episode_meta.py), [\_\_init\_\_.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py)
  * **Layer 2 (`spwm/training/`):** [engine.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py), [losses.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/losses.py), [metrics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/metrics.py), [noise.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/noise.py), [checkpoints.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/checkpoints.py), [replay_buffer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/replay_buffer.py), [diagnostics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/diagnostics.py), [phases/representation.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/representation.py), [phases/m1_nominal.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/m1_nominal.py), [phases/\_\_init\_\_.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/__init__.py)
  * **Layer 3 (`spwm/evaluation/`):** [evaluator.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/evaluator.py), [mppi_controller.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_controller.py), [mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py), [mppi.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi.py), [robustness.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/robustness.py), [visualizer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/visualizer.py), [metrics_resume.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/metrics_resume.py), [\_\_init\_\_.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/__init__.py)
  * **Utilities & Facade:** [spwm/utils/reproducibility.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/utils/reproducibility.py), [spwm/version.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/version.py)
* **`experiments/` (14 runners + `__init__.py`):** [train.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/train.py), [train_multiseed.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/train_multiseed.py), [train_ann_ablation.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/train_ann_ablation.py), [train_baselines.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/train_baselines.py), [benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py), [benchmark_robustness_stress_test.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_robustness_stress_test.py), [benchmark_classical_and_bc.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_classical_and_bc.py) *(implements Privileged State-Oracle Jacobian-IK + PD reading true physical joint angles $\mathbf{q}_{\text{phys}}$ directly without vision, calibrated to mean `2.85 kg` payload, plus Reactive BC `322,210` params)*, [profile_energy_sops.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/profile_energy_sops.py), [evaluate.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/evaluate.py), [ablation.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/ablation.py), [optimize_mppi_cost.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/optimize_mppi_cost.py), [export_mppi_trajectory.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/export_mppi_trajectory.py), [export_ghost_trajectory.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/export_ghost_trajectory.py), [test_smooth_rollout.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/test_smooth_rollout.py).
* **`scripts/` (16 files):** [eval_table1_ssot.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/eval_table1_ssot.py), [aggregate_corl_final_verification.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/aggregate_corl_final_verification.py), [scan_fallbacks.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/scan_fallbacks.py), [build_paper_context.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/build_paper_context.py), [run_remaining_cpu_suite.sh](file:///Users/Mattia/Desktop/Studies/Temp/scripts/run_remaining_cpu_suite.sh), [run_pipeline.sh](file:///Users/Mattia/Desktop/Studies/Temp/scripts/run_pipeline.sh), [run_baselines.sh](file:///Users/Mattia/Desktop/Studies/Temp/scripts/run_baselines.sh), `diagnostics/` ([forensic_rssm_gru_failure.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/forensic_rssm_gru_failure.py), [audit_spwm_seed_variance.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/audit_spwm_seed_variance.py), [eval_phase0_kinematics.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/diagnostics/eval_phase0_kinematics.py)), and utility scripts ([generate_metrics_resumes.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/generate_metrics_resumes.py), [json_compressor.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/json_compressor.py), [parse_logs.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/parse_logs.py), [prune_stale_phase_flags.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/prune_stale_phase_flags.py), [render_simulation_video.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/render_simulation_video.py), [sanity_check_data.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/sanity_check_data.py)).
* **`figures/`:** Retains only [architecture.pdf](file:///Users/Mattia/Desktop/Studies/Temp/figures/architecture.pdf) and the LaTeX manuscript sources ([main_v0_12_4.tex](file:///Users/Mattia/Desktop/Studies/Temp/figures/main_v0_12_4.tex), [main_v0_12_4.pdf](file:///Users/Mattia/Desktop/Studies/Temp/figures/main_v0_12_4.pdf)); all non-architecture figure artifacts and plotting scripts were removed.
* **`sim/`:** Unity `6000.5.7f1` project (`Assets/`, `Library/`, `Logs/`, `Packages/`, `ProjectSettings/`, `UserSettings/`, `Stevedore.conf`) and `ExportedTrajectories/` containing `4,000` `.json.gz` files (`trajectory_00000.json.gz` .. `trajectory_03999.json.gz`; zero HDF5/NPZ files).
* **`tests/` (17 test modules + `__init__.py`):** Enforcing architectural contracts, zero-fallback AST scanning, PIL physics/event integrity, and zero state leakage (`102` tests total).

### 1.2 Layer Rules & Strict Anti-Fallback Policy
* **Dependency Rule:** Lower layers never import higher layers (`Layer 0` zero internal/PyTorch-unrelated deps; `Layer 1` imports `Layer 0`; `Layer 2` imports `Layers 0–1` and `spwm.utils`; `Layer 3` imports `Layers 0–2` and `spwm.utils`). Phase strategies in `spwm/training/phases/` are strictly `< 150` lines (`representation.py` = `123` lines, `m1_nominal.py` = `130` lines).
* **AST Enforcement:** 3-argument `getattr(obj, attr, default)`, ternary `hasattr(...)` fallbacks, `strict=False` checkpoint loads, silent `except` swallowing, and silent `dict.get` config fallbacks are banned and checked by [scripts/scan_fallbacks.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/scan_fallbacks.py) and [tests/test_anti_fallback.py](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_anti_fallback.py).

---

## 2. Layer 0 Contracts & SSOT Configuration

* **Version Solver ([spwm/contracts/version.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/version.py#L29-L32)):** `VersionSolver(version=0, minor=12, patch=3, prefix="SPWM")` evaluates `GLOBAL_VERSION.tag == "SPWM-v0.12.3"` and `GLOBAL_VERSION.snake_case == "spwm_v0_12_3"`. Production YAMLs ([results/spwm_v0_12_4_train_m1/config.yaml](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/config.yaml#L2)) explicitly set `project.name: spwm_v0_12_4`, while scripts that read `GLOBAL_VERSION.tag` directly (e.g., [benchmark_robustness_stress_test.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_robustness_stress_test.py)) record `"SPWM-v0.12.3"`.
* **Typed Config ([spwm/contracts/config.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py#L61-L484)):** `ExperimentConfig` composes 10 frozen dataclasses (`ProjectConfig`, `EnvironmentConfig`, `DataConfig`, `ModelConfig`, `LatentNeuronConfig`, `MemoryConfig`, `TrainingConfig`, `LossConfig`, `EvaluationConfig`, `MPPIConfig`). `_strict_dict_to_dataclass` rejects unknown keys; `ExperimentConfig.with_overrides` exposes `31` explicit keyword-only parameters covering `project`, `data`, `training`, and `loss` subsets (`config.py:446-480`).
  * `MPPIConfig.__post_init__` (`config.py:372-398`) validates `num_samples > 0`, `horizon > 0`, `action_dim > 0`, `temperature > 0`, `shoulder_sigma_multiplier > 0`, `elbow_sigma_multiplier > 0`, `nominal_update_gamma in (0, 1]`, `noise_beta in [0, 1)`, `chunk_size > 0`, `num_iterations > 0`, `w_joint_accel >= 0`, `w_joint_limit >= 0`, and `max_action_delta > 0`.
* **Tensor & Phase-Space Invariants ([spwm/contracts/tensors.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py#L12-L114)):** `J = 6`, `C_EVENT = 4`, `latent_dim = 128 == q_dim (32) + p_dim (96)`, `p_dim % q_dim == 0` (`3x`), and `2 * num_keypoints (16) == q_dim (32)`. `assert_phase_space_contract` checks `actions.shape[-1] == action_dim` when `action_dim` is passed (`tensors.py:89`) or `> 0` otherwise (`tensors.py:94`).
* **State Containers & Protocols ([spwm/contracts/states.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py), [spwm/contracts/protocols.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)):** Standardized containers include `ALIFState`, `MultiTimescaleState`, `DynamicsState`, `RolloutResult` (returned by `rollout_latent`, holding `trajectory` and `fast_trajectory`), `ClosedLoopTrajectory`, and `Tier2EvalResult`. Note: `MPPIConfig`, `ClosedLoopTrajectory`, `Tier2EvalResult`, and `StatefulTier2EvaluatorProtocol` live in their respective submodules and are not re-exported by `spwm/contracts/__init__.py`.

---

## 3. Physical World, Simulator & Dataset Splits

### 3.1 Unity `6000.5.7f1` Articulated Arm & Spawner
* **Engine & Solver:** Unity `6000.5.7f1` ([ProjectVersion.txt](file:///Users/Mattia/Desktop/Studies/Temp/sim/ProjectSettings/ProjectVersion.txt#L1-L2)), macro timestep $\Delta t = 0.05\,\text{s}$ (`20 Hz`), `physicsSubSteps = 5` ($\Delta t_{\text{fixed}} = 0.01\,\text{s}$ / `100 Hz`), `24` solver position / `8` velocity iterations ([RoboticTelemetryLogger.cs](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Scripts/RoboticTelemetryLogger.cs#L65-L108)).
* **7-Segment Kinematic Chain ([ArticulatedRobotRig.cs](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Scripts/ArticulatedRobotRig.cs#L722-L833), [kinematics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/kinematics.py#L18-L49)):** Total zero-configuration height is `1.25 + 0.05 + 0.05 + 0.28 + 0.25 + 0.04 + 0.04 + 0.06 = 2.02 m`.
  * *Visual Note:* Both `ArticulatedRobotRig.cs:277-319` and `GhostPlaybackManager.cs:89-138` instantiate `_JointHub` spheres on all joints except `Shoulder_Pan` and separate cylinders (`0.04m`, `0.04m`, `0.06m`) for `Wrist_Pitch`, `Wrist_Yaw`, and `Wrist_Roll`.

| Link | Parent | Anchor $(0, \Delta y, 0)$ [m] | Mass [kg] | Limits [deg] | Drive Stiffness / Damping / MaxForce | Rotation Axis (from `AxisRotationEuler`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Base_Root` | `-1` | `1.25` | `20.0` | Fixed | — | — |
| `Shoulder_Pan` ($q_0$) | `0` | `0.05` | `2.5` | `[-120, 120]` | `300 / 75 / 600` | `(0.0, 1.0, 0.0)` |
| `Shoulder_Lift` ($q_1$) | `1` | `0.05` | `2.2` | `[-90, 185]` | `320 / 80 / 600` | `(1.0, 0.0, 0.0)` |
| `Elbow_Flex` ($q_2$) | `2` | `0.28` | `1.6` | `[-155, 15]` | `260 / 65 / 500` | `(0.5375, -0.2756, -0.7969)` |
| `Wrist_Pitch` ($q_3$) | `3` | `0.25` | `0.6` | `[-95, 95]` | `200 / 50 / 350` | `(0.5375, -0.2756, -0.7969)` |
| `Wrist_Yaw` ($q_4$) | `4` | `0.04` | `0.5` | `[-75, 75]` | `180 / 45 / 300` | `(0.0, 0.0, -1.0)` |
| `Wrist_Roll` ($q_5$) | `5` | `0.04` (tip `+0.06`) | `0.5` | `[-180, 180]` | `150 / 40 / 250` | `(0.0, 1.0, 0.0)` |

* **Unity Actuation ([ActionConditionedController.cs](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Scripts/ActionConditionedController.cs#L459-L475)):** Default `MotorTargetPD` writes `qDes` / `dqDes` to `ArticulationBody.xDrive` using the table's per-joint `DriveStiffness` and `DriveDamping` (`setupHoldSteps = 6`).
* **Payloads & Grasp ([ProceduralObjectSpawner.cs](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Scripts/ProceduralObjectSpawner.cs#L152-L214)):** Three elevation planes (`Low ~1.04 m`, `Mid ~1.14 m`, `High ~1.26 m`; classified in [kinematics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/kinematics.py#L248-L255) by `y < 1.08`, `1.08 <= y < 1.18`, `y >= 1.18`), forward reach `[0.42, 0.52] m`, mass $m \in [1.20, 4.50]\,\text{kg}$ (`Light < 2.30`, `Normal [2.30, 3.40)`, `Heavy >= 3.40` per [unity_episode_meta.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/unity_episode_meta.py#L68)), geometries `Sphere`/`Box`/`ElongatedCylinder`, locked via `FixedJoint` on contact.

### 3.2 Python 6-DOF Physics-in-the-Loop (PIL) Stepper & Dual Event Camera
* **Closed-Loop Physics ([sim_6dof.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/sim_6dof.py#L282-L359)):** Unlike Unity PhysX 4's full Featherstone solver, `ArticulatedArmPhysicsStepper` implements an analytical `100 Hz` decoupled 6-DOF articulated dynamics stepper (diagonal effective inertia, zero Coriolis/inter-link mass coupling, symmetric `[-3.14, 3.14] rad` joint limits) running `sub_steps=5` (`0.01 s`) per `0.05 s` macro-step with `drive_stiffness = 16.5`, `damping = 0.15`, `gravity_scale = 0.22`, `max_joint_angle = 3.14 rad`, and PD tracking (`kp = 18.0`, `kd = 0.80`) with gravity feedforward:
  $$\mathbf{u}_{\text{PD}} = \operatorname{clip}\!\left(\frac{\mathbf{q}_{\text{des}} + k_p(\mathbf{q}_{\text{des}}-\mathbf{q}) + k_d(\dot{\mathbf{q}}_{\text{des}}-\mathbf{p}) + g_{\text{eff}}\sin(\mathbf{q})/K_{\text{stiff}}}{\theta_{\max}},\,-1,\,1\right)$$
  $$\ddot{\mathbf{q}} = \frac{1}{I_{\text{eff}}}\Big[K_{\text{stiff}}(\theta_{\max}\mathbf{u}_{\text{PD}} - \mathbf{q}) - g_{\text{eff}}\sin(\mathbf{q}) - 0.04\tanh(5\mathbf{p}) - (7.5 \cdot \text{damping})\mathbf{p}\Big]$$
  When `payload_attached = (t >= t_c)` becomes `True` (where $t_c = \texttt{p1\_indices[0]} = t_{\text{contact}} - 7$ is the $\phi_t=1$ sensory-window onset and PIL evaluates the 7-step loaded transition through $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$; rigid palm-latch matching Unity's `FixedJoint`), effective inertia jumps from `unloaded_inertia` (`[0.80, 0.69, 0.58, 0.47, 0.36, 0.25]`) by `+ 0.16 * m_payload * factor` (proximal `1.25`, distal `0.85`/`1.18`/`1.45` by shape) and gravity increases by `(1 + 0.42 * m_payload)`. Joint velocity clips to `[-4.5, 4.5] rad/s`.
* **Dual-Camera Event Stream (`C=4, H=32, W=32`, $\theta = 0.08$):** Exported `.json.gz` files record `sensor_height=32, sensor_width=32, sensor_channels=4` (`trajectory_00000.json.gz`). Unity's [EventCameraLogDiff.compute](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Shaders/EventCameraLogDiff.compute#L29) computes `log(max(lum, 0) + 1e-3)` at `20 Hz` macro-steps during logging, whereas Python's [EventCameraSimulator](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/event_camera.py#L248-L255) (and `OnlineEventCameraStream` in PIL) differences linear grayscale frames directly without `log()`.
* **Dataset Splits ([datasets.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/datasets.py#L40-L55,L265-L366)):** `4,000` trajectories (`T = 105` steps) split strictly by ID into **Train `[0, 3200)`**, **Val `[3200, 3600)`**, and **Test `[3600, 4000)`**. When `strict_dataset=True` (default), `extrap_ds` reuses `[3600, 3700)` (`datasets.py:270`). `weighted_sampling=True` in Phase 1 applies `WeightedRandomSampler` (`2.5x` heavy `mass >= 3.40`, `2.5x` mid-plane `1.08 <= y < 1.18`, `5.0x` both).

---

## 4. Neural Architecture & Verified Parameter Census

### 4.1 Exact Parameter Count Breakdown
Measured programmatically from [world_model.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/world_model.py#L43-L176) and baseline modules using `results/spwm_v0_12_4_train_m1/config.yaml`:

| Model / Submodule | Total Params | Active Inference Params | Notes |
| :--- | :---: | :---: | :--- |
| `encoder` (`conv` `5,856` + `wrist_stem` `5,040` + `fusion_head` `528` + `proj` `4,480`) | `15,904` | `15,904` | Dual-camera `4x32x32 -> 8x8` SpatialSoftmax (`0` params) $\to 128$-d |
| `sensory_predictor` (`Linear(128, 128, bias=False)`) | `16,384` | `16,384` | Top-down error prediction $\hat{\mathbf{x}}_t = \mathbf{W}_{\text{sens}}\mathbf{z}_{t-1}$ |
| `dynamics` (`memory` `512` + `input_proj` `33,024` + `recurrent_proj` `24,576` + `q_proj` `8,192` + `action_proj` `1,536` + `W_vel` `8,192` + `fuse_spikes` `24,672` + `fuse_mems` `24,576` + `norm` `192` + `gate_scale` `32`) | `125,504` | `117,312` | `W_vel.weight` (`8,192`) is a legacy buffer for `strict=True`; velocity uses slice `p[..., :q_dim]` |
| `predictor` (`backbone` `101,376` + `force_head` `24,672` + `dissipation_head` `24,672` + `restorative_q_head` `8,224`) | `158,944` | `158,944` | Dissipative phase-space attractor $M_1(\mathbf{z}_t, \mathbf{u}_t)$ |
| `physical_decoder` (`pos_head` `2,630` + `vel_head` `6,726`) | `9,356` | `9,356` | Decodes $\hat{\mathbf{q}}^* \in \mathbb{R}^6$, $\hat{\mathbf{p}}^* \in \mathbb{R}^6$ |
| `critic` (`critic.net` `128 -> 256 -> 128 -> 1`) | `66,049` | `0` | Training-only (`MPPIConfig.w_critic = 0.0` at test time) |
| **`SPWM` Total** | **`392,141`** | **`317,900`** | `74,241` inactive at closed-loop inference |
| `ANNWorldModel` ([ann_ablation.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/ann_ablation.py)) | `392,141` | `317,900` | Continuous leaky rate (`__call__` returns `Dict[str, Tensor]`) |
| `ContinuousRSSM` ([baselines.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/baselines.py)) | `380,460` | `380,460` | Deterministic `d=256` GRU + `s=32` stochastic state (`8,256` `raw_std` output weights discarded in deterministic inference) |
| `DeepGRUPredictor` ([baselines.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/baselines.py)) | `338,868` | `338,868` | `hidden_dim=164` 2-layer GRU (`encoder` `15,904`, `gru1` `147,600`, `gru2` `162,360`, `kin_to_feat` `1,664`, `decoder` `11,340`) |
| `ContinuousConvLSTM` ([baselines.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/baselines.py)) | `296,892` | `296,892` | Spatial 2D ConvLSTM (`conv_channels=(32,48)`, `hidden=56`; open-loop rollout only, not wired to PIL MPPI) |

### 4.2 Core Forward Equations (`file:line` Verified)
1. **Dual-Stream Visual Encoder ([encoder.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py#L43-L160)):**
   `feat = conv(E_t) + fusion_head(wrist_stem(E_t[:, 2:4]))` $\in \mathbb{R}^{B \times 16 \times 8 \times 8}$. `SpatialSoftmax` computes 2D centers of mass $\mathbf{k}_t \in [-1, 1]^{32}$, and `proj` (`Linear(32, 128) + LayerNorm + SiLU`) yields $\mathbf{x}_t \in \mathbb{R}^{128}$.
2. **Multi-Timescale ALIF Memory ([neurons.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/neurons.py#L40-L213), [memory.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/memory.py#L27-L109)):**
   Two pools of `128` ALIF neurons (`N=256`) with clamped decays $\beta_{\text{mem}} \in [0.50, 0.98]$ (init `0.80`) and $\beta_{\text{adapt}} \in [0.80, 0.995]$ (init `0.90` reactive, `0.985` deep), $\gamma = 0.18$, $V_{\text{th},0} = 1.0$, and `AtanSurrogate` ($\alpha = 2.0$):
   $$\mathbf{V}_t = \beta_{\text{mem}}\mathbf{V}_{t-1} + \mathbf{I}_{\text{soma}, t} - \mathbf{s}_{t-1}V_{\text{th},0}, \quad \mathbf{A}_t = \beta_{\text{adapt}}\mathbf{A}_{t-1} + \mathbf{s}_{t-1}, \quad \mathbf{s}_t = \Theta(\mathbf{V}_t - (V_{\text{th},0} + \gamma \mathbf{A}_t))$$
3. **Spiking Latent Dynamics & Anti-Evanescence Stasis Gate ([dynamics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py#L80-L210), [rollout.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/rollout.py#L32-L35)):**
   $$\mathbf{I}_{\text{soma}, t} = \mathbf{W}_{\text{in}}\boldsymbol{\varepsilon}_t + \mathbf{W}_p\mathbf{p}_{t-1} + \mathbf{W}_q\mathbf{q}_{t-1} + \mathbf{W}_u\mathbf{u}_t, \quad \mathbf{p}_t = \operatorname{LayerNorm}\big(\mathbf{W}_{sp}\mathbf{s}_t + \mathbf{W}_m \tanh(\mathbf{V}_t)\big)$$
   Velocity readout slices $\mathbf{v}_p = \mathbf{p}_t[:, :32]$, damped by event activity $a_t = \max(\operatorname{mean}|E_{\text{global}}|, \operatorname{mean}|E_{\text{wrist}}|)$ (`stasis_sigma=0.003`, `activity_threshold=0.001`, `max_k_gate=0.35`, `gate_ramp_steps=4`, `max_correction_norm=0.50`):
   $$\mathbf{v}_{p,\text{eff}} = \tanh\!\left(\frac{a_t}{\sigma_{\text{stasis}}}\right)\mathbf{v}_p, \quad a_{\text{eff}} = \max(0, a_t - \theta_{\text{act}}), \quad K_{\text{gate}} = \operatorname{ramp}(g_{\text{steps}}) \cdot \left(1 - e^{-\operatorname{softplus}(g)\cdot a_{\text{eff}}}\right) K_{\max} \cdot \tanh^2\!\left(\frac{a_{\text{eff}}}{\sigma_{\text{stasis}}}\right)$$
   $$\mathbf{q}_t = \mathbf{q}_{t-1} + \Delta t\,\mathbf{v}_{p,\text{eff}} + K_{\text{gate}}\,\operatorname{clip\_norm}(\mathbf{k}_t - \mathbf{q}_{t-1}, 0.50)$$
   *(Note: $\mathbf{q}_t$ is not explicitly clamped to $[-1, 1]$ after integration; only $\mathbf{k}_t$ is bounded via `SpatialSoftmax`.)*
4. **Action-Conditioned Attractor $M_1$ ([predictors.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py#L69-L115)):**
   $$\mathbf{h}_t = \operatorname{Backbone}([\mathbf{z}_t, \mathbf{u}_t]), \quad \gamma_t = \sigma(\mathbf{W}_{\text{diss}}\mathbf{h}_t + b_{\text{diss}}), \quad (b_{\text{diss},0} = -2.0)$$
   $$\hat{\mathbf{p}}_{t+1} = (1 - \Delta t\,\gamma_t)\mathbf{p}_t + \mathbf{f}_p(\mathbf{h}_t), \quad \hat{\mathbf{q}}_{t+1} = \mathbf{q}_t + \Delta t\big(\hat{\mathbf{p}}_{t+1}[:, :32] + \mathbf{f}_q(\mathbf{h}_t)\big)$$

---

## 5. Design Decisions & Negative Ablations

Key architectural rejections verified against the historical engineering log ([History.md](file:///Users/Mattia/Desktop/Studies/Temp/History.md)) and final codebase:

1. **Strict Port-Hamiltonian Skew-Symmetric Recurrence (`J - R`) Rejected:** Constraining recurrence to $\mathbf{W}_{\text{rec}} = \mathbf{J} - \mathbf{R}$ ($\mathbf{J}$ skew-symmetric, $\mathbf{R} = \operatorname{diag}(\operatorname{softplus}(\mathbf{r}) + 10^{-4})$) with hard boundary reflections ($|\mathbf{q}| \ge 0.98$) disrupted flow continuity and caused catastrophic long-horizon freezing: in `v5.3/v5.4`, $H=1$ position error (`0.0919 rad`) collapsed to `0.8942 rad` at $H=50$ and `1.1479 rad` at $H=100$; in `v6.1` (momentum-only $\mathbf{J} - \mathbf{R}$ on $\mathbf{p} \in \mathbb{R}^{96}$), $H=1$ position error degraded `+280%` (`0.145 -> 0.552 rad`) and $H=100$ latent MSE doubled (`0.653 -> 1.136`). Consequently, [dynamics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py#L81-L84) uses unconstrained linear projections and [predictors.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py#L20-L29) learns a state-dependent dissipative field ($\gamma_t, \mathbf{f}_p, \mathbf{f}_q$).
2. **Zero-Lag `PhysicalDecoder` (`alpha_pos=0.0, alpha_vel=0.0`) Over EMA Filtering:** A 9-configuration sweep over `decoder_pos_alpha` $\in \{0.00, 0.20, 0.60\}$ and `decoder_ema_alpha` $\in \{0.00, 0.20, 0.60\}$ (`v0.7.2`) showed causal EMA filtering introduced phase lag; zeroing both (`config.py:143-144`, [predictors.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py#L136-L137,L221-L249)) cut mean position/velocity error by `2.1%` (`0.2114 -> 0.2069 rad`, `0.1783 -> 0.1745 rad/s`) and final velocity error by `12.5%` (`0.2187 -> 0.1914 rad/s`).
3. **Stasis Noise Floor `activity_threshold = 0.001` Over `0.012`:** Setting `activity_threshold = 0.012` (`v0.6.2`) triggered aggressive stasis clamping on low-event frames, attenuating joint excursion (`v0.5.4`: `-81%` shoulder pan motion) and raising Phase 1 `val_loss` by `+71%` (`0.00423 -> 0.00722`). Recalibrating to `0.001` ([config.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py#L148), [dynamics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py#L43-L44,L100-L123)) with channel-max activity `torch.maximum(act_global, act_wrist)` ([world_model.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/world_model.py#L256-L259)) cut Phase 1 `val_loss` by `-40.4%` (`0.00722 -> 0.00430`) and avoided static attractor deadlock.

### `docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md`
> Verified loss formulations, 2-phase/4-latch curriculum, MPPI controls, and equalized baseline training regimes

# 02 — Training Curriculum, Loss Formulations & MPPI Control

---

## 1. Phase 0: Grounded State Estimator (`train_representation`)

Implemented in [RepresentationPhaseStrategy](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/representation.py#L15-L123) (`123` lines) using config [results/spwm_v0_12_4_train_representation/config.yaml](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_representation/config.yaml):

* **Gradient Isolation (`representation.py:21-31,47-51`):** `predictor` and `critic` are frozen (`requires_grad=False`); only `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` train. `dynamics.action_proj` is zero-initialized ([dynamics.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py#L84)).
* **Optimizer & Cosine Schedule (`representation.py:34-45,100-103`):** Adam across 5 parameter groups:
  * `encoder`, `dynamics` (non-`beta_`), `sensory_predictor`: `lr = 1e-4` (`TrainingConfig.learning_rate`), `weight_decay = 5e-4`.
  * `dynamics` (`beta_*` leak parameters): `lr = 0.5 * 1e-4 = 5e-5`, `weight_decay = 0.0`.
  * `physical_decoder`: `probe_lr = 2e-4`, `probe_weight_decay = 1e-4`.
  * Cosine annealing scales all parameter group LRs from $1.0\times \to 0.20\times$ across `epochs = 160` (`grad_clip_norm = 0.8`).
* **Phase 0 Total Loss ([losses.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/losses.py#L124-L200)):**
  $$\mathcal{L}_{\text{Phase 0}} = \lambda_{\text{kin}}\mathcal{L}_{\text{kinematic}} + \lambda_{\text{deriv}}\mathcal{L}_{\text{deriv}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{kin\_sm}}\mathcal{L}_{\text{kin\_smooth}} + \lambda_{\text{spk}}\mathcal{L}_{\text{spike\_bal}} + \lambda_{\text{sens}}\mathcal{L}_{\text{sensory}}$$
  with defaults (`LossConfig`, `config.py:303-315`): $\lambda_{\text{kin}}=5.0$, $\lambda_{\text{deriv}}=1.0$, $\lambda_{\text{smooth}}=0.1$, $\lambda_{\text{kin\_sm}}=0.5$, $\lambda_{\text{spk}}=0.25$, $\lambda_{\text{sens}}=1.0$:
  1. `l_kinematic`: $\operatorname{MSE}_{\text{sum}}(\hat{\mathbf{q}}^*, \mathbf{q}^*) + \operatorname{MSE}_{\text{sum}}(\hat{\mathbf{p}}^*, \mathbf{p}^*)/\sigma_v^2 + 40.0\big(\operatorname{MSE}_{\text{sum}}(\mathbf{p}_{\text{ee}}, \mathbf{p}_{\text{ee}}^*) + 0.5\,\operatorname{MSE}_F(\mathbf{R}_{\text{ee}}, \mathbf{R}_{\text{ee}}^*)\big)$ via [DifferentiableForwardKinematics](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/kinematics.py#L52).
  2. `l_derivative`: $\operatorname{MSE}_{\text{sum}}(\mathbf{q}_{t+1}-\mathbf{q}_t, \Delta t\,\mathbf{p}_t[:, :q_{\text{dim}}])$.
  3. `l_smoothness`: $\operatorname{SmoothL1}_{\beta=0.1,\text{sum}}(\mathbf{p}_{t+1}, \mathbf{p}_t)$.
  4. `l_kinematic_smooth` (`losses.py:95-121,166-168`): Calls `_decoded_kinematic_temporal_loss(..., reduce_joint_sum=True)` summing across joints: finite-difference velocity loss + `0.25 * vel_mid` consistency + `0.5 * ddq_loss`, where `ddq_pred = (pos_pred[:, 2:] - 2*pos_pred[:, 1:-1] + pos_pred[:, :-2]) / dt` (dividing by `dt`, not `dt^2`).
  5. `l_spike_balance` (`losses.py:171-180`): $\sum_i (\bar{s}_i - 0.12)^2 + 4.0\sum_i \operatorname{ReLU}(0.02 - \bar{s}_i)^2 + 2.0\sum_i \operatorname{ReLU}(\bar{s}_i - 0.45)^2$ (`target_spike_rate = 0.12`).
  6. `l_sensory`: $\operatorname{MSE}_{\text{sum}}(\boldsymbol{\varepsilon}_t, \mathbf{0})$.
* **Scoped Checkpoint Handoff ([checkpoints.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/checkpoints.py#L35-L58)):** `load_representation_checkpoint` loads `encoder`, `sensory_predictor`, `dynamics`, and `physical_decoder` each with `strict=True`, leaving `predictor` and `critic` untouched.

---

## 2. Phase 1: Action-Conditioned Attractor $M_1$ & 4-Phase Curriculum (`train_m1`)

Implemented in [M1PhaseStrategy](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/m1_nominal.py#L17-L130) (`130` lines) and orchestrated by [BaseTrainer](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py#L190-L316) using [results/spwm_v0_12_4_train_m1/config.yaml](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/config.yaml):

* **Frozen Representation & Optimizer (`m1_nominal.py:23-29`, `engine.py:236-239`):** Representation stays in `eval()` with `requires_grad=False`. Adam updates **both** `model.predictor` and `model.critic` at `lr = 1e-4`, `weight_decay = 5e-4`, dropping dynamically to `lr = 5e-5` once Latch Phase `4` engages (`engine.py:236`).
* **Phase 1 Loss ([losses.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/losses.py#L203-L298)):**
  $$\mathcal{L}_{\text{Phase 1}} = \mathcal{L}_{M_1,\text{latent}} + 1.0\,\mathcal{L}_{\text{kin\_M1}} + w_{\text{critic}}\,\mathcal{L}_{\text{critic}}$$
  * **Latent Rollout ($\mathcal{L}_{M_1,\text{latent}}$, `losses.py:228-241,265-267`):** Evaluates `.mean()` across `(B, H, D)` (dividing by `B * H * 32` for `q` and `B * H * 96` for `p`), with half-open contact-heft weighting where `h0, h_end` are the first/last `phase_flags == 1` indices:
    `step_weights[b, max(0, h0 - 5) : min(H, h_end + 10)] = 2.0` ($\mathbf{q}$) and `vel_step_weights[b, max(0, h0) : min(H, h_end + 12)] = 3.0` ($\mathbf{p}$), weighted by $\lambda_{\text{position}}=1.0, \lambda_{\text{velocity}}=1.5, \lambda_{\text{latent}}=1.0$.
  * **DAgger Mixed BPTT (`losses.py:246-263`, [replay_buffer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/replay_buffer.py#L85-L109)):** Concatenates open-loop transitions with closed-loop transitions sampled from `ClosedLoopBuffer` (FIFO rolling window `max_eval_generations = 3`).
  * **Decoded Kinematic Loss ($\mathcal{L}_{\text{kin\_M1}}$, `losses.py:269-286`):** $\operatorname{MSE}(\hat{\mathbf{q}}^*, \mathbf{q}^*) + 0.5\,\operatorname{MSE}(\hat{\mathbf{p}}^*, \mathbf{p}^*) + 40.0(\mathcal{L}_{\text{ee\_pos}} + 0.5\,\mathcal{L}_{\text{ee\_rot}}) + 0.5\,\mathcal{L}_{\text{kin\_smooth}}$ (`reduce_joint_sum=False`).
  * **Monte Carlo Value Critic ($\mathcal{L}_{\text{critic}}$, `losses.py:54-92,264`):** Discounted return recursion $V^*_t = \|\mathbf{p}_{\text{ee}}-\mathbf{p}^*_{\text{ee}}\|_2 + (1 - \cos\theta_{\text{err}}) + 0.95\,V^*_{t+1}$, weighted by $w_{\text{critic}} = 0.2$ in Phases 1–2 and $1.0$ (`5.0 * 0.2`) in Phases 3–4.
* **4-Phase Irreversible Latch Curriculum ([noise.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/noise.py#L46-L154), [engine.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py#L194-L200,L234-L265)):**
  * `BaseTrainer.__init__` overrides `LatchCurriculumState`'s standalone class defaults (`0.01/0.008/0.4`) with `TrainingConfig` values (`phase1_loss_threshold=0.10`, `phase2_loss_threshold=0.095`, `target_alpha=0.25`, `noise_ramp_epochs=20`, `tier2_frequency=5`) and always injects `batch["curriculum_alpha"]` (bypassing `compute_mppi_noise_alpha` and `mppi_noise_warmup_epochs=10` in `M1PhaseStrategy:58`). Latch transitions depend strictly on `loss_m1_latent`:
    1. **Phase 1 (Warmup Nominal):** `w_t = 0.60`, `alpha = 0.0`, `dagger_ratio = 0.0`. Latches to Phase 2 when `loss_m1_latent < 0.10`.
    2. **Phase 2 (Sensory Fade-Out):** Linearly decays `w_t` from `0.60 -> 0.0` over `7` epochs (`alpha = 0.0`). Latches to Phase 3 when `w_t == 0.0` and `loss_m1_latent < 0.095`.
    3. **Phase 3 (AR(1) Noise Ramp):** `w_t = 0.0`, linearly ramps `alpha` from `0.0 -> 0.25` over `20` epochs via `sample_ar1_noise` ($\beta = 0.90, \sigma = 0.08$, shoulder `1.6x`, elbow `1.4x`). Latches to Phase 4 when `alpha >= 0.25`.
    4. **Phase 4 (DAgger + Fine-Tuning):** `lr = 5e-5`, `w_t = 0.0`, `alpha = 0.25`, ramps `dagger_ratio` `0.10 -> 0.50` over `2 * tier2_frequency = 10` epochs.

---

## 3. Open-Loop Soft-Fade vs. Closed-Loop Residual Handover

1. **Open-Loop Rollout Soft-Fade ([rollout.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/rollout.py#L208-L277)):** During `phase_flags == 1`, `rollout_latent` blends $\mathbf{z}_{t+1} = (1 - w_t)\mathbf{z}_{\text{pred}} + w_t \mathbf{z}_{\text{grounded}}$ where $w_t = \operatorname{clamp}(g_{\text{steps}} / 4, 0, 1) \cdot w_{\max}$ ($w_{\max} = 0.60$, reaching `0.15 -> 0.30 -> 0.45 -> 0.60` only while `act_val > activity_threshold` in `dynamics.py:192-195`). Upon entering `phase_flags == 2`, `fade_counter == 1` applies $0.50\,w_{\max} = 0.30$, then drops to `0.00`.
2. **Closed-Loop PIL Residual-Gated Handover ([benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py#L234-L256,L415-L455), [mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py#L381-L394)):** During $\phi_t = 1$, causal `OnlineEventCameraStream` produces $\mathbf{z}_{\text{obs}}$ via `step_grounded` (for `SPWM` / `ANN-Rate`) or `step_online` (for `ContinuousRSSM` / `DeepGRUPredictor`). Under the shared residual-gated regime (`--baseline-handover-mode residual_gated`, default), decoded SE(3) errors `dist_pred`, `dist_obs`, `rot_obs` against `target_final_mat` govern $w_t$ identically across all architectures:
   * If `dist_pred > tau` (`0.026 m` default; passed via `--handover-dist-threshold`, while `mppi_scorer.py:388` hardcodes `0.026`) **and** `dist_obs < 0.025 m` **and** `rot_obs < 0.049 rad`: $w_t = 0.85$.
   * Else if `dist_obs < dist_pred` and `t <= t_c + 2` (where $t_c = \texttt{p1\_indices[0]} = t_{\text{contact}} - 7$ is the $\phi_t=1$ sensory-window onset and $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$): $w_t = 0.40 \cdot \operatorname{clip}((\text{dist\_pred} - 0.032)/0.04,\, 0,\, 1)$.
   * Otherwise: $w_t = 0.00$ (rejecting corrupted post-$t_c$ (post-attachment) event bursts where `dist_obs` is `0.16–0.22 m` while `M1` remains within `1.39–1.74 cm` median).

---

## 4. MPPI Control & Two-Tiered Validation

* **Latent MPPI Cost ([mppi_controller.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_controller.py#L406-L551), [MPPIConfig](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py#L341-L370)):** Evaluates `K=256`, `H=10`, `N_iter=3`, `temperature=0.20`, `nominal_update_gamma=0.60`, `max_action_delta=0.10`, `noise_sigma=0.08` (`1.6x` shoulder, `1.4x` elbow, `noise_beta=0.90`):
  $$\mathcal{C}_k = \sum_{h=1}^{H}\Big[w_{\text{pos}}\|\mathbf{p}_{\text{ee},h}-\mathbf{p}^*_h\|_2 + w_{\text{rot}}\big((1-\cos\theta_h) + 0.5\,\theta_h\big) + w_{\text{jerk}}\|\Delta\mathbf{u}_h\|_2^2 - w_{\text{prog}}\,\alpha_h\,(\Delta\mathbf{p}_h\cdot \hat{\mathbf{d}}_h) + w_{\text{damp}}(1-\alpha_h)\|\Delta\mathbf{p}_h\|_2\Big]$$
  $$+\; w_{\text{j\_acc}}\sum \|\Delta^2\hat{\mathbf{q}}_h\|_2^2 + w_{\text{j\_lim}}\sum \big(\operatorname{ReLU}(\hat{\mathbf{q}}_h - \mathbf{q}_{\max}) + \operatorname{ReLU}(\mathbf{q}_{\min} - \hat{\mathbf{q}}_h)\big)^2 + \gamma^H (w_{\text{critic}}\,w_{\text{pos}}\,V(\mathbf{z}_H))$$
  with `w_pos=80.0`, `w_rot=100.0`, `w_jerk=4.0`, `w_progress=50.0`, `w_damping=12.0`, `w_joint_accel=15.0`, `w_joint_limit=50.0`, `w_critic=0.0`, `gamma_discount=0.95`, and deadband factor $\alpha_h = \operatorname{clamp}((e_{\text{pos},h} - 0.03)/0.10, 0, 1)$.
  * **2-Stage SE(3) Lookahead (`benchmark_mppi_control.py:366-370`):** Horizon steps `t + tau <= t_c` target `target_grasp_mat`, while `t + tau > t_c` target `target_final_mat`.
  * **Grasp Success Criteria (`benchmark_mppi_control.py:497-513`, `mppi_scorer.py:21-61`):** An episode succeeds iff (i) end-effector position error stays `< 3.0 cm` across the final 3-step settling window, (ii) terminal $\text{SO}(3)$ orientation error `< 0.05 rad`, and (iii) terminal joint velocity $\|\dot{\mathbf{q}}(t_{\text{grasp}})\|_2 < \tau_{\text{vel}}$, where $\tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)\,\text{rad/s}$.
* **Two-Tiered Validation & EMA ([mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py#L541-L699), [engine.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/engine.py#L444-L466,L649-L651)):**
  * Tier 1 open-loop MSE (`metrics.py:26-51`) is logged every epoch but decoupled from model selection.
  * Tier 2 `MPPIStructuralScorer` runs every `tier2_frequency = 5` epochs on `N=6` validation trajectories (`[3200..3205]`) with deterministic RNG seed `42000 + traj_id`, evaluates EMA weights (`ema_decay = 0.990`), and decays expert trajectories in `ClosedLoopBuffer` as `4 -> 3 -> 2 -> 1 -> 0` across Phase 4 evaluations (`mppi_scorer.py:546-549`). Emits `mppi_score`, `mppi_grasp_rot_err_rad`, and `mppi_final_rot`.
  * When `is_best=True`, `save_checkpoint` writes **both** `model.pt` and `best_model.pt` (`engine.py:649-651`). With `classic_patience: false`, training saves metrics/artifacts when `tier2_patience=5` triggers and continues to `epochs`.

---

## 5. Training Replication Commands

```bash
# 1. Phase 0: Representation Estimator (160 epochs)
.venv/bin/python experiments/train.py \
  --config results/spwm_v0_12_4_train_representation/config.yaml \
  --mode train_representation --epochs 160 --output-dir results/spwm_v0_12_4_train_representation

# 2. Phase 1: Action-Conditioned Attractor M1 (240 epochs, Seed 42)
.venv/bin/python experiments/train.py \
  --config results/spwm_v0_12_4_train_m1/config.yaml \
  --mode train_m1 --epochs 240 --weighted-sampling \
  --representation-checkpoint results/spwm_v0_12_4_train_representation/model.pt \
  --output-dir results/spwm_v0_12_4_train_m1

# 3. Multi-Seed SPWM (Seeds 42, 101, 2024; Seed 2024 resumed to 260 epochs in results/multiseed/seed_2024_m1)
.venv/bin/python experiments/train_multiseed.py \
  --seeds 101 2024 --rep-epochs 160 --m1-epochs 240 --output-dir results/multiseed

# 4. Parameter-Matched ANN-Rate Ablation (per seed, 160 rep + 240 M1 epochs)
.venv/bin/python experiments/train_ann_ablation.py --seed 42 --output-dir results/ann_ablation

# 5. Equalized Continuous Recurrent Baselines (Multi-Step SE(3) + AR(1) + ClosedLoopBuffer DAgger)
.venv/bin/python experiments/train_baselines.py \
  --config results/spwm_v0_12_4_train_m1/config.yaml \
  --checkpoint results/spwm_v0_12_4_train_m1/model.pt \
  --m1-metrics results/spwm_v0_12_4_train_m1/metrics_spwm_v0_12_4.json \
  --models deep_gru --seed 101 --epochs 80 --patience 20 --lr 3e-4 \
  --enable-dagger-ar1 --ar1-warmup-epochs 10 --ar1-ramp-epochs 20 --ar1-target-alpha 0.25 \
  --dagger-start-epoch 15 --dagger-frequency 5 --device mps --output-dir results/multiseed_rssm/seed_101
```

---

## 6. Baselines & Three PIL Operating Regimes

1. **Reactive Behavior Cloning (`ReactiveBCPolicy`, `322,210` params; [benchmark_classical_and_bc.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_classical_and_bc.py#L44-L105,L295-L373)):**
   * **Architecture:** Dual-stream `EventEncoder` (`15,904` params, `4x32x32 -> 128`) + `GRUCell(134, 242)` (`274,428` params on `[enc_out, prev_action]`) + `MLP(242 -> 128 -> SiLU -> 6 -> Tanh)` (`31,878` params).
   * **Training:** `50` epochs on `[0, 3200)` / val `[3200, 3600)`, `batch_size=32`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(eta_min=1e-5)`, `clip_grad_norm_(1.0)`, scheduled sampling `ss_prob = min(0.5, 0.5 * ep / 50)`, and loss $\mathcal{L} = \operatorname{MSE}(\hat{u}, u) + 0.1\,\frac{1}{B(T-1)A}\sum_{t=1}^{T-1}\|\hat{u}_t - \hat{u}_{t-1}\|_2^2$.
2. **Privileged State-Oracle Jacobian-IK + PD (`mode == "fixed_pd"`; [benchmark_classical_and_bc.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_classical_and_bc.py#L108-L138,L231-L242)):**
   * Reads `q_curr = stepper.q` directly without vision (`OnlineEventCameraStream` is never called) and uses oracle `t_c` for `payload_attached = (t >= t_c)`.
   * Computes 6D SE(3) error `[p_err, 0.35 * rot_err]` via `fk.forward_homogeneous(q_curr)`, autograd Jacobian $J \in \mathbb{R}^{6 \times 6}$, and Levenberg–Marquardt update $dq = J^\top(JJ^\top + \lambda^2 I_6)^{-1}\text{err}_{6\text{d}}$ ($\lambda = 0.05$, $\lambda^2 = 0.0025$), setting $q_{\text{des}} = \operatorname{clamp}(q_{\text{curr}} + 0.65\,dq, -3.14, 3.14)$ and $\dot{q}_{\text{des}} = \mathbf{0}$ with fixed gains `kp = 18.0`, `kd = 0.80` calibrated to `mean_calib_mass = 2.85 kg`.
3. **Equalized Baseline Training & 3 PIL Operating Regimes (`ContinuousRSSM` & `DeepGRUPredictor`; [train_baselines.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/train_baselines.py), [benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py#L160-L193,L415-L445)):**
   * **Equalized AR(1) + DAgger Training (`--enable-dagger-ar1`):** Perturbs 80-step rollout actions with MPPI-matched `sample_ar1_noise` ($\beta=0.90$, $\sigma=0.08$, shoulder `1.6x`, elbow `1.4x`, ramped $\alpha \to 0.25$), collects `N=6` stratified validation MPPI trajectories every `5` epochs into a 3-generation FIFO `ClosedLoopBuffer`, and mixes BPTT replay (`dagger_ratio` `0.10 -> 0.50`) alongside teacher-forcing + 80-step SE(3) rollout supervision.
   * **Causal Candidate Update (`step_online`) During $\phi_t = 1$:**
     * **`ContinuousRSSM`:** `h_next = gru_cell([s, u_pd_t], h)`, `e_step, _ = encoder(ev_online)`, `post_mu, _ = _get_stats(posterior_net([h_next, e_step]))`, `z_obs = [h_next, post_mu]`.
     * **`DeepGRUPredictor`:** `e_step, _ = encoder(ev_online)`, `h1_next = gru1([e_step, u_pd_t], h1)`, `h2_next = gru2(h1_next, h2)`, `z_obs = [h1_next, h2_next]`.
   * **Three Evaluated PIL Operating Regimes (`--baseline-handover-mode`):**
     * **`open_loop` (`--disable-handover`, $w_t \equiv 0$):** Evolves strictly via prior/feedback dynamics without online `ev_online` updates.
     * **`ungated` (`--baseline-handover-mode ungated`, $w_t \equiv 1.0$):** Forces unconditional replacement `curr_z = z_obs` on every $\phi_t=1$ step.
     * **`residual_gated` (`--baseline-handover-mode residual_gated`, default):** Applies `_compute_residual_handover_weight(fk, physical_decoder, z_pred, z_obs, target_final_mat, t, t_c)` identically to `SPWM` and `ANN-Rate` (`curr_z = (1 - w_t)*z_pred + w_t*z_obs`).

### `docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md`
> Verified SSOT benchmarks, residual-gated baseline fairness, 3-seed ablation, neuromorphic standby energy framing, and replication commands

# 03 — Benchmarks, Verified Evidence & Replication Runbook

All closed-loop control metrics below were executed under causal **Physics-in-the-Loop (PIL)** (`ArticulatedArmPhysicsStepper`, rigid palm-latch `payload_attached = (t >= t_c)` where $t_c = \texttt{p1\_indices[0]} = t_{\text{contact}} - 7$ is the $\phi_t=1$ sensory-window onset and $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$, causal `OnlineEventCameraStream`, zero `z_gt_next` leakage) on `50` unseen test episodes (`[3600, 3649]`) with `K=256, H=10, N_iter=3`, slew clamp `0.10`. An episode counts as a grasp success iff end-effector error stays `< 3.0 cm` across the final 3-step settling window, terminal $\text{SO}(3)$ rotation error `< 0.05 rad`, and terminal joint velocity $\|\dot{\mathbf{q}}(t_{\text{grasp}})\|_2 < \tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)\,\text{rad/s}$.

---

## 0. Checkpoint Provenance & Equalized Training Budgets

`model.pt` and `best_model.pt` in `results/spwm_v0_12_4_train_m1/` are distinct files on disk (`sha256` prefixes `d358cccbd678` vs `f5cabfca3635` due to non-weight metadata), **but both store identical Epoch 200 EMA `model_state` tensors (`0` weight diffs)**; SPWM, ANN-Rate, and equalized baseline checkpoints are documented below:

| Model / Seed | Checkpoint File | Epochs (Best) | Wall-Clock (`MPS`) | Primary Consumers |
| :--- | :--- | :---: | :---: | :--- |
| `SPWM` (`42`) | [results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/checkpoint_phase_4.pt) | `160+240` (`240`) | `10.40 h` | Table 2 (`N=25` Val-Sel.), Table 3 (Stages 3a–4b), Table 4, `forensic` |
| `SPWM` (`42`) | [results/spwm_v0_12_4_train_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/model.pt) / `best_model.pt` | `160+240` (`200`) | `10.40 h` | Table 1, Table 2 (`N=6`), Table 5 |
| `SPWM` (`42`) | `checkpoint_phase_1.pt` | `181` | — | Table 3 (Stages 1–2) |
| `SPWM` (`101`) | [results/multiseed/seed_101_m1/checkpoint_phase_1.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_m1/checkpoint_phase_1.pt) | `160+240` (`96`) | `5.86 h` | Table 2 (`N=25` Val-Sel.), Table 3 (Stages 1, 2, 3b, 4a, 4b), Table 4, `forensic` |
| `SPWM` (`101`) | [results/multiseed/seed_101_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_101_m1/model.pt) / `checkpoint_phase_4.pt` | `80` / `240` | `5.86 h` | Table 1, Table 2 (`N=6`), Table 3 (Stage 3a), Table 5 |
| `SPWM` (`2024`) | [results/multiseed/seed_2024_m1/checkpoint_phase_2.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_m1/checkpoint_phase_2.pt) | `160+260` (`102`) | `6.41 h` | Table 2 (`N=25` Val-Sel.), Table 3 (Stages 3b, 4a, 4b), Table 4, `forensic` |
| `SPWM` (`2024`) | [results/multiseed/seed_2024_m1/model.pt](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/seed_2024_m1/model.pt) / `checkpoint_phase_1.pt` / `phase_4.pt` | `130` / `94` / `260` | `6.41 h` | Table 1, Table 2 (`N=6`), Table 3 (Stages 1–3a), Table 5 |
| `ANN-Rate` (`42/101/2024`) | `results/ann_ablation/seed_{42,101,2024}_m1/model.pt` | `160+240` | — | Tables 1, 2, 4, 5, `forensic` |
| `Continuous RSSM` (`42/101/2024`) | `results/baselines/continuous_rssm/model.pt`, `results/multiseed_rssm/seed_{101,2024}/continuous_rssm/model.pt` *(60-ep backups in `model_pre_dagger_60ep.pt`)* | `60+16=76` (`61`), `60+16=76` (`61`), `60+17=77` (`62`) | `46.0+13.0 m`, `44.0+12.9 m`, `44.0+13.8 m` | Tables 1, 2 (Open-Loop, Un-Gated, Residual-Gated), `forensic` |
| `Deep GRU` (`42/101/2024`) | `results/baselines/deep_gru/model.pt`, `results/multiseed_rssm/seed_{101,2024}/deep_gru/model.pt` *(backups in `model_pre_dagger_60ep.pt` / `model_pre_equalized_18ep.pt`)* | `59+40=99` (`90`), `80` (`68`), `80` (`80`) | `46.6+33.8 m`, `67.1 m`, `66.8 m` | Tables 1, 2 (Open-Loop, Un-Gated, Residual-Gated), Table 4 (Residual-Gated), `forensic` |
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
  2. Once equalized across all 3 seeds (`99 / 80 / 80` epochs with AR(1)+DAgger) and gated identically, `Deep GRU` achieves `76.7 ± 5.0%` PIL success (`77.3 ± 4.2%` open-loop)—exceeding `SPWM` (`60.7 ± 2.3%`) in nominal grasp rate and smoothness (`|Δ²q| = 0.00449`), **but** (i) violates the `50.0 ms` (`20 Hz`) real-time budget (`57.7 ms/step` vs `SPWM` `26.6 ms/step`), (ii) suffers `9.0×` higher $H=100$ open-loop velocity MSE (`0.08716` vs `0.00971 rad²/s²`), (iii) **collapses under zero-shot mechanical perturbations** (`23.3 ± 12.9%` at `4.80 kg` payload, `26.7 ± 23.9%` at `0.80×` friction, `30.0 ± 15.1%` at `1.20×` friction vs. `SPWM` `44.7 ± 13.3% / 52.7 ± 13.3% / 48.0 ± 19.3%`, trailing `SPWM` by `+18.0 to +26.0 pp`), and (iv) consumes `22.1×` higher standby core energy / `1.59×` higher E2E step energy.
  3. Compared against parameter-matched `ANN-Rate` (`37.3 ± 9.5%`), `SPWM` retains a **`+23.4 pp`** 3-seed PIL gain (`60.7%` vs `37.3%`) and **`+24.7 pp`** under `1.0%` sensory noise (`54.7 ± 1.2%` vs `30.0 ± 3.5%`, `2.0×` better relative retention) isolating discrete ALIF spike thresholding.

---

## 3. Table 3 — Full 3-Seed 6-Row SPWM PIL Ablation (`[3600, 3649]`, Seeds `42, 101, 2024`)

**Evidence:** [ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json) (`corl_final_verification.disentangled_ablation_3seed`), `results/ablations/multiseed/stage*_seed{42,101,2024}/mppi_benchmark_metrics.json`.

| Ablation Stage (Seeds `42 / 101 / 2024`) | Per-Seed Succ (%) | 3-Seed PIL Succ (%) | Min Err (cm) | Term Err (cm) | $\|\Delta^2 q\|$ | $\|\Delta^3 q\|$ | Flips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. SPWM-Base (`checkpoint_phase_1.pt`, Unsmoothed) | `62 / 44 / 30` | `45.3 ± 16.0` | `1.03 ± 0.21` | `2.50 ± 0.11` | `0.00983` | `0.00765` | `23.03` |
| 2. + Smooth MPPI Controls (`checkpoint_phase_1.pt`) | `62 / 44 / 48` | `51.3 ± 9.5` | `1.02 ± 0.08` | `2.41 ± 0.13` | `0.00991` | `0.00705` | `21.99` |
| 3a. + Phase 3/4 Curriculum (AR(1)+DAgger, `checkpoint_phase_4.pt`, No Lookahead/Handover) | `56 / 18 / 40` | `38.0 ± 19.1` | `1.53 ± 0.59` | `3.27 ± 1.13` | `0.00589` | `0.00354` | `11.17` |
| 3b. + $N=25$ Validation Checkpoint Selection (Val-Sel., No Lookahead/Handover) | `56 / 44 / 48` | `49.3 ± 6.1` | `1.00 ± 0.16` | `2.29 ± 0.23` | `0.00921` | `0.00659` | `21.18` |
| 4a. + 2-Stage SE(3) Lookahead ($N=25$ Val-Sel., $w_t \equiv 0$) | `58 / 62 / 62` | `60.7 ± 2.3` | `0.88 ± 0.11` | `2.01 ± 0.12` | `0.00931` | `0.00663` | `21.14` |
| **4b. + Residual Rejection Gate (Full SPWM, $N=25$ Val-Sel.)** | **`58 / 62 / 62`** | **`60.7 ± 2.3`** | **`0.88 ± 0.11`** | **`2.01 ± 0.12`** | **`0.00931`** | **`0.00663`** | **`21.14`** |

* **Transparent Disentangled Diagnostic (Stage 2 vs. Stage 3a vs. Stage 3b vs. Stage 4a):** Comparing Stage 2 (`51.3 ± 9.5%`) vs. Stage 3a (`38.0 ± 19.1%`) across all 3 seeds shows that Phase 3/4 AR(1)+DAgger replay (`checkpoint_phase_4.pt`) cuts trajectory jitter (`|Δ²q|: 0.00991 -> 0.00589`, flips `21.99 -> 11.17`) **while degrading 3-seed mean PIL success by `-13.3 pp` and doubling inter-seed std (`9.5% -> 19.1%`)**. Isolating **`N=25` validation-split (`[3200, 3224]`) checkpoint selection** *without* Lookahead (`Stage 3b`: `42: phase_4`, `101: phase_1`, `2024: phase_2`) recovers 3-seed PIL success to **`49.3 ± 6.1%`** (`56% / 44% / 48%`, `+11.3 pp` recovery and `3.1×` std reduction from checkpoint selection alone). Adding **2-Stage SE(3) Lookahead** on those same validation-selected checkpoints (`Stage 4a/4b`) then delivers a further **`+11.3 pp`** gain (`49.3% -> 60.7 ± 2.3%`, `58% / 62% / 62%`), cleanly separating validation checkpoint selection from 2-Stage SE(3) Lookahead.

---

## 4. Table 4 — 3-Seed PIL Robustness (`$N=25$` Val-Sel. SPWM vs. ANN-Rate vs. Equalized Deep GRU)

**Evidence:** [robustness_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/robustness_benchmark/robustness_metrics.json) (`42: checkpoint_phase_4.pt`), `results/robustness_benchmark_pil_spwm_seed{101,2024}/robustness_metrics.json` (`101: checkpoint_phase_1.pt`, `2024: checkpoint_phase_2.pt`), `results/robustness_benchmark_pil_ann_rate_seed{42,101,2024}/robustness_metrics.json`, `results/robustness_benchmark_pil_deep_gru_seed{42,101,2024}/robustness_metrics.json` (Equalized Deep GRU under `residual_gated` mode).

| Perturbation Condition (`50` Test Ep) | SPWM ($N=25$ Val-Sel., 3 Seeds: Succ / Min / Term) | ANN-Rate (3 Seeds: Succ / Min / Term) | Equalized Deep GRU (Residual-Gated, 3 Seeds: Succ / Min / Term) |
| :--- | :---: | :---: | :---: |
| Nominal Baseline | **`60.7 ± 2.3%`** / **`0.88`** / `2.01 cm` | `37.3 ± 9.5%` / `1.07` / `2.98 cm` | `76.7 ± 5.0%` / `1.14` / `1.90 cm` |
| Sensory Noise (`1.0%`) | **`54.7 ± 1.2%`** (`-9.9% rel`) / **`0.92`** / `2.26 cm` | `30.0 ± 3.5%` (`-19.6% rel`) / `1.04` / `3.36 cm` | `74.7 ± 3.1%` (`-2.6% rel`) / `1.52` / `1.85 cm` |
| Payload Extrapolation (`4.80 kg`) | **`44.7 ± 13.3%`** (`-26.4% rel`) / **`1.31`** / **`2.49 cm`** | `26.7 ± 16.2%` (`-28.6% rel`) / `1.36` / `3.64 cm` | `23.3 ± 12.9%` (`-69.6% rel`) / `3.21` / `4.35 cm` |
| Actuator Drift (`0.80x` Friction) | **`52.7 ± 13.3%`** (`-13.2% rel`) / **`1.31`** / **`2.39 cm`** | `30.7 ± 14.0%` (`-17.9% rel`) / `1.42` / `3.74 cm` | `26.7 ± 23.9%` (`-65.2% rel`) / `3.24` / `4.38 cm` |
| Actuator Drift (`1.20x` Friction) | **`48.0 ± 19.3%`** (`-20.9% rel`) / **`1.29`** / **`2.28 cm`** | `33.3 ± 16.7%` (`-10.7% rel`) / `1.34` / `3.51 cm` | `30.0 ± 15.1%` (`-60.9% rel`) / `3.10` / `4.19 cm` |

* **Key Robustness Findings ($N=25$ Val-Selected SPWM vs. ANN-Rate vs. Deep GRU):**
  1. **Sensory Noise (`1.0%`):** `$N=25$` validation-selected `SPWM` achieves `54.7 ± 1.2%` (`-9.9%` relative drop, reducing inter-seed std `16×` vs. `N=6` Tier-2 `55.3 ± 19.4%`), outperforming `ANN-Rate` (`30.0 ± 3.5%`, `-19.6%` relative drop) by **`+24.7 pp`** (`2.0×` better relative retention).
  2. **Zero-Shot Physical / Mechanical OOD Collapse of `Deep GRU`:** While equalized `Deep GRU` leads on in-distribution nominal (`76.7%`) and sensory-noise (`74.7%`) episodes, it **collapses by `-60.9% to -69.6%` relative under unmodeled physical perturbations** (`23.3 ± 12.9%` on `4.80 kg` payloads, `26.7 ± 23.9%` on `0.80×` friction, `30.0 ± 15.1%` on `1.20×` friction; terminal error `4.19–4.38 cm`). Because `SPWM` factorizes dynamics into dissipative `(q, p)` phase space rather than an unstructured `164`-d GRU state, `SPWM` outperforms `Deep GRU` by **`+21.4 pp`** (`44.7%` vs. `23.3%`), **`+26.0 pp`** (`52.7%` vs. `26.7%`), and **`+18.0 pp`** (`48.0%` vs. `30.0%`) across all three mechanical distribution shifts.
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
  1. `SPWM`'s primary advantage over `ANN-Rate` is **closed-loop dynamical stability and noise/dynamics robustness** (`+23.4 pp` 3-seed PIL grasp success, `+24.7 pp` and `2.0×` better relative retention under `1.0%` sensory noise on `$N=25$` validation-selected checkpoints), **not** end-to-end energy savings: saving `48.14 nJ/step` on Loihi-2 (`60.36 -> 12.22 nJ/step`) atop `1.0971 mJ/step` (`1,097,100 nJ/step`) of dense GPU MPPI rollouts is a **`0.004%`** end-to-end difference (`1.0979 mJ/step`, `21.96 mW` @ `20 Hz` for both).
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

### `results/spwm_v0_12_4_train_m1/config.yaml`
> Exact hyperparameters used for the v0.12.4 run

```yaml
project: {name:spwm_v0_12_4,seed:42}
environment: {joint_count:6,action_dim:6,height:32,width:32,dt:0.05,event_threshold:0.08}
data:
  dataset_dir: sim/ExportedTrajectories
  joint_count: 6
  action_dim: 6
  total_trajectories: 4000
  train_split: 0.8
  val_split: 0.1
  sequence_length: 105
  batch_size: 32
  standard_velocity_range: [-1.0,1.0]
  extrapolation_velocity_range: [-2.0,2.0]
  num_workers: 0
  cache_data: true
  strict_dataset: true
  pre_contact_sensory_steps: 7
  post_contact_sensory_steps: 7
  weighted_sampling: true
model:
  type: spwm
  num_keypoints: 16
  encoder_dim: 128
  encoder_conv_channels: [16,32]
  latent_dim: 128
  q_dim: 32
  p_dim: 96
  predictor_hidden_dim: 256
  ema_decay: 0.9
  decoder_ema_alpha: 0.0
  decoder_pos_alpha: 0.0
  gate_ramp_steps: 4
  max_k_gate: 0.35
  max_correction_norm: 0.5
  activity_threshold: 0.001
  stasis_sigma: 0.003
  in_channels: 4
  symplectic: true
neuron: {model:alif,threshold:1.0,beta_mem:0.8,gamma:0.18,surrogate:atan,surrogate_alpha:2.0}
memory: {num_timescales:2,timescale_dims:[128,128],betas:[0.9,0.985]}
training:
  mode: train_m1
  epochs: 240
  early_stopping_patience: 10
  batch_size: 32
  weight_decay: 0.0005
  grad_clip_norm: 0.8
  learning_rate: 0.0001
  probe_lr: 0.0002
  probe_weight_decay: 0.0001
  temporal_burn_in: 15
  representation_checkpoint: results/spwm_v0_12_4_train_representation/model.pt
  phase_horizon: 100
  tier2_frequency: 5
  tier2_trajectories: 6
  tier2_seed: 42000
  tier2_num_samples: 1024
  tier2_horizon: 10
  tier2_iterations: 3
  tier2_candidate_selection: ema
  tier2_patience: 5
  tier2_min_delta: 0.001
  metrics_storage: 5
  classic_patience: false
  ema_decay: 0.99
  mppi_noise_warmup_epochs: 10
  mppi_noise_ramp_epochs: 20
  mppi_noise_target_alpha: 0.25
  phase1_loss_threshold: 0.1
  phase2_loss_threshold: 0.095
  max_eval_generations: 3
loss:
  lambda_encoder: 1.0
  lambda_decoder: 1.0
  lambda_latent: 1.0
  lambda_sensory: 1.0
  lambda_position: 1.0
  lambda_velocity: 1.5
  lambda_kinematic: 5.0
  lambda_derivative: 1.0
  lambda_smoothness: 0.1
  lambda_critic: 0.2
  cartesian_weight: 40.0
  ee_rot_weight: 0.5
  lambda_kinematic_smooth: 0.5
  lambda_spike_balance: 0.25
  target_spike_rate: 0.12
  velocity_loss_scale: 1.0
evaluation: {rollout_horizons:[1,5,10,25,50,100]}
mppi:
  num_samples: 256
  horizon: 10
  action_dim: 6
  action_min: -1.0
  action_max: 1.0
  noise_sigma: 0.08
  shoulder_sigma_multiplier: 1.6
  elbow_sigma_multiplier: 1.4
  noise_beta: 0.9
  temperature: 0.2
  nominal_update_gamma: 0.6
  num_iterations: 3
  chunk_size: 1
  gamma_discount: 0.95
  terminal_steps: null
  w_progress: 50.0
  w_jerk: 4.0
  w_damping: 12.0
  w_rot: 100.0
  w_pos: 80.0
  w_critic: 0.0
  w_joint_accel: 15.0
  w_joint_limit: 50.0
  max_action_delta: 0.1
```

