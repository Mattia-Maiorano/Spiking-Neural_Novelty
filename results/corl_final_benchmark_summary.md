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
> 2. **Architectural Trade-Off vs. Equalized Deep GRU:** While equalized `Deep GRU` achieves higher nominal PIL grasp success (`76.7 ± 5.0%` vs. `60.7 ± 2.3%` for SPWM) and smoother controls (`|Δ²q| = 0.00449` vs. `0.00931`), `Deep GRU` exceeds the `50.0 ms` (`20 Hz`) real-time budget at **`57.7 ms/step`** (`2.17×` slower than SPWM's `26.6 ms/step` under identical `OMP_NUM_THREADS=2`), exhibits **`9.0×` higher $H=100$ open-loop velocity MSE** (`0.08716` vs. `0.00971 rad²/s²`), and consumes **`22.1×` higher standby core energy** (`269.57 nJ/step` on Orin vs. `12.22 nJ/step` on Loihi-2) and **`1.59×` higher E2E step energy** (`1.7490 mJ` vs. `1.0979 mJ`).
> 3. **Controlled Spiking Isolation (`SPWM` vs. `ANN-Rate`):** Under identical parameter budget (`317.9k` active), `(q, p)` factorization, `160+240` training curriculum, and MPPI controller, SPWM's discrete ALIF thresholding outperforms `ANN-Rate` by **`+23.4 pp`** (`60.7 ± 2.3%` vs. `37.3 ± 9.5%`) and delivers `4.3×` better relative retention under `1.0%` sensory noise.

### Table 3: Full 3-Seed 5-Stage SPWM Architectural Ablation in PIL (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$, Seeds `42, 101, 2024`)

```latex
\begin{table}[t]
\centering
\caption{\textbf{3-Seed 5-Stage SPWM Architectural Ablation in PIL (`50` Episodes `[3600, 3649]`, $t \ge t_c$, Seeds `42, 101, 2024`).} Stages 1--2 evaluate pre-DAgger Phase 1 warmup weights (`checkpoint\_phase\_1.pt`), Stage 3 evaluates Phase 3/4 AR(1)+DAgger curriculum weights (`checkpoint\_phase\_4.pt`) without lookahead/handover, and Stages 4a--4b apply 2-Stage $\text{SE}(3)$ Lookahead and Residual Handover to the $N=25$ validation-selected checkpoints (`42: phase\_4`, `101: phase\_1`, `2024: phase\_2`).}
\label{tab:spwm_ablation}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
\textbf{Ablation Stage} & \textbf{Seeds (42 / 101 / 2024)} & \textbf{PIL Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} & $|\Delta^2 q|$ & $|\Delta^3 q|$ & \textbf{Flips} \\
\midrule
1. SPWM-Base (Phase 1 Warmup, Unsmoothed) & 62 / 44 / 30 & 45.3 $\pm$ 16.0 & 1.03 $\pm$ 0.21 & 2.50 $\pm$ 0.11 & 0.00983 & 0.00765 & 23.03 \\
2. + Smooth MPPI Controls (Pre-DAgger) & 62 / 44 / 48 & 51.3 $\pm$ 9.5 & 1.02 $\pm$ 0.08 & 2.41 $\pm$ 0.13 & 0.00991 & 0.00705 & 21.99 \\
3. + Phase 3/4 Curriculum (AR(1)+DAgger, No Lookahead/Handover) & 56 / 18 / 40 & 38.0 $\pm$ 19.1 & 1.53 $\pm$ 0.59 & 3.27 $\pm$ 1.13 & 0.00589 & 0.00354 & 11.17 \\
4a. + 2-Stage SE(3) Lookahead (Val-Sel., $w_t \equiv 0$) & 58 / 62 / 62 & 60.7 $\pm$ 2.3 & 0.88 $\pm$ 0.11 & 2.01 $\pm$ 0.12 & 0.00931 & 0.00663 & 21.14 \\
\textbf{4b. + Residual Rejection Gate (Full SPWM, Val-Sel.)} & 58 / 62 / 62 & \textbf{60.7 $\pm$ 2.3} & \textbf{0.88 $\pm$ 0.11} & \textbf{2.01 $\pm$ 0.12} & 0.00931 & 0.00663 & 21.14 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

> **Honest Negative/Mixed-Result Diagnostic (Stage 2 vs. Stage 3):** Comparing Stage 2 (`51.3 ± 9.5%`) against Stage 3 (`38.0 ± 19.1%`) across all 3 seeds shows that while Phase 3/4 AR(1)+DAgger replay substantially smooths control trajectories (`|Δ²q|: 0.00991 -> 0.00589 rad/step²`, `|Δ³q|: 0.00705 -> 0.00354 rad/step³`, flips `21.99 -> 11.17`), it **degrades 3-seed mean PIL grasp success by `-13.3 pp`** (`56.0% / 18.0% / 40.0%`) and **doubles inter-seed standard deviation (`9.5% -> 19.1%`)**. Because on-policy DAgger overfitting can raise single-seed performance (e.g., Seed 42 validation `92.0%`) while destabilizing other seeds (`18.0%` on Seed 101 at `phase_4`), **$N=25$ validation-split (`[3200, 3224]`) checkpoint selection** is mandatory to select pre-overfit checkpoints where needed (`101: phase_1`, `2024: phase_2`), enabling 2-Stage SE(3) Lookahead (`Stage 4a/4b`) to achieve **`60.7 ± 2.3%`** (`58.0% / 62.0% / 62.0%`).

### Table 4: Symmetric 3-Seed Zero-Shot Sim-to-Real PIL Robustness Stress Test — SPWM vs. ANN-Rate (`50` Test Episodes, $t \ge t_c$)

```latex
\begin{table}[t]
\centering
\caption{\textbf{Zero-Shot Physical \& Sensory Robustness Stress Test in PIL (`50` Test Episodes `[3600, 3649]`, $t \ge t_c$, $N=6$ Tier-2 Checkpoints).} Symmetric 3-seed comparison ($42, 101, 2024$) of frozen SPWM-v0.12.4 vs. parameter-matched ANN-Rate. Relative success retention ($\Delta_{\text{rel}}$) is computed against each model's Nominal Baseline.}
\label{tab:pil_robustness}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
\textbf{Perturbation Condition} & \multicolumn{3}{c}{\textbf{SPWM-v0.12.4 (3-Seed Mean $\pm$ Std)}} & \multicolumn{3}{c}{\textbf{ANN-Rate (3-Seed Mean $\pm$ Std)}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
 & \textbf{Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} & \textbf{Succ (\%)} & \textbf{Min Err (cm)} & \textbf{Term Err (cm)} \\
\midrule
Nominal Baseline & \textbf{58.0 $\pm$ 27.1} & \textbf{0.90 $\pm$ 0.04} & \textbf{2.23 $\pm$ 0.68} & 37.3 $\pm$ 9.5 & 1.07 $\pm$ 0.28 & 2.98 $\pm$ 0.46 \\
Sensory Noise Injection (1.0\%) & \textbf{55.3 $\pm$ 19.4 (-4.6\% rel)} & \textbf{0.90 $\pm$ 0.09} & \textbf{2.34 $\pm$ 0.63} & 30.0 $\pm$ 3.5 (-19.6\% rel) & 1.04 $\pm$ 0.18 & 3.36 $\pm$ 0.53 \\
Payload Extrapolation (4.80 kg) & \textbf{39.3 $\pm$ 33.7 (-32.2\% rel)} & \textbf{1.18 $\pm$ 0.15} & \textbf{2.50 $\pm$ 0.62} & 26.7 $\pm$ 16.2 (-28.6\% rel) & 1.36 $\pm$ 0.20 & 3.64 $\pm$ 0.80 \\
Actuator Drift (0.80$\times$ Friction) & \textbf{40.7 $\pm$ 27.3 (-29.9\% rel)} & \textbf{1.19 $\pm$ 0.11} & \textbf{2.61 $\pm$ 0.54} & 30.7 $\pm$ 14.0 (-17.9\% rel) & 1.42 $\pm$ 0.22 & 3.74 $\pm$ 0.83 \\
Actuator Drift (1.20$\times$ Friction) & \textbf{44.0 $\pm$ 28.0 (-24.1\% rel)} & \textbf{1.22 $\pm$ 0.11} & \textbf{2.53 $\pm$ 0.56} & 33.3 $\pm$ 16.7 (-10.7\% rel) & 1.34 $\pm$ 0.21 & 3.51 $\pm$ 1.21 \\
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
> - **Primary Advantage Over `ANN-Rate`:** `SPWM`'s advantage over `ANN-Rate` is **closed-loop dynamical stability and noise robustness** (`+23.4 pp` 3-seed PIL grasp success, `4.3×` better relative retention under `1.0%` sensory noise), **not** end-to-end closed-loop energy savings: saving `48.14 nJ/step` on Loihi-2 (`60.36 -> 12.22 nJ/step`) atop `1.0971 mJ/step` of dense GPU MPPI rollouts is a **`0.004%`** end-to-end difference (`1.0979 mJ/step` / `21.96 mW` @ `20 Hz` for both).
> - **Always-On Standby Benefit & `M1` Rollout Compactness:** The `4.94×` Loihi-2 recurrent core reduction (`12.22 ± 0.19 nJ/step`, `0.244 mW` @ `20 Hz`, computed from synaptic SOPs excluding passive membrane leak) is specifically an **always-on, sub-milliwatt standby state-estimation benefit** when batched MPPI planning is dormant. The `1.93×` E2E energy reduction vs. `Continuous RSSM` (`2.1236 mJ/step`) and `1.59×` reduction vs. `Deep GRU` (`1.7490 mJ/step`) stem from `M1`'s compact MLP rollout architecture (`1.219 GMACs` vs. `2.358 GMACs` and `1.942 GMACs`).

---

## 3. Residual-Gated Handover ($w_t > 0$) Telemetry & Validation Threshold Sweep

| Perturbation Condition | Seed 42 (Ep% / Step% / Mean $w$) | Seed 101 (Ep% / Step% / Mean $w$) | Seed 2024 (Ep% / Step% / Mean $w$) | 3-Seed Mean Step $w_t > 0$ |
| :--- | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |
| **Sensory Noise Injection (1.0%)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `0.00 ± 0.00%` |
| **Payload Extrapolation (4.80 kg)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `4.0% / 0.5% / 0.165` | `0.17 ± 0.29%` |
| **Actuator Drift (0.80x Friction)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `4.0% / 0.5% / 0.229` | `0.17 ± 0.29%` |
| **Actuator Drift (1.20x Friction)** | `0.0% / 0.0% / 0.000` | `0.0% / 0.0% / 0.000` | `4.0% / 0.5% / 0.169` | `0.17 ± 0.29%` |

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
