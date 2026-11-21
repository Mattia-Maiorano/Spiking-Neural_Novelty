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
3. **Baseline Equalization (AR(1)+DAgger) & Identical FK Residual Rejection Gate:** Fixed the truncated `18`-epoch `Deep GRU` runs on Seeds `101, 2024`, trained both `Continuous RSSM` and `Deep GRU` across all 3 seeds (`42, 101, 2024`) with multi-step SE(3) rollout + AR(1) MPPI-matched action noise + `ClosedLoopBuffer` DAgger replay, and evaluated all three PIL regimes (**Open-Loop $w_t=0$**, **Un-Gated Online $w_t=1.0$**, and **Residual-Gated Online** using the identical FK rejection gate as SPWM). Under residual gating, `Continuous RSSM` remains at `15.3 ± 5.0%` (`0.0%` un-gated, confirming its ceiling is architectural), while equalized `Deep GRU` recovers from `0.0%` (un-gated) to `76.7 ± 5.0%` (`77.3 ± 4.2%` open-loop)—achieving higher nominal PIL success than `SPWM` (`60.7 ± 2.3%`) while exceeding the `50 ms` (`20 Hz`) real-time budget (`57.7 ms/step` vs `26.6 ms/step`), trailing `SPWM` by `9.0×` on $H=100$ velocity MSE (`0.08716` vs `0.00971`), and using `22.1×` higher standby core energy / `1.59×` higher E2E step energy.
4. **Full 3-Seed 5-Stage Disentangled Ablation (`45.3 ± 16.0% -> 51.3 ± 9.5% -> 38.0 ± 19.1% -> 60.7 ± 2.3% -> 60.7 ± 2.3%`):** Evaluated across all 3 seeds (`42, 101, 2024`) on `[3600, 3649]`. Transparently reveals that Phase 3/4 AR(1)+DAgger replay (`Stage 3`) reduces trajectory jitter (`|Δ²q|: 0.00991 -> 0.00589`) while degrading 3-seed mean PIL success (`-13.3 pp`) and doubling seed std (`9.5% -> 19.1%`), making `N=25` validation checkpoint selection (`[3200, 3224]`) essential before 2-Stage SE(3) Lookahead (`Stage 4a/4b`) yields `60.7 ± 2.3%` (`11.7×` variance cut vs. `N=6` Tier-2 `58.0 ± 27.1%`).
5. **Standby Core vs. E2E Energy Recalibration:** Frames the `4.94×` Loihi-2 core reduction (`12.22 ± 0.19 nJ/step`, `0.244 mW` @ `20 Hz`, excluding passive membrane leak) strictly as an **always-on standby state-estimation benefit** when dense MPPI planning is dormant; E2E step energy (`1.0979 mJ/step`) matches `ANN-Rate` (`0.004%` diff), so `SPWM`'s primary advantage over `ANN-Rate` is dynamical stability (`+23.4 pp` PIL success) and `4.3×` better `1.0%` noise retention.

---

## 2. Corrections & Reconciliations Applied

| Historical Issue Resolved | Ground Truth Verified in Code / Artifacts | Where Resolved Here |
| :--- | :--- | :--- |
| Un-gated (`w_t=1.0`) online baseline collapse & truncated `Deep GRU` epochs (`59/18/18`) | Equalized `Continuous RSSM` (`60+16/16/17`) & `Deep GRU` (`59+40=99`, `80`, `80`) with AR(1)+DAgger; evaluated Open-Loop, Un-Gated, and Residual-Gated PIL regimes across all 3 seeds | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L22-L70) |
| Single-seed `"Canonical Seed 42"` (`86.0%`) headline & Seed-42-only ablation | Removed all `"Canonical Seed 42"` headline claims; lead strictly with $N=25$ val-selected (`60.7 ± 2.3%`) and 3-seed 5-stage ablation (`45.3 -> 51.3 -> 38.0 -> 60.7 -> 60.7%`) | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L72-L88) |
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
