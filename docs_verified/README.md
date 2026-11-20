# SPWM Verified Documentation (`docs_verified/`)

> **Consolidated Single-Source-of-Truth Documentation** replacing the 10 overlapping files in [docs/](file:///Users/Mattia/Desktop/Studies/Temp/docs) without breaking scripts that write to `docs/` ([run_ssot_benchmark_and_plot.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/run_ssot_benchmark_and_plot.py#L555), [aggregate_corl_final_verification.py](file:///Users/Mattia/Desktop/Studies/Temp/scripts/aggregate_corl_final_verification.py#L624)). Every claim, equation, hyperparameter, and benchmark value below is directly linked to source code, JSON/checkpoint artifacts, background execution logs (`task-*.log`), and reproducible CLI commands.

---

## 1. Executive Audit Verdict: Are the Documentation Values Correct and Backed Up?

**Yes for all core empirical results.** Every benchmark metric in the latest results pipeline ([ssot_eval_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/ssot_eval_metrics.json), [multiseed_metrics.json](file:///Users/Mattia/Desktop/Studies/Temp/results/multiseed/multiseed_metrics.json), [comparative_metrics_v0_12_4.json](file:///Users/Mattia/Desktop/Studies/Temp/results/spwm_v0_12_4_train_m1/comparative_metrics_v0_12_4.json), and `results/`) matches actual command runs recorded across three sequential Antigravity sessions:
* **Agent 3 (Iteration 8, finished `2026-11-20 13:23 CET`):** [transcript.jsonl](file:///Users/Mattia/.gemini/antigravity/brain/b270518d-34f7-4b20-a61b-c71cf07c3fb8/.system_generated/logs/transcript.jsonl) (`task-346`, `352`, `361`, `402`, `524`, `557`, `586`).
* **Agent 2 (Iteration 9, `13:24–16:01 CET`):** [transcript.jsonl](file:///Users/Mattia/.gemini/antigravity/brain/a62a3ce4-cc2d-4e89-a137-0cb6fa9b42a6/.system_generated/logs/transcript.jsonl) (`task-29`, `58`, `60`, `84`, `107`, `127`, `141`, `152`).
* **Agent 1 (Iteration 10, `15:11–16:01 CET`):** [transcript.jsonl](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/logs/transcript.jsonl) (`task-34`, `108–111`, `118–119`, `126`, `138`, `152–153`, `161`, `206`, `213`, `249`).

### Progression of Testing Precision
The reported SPWM success numbers moved from `96.0%` $\to$ `88.0%` $\to$ `86.0%` $\to$ `58.0 ± 27.1%` $\to$ `60.7 ± 2.3%` directly as evaluation rigor tightened:
1. **Non-PIL Latent-Grounded (`96.0%`) & Pre-Fix PIL (`88.0%`):** Early runs either blended ground-truth latents (`z_gt_next`) during $\phi_t = 1$ or delayed payload attachment to `t >= t_grasp` (historical Nov 17 snapshot in [docs/CORL_GROUNDED_AUDIT.md](file:///Users/Mattia/Desktop/Studies/Temp/docs/CORL_GROUNDED_AUDIT.md)).
2. **Causal Physics-in-the-Loop (`86.0%` Canonical Seed 42):** Enforced causal `payload_attached=(t >= t_c)` in [experiments/benchmark_mppi_control.py](file:///Users/Mattia/Desktop/Studies/Temp/experiments/benchmark_mppi_control.py#L390) and [spwm/evaluation/mppi_scorer.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/mppi_scorer.py#L369), where $t_c$ indexes the $\phi_t=1$ sensory-window onset (`p1_indices[0] = t_contact - 7`) and PIL evaluates the 7-step loaded transition through $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$, with zero `z_gt_next` leakage, causal `OnlineEventCameraStream`, and strict grasp success criteria (`< 3.0 cm` 3-step settling, `< 0.05 rad` $\text{SO}(3)$ error, $\tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)$) ([tests/test_leakage.py](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_leakage.py), [tests/test_pil_integrity.py](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_pil_integrity.py)).
3. **3-Seed PIL + Online vs. Open-Loop Baseline Disentanglement:** Evaluated Seeds `42, 101, 2024` symmetrically (`58.0 ± 27.1%` for SPWM `N=6` Tier-2, `37.3 ± 9.5%` for `ANN-Rate`), and separated **Open-Loop PIL** vs. **Online Observation Update** for `Continuous RSSM` (`15.3 ± 4.6%` vs. `0.0%`) and `Deep GRU` (`1.3 ± 2.3%` vs. `0.0%`).
4. **5-Stage Disentangled Ablation (`62.0% -> 62.0% -> 72.0% -> 86.0% -> 86.0%`):** Isolated Stage 4a (`+ 2-Stage SE(3) Lookahead` with handover strictly disabled, $w_t \equiv 0$) from Stage 4b (Full), proving the `+14.0 pp` gain comes from lookahead while residual handover keeps `0.0%` false triggers on nominal episodes.
5. **$N=25$ Validation Selection (`60.7 ± 2.3%`) & Forensic Diagnostics:** Audited all 15 checkpoints across Seeds `42, 101, 2024` on held-out validation split `[3200, 3224]` (cutting inter-seed test std $11.7\times$ from `27.06%` to `2.31%`), alongside per-episode failure decomposition ([forensic_rssm_gru_failure.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/forensic_rssm_gru_failure.json)) and weight/spike audits ([spwm_seed_variance_audit.json](file:///Users/Mattia/Desktop/Studies/Temp/results/diagnostics/spwm_seed_variance_audit.json)).

---

## 2. Corrections & Reconciliations Applied Over Legacy `docs/`

Because the three prior sessions ran without intermediate git commits after `HEAD` (`c413bcc`), and because legacy `docs/` accumulated stale equations and metadata, this directory resolves every discovered discrepancy:

| Issue Found in Legacy `docs/` | Ground Truth Verified in Code / Artifacts | Where Resolved Here |
| :--- | :--- | :--- |
| Agent 1 overwrote Agent 2's Section 2 in `docs/CORL_FINAL_BENCHMARK_SUMMARY.md`, dropping forensic diagnostics | Restored `forensic_rssm_gru_failure.json`, `spwm_seed_variance_audit.json`, `pil_ep240_seed{42,101}`, `pil_final_seed2024`, and `robustness_benchmark_pil_ann_rate_seed*` | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L128-L150) |
| Table 1 caption claimed `"160+240 training budget for SPWM, ANN-Rate, and Continuous RSSM"` | `Continuous RSSM` trained **60 epochs** (all 3 seeds); `Deep GRU` **59/18/18** ep; `ConvLSTM` **54** ep; `SPWM Seed 2024` **160+260** ep (`best_model.pt` locked at Ep 130) | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L15-L34) |
| Seed 42 latency mismatch (`27.51 ms` in `ssot_eval_metrics.json` vs `23.07 ms` in `multiseed_metrics.json`) | `results/mppi_benchmark/mppi_benchmark_metrics.json` (`36.65 ms`) ran under `OMP_NUM_THREADS=2` CPU throttling (Agent 1 `task-111`), whereas `physics_in_loop/spwm/` (`23.33 ms`, Agent 3) ran unthrottled | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L56-L65) |
| `model.pt` vs `best_model.pt` conflation across sections | Explicitly recorded SHA-256 hashes (`d358cccbd678` vs `f5cabfca3635`, identical Epoch 200 EMA `model_state` tensors with `0` weight diffs) and which checkpoint each benchmark evaluated | [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md#L5-L19) |
| `GLOBAL_VERSION` claimed as `0.12.4` / `"spwm_v0_12_4"` | [spwm/contracts/version.py](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/version.py#L29-L32) defines `0.12.3` (`spwm_v0_12_3`), so robustness JSONs log `"SPWM-v0.12.3"` while YAML configs override `project.name: spwm_v0_12_4` | [01_SYSTEM_AND_ARCHITECTURE.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md#L26-L28) |
| Legacy `DOSSIER.md` / `Architecture.md` / `WORLD.md` formula & spec drift | Fixed `32x32` sensor resolution, Phase 0/1 `lr = 1e-4` (`probe_lr = 2e-4`), Phase 1 `predictor + critic` scope, `v_p = p_next[:, :q_dim]`, `.mean()` M1 loss normalization, `ddq / dt`, $\theta_{\max} = 3.14$ scaling in `sim_6dof.py`, Unity `6000.5.7f1`, and segmented wrist cylinders | [01_SYSTEM_AND_ARCHITECTURE.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md), [02_TRAINING_CURRICULUM_AND_MPPI.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md) |

---

## 3. Directory Map (`docs_verified/`)

| Document | Scope | Replaces Legacy `docs/` Files |
| :--- | :--- | :--- |
| [README.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/README.md) | Audit verdict, testing precision progression, and discrepancy reconciliations | `CORL_GROUNDED_AUDIT.md`, `History.md` (meta) |
| [01_SYSTEM_AND_ARCHITECTURE.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md) | Repo skeleton, 4-layer contracts, Unity `6000.5.7f1` / Python PIL physics, dataset splits, neural equations, exact parameter census, and design decisions / negative ablations | `REPO_SKELETON.md`, `WORKSPACE_AND_CONTRACTS.md`, `WORLD.md`, `Architecture.md`, `DOSSIER.md` (ablations) |
| [02_TRAINING_CURRICULUM_AND_MPPI.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md) | Phase 0 & Phase 1 training strategies, exact losses, 4-phase latch curriculum, active perception / residual handover, MPPI cost, Tier-2 validation, training CLIs, and baseline / online update specs | `TRAINING_AND_LOSSES.md`, `Architecture.md` (MPPI) |
| [03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md) | Checkpoint provenance, Tables 1–5, Figs 1–4 + `heft_shock_timeseries`, SOP/MAC derivation, Phase 0 metrics, validation sweeps, forensic diagnostics, compute/wall-clock stack, and replication runbook | `CORL_FINAL_BENCHMARK_SUMMARY.md`, `FIGURE_METRICS.md`, `DOSSIER.md` (benchmarks), `History.md` (Iter 8–10) |

---

## 4. Quick Verification Gate

Run the repository AST anti-fallback check and the 101-test integrity suite (verified passing in [task-249.log](file:///Users/Mattia/.gemini/antigravity/brain/1ca9af8a-f1e4-4a6f-ba2d-912890fe3ec8/.system_generated/tasks/task-249.log)):

```bash
.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/
.venv/bin/pytest tests/ -q
```
