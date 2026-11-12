# Agent-Friendly Refactoring Master Plan (`AGENT_REFACTORING_PLAN.md`)

This document is the **single coordination blueprint** for refactoring the SPWM repository into an **Agent-Friendly** codebase. It is designed so that **independent, disconnected agents** can claim and execute individual checklist items without reading the entire repository or introducing regressions.

---

## 1. Global Context (Mandatory Reading for Every Assigned Agent)

Before touching any file, every assigned agent must internalize the following non-negotiable architectural invariants:

### 1.1 The 4-Layer Strict Dependency Hierarchy
Imports inside `spwm/` may only flow **downward** (enforced by `tests/test_architecture_contracts.py::test_strict_4_layer_import_hierarchy`):
- **Layer 0 (`spwm/contracts/`)**: Pure immutable (`@dataclass(frozen=True)`) configs (`config.py`), state containers (`states.py`), runtime-checkable protocols (`protocols.py`), tensor/phase-space assertions (`tensors.py`), and `GLOBAL_VERSION` (`version.py`). **Zero `spwm.*` imports outside `spwm.contracts`.**
- **Layer 1 (`spwm/models/`, `spwm/data/`)**: Neural architectures, kinematics, unified rollout (`spwm/models/`), and dataset loaders / event camera / trajectory simulation (`spwm/data/`). May import `spwm.contracts` and intra-layer `spwm.models`/`spwm.data`. **Never `spwm.training` or `spwm.evaluation`.**
- **Layer 2 (`spwm/training/`)**: Losses, phase strategies (`representation`, `m1_nominal`), curriculum/noise schedules, replay buffer, checkpoints, and `BaseTrainer` (`engine.py`). May import `spwm.contracts`, `spwm.models`, `spwm.data`, `spwm.utils`. **Never `spwm.evaluation`.**
- **Layer 3 (`spwm/evaluation/`)**: Rollout evaluation, MPPI closed-loop control, Tier 2 structural scoring, robustness benchmarks, and visualizers.
- **Layer 4 (`experiments/`, `scripts/`)**: Thin CLI runners (`experiments/`) and CI/figure/diagnostic utilities (`scripts/`).

### 1.2 Strict Single Source of Truth (SSOT) & Target Pillars
1. **Structural Determinism**: One canonical location for every concept. No empty ghost folders, no duplicate documentation roots, no compatibility facades that obscure where logic lives.
2. **Locality of Behavior**: Modules should remain focused and concise (ideally `< 400` LOC) so any task requires loading $\le 3$ files into context.
3. **Configuration SSOT**: `ExperimentConfig` (`spwm/contracts/config.py`) is the sole owner of hyperparameters and defaults. Downstream classes and dataloader factories must accept typed configs (`.from_config(cfg)`) rather than duplicating defaults across 20-argument signatures.
4. **Zero-Fallback Policy (Enforced by AST CI)**:
   - **NO** 3-argument `getattr(obj, attr, default)`.
   - **NO** ternary `hasattr(...)` fallbacks.
   - **NO** `load_state_dict(..., strict=False)`.
   - **NO** silent `except TypeError:`, `except AttributeError:`, or `except ...: pass`.
   - **NO** silent `dict.get(key, fallback)` masking missing schemas.
   - Verify after every edit with: `.venv/bin/python scripts/scan_fallbacks.py spwm/` and `.venv/bin/pytest tests/test_anti_fallback.py tests/test_architecture_contracts.py`.

### 1.3 Repository Rules for Task Completion
- Every task must leave the repository in a **100% passing state** before marking the item `[x]`.
- Do **not** expand scope beyond the files listed in your assigned `Notes for the Assigned Agent` block.
- When a step modifies contracts, models, training logic, or simulation scripts, synchronize `./docs/` and record the completed iteration in `docs/History.md` using the 5-item structure defined in `AGENTS.md` / `GEMINI.md`.

---

## 2. Master Task Checklist

### Phase A: Workspace Hygiene & Import Determinism
- [ ] **Step 1: Workspace Hygiene, Ghost Directory Removal & Documentation SSOT Consolidation**
- [x] **Step 2: Prune Redundant Re-Export Facades & Enforce Direct Layer Imports**

### Phase B: Mathematical & Config SSOT Reconciliation
- [ ] **Step 3: Reconcile Loss SSOT (`spwm/training/losses.py` vs. Phase Strategies)**
- [x] **Step 4: Unify Kinematics, Rotation Metrics & MPPI Evaluation SSOT**

### Phase C: Monolithic Module Decomposition (Locality of Behavior)
- [x] **Step 5.1: Decompose `spwm/training/engine.py` (Checkpoints, Replay Buffer & Trainer)**
- [x] **Step 5.2: Decompose `spwm/evaluation/mppi.py` (Controller vs. Scorer & Relocate `NeuralCostMPPIController`)**
- [x] **Step 5.3: Decompose `spwm/data/datasets.py` (6-DOF Sim, Unity I/O & PyTorch Datasets)**

### Phase D: Typed Boundaries, Config Injection & CI Hardening
- [x] **Step 6: Standardize Config-First Object Injection & Typed Layer Boundaries**
- [x] **Step 7: Reconcile `experiments/` vs. `scripts/`, Harden AST CI Guards & Finalize Docs**

---

## 3. Detailed Execution Steps & Agent Handover Notes

---

### Step 1: Workspace Hygiene, Ghost Directory Removal & Documentation SSOT Consolidation

- [ ] **Step 1: Workspace Hygiene, Ghost Directory Removal & Documentation SSOT Consolidation**

> **Notes for the Assigned Agent:**
> - **Objective:** Eliminate split-brain documentation (`docs/` vs. `documentation/`), remove empty legacy directories (`spwm/core/`, `spwm/learning/`, `training/`), and synchronize version references with `GLOBAL_VERSION`.
> - **Allowed Files to Touch:**
>   - Read/Merge/Delete: `documentation/MODELS.md`, `documentation/TRAINING.md`, `documentation/WORLD.md` (delete `documentation/` once merged).
>   - Update: `docs/ARCHITECTURE.md`, `docs/TRAINING_AND_LOSSES.md`, `docs/WORLD.md`, `docs/WORKSPACE_AND_CONTRACTS.md`, `docs/History.md`, `AGENTS.md`, `GEMINI.md`, `REPO_SKELETON.md`.
>   - Delete empty ghost directories (if only `__pycache__` / `.DS_Store` remain): `spwm/core/`, `spwm/learning/`, `training/`.
> - **Traps to Avoid:**
>   - Do **not** delete any file inside `docs/`; merge unique content from `documentation/*.md` into the corresponding `docs/*.md` files first before removing `documentation/`.
>   - In `AGENTS.md` and `GEMINI.md`, update the synchronization paths (e.g., change `documentation/MODELS.md` to `docs/ARCHITECTURE.md` and `docs/TRAINING.md` to `docs/TRAINING_AND_LOSSES.md`) while keeping all other rules intact.
>   - Verify `REPO_SKELETON.md` and `docs/WORKSPACE_AND_CONTRACTS.md` match `spwm/contracts/version.py` (`GLOBAL_VERSION`).
>   - Do **not** touch any `.py` source files in `spwm/`, `experiments/`, `scripts/`, or `tests/` in this step.
> - **Definition of Done (Must exit with code 0):**
>   - `test ! -d documentation && test ! -d spwm/core && test ! -d spwm/learning && test ! -d training`
>   - `.venv/bin/pytest`

---

### Step 2: Prune Redundant Re-Export Facades & Enforce Direct Layer Imports

- [x] **Step 2: Prune Redundant Re-Export Facades & Enforce Direct Layer Imports**

> **Notes for the Assigned Agent:**
> - **Objective:** Remove thin re-export shims (`spwm/config/`, `spwm/models/spwm.py`, `spwm/training/rollout.py`) so every symbol has a single canonical import path.
> - **Allowed Files to Touch:**
>   - Update imports in: `tests/test_config_schema.py` (switch from `spwm.config.schema` to `spwm.contracts.config`), `tests/test_closed_loop_buffer.py` (switch from `spwm.models.spwm` to `spwm.models.world_model`), `spwm/training/__init__.py` (import rollout symbols directly from `spwm.models.rollout`).
>   - Update `tests/test_dead_code.py` (`OBSOLETE_FILE_PATHS`) to assert these deleted shims never return.
>   - Delete: `spwm/config/__init__.py`, `spwm/config/schema.py` (and `spwm/config/`), `spwm/models/spwm.py`, `spwm/training/rollout.py`.
>   - Update `REPO_SKELETON.md` and `docs/WORKSPACE_AND_CONTRACTS.md` if they still list the removed facades.
> - **Traps to Avoid:**
>   - **Do NOT delete `spwm/version.py`** yet if `scripts/run_pipeline.sh`, `scripts/run_baselines.sh`, and `tests/test_version_solver.py` invoke `python -m spwm.version`. Leave `spwm/version.py` intact unless explicitly scoped.
>   - Check `spwm/`, `experiments/`, `scripts/`, and `tests/` with `grep` before deleting any facade to make sure no hidden callers remain.
>   - Preserve the 4-layer import hierarchy (`spwm/training/__init__.py` importing from `spwm.models.rollout` is valid Layer 2 $\rightarrow$ Layer 1).
> - **Definition of Done (Must exit with code 0):**
>   - `grep -rn "spwm\.config\|spwm\.models\.spwm\|spwm\.training\.rollout" spwm/ experiments/ scripts/ tests/` returns 0 matches.
>   - `.venv/bin/pytest tests/test_dead_code.py tests/test_architecture_contracts.py tests/test_config_schema.py tests/test_closed_loop_buffer.py`
>   - `.venv/bin/pytest`

---

### Step 3: Reconcile Loss SSOT (`spwm/training/losses.py` vs. Phase Strategies)

- [ ] **Step 3: Reconcile Loss SSOT (`spwm/training/losses.py` vs. Phase Strategies)**

> **Notes for the Assigned Agent:**
> - **Objective:** Make `spwm/training/losses.py` the true Single Source of Truth for Phase 0 (`train_representation`) and Phase 1 (`train_m1`) losses. Currently, `BaseTrainer` instantiates `SPWMLoss` (`PredictorLoss`), but `RepresentationPhaseStrategy._compute_representation_objectives` and `M1PhaseStrategy._compute_m1_objectives` bypass `self.loss_fn` and compute losses inline.
> - **Allowed Files to Touch:**
>   - `spwm/training/losses.py`
>   - `spwm/training/phases/representation.py`
>   - `spwm/training/phases/m1_nominal.py`
>   - `spwm/training/__init__.py`
>   - `spwm/contracts/config.py` (only if pruning genuinely dead fields in `LossConfig`, after checking `tests/test_config_usage.py` and `tests/test_config_schema.py`)
>   - `tests/test_training_phases.py`, `tests/test_relay.py` (and a new or temporary frozen baseline test file under `tests/`)
> - **Traps to Avoid:**
>   - **MANDATORY FROZEN BASELINE PREREQUISITE:** Before refactoring any loss math, you **MUST** write and run a frozen fixture test using fixed PyTorch seeds (`torch.manual_seed(42)`), dummy `SPWM` models, dummy `events`/`flat_kin`/`actions`/`phase_flags`/`closed_loop_batch` tensors, and save/hardcode the exact scalar loss outputs (`total_loss`, sub-losses) and parameter gradient norms produced by the *unmodified* `RepresentationPhaseStrategy` and `M1PhaseStrategy`. After extracting the loss functions into `spwm/training/losses.py`, assert **bit-for-bit identical** (or `atol=0.0, rtol=0.0` / exact float equality on CPU) forward loss values and backward gradients!
>   - Do **not** alter `LossConfig` field names expected by `tests/test_config_schema.py` (`lambda_encoder`, `lambda_decoder`, `lambda_position`, `lambda_velocity`, etc.) without checking whether existing YAML configs or tests still pass them to `LossConfig(...)`.
>   - Keep `RepresentationPhaseStrategy` and `M1PhaseStrategy` under 150 lines each (per `docs/WORKSPACE_AND_CONTRACTS.md` contract).
>   - Strictly obey Zero-Fallback rules (no `getattr` defaults, no `dict.get(..., fallback)` on config).
> - **Definition of Done (Must exit with code 0):**
>   - `.venv/bin/pytest tests/test_training_phases.py tests/test_relay.py tests/test_config_usage.py tests/test_config_schema.py tests/test_anti_fallback.py`
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/`
>   - `.venv/bin/pytest`

---

### Step 4: Unify Kinematics, Rotation Metrics & MPPI Evaluation SSOT

- [x] **Step 4: Unify Kinematics, Rotation Metrics & MPPI Evaluation SSOT**

> **Notes for the Assigned Agent:**
> - **Objective:** Eliminate divergent `compute_rot_error` implementations, duplicate `classify_elevation` helpers, competing Tier 2 `mppi_score` formulas, and hardcoded MPPI parameters bypassing `MPPIConfig`.
> - **Allowed Files to Touch:**
>   - `spwm/contracts/config.py` (`MPPIConfig` only, if moving hardcoded `gamma = 0.75` or anisotropic sigma multipliers into config)
>   - `spwm/models/kinematics.py` (canonical `compute_rot_error` and/or `classify_elevation`, or keep in `spwm/evaluation/mppi.py` and import everywhere)
>   - `spwm/evaluation/mppi.py`
>   - `spwm/training/engine.py` (lines ~847–909 where `tier2_metrics["mppi_score"]` is currently overwritten)
>   - `experiments/benchmark_mppi_control.py`
>   - `spwm/utils/MPPI_loss_finder.py`
>   - `tests/test_mppi.py`, `tests/test_two_tiered_validation.py`, `tests/test_kinematics.py`
> - **Traps to Avoid:**
>   - **MANDATORY FROZEN BASELINE PREREQUISITE:** Before modifying `LatentMPPIController.compute_trajectory_cost`, `plan_step`, `compute_rot_error`, or `MPPIStructuralScorer`, you **MUST** write and run a frozen fixture test with fixed seeds (`seed=42000`) capturing exact numeric outputs from the *unmodified* `LatentMPPIController.plan_step`, `compute_trajectory_cost`, `compute_rot_error`, and `MPPIStructuralScorer.evaluate_trajectory`. Verify the unified code produces **bit-for-bit identical** outputs when using the same config values!
>   - **Semantic Divergence Trap:** Note that `spwm/evaluation/mppi.py::compute_rot_error` computes geodesic $\arccos((\text{Tr}(R^T R_{\text{tgt}}) - 1)/2)$ while `experiments/benchmark_mppi_control.py::compute_rot_error` computes cosine dot-product $(1 - \mathbf{z}\cdot\mathbf{z}_{\text{tgt}}) + (1 - \mathbf{y}\cdot\mathbf{y}_{\text{tgt}})$. Document clearly in the canonical function docstring which formulation is canonical, and verify `tests/test_mppi.py` and `tests/test_two_tiered_validation.py` expectations before changing formulas.
>   - **Tier 2 Score Overwrite Trap:** Notice `spwm/training/engine.py` currently overwrites `tier2_metrics["mppi_score"]` with `0.5 * grasp_error_cm + 0.5 * settling_pos_error_cm + 0.25 * mean_cart_error_cm + 0.1 * final_velocity + 2.0 * geodesic_rot_error`. Move that exact canonical scoring into `MPPIStructuralScorer` (or make `MPPIStructuralScorer` compute both `structural_score` and canonical `mppi_score`) so `engine.py` consumes `tier2_metrics["mppi_score"]` without recomputing it, and verify `tests/test_two_tiered_validation.py` still passes identically.
>   - In `experiments/benchmark_mppi_control.py`, wire `--noise-beta`, `--w-act`, `--w-rot`, etc. cleanly through `MPPIConfig` (using `cfg.mppi` as base) instead of ignoring CLI args and hardcoding floats.
>   - Check `tests/test_config_usage.py` if adding any field to `MPPIConfig`—every field in `MPPIConfig` must be referenced in non-test code.
> - **Definition of Done (Must exit with code 0):**
>   - `grep -rn "def compute_rot_error\|def classify_elevation" spwm/ experiments/ scripts/` returns at most 1 definition per symbol.
>   - `.venv/bin/pytest tests/test_mppi.py tests/test_two_tiered_validation.py tests/test_kinematics.py tests/test_config_usage.py tests/test_anti_fallback.py`
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/`
>   - `.venv/bin/pytest`

---

### Step 5.1: Decompose `spwm/training/engine.py` (Checkpoints, Replay Buffer & Trainer)

- [x] **Step 5.1: Decompose `spwm/training/engine.py` (Checkpoints, Replay Buffer & Trainer)**

> **Notes for the Assigned Agent:**
> - **Objective:** Split the 1,278-line `spwm/training/engine.py` into three focused, co-located modules without changing runtime behavior or breaking checkpoint compatibility:
>   1. `spwm/training/checkpoints.py`: `load_representation_checkpoint`, `resolve_version_checkpoint`, `resolve_closed_loop_buffer`, `enforce_phase_prerequisites`, `_extract_submodel_state`.
>   2. `spwm/training/replay_buffer.py`: `ClosedLoopBuffer`, `EMATracker`.
>   3. `spwm/training/engine.py`: `BaseTrainer`, `Trainer` (shrunk to orchestration only).
> - **Allowed Files to Touch:**
>   - Create: `spwm/training/checkpoints.py`, `spwm/training/replay_buffer.py`
>   - Modify: `spwm/training/engine.py`, `spwm/training/__init__.py`
>   - Modify tests only if needed to test round-trip serialization: `tests/test_closed_loop_buffer.py`, `tests/test_training_phases.py`
> - **Traps to Avoid:**
>   - **CRITICAL CHECKPOINT / SERIALIZATION SAFETY CONSTRAINT:**
>     - Moving `ClosedLoopBuffer` or `EMATracker` to `spwm/training/replay_buffer.py` can break existing `.pt` files (`checkpoint.pt`, `closed_loop_buffer.pt`, `model.pt`) if any checkpoint ever pickled a class instance instead of pure `state_dict()` dicts/tensors!
>     - Verify all `save()` / `save_checkpoint()` / `state_dict()` methods in `ClosedLoopBuffer`, `EMATracker`, and `BaseTrainer` serialize **strictly** plain Python primitives (`dict`, `list`, `int`, `float`, `str`, `bool`) and `torch.Tensor` objects—**never** class instances.
>     - Keep backward-compatible re-export imports of `ClosedLoopBuffer`, `EMATracker`, `load_representation_checkpoint`, `resolve_version_checkpoint`, `resolve_closed_loop_buffer`, and `enforce_phase_prerequisites` inside `spwm/training/engine.py` (and `spwm/training/__init__.py`) so all existing callers (`from spwm.training.engine import ClosedLoopBuffer`, `experiments/train.py`, `tests/*`, and any pickled module path references to `spwm.training.engine`) continue to resolve cleanly.
>   - **Do NOT touch** `spwm/evaluation/` or `spwm/data/` in this sub-step.
>   - Enforce Layer 2 import rules (no `spwm.evaluation` imports in `checkpoints.py`, `replay_buffer.py`, or `engine.py`) and Zero-Fallback AST rules.
> - **Definition of Done (Must exit with code 0):**
>   - A round-trip test confirms `ClosedLoopBuffer.save()` / `load()` and `BaseTrainer.save_checkpoint()` / `load_checkpoint()` succeed with `strict=True` and contain only `dict`/`list`/primitive/`Tensor` types in `state_dict()`.
>   - `.venv/bin/pytest tests/test_closed_loop_buffer.py tests/test_training_phases.py tests/test_two_tiered_validation.py tests/test_architecture_contracts.py tests/test_anti_fallback.py`
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/`
>   - `.venv/bin/pytest`

---

### Step 5.2: Decompose `spwm/evaluation/mppi.py` (Controller vs. Scorer & Relocate `NeuralCostMPPIController`)

- [x] **Step 5.2: Decompose `spwm/evaluation/mppi.py` (Controller vs. Scorer & Relocate `NeuralCostMPPIController`)**

> **Notes for the Assigned Agent:**
> - **Objective:** Split the 964-line `spwm/evaluation/mppi.py` into focused modules and relocate `NeuralCostMPPIController` out of `spwm/utils/MPPI_loss_finder.py` into Layer 3:
>   1. `spwm/evaluation/mppi_controller.py`: `MPPIPlanResult`, `LatentMPPIController`, `NeuralCostMPPIController` (moved from `spwm/utils/MPPI_loss_finder.py`).
>   2. `spwm/evaluation/mppi_scorer.py`: `MPPIStructuralScorer`, `compute_structural_score` (and shared closed-loop evaluation logic).
>   3. `spwm/evaluation/mppi.py`: Clean thin barrel re-exporting symbols from `mppi_controller.py` and `mppi_scorer.py` so existing imports (`from spwm.evaluation.mppi import ...`) remain unbroken.
> - **Allowed Files to Touch:**
>   - Create: `spwm/evaluation/mppi_controller.py`, `spwm/evaluation/mppi_scorer.py`
>   - Modify: `spwm/evaluation/mppi.py`, `spwm/evaluation/__init__.py`, `spwm/utils/MPPI_loss_finder.py` (import `NeuralCostMPPIController` from `spwm.evaluation.mppi_controller` instead of defining it inline)
> - **Traps to Avoid:**
>   - **CRITICAL CHECKPOINT / SERIALIZATION SAFETY CONSTRAINT:**
>     - Ensure `MPPIPlanResult` or controller instances are never pickled directly into checkpoints; keep `spwm/evaluation/mppi.py` re-exporting `LatentMPPIController`, `MPPIPlanResult`, `MPPIStructuralScorer`, `compute_structural_score`, `compute_rot_error`, and `classify_elevation` so any external script or test importing from `spwm.evaluation.mppi` continues working without modification.
>   - **Notice existing bug in `spwm/utils/MPPI_loss_finder.py`:** Lines 39–44 reference `LatentPredictor`, `PhysicalDecoder`, `LatentCritic` in type annotations without importing them (masked by `from __future__ import annotations`). When moving `NeuralCostMPPIController` to `spwm/evaluation/mppi_controller.py`, explicitly import `LatentPredictor`, `PhysicalDecoder`, and `LatentCritic` from `spwm.models.predictors`.
>   - **Do NOT touch** `spwm/training/` or `spwm/data/` in this sub-step.
>   - Strictly verify Zero-Fallback AST compliance across `spwm/evaluation/`.
> - **Definition of Done (Must exit with code 0):**
>   - `.venv/bin/pytest tests/test_mppi.py tests/test_two_tiered_validation.py tests/test_robustness_stress_test.py tests/test_architecture_contracts.py tests/test_anti_fallback.py`
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/`
>   - `.venv/bin/pytest`

---

### Step 5.3: Decompose `spwm/data/datasets.py` (6-DOF Sim, Unity I/O & PyTorch Datasets)

- [x] **Step 5.3: Decompose `spwm/data/datasets.py` (6-DOF Sim, Unity I/O & PyTorch Datasets)**

> **Notes for the Assigned Agent:**
> - **Objective:** Split the 831-line `spwm/data/datasets.py` into three focused Layer 1 modules:
>   1. `spwm/data/sim_6dof.py`: `_solve_6dof_industrial_ik`, `_compute_default_phase_schedule`, `simulate_articulated_trajectory`.
>   2. `spwm/data/io_unity.py`: `parse_unity_trajectory_file`, `export_unity_trajectory_file`.
>   3. `spwm/data/datasets.py`: `partition_trajectories`, `EventWorldDataset`, `collate_event_batches`, `create_dataloaders` (re-exporting moved functions or importing them cleanly).
> - **Allowed Files to Touch:**
>   - Create: `spwm/data/sim_6dof.py`, `spwm/data/io_unity.py`
>   - Modify: `spwm/data/datasets.py`, `spwm/data/__init__.py`
> - **Traps to Avoid:**
>   - **Serialization & Caching Safety:** `EventWorldDataset` and `export_unity_trajectory_file` write `.npz` (`allow_pickle=False`) and `.json`/`.json.gz` files, and return `RoboticTrajectory` (defined in `spwm/data/event_camera.py`). Keep `RoboticTrajectory` in `spwm/data/event_camera.py` so no serialized trajectory metadata breaks.
>   - Keep re-exports of `simulate_articulated_trajectory`, `parse_unity_trajectory_file`, and `export_unity_trajectory_file` in `spwm/data/datasets.py` and `spwm/data/__init__.py` so all tests and `experiments/*.py` scripts remain 100% compatible.
>   - **Do NOT touch** `spwm/training/` or `spwm/evaluation/` in this sub-step.
>   - Strictly obey Layer 1 import boundaries (no `spwm.training` or `spwm.evaluation`) and Zero-Fallback rules.
> - **Definition of Done (Must exit with code 0):**
>   - `.venv/bin/pytest tests/test_event_camera.py tests/test_leakage.py tests/test_overfit.py tests/test_architecture_contracts.py tests/test_anti_fallback.py`
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/`
>   - `.venv/bin/pytest`

---

### Step 6: Standardize Config-First Object Injection & Typed Layer Boundaries

- [x] **Step 6: Standardize Config-First Object Injection & Typed Layer Boundaries**

> **Notes for the Assigned Agent:**
> - **Objective:** Eliminate 19-argument `create_dataloaders(...)` unpacking boilerplate, align `SPWM.__init__` / `Trainer` defaults with `ExperimentConfig`, and replace untyped cross-layer dictionaries with frozen Layer 0 dataclasses.
> - **Allowed Files to Touch:**
>   - `spwm/contracts/states.py` (add typed containers such as `ClosedLoopTrajectory` / `Tier2EvalResult` if replacing ad-hoc dicts, keeping `.to_dict()` / `state_dict()` pure primitive+tensor compatible)
>   - `spwm/data/datasets.py` (add `create_dataloaders_from_config(cfg: ExperimentConfig, ...)` while preserving `create_dataloaders(...)` if tests call it directly)
>   - `spwm/models/world_model.py` (align `SPWM.__init__` fallback defaults with `ModelConfig` / `TrainingConfig` or make `SPWM.from_config` canonical across callers)
>   - `spwm/training/engine.py`, `spwm/training/replay_buffer.py`
>   - `spwm/evaluation/mppi_scorer.py`
>   - `experiments/*.py` and `scripts/*.py` callers of `create_dataloaders`
> - **Traps to Avoid:**
>   - **CRITICAL CHECKPOINT / SERIALIZATION SAFETY CONSTRAINT:** If introducing `ClosedLoopTrajectory` or `Tier2EvalResult` dataclasses in `spwm/contracts/states.py`, **never** pickle dataclass instances directly inside `ClosedLoopBuffer.state_dict()` or `checkpoint.pt`. Convert dataclasses to plain `dict`s of primitives and `torch.Tensor`s inside `state_dict()`, and validate/reconstruct on `load_state_dict()`.
>   - **Test Compatibility:** Many unit tests (`tests/test_models_contracts.py`, `tests/test_anti_fallback.py`, `tests/test_closed_loop_buffer.py`, `tests/test_leakage.py`) instantiate `SPWM(...)`, `Trainer(...)`, or `create_dataloaders(...)` directly with small dimensions (e.g., `latent_dim=32`). Do **not** break existing unit test signatures without updating or preserving keyword compatibility cleanly.
>   - **Zero-Fallback AST Guard:** Do **not** use `dict.get(key, default)` for required fields or 3-argument `getattr`.
> - **Definition of Done (Must exit with code 0):**
>   - Every `experiments/*.py` script constructs dataloaders via `create_dataloaders_from_config(cfg, ...)` (or equivalent single-config helper) rather than manually unpacking 15+ fields.
>   - `.venv/bin/pytest tests/test_anti_fallback.py tests/test_architecture_contracts.py tests/test_config_schema.py tests/test_config_usage.py`
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/`
>   - `.venv/bin/pytest`

---

### Step 7: Reconcile `experiments/` vs. `scripts/`, Harden AST CI Guards & Finalize Docs

- [x] **Step 7: Reconcile `experiments/` vs. `scripts/`, Harden AST CI Guards & Finalize Docs**

> **Notes for the Assigned Agent:**
> - **Objective:** Enforce a clean boundary between `experiments/` (CLI model training, evaluation, optimization, and benchmarks) and `scripts/` (CI verification, diagnostics, and paper figure generation), extend the AST Zero-Fallback CI scanner across `experiments/` and `scripts/`, and synchronize all repository documentation.
> - **Allowed Files to Touch:**
>   - Relocate CLI runner from `spwm/utils/MPPI_loss_finder.py` to `experiments/optimize_mppi_cost.py` (delete `spwm/utils/MPPI_loss_finder.py` after `NeuralCostMPPIController` has been moved in Step 5.2).
>   - Audit & remove/archive stale `experiments/generate_figures.py` (which references obsolete M2/M3 phases) after ensuring `scripts/generate_all_figures.py` covers all active figure workflows.
>   - Fix the self-violation in `scripts/scan_fallbacks.py:186` (`except ValueError: pass` when computing `relative_to`; use `v.file_path.is_relative_to(Path.cwd())` instead).
>   - Extend `tests/test_anti_fallback.py` (or add a test) to scan `spwm/`, `experiments/`, and `scripts/`—fixing any AST violations uncovered in `experiments/` and `scripts/` (e.g., `except Exception` / `strict=False` / 3-arg `getattr`).
>   - Update `docs/WORKSPACE_AND_CONTRACTS.md`, `REPO_SKELETON.md`, and append a clean 5-item completion entry to `docs/History.md`.
> - **Traps to Avoid:**
>   - **Check `scripts/run_pipeline.sh` and `scripts/run_baselines.sh`** before renaming or deleting any file in `experiments/` or `scripts/` so bash orchestration scripts do not point to missing files.
>   - Fixing `scan_fallbacks.py` to scan `experiments/` and `scripts/` may surface existing `except Exception` or `getattr` calls in scripts like `experiments/generate_figures.py` or `scripts/parse_logs.py`. Inspect and fix all surfaced violations cleanly with explicit error propagation rather than suppressing scanner rules.
>   - Keep `docs/History.md` strictly lean and numerical/factual per `AGENTS.md` / `GEMINI.md` rules.
> - **Definition of Done (Must exit with code 0):**
>   - `.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/` exits with code `0` (0 violations).
>   - `.venv/bin/pytest tests/test_anti_fallback.py tests/test_dead_code.py tests/test_architecture_contracts.py tests/test_config_usage.py`
>   - `.venv/bin/pytest`

