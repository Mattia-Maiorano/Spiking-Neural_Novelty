# Repository Rules & Guidelines for Agents

## Mandatory Documentation Update (`./docs`) at Each Run Ending

Every agent operating in this repository MUST adhere to this rule:

### 1. Trigger
At the conclusion of **every completed run or milestone** (including training phases, pipeline executions, evaluations, simulation exports, or any completed experimental/architectural modification session):

### 2. Mandatory Update to `docs/History.md`
The agent MUST update (or append to) `docs/History.md` with the entry for the version and iteration just tested or modified.

#### Definition of Versions vs. Iterations
- **Iterations increment within the same version**: The "iteration" can and should increase also upon the same version (e.g., multiple successive iterations under `SPWM-v0.6.4` progressing through `Iteration 5`, `Iteration 6`, `Iteration 7`).
- **What constitutes a new iteration**: A new iteration is defined as the **full implementation of a feature, refactor, or fix** (or a completed training run, evaluation benchmark, or formal post-mortem).
- **Agent execution scope**: **Not each agent execution is a new iteration.** Tuning an implementation across prompts in the same chat session does **not** always account for a new iteration. Incremental prompt turns, minor parameter tweaks, and active debugging within the same chat should be consolidated into the single iteration representing that completed feature, refactor, or fix.

#### Entry Structure
Every entry in `docs/History.md` MUST strictly adhere to this 5-item structure:
- **version afflicted**: The version afflicted (e.g., `SPWM-v0.6.4`).
- **iteration on version**: The iteration or phase identifier (e.g., `Iteration 5 (Simulation Heft Smoothing & Smooth Stasis Gate)`).
- **changes made**: Concise summary of what code, architecture, parameters, or simulation scripts were modified.
- **new results (if any)**: Majorly numerical metrics (losses, evaluation errors, MAE, MSE, spike rates, horizon breakdowns). **MUST explicitly tell if no runs were made** (e.g., "No runs made - code refactoring / configuration only").
- **verdict on change made**: Explicit statement of whether the change has been **preserved**, **reverted**, or **discarded**.

> **Constraint on history format**: Keep `History.md` lean, structured, and strictly numerical. Do NOT introduce bloated narrative prose, discursive commentary, or redundant derivations.

### 3. Synchronization of Topic Documentation (`./docs/`)
If the run or change introduced modifications to contracts, models, training logic, or simulation environments, the corresponding documentation in `./docs/` MUST be kept synchronized:
- **`docs/ARCHITECTURE.md`**: For changes to neural networks, models, encoders, dynamics, decoders, or visual pipelines.
- **`docs/WORKSPACE_AND_CONTRACTS.md`**: For changes to config schemas, tensor contracts, data structures, or layer interfaces.
- **`docs/TRAINING_AND_LOSSES.md`**: For changes to loss formulations, optimization curricula, or training strategies.
- **`docs/WORLD.md`**: For changes to Unity physics, C# simulation scripts, action spaces, or robotic rigs.

---

## Strict Anti-Fallback Policy (Zero Fallbacks Allowed)

Every agent operating in this repository MUST adhere to this strict anti-fallback policy. **Fallbacks are strictly prohibited and considered a primary source of architectural decay.**

### 1. Why Fallbacks Are Toxic to this Project
- **Silent Defect Masking**: Fallbacks silently swallow contract violations, missing tensor dimensions, misconfigured models, or corrupted telemetry files. Instead of an immediate, actionable stack trace, corrupt states propagate downstream, causing silent numerical divergence or phantom stability.
- **Divergence Between Environments**: Fallback branches create multiple untracked execution modes between simulation, unit tests, and hardware deployment, invalidating benchmarks.
- **Zombie Code & Technical Debt**: Fallback branches rarely get tested or maintained, accumulating dead code paths that complicate refactoring and AST verification.

### 2. Prohibited Fallback Patterns (Enforced by AST CI)
Every agent MUST avoid the following patterns across Python and simulation scripts:

1. **NO 3-Argument `getattr(obj, attr, default)`**:
   - ❌ FORBIDDEN: `val = getattr(model, "encoder_dim", 64)` or `getattr(f_type, "__origin__", None)`.
   - ✅ REQUIRED: Direct typed attribute access `val = model.encoder_dim` or standard introspector `get_origin(f_type)`. If an attribute is contractually required, access it directly so missing attributes raise `AttributeError`.

2. **NO Ternary `hasattr(...)` Fallbacks**:
   - ❌ FORBIDDEN: `x = model.dim if hasattr(model, "dim") else 128`.
   - ✅ REQUIRED: Rely on strict Layer 0 protocols and contract dataclasses (`WorldModelProtocol`, `DynamicsProtocol`). If an optional capability is checked, do not branch to a synthetic fallback default.

3. **NO Checkpoint Loading with `strict=False`**:
   - ❌ FORBIDDEN: `model.load_state_dict(ckpt, strict=False)`.
   - ✅ REQUIRED: `model.load_state_dict(ckpt, strict=True)`. Checkpoints must strictly match the model architecture.

4. **NO Silent Exception Swallowing**:
   - ❌ FORBIDDEN: `except Exception: pass`, `except TypeError: return default`, or `except AttributeError: return fallback`.
   - ✅ REQUIRED: Catch only expected errors when necessary, and always propagate or re-raise with informative diagnostic context (`raise ValueError(...) from err`).

5. **NO Silent Configuration or Data Fallbacks**:
   - ❌ FORBIDDEN: `dict.get(key, fallback)` masking missing configuration schemas, or generating synthetic trajectories when real data is requested.
   - ✅ REQUIRED: `ExperimentConfig` and `_strict_dict_to_dataclass` must raise `ValueError` on unknown or missing parameters. Fail loudly and immediately.

### 3. Verification & Spotting Tool
To spot and eliminate fallbacks across the codebase:
- Run the AST scanner:
  ```bash
  python scripts/scan_fallbacks.py spwm/
  ```
- Run the anti-fallback test suite:
  ```bash
  .venv/bin/pytest tests/test_anti_fallback.py tests/test_architecture_contracts.py
  ```


