# Repository Rules & Guidelines for Agents
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


