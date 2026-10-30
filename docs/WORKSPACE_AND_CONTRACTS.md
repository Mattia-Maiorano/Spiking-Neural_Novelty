# Workspace Topology & Layer 0 Contracts (`spwm/contracts/`)

This guide provides a file-by-file breakdown of the **4-Layer Strict Contract Architecture** in SPWM-v0.5.0, the immutable configuration and tensor validation contracts in `spwm/contracts/`, and the context-window boundary rules designed for AI-assisted and small-LLM engineering.

---

## 1. The 4-Layer Strict Dependency Hierarchy

Every Python module inside `spwm/` belongs to one of four strictly ordered layers. Imports may only flow **downward** (from higher layer numbers to lower layer numbers):

| Layer | Package Path | Role | Permitted Internal Imports |
| :--- | :--- | :--- | :--- |
| **Layer 0** | [`spwm/contracts/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py) | Pure immutable dataclasses, state containers, tensor shape validators (including 6-DOF joint count $J=6$ and action vectors $\mathbf{u}_t$), and `Protocol` / `ABC` interfaces. | **None** (Zero `spwm.*` imports) |
| **Layer 1** | [`spwm/models/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py), [`spwm/data/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py) | Pure neural network components, action-conditioned attractor predictor, unified latent rollout engine, robotic trajectory containers with `phase_flags`, and neuromorphic event sensor simulators. | `spwm.contracts` (and intra-layer `spwm.models` / `spwm.data`) |
| **Layer 2** | [`spwm/training/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/__init__.py) | Composable loss formulations, isolated $<150$-line training phase strategies (`representation`, `m1_nominal`), articulated metrics, and `BaseTrainer` / `Trainer`. | `spwm.contracts`, `spwm.models`, `spwm.data`, `spwm.utils` |
| **Layer 3** | [`spwm/evaluation/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/__init__.py) | Action-conditioned multi-horizon rollout benchmarks (`RolloutEvaluator`), in-engine ghost trajectory exporters (`GhostVisualizerExporter`), segmented telemetry resumes (`metrics_resume.py`), and publication plotting (`visualizer.py`). | `spwm.contracts`, `spwm.models`, `spwm.data`, `spwm.training`, `spwm.utils` |

> **Compatibility Facades**: `spwm/config/`, `spwm/core/`, and `spwm/learning/` are thin, zero-logic re-export packages maintained so external scripts importing legacy paths resolve directly to the canonical Layer 0, Layer 1, and Layer 2 implementations.

---

## 2. Layer 0 Contract Modules (`spwm/contracts/`)

### 2.1 [`spwm/contracts/config.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py) — Immutable Single Source of Truth (SSOT)
All configuration schemas are `@dataclass(frozen=True)` classes:
- [`ProjectConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `name` (`"spwm_v0_6_0"`), `seed` (`42`).
- [`EnvironmentConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `joint_count` ($J=6$), `action_dim` ($D_u=6$), `height`, `width`, `dt`, `event_threshold`. Validates in `__post_init__` that `height` and `width` are positive multiples of $4$ (matching the two stride-2 convolutions in `EventEncoder`).
- [`DataConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `dataset_dir` (`"sim/ExportedTrajectories"`), `joint_count` ($J=6$), `action_dim` ($D_u=6$), `total_trajectories`, `train_split`, `val_split`, `sequence_length`, `batch_size`, `standard_velocity_range`, `extrapolation_velocity_range`, `strict_dataset`, `pre_contact_sensory_steps`.
- [`ModelConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `in_channels`, `encoder_conv_channels`, `encoder_dim`, `latent_dim`, `q_dim`, `p_dim`, `num_keypoints`, `predictor_hidden_dim`, `ema_decay`, `decoder_ema_alpha`, `decoder_pos_alpha`, `gate_ramp_steps`, `max_k_gate`, `max_correction_norm`, `symplectic`. Validates phase-space invariants (`q_dim + p_dim == latent_dim`, `q_dim == 2 * num_keypoints`, and `p_dim` is a multiple of `q_dim`).
- [`LatentNeuronConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `beta_mem`, `threshold`, `gamma`, `surrogate`, `surrogate_alpha`.
- [`MemoryConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `num_timescales`, `timescale_dims`, `betas`. Validates `len(timescale_dims) == num_timescales == len(betas)` and $0 < \beta_k < 1$.
- [`TrainingConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `mode` (`"train_representation"`, `"train_m1"`), `epochs`, `learning_rate`, `weight_decay`, `grad_clip_norm`, `probe_lr`, `probe_weight_decay`, `phase_horizon`, `temporal_burn_in`, `early_stopping_patience`, `representation_checkpoint`.
- [`LossConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `lambda_position`, `lambda_velocity`, `lambda_latent`, `lambda_encoder`, `lambda_decoder`, `lambda_sensory`, `lambda_kinematic`, `lambda_derivative`, `lambda_smoothness`, `velocity_loss_scale`.
- [`EvaluationConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `rollout_horizons`.
- [`ExperimentConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): Root composite schema.
  - Cross-validates `data.sequence_length` against the active training phase (`train_representation` requires $T \ge 2$; `train_m1` requires $T \ge H + 1$).
  - Validates full phase-space contracts including `action_dim`.
  - Provides `cfg.with_overrides(**kwargs)` to create a new frozen configuration with updated fields and full re-validation.
  - Rejects unknown dictionary keys in `ExperimentConfig.from_dict()` with an immediate `ValueError`.

### 2.2 [`spwm/contracts/tensors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py) — Runtime Tensor & Phase-Space Validators
- `J`: Master 6-DOF manipulator joint count constant ($J=6$).
- `C_EVENT`: Master dual-camera neuromorphic event channel constant ($C_{\text{event}}=4$: 2 static global + 2 egocentric wrist).
- [`assert_tensor_contract(tensor, expected_shape, name, allow_nan=False)`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py): Verifies rank, concrete dimensions (where `None` denotes any positive size), and finite values (raising `ValueError` on `NaN`/`Inf`).
- [`assert_phase_space_contract(latent_dim, q_dim, p_dim, num_keypoints, action_dim, actions)`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py): Enforces $D = D_q + D_p$, $D_q = 2K$, integer divisibility $D_p \equiv 0 \pmod{D_q}$, and validates motor action tensor $\mathbf{u}_t$ dimensionality ($D_u=6$), rank $\ge 2$, and finiteness.
- [`assert_euclidean_coordinates(q, name, bound=1.0)`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py): Verifies $\mathbf{q} \in [-1, 1]^{D_q}$.

### 2.3 [`spwm/contracts/states.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py) — Neural State & Rollout Containers
- [`ALIFState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(v_mem, a_adapt, spikes)` with `.detach()`.
- [`MultiTimescaleState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `population_states: List[ALIFState]` with `.concatenated_spikes`, `.concatenated_mems`, `.fast_spikes`, `.slow_spikes`.
- [`DynamicsState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(memory_state, z_prev, ema_spikes, grounded_steps)`.
- [`DynamicsOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py), [`PredictorOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py), [`SPWMState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py) (includes `action: Optional[torch.Tensor]`), [`SPWMStepOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py), [`SPWMSequenceOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py).
- [`RolloutResult`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): Standardized container for action-conditioned `rollout_latent` with articulated `.positions` (`[..., :joint_count]`) and `.velocities` (`[..., joint_count : 2 * joint_count]`), `actions`, and telemetry (default `joint_count = 6`).

### 2.4 [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py) — Structural Protocols & `PhaseStrategy` ABC
Defines `@runtime_checkable` protocols (`EncoderProtocol`, `DynamicsProtocol`, `PredictorProtocol`, `PhysicalDecoderProtocol`, `WorldModelProtocol`) and the `PhaseStrategy` abstract base class (`setup`, `verify_gradient_isolation`, `train_step`, `validate_step`).
`DynamicsProtocol.step` and `forward` incorporate `event_activity`, `in_sensory_phase`, and `unconstrained` gating parameters.

---

## 3. Small-LLM Context Boundary Rules (`< 3 Files` Rule)

When modifying any component of SPWM-v0.5.0, follow this table to keep context windows minimal and prevent cross-module regressions:

| Modification Task | Files to Load into Context ($\le 3$ Files) | Verification Command |
| :--- | :--- | :--- |
| **Modify Phase 0 (`train_representation`) logic** | 1. [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)<br>2. [`spwm/training/phases/representation.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/representation.py)<br>3. [`tests/test_training_phases.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_training_phases.py) | `.venv/bin/pytest tests/test_training_phases.py` |
| **Modify Phase 1 (`train_m1`) action conditioning** | 1. [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)<br>2. [`spwm/training/phases/m1_nominal.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/m1_nominal.py)<br>3. [`tests/test_training_phases.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_training_phases.py) | `.venv/bin/pytest tests/test_training_phases.py tests/test_relay.py` |
| **Modify Action-Conditioned `LatentPredictor`** | 1. [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)<br>2. [`spwm/models/predictors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py)<br>3. [`tests/test_relay.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_relay.py) | `.venv/bin/pytest tests/test_relay.py` |
| **Modify Articulated `PhysicalDecoder`** | 1. [`spwm/contracts/tensors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py)<br>2. [`spwm/models/predictors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py)<br>3. [`tests/test_models_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models_contracts.py) | `.venv/bin/pytest tests/test_models_contracts.py` |
| **Modify Unified `rollout_latent` with actions & Active Perception** | 1. [`spwm/contracts/states.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py)<br>2. [`spwm/models/rollout.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/rollout.py)<br>3. [`tests/test_unified_rollout.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_unified_rollout.py) | `.venv/bin/pytest tests/test_unified_rollout.py` |
| **Modify `EventEncoder` / `SpatialSoftmax`** | 1. [`spwm/contracts/tensors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py)<br>2. [`spwm/models/encoder.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py)<br>3. [`tests/test_models_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models_contracts.py) | `.venv/bin/pytest tests/test_models_contracts.py` |
| **Modify `SpikingLatentDynamics` / `ALIFCell`** | 1. [`spwm/contracts/states.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py)<br>2. [`spwm/models/dynamics.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py)<br>3. [`tests/test_models_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models_contracts.py) | `.venv/bin/pytest tests/test_models_contracts.py tests/test_alif.py` |
