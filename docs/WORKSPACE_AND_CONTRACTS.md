# Workspace Topology & Layer 0 Contracts (`spwm/contracts/`)

This guide provides a file-by-file breakdown of the **4-Layer Strict Contract Architecture** in SPWM-v0.7.3, the immutable configuration and tensor validation contracts in `spwm/contracts/`, and the context-window boundary rules designed for AI-assisted and small-LLM engineering.

---

## 1. The 4-Layer Strict Dependency Hierarchy

Every Python module inside `spwm/` belongs to one of four strictly ordered layers. Imports may only flow **downward** (from higher layer numbers to lower layer numbers):

| Layer | Package Path | Role | Permitted Internal Imports |
| :--- | :--- | :--- | :--- |
| **Layer 0** | [`spwm/contracts/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py) | Pure immutable dataclasses, state containers, tensor shape validators (including 6-DOF joint count $J=6$ and action vectors $\mathbf{u}_t$), and `Protocol` / `ABC` interfaces. | **None** (Zero `spwm.*` imports) |
| **Layer 1** | [`spwm/models/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py), [`spwm/data/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py) | Pure neural network components, action-conditioned attractor predictor, unified latent rollout engine, robotic trajectory containers with `phase_flags`, and neuromorphic event sensor simulators. | `spwm.contracts` (and intra-layer `spwm.models` / `spwm.data`) |
| **Layer 2** | [`spwm/training/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/__init__.py) | Composable loss formulations, isolated $<150$-line training phase strategies (`representation`, `m1_nominal`), articulated metrics, and `BaseTrainer` / `Trainer`. | `spwm.contracts`, `spwm.models`, `spwm.data`, `spwm.utils` |
| **Layer 3** | [`spwm/evaluation/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/__init__.py) | Action-conditioned multi-horizon rollout benchmarks (`RolloutEvaluator`), in-engine ghost trajectory exporters (`GhostVisualizerExporter`), segmented telemetry resumes (`metrics_resume.py`), downstream MPPI controller (`mppi.py`), sim-to-real robustness benchmarks (`robustness.py`), and publication plotting (`visualizer.py`). | `spwm.contracts`, `spwm.models`, `spwm.data`, `spwm.training`, `spwm.utils` |

> **Compatibility Facades**: `spwm/config/`, `spwm/core/`, and `spwm/learning/` are thin, zero-logic re-export packages maintained so external scripts importing legacy paths resolve directly to the canonical Layer 0, Layer 1, and Layer 2 implementations.

---

## 2. Layer 0 Contract Modules (`spwm/contracts/`)

### 2.1 [`spwm/contracts/config.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py) — Immutable Single Source of Truth (SSOT)
All configuration schemas are `@dataclass(frozen=True)` classes:
- [`ProjectConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `name` (`"spwm_v0_7_3"`), `seed` (`42`).
- [`EnvironmentConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `joint_count` ($J=6$), `action_dim` ($D_u=6$), `height`, `width`, `dt` ($0.05\,\text{s}$), `event_threshold` ($0.08$). Validates in `__post_init__` that `joint_count > 0`, `action_dim > 0`, and `height` and `width` are positive multiples of $4$ (matching the two stride-2 convolutions in `EventEncoder`).
- [`DataConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `dataset_dir` (`"sim/ExportedTrajectories"`), `joint_count` ($J=6$), `action_dim` ($D_u=6$), `total_trajectories` ($4000$), `train_split` ($0.8$), `val_split` ($0.1$), `sequence_length` ($105$), `batch_size` ($32$), `standard_velocity_range` ($(-1.0, 1.0)$), `extrapolation_velocity_range` ($(-2.0, 2.0)$), `num_workers` ($0$), `cache_data` ($True$), `strict_dataset` ($True$), `pre_contact_sensory_steps` ($7$), `post_contact_sensory_steps` ($7$). Supports `weighted_sampling` via `WeightedRandomSampler` exposing $M_1$ to $2.5\times$ Heavy payloads and $2.5\times$ Mid-plane elevations ($5.0\times$ joint).
- [`ModelConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `type` (`"spwm"`), `num_keypoints` ($16$), `encoder_dim` ($128$), `encoder_conv_channels` (`(16, 32)`), `latent_dim` ($128$), `q_dim` ($32$), `p_dim` ($96$), `predictor_hidden_dim` ($256$), `ema_decay` ($0.9$), `decoder_ema_alpha` ($0.0$), `decoder_pos_alpha` ($0.0$), `gate_ramp_steps` ($4$), `max_k_gate` ($0.35$), `max_correction_norm` ($0.50$), `activity_threshold` ($0.001$), `in_channels` ($C_{\text{event}}=4$), `symplectic` ($True$). Validates phase-space invariants (`q_dim + p_dim == latent_dim`, `q_dim == 2 * num_keypoints`, and `p_dim` is an integer multiple of `q_dim`), `activity_threshold >= 0`, `in_channels in (2, 4)`, and `len(encoder_conv_channels) == 2`.
- [`LatentNeuronConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `model` (`"alif"`), `threshold` ($1.0$), `beta_mem` ($0.80$), `gamma` ($0.18$), `surrogate` (`"atan"`), `surrogate_alpha` ($2.0$).
- [`MemoryConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `num_timescales` ($2$), `timescale_dims` (`(128, 128)`), `betas` (`(0.90, 0.985)`). Validates `len(timescale_dims) == num_timescales == len(betas)` and $0 < \beta_k < 1$.
- [`TrainingConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `mode` (`"train_m1"` default; allowed `"train_representation"`, `"train_m1"`), `epochs` ($100$), `early_stopping_patience` ($10$), `batch_size` ($32$), `weight_decay` ($0.0005$), `grad_clip_norm` ($0.8$), `learning_rate` ($0.0002$), `probe_lr` ($0.0002$), `probe_weight_decay` ($0.0001$), `temporal_burn_in` ($15$), `representation_checkpoint` (`None`), `phase_horizon` ($50$), `tier2_frequency` ($10$), `tier2_trajectories` ($6$), `tier2_seed` ($42000$), `tier2_num_samples` ($256$), `tier2_horizon` ($30$), `tier2_iterations` ($3$), `tier2_candidate_selection` (`"best_in_block"`), `tier2_patience` ($3$), `tier2_min_delta` ($0.001$).
- [`LossConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `lambda_encoder` ($1.0$), `lambda_decoder` ($1.0$), `lambda_latent` ($1.0$), `lambda_sensory` ($1.0$), `lambda_position` ($1.0$), `lambda_velocity` ($1.5$), `lambda_kinematic` ($5.0$), `lambda_derivative` ($1.0$), `lambda_smoothness` ($0.1$), `velocity_loss_scale` ($1.0$).
- [`EvaluationConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): `rollout_horizons` (`(1, 5, 10, 25, 50, 100)`).
- [`ExperimentConfig`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/config.py): Root composite schema.
  - Cross-validates `data.sequence_length` against the active training phase (`train_representation` requires $T \ge 2$; `train_m1` requires $T \ge H + 1$).
  - Validates full phase-space contracts including `action_dim`.
  - Provides `cfg.with_overrides(**kwargs)` to create a new frozen configuration with updated fields and full re-validation.
  - Rejects unknown dictionary keys in `ExperimentConfig.from_dict()` with an immediate `ValueError` (zero fallbacks allowed).
  - Enforces zero duck typing and zero 3-argument `getattr` calls via standard `typing.get_origin`.

### 2.2 [`spwm/contracts/tensors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py) — Runtime Tensor & Phase-Space Validators
- `J`: Master 6-DOF manipulator joint count constant ($J=6$).
- `C_EVENT`: Master dual-camera neuromorphic event channel constant ($C_{\text{event}}=4$: 2 static global + 2 egocentric wrist).
- [`assert_tensor_contract(tensor, expected_shape, name, check_finite=False)`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py): Verifies rank, concrete dimensions (where `None` denotes any wildcard size), and finite values when `check_finite=True` (raising `ValueError` on `NaN`/`Inf`).
- [`assert_phase_space_contract(latent_dim, q_dim, p_dim, num_keypoints=None, action_dim=None, actions=None)`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py): Enforces $D = D_q + D_p$, $D_q = 2K$, integer divisibility $D_p \equiv 0 \pmod{D_q}$, and validates motor action tensor $\mathbf{u}_t$ dimensionality ($D_u=6$), rank $\ge 2$, and finiteness.
- [`assert_euclidean_coordinates(q, name="q_coordinates", tolerance=1e-5)`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py): Verifies $\mathbf{q} \in [-1.0 - \text{tolerance}, 1.0 + \text{tolerance}]^{D_q}$.

### 2.3 [`spwm/contracts/states.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py) — Neural State & Rollout Containers
- [`ALIFState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(v_mem, a_adapt, spikes)` with `.detach()`.
- [`MultiTimescaleState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `population_states: List[ALIFState]` with `.v_mems`, `.a_adapts`, `.spikes`, `.concatenated_spikes`, `.concatenated_mems`, `.fast_spikes`, `.slow_spikes`, and `.detach()`.
- [`DynamicsState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(memory_state, z_prev, ema_spikes, grounded_steps)` with `.detach()`.
- [`DynamicsOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(latent_states, fast_spikes, slow_spikes, fast_mems, slow_mems, mean_spike_rate)`.
- [`PredictorOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(predicted_latent, mean, log_var)`.
- [`SPWMState`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(encoder_states, dynamics_state, action)`.
- [`SPWMStepOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(latent, predicted_next_latent, error_neurons, prediction_error, decoded_kinematics, ema_spikes, keypoints, action)`.
- [`SPWMSequenceOutput`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): `(latent_states, predicted_latents, prediction_errors, error_neurons, fast_spikes, slow_spikes, encoder_spike_rate, dynamics_spike_rate, mean_spike_rate, decoded_kinematics, keypoints)`.
- [`TelemetryContainer`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): Container for logging fine-grained spiking and latent telemetry (`spikes`, `v_mem`, `v_th`, `gain_mask`) with `.append(...)` and `.as_stacked_tensors()`.
- [`RolloutResult`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py): Standardized container for action-conditioned `rollout_latent` with articulated `.positions` (`[..., :joint_count]`) and `.velocities` (`[..., joint_count : 2 * joint_count]`), `actions`, and telemetry (default `joint_count = 6`).

### 2.4 [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py) — Structural Protocols & `PhaseStrategy` ABC
Defines `@runtime_checkable` protocols:
- `EncoderProtocol`: `in_channels`, `num_keypoints`, `out_dim`, `encode_step`, `__call__`.
- `DynamicsProtocol`: `input_dim`, `latent_dim`, `q_dim`, `p_dim`, `init_state`, `step` (incorporating `event_activity`, `in_sensory_phase`, `unconstrained`, and `action`), `__call__`.
- `PredictorModuleProtocol`: `latent_dim`, `action_dim`, `__call__`.
- `PredictorProtocol`: `latent_dim`, `action_dim`, `hidden_dim`, `symplectic`, `dt`, `__call__`.
- `PhysicalDecoderProtocol`: `latent_dim`, `q_dim`, `p_dim`, `joint_count`, `decode_pos`, `decode_vel`, `warm_up_filter`, `__call__`.
- `WorldModelProtocol`: `latent_dim`, `encoder_dim`, `q_dim`, `p_dim`, `joint_count`, `action_dim`, `encoder`, `dynamics`, `predictor`, `physical_decoder`, `m1`, `forward_representation`, `__call__`.
- `Tier2EvaluatorProtocol`: `frequency`, `__call__(model, epoch) -> Dict[str, float]` for closed-loop validation evaluation.
- `PhaseStrategy` abstract base class: `setup`, `verify_gradient_isolation`, `train_step`, `validate_step`, `log_epoch`, `get_selection_score`, `get_optimizers_state`, `load_optimizers_state` (`PhaseHandler` alias for backward compatibility).

---

## 3. Small-LLM Context Boundary Rules (`< 3 Files` Rule)

When modifying any component of SPWM-v0.7.3, follow this table to keep context windows minimal and prevent cross-module regressions:

| Modification Task | Files to Load into Context ($\le 3$ Files) | Verification Command |
| :--- | :--- | :--- |
| **Modify Phase 0 (`train_representation`) logic** | 1. [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)<br>2. [`spwm/training/phases/representation.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/representation.py)<br>3. [`tests/test_training_phases.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_training_phases.py) | `.venv/bin/pytest tests/test_training_phases.py` |
| **Modify Phase 1 (`train_m1`) action conditioning** | 1. [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)<br>2. [`spwm/training/phases/m1_nominal.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/phases/m1_nominal.py)<br>3. [`tests/test_training_phases.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_training_phases.py) | `.venv/bin/pytest tests/test_training_phases.py tests/test_relay.py` |
| **Modify Action-Conditioned `LatentPredictor`** | 1. [`spwm/contracts/protocols.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/protocols.py)<br>2. [`spwm/models/predictors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py)<br>3. [`tests/test_relay.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_relay.py) | `.venv/bin/pytest tests/test_relay.py` |
| **Modify Articulated `PhysicalDecoder`** | 1. [`spwm/contracts/tensors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py)<br>2. [`spwm/models/predictors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictors.py)<br>3. [`tests/test_models_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models_contracts.py) | `.venv/bin/pytest tests/test_models_contracts.py` |
| **Modify Unified `rollout_latent` with actions & Active Perception** | 1. [`spwm/contracts/states.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py)<br>2. [`spwm/models/rollout.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/rollout.py)<br>3. [`tests/test_unified_rollout.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_unified_rollout.py) | `.venv/bin/pytest tests/test_unified_rollout.py` |
| **Modify `EventEncoder` / `SpatialSoftmax`** | 1. [`spwm/contracts/tensors.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/tensors.py)<br>2. [`spwm/models/encoder.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py)<br>3. [`tests/test_models_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models_contracts.py) | `.venv/bin/pytest tests/test_models_contracts.py` |
| **Modify `SpikingLatentDynamics` / `ALIFCell`** | 1. [`spwm/contracts/states.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/states.py)<br>2. [`spwm/models/dynamics.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/dynamics.py)<br>3. [`tests/test_models_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_models_contracts.py) | `.venv/bin/pytest tests/test_models_contracts.py tests/test_alif.py` |

