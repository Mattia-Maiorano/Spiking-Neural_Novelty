# Repository Skeleton & Configuration Inventory

A factual mapping of directory responsibilities and the configuration inventory of the repository following the SSOT refactoring.

---

## 1. Directory Tree & Module Responsibilities

```text
├── docs/                                # Technical specifications, contracts, and chronological iteration logs
│   ├── Architecture.md                  # Comprehensive SPWM model and layer definitions
│   ├── History.md                       # Chronological numerical metric logs across versions/iterations
│   ├── TRAINING_AND_LOSSES.md           # Loss definitions, curricula, and optimization stages
│   ├── WORKSPACE_AND_CONTRACTS.md       # Layered contract specs, schemas, and interface hierarchy
│   └── WORLD.md                         # Unity simulation and robotic physical setup specs
├── experiments/                         # Executable CLI benchmarks and experiment runners
│   ├── benchmark_mppi_control.py        # Benchmark closed-loop MPPI trajectory tracking across elevation planes
│   ├── benchmark_robustness_stress_test.py # Stress-testing models against event noise/sensor dropouts
│   ├── export_mppi_trajectory.py        # Exports closed-loop MPPI trajectory runs for Unity visualization
│   └── train.py                         # Multi-stage training runner for representation and M1 phases
├── scripts/                             # Utility scripts for AST verification, plotting, and analysis
│   ├── scan_fallbacks.py                # AST CI script enforcing the strict Zero-Fallback policy
│   └── ...
├── sim/                                 # Robotic simulation assets and exported Unity trajectories
│   └── ExportedTrajectories/            # HDF5/NPZ/JSON trajectory files recorded from 6-DOF robot
├── spwm/                                # Core Spiking Predictive World Model package
│   ├── config/                          # Compatibility re-export facade for spwm.contracts.config
│   │   ├── __init__.py                  # Re-exports typed configuration schemas and load_app_config
│   │   └── schema.py                    # Re-export facade to spwm.contracts.config
│   ├── contracts/                       # Layer 0: Immutable dataclasses, protocols, and tensor assertions
│   │   ├── config.py                    # Master SSOT frozen dataclass hierarchy (ExperimentConfig, MPPIConfig, etc.)
│   │   ├── protocols.py                 # Structural typing protocols (WorldModelProtocol, PhaseStrategy, etc.)
│   │   ├── states.py                    # State containers and rollout results (DynamicsState, RolloutResult, etc.)
│   │   ├── tensors.py                   # Runtime tensor dimension/finite assertions and phase-space invariants
│   │   └── version.py                   # Canonical project semantic version tracker (GLOBAL_VERSION)
│   ├── data/                            # Layer 1: Robotics dataset loaders and neuromorphic camera simulation
│   │   ├── datasets.py                  # PyTorch Dataset for loading and windowing robotic trajectories
│   │   ├── event_camera.py              # Log-intensity change spike generator simulator
│   │   └── unity_episode_meta.py        # C# DotNetRandom emulator for trajectory parity
│   ├── models/                          # Layer 1: Neural network architectures and forward kinematics
│   │   ├── dynamics.py                  # SpikingLatentDynamics wrapping ALIF recurrent population
│   │   ├── encoder.py                   # EventEncoder and SpatialSoftmax coordinate keypoint extractor
│   │   ├── kinematics.py                # Differentiable forward kinematics for 6-DOF manipulator
│   │   ├── memory.py                    # Multi-timescale recurrent memory cell
│   │   ├── neurons.py                   # ALIF spiking neuron dynamics and surrogate gradients
│   │   ├── predictors.py                # LatentPredictor (M1), PhysicalDecoder, and LatentCritic
│   │   ├── rollout.py                   # Unified action-conditioned autoregressive latent rollout
│   │   └── world_model.py               # SPWM top-level world model coordinator
│   ├── training/                        # Layer 2: Training loops, losses, and curriculum phases
│   │   ├── engine.py                    # BaseTrainer, Trainer, EMATracker, and ClosedLoopBuffer
│   │   ├── losses.py                    # PredictorLoss and composite SPWMLoss
│   │   ├── metrics.py                   # Multi-horizon error and kinematic metrics
│   │   ├── noise.py                     # AR(1) correlated action noise injection and curriculum schedule
│   │   └── phases/                      # PhaseStrategy implementations (representation, m1_nominal)
│   ├── evaluation/                      # Layer 3: Evaluation benchmarks, controllers, and visualizers
│   │   ├── evaluator.py                 # Multi-horizon test rollout evaluator
│   │   ├── metrics_resume.py            # Summary formatting for benchmark metrics
│   │   ├── mppi.py                      # LatentMPPIController and MPPIStructuralScorer
│   │   ├── robustness.py                # Perturbation robustness testing suite
│   │   └── visualizer.py                # Ghost visualizer exporter and publication plotting
│   └── utils/                           # Cross-cutting utilities
│       ├── reproducibility.py           # Deterministic seeding across PyTorch, NumPy, Python
│       └── MPPI_loss_finder.py          # Optuna optimization utility for MPPI neural cost weights
└── tests/                               # Pytest test suite
    ├── test_anti_fallback.py            # AST and contract anti-fallback verification
    ├── test_architecture_contracts.py   # Tensor dimension, protocol, and interface contract tests
    ├── test_config_schema.py            # Validation and immutability tests for configuration schema
    ├── test_config_usage.py             # AST test verifying that all config fields are referenced in codebase
    ├── test_mppi.py                     # LatentMPPIController and MPPIStructuralScorer tests
    └── test_two_tiered_validation.py    # Two-tiered validation orchestration tests
```

---

## 2. Configuration Schema & Object Injection Inventory

### 2.1 Central SSOT Schema (`spwm/contracts/config.py`)
All configuration dataclasses are `@dataclass(frozen=True)` and reject unknown parameters via `_strict_dict_to_dataclass`:
- `ProjectConfig`: `name: str = "spwm_v0_10_4"`, `seed: int = 42`.
- `EnvironmentConfig`: `joint_count: int = 6`, `action_dim: int = 6`, `height: int = 32`, `width: int = 32`, `dt: float = 0.05`, `event_threshold: float = 0.08`.
- `DataConfig`: `dataset_dir: str = "sim/ExportedTrajectories"`, `joint_count: int = 6`, `action_dim: int = 6`, `total_trajectories: int = 4000`, `train_split: float = 0.8`, `val_split: float = 0.1`, `sequence_length: int = 105`, `batch_size: int = 32`, `standard_velocity_range: Tuple[float, float] = (-1.0, 1.0)`, `extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0)`, `num_workers: int = 0`, `cache_data: bool = True`, `strict_dataset: bool = True`, `pre_contact_sensory_steps: int = 7`, `post_contact_sensory_steps: int = 7`, `weighted_sampling: bool = False`.
- `ModelConfig`: `type: str = "spwm"`, `num_keypoints: int = 16`, `encoder_dim: int = 128`, `encoder_conv_channels: Tuple[int, ...] = (16, 32)`, `latent_dim: int = 128`, `q_dim: int = 32`, `p_dim: int = 96`, `predictor_hidden_dim: int = 256`, `ema_decay: float = 0.9`, `decoder_ema_alpha: float = 0.0`, `decoder_pos_alpha: float = 0.0`, `gate_ramp_steps: int = 4`, `max_k_gate: float = 0.35`, `max_correction_norm: float = 0.50`, `activity_threshold: float = 0.001`, `in_channels: int = 4`, `symplectic: bool = True`.
- `LatentNeuronConfig`: `model: str = "alif"`, `threshold: float = 1.0`, `beta_mem: float = 0.80`, `gamma: float = 0.18`, `surrogate: str = "atan"`, `surrogate_alpha: float = 2.0`.
- `MemoryConfig`: `num_timescales: int = 2`, `timescale_dims: Tuple[int, ...] = (128, 128)`, `betas: Tuple[float, ...] = (0.90, 0.985)`.
- `TrainingConfig`: `mode: str = "train_m1"`, `epochs: int = 100`, `early_stopping_patience: int = 10`, `batch_size: int = 32`, `weight_decay: float = 0.0005`, `grad_clip_norm: float = 0.8`, `learning_rate: float = 0.0001`, `probe_lr: float = 0.0002`, `probe_weight_decay: float = 0.0001`, `temporal_burn_in: int = 15`, `representation_checkpoint: Optional[str] = None`, `phase_horizon: int = 100`, `tier2_frequency: int = 5`, `tier2_trajectories: int = 6`, `tier2_seed: int = 42000`, `tier2_num_samples: int = 1024`, `tier2_horizon: int = 10`, `tier2_iterations: int = 3`, `tier2_candidate_selection: str = "ema"`, `tier2_patience: int = 5`, `tier2_min_delta: float = 0.001`, `ema_decay: float = 0.990`, `mppi_noise_warmup_epochs: int = 10`, `mppi_noise_ramp_epochs: int = 20`, `mppi_noise_target_alpha: float = 0.4`, `max_eval_generations: int = 3`.
- `LossConfig`: `lambda_encoder: float = 1.0`, `lambda_decoder: float = 1.0`, `lambda_latent: float = 1.0`, `lambda_sensory: float = 1.0`, `lambda_position: float = 1.0`, `lambda_velocity: float = 1.5`, `lambda_kinematic: float = 5.0`, `lambda_derivative: float = 1.0`, `lambda_smoothness: float = 0.1`, `lambda_critic: float = 0.2`, `velocity_loss_scale: float = 1.0`.
- `EvaluationConfig`: `rollout_horizons: Tuple[int, ...] = (1, 5, 10, 25, 50, 100)`.
- `MPPIConfig`: `num_samples: int = 256`, `horizon: int = 10`, `action_dim: int = 6`, `action_min: float = -1.0`, `action_max: float = 1.0`, `noise_sigma: Union[float, Tuple[float, ...]] = 0.28`, `noise_beta: float = 0.75`, `temperature: float = 0.15`, `num_iterations: int = 3`, `chunk_size: int = 4`, `gamma_discount: float = 0.95`, `terminal_steps: Optional[int] = None`, `w_progress: float = 50.0`, `w_jerk: float = 0.005`, `w_damping: float = 10.0`, `w_rot: float = 50.0`, `w_pos: float = 20.0`.
- `ExperimentConfig`: Master composite root schema combining all sub-configs (`project`, `environment`, `data`, `model`, `neuron`, `memory`, `training`, `loss`, `evaluation`, `mppi`).
- `load_app_config(path: Optional[Union[str, Path]] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig`: Master factory function.

### 2.2 Class Injection Status
- `LatentMPPIController`: Constructor signature is `__init__(predictor, physical_decoder, config: MPPIConfig, forward_kinematics=None, critic=None, device=None, seed=None)`. Accepts no loose primitive parameters and no `**kwargs`.
- `MPPIStructuralScorer`: Accepts `config: Optional[MPPIConfig]` and instantiates `LatentMPPIController` using `self.config`.
- `NeuralCostMPPIController`: Subclasses `LatentMPPIController`, constructor signature is `__init__(predictor, physical_decoder, config: MPPIConfig, neural_weights: List[float], forward_kinematics=None, critic=None, device=None, seed=None)`.
- Entry points (`benchmark_mppi_control.py`, `benchmark_robustness_stress_test.py`, `export_mppi_trajectory.py`, `visualize_fig1_teaser.py`, `test_mppi.py`, `test_two_tiered_validation.py`): Explicitly instantiate and inject `MPPIConfig`.
