"""
Layer 0 Typed Configuration Schema for SPWM (Single Source of Truth).
All configuration dataclasses are strictly frozen (immutable) and validated on construction.
Unknown keys or violated cross-field invariants raise ValueError immediately.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union, get_origin, get_type_hints
import yaml

from spwm.contracts.tensors import C_EVENT, J, assert_phase_space_contract
from spwm.contracts.version import GLOBAL_VERSION

T = TypeVar("T")


def _strict_dict_to_dataclass(cls: Type[T], data: Dict[str, Any], context: str = "") -> T:
    """Instantiate a frozen dataclass strictly: unknown keys raise ValueError immediately."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary for {context or cls.__name__}, got {type(data).__name__}: {data}")

    type_hints = get_type_hints(cls, globalns=sys.modules[__name__].__dict__)
    valid_fields = {f.name: f for f in fields(cls)}
    unknown_keys = set(data.keys()) - set(valid_fields.keys())
    if unknown_keys:
        ctx_str = f" in {context}" if context else f" in {cls.__name__}"
        raise ValueError(f"Unknown configuration parameter(s){ctx_str}: {sorted(list(unknown_keys))}. No fallbacks allowed.")

    kwargs: Dict[str, Any] = {}
    for f_name, f_field in valid_fields.items():
        f_type = type_hints.get(f_name, f_field.type)
        if f_name in data:
            val = data[f_name]
            if isinstance(f_type, type) and is_dataclass(f_type) and isinstance(val, dict):
                kwargs[f_name] = _strict_dict_to_dataclass(f_type, val, context=f"{context}.{f_name}" if context else f_name)
            elif val is not None:
                origin = get_origin(f_type)
                if origin is tuple and isinstance(val, (list, tuple)):
                    val = tuple(val)
                elif origin is list and isinstance(val, (list, tuple)):
                    val = list(val)
                kwargs[f_name] = val
            else:
                kwargs[f_name] = None
        else:
            if f_field.default is not MISSING:
                kwargs[f_name] = f_field.default
            elif f_field.default_factory is not MISSING:
                kwargs[f_name] = f_field.default_factory()
            else:
                ctx_str = f" for {context}.{f_name}" if context else f" for {cls.__name__}.{f_name}"
                raise ValueError(f"Missing required configuration parameter{ctx_str}.")

    return cls(**kwargs)


@dataclass(frozen=True)
class ProjectConfig:
    """Project-level identity, tracking tags, and pseudo-random reproducibility configuration."""
    name: str = GLOBAL_VERSION.snake_case  #: Unique identifier/tag for the project and run artifacts.
    seed: int = 42                         #: Global pseudo-random seed for PyTorch, NumPy, and trajectory generation.


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    3D Unity Articulated Robotics environment and neuromorphic event camera sensor configuration.
    Controls the 6-DOF articulated manipulator chain (joint_count=6), motor command dimension (action_dim=6),
    discrete frame timesteps, and event sensor log-intensity spike generation thresholds.
    """
    joint_count: int = J           #: Number of articulated joints J=6 in the robotic manipulator chain.
    action_dim: int = J            #: Dimensionality D_u=6 of continuous motor action vectors u_t.
    height: int = 32               #: Pixel height of the event camera sensor plane (must be divisible by 4).
    width: int = 32                #: Pixel width of the event camera sensor plane (must be divisible by 4).
    dt: float = 0.05               #: Simulation time step size in seconds per discrete physics frame.
    event_threshold: float = 0.08  #: Log-intensity change threshold required to trigger a polarity spike (+1/-1).

    def __post_init__(self) -> None:
        if self.joint_count <= 0:
            raise ValueError(f"joint_count must be > 0, got {self.joint_count}")
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.action_dim}")
        if self.height <= 0 or self.height % 4 != 0:
            raise ValueError(f"Environment height ({self.height}) must be positive and divisible by 4 for 2-stage stride-2 conv encoder.")
        if self.width <= 0 or self.width % 4 != 0:
            raise ValueError(f"Environment width ({self.width}) must be positive and divisible by 4 for 2-stage stride-2 conv encoder.")


@dataclass(frozen=True)
class DataConfig:
    """
    Unity robotics dataset splitting, trajectory directory, sequence length, batching,
    joint count, and action dimensionality configuration.
    """
    dataset_dir: str = "sim/ExportedTrajectories"   #: Path to exported Unity robotic trajectory files.
    joint_count: int = J                            #: Number of articulated joints J=6 expected in dataset trajectories.
    action_dim: int = J                             #: Dimensionality D_u=6 of motor action vectors in dataset trajectories.
    total_trajectories: int = 4000                  #: Total number of trajectories across all dataset splits.
    train_split: float = 0.8                        #: Fraction of trajectories dedicated to training (must satisfy train+val < 1.0).
    val_split: float = 0.1                          #: Fraction of trajectories dedicated to validation (remaining is test).
    sequence_length: int = 105                      #: Number of consecutive temporal steps per trajectory.
    batch_size: int = 32                            #: Dataloader batch size during training and validation.
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0)       #: Nominal joint angular velocity scaling range.
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0)  #: Extrapolation joint angular velocity range.
    num_workers: int = 0                            #: Number of subprocesses used by PyTorch DataLoaders.
    cache_data: bool = True                         #: Whether to cache rendered event frames in RAM to accelerate training epochs.
    strict_dataset: bool = True                     #: Whether to strictly require real exported Unity trajectories (fail if missing).
    pre_contact_sensory_steps: int = 7              #: Number of approach steps before physical contact where Phase 1.5 sensory grounding begins.
    post_contact_sensory_steps: int = 7             #: Number of steps after physical contact where Phase 1.5 sensory grounding is maintained before release.
    weighted_sampling: bool = False                 #: Whether to apply weighted oversampling for heavy payloads and mid-plane reach trajectories.

    def __post_init__(self) -> None:
        if self.joint_count <= 0:
            raise ValueError(f"DataConfig.joint_count must be > 0, got {self.joint_count}")
        if self.action_dim <= 0:
            raise ValueError(f"DataConfig.action_dim must be > 0, got {self.action_dim}")
        if self.train_split + self.val_split >= 1.0:
            raise ValueError(f"train_split ({self.train_split}) + val_split ({self.val_split}) must be < 1.0")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.sequence_length}")


@dataclass(frozen=True)
class ModelConfig:
    """
    Spiking Predictive World Model (SPWM) neural architecture dimensions and coupling parameters.
    Defines coordinate/momentum subspace partitioning (latent_dim = q_dim + p_dim)
    and spatial keypoint bottlenecks.
    """
    type: str = "spwm"                                #: Architecture family ("spwm").
    num_keypoints: int = 16                           #: Number of spatial feature keypoints extracted by the encoder.
    encoder_dim: int = 128                            #: Output feature dimension of the convolutional feature extractor.
    encoder_conv_channels: Tuple[int, ...] = (16, 32) #: Channel depths for 2D convolutional encoder layers.
    latent_dim: int = 128                             #: Total latent state dimension; MUST equal q_dim + p_dim.
    q_dim: int = 32                                   #: Coordinate/position subspace dimension; MUST equal num_keypoints * 2.
    p_dim: int = 96                                   #: Momentum/velocity subspace dimension; MUST be a multiple of q_dim.
    predictor_hidden_dim: int = 256                   #: Hidden layer dimensionality of the predictor residual MLP network.
    ema_decay: float = 0.9                            #: Exponential moving average decay for tracking latent spike statistics.
    decoder_ema_alpha: float = 0.0                    #: EMA filter smoothing coefficient for PhysicalDecoder momentum readout (0.0 for raw/unfiltered).
    decoder_pos_alpha: float = 0.0                    #: EMA filter smoothing coefficient for PhysicalDecoder position readout (0.0 to disable).
    gate_ramp_steps: int = 4                          #: Number of transition steps for smooth sensory gate ramp (reaching full injection 3-4 steps ahead of contact).
    max_k_gate: float = 0.35                          #: Maximum sensory injection gate weight to maintain inertial damping.
    max_correction_norm: float = 0.50                 #: Maximum Euclidean norm of single-step sensory correction in coordinate space.
    activity_threshold: float = 0.001                 #: Minimum event activity threshold below which sensory correction is suppressed (Anti-Evanescence gate).
    in_channels: int = C_EVENT                        #: Dual-camera event input polarity channels (4 = 2 static global + 2 egocentric wrist).
    symplectic: bool = True                           #: Whether to use dissipative phase-space integration for M1.

    def __post_init__(self) -> None:
        assert_phase_space_contract(
            latent_dim=self.latent_dim,
            q_dim=self.q_dim,
            p_dim=self.p_dim,
            num_keypoints=self.num_keypoints,
        )
        if self.activity_threshold < 0.0:
            raise ValueError(f"activity_threshold must be non-negative, got {self.activity_threshold}")
        if self.in_channels not in (2, C_EVENT):
            raise ValueError(
                f"in_channels must be 2 (monocular) or {C_EVENT} (dual-camera global+wrist), got {self.in_channels}"
            )
        if len(self.encoder_conv_channels) != 2:
            raise ValueError(
                f"encoder_conv_channels must specify exactly 2 channel depths, got {self.encoder_conv_channels}"
            )


@dataclass(frozen=True)
class LatentNeuronConfig:
    """
    Biophysical neuron hyperparameters for the recurrent Spiking Latent Dynamics core (z_t).
    Controls the Adaptive Leaky Integrate-and-Fire (ALIF) membrane decay, dynamic threshold adaptation,
    and pseudo-derivative surrogate gradient shape for non-differentiable spike events.
    """
    model: str = "alif"             #: Spiking neuron model ("alif").
    threshold: float = 1.0          #: Baseline membrane potential firing threshold v_th0.
    beta_mem: float = 0.80          #: Membrane potential leak decay factor: exp(-dt / tau_mem).
    gamma: float = 0.18             #: Spike threshold adaptation coefficient: increase in firing threshold per emitted spike.
    surrogate: str = "atan"         #: Surrogate derivative function for backward pass ("atan", "fast_sigmoid", "sigmoid").
    surrogate_alpha: float = 2.0    #: Sharpness / scale parameter of the surrogate gradient derivative.


@dataclass(frozen=True)
class MemoryConfig:
    """
    Multi-timescale recurrent working memory dynamics for the ALIF latent core.
    Splits the latent population into distinct pools with varying temporal retention factors.
    """
    num_timescales: int = 2                       #: Number of hierarchical time constants in the recurrent memory.
    timescale_dims: Tuple[int, ...] = (128, 128)  #: State vector dimensionality allocated to each timescale.
    betas: Tuple[float, ...] = (0.90, 0.985)      #: Threshold adaptation decay factors for each timescale (reactive vs deep context).

    def __post_init__(self) -> None:
        if len(self.timescale_dims) != self.num_timescales:
            raise ValueError(f"timescale_dims length ({len(self.timescale_dims)}) != num_timescales ({self.num_timescales})")
        if len(self.betas) != self.num_timescales:
            raise ValueError(f"betas length ({len(self.betas)}) != num_timescales ({self.num_timescales})")


@dataclass(frozen=True)
class TrainingConfig:
    """
    Training regime, optimization hyperparameters, phases, and unroll horizons.
    Two-Phase Action-Conditioned Attractor Architecture:
      1. 'train_representation' (Phase 0 - Grounded Articulated State Estimator):
         Jointly trains EventEncoder, SpikingLatentDynamics, SensoryPredictor, and PhysicalDecoder
         over sensory streams to decode complex joint angles q* and angular velocities p*.
      2. 'train_m1' (Phase 1 - Action-Conditioned Latent Attractor Predictor):
         Freezes representation modules and trains LatentPredictor (M1) autoregressively
         conditioned on motor action sequences u_{0:H}.
    """
    mode: str = "train_m1"                           #: Active training stage: "train_representation" or "train_m1".
    epochs: int = 100                                #: Total number of training epochs to execute for the active phase.
    early_stopping_patience: int = 10                #: Epochs without validation improvement before stopping.
    batch_size: int = 32                             #: Mini-batch size per optimization step.
    weight_decay: float = 0.0005                     #: Base L2 weight decay penalty applied to Adam optimizer across active submodules.
    grad_clip_norm: float = 0.8                      #: Maximum L2 gradient norm threshold for torch.nn.utils.clip_grad_norm_.
    learning_rate: float = 0.0001                    #: Base Adam learning rate for model weights.
    probe_lr: float = 0.0002                         #: Dedicated learning rate for downstream kinematic readout probes.
    probe_weight_decay: float = 0.0001               #: L2 regularization penalty applied to kinematic probes.
    temporal_burn_in: int = 15                       #: Initial latent steps for recurrent dynamics warm-up before launching evaluation rollouts.
    representation_checkpoint: Optional[str] = None  #: Optional path to pre-trained representation checkpoint .pt.
    phase_horizon: int = 100                          #: Duration H of predictor unrolling during training.
    tier2_frequency: int = 5                         #: Frequency (in epochs, 5..10) to evaluate Tier 2 closed-loop MPPI scoring.
    tier2_trajectories: int = 6                       #: Number of fixed validation trajectories for Tier 2 evaluation (covering Low, Mid, High).
    tier2_seed: int = 42000                           #: Deterministic base RNG seed for MPPI noise generation in Tier 2 validation.
    tier2_num_samples: int = 1024                     #: Number of candidate trajectory samples K in MPPI planning during Tier 2 validation.
    tier2_horizon: int = 10                           #: Planning horizon H in MPPI planning during Tier 2 validation.
    tier2_iterations: int = 3                         #: Number of MPPI optimization iterations per control step during Tier 2 validation.
    tier2_candidate_selection: str = "ema"            #: Candidate model to evaluate in Tier 2: "ema" (default: evaluate exponential moving average weights) or "latest".
    tier2_patience: int = 5                           #: Consecutive Tier 2 evaluation checks without MPPI score improvement before early stopping.
    tier2_min_delta: float = 0.001                    #: Minimum improvement threshold on MPPI cost score to reset early stopping patience.
    ema_decay: float = 0.990                          #: Decay factor tau for Exponential Moving Average (EMA) of model weights.
    mppi_noise_warmup_epochs: int = 10                #: Initial training epochs with zero AR(1) action noise before curriculum ramp.
    mppi_noise_ramp_epochs: int = 20                  #: Number of epochs over which AR(1) action noise linearly ramps to target alpha.
    mppi_noise_target_alpha: float = 0.4              #: Maximum target intensity alpha for AR(1) action noise injection.
    max_eval_generations: int = 3                     #: Rolling window of Tier 2 evaluation generations to preserve in ClosedLoopBuffer.

    def __post_init__(self) -> None:
        valid_modes = ("train_representation", "train_m1")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid training mode '{self.mode}'. Allowed modes: {valid_modes}")
        if self.phase_horizon <= 0:
            raise ValueError(f"phase_horizon must be > 0, got {self.phase_horizon}")
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be > 0, got {self.early_stopping_patience}")
        if self.tier2_frequency <= 0:
            raise ValueError(f"tier2_frequency must be > 0, got {self.tier2_frequency}")
        if self.tier2_trajectories <= 0:
            raise ValueError(f"tier2_trajectories must be > 0, got {self.tier2_trajectories}")
        if self.tier2_patience <= 0:
            raise ValueError(f"tier2_patience must be > 0, got {self.tier2_patience}")
        if self.tier2_min_delta < 0.0:
            raise ValueError(f"tier2_min_delta must be non-negative, got {self.tier2_min_delta}")
        valid_selections = ("ema", "latest")
        if self.tier2_candidate_selection not in valid_selections:
            raise ValueError(
                f"Invalid tier2_candidate_selection '{self.tier2_candidate_selection}'. Allowed: {valid_selections}"
            )
        if not (0.0 <= self.ema_decay <= 1.0):
            raise ValueError(f"ema_decay must be in [0.0, 1.0], got {self.ema_decay}")
        if self.mppi_noise_warmup_epochs < 0:
            raise ValueError(f"mppi_noise_warmup_epochs must be >= 0, got {self.mppi_noise_warmup_epochs}")
        if self.mppi_noise_ramp_epochs < 0:
            raise ValueError(f"mppi_noise_ramp_epochs must be >= 0, got {self.mppi_noise_ramp_epochs}")
        if self.mppi_noise_target_alpha < 0.0:
            raise ValueError(f"mppi_noise_target_alpha must be non-negative, got {self.mppi_noise_target_alpha}")
        if self.max_eval_generations <= 0:
            raise ValueError(f"max_eval_generations must be > 0, got {self.max_eval_generations}")


@dataclass(frozen=True)
class LossConfig:
    """
    Loss function scaling weights, horizons, and balancing coefficients for SPWM.
    Coordinates the objective across Phase 0 and Phase 1 (M1):
      - Coordinate (q) subspace error weight: lambda_position
      - Momentum/velocity (p) subspace error weight: lambda_velocity
      - Overall latent trajectory scale: lambda_latent
      - Phase 0 Grounded Representation: lambda_kinematic, lambda_derivative, lambda_smoothness, lambda_sensory
    """
    lambda_encoder: float = 1.0     #: Weight of SpatialSoftmax keypoint MSE loss.
    lambda_decoder: float = 1.0     #: Weight of PhysicalDecoder probe MSE loss.
    lambda_latent: float = 1.0      #: Weight of overall latent trajectory MSE loss.
    lambda_sensory: float = 1.0     #: Weight of top-down sensory prediction MSE loss.
    lambda_position: float = 1.0    #: Weight of coordinate/position (q) subspace MSE in latent rollout losses.
    lambda_velocity: float = 1.5    #: Weight of momentum/velocity (p) subspace MSE in latent rollout losses.
    lambda_kinematic: float = 5.0   #: Weight of physical decoder joint kinematic MSE loss in Phase 0.
    lambda_derivative: float = 1.0  #: Weight of strict phase-space derivative constraint (q_{t+1}-q_t - dt*p_t) MSE.
    lambda_smoothness: float = 0.1  #: Weight of ALIF momentum smoothness loss in Phase 0.
    lambda_critic: float = 0.2      #: Weight of terminal cost-to-go LatentCritic MC MSE loss in Phase 1 (train_m1).
    velocity_loss_scale: float = 1.0 #: Normalization scale for angular velocities (rad/s) in kinematic loss (L_vel / s^2).


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Evaluation protocol and benchmarking horizons.
    Governs test-time unrolled rollout evaluations across multiple temporal horizons H.
    """
    rollout_horizons: Tuple[int, ...] = (1, 5, 10, 25, 50, 100)  #: Unrolled rollout evaluation horizons H evaluated on test set.


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Complete typed configuration for an SPWM experiment.
    Serves as the Single Source of Truth: immutable, validated across domain boundaries,
    and serializable to/from YAML.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    neuron: LatentNeuronConfig = field(default_factory=LatentNeuronConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        assert_phase_space_contract(
            latent_dim=self.model.latent_dim,
            q_dim=self.model.q_dim,
            p_dim=self.model.p_dim,
            num_keypoints=self.model.num_keypoints,
            action_dim=self.environment.action_dim,
        )
        H = self.training.phase_horizon
        seq_len = self.data.sequence_length
        mode = self.training.mode

        if mode == "train_representation":
            min_len = 2
            if seq_len < min_len:
                raise ValueError(
                    f"DataConfig.sequence_length ({seq_len}) is too short for {mode} "
                    f"(requires >= {min_len})."
                )
        elif mode == "train_m1":
            min_len = H + 1
            if seq_len < min_len:
                raise ValueError(
                    f"DataConfig.sequence_length ({seq_len}) is too short for train_m1 "
                    f"with phase_horizon={H} (requires >= {min_len})."
                )

    def with_overrides(
        self,
        *,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        mode: Optional[str] = None,
        epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        representation_checkpoint: Optional[str] = None,
        phase_horizon: Optional[int] = None,
        temporal_burn_in: Optional[int] = None,
        sequence_length: Optional[int] = None,
        total_trajectories: Optional[int] = None,
        weighted_sampling: Optional[bool] = None,
        tier2_frequency: Optional[int] = None,
        tier2_trajectories: Optional[int] = None,
        tier2_seed: Optional[int] = None,
        tier2_num_samples: Optional[int] = None,
        learning_rate: Optional[float] = None,
        tier2_candidate_selection: Optional[str] = None,
        tier2_patience: Optional[int] = None,
        tier2_min_delta: Optional[float] = None,
        ema_decay: Optional[float] = None,
        mppi_noise_warmup_epochs: Optional[int] = None,
        mppi_noise_ramp_epochs: Optional[int] = None,
        mppi_noise_target_alpha: Optional[float] = None,
        max_eval_generations: Optional[int] = None,
    ) -> "ExperimentConfig":
        """
        Creates a new validated, frozen ExperimentConfig with explicit overrides.
        Re-runs all dataclass and cross-domain __post_init__ checks.
        """
        proj_updates: Dict[str, Any] = {}
        if name is not None:
            proj_updates["name"] = name
        if seed is not None:
            proj_updates["seed"] = seed
        proj = dataclasses.replace(self.project, **proj_updates) if proj_updates else self.project

        data_updates: Dict[str, Any] = {}
        if sequence_length is not None:
            data_updates["sequence_length"] = sequence_length
        if total_trajectories is not None:
            data_updates["total_trajectories"] = total_trajectories
        if weighted_sampling is not None:
            data_updates["weighted_sampling"] = weighted_sampling
        data = dataclasses.replace(self.data, **data_updates) if data_updates else self.data

        train_updates: Dict[str, Any] = {}
        if mode is not None:
            train_updates["mode"] = mode
        if epochs is not None:
            train_updates["epochs"] = epochs
        if early_stopping_patience is not None:
            train_updates["early_stopping_patience"] = early_stopping_patience
        if representation_checkpoint is not None:
            train_updates["representation_checkpoint"] = representation_checkpoint
        if phase_horizon is not None:
            train_updates["phase_horizon"] = phase_horizon
        if temporal_burn_in is not None:
            train_updates["temporal_burn_in"] = temporal_burn_in
        if tier2_frequency is not None:
            train_updates["tier2_frequency"] = tier2_frequency
        if tier2_trajectories is not None:
            train_updates["tier2_trajectories"] = tier2_trajectories
        if tier2_seed is not None:
            train_updates["tier2_seed"] = tier2_seed
        if tier2_num_samples is not None:
            train_updates["tier2_num_samples"] = tier2_num_samples
        if learning_rate is not None:
            train_updates["learning_rate"] = learning_rate
        if tier2_candidate_selection is not None:
            train_updates["tier2_candidate_selection"] = tier2_candidate_selection
        if tier2_patience is not None:
            train_updates["tier2_patience"] = tier2_patience
        if tier2_min_delta is not None:
            train_updates["tier2_min_delta"] = tier2_min_delta
        if ema_decay is not None:
            train_updates["ema_decay"] = ema_decay
        if mppi_noise_warmup_epochs is not None:
            train_updates["mppi_noise_warmup_epochs"] = mppi_noise_warmup_epochs
        if mppi_noise_ramp_epochs is not None:
            train_updates["mppi_noise_ramp_epochs"] = mppi_noise_ramp_epochs
        if mppi_noise_target_alpha is not None:
            train_updates["mppi_noise_target_alpha"] = mppi_noise_target_alpha
        if max_eval_generations is not None:
            train_updates["max_eval_generations"] = max_eval_generations

        training = dataclasses.replace(self.training, **train_updates) if train_updates else self.training
        return dataclasses.replace(self, project=proj, data=data, training=training)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return _strict_dict_to_dataclass(cls, data, context="ExperimentConfig")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        class _SafeLoaderWithTuple(yaml.SafeLoader):
            pass

        _SafeLoaderWithTuple.add_constructor(
            "tag:yaml.org,2002:python/tuple",
            lambda loader, node: tuple(loader.construct_sequence(node)),
        )

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.load(f, Loader=_SafeLoaderWithTuple)
        if not isinstance(raw, dict):
            raise ValueError(f"YAML content at {path} must be a dictionary, got {type(raw).__name__}")
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
