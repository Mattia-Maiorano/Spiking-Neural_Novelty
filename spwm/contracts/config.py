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
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union, get_type_hints
import yaml

from spwm.contracts.tensors import assert_phase_space_contract

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
                origin = getattr(f_type, "__origin__", None)
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
    name: str = "spwm_v0_4_0"  #: Unique identifier/tag for the project and run artifacts.
    seed: int = 42             #: Global pseudo-random seed for PyTorch, NumPy, and trajectory generation.


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    3D Unity Articulated Robotics environment and neuromorphic event camera sensor configuration.
    Controls the articulated kinematic chain DOF (joint_count), motor command dimension (action_dim),
    discrete frame timesteps, and event sensor log-intensity spike generation thresholds.
    """
    joint_count: int = 8           #: Number of articulated joints J in the robotic kinematic chain.
    action_dim: int = 8            #: Dimensionality D_u of continuous motor action vectors u_t.
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
    joint_count: int = 8                            #: Number of articulated joints J expected in dataset trajectories.
    action_dim: int = 8                             #: Dimensionality D_u of motor action vectors in dataset trajectories.
    total_trajectories: int = 4000                  #: Total number of trajectories across all dataset splits.
    train_split: float = 0.8                        #: Fraction of trajectories dedicated to training (must satisfy train+val < 1.0).
    val_split: float = 0.1                          #: Fraction of trajectories dedicated to validation (remaining is test).
    sequence_length: int = 105                      #: Number of consecutive temporal steps per trajectory.
    batch_size: int = 32                            #: Dataloader batch size during training and validation.
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0)       #: Nominal joint angular velocity scaling range.
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0)  #: Extrapolation joint angular velocity range.
    num_workers: int = 0                            #: Number of subprocesses used by PyTorch DataLoaders.
    cache_data: bool = True                         #: Whether to cache rendered event frames in RAM to accelerate training epochs.

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
    decoder_ema_alpha: float = 0.2                    #: EMA filter smoothing coefficient for PhysicalDecoder momentum readout.
    in_channels: int = 2                              #: Event camera input polarity channels (positive and negative spikes).
    symplectic: bool = True                           #: Whether to use dissipative phase-space integration for M1.

    def __post_init__(self) -> None:
        assert_phase_space_contract(
            latent_dim=self.latent_dim,
            q_dim=self.q_dim,
            p_dim=self.p_dim,
            num_keypoints=self.num_keypoints,
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
    early_stopping_patience: int = 20                #: Epochs without validation improvement before stopping.
    batch_size: int = 32                             #: Mini-batch size per optimization step.
    weight_decay: float = 0.0005                     #: Base L2 weight decay penalty applied to Adam optimizer across active submodules.
    grad_clip_norm: float = 0.8                      #: Maximum L2 gradient norm threshold for torch.nn.utils.clip_grad_norm_.
    learning_rate: float = 0.0002                    #: Base Adam learning rate for model weights.
    probe_lr: float = 0.0002                         #: Dedicated learning rate for downstream kinematic readout probes.
    probe_weight_decay: float = 0.01                 #: L2 regularization penalty applied to kinematic probes.
    temporal_burn_in: int = 15                       #: Initial latent steps for recurrent dynamics warm-up before launching evaluation rollouts.
    representation_checkpoint: Optional[str] = None  #: Optional path to pre-trained representation checkpoint .pt.
    phase_horizon: int = 50                          #: Duration H of predictor unrolling during training.

    def __post_init__(self) -> None:
        valid_modes = ("train_representation", "train_m1")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid training mode '{self.mode}'. Allowed modes: {valid_modes}")
        if self.phase_horizon <= 0:
            raise ValueError(f"phase_horizon must be > 0, got {self.phase_horizon}")
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be > 0, got {self.early_stopping_patience}")


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
    lambda_velocity: float = 1.0    #: Weight of momentum/velocity (p) subspace MSE in latent rollout losses.
    lambda_kinematic: float = 1.0   #: Weight of physical decoder joint kinematic MSE loss in Phase 0.
    lambda_derivative: float = 1.0  #: Weight of strict phase-space derivative constraint (q_{t+1}-q_t - dt*p_t) MSE.
    lambda_smoothness: float = 1.0  #: Weight of ALIF momentum smoothness MSE loss in Phase 0.


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
        data = dataclasses.replace(self.data, sequence_length=sequence_length) if sequence_length is not None else self.data

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
