"""
Typed Configuration Schema for SPWM.
Single Source of Truth for all hyperparameter declarations, validation, and serialization.
Strict: Every non-aligned parameter surfaces and raises an error; no silent fallbacks.
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import yaml

T = TypeVar("T")


def _strict_dict_to_dataclass(cls: Type[T], data: Dict[str, Any], context: str = "") -> T:
    """Instantiate a dataclass strictly: unknown keys raise ValueError immediately."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dictionary for {context or cls.__name__}, got {type(data).__name__}: {data}")

    from typing import get_type_hints
    import sys

    # Resolve type hints in current module scope
    type_hints = get_type_hints(cls, globalns=sys.modules[__name__].__dict__)
    valid_fields = {f.name: f for f in fields(cls)}
    unknown_keys = set(data.keys()) - set(valid_fields.keys())
    if unknown_keys:
        ctx_str = f" in {context}" if context else f" in {cls.__name__}"
        raise ValueError(f"Unknown configuration parameter(s){ctx_str}: {sorted(list(unknown_keys))}. No fallbacks allowed.")

    kwargs = {}
    for f_name, f_field in valid_fields.items():
        f_type = type_hints.get(f_name, f_field.type)
        if f_name in data:
            val = data[f_name]
            # Handle nested dataclass
            if isinstance(f_type, type) and is_dataclass(f_type) and isinstance(val, dict):
                kwargs[f_name] = _strict_dict_to_dataclass(f_type, val, context=f"{context}.{f_name}" if context else f_name)
            elif val is not None:
                # Handle tuples and lists
                origin = getattr(f_type, "__origin__", None)
                if origin is tuple and isinstance(val, (list, tuple)):
                    val = tuple(val)
                elif origin is list and isinstance(val, (list, tuple)):
                    val = list(val)
                kwargs[f_name] = val
        else:
            # Check if field has default or default_factory
            if f_field.default is not MISSING:
                kwargs[f_name] = f_field.default
            elif f_field.default_factory is not MISSING:
                kwargs[f_name] = f_field.default_factory()
            else:
                ctx_str = f" for {context}.{f_name}" if context else f" for {cls.__name__}.{f_name}"
                raise ValueError(f"Missing required configuration parameter{ctx_str}.")

    return cls(**kwargs)



@dataclass
class ProjectConfig:
    """Project-level identity, tracking tags, and pseudo-random reproducibility configuration."""
    name: str = "spwm_v0_1_0"  #: Unique identifier/tag for the project and run artifacts.
    seed: int = 42             #: Global pseudo-random seed for PyTorch, NumPy, and synthetic data generation.


@dataclass
class EnvironmentConfig:
    """
    Synthetic physics environment and event camera sensor configuration.
    Controls the continuous billiard simulation arena, discrete frame timesteps,
    and event sensor log-intensity spike generation thresholds.
    """
    num_objects: int = 1        #: Number of moving billiard ball objects in the 2D bounding arena.
    height: int = 32            #: Pixel height of the synthetic event camera sensor plane.
    width: int = 32             #: Pixel width of the synthetic event camera sensor plane.
    box_size: float = 1.0       #: Half-width/box boundaries of the normalized simulation arena [-box_size, box_size].
    dt: float = 0.05            #: Simulation time step size in seconds per discrete physics frame.
    event_threshold: float = 0.08  #: Log-intensity change threshold required to trigger a polarity spike (+1/-1).


@dataclass
class DataConfig:
    """
    Dataset splitting, sequence length, batching, and kinematic drift configuration.
    Configures nominal trajectory generation as well as drift injection probabilities
    and magnitude parameters for robustness benchmarking.
    """
    total_trajectories: int = 600           #: Total number of synthetic trajectories generated in the dataset.
    train_split: float = 0.8                #: Fraction of trajectories dedicated to training (must satisfy train+val < 1.0).
    val_split: float = 0.1                  #: Fraction of trajectories dedicated to validation (remaining is test).
    sequence_length: int = 150              #: Number of consecutive temporal steps per generated trajectory.
    batch_size: int = 32                    #: Dataloader batch size during training and validation.
    drift_injection_prob: float = 0.20      #: Probability per timestep of injecting an unannounced velocity drift/perturbation.
    drift_magnitude: float = 0.05           #: Instantaneous impulse magnitude added to velocity vectors upon drift injection.
    mixed_drift: bool = False               #: Whether to combine discrete velocity shocks with continuous Brownian random walk drift.
    standard_velocity_range: Tuple[float, float] = (-1.0, 1.0)        #: Initial object velocity range (min, max) for nominal regimes.
    extrapolation_velocity_range: Tuple[float, float] = (-2.0, 2.0)   #: Velocity range for out-of-distribution extrapolation evaluation.
    num_workers: int = 0                    #: Number of subprocesses used by PyTorch DataLoaders.
    cache_data: bool = True                 #: Whether to cache rendered event frames in RAM to accelerate training epochs.

    def __post_init__(self):
        if self.train_split + self.val_split >= 1.0:
            raise ValueError(f"train_split ({self.train_split}) + val_split ({self.val_split}) must be < 1.0")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.sequence_length}")


@dataclass
class ModelConfig:
    """
    Spiking Predictive World Model (SPWM) neural architecture dimensions and coupling parameters.
    Defines coordinate/momentum subspace partitioning (latent_dim = q_dim + p_dim),
    spatial keypoint bottlenecks, and eligibility trace feedback parameters.
    """
    type: str = "spwm"                                #: Architecture family ("spwm" or baseline variants).
    encoder_type: str = "spatial_softmax"             #: Visual front-end encoder type ("spatial_softmax" keypoints).
    num_keypoints: int = 16                           #: Number of spatial feature keypoints extracted by the encoder.
    encoder_dim: int = 128                            #: Output feature dimension of the convolutional feature extractor.
    encoder_conv_channels: Tuple[int, ...] = (16, 32) #: Channel depths for 2D convolutional encoder layers.
    latent_dim: int = 128                             #: Total latent state dimension; MUST equal q_dim + p_dim.
    q_dim: int = 32                                   #: Coordinate/position subspace dimension; MUST equal num_keypoints * 2.
    p_dim: int = 96                                   #: Momentum/velocity subspace dimension; MUST be a multiple of q_dim.
    predictor_hidden_dim: int = 256                   #: Hidden layer dimensionality of the predictor ALIF/RNN network.
    local_lr: float = 0.0002                          #: Local learning rate for online e-prop eligibility trace updates.
    lambda_kin_feedback: float = 1.0                  #: Weight of kinematic boundary error feedback coupled back into momentum p.
    ema_decay: float = 0.9                            #: Exponential moving average decay for tracking latent state statistics.
    in_channels: int = 2                              #: Event camera input polarity channels (positive and negative spikes).

    def __post_init__(self):
        if self.q_dim + self.p_dim != self.latent_dim:
            raise ValueError(f"q_dim ({self.q_dim}) + p_dim ({self.p_dim}) != latent_dim ({self.latent_dim})")
        if self.num_keypoints * 2 != self.q_dim:
            raise ValueError(f"num_keypoints ({self.num_keypoints}) * 2 must equal q_dim ({self.q_dim})")
        if self.p_dim % self.q_dim != 0:
            raise ValueError(f"p_dim ({self.p_dim}) must be an integer multiple of q_dim ({self.q_dim}) for spatial boundary coupling.")


@dataclass
class LatentNeuronConfig:
    """
    Biophysical neuron hyperparameters for the recurrent Spiking Latent Dynamics core (z_t).
    Controls the Adaptive Leaky Integrate-and-Fire (ALIF) membrane decay, dynamic threshold adaptation,
    and pseudo-derivative surrogate gradient shape for non-differentiable spike events.
    """
    model: str = "alif"             #: Spiking neuron model ("alif" or standard "lif").
    threshold: float = 1.0          #: Baseline membrane potential firing threshold v_th.
    beta_mem: float = 0.80          #: Membrane potential leak decay factor: exp(-dt / tau_mem).
    gamma: float = 0.18             #: Spike threshold adaptation coefficient: increase in firing threshold per emitted spike.
    surrogate: str = "atan"         #: Surrogate derivative function for backward pass ("atan", "sigmoid", etc.).
    surrogate_alpha: float = 2.0    #: Sharpness / scale parameter of the surrogate gradient derivative.


@dataclass
class MemoryConfig:
    """
    Multi-timescale eligibility trace and recurrent working memory dynamics for the latent core.
    Splits the latent state into distinct populations with varying temporal retention factors
    (e.g., reactive short-term integration vs. persistent context accumulation).
    """
    num_timescales: int = 2                       #: Number of hierarchical time constants in the recurrent memory.
    timescale_dims: Tuple[int, ...] = (128, 128)  #: State vector dimensionality allocated to each timescale.
    betas: Tuple[float, ...] = (0.90, 0.985)      #: Leak decay factors for each timescale (fast decay, slow persistent memory).

    def __post_init__(self):
        if len(self.timescale_dims) != self.num_timescales:
            raise ValueError(f"timescale_dims length ({len(self.timescale_dims)}) != num_timescales ({self.num_timescales})")
        if len(self.betas) != self.num_timescales:
            raise ValueError(f"betas length ({len(self.betas)}) != num_timescales ({self.num_timescales})")


@dataclass
class CorrectorConfig:
    """
    Slow Neuromodulatory Gain Field Corrector hyperparameters and adaptation dynamics.
    Configures the macro module parameters (authority alpha_gain, population dimension)
    along with its internal slow ALIF biophysics (beta_mem, beta_adapt, v_th0, gamma)
    used to emit bounded multiplicative gain masks in [-1, 1].
    """
    enabled: bool = True       #: Master switch enabling the slow neuromodulatory gain field corrector network.
    dim: int = 128             #: Hidden dimension of the ALIF corrector population.
    beta_mem: float = 0.20     #: Membrane decay factor for corrector neurons (fast responsive integration).
    beta_adapt: float = 0.05   #: Threshold adaptation decay factor for slow accumulation of persistent drift evidence.
    v_th0: float = 0.50        #: Baseline firing threshold for corrector ALIF neurons.
    gamma: float = 0.08        #: Adaptation strength incrementing threshold upon spike emission.
    alpha_gain: float = 1.0    #: Maximum gain modulation authority scaling Δz_fast: Δz = Δz_fast * (1 + alpha_gain * mask).


@dataclass
class TrainingConfig:
    """
    Training regime, optimization hyperparameters, phases, and unroll horizons.
    Supports decoupled two-phase training ("train_predictor" for nominal fast physics,
    and "train_corrector" for the slow gain field modulator under drift perturbations).
    """
    mode: str = "train_predictor"                  #: Training stage: "train_predictor" or "train_corrector".
    epochs: int = 100                              #: Total number of training epochs to execute.
    learning_rate: float = 0.0002                  #: Base AdamW learning rate for the main model weights.
    probe_lr: float = 0.0005                       #: Learning rate for downstream kinematic readout linear probes.
    probe_weight_decay: float = 0.01               #: L2 regularization penalty applied to linear probes.
    weight_decay: float = 0.0005                   #: L2 regularization penalty applied to main model parameters.
    batch_size: int = 32                           #: Mini-batch size per optimization step.
    grad_clip_norm: float = 0.8                    #: Maximum L2 gradient norm threshold for gradient clipping.
    encoder_warmup_epochs: int = 60                #: Epochs during which spatial encoder is warmed up before locking dynamics.
    corrector_lr: float = 0.0005                   #: Learning rate specifically for the corrector module in Stage 2.
    corrector_horizon: int = 25                    #: Temporal unroll horizon H over which asymptotic rollout loss is computed.
    corrector_burn_in_steps: int = 0               #: Steps at start of rollout where corrector integrates state but gain is bypassed.
    perturb_corrector_states: bool = False          #: Whether to inject noise into corrector inputs to regularize robustness.
    corrector_perturbation_std: float = 0.08       #: Standard deviation of Gaussian perturbation injected into corrector states.
    predictor_checkpoint: Optional[str] = None     #: Path to pre-trained predictor checkpoint .pt loaded when in train_corrector mode.

    # Deprecated / obsolete fields:
    # learning_algorithm: str = "online_eprop"       #: [Deprecated - Last used in v4.3/v5, superseded by modular phase handlers & fixed online eprop]
    # curriculum_multi_step: bool = False            #: [Deprecated - Last used in v7.3 curriculum experiments, removed in v8+ decoupled training]
    # smooth_horizon_sampling: bool = False          #: [Deprecated - Last used in v7.3, rejected due to training instability]
    # k_max: int = 50                                #: [Deprecated - Last used in v7.3 curriculum horizon bounds]

    def __post_init__(self):
        valid_modes = ("train_predictor", "train_corrector")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid training mode '{self.mode}'. Allowed modes: {valid_modes}")


@dataclass
class LossConfig:
    """
    Multi-objective loss function balance weights, normalization scales, and regularizers.
    Balances 1-step prediction error, multi-step autoregressive rollout, kinematic probe tracking,
    metabolic spike sparsity, variance preservation, and asymptotic corrector rollout error.
    """
    lambda_pred: float = 1.0                  #: Weight of next-step latent state prediction error loss.
    lambda_multi: float = 0.5                 #: Weight of multi-step rollout autoregressive prediction loss.
    lambda_var: float = 0.1                   #: Weight of variance regularization loss (preventing latent representation collapse).
    lambda_sparse: float = 0.5                #: Weight of metabolic spike sparsity penalty.
    lambda_vel: float = 0.5                   #: Weight of velocity consistency loss between q derivative and momentum p.
    lambda_probe: float = 2.0                 #: Weight of supervised kinematic probe loss (position and velocity).
    lambda_coord: float = 0.15                #: Weight of coordinate preservation loss on keypoint encoder representations.
    beta_v: float = 2.0                       #: Weight balancing velocity error relative to position error in kinematic losses.
    sigma_q: float = 1.0                      #: Normalization scaling standard deviation for coordinate positions.
    sigma_v: float = 1.0                      #: Normalization scaling standard deviation for velocities.
    use_empirical_variance: bool = False      #: Whether to estimate normalization scales empirically from the training batch.
    multi_step_horizon: int = 3               #: Default rollout horizon for multi-step loss in predictor training.
    target_variance: float = 1.0              #: Target variance for latent dimension variance regularization.
    target_spike_rate: float = 0.10           #: Desired biological target spike rate per neuron (e.g. 10% active spikes).
    # lambda_corrector_asymptotic: float = 1.0  #: [Deprecated - Redundant 1.0 multiplier on corrector rollout loss]
    # curriculum_thresholds: Optional[Dict[str, Any]] = None  #: [Deprecated - Last used in v7.3 curriculum progression, replaced by fixed horizons]


@dataclass
class EvaluationConfig:
    """
    Evaluation protocol and benchmarking horizons.
    Governs test-time unrolled rollout evaluations across multiple temporal horizons H.
    """
    rollout_horizons: Tuple[int, ...] = (1, 5, 10, 25, 50, 100) #: Unrolled rollout evaluation horizons H evaluated on test set.
    # max_drift_ratio: float = 2.0                                #: [Deprecated - Dead residual gate metric, unused in evaluation pipeline]


@dataclass
class ExperimentConfig:
    """
    Complete typed configuration for an SPWM experiment.
    Serves as the Single Source of Truth: instantiable directly in Python code
    with documented fields, serializable to/from YAML for run persistence and versioning.
    """
    project: ProjectConfig = field(default_factory=ProjectConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    neuron: LatentNeuronConfig = field(default_factory=LatentNeuronConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return _strict_dict_to_dataclass(cls, data, context="ExperimentConfig")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"YAML content at {path} must be a dictionary, got {type(raw).__name__}")
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

