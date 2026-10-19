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
    total_trajectories: int = 4000           #: Total number of synthetic trajectories generated in the dataset.
    train_split: float = 0.8                #: Fraction of trajectories dedicated to training (must satisfy train+val < 1.0).
    val_split: float = 0.1                  #: Fraction of trajectories dedicated to validation (remaining is test).
    sequence_length: int = 31              #: Number of consecutive temporal steps per generated trajectory.
    batch_size: int = 32                    #: Dataloader batch size during training and validation.
    drift_injection_prob: float = 0.20      #: Probability per timestep of injecting an unannounced velocity drift/perturbation.
    drift_magnitude: float = 0.05           #: Instantaneous impulse magnitude added to velocity vectors upon drift injection.
    mixed_drift: bool = True               #: Whether to combine discrete velocity shocks with continuous Brownian random walk drift.
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
    Slow Corrector hyperparameters and dynamics.
    Supports 'gru' (control experiment), 'mlp' (residual MLP control), and 'alif' (neuromorphic gain field).
    """
    enabled: bool = True       #: Master switch enabling the corrector network.
    type: str = "gru"          #: Corrector architecture type: 'gru', 'mlp', or 'alif'.
    dim: int = 512             #: Hidden dimension of the corrector network.
    beta_mem: float = 0.995    #: Membrane decay factor for ALIF neurons.
    beta_adapt: float = 0.98   #: Threshold adaptation decay factor for ALIF neurons.
    v_th0: float = 0.50        #: Baseline firing threshold for ALIF neurons.
    gamma: float = 0.08        #: Adaptation strength for ALIF neurons.
    alpha_gain: float = 0.4    #: Maximum gain modulation authority scaling.


@dataclass
class TrainingConfig:
    """
    Training regime, optimization hyperparameters, phases, and unroll horizons.
    Supports decoupled three-stage curriculum training:
      1. 'train_representation': Train SpatialSoftmax encoder and physical decoder probe;
         ALIF recurrent predictor / dynamics is frozen.
      2. 'train_predictor': Train ALIF recurrent predictor / dynamics on multi-step rollouts;
         SpatialSoftmax encoder and physical decoder are frozen.
      3. 'train_corrector': Train neuromodulatory gain-field corrector on drift-perturbed sequences;
         predictor and encoder are frozen.

    Train & Validation Loss Formulas:
    ---------------------------------
    - 'train_representation':
        Train Loss:
            train_loss = lambda_encoder * train_loss_encoder + lambda_decoder * train_loss_decoder
            where:
                train_loss_encoder = MSE(keypoints, target_positions)
                train_loss_decoder = MSE(decoded_kinematics, target_kinematics)
        Val Loss:
            val_loss = lambda_encoder * val_loss_encoder + lambda_decoder * val_loss_decoder
        Model Selection:
            min(val_loss) -> model.pt

    - 'train_predictor':
        Train Loss (evaluated over rollout horizon H = multi_step_horizon):
            train_loss = lambda_position * train_loss_position + lambda_velocity * train_loss_velocity
            where:
                train_loss_position = MSE(rollout_positions[:, :H], target_positions[:, :H])
                train_loss_velocity = MSE(rollout_velocities[:, :H], target_velocities[:, :H])
        Val Loss:
            val_loss = lambda_position * val_loss_position + lambda_velocity * val_loss_velocity
        Model Selection:
            min(val_loss) -> model.pt

    - 'train_corrector':
        Train Loss (evaluated over horizon H = corrector_horizon with Gaussian comb temporal weighting w_t):
            train_loss = train_kin_loss = train_pos_loss + beta_v * train_vel_loss
            where:
                w_t = normalized Gaussian comb weights centered at multiples of 25 timesteps:
                      w_t ∝ exp(-0.5 * (min(t % 25, 25 - t % 25) / comb_sigma)^2)
                train_pos_loss = mean(w_t * (rollout_positions - target_positions)^2)
                train_vel_loss = mean(w_t * (rollout_velocities - target_velocities)^2)
        Val Loss:
            val_loss = val_kin_loss (comb-weighted kinematic error over active rollout horizon)
        Model Selection:
            min(val_loss) -> model.pt
    """
    # -------------------------------------------------------------------------
    # 1. General Training Parameters (Shared Across Phases)
    # -------------------------------------------------------------------------
    mode: str = "train_predictor"                  #: Active training stage: "train_representation", "train_predictor", "train_joint", or "train_corrector".
    epochs: int = 100                              #: Total number of training epochs to execute for the active phase.
    batch_size: int = 32                           #: Mini-batch size per optimization step.
    weight_decay: float = 0.0005                   #: Base L2 weight decay penalty applied to Adam optimizer across active submodules.
    grad_clip_norm: float = 0.8                    #: Maximum L2 gradient norm threshold for torch.nn.utils.clip_grad_norm_.

    # -------------------------------------------------------------------------
    # 2. Predictor & Representation Training Parameters ('train_predictor' / 'train_representation' / 'train_joint')
    # -------------------------------------------------------------------------
    learning_rate: float = 0.0002                  #: Base Adam learning rate for model weights (encoder/decoder in representation, predictor in predictor/joint phase).
    probe_lr: float = 0.0005                       #: Dedicated learning rate for downstream kinematic readout linear probes (when separately parameterized).
    probe_weight_decay: float = 0.01               #: L2 regularization penalty applied to linear probes.
    temporal_burn_in: int = 5                      #: Initial latent steps for recurrent dynamics warm-up before launching predictor rollout in train_joint / train_predictor.
    representation_checkpoint: Optional[str] = None  #: Path to pre-trained representation checkpoint .pt for 'train_predictor' mode. If None, auto-checks default location.

    # -------------------------------------------------------------------------
    # 3. Corrector-Only Training Parameters ('train_corrector')
    # -------------------------------------------------------------------------
    corrector_lr: float = 0.0004                   #: Dedicated learning rate for neuromodulatory gain-field corrector parameters in 'train_corrector'.
    corrector_horizon: int = 25                    #: Temporal unroll horizon H over which corrector rollout comb loss is computed.
    corrector_burn_in_steps: int = 5               #: Initial unroll steps where corrector integrates history but gain modulation is bypassed.
    perturb_corrector_states: bool = True          #: Whether to inject Gaussian noise into corrector inputs to regularize robustness under drift.
    corrector_perturbation_std: float = 0.08       #: Standard deviation sigma of Gaussian perturbation injected into corrector states.
    predictor_checkpoint: Optional[str] = None     #: Path to pre-trained predictor checkpoint .pt required when running in 'train_corrector' mode.

    # Deprecated / obsolete fields:
    # encoder_warmup_epochs: int = 60                #: [Deprecated - Legacy warmup epochs for encoder before locking dynamics in legacy joint trainer, superseded by decoupled phases]
    # learning_algorithm: str = "online_eprop"       #: [Deprecated - Last used in v4.3/v5, superseded by modular phase handlers & fixed online eprop]
    # curriculum_multi_step: bool = False            #: [Deprecated - Last used in v7.3 curriculum experiments, removed in v8+ decoupled training]
    # smooth_horizon_sampling: bool = False          #: [Deprecated - Last used in v7.3, rejected due to training instability]
    # k_max: int = 50                                #: [Deprecated - Last used in v7.3 curriculum horizon bounds]

    def __post_init__(self):
        valid_modes = ("train_representation", "train_predictor", "train_joint", "train_corrector")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid training mode '{self.mode}'. Allowed modes: {valid_modes}")


@dataclass
class LossConfig:
    """
    Loss function scaling weights, horizons, and balancing coefficients for SPWM.
    Governs loss_encoder, loss_decoder, loss_position, loss_velocity, and corrector comb loss.

    Weight Scopes & Formulations:
    -----------------------------
    1. Scope: 'train_representation'
       - lambda_encoder: Multiplier on SpatialSoftmax keypoint distribution MSE loss:
           L_encoder = MSE(keypoints, target_positions)
       - lambda_decoder: Multiplier on kinematic probe reconstruction MSE loss from immediate latent:
           L_decoder = MSE(decoded_kinematics, target_kinematics)
       Total Train/Val Loss:
           L_total = lambda_encoder * L_encoder + lambda_decoder * L_decoder

    2. Scope: 'train_predictor'
       - lambda_position: Multiplier on position MSE error across multi-step autonomous rollout:
           L_position = MSE(rollout_positions[:, :H], target_positions[:, :H])
       - lambda_velocity: Multiplier on velocity MSE error across multi-step autonomous rollout:
           L_velocity = MSE(rollout_velocities[:, :H], target_velocities[:, :H])
       - multi_step_horizon (H): Number of future autonomous rollout steps used for computing L_position and L_velocity.
       Total Train/Val Loss:
           L_total = lambda_position * L_position + lambda_velocity * L_velocity

    3. Scope: 'train_corrector'
       - beta_v: Relative weighting coefficient for velocity error in the comb rollout loss:
           L_kin = L_pos + beta_v * L_vel
       - comb_sigma: Gaussian kernel bandwidth for periodic comb temporal weighting centered at multiples of 25 timesteps:
           w_t ∝ exp(-0.5 * (min(t % 25, 25 - t % 25) / comb_sigma)^2)
       Total Train/Val Loss:
           L_total = L_kin
    """
    lambda_encoder: float = 1.0     #: Weight of SpatialSoftmax keypoint MSE loss in 'train_representation'. Formula: L_encoder = MSE(keypoints, xy_gt).
    lambda_decoder: float = 1.0     #: Weight of physical decoder probe MSE loss in 'train_representation'. Formula: L_decoder = MSE(decoded_kin, kin_gt).
    lambda_position: float = 1.0    #: Weight of position MSE loss in 'train_predictor'. Formula: L_pos = MSE(pred_pos[:, :H], true_pos[:, :H]).
    lambda_velocity: float = 2.0    #: Weight of velocity MSE loss in 'train_predictor'. Formula: L_vel = MSE(pred_vel[:, :H], true_vel[:, :H]).
    multi_step_horizon: int = 5     #: Multi-step autonomous rollout horizon H in 'train_predictor'.
    beta_v: float = 2.0             #: Velocity loss scaling factor in 'train_corrector' comb loss. Formula: L_kin = L_pos + beta_v * L_vel.
    comb_sigma: float = 3.5         #: Gaussian bandwidth parameter for smooth comb temporal weighting in 'train_corrector'.



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
        import dataclasses
        return dataclasses.asdict(self)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

