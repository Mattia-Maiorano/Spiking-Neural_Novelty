"""
SPWM: Spiking Predictive World Model with Neuromodulatory Gain Field Corrector.
Composes core primitives (Encoder, Dynamics, Predictor, PhysicalDecoder, Corrector)
into a unified world model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from spwm.config.schema import ExperimentConfig
from spwm.core.encoder import EventEncoder
from spwm.core.dynamics import SpikingLatentDynamics, DynamicsState, DynamicsOutput
from spwm.core.predictor import LatentPredictor, PhysicalDecoder, PredictorOutput
from spwm.core.corrector import SlowCorrector, SlowCorrectorState, SlowCorrectorOutput
from spwm.models.rollout import rollout_latent, RolloutResult


@dataclass
class SPWMState:
    """Complete recurrent state of SPWM model."""
    encoder_states: Optional[Any]
    dynamics_state: DynamicsState


@dataclass
class SPWMStepOutput:
    """Output from a single step execution in SPWM."""
    latent: torch.Tensor                        # [B, latent_dim] (z_t)
    predicted_next_latent: torch.Tensor         # [B, latent_dim] (z_hat_(t+1))
    error_neurons: torch.Tensor                 # [B, encoder_dim] (epsilon_t = x_t - W_pred z_(t-1))
    prediction_error: Optional[torch.Tensor]    # [B] if target latent is available
    decoded_kinematics: Optional[torch.Tensor]  # [B, 4 * N]
    ema_spikes: Optional[torch.Tensor] = None   # [B, total_memory_dim]


@dataclass
class SPWMSequenceOutput:
    """Output from a full sequence forward pass in SPWM."""
    latent_states: torch.Tensor          # [B, T, latent_dim]
    predicted_latents: torch.Tensor      # [B, T, latent_dim]
    prediction_errors: torch.Tensor      # [B, T-1]
    error_neurons: torch.Tensor          # [B, T, encoder_dim]
    fast_spikes: torch.Tensor            # [B, T, dim_fast]
    slow_spikes: torch.Tensor            # [B, T, dim_slow]
    encoder_spike_rate: torch.Tensor
    dynamics_spike_rate: torch.Tensor
    mean_spike_rate: torch.Tensor
    decoded_kinematics: Optional[torch.Tensor] = None
    corrector_spikes: Optional[torch.Tensor] = None


class SPWM(nn.Module):
    """
    SPWM: Spiking Predictive World Model.
    Unified composition of spatial bottleneck encoder, ALIF latent dynamics,
    fast transition predictor, kinematic probe, and slow gain-field corrector.
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        encoder_conv_channels: Tuple[int, int] = (16, 32),
        encoder_dim: int = 128,
        latent_dim: int = 128,
        timescale_dims: Optional[Tuple[int, ...]] = None,
        betas: Tuple[float, ...] = (0.90, 0.985),
        beta_mem: float = 0.80,
        threshold: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        num_keypoints: Optional[int] = None,
        predictor_hidden_dim: int = 256,
        num_objects: int = 1,
        local_lr: float = 1e-3,
        lambda_kin_feedback: float = 0.0,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        ema_decay: float = 0.9,
        enable_corrector: bool = True,
        corrector_dim: int = 128,
        corrector_beta_mem: float = 0.20,
        corrector_beta_adapt: float = 0.05,
        corrector_v_th0: float = 0.50,
        corrector_gamma: float = 0.08,
        corrector_alpha_gain: float = 1.0,
    ) -> None:
        super().__init__()
        if q_dim is None and p_dim is None:
            if latent_dim >= 128:
                q_dim = 32
                p_dim = latent_dim - 32
            else:
                q_dim = max(2, (latent_dim // 4) // 2 * 2)
                p_dim = latent_dim - q_dim
        elif q_dim is None:
            q_dim = latent_dim - p_dim
        elif p_dim is None:
            p_dim = latent_dim - q_dim

        if q_dim + p_dim != latent_dim:
            raise ValueError(f"SPWM: q_dim ({q_dim}) + p_dim ({p_dim}) != latent_dim ({latent_dim})")

        if num_keypoints is None:
            num_keypoints = max(1, q_dim // 2)
        elif num_keypoints * 2 != q_dim:
            raise ValueError(f"SPWM: num_keypoints ({num_keypoints}) * 2 != q_dim ({q_dim})")



        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.local_lr = local_lr
        self.lambda_kin_feedback = lambda_kin_feedback
        self.num_objects = num_objects

        # 1. Spatial Softmax Keypoint Event Encoder
        self.encoder = EventEncoder(
            in_channels=in_channels,
            conv_channels=encoder_conv_channels,
            num_keypoints=num_keypoints,
            out_dim=encoder_dim,
        )

        # 2. Predictive Sensory Decoder (top-down predictive coding)
        self.sensory_predictor = nn.Linear(latent_dim, encoder_dim, bias=False)

        # 3. Recurrent Spiking Latent Dynamics with ALIF Memory Core
        self.dynamics = SpikingLatentDynamics(
            input_dim=encoder_dim,
            latent_dim=latent_dim,
            q_dim=q_dim,
            p_dim=p_dim,
            ema_decay=ema_decay,
            timescale_dims=timescale_dims,
            betas=betas,
            beta_mem=beta_mem,
            v_th0=threshold,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # 4. Latent Predictor (Fast Nominal Physics)
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            residual=True,
        )

        # 5. Slow Neuromodulatory Corrector (Gain Field System)
        self.enable_corrector = enable_corrector
        if enable_corrector:
            self.corrector: Optional[SlowCorrector] = SlowCorrector(
                q_dim=q_dim,
                p_dim=p_dim,
                corrector_dim=corrector_dim,
                beta_mem=corrector_beta_mem,
                beta_adapt=corrector_beta_adapt,
                v_th0=corrector_v_th0,
                gamma=corrector_gamma,
                surrogate_name=surrogate_name,
                surrogate_alpha=surrogate_alpha,
                alpha_gain=corrector_alpha_gain,
            )
        else:
            self.corrector = None

        # 6. Physical Kinematic Probe Decoder
        self.physical_decoder = PhysicalDecoder(
            latent_dim=latent_dim,
            q_dim=q_dim,
            p_dim=p_dim,
            num_objects=num_objects,
        )

        B_feedback = torch.randn(4 * num_objects, latent_dim) / (4.0 ** 0.5)
        self.register_buffer("B_kin_feedback", B_feedback)

        self.init_buffer()

    @classmethod
    def from_config(cls, cfg: ExperimentConfig, enable_corrector: Optional[bool] = None) -> "SPWM":
        """Instantiate SPWM directly from the typed ExperimentConfig. Single source of truth."""
        corr_enabled = cfg.corrector.enabled if enable_corrector is None else enable_corrector
        return cls(
            in_channels=cfg.model.in_channels,
            height=cfg.environment.height,
            width=cfg.environment.width,
            encoder_conv_channels=cfg.model.encoder_conv_channels,
            encoder_dim=cfg.model.encoder_dim,
            latent_dim=cfg.model.latent_dim,
            timescale_dims=cfg.memory.timescale_dims,
            betas=cfg.memory.betas,
            beta_mem=cfg.neuron.beta_mem,
            threshold=cfg.neuron.threshold,
            gamma=cfg.neuron.gamma,
            surrogate_name=cfg.neuron.surrogate,
            surrogate_alpha=cfg.neuron.surrogate_alpha,
            num_keypoints=cfg.model.num_keypoints,
            predictor_hidden_dim=cfg.model.predictor_hidden_dim,
            num_objects=cfg.environment.num_objects,
            local_lr=cfg.model.local_lr,
            lambda_kin_feedback=cfg.model.lambda_kin_feedback,
            q_dim=cfg.model.q_dim,
            p_dim=cfg.model.p_dim,
            ema_decay=cfg.model.ema_decay,
            enable_corrector=corr_enabled,
            corrector_dim=cfg.corrector.dim,
            corrector_beta_mem=cfg.corrector.beta_mem,
            corrector_beta_adapt=cfg.corrector.beta_adapt,
            corrector_v_th0=cfg.corrector.v_th0,
            corrector_gamma=cfg.corrector.gamma,
            corrector_alpha_gain=cfg.corrector.alpha_gain,
        )

    def init_buffer(self) -> None:
        for name, param in self.named_parameters():
            if param.requires_grad:
                buf_name = f"delta_w_{name.replace('.', '_')}"
                self.register_buffer(buf_name, torch.zeros_like(param.data), persistent=False)

    @property
    def delta_w_buffer(self) -> Dict[str, torch.Tensor]:
        bufs = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                buf_name = f"delta_w_{name.replace('.', '_')}"
                if hasattr(self, buf_name):
                    bufs[name] = getattr(self, buf_name)
        return bufs

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> SPWMState:
        return SPWMState(
            encoder_states=None,
            dynamics_state=self.dynamics.init_state(batch_size, device=device),
        )

    def step(
        self,
        event_frame: torch.Tensor,
        state: Optional[SPWMState] = None,
        target_next_latent: Optional[torch.Tensor] = None,
        target_kinematics: Optional[torch.Tensor] = None,
        accumulate_local_updates: bool = True,
        learning_rate: Optional[float] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        B = event_frame.shape[0]
        device = event_frame.device

        if state is None:
            state = self.init_state(B, device=device)

        seq_len_val = 1.0 if seq_len is None else float(seq_len)

        with torch.no_grad():
            sensory_x, new_enc_states = self.encoder.step(event_frame, state.encoder_states)
            sensory_keypoints = self.encoder.last_keypoints

            z_prev = state.dynamics_state.z_prev
            predicted_x = self.sensory_predictor(z_prev)
            epsilon_t = sensory_x - predicted_x

            z_t, new_dyn_state = self.dynamics.step(
                sensory_input=epsilon_t,
                state=state.dynamics_state,
                sensory_keypoints=sensory_keypoints,
            )

            pred_out = self.predictor(z_t)
            z_hat_next = pred_out.predicted_latent

            pred_error = None
            if target_next_latent is not None:
                pred_error = torch.norm(z_hat_next - target_next_latent, dim=-1)

            decoded_kinematics = self.physical_decoder(z_t)

            if accumulate_local_updates:
                scale = B * seq_len_val
                delta_w_pred = (epsilon_t.T @ z_prev) / scale
                if "sensory_predictor.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["sensory_predictor.weight"].add_(delta_w_pred)

                l_lat = epsilon_t @ self.sensory_predictor.weight

                if self.lambda_kin_feedback > 0.0 and target_kinematics is not None:
                    if decoded_kinematics is not None:
                        epsilon_kin = decoded_kinematics - target_kinematics
                        l_kin = epsilon_kin @ self.B_kin_feedback
                    else:
                        l_kin = 0.0
                    l_lat = l_lat + self.lambda_kin_feedback * l_kin

                l_q = l_lat[:, :self.dynamics.q_dim]
                l_p = l_lat[:, self.dynamics.q_dim:]
                l_mem = l_p @ self.dynamics.fuse_spikes.weight

                delta_w_in = (l_mem.T @ epsilon_t) / scale
                if "dynamics.input_proj.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.input_proj.weight"].add_(delta_w_in)

                p_prev = z_prev[:, self.dynamics.q_dim:]
                delta_w_rec = (l_mem.T @ p_prev) / scale
                if "dynamics.recurrent_proj.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.recurrent_proj.weight"].add_(delta_w_rec)

                q_prev = z_prev[:, :self.dynamics.q_dim]
                delta_w_q = (l_mem.T @ q_prev) / scale
                if "dynamics.q_proj.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.q_proj.weight"].add_(delta_w_q)

                ema_spikes = state.dynamics_state.ema_spikes
                delta_w_vel = (l_q.T @ ema_spikes) / scale
                if "dynamics.W_vel.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.W_vel.weight"].add_(delta_w_vel)

            new_state = SPWMState(
                encoder_states=new_enc_states,
                dynamics_state=new_dyn_state,
            )

            output = SPWMStepOutput(
                latent=z_t,
                predicted_next_latent=z_hat_next,
                error_neurons=epsilon_t,
                prediction_error=pred_error,
                decoded_kinematics=decoded_kinematics,
                ema_spikes=state.dynamics_state.ema_spikes,
            )
            return output, new_state

    def apply_accumulated_updates(self, learning_rate: Optional[float] = None) -> None:
        lr = self.local_lr if learning_rate is None else learning_rate
        for name, param in self.named_parameters():
            if name in self.delta_w_buffer:
                update = self.delta_w_buffer[name]
                if torch.count_nonzero(update) > 0:
                    clamped_update = torch.clamp(update * lr, -0.1, 0.1)
                    param.data.add_(clamped_update)
                self.delta_w_buffer[name].zero_()

    def forward(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
        accumulate_local_updates: bool = True,
        learning_rate: Optional[float] = None,
        target_kinematics: Optional[torch.Tensor] = None,
    ) -> SPWMSequenceOutput:
        B, T, C, H, W = event_sequence.shape
        device = event_sequence.device
        state = initial_state if initial_state is not None else self.init_state(B, device=device)

        latent_steps = []
        pred_steps = []
        error_neuron_steps = []
        fast_spk_steps = []
        slow_spk_steps = []
        decoded_steps = []

        for t in range(T):
            event_frame = event_sequence[:, t]
            kin_t = target_kinematics[:, t] if target_kinematics is not None else None
            step_out, state = self.step(
                event_frame=event_frame,
                state=state,
                target_kinematics=kin_t,
                accumulate_local_updates=accumulate_local_updates,
                learning_rate=learning_rate,
                seq_len=T,
            )
            latent_steps.append(step_out.latent)
            pred_steps.append(step_out.predicted_next_latent)
            error_neuron_steps.append(step_out.error_neurons)
            if step_out.decoded_kinematics is not None:
                decoded_steps.append(step_out.decoded_kinematics)

            pop_spks = state.dynamics_state.memory_state.spikes
            fast_spk_steps.append(pop_spks[0])
            slow_spk_steps.append(pop_spks[-1])

        latents = torch.stack(latent_steps, dim=1)
        predicted_latents = torch.stack(pred_steps, dim=1)
        error_neurons = torch.stack(error_neuron_steps, dim=1)
        pred_errors = torch.norm(predicted_latents[:, :-1] - latents[:, 1:], dim=-1)

        fast_spikes = torch.stack(fast_spk_steps, dim=1)
        slow_spikes = torch.stack(slow_spk_steps, dim=1)
        decoded_kinematics = torch.stack(decoded_steps, dim=1) if decoded_steps else None

        all_spikes = torch.cat([fast_spikes, slow_spikes], dim=-1)
        dyn_spike_rate = (all_spikes > 0).float().mean()
        enc_spike_rate = torch.tensor(0.0, device=device)
        mean_spike_rate = dyn_spike_rate

        return SPWMSequenceOutput(
            latent_states=latents,
            predicted_latents=predicted_latents,
            prediction_errors=pred_errors,
            error_neurons=error_neurons,
            fast_spikes=fast_spikes,
            slow_spikes=slow_spikes,
            encoder_spike_rate=enc_spike_rate,
            dynamics_spike_rate=dyn_spike_rate,
            mean_spike_rate=mean_spike_rate,
            decoded_kinematics=decoded_kinematics,
        )

    def predict_future(
        self,
        initial_latent: torch.Tensor,
        horizon: int = 50,
        use_corrector: bool = True,
        initial_corrector_state: Optional[SlowCorrectorState] = None,
        burn_in_steps: int = 0,
    ) -> torch.Tensor:
        """Autonomous Rollout. Delegates directly to single source of truth rollout_latent."""
        res = rollout_latent(
            model=self,
            z_init=initial_latent,
            horizon=horizon,
            use_corrector=use_corrector,
            burn_in_steps=burn_in_steps,
            with_grad=False,
            initial_corrector_state=initial_corrector_state,
            decode_kinematics=False,
        )
        return res.trajectory

    def predict_rollout(
        self,
        initial_latent: torch.Tensor,
        horizon: int = 100,
        use_corrector: bool = True,
        initial_corrector_state: Optional[SlowCorrectorState] = None,
        return_corrector_state: bool = False,
        return_fast_trajectory: bool = False,
        burn_in_steps: int = 0,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[SlowCorrectorState]],
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[SlowCorrectorState], torch.Tensor],
    ]:
        """Differentiable rollout for multi-step training. Delegates to rollout_latent."""
        res = rollout_latent(
            model=self,
            z_init=initial_latent,
            horizon=horizon,
            use_corrector=use_corrector,
            burn_in_steps=burn_in_steps,
            with_grad=True,
            initial_corrector_state=initial_corrector_state,
            decode_kinematics=False,
        )
        if return_corrector_state and return_fast_trajectory:
            return res.trajectory, res.spikes, res.corrector_state, res.fast_trajectory
        if return_corrector_state:
            return res.trajectory, res.spikes, res.corrector_state
        if return_fast_trajectory:
            return res.trajectory, res.spikes, res.fast_trajectory
        return res.trajectory, res.spikes

    def online_step(
        self,
        event_frame: torch.Tensor,
        state: Optional[SPWMState] = None,
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        return self.step(event_frame, state=state, accumulate_local_updates=False)