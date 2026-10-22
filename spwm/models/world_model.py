"""
Layer 1 SPWM: Spiking Predictive World Model with Cascaded Temporal Relay.
Composes pure Layer 1 neural primitives (EventEncoder, SpikingLatentDynamics,
CascadedTemporalRelay, PhysicalDecoder) into a unified, strictly typed world model.
Zero dead plasticity buffers, zero legacy corrector branches, zero upward imports.
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.states import (
    SPWMSequenceOutput,
    SPWMState,
    SPWMStepOutput,
)
from spwm.contracts.tensors import assert_phase_space_contract, assert_tensor_contract
from spwm.models.dynamics import SpikingLatentDynamics
from spwm.models.encoder import EventEncoder
from spwm.models.predictors import (
    CascadedTemporalRelay,
    LatentPredictor,
    PhysicalDecoder,
)


class SPWM(nn.Module):
    """
    SPWM: Spiking Predictive World Model (v0.2.0).
    Unified composition of:
      1. SpatialSoftmax Keypoint EventEncoder
      2. Top-down Predictive Coding sensory feedback (`sensory_predictor`)
      3. Dual-timescale ALIF SpikingLatentDynamics
      4. CascadedTemporalRelay of Stateless Predictors (M1, M2, M3)
      5. PhysicalDecoder kinematic linear probe
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        encoder_conv_channels: Tuple[int, int] = (16, 32),
        encoder_dim: int = 128,
        latent_dim: int = 128,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        num_keypoints: Optional[int] = None,
        timescale_dims: Tuple[int, ...] = (128, 128),
        betas: Tuple[float, ...] = (0.90, 0.985),
        beta_mem: float = 0.80,
        threshold: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        predictor_hidden_dim: int = 256,
        num_objects: int = 1,
        ema_decay: float = 0.9,
        decoder_ema_alpha: float = 0.85,
        phase_horizon: int = 10,
    ) -> None:
        super().__init__()
        # Canonical symplectic partition resolution when instantiated with shorthand latent_dim in unit tests
        resolved_q = q_dim if q_dim is not None else (latent_dim // 4)
        resolved_p = p_dim if p_dim is not None else (latent_dim - resolved_q)
        resolved_kp = num_keypoints if num_keypoints is not None else (resolved_q // 2)

        assert_phase_space_contract(
            latent_dim=latent_dim,
            q_dim=resolved_q,
            p_dim=resolved_p,
            num_keypoints=resolved_kp,
        )

        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.q_dim = resolved_q
        self.p_dim = resolved_p
        self.num_keypoints = resolved_kp
        self.timescale_dims = tuple(timescale_dims)
        self.num_objects = num_objects
        self.phase_horizon = phase_horizon

        # 1. Spatial Softmax Keypoint Event Encoder
        self.encoder = EventEncoder(
            in_channels=in_channels,
            conv_channels=encoder_conv_channels,
            num_keypoints=resolved_kp,
            out_dim=encoder_dim,
            input_height=height,
            input_width=width,
        )

        # 2. Predictive Sensory Decoder (top-down predictive coding)
        self.sensory_predictor = nn.Linear(latent_dim, encoder_dim, bias=False)

        # 3. Recurrent Spiking Latent Dynamics with ALIF Memory Core
        self.dynamics = SpikingLatentDynamics(
            input_dim=encoder_dim,
            latent_dim=latent_dim,
            q_dim=resolved_q,
            p_dim=resolved_p,
            ema_decay=ema_decay,
            timescale_dims=timescale_dims,
            betas=betas,
            beta_mem=beta_mem,
            v_th0=threshold,
            gamma=gamma,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # 4. Cascaded Temporal Relay of Stateless Predictors (M1, M2, M3)
        self.predictor = CascadedTemporalRelay(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            residual=True,
            phase_horizon=phase_horizon,
        )

        # 5. Physical Kinematic Probe Decoder
        self.physical_decoder = PhysicalDecoder(
            latent_dim=latent_dim,
            q_dim=resolved_q,
            p_dim=resolved_p,
            num_objects=num_objects,
            alpha=decoder_ema_alpha,
        )

    @property
    def relay(self) -> CascadedTemporalRelay:
        return self.predictor

    @property
    def m1(self) -> LatentPredictor:
        return self.predictor.m1

    @property
    def m2(self) -> LatentPredictor:
        return self.predictor.m2

    @property
    def m3(self) -> LatentPredictor:
        return self.predictor.m3

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "SPWM":
        """Instantiate SPWM directly from the typed ExperimentConfig (Single Source of Truth)."""
        return cls(
            in_channels=cfg.model.in_channels,
            height=cfg.environment.height,
            width=cfg.environment.width,
            encoder_conv_channels=cfg.model.encoder_conv_channels,
            encoder_dim=cfg.model.encoder_dim,
            latent_dim=cfg.model.latent_dim,
            q_dim=cfg.model.q_dim,
            p_dim=cfg.model.p_dim,
            num_keypoints=cfg.model.num_keypoints,
            timescale_dims=cfg.memory.timescale_dims,
            betas=cfg.memory.betas,
            beta_mem=cfg.neuron.beta_mem,
            threshold=cfg.neuron.threshold,
            gamma=cfg.neuron.gamma,
            surrogate_name=cfg.neuron.surrogate,
            surrogate_alpha=cfg.neuron.surrogate_alpha,
            predictor_hidden_dim=cfg.model.predictor_hidden_dim,
            num_objects=cfg.environment.num_objects,
            ema_decay=cfg.model.ema_decay,
            decoder_ema_alpha=cfg.model.decoder_ema_alpha,
            phase_horizon=cfg.training.phase_horizon,
        )

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
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        """
        Executes a single sensory-driven predictive coding step.
        Respects the active PyTorch autograd context (`torch.enable_grad()` vs `torch.no_grad()`).
        """
        assert_tensor_contract(
            event_frame,
            (None, self.in_channels, self.height, self.width),
            "SPWM.step.event_frame",
        )
        B = event_frame.shape[0]
        device = event_frame.device

        if state is None:
            state = self.init_state(B, device=device)

        sensory_x, sensory_keypoints = self.encoder.encode_step(event_frame)

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

        new_state = SPWMState(
            encoder_states=None,
            dynamics_state=new_dyn_state,
        )

        output = SPWMStepOutput(
            latent=z_t,
            predicted_next_latent=z_hat_next,
            error_neurons=epsilon_t,
            prediction_error=pred_error,
            decoded_kinematics=decoded_kinematics,
            ema_spikes=state.dynamics_state.ema_spikes,
            keypoints=sensory_keypoints,
        )
        return output, new_state

    def forward(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
    ) -> SPWMSequenceOutput:
        """
        Processes a full sequence of differential event frames [B, T, C, H, W].
        Preserves autograd graph when called in a gradient-enabled context.
        """
        assert_tensor_contract(
            event_sequence,
            (None, None, self.in_channels, self.height, self.width),
            "SPWM.forward.event_sequence",
        )
        B, T, _, _, _ = event_sequence.shape
        device = event_sequence.device
        state = initial_state if initial_state is not None else self.init_state(B, device=device)

        latent_steps = []
        pred_steps = []
        error_neuron_steps = []
        fast_spk_steps = []
        slow_spk_steps = []
        decoded_steps = []
        keypoint_steps = []

        for t in range(T):
            event_frame = event_sequence[:, t]
            step_out, state = self.step(
                event_frame=event_frame,
                state=state,
            )
            latent_steps.append(step_out.latent)
            pred_steps.append(step_out.predicted_next_latent)
            error_neuron_steps.append(step_out.error_neurons)
            if step_out.decoded_kinematics is not None:
                decoded_steps.append(step_out.decoded_kinematics)
            if step_out.keypoints is not None:
                keypoint_steps.append(step_out.keypoints)

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
        keypoints = torch.stack(keypoint_steps, dim=1) if keypoint_steps else None

        all_spikes = torch.cat([fast_spikes, slow_spikes], dim=-1)
        dyn_spike_rate = (all_spikes > 0).float().mean()
        enc_spike_rate = torch.tensor(0.0, device=device)

        return SPWMSequenceOutput(
            latent_states=latents,
            predicted_latents=predicted_latents,
            prediction_errors=pred_errors,
            error_neurons=error_neurons,
            fast_spikes=fast_spikes,
            slow_spikes=slow_spikes,
            encoder_spike_rate=enc_spike_rate,
            dynamics_spike_rate=dyn_spike_rate,
            mean_spike_rate=dyn_spike_rate,
            decoded_kinematics=decoded_kinematics,
            keypoints=keypoints,
        )

    def predict_future(
        self,
        initial_latent: torch.Tensor,
        horizon: int = 50,
    ) -> torch.Tensor:
        """
        Autonomous multi-step rollout over the Cascaded Temporal Relay (M1 -> M2 <-> M3).
        Delegates to unified `rollout_latent`.
        """
        from spwm.models.rollout import rollout_latent

        res = rollout_latent(
            model=self,
            z_init=initial_latent,
            horizon=horizon,
            with_grad=False,
            decode_kinematics=False,
        )
        return res.trajectory