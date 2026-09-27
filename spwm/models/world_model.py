"""
SPWM-v3.4: Kinematic-Guided e-prop — Closing the Representational Gap.
Features:
- Predictive coding core with error routing (ϵ_t = x_t - W_pred z_(t-1))
- Adaptive Leaky Integrate-and-Fire (ALIF) latent core with dynamic threshold homeostasis
- Dual-timescale hierarchy (50% Reactive β_adapt=0.90, 50% Deep Context β_adapt=0.985)
- Deterministic forward-only e-prop plasticity with O(1) memory complexity over long horizons
- v3.4: Kinematic feedback injection — ε_kin projected via W_probe into L^total augments
  the three-factor plasticity rule so ALIF dynamics align with geometric coordinates
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.models.encoder import EventEncoder
from spwm.models.latent_dynamics import SpikingLatentDynamics, DynamicsState, DynamicsOutput
from spwm.models.predictor import LatentPredictor, PhysicalDecoder, PredictorOutput
from spwm.models.neurons import NeuronState


@dataclass
class SPWMState:
    """Complete recurrent state of SPWM-v3 model."""
    encoder_states: Optional[Tuple[NeuronState, NeuronState, NeuronState]]
    dynamics_state: DynamicsState


@dataclass
class SPWMStepOutput:
    """Output from a single step execution in SPWM-v3."""
    latent: torch.Tensor  # [B, latent_dim] (z_t)
    predicted_next_latent: torch.Tensor  # [B, latent_dim] (z_hat_(t+1))
    error_neurons: torch.Tensor  # [B, encoder_dim] (ϵ_t = x_t - W_pred z_(t-1))
    prediction_error: Optional[torch.Tensor]  # [B] if target latent is available
    decoded_kinematics: Optional[torch.Tensor]  # [B, 4 * N]
    ema_spikes: Optional[torch.Tensor] = None  # [B, total_memory_dim]


@dataclass
class SPWMSequenceOutput:
    """Output from a full sequence forward pass in SPWM-v3."""
    latent_states: torch.Tensor  # [B, T, latent_dim]
    predicted_latents: torch.Tensor  # [B, T, latent_dim]
    prediction_errors: torch.Tensor  # [B, T-1]
    error_neurons: torch.Tensor  # [B, T, encoder_dim]
    fast_spikes: torch.Tensor  # [B, T, dim_fast]
    slow_spikes: torch.Tensor  # [B, T, dim_slow]
    encoder_spike_rate: torch.Tensor
    dynamics_spike_rate: torch.Tensor
    mean_spike_rate: torch.Tensor
    decoded_kinematics: Optional[torch.Tensor] = None


class SPWM(nn.Module):
    """
    SPWM-v3.1: Spiking Predictive World Model with ALIF Core, Normalized e-prop, and Recalibrated Dynamics.
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        encoder_conv_channels: Tuple[int, int] = (32, 64),
        encoder_dim: int = 128,
        latent_dim: int = 128,
        timescale_dims: Optional[Tuple[int, ...]] = None,
        betas: Tuple[float, ...] = (0.90, 0.985),
        beta_mem: float = 0.80,
        threshold: float = 1.0,
        gamma: float = 0.18,
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        num_keypoints: int = 16,
        predictor_hidden_dim: int = 256,
        num_objects: int = 1,
        local_lr: float = 1e-3,
        lambda_kin_feedback: float = 0.0,
        rls_enabled: bool = False,
        rls_forgetting: float = 0.99,
        rls_delta: float = 1.0,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.local_lr = local_lr
        # v3.4+: weight for kinematic feedback injection into e-prop learning signal
        self.lambda_kin_feedback = lambda_kin_feedback
        # v3.5: whether to use online RLS decoder for kinematic feedback
        self.rls_enabled = rls_enabled

        # Determine phase space dimensions consistently
        if q_dim is not None:
            num_keypoints = max(1, q_dim // 2)
            p_dim = p_dim if p_dim is not None else (latent_dim - q_dim)
        else:
            if latent_dim >= 2 * num_keypoints + 16:
                q_dim = 2 * num_keypoints
                p_dim = latent_dim - q_dim
            else:
                num_keypoints = max(1, (latent_dim // 4) // 2)
                q_dim = 2 * num_keypoints
                p_dim = latent_dim - q_dim

        # 1. Spatial Softmax Keypoint Event Encoder (SPWM-v4.0)
        self.encoder = EventEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            conv_channels=encoder_conv_channels,
            num_keypoints=num_keypoints,
            out_dim=encoder_dim,
        )


        # 2. Predictive Sensory Decoder (Predictive Coding Core: x_hat_t = W_pred * z_(t-1))
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

        # 4. Latent Predictor (z_hat_(t+1) = LatentPredictor(z_t))
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            residual=True,
        )

        # 5. Physical Decoder Probe (ground truth kinematics evaluation)
        self.physical_decoder = PhysicalDecoder(
            latent_dim=latent_dim,
            q_dim=self.dynamics.q_dim,
            p_dim=self.dynamics.p_dim,
            num_objects=num_objects,
            hidden_dim=latent_dim,
            rls_enabled=rls_enabled,
            rls_forgetting=rls_forgetting,
            rls_delta=rls_delta,
        )

        # Feedback Alignment: Fixed non-trainable feedback matrix for kinematic error retroprojection
        B_feedback = torch.randn(4 * num_objects, latent_dim) / (4.0 ** 0.5)
        self.register_buffer("B_kin_feedback", B_feedback)

        # Register plasticity buffers for online forward-only weight updates
        self.init_buffer()

    def init_buffer(self) -> None:
        """Initializes the plasticity accumulation buffer ΔW_buffer as PyTorch buffers."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                buf_name = f"delta_w_{name.replace('.', '_')}"
                self.register_buffer(buf_name, torch.zeros_like(param.data), persistent=False)

    @property
    def delta_w_buffer(self) -> Dict[str, torch.Tensor]:
        """Provides a dictionary view of delta_w buffers keyed by parameter name."""
        bufs = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                buf_name = f"delta_w_{name.replace('.', '_')}"
                if hasattr(self, buf_name):
                    bufs[name] = getattr(self, buf_name)
        return bufs

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> SPWMState:
        """Initializes quiescent model state across all sub-modules."""
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
        """
        Executes a single forward-only O(1) step:
            1. Encode raw sensory event: x_t = encoder(event_frame), extract keypoints
            2. Compute sensory prediction: x_hat_t = sensory_predictor(z_(t-1))
            3. Error neuron routing: ϵ_t = x_t - x_hat_t
            4. Latent dynamics update with ϵ_t and keypoints -> z_t = [q_t, p_t]
            5. Latent state prediction: z_hat_(t+1) = predictor(z_t)
            6. Accumulate forward-only e-prop updates (coherent q and p routing)
        """
        B = event_frame.shape[0]
        device = event_frame.device
        lr = self.local_lr if learning_rate is None else learning_rate

        if state is None:
            state = self.init_state(B, device=device)

        seq_len_val = 1.0 if seq_len is None else float(seq_len)

        with torch.no_grad():
            # 1. Encode sensory input frame and extract raw keypoints in [-1, 1]
            sensory_x, new_enc_states = self.encoder.step(event_frame, state.encoder_states)
            sensory_keypoints = getattr(self.encoder, "last_keypoints", None)  # [B, q_dim=32]

            # 2. Predictive coding: predict sensory observation from previous latent state
            z_prev = state.dynamics_state.z_prev  # [B, latent_dim=128]
            predicted_x = self.sensory_predictor(z_prev)  # [B, encoder_dim=128]

            # 3. Error Neurons ϵ_t
            epsilon_t = sensory_x - predicted_x  # [B, encoder_dim=128]

            # 4. Latent dynamics receives sensory prediction error ϵ_t and sensory keypoints
            z_t, new_dyn_state = self.dynamics.step(
                sensory_input=epsilon_t,
                state=state.dynamics_state,
                sensory_keypoints=sensory_keypoints,
            )

            # 5. Latent prediction of next state
            pred_out = self.predictor(z_t)
            z_hat_next = pred_out.predicted_latent

            # Prediction error if target latent is available
            pred_error = None
            if target_next_latent is not None:
                pred_error = torch.norm(z_hat_next - target_next_latent, dim=-1)

            # Kinematics decoding probe
            decoded_kinematics = self.physical_decoder(z_t)

            # 6. Forward-Only e-prop Plasticity Update Accumulation (Normalized by B and T)
            if accumulate_local_updates:
                scale = B * seq_len_val
                # Top-down sensory predictor update: ΔW_pred = (ϵ_t ⊗ z_prev) / (B * T)
                # Shape: [encoder_dim, latent_dim] = [128, 128]
                delta_w_pred = (epsilon_t.T @ z_prev) / scale
                if "sensory_predictor.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["sensory_predictor.weight"].add_(delta_w_pred)

                # Deterministic feedback in full latent space: L_lat = ϵ_t @ W_pred  [B, latent_dim=128]
                l_lat = epsilon_t @ self.sensory_predictor.weight

                # Kinematic Feedback Alignment: L^kin_t = ε_kin @ B_kin_feedback   [B, latent_dim=128]
                if (
                    self.lambda_kin_feedback > 0.0
                    and target_kinematics is not None
                ):
                    if self.rls_enabled and self.physical_decoder.rls is not None:
                        s_hat_rls, _ = self.physical_decoder.rls_update_step(z_t, target_kinematics)
                        epsilon_kin = s_hat_rls - target_kinematics  # [B, out_dim]
                    elif decoded_kinematics is not None:
                        epsilon_kin = decoded_kinematics - target_kinematics
                    else:
                        epsilon_kin = None

                    if epsilon_kin is not None:
                        l_kin = epsilon_kin @ self.B_kin_feedback    # [B, latent_dim=128]
                    else:
                        l_kin = 0.0

                    l_lat = l_lat + self.lambda_kin_feedback * l_kin

                # Decompose latent feedback into q [B, q_dim=32] and p [B, p_dim=96] components
                l_q = l_lat[:, :self.dynamics.q_dim]
                l_p = l_lat[:, self.dynamics.q_dim:]

                # Retroproject momentum feedback l_p [B, 96] through fuse_spikes [96, 128] to somatic currents:
                # l_mem: [B, total_memory_dim=128]
                l_mem = l_p @ self.dynamics.fuse_spikes.weight

                # Input projection update: ΔW_in = (l_mem^T @ ϵ_t) / (B * T) -> [128, 128]
                delta_w_in = (l_mem.T @ epsilon_t) / scale
                if "dynamics.input_proj.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.input_proj.weight"].add_(delta_w_in)

                # Recurrent projection update: ΔW_rec = (l_mem^T @ p_prev) / (B * T) -> [128, 96]
                p_prev = z_prev[:, self.dynamics.q_dim:]
                delta_w_rec = (l_mem.T @ p_prev) / scale
                if "dynamics.recurrent_proj.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.recurrent_proj.weight"].add_(delta_w_rec)

                # q projection update: ΔW_q = (l_mem^T @ q_prev) / (B * T) -> [128, 32]
                q_prev = z_prev[:, :self.dynamics.q_dim]
                delta_w_q = (l_mem.T @ q_prev) / scale
                if "dynamics.q_proj.weight" in self.delta_w_buffer:
                    self.delta_w_buffer["dynamics.q_proj.weight"].add_(delta_w_q)

                # Velocity map update: ΔW_vel = (l_q^T @ ema_spikes) / (B * T) -> [32, 128]
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
        """Applies accumulated plasticity buffer updates to model parameters and clears buffer."""
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
        """
        Processes full event sequence [B, T, C, H, W] in a forward-only stream.
        Maintains O(1) computational graph footprint.
        """
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
            # v3.4: pass per-timestep kinematic target if available
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

        latents = torch.stack(latent_steps, dim=1)  # [B, T, latent_dim]
        predicted_latents = torch.stack(pred_steps, dim=1)  # [B, T, latent_dim]
        error_neurons = torch.stack(error_neuron_steps, dim=1)  # [B, T, enc_dim]

        # Prediction errors: ||z_hat_(t+1) - z_(t+1)||
        pred_errors = torch.norm(predicted_latents[:, :-1] - latents[:, 1:], dim=-1)

        fast_spikes = torch.stack(fast_spk_steps, dim=1)
        slow_spikes = torch.stack(slow_spk_steps, dim=1)

        decoded_kinematics = torch.stack(decoded_steps, dim=1) if decoded_steps else None

        # Spike rates
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
    ) -> torch.Tensor:
        """Autonomous Rollout without future sensory observations."""
        predictions = []
        curr_z = initial_latent

        with torch.no_grad():
            for _ in range(horizon):
                pred_out = self.predictor(curr_z)
                curr_z = pred_out.predicted_latent
                predictions.append(curr_z)

        return torch.stack(predictions, dim=1)

    def online_step(
        self,
        event_frame: torch.Tensor,
        state: Optional[SPWMState] = None,
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        """Online forward step without autograd overhead."""
        return self.step(event_frame, state=state, accumulate_local_updates=False)
