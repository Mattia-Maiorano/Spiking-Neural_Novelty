"""
SPWM: Spiking Predictive World Model (v1).
End-to-end integration of event encoding, multi-timescale recurrent spiking dynamics,
predictive latent modeling, and autonomous multi-step rollouts.
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
    """Complete recurrent state of SPWM model."""
    encoder_states: Optional[Tuple[NeuronState, NeuronState, NeuronState]]
    dynamics_state: DynamicsState


@dataclass
class SPWMStepOutput:
    """Output from a single step execution."""
    latent: torch.Tensor  # [B, latent_dim]
    predicted_next_latent: torch.Tensor  # [B, latent_dim]
    prediction_error: Optional[torch.Tensor]  # [B] if target latent is available
    decoded_kinematics: Optional[torch.Tensor]  # [B, 4 * N]


@dataclass
class SPWMSequenceOutput:
    """Output from a full sequence forward pass."""
    latent_states: torch.Tensor  # [B, T, latent_dim] (z_1 ... z_T)
    predicted_latents: torch.Tensor  # [B, T, latent_dim] (z_hat_2 ... z_hat_(T+1))
    prediction_errors: torch.Tensor  # [B, T-1] distance(z_hat_(t+1), z_(t+1))
    fast_spikes: torch.Tensor  # [B, T, dim_fast]
    slow_spikes: torch.Tensor  # [B, T, dim_slow]
    encoder_spike_rate: torch.Tensor
    dynamics_spike_rate: torch.Tensor
    mean_spike_rate: torch.Tensor
    decoded_kinematics: Optional[torch.Tensor] = None  # [B, T, 4 * N]


class SPWM(nn.Module):
    """
    Spiking Predictive World Model (V1).
    Receives event streams, generates multi-timescale spiking dynamics,
    and learns predictive models of the future in an abstract latent space.
    """

    def __init__(
        self,
        in_channels: int = 2,
        height: int = 32,
        width: int = 32,
        encoder_conv_channels: Tuple[int, int] = (32, 64),
        encoder_dim: int = 128,
        latent_dim: int = 128,
        timescale_dims: Tuple[int, int] = (64, 64),
        betas: Tuple[float, float] = (0.8, 0.98),
        threshold: float = 1.0,
        reset_mechanism: str = "hard",
        surrogate_name: str = "atan",
        surrogate_alpha: float = 2.0,
        predictor_hidden_dim: int = 256,
        num_objects: int = 1,
        predictive_coding_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.predictive_coding_enabled = predictive_coding_enabled

        # 1. Spiking Sensory Event Encoder
        self.encoder = EventEncoder(
            in_channels=in_channels,
            height=height,
            width=width,
            conv_channels=encoder_conv_channels,
            out_dim=encoder_dim,
            beta=betas[0],
            threshold=threshold,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # 2. Multi-timescale Recurrent Spiking Dynamics
        self.dynamics = SpikingLatentDynamics(
            input_dim=encoder_dim,
            latent_dim=latent_dim,
            timescale_dims=timescale_dims,
            betas=betas,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate_name=surrogate_name,
            surrogate_alpha=surrogate_alpha,
        )

        # 3. Latent Predictor
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim,
            residual=True,
        )

        # 4. Physical Decoder Probe (ground-truth validation)
        self.physical_decoder = PhysicalDecoder(
            latent_dim=latent_dim,
            num_objects=num_objects,
            hidden_dim=latent_dim,
        )

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
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        """
        Executes a single-step recurrent forward pass:
            event_t -> sensory embedding -> spiking dynamics -> z_t -> z_hat_(t+1)
        """
        B = event_frame.shape[0]
        device = event_frame.device

        if state is None:
            state = self.init_state(B, device=device)

        # Encode sensory input
        sensory_t, new_enc_states = self.encoder.step(event_frame, state.encoder_states)

        # Update recurrent spiking dynamics
        z_t, new_dyn_state = self.dynamics.step(sensory_t, state.dynamics_state)

        # Predict next latent state
        pred_out = self.predictor(z_t)
        z_hat_next = pred_out.predicted_latent

        # Calculate prediction error if target is provided
        pred_error = None
        if target_next_latent is not None:
            pred_error = torch.norm(z_hat_next - target_next_latent, dim=-1)

        # Physical probe decoding
        decoded_kinematics = self.physical_decoder(z_t)

        new_state = SPWMState(
            encoder_states=new_enc_states,
            dynamics_state=new_dyn_state,
        )

        output = SPWMStepOutput(
            latent=z_t,
            predicted_next_latent=z_hat_next,
            prediction_error=pred_error,
            decoded_kinematics=decoded_kinematics,
        )

        return output, new_state

    def forward(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
    ) -> SPWMSequenceOutput:
        """
        Processes full event sequence [B, T, 2, H, W].
        Computes latent states z_1:T, one-step predictions z_hat_2:T+1,
        and prediction errors ||z_hat_(t+1) - z_(t+1)||_2.
        """
        B, T, C, H, W = event_sequence.shape
        device = event_sequence.device

        # 1. Encode sensory sequence
        sensory_seq, enc_spike_rate = self.encoder(event_sequence)  # [B, T, enc_dim]

        # 2. Unroll recurrent spiking dynamics
        init_dyn_state = initial_state.dynamics_state if initial_state is not None else None
        dyn_out, final_dyn_state = self.dynamics(sensory_seq, initial_state=init_dyn_state)
        latent_states = dyn_out.latent_states  # [B, T, latent_dim]

        # 3. Predict next latent states: z_hat_(t+1) = predictor(z_t)
        pred_out = self.predictor(latent_states)
        predicted_latents = pred_out.predicted_latent  # [B, T, latent_dim]

        # 4. Compute prediction errors:
        # z_hat from t=0 predicts z at t=1:
        # predicted_latents[:, :-1] compared with latent_states[:, 1:]
        z_targets = latent_states[:, 1:].detach()  # Stop gradient for target latent
        z_predictions = predicted_latents[:, :-1]
        pred_errors = torch.norm(z_predictions - z_targets, dim=-1)  # [B, T-1]

        # 5. Decode physical kinematics probe
        decoded_kinematics = self.physical_decoder(latent_states)

        # 6. Overall spike statistics
        mean_spike_rate = 0.5 * (enc_spike_rate + dyn_out.mean_spike_rate)

        return SPWMSequenceOutput(
            latent_states=latent_states,
            predicted_latents=predicted_latents,
            prediction_errors=pred_errors,
            fast_spikes=dyn_out.fast_spikes,
            slow_spikes=dyn_out.slow_spikes,
            encoder_spike_rate=enc_spike_rate,
            dynamics_spike_rate=dyn_out.mean_spike_rate,
            mean_spike_rate=mean_spike_rate,
            decoded_kinematics=decoded_kinematics,
        )

    def predict_future(
        self,
        initial_latent: torch.Tensor,
        horizon: int = 50,
    ) -> torch.Tensor:
        """
        Autonomous Rollout without future sensory observations.
        Iteratively propagates predictions through the latent predictor:
            z_t -> z_hat_(t+1) -> z_hat_(t+2) -> ... -> z_hat_(t+H)
        initial_latent: [B, latent_dim]
        Returns: [B, horizon, latent_dim]
        """
        predictions = []
        curr_z = initial_latent

        for _ in range(horizon):
            pred_out = self.predictor(curr_z)
            curr_z = pred_out.predicted_latent
            predictions.append(curr_z)

        return torch.stack(predictions, dim=1)  # [B, horizon, latent_dim]

    def online_step(
        self,
        event_frame: torch.Tensor,
        state: Optional[SPWMState] = None,
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        """
        Online inference step without autograd tracking.
        Enables streaming evaluation and neuromorphic event-by-event inference.
        """
        with torch.no_grad():
            return self.step(event_frame, state)

    def predictive_coding_update(
        self,
        predicted_state: SPWMState,
        observation: torch.Tensor,
        prediction_error: torch.Tensor,
    ) -> SPWMState:
        """
        Predictive coding state update interface (prepared for V2).
        Allows error-driven local state updates.
        """
        # In V1, recurrent dynamics already update state from input; this hook establishes
        # the required API for V2 local error-correction.
        return predicted_state
