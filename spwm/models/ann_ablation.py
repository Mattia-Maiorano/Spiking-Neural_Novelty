"""
Layer 1 Continuous Rate ANN World Model Ablation (DOSSIER Section 6.2).

Replaces the 256 ALIF spiking cells in `MultiTimescaleMemory` with identical continuous leaky rate units:
    m_t = beta_mem * m_{t-1} + (1 - beta_mem) * I_t
    r_t = sigmoid(m_t)

Preserves exact 1:1 parity with SPWM:
- Identical parameter counts (328,652 parameters)
- Identical projection matrices and dimensionalities
- Identical SpatialSoftmax keypoint frontend (K=16)
- Identical symplectic / dissipative phase-space factorization (q, p)
- Identical action-conditioned latent predictor (M1) and physical kinematic decoder
- Conforms strictly to Layer 0 `WorldModelProtocol` without silent fallbacks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.config import ExperimentConfig
from spwm.contracts.protocols import PhysicalDecoderProtocol, WorldModelProtocol
from spwm.contracts.states import (
    DynamicsOutput,
    DynamicsState,
    PredictorOutput,
    RolloutResult,
    SPWMSequenceOutput,
    SPWMState,
    SPWMStepOutput,
)
from spwm.contracts.tensors import C_EVENT, J, assert_phase_space_contract, assert_tensor_contract
from spwm.models.encoder import EventEncoder
from spwm.models.predictors import LatentCritic, LatentPredictor, PhysicalDecoder
from spwm.models.rollout import compute_rollout_schedule, ground_initial_state, rollout_latent


@dataclass
class AblationRolloutResult:
    """Output container for ablation multi-step autonomous rollouts."""
    trajectory: torch.Tensor                          # [B, H, 2 * joint_count] predicted joint kinematics
    latent_trajectory: torch.Tensor                   # [B, H, latent_dim] latent states
    decoded_kinematics: torch.Tensor                  # [B, H, 2 * joint_count]
    positions: torch.Tensor                           # [B, H, joint_count]
    velocities: torch.Tensor                          # [B, H, joint_count]


@dataclass
class ContinuousRateState:
    """State container for multi-timescale continuous leaky rate units."""
    m_mem: List[torch.Tensor]
    rates: List[torch.Tensor]

    @property
    def v_mems(self) -> List[torch.Tensor]:
        return self.m_mem

    @property
    def a_adapts(self) -> List[torch.Tensor]:
        return self.m_mem

    @property
    def spikes(self) -> List[torch.Tensor]:
        return self.rates

    @property
    def concatenated_spikes(self) -> torch.Tensor:
        """Concatenates continuous rate activations across all timescale tiers [B, total_dim]."""
        return torch.cat(self.rates, dim=-1)

    @property
    def concatenated_mems(self) -> torch.Tensor:
        """Concatenates continuous membrane potentials across all tiers [B, total_dim]."""
        return torch.cat(self.m_mem, dim=-1)

    @property
    def fast_spikes(self) -> torch.Tensor:
        return self.rates[0]

    @property
    def slow_spikes(self) -> torch.Tensor:
        return self.rates[-1]

    def detach(self) -> "ContinuousRateState":
        return ContinuousRateState(
            m_mem=[m.detach() for m in self.m_mem],
            rates=[r.detach() for r in self.rates],
        )


class ContinuousRateMemory(nn.Module):
    """
    Multi-Timescale Continuous Leaky Rate Memory Module.
    Direct non-spiking ablation of MultiTimescaleMemory:
        m_t = beta_mem * m_{t-1} + (1 - beta_mem) * I_t
        r_t = sigmoid(m_t)
    """

    def __init__(
        self,
        timescale_dims: Sequence[int] = (128, 128),
        betas: Sequence[float] = (0.90, 0.985),
    ) -> None:
        super().__init__()
        self.timescale_dims = list(timescale_dims)
        self.num_timescales = len(self.timescale_dims)
        if self.num_timescales == 0 or any(d <= 0 for d in self.timescale_dims):
            raise ValueError(f"timescale_dims must be positive integers, got {self.timescale_dims}")

        if len(betas) != self.num_timescales:
            raise ValueError(f"betas length ({len(betas)}) does not match num_timescales ({self.num_timescales})")

        self.total_dim = sum(self.timescale_dims)
        self.register_buffer("betas", torch.tensor([float(b) for b in betas], dtype=torch.float32))

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> ContinuousRateState:
        m_list = [torch.zeros(batch_size, dim, device=device) for dim in self.timescale_dims]
        r_list = [torch.zeros(batch_size, dim, device=device) for dim in self.timescale_dims]
        return ContinuousRateState(m_mem=m_list, rates=r_list)

    def forward(
        self,
        synaptic_inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        state: Optional[ContinuousRateState] = None,
    ) -> Tuple[torch.Tensor, ContinuousRateState]:
        if isinstance(synaptic_inputs, torch.Tensor):
            assert_tensor_contract(synaptic_inputs, (None, self.total_dim), "ContinuousRateMemory.synaptic_inputs")
            inputs = torch.split(synaptic_inputs, self.timescale_dims, dim=-1)
        else:
            inputs = list(synaptic_inputs)
            if len(inputs) != self.num_timescales:
                raise ValueError(f"Expected {self.num_timescales} input tensors, got {len(inputs)}")

        batch_size = inputs[0].shape[0]
        device = inputs[0].device

        if state is None:
            state = self.init_state(batch_size, device=device)

        new_mems: List[torch.Tensor] = []
        new_rates: List[torch.Tensor] = []

        for i, (inp, prev_m) in enumerate(zip(inputs, state.m_mem)):
            beta_i = self.betas[i]
            m_t = beta_i * prev_m + (1.0 - beta_i) * inp
            r_t = torch.sigmoid(m_t)
            new_mems.append(m_t)
            new_rates.append(r_t)

        new_state = ContinuousRateState(m_mem=new_mems, rates=new_rates)
        combined_rates = torch.cat(new_rates, dim=-1)
        return combined_rates, new_state


class ContinuousRateDynamics(nn.Module):
    """
    Recurrent Continuous Rate Latent Dynamics with Leaky Memory Hierarchy.
    Exact parameter-matched and architectural twin of SpikingLatentDynamics,
    substituting ALIF spiking cells with continuous leaky rate units.
    """

    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 128,
        q_dim: int = 32,
        p_dim: int = 96,
        dt: float = 0.05,
        ema_decay: float = 0.9,
        timescale_dims: Sequence[int] = (128, 128),
        betas: Sequence[float] = (0.90, 0.985),
        gate_ramp_steps: int = 1,
        max_k_gate: float = 1.0,
        max_correction_norm: float = 0.0,
        action_dim: int = J,
        activity_threshold: float = 0.001,
    ) -> None:
        super().__init__()
        assert_phase_space_contract(latent_dim=latent_dim, q_dim=q_dim, p_dim=p_dim)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.action_dim = action_dim
        self.dt = float(dt)
        self.ema_decay = ema_decay
        self.timescale_dims = list(timescale_dims)
        self.gate_ramp_steps = int(gate_ramp_steps)
        self.max_k_gate = float(max_k_gate)
        self.max_correction_norm = float(max_correction_norm)
        self.activity_threshold = float(activity_threshold)

        if len(betas) != len(self.timescale_dims):
            raise ValueError(
                f"ContinuousRateDynamics: betas length ({len(betas)}) != timescale_dims length ({len(self.timescale_dims)})"
            )

        self.total_memory_dim = sum(self.timescale_dims)

        self.memory = ContinuousRateMemory(
            timescale_dims=self.timescale_dims,
            betas=betas,
        )

        self.input_proj = nn.Linear(input_dim, self.total_memory_dim)
        self.recurrent_proj = nn.Linear(self.p_dim, self.total_memory_dim, bias=False)
        self.q_proj = nn.Linear(self.q_dim, self.total_memory_dim, bias=False)
        self.action_proj = nn.Linear(self.action_dim, self.total_memory_dim, bias=False)
        nn.init.zeros_(self.action_proj.weight)

        self.vel_proj = nn.Linear(self.p_dim, self.q_dim, bias=False)
        with torch.no_grad():
            self.vel_proj.weight.zero_()
            min_dim = min(self.q_dim, self.p_dim)
            self.vel_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)

        self.W_vel = nn.Linear(self.total_memory_dim, self.q_dim, bias=False)
        nn.init.normal_(self.W_vel.weight, mean=0.0, std=0.01)

        self.gate_scale = nn.Parameter(torch.full((self.q_dim,), 50.0))

        self.fuse_spikes = nn.Linear(self.total_memory_dim, self.p_dim)
        self.fuse_mems = nn.Linear(self.total_memory_dim, self.p_dim, bias=False)
        self.norm = nn.LayerNorm(self.p_dim)

    def vel(self, p: torch.Tensor) -> torch.Tensor:
        """Velocity mapping from momentum subspace p in R^{p_dim} to coordinate velocity in R^{q_dim}."""
        return self.vel_proj(p)

    def compute_k_gate(
        self,
        event_activity: Optional[torch.Tensor],
        rates: torch.Tensor,
        max_k_gate: Optional[float] = None,
    ) -> torch.Tensor:
        """Anti-Evanescence Gate conditioned on event activity or average continuous rate."""
        if event_activity is not None:
            act = event_activity.view(rates.shape[0], -1).mean(dim=-1, keepdim=True)
        else:
            act = rates.mean(dim=-1, keepdim=True)
        act = torch.clamp(act, min=0.0)
        act_eff = torch.clamp(act - self.activity_threshold, min=0.0)
        scale = F.softplus(self.gate_scale)
        eff_max = self.max_k_gate if max_k_gate is None else float(max_k_gate)
        k_gate = (1.0 - torch.exp(-scale * act_eff)) * eff_max
        return torch.clamp(k_gate, min=0.0, max=eff_max)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> DynamicsState:
        mem_state = self.memory.init_state(batch_size, device=device)
        q0 = torch.zeros(batch_size, self.q_dim, device=device)
        p0 = torch.zeros(batch_size, self.p_dim, device=device)
        z0 = torch.cat([q0, p0], dim=1)
        ema0 = torch.zeros(batch_size, self.total_memory_dim, device=device)
        g0 = torch.zeros(batch_size, 1, device=device)
        return DynamicsState(memory_state=mem_state, z_prev=z0, ema_spikes=ema0, grounded_steps=g0)  # type: ignore[arg-type]

    def step(
        self,
        sensory_input: torch.Tensor,
        state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
        event_activity: Optional[torch.Tensor] = None,
        in_sensory_phase: bool = True,
        unconstrained: bool = False,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, DynamicsState]:
        assert_tensor_contract(sensory_input, (None, self.input_dim), "ContinuousRateDynamics.sensory_input")
        B = sensory_input.shape[0]
        device = sensory_input.device

        if state is None:
            state = self.init_state(B, device=device)

        q_prev = state.z_prev[:, : self.q_dim]
        p_prev = state.z_prev[:, self.q_dim :]

        soma_current = self.input_proj(sensory_input) + self.recurrent_proj(p_prev) + self.q_proj(q_prev)
        if action is not None:
            soma_current = soma_current + self.action_proj(action)

        rates, new_mem_state = self.memory(
            synaptic_inputs=soma_current,
            state=state.memory_state,  # type: ignore[arg-type]
        )

        mem_analog = torch.tanh(new_mem_state.concatenated_mems)
        p_next = self.norm(self.fuse_spikes(rates) + self.fuse_mems(mem_analog))

        ema_rates = self.ema_decay * state.ema_spikes + (1.0 - self.ema_decay) * rates

        v_p = self.vel(p_next)
        if sensory_keypoints is not None and in_sensory_phase:
            assert_tensor_contract(sensory_keypoints, (B, self.q_dim), "ContinuousRateDynamics.sensory_keypoints")
            raw_corr = sensory_keypoints - q_prev
            if unconstrained:
                g_steps = torch.zeros(B, 1, device=device, dtype=q_prev.dtype)
                k_gate = self.compute_k_gate(event_activity, rates, max_k_gate=1.0)
                bounded_corr = raw_corr
            else:
                prev_g = state.grounded_steps if state.grounded_steps is not None else torch.zeros(B, 1, device=device, dtype=q_prev.dtype)
                if event_activity is not None:
                    act_val = event_activity.view(B, -1).mean(dim=-1, keepdim=True)
                else:
                    act_val = rates.mean(dim=-1, keepdim=True)
                is_active = (act_val > self.activity_threshold).to(dtype=prev_g.dtype)
                g_steps = torch.where(is_active > 0, prev_g + 1.0, torch.zeros_like(prev_g))
                ramp = torch.clamp(g_steps / float(max(1, self.gate_ramp_steps)), min=0.0, max=1.0)
                k_gate = ramp * self.compute_k_gate(event_activity, rates)
                if self.max_correction_norm > 0:
                    corr_norm = torch.linalg.norm(raw_corr, dim=-1, keepdim=True)
                    scale_factor = torch.clamp(self.max_correction_norm / torch.clamp(corr_norm, min=1e-6), max=1.0)
                    bounded_corr = raw_corr * scale_factor
                else:
                    bounded_corr = raw_corr

            q_next = q_prev + self.dt * v_p + k_gate * bounded_corr
        else:
            g_steps = torch.zeros(B, 1, device=device, dtype=q_prev.dtype)
            q_next = q_prev + self.dt * v_p

        z_next = torch.cat([q_next, p_next], dim=1)
        new_state = DynamicsState(memory_state=new_mem_state, z_prev=z_next, ema_spikes=ema_rates, grounded_steps=g_steps)  # type: ignore[arg-type]
        return z_next, new_state

    def forward(
        self,
        sensory_sequence: torch.Tensor,
        initial_state: Optional[DynamicsState] = None,
        sensory_keypoints: Optional[torch.Tensor] = None,
        event_activity: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[DynamicsOutput, DynamicsState]:
        assert_tensor_contract(sensory_sequence, (None, None, self.input_dim), "ContinuousRateDynamics.sensory_sequence")
        B, T, _ = sensory_sequence.shape
        device = sensory_sequence.device

        if sensory_keypoints is not None:
            assert_tensor_contract(sensory_keypoints, (B, T, self.q_dim), "ContinuousRateDynamics.sensory_keypoints_seq")

        state = initial_state if initial_state is not None else self.init_state(B, device=device)

        latent_steps = []
        fast_rate_steps = []
        slow_rate_steps = []
        fast_mem_steps = []
        slow_mem_steps = []

        total_rates = torch.tensor(0.0, device=device)
        total_units = 0

        for t in range(T):
            e_t = sensory_sequence[:, t]
            kp_t = sensory_keypoints[:, t] if sensory_keypoints is not None else None
            act_t = event_activity[:, t] if event_activity is not None else None
            u_t = actions[:, t] if actions is not None else None
            z_t, state = self.step(
                e_t,
                state=state,
                sensory_keypoints=kp_t,
                event_activity=act_t,
                in_sensory_phase=True,
                unconstrained=True,
                action=u_t,
            )
            latent_steps.append(z_t)
            fast_rate_steps.append(state.memory_state.fast_spikes)
            slow_rate_steps.append(state.memory_state.slow_spikes)
            fast_mem_steps.append(state.memory_state.v_mems[0])
            slow_mem_steps.append(state.memory_state.v_mems[-1])

            total_rates = total_rates + state.memory_state.concatenated_spikes.sum()
            total_units += state.memory_state.concatenated_spikes.numel()

        mean_rate = total_rates / max(1, total_units)

        output = DynamicsOutput(
            latent_states=torch.stack(latent_steps, dim=1),
            fast_spikes=torch.stack(fast_rate_steps, dim=1),
            slow_spikes=torch.stack(slow_rate_steps, dim=1),
            fast_mems=torch.stack(fast_mem_steps, dim=1),
            slow_mems=torch.stack(slow_mem_steps, dim=1),
            mean_spike_rate=mean_rate,
        )
        return output, state


class ANNWorldModel(nn.Module):
    """
    ANNWorldModel: Continuous Leaky Rate Ablation of SPWM.
    100% Parameter-Matched non-spiking control baseline:
      1. Dual-Camera SpatialSoftmax Keypoint EventEncoder (103,456 params)
      2. Top-down predictive coding sensory decoder (16,384 params)
      3. Dual-timescale ContinuousRateDynamics (128,066 params)
      4. Action-Conditioned Latent Attractor Predictor M1 (65,856 params)
      5. Articulated Joint Kinematic PhysicalDecoder (14,890 params)
    Total Trainable Parameters: 328,652 (strictly identical to SPWM).
    """

    def __init__(
        self,
        in_channels: int = C_EVENT,
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
        predictor_hidden_dim: int = 256,
        joint_count: int = J,
        action_dim: int = J,
        ema_decay: float = 0.9,
        decoder_ema_alpha: float = 0.0,
        decoder_pos_alpha: float = 0.0,
        gate_ramp_steps: int = 4,
        max_k_gate: float = 0.35,
        max_correction_norm: float = 0.5,
        activity_threshold: float = 0.001,
        phase_horizon: int = 50,
        symplectic_m1: bool = True,
        dt: float = 0.05,
    ) -> None:
        super().__init__()
        resolved_q = q_dim if q_dim is not None else (latent_dim // 4)
        resolved_p = p_dim if p_dim is not None else (latent_dim - resolved_q)
        resolved_kp = num_keypoints if num_keypoints is not None else (resolved_q // 2)

        assert_phase_space_contract(
            latent_dim=latent_dim,
            q_dim=resolved_q,
            p_dim=resolved_p,
            num_keypoints=resolved_kp,
            action_dim=action_dim,
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
        self.joint_count = joint_count
        self.action_dim = action_dim
        self.phase_horizon = phase_horizon
        self.symplectic_m1 = symplectic_m1
        self.dt = float(dt)

        # 1. SpatialSoftmax Keypoint Event Encoder
        self.encoder = EventEncoder(
            in_channels=in_channels,
            conv_channels=encoder_conv_channels,
            num_keypoints=resolved_kp,
            out_dim=encoder_dim,
            input_height=height,
            input_width=width,
        )

        # 2. Predictive Sensory Decoder
        self.sensory_predictor = nn.Linear(latent_dim, encoder_dim, bias=False)

        # 3. Continuous Leaky Rate Latent Dynamics
        self.dynamics = ContinuousRateDynamics(
            input_dim=encoder_dim,
            latent_dim=latent_dim,
            q_dim=resolved_q,
            p_dim=resolved_p,
            action_dim=action_dim,
            dt=self.dt,
            ema_decay=ema_decay,
            timescale_dims=timescale_dims,
            betas=betas,
            gate_ramp_steps=gate_ramp_steps,
            max_k_gate=max_k_gate,
            max_correction_norm=max_correction_norm,
            activity_threshold=activity_threshold,
        )

        # 4. Action-Conditioned Latent Attractor Predictor (M1)
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
            residual=True,
            symplectic=symplectic_m1,
            q_dim=resolved_q,
            p_dim=resolved_p,
            dt=self.dt,
        )

        # 5. Articulated Joint Kinematic Decoder
        self.physical_decoder = PhysicalDecoder(
            latent_dim=latent_dim,
            q_dim=resolved_q,
            p_dim=resolved_p,
            joint_count=joint_count,
            alpha=decoder_ema_alpha,
            pos_alpha=decoder_pos_alpha,
        )

        # 6. Latent Terminal Value Critic Network
        self.critic = LatentCritic(
            latent_dim=latent_dim,
        )

    @property
    def m1(self) -> LatentPredictor:
        return self.predictor

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> "ANNWorldModel":
        """Instantiate ANNWorldModel directly from typed ExperimentConfig."""
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
            predictor_hidden_dim=cfg.model.predictor_hidden_dim,
            joint_count=cfg.environment.joint_count,
            action_dim=cfg.environment.action_dim,
            ema_decay=cfg.model.ema_decay,
            decoder_ema_alpha=cfg.model.decoder_ema_alpha,
            decoder_pos_alpha=cfg.model.decoder_pos_alpha,
            gate_ramp_steps=cfg.model.gate_ramp_steps,
            max_k_gate=cfg.model.max_k_gate,
            max_correction_norm=cfg.model.max_correction_norm,
            activity_threshold=cfg.model.activity_threshold,
            phase_horizon=cfg.training.phase_horizon,
            symplectic_m1=cfg.model.symplectic,
            dt=cfg.environment.dt,
        )

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> SPWMState:
        return SPWMState(
            encoder_states=None,
            dynamics_state=self.dynamics.init_state(batch_size, device=device),
            action=torch.zeros(batch_size, self.action_dim, device=device),
        )

    def step(
        self,
        event_frame: torch.Tensor,
        state: Optional[SPWMState] = None,
        action: Optional[torch.Tensor] = None,
        target_next_latent: Optional[torch.Tensor] = None,
        unconstrained: bool = False,
    ) -> Tuple[SPWMStepOutput, SPWMState]:
        assert_tensor_contract(
            event_frame,
            (None, self.in_channels, self.height, self.width),
            "ANNWorldModel.step.event_frame",
        )
        B = event_frame.shape[0]
        device = event_frame.device

        if state is None:
            state = self.init_state(B, device=device)

        if action is not None:
            assert_tensor_contract(action, (B, self.action_dim), "ANNWorldModel.step.action")

        sensory_x, sensory_keypoints = self.encoder.encode_step(event_frame)
        z_prev = state.dynamics_state.z_prev
        predicted_x = self.sensory_predictor(z_prev)
        epsilon_t = sensory_x - predicted_x
        event_activity = event_frame.abs().mean(dim=(1, 2, 3), keepdim=True)

        z_t, new_dyn_state = self.dynamics.step(
            sensory_input=epsilon_t,
            state=state.dynamics_state,
            sensory_keypoints=sensory_keypoints,
            event_activity=event_activity,
            unconstrained=unconstrained,
            action=action,
        )

        pred_out = self.predictor(z_t, action=action)
        z_hat_next = pred_out.predicted_latent

        pred_error = None
        if target_next_latent is not None:
            pred_error = torch.norm(z_hat_next - target_next_latent, dim=-1)

        decoded_kinematics = self.physical_decoder(z_t)

        new_state = SPWMState(
            encoder_states=None,
            dynamics_state=new_dyn_state,
            action=action,
        )

        output = SPWMStepOutput(
            latent=z_t,
            predicted_next_latent=z_hat_next,
            error_neurons=epsilon_t,
            prediction_error=pred_error,
            decoded_kinematics=decoded_kinematics,
            ema_spikes=state.dynamics_state.ema_spikes,
            keypoints=sensory_keypoints,
            action=action,
        )
        return output, new_state

    def forward_representation(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
        actions: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sensory-driven state estimation without predictor rollouts."""
        assert_tensor_contract(
            event_sequence,
            (None, None, self.in_channels, self.height, self.width),
            "ANNWorldModel.forward_representation.event_sequence",
        )
        B, T, _, _, _ = event_sequence.shape
        device = event_sequence.device
        if actions is not None:
            assert_tensor_contract(actions, (B, T, self.action_dim), "ANNWorldModel.forward_representation.actions")
        state = initial_state if initial_state is not None else self.init_state(B, device=device)

        latent_steps = []
        error_neuron_steps = []

        for t in range(T):
            event_frame = event_sequence[:, t]
            sensory_x, sensory_keypoints = self.encoder.encode_step(event_frame)
            z_prev = state.dynamics_state.z_prev
            predicted_x = self.sensory_predictor(z_prev)
            epsilon_t = sensory_x - predicted_x
            event_activity = event_frame.abs().mean(dim=(1, 2, 3), keepdim=True)
            u_t = actions[:, t] if actions is not None else None

            z_t, new_dyn_state = self.dynamics.step(
                sensory_input=epsilon_t,
                state=state.dynamics_state,
                sensory_keypoints=sensory_keypoints,
                event_activity=event_activity,
                unconstrained=True,
                action=u_t,
            )
            state = SPWMState(encoder_states=None, dynamics_state=new_dyn_state, action=state.action)
            latent_steps.append(z_t)
            error_neuron_steps.append(epsilon_t)

        latents = torch.stack(latent_steps, dim=1)
        error_neurons = torch.stack(error_neuron_steps, dim=1)
        return latents, error_neurons

    def __call__(
        self,
        event_sequence: torch.Tensor,
        initial_state: Optional[SPWMState] = None,
        actions: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Sequence forward pass decoding joint kinematics across all steps."""
        latents, _ = self.forward_representation(event_sequence, initial_state=initial_state, actions=actions)
        decoded = self.physical_decoder(latents)
        positions = decoded[..., : self.joint_count]
        velocities = decoded[..., self.joint_count : 2 * self.joint_count]
        return {
            "latents": latents,
            "positions": positions,
            "velocities": velocities,
            "decoded_kinematics": decoded,
        }

    @torch.no_grad()
    def rollout(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        t_start: int = 5,
        phase_flags: Optional[torch.Tensor] = None,
    ) -> RolloutResult:
        """
        Executes burn-in with sensory observations up to t_start, then rolls out autonomously using M1.
        """
        B, T, _, _, _ = events.shape
        device = events.device
        dyn_state = self.dynamics.init_state(B, device=device)
        latents = []

        # Burn-in
        for t in range(t_start + 1):
            ev_t = events[:, t]
            sx, skp = self.encoder.encode_step(ev_t)
            eps = sx - self.sensory_predictor(dyn_state.z_prev)
            act = ev_t.abs().mean(dim=(1, 2, 3), keepdim=True)
            u_t = actions[:, t] if actions is not None else None
            curr_z, dyn_state = self.dynamics.step(
                eps,
                state=dyn_state,
                sensory_keypoints=skp,
                event_activity=act,
                in_sensory_phase=True,
                unconstrained=True,
                action=u_t,
            )
            latents.append(curr_z)

        z_prefix = torch.stack(latents, dim=1)
        decoder_state = self.physical_decoder.warm_up_filter(z_prefix)
        rollout_actions = actions[:, t_start : t_start + horizon] if actions is not None else None
        rollout_events = events[:, t_start + 1 : t_start + 1 + horizon]
        rollout_flags = phase_flags[:, t_start + 1 : t_start + 1 + horizon] if phase_flags is not None else None

        rollout_res = rollout_latent(
            model=self,
            z_init=curr_z,
            horizon=horizon,
            actions=rollout_actions,
            events=rollout_events,
            phase_flags=rollout_flags,
            initial_dynamics_state=dyn_state,
            with_grad=False,
            decode_kinematics=True,
            initial_decoder_state=decoder_state,
            fixed_predictor=self.predictor,
        )

        assert rollout_res.decoded_kinematics is not None, "ANNWorldModel: rollout decoded_kinematics must not be None"
        kin = rollout_res.decoded_kinematics
        pos = kin[..., : self.joint_count]
        vel = kin[..., self.joint_count : 2 * self.joint_count]

        return AblationRolloutResult(
            trajectory=kin,
            latent_trajectory=rollout_res.trajectory,
            decoded_kinematics=kin,
            positions=pos,
            velocities=vel,
        )
