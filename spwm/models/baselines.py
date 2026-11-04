"""
Comparative Non-Spiking Baseline World Models (DOSSIER Section 6.1).

Implements three standard deep recurrent world model architectures matched in parameter budget
and input/output contracts to SPWM for rigorous empirical benchmarking:
1. ContinuousRSSM: Recurrent State-Space Model (Dreamer-style) with deterministic GRU state
   h_t in R^256 and continuous stochastic Gaussian state s_t in R^32.
2. DeepGRUPredictor: 2-layer action-conditioned GRU predictor matched to ~328k parameters.
3. ContinuousConvLSTM: Spatial-temporal ConvLSTM operating directly on 2D convolutional feature maps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from spwm.contracts.tensors import C_EVENT, J, assert_tensor_contract
from spwm.models.encoder import EventEncoder


@dataclass
class BaselineRolloutResult:
    """Output container for baseline multi-step autonomous rollouts."""
    trajectory: torch.Tensor           # [B, H, 12] predicted joint kinematics (q*, p*)
    latent_trajectory: torch.Tensor    # [B, H, D] recurrent/latent states
    positions: torch.Tensor            # [B, H, J] joint positions
    velocities: torch.Tensor           # [B, H, J] joint velocities


# ==============================================================================
# Baseline 1: Continuous RSSM (Dreamer-style Recurrent State-Space Model)
# ==============================================================================

class ContinuousRSSM(nn.Module):
    """
    Continuous Recurrent State-Space Model (Dreamer / RSSM).
    - Deterministic recurrent GRU state: h_t in R^{256}
    - Stochastic continuous state: s_t in R^{32}
    - Visual encoder: EventEncoder with SpatialSoftmax keypoint bottleneck -> e_t in R^{128}
    - Prior: p(s_t | h_t) ~ N(mu_prior, sigma_prior^2)
    - Posterior: q(s_t | h_t, e_t) ~ N(mu_post, sigma_post^2)
    - Kinematic Decoder: maps [h_t, s_t] in R^{288} -> [q_t, p_t] in R^{12}
    """

    def __init__(
        self,
        in_channels: int = C_EVENT,
        height: int = 32,
        width: int = 32,
        encoder_conv_channels: Tuple[int, int] = (16, 32),
        encoder_dim: int = 128,
        num_keypoints: int = 16,
        deterministic_dim: int = 256,
        stochastic_dim: int = 32,
        action_dim: int = J,
        joint_count: int = J,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.encoder_dim = encoder_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.action_dim = action_dim
        self.joint_count = joint_count
        self.min_std = float(min_std)

        # 1. Visual Event Encoder
        self.encoder = EventEncoder(
            in_channels=in_channels,
            conv_channels=encoder_conv_channels,
            num_keypoints=num_keypoints,
            out_dim=encoder_dim,
            input_height=height,
            input_width=width,
        )

        # 2. Deterministic GRU Transition: inputs are [s_{t-1}, a_{t-1}]
        self.gru_cell = nn.GRUCell(
            input_size=stochastic_dim + action_dim,
            hidden_size=deterministic_dim,
        )

        # 3. Stochastic Prior Network: p(s_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2 * stochastic_dim),
        )

        # 4. Stochastic Posterior Network: q(s_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_dim + encoder_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2 * stochastic_dim),
        )

        # 5. Articulated Joint Kinematic Readout: [h_t, s_t] -> [q_t, p_t]
        self.decoder = nn.Sequential(
            nn.Linear(deterministic_dim + stochastic_dim, 128),
            nn.ELU(),
            nn.Linear(128, 2 * joint_count),
        )

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes (h_0, s_0) to zeros."""
        h = torch.zeros(batch_size, self.deterministic_dim, device=device)
        s = torch.zeros(batch_size, self.stochastic_dim, device=device)
        return h, s

    def _get_stats(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, raw_std = torch.chunk(tensor, 2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return mu, std

    def forward(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward teacher-forcing pass across sequence.
        Args:
            events: [B, T, C, H, W]
            actions: [B, T, action_dim]
        Returns:
            Dictionary containing predicted kinematics, latent states, prior/post parameters, and KL divergence.
        """
        b, t, c, h_in, w_in = events.shape
        device = events.device

        # Encode all sensory frames: [B * T, C, H, W] -> [B, T, encoder_dim]
        flat_events = events.reshape(b * t, c, h_in, w_in)
        flat_enc, _ = self.encoder(flat_events, return_keypoints=True)
        e_seq = flat_enc.reshape(b, t, self.encoder_dim)

        h, s = self.init_state(b, device)

        pred_kinematics_list = []
        prior_mu_list = []
        prior_std_list = []
        post_mu_list = []
        post_std_list = []
        h_list = []
        s_list = []

        for step in range(t):
            # 1. Deterministic update
            a_prev = actions[:, step] if step == 0 else actions[:, step - 1]
            gru_in = torch.cat([s, a_prev], dim=-1)
            h = self.gru_cell(gru_in, h)

            # 2. Prior distribution
            prior_raw = self.prior_net(h)
            prior_mu, prior_std = self._get_stats(prior_raw)

            # 3. Posterior distribution (conditioned on current observation e_step)
            post_in = torch.cat([h, e_seq[:, step]], dim=-1)
            post_raw = self.posterior_net(post_in)
            post_mu, post_std = self._get_stats(post_raw)

            # 4. Reparameterized sampling from posterior during observation
            eps = torch.randn_like(post_std)
            s = post_mu + post_std * eps

            # 5. Kinematic decoding
            state_cat = torch.cat([h, s], dim=-1)
            kin = self.decoder(state_cat)

            pred_kinematics_list.append(kin)
            prior_mu_list.append(prior_mu)
            prior_std_list.append(prior_std)
            post_mu_list.append(post_mu)
            post_std_list.append(post_std)
            h_list.append(h)
            s_list.append(s)

        pred_kinematics = torch.stack(pred_kinematics_list, dim=1)  # [B, T, 12]
        prior_mu_all = torch.stack(prior_mu_list, dim=1)
        prior_std_all = torch.stack(prior_std_list, dim=1)
        post_mu_all = torch.stack(post_mu_list, dim=1)
        post_std_all = torch.stack(post_std_list, dim=1)
        latents = torch.cat([torch.stack(h_list, dim=1), torch.stack(s_list, dim=1)], dim=-1)

        # Analytical Gaussian KL divergence: KL(q_post || p_prior)
        kl_div = (
            torch.log(prior_std_all / post_std_all)
            + (post_std_all**2 + (post_mu_all - prior_mu_all) ** 2) / (2.0 * prior_std_all**2)
            - 0.5
        )
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return {
            "kinematics": pred_kinematics,
            "latent_states": latents,
            "kl_loss": kl_loss,
            "positions": pred_kinematics[..., :self.joint_count],
            "velocities": pred_kinematics[..., self.joint_count:],
        }

    @torch.no_grad()
    def rollout(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        t_start: int = 5,
    ) -> BaselineRolloutResult:
        """
        Executes burn-in with sensory observations up to t_start, then rolls out autonomously using prior dynamics.
        """
        b, t_seq, c, h_in, w_in = events.shape
        device = events.device
        h, s = self.init_state(b, device)

        # 1. Warm-up recurrent state during burn-in (steps 0 .. t_start)
        for step in range(t_start + 1):
            flat_ev = events[:, step]
            e_step, _ = self.encoder(flat_ev, return_keypoints=True)
            a_prev = actions[:, step] if step == 0 else actions[:, step - 1]
            gru_in = torch.cat([s, a_prev], dim=-1)
            h = self.gru_cell(gru_in, h)

            post_in = torch.cat([h, e_step], dim=-1)
            post_raw = self.posterior_net(post_in)
            post_mu, post_std = self._get_stats(post_raw)
            s = post_mu  # deterministic mean at inference

        # 2. Autonomous rollout for horizon steps
        pred_kin_list = []
        latents_list = []

        for h_step in range(horizon):
            act_idx = min(t_start + h_step, actions.shape[1] - 1)
            a_cur = actions[:, act_idx]

            gru_in = torch.cat([s, a_cur], dim=-1)
            h = self.gru_cell(gru_in, h)

            # Prior dynamics (no sensory input available!)
            prior_raw = self.prior_net(h)
            prior_mu, _ = self._get_stats(prior_raw)
            s = prior_mu  # mean of prior

            state_cat = torch.cat([h, s], dim=-1)
            kin = self.decoder(state_cat)

            pred_kin_list.append(kin)
            latents_list.append(state_cat)

        traj = torch.stack(pred_kin_list, dim=1)
        lat_traj = torch.stack(latents_list, dim=1)

        return BaselineRolloutResult(
            trajectory=traj,
            latent_trajectory=lat_traj,
            positions=traj[..., :self.joint_count],
            velocities=traj[..., self.joint_count:],
        )


# ==============================================================================
# Baseline 2: Deep GRU / LSTM Predictor (~328k parameters)
# ==============================================================================

class DeepGRUPredictor(nn.Module):
    """
    Standard 2-Layer Action-Conditioned Deep GRU Predictor.
    - Parameter budget matched to SPWM (~328k parameters)
    - Visual encoder: EventEncoder with SpatialSoftmax keypoint bottleneck -> e_t in R^{128}
    - 2-layer stacked GRU cells: hidden_dim = 164
    - Kinematic projection feedback for closed-loop autonomous unrolling
    - Kinematic decoder: maps GRU hidden state -> [q_t, p_t] in R^{12}
    """

    def __init__(
        self,
        in_channels: int = C_EVENT,
        height: int = 32,
        width: int = 32,
        encoder_conv_channels: Tuple[int, int] = (16, 32),
        encoder_dim: int = 128,
        num_keypoints: int = 16,
        hidden_dim: int = 164,
        action_dim: int = J,
        joint_count: int = J,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.joint_count = joint_count

        # 1. Visual Event Encoder
        self.encoder = EventEncoder(
            in_channels=in_channels,
            conv_channels=encoder_conv_channels,
            num_keypoints=num_keypoints,
            out_dim=encoder_dim,
            input_height=height,
            input_width=width,
        )

        # 2. Two-layer stacked GRU
        self.gru1 = nn.GRUCell(input_size=encoder_dim + action_dim, hidden_size=hidden_dim)
        self.gru2 = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 3. Closed-loop kinematic feedback projection (for autonomous rollout)
        self.kin_to_feat = nn.Linear(2 * joint_count, encoder_dim)

        # 4. Kinematic Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2 * joint_count),
        )

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        h2 = torch.zeros(batch_size, self.hidden_dim, device=device)
        return h1, h2

    def forward(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, t, c, h_in, w_in = events.shape
        device = events.device

        flat_events = events.reshape(b * t, c, h_in, w_in)
        flat_enc, _ = self.encoder(flat_events, return_keypoints=True)
        e_seq = flat_enc.reshape(b, t, self.encoder_dim)

        h1, h2 = self.init_state(b, device)
        pred_kin_list = []
        h2_list = []

        for step in range(t):
            gru_in = torch.cat([e_seq[:, step], actions[:, step]], dim=-1)
            h1 = self.gru1(gru_in, h1)
            h2 = self.gru2(h1, h2)
            kin = self.decoder(h2)

            pred_kin_list.append(kin)
            h2_list.append(h2)

        pred_kinematics = torch.stack(pred_kin_list, dim=1)
        latents = torch.stack(h2_list, dim=1)

        return {
            "kinematics": pred_kinematics,
            "latent_states": latents,
            "positions": pred_kinematics[..., :self.joint_count],
            "velocities": pred_kinematics[..., self.joint_count:],
        }

    @torch.no_grad()
    def rollout(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        t_start: int = 5,
    ) -> BaselineRolloutResult:
        b, t_seq, c, h_in, w_in = events.shape
        device = events.device
        h1, h2 = self.init_state(b, device)
        last_kin = torch.zeros(b, 2 * self.joint_count, device=device)

        # 1. Sensory burn-in (steps 0 .. t_start)
        for step in range(t_start + 1):
            e_step, _ = self.encoder(events[:, step], return_keypoints=True)
            gru_in = torch.cat([e_step, actions[:, step]], dim=-1)
            h1 = self.gru1(gru_in, h1)
            h2 = self.gru2(h1, h2)
            last_kin = self.decoder(h2)

        # 2. Autonomous rollout
        pred_kin_list = []
        latents_list = []

        for h_step in range(horizon):
            act_idx = min(t_start + h_step, actions.shape[1] - 1)
            a_cur = actions[:, act_idx]
            feat_in = self.kin_to_feat(last_kin)
            gru_in = torch.cat([feat_in, a_cur], dim=-1)

            h1 = self.gru1(gru_in, h1)
            h2 = self.gru2(h1, h2)
            last_kin = self.decoder(h2)

            pred_kin_list.append(last_kin)
            latents_list.append(h2)

        traj = torch.stack(pred_kin_list, dim=1)
        lat_traj = torch.stack(latents_list, dim=1)

        return BaselineRolloutResult(
            trajectory=traj,
            latent_trajectory=lat_traj,
            positions=traj[..., :self.joint_count],
            velocities=traj[..., self.joint_count:],
        )


# ==============================================================================
# Baseline 3: Continuous ConvLSTM (Spatial-temporal Convolutional Recurrence)
# ==============================================================================

class ConvLSTMCell(nn.Module):
    """2D Convolutional LSTM Cell for spatial-temporal representation."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur


class ContinuousConvLSTM(nn.Module):
    """
    Spatial-Temporal Continuous ConvLSTM Baseline.
    - Processes 2D feature maps directly with 2D spatial convolutions
    - Parameter budget matched to ~328k parameters (~300k-330k)
    - Spatial action conditioning via broadcast channel tiling
    - Kinematic readout via spatial adaptive pooling and MLP
    """

    def __init__(
        self,
        in_channels: int = C_EVENT,
        height: int = 32,
        width: int = 32,
        conv_channels: Tuple[int, int] = (32, 48),
        action_dim: int = J,
        action_embed_dim: int = 16,
        hidden_channels: int = 56,
        joint_count: int = J,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.action_embed_dim = action_embed_dim
        self.hidden_channels = hidden_channels
        self.joint_count = joint_count

        # 1. Convolutional Spatial Stem: [B, C, 32, 32] -> [B, 48, 8, 8]
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, conv_channels[0]),
            nn.SiLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, conv_channels[1]),
            nn.SiLU(),
        )

        # 2. Action Spatial Projection
        self.action_proj = nn.Linear(action_dim, action_embed_dim)

        # 3. ConvLSTM Cell
        total_cell_in = conv_channels[1] + action_embed_dim  # 48 + 16 = 64
        self.cell = ConvLSTMCell(
            in_channels=total_cell_in,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )

        # 4. Autonomous Feature Generator (produces feature map from hidden state during unrolling)
        self.rollout_trans = nn.Conv2d(hidden_channels, conv_channels[1], kernel_size=3, padding=1)

        # 5. Spatial Readout: [B, hidden_channels, 8, 8] -> [B, 2 * joint_count]
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels * 4, 64),
            nn.SiLU(),
            nn.Linear(64, 2 * joint_count),
        )

    def init_state(
        self,
        batch_size: int,
        spatial_h: int,
        spatial_w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_channels, spatial_h, spatial_w, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, spatial_h, spatial_w, device=device)
        return h, c

    def _embed_action_spatial(self, action: torch.Tensor, h_dim: int, w_dim: int) -> torch.Tensor:
        b = action.shape[0]
        a_emb = self.action_proj(action).reshape(b, self.action_embed_dim, 1, 1)
        return a_emb.expand(b, self.action_embed_dim, h_dim, w_dim)

    def forward(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b, t, c, h_in, w_in = events.shape
        device = events.device

        flat_events = events.reshape(b * t, c, h_in, w_in)
        stem_out = self.conv_stem(flat_events)
        _, c_feat, h_feat, w_feat = stem_out.shape
        feats = stem_out.reshape(b, t, c_feat, h_feat, w_feat)

        state = self.init_state(b, h_feat, w_feat, device)
        pred_kin_list = []
        latents_list = []

        for step in range(t):
            f_step = feats[:, step]
            a_spatial = self._embed_action_spatial(actions[:, step], h_feat, w_feat)
            cell_in = torch.cat([f_step, a_spatial], dim=1)

            h_cur, c_cur = self.cell(cell_in, state)
            state = (h_cur, c_cur)

            pooled = self.pool(h_cur).reshape(b, -1)
            kin = self.readout(pooled)

            pred_kin_list.append(kin)
            latents_list.append(pooled)

        pred_kinematics = torch.stack(pred_kin_list, dim=1)
        latents = torch.stack(latents_list, dim=1)

        return {
            "kinematics": pred_kinematics,
            "latent_states": latents,
            "positions": pred_kinematics[..., :self.joint_count],
            "velocities": pred_kinematics[..., self.joint_count:],
        }

    @torch.no_grad()
    def rollout(
        self,
        events: torch.Tensor,
        actions: torch.Tensor,
        horizon: int,
        t_start: int = 5,
    ) -> BaselineRolloutResult:
        b, t_seq, c, h_in, w_in = events.shape
        device = events.device

        sample_stem = self.conv_stem(events[:, 0])
        _, c_feat, h_feat, w_feat = sample_stem.shape

        state = self.init_state(b, h_feat, w_feat, device)

        # 1. Sensory burn-in (steps 0 .. t_start)
        for step in range(t_start + 1):
            f_step = self.conv_stem(events[:, step])
            a_spatial = self._embed_action_spatial(actions[:, step], h_feat, w_feat)
            cell_in = torch.cat([f_step, a_spatial], dim=1)
            h_cur, c_cur = self.cell(cell_in, state)
            state = (h_cur, c_cur)

        # 2. Autonomous rollout
        pred_kin_list = []
        latents_list = []

        for h_step in range(horizon):
            act_idx = min(t_start + h_step, actions.shape[1] - 1)
            a_cur = actions[:, act_idx]

            # Generate autonomous feature map from hidden state
            f_auto = self.rollout_trans(state[0])
            a_spatial = self._embed_action_spatial(a_cur, h_feat, w_feat)
            cell_in = torch.cat([f_auto, a_spatial], dim=1)

            h_cur, c_cur = self.cell(cell_in, state)
            state = (h_cur, c_cur)

            pooled = self.pool(h_cur).reshape(b, -1)
            kin = self.readout(pooled)

            pred_kin_list.append(kin)
            latents_list.append(pooled)

        traj = torch.stack(pred_kin_list, dim=1)
        lat_traj = torch.stack(latents_list, dim=1)

        return BaselineRolloutResult(
            trajectory=traj,
            latent_trajectory=lat_traj,
            positions=traj[..., :self.joint_count],
            velocities=traj[..., self.joint_count:],
        )
