"""
Latent Predictor and Physical Ground-Truth Decoding Heads.
Predicts future latent state p(z_(t+1) | z_t) and decodes physical kinematic variables.

v3.5: PhysicalDecoder gains an optional online RLS linear decoder that is updated
at each timestep via Recursive Least Squares, maintaining O(1) memory and zero autograd.
The freshly-updated W_rls is exposed for kinematic e-prop back-projection so the
learning signal is never stale with respect to the current probe weights.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class PredictorOutput:
    """Container for latent prediction output."""
    predicted_latent: torch.Tensor  # [B, ..., latent_dim]
    mean: torch.Tensor
    log_var: Optional[torch.Tensor] = None


class LatentPredictor(nn.Module):
    """
    Predictive Transition Head: p(z_(t+1) | z_t).
    Uses a residual MLP block to predict the temporal differential or next latent state,
    with an optional Port-Hamiltonian recurrent transition operator (W_rec = J - R)
    on the momentum p subspace to prevent kinetic energy dissipation and velocity divergence.

    Formulation:
        z = [q, p]
        W_rec = J - R
        J = 0.5 * (W_skew - W_skew^T)  (pure skew-symmetric: J^T = -J)
        R = diag(softplus(gamma_diss)) + eps_diss (positive semi-definite damping)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        residual: bool = True,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        use_port_hamiltonian: bool = True,
        eps_diss: float = 1e-4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.residual = residual
        self.use_port_hamiltonian = use_port_hamiltonian
        self.eps_diss = eps_diss

        if q_dim is None and p_dim is None:
            self.q_dim = max(1, latent_dim // 4)
            self.p_dim = max(1, latent_dim - self.q_dim)
        elif q_dim is None:
            self.p_dim = min(p_dim, latent_dim - 1)
            self.q_dim = max(1, latent_dim - self.p_dim)
        elif p_dim is None:
            self.q_dim = min(q_dim, latent_dim - 1)
            self.p_dim = max(1, latent_dim - self.q_dim)
        else:
            self.q_dim = q_dim
            self.p_dim = p_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        if self.use_port_hamiltonian:
            # Parametrization for momentum recurrence W_rec = J - R in R^{p_dim x p_dim}
            self.W_skew = nn.Parameter(torch.empty(self.p_dim, self.p_dim))
            nn.init.orthogonal_(self.W_skew, gain=0.5)
            # Damping parameter gamma: R = diag(softplus(gamma_diss) + eps_diss)
            self.gamma_diss = nn.Parameter(torch.zeros(self.p_dim))

    def get_port_hamiltonian_w_rec(self) -> torch.Tensor:
        """
        Computes discrete Port-Hamiltonian recurrence matrix: W_rec = J - R.
        J is skew-symmetric: J = 0.5 * (W_skew - W_skew^T)
        R is positive diagonal: R = diag(softplus(gamma_diss) + eps_diss)
        """
        J = 0.5 * (self.W_skew - self.W_skew.T)
        R = torch.diag(torch.nn.functional.softplus(self.gamma_diss) + self.eps_diss)
        return J - R

    def forward(self, z: torch.Tensor) -> PredictorOutput:
        """
        z: [B, latent_dim] or [B, T, latent_dim]
        Returns: PredictorOutput with predicted_latent [B, ..., latent_dim]
        """
        delta_z = self.net(z)

        if self.use_port_hamiltonian:
            # Extract momentum component p
            p = z[..., self.q_dim:self.q_dim + self.p_dim]
            W_rec = self.get_port_hamiltonian_w_rec()
            p_ph = p @ W_rec.T
            # Augment p prediction with Port-Hamiltonian conservative/dissipative dynamics
            delta_p = delta_z[..., self.q_dim:self.q_dim + self.p_dim] + p_ph
            delta_q = delta_z[..., :self.q_dim]
            delta_z = torch.cat([delta_q, delta_p], dim=-1)

        if self.residual:
            z_next = z + delta_z
        else:
            z_next = delta_z

        return PredictorOutput(
            predicted_latent=z_next,
            mean=z_next,
            log_var=None,  # Prepared for V4 probabilistic world model
        )


class OnlineRLSDecoder(nn.Module):
    """
    Online Linear Decoder via Recursive Least Squares (RLS).

    Maintains a linear map  W: [latent_dim -> out_dim]  and a covariance estimate
    P: [latent_dim x latent_dim]  updated at each call with forgetting factor lambda.

    RLS update (Sherman-Morrison rank-1 form, batch-averaged):
        z_mean = mean(z, dim=0)           [D]
        k_t    = P_{t-1} z_mean / (lam + z_mean^T P_{t-1} z_mean)
        P_t    = (P_{t-1} - k_t z_mean^T P_{t-1}) / lam
        e_t    = s_mean - W_{t-1} z_mean
        W_t    = W_{t-1} + e_t outer k_t

    This is O(D^2) per step; with D=128, that is 16K ops — negligible vs. ALIF.
    No autograd graph is built; W_rls is always the freshest causal estimate.

    Args:
        latent_dim : input dimensionality D
        out_dim    : output dimensionality K = 4 * num_objects
        forgetting : RLS forgetting factor lambda in (0, 1]
        delta      : initial covariance diagonal (P_0 = delta * I)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        out_dim: int = 4,
        forgetting: float = 0.99,
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.forgetting = forgetting

        # W_rls: [out_dim, latent_dim]  -- the live linear decoder
        self.register_buffer("W_rls", torch.zeros(out_dim, latent_dim))
        # P: [latent_dim, latent_dim]  -- inverse covariance estimate
        self.register_buffer("P_rls", delta * torch.eye(latent_dim))

    @torch.no_grad()
    def update(self, z: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """
        Performs one RLS step and returns the causal prediction (W_{t-1} before update).

        Args:
            z      : [B, D]  current latent state (should be detached)
            s_true : [B, K]  ground-truth kinematics at this timestep

        Returns:
            s_hat  : [B, K]  causal prediction using W_{t-1}
        """
        lam = self.forgetting

        # Causal prediction with current W (before update)
        s_hat = z @ self.W_rls.T  # [B, K]

        # Batch-averaged update
        z_mean = z.mean(0)       # [D]
        s_mean = s_true.mean(0)  # [K]

        # Gain vector: k = P z / (lam + z^T P z)
        Pz = self.P_rls @ z_mean            # [D]
        denom = lam + z_mean.dot(Pz)         # scalar
        k = Pz / denom.clamp(min=1e-6)       # [D]

        # Covariance update (Sherman-Morrison with symmetrization & stability guarding)
        outer_kPz = k.unsqueeze(1) * Pz.unsqueeze(0)   # [D, D]
        self.P_rls = (self.P_rls - outer_kPz) / lam
        self.P_rls = 0.5 * (self.P_rls + self.P_rls.T)
        self.P_rls = torch.clamp(self.P_rls, -100.0, 100.0)

        # Prediction error with current W
        e = s_mean - self.W_rls @ z_mean    # [K]

        # Weight update: W += e outer k (clamped)
        delta_W = e.unsqueeze(1) * k.unsqueeze(0)
        delta_W = torch.clamp(delta_W, -0.5, 0.5)
        self.W_rls = torch.clamp(self.W_rls + delta_W, -10.0, 10.0)

        return s_hat

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Pure inference without updating state. z: [B, D] -> [B, K]"""
        return z @ self.W_rls.T


class PhysicalDecoder(nn.Module):
    """
    Diagnostic Linear Probe for Physical Kinematics (x, y, vx, vy).
    Separated Architecture (v4.3.1):
      - pos_pred = self.pos_head(q_t)           [B, ..., 2 * num_objects]
      - vel_pred = self.vel_head(smooth_p)      [B, ..., 2 * num_objects]
      - decoded  = torch.cat([pos_pred, vel_pred], dim=-1)
    """
    def __init__(
        self,
        latent_dim: int = 128,
        q_dim: Optional[int] = None,
        p_dim: Optional[int] = None,
        num_objects: int = 1,
        alpha: float = 0.85,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        if q_dim is None and p_dim is None:
            self.q_dim = max(1, latent_dim // 4)
            self.p_dim = max(1, latent_dim - self.q_dim)
        elif q_dim is None:
            self.p_dim = min(p_dim, latent_dim - 1)
            self.q_dim = max(1, latent_dim - self.p_dim)
        elif p_dim is None:
            self.q_dim = min(q_dim, latent_dim - 1)
            self.p_dim = max(1, latent_dim - self.q_dim)
        else:
            self.q_dim = q_dim
            self.p_dim = p_dim

        self.num_objects = num_objects
        self.alpha = alpha
        self.out_dim = 4 * num_objects

        self.pos_head = nn.Linear(self.q_dim, 2 * num_objects)
        self.vel_head = nn.Linear(self.p_dim, 2 * num_objects)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, latent_dim] or [B, T, latent_dim]
        if z.shape[-1] >= self.q_dim + self.p_dim:
            q = z[..., :self.q_dim]
            p = z[..., self.q_dim:self.q_dim + self.p_dim]
        else:
            q = z[..., :min(self.q_dim, z.shape[-1])]
            p = z[..., min(self.q_dim, z.shape[-1]):]

        pos_pred = self.pos_head(q)

        if z.dim() == 3:
            B, T, _ = p.shape
            filtered_p = []
            r = torch.zeros(B, self.p_dim, device=z.device, dtype=z.dtype)
            for t in range(T):
                r = self.alpha * r + (1.0 - self.alpha) * p[:, t]
                filtered_p.append(r)
            smooth_p = torch.stack(filtered_p, dim=1)
            vel_pred = self.vel_head(smooth_p)
        else:
            vel_pred = self.vel_head(p)

        return torch.cat([pos_pred, vel_pred], dim=-1)
