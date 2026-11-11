# SPWM Model & Prediction Architecture

This document specifies the exact architecture, data ingestion pipeline, network dimensions, layer configurations, input/output tensor contracts, and mathematical state transitions implemented in `spwm/models/`.

---

## 1. Top-Level Architectural Overview

The core model is `SPWM` (defined in `spwm/models/world_model.py` and exported through `spwm/models/spwm.py`). It implements a dual-stream neuromorphic world model structured into six specialized sub-modules:

1. **Dual-Camera Visual Fusion Event Encoder** (`EventEncoder`, `SpatialSoftmax`)
2. **Predictive Sensory Decoder / Top-Down Prediction** (`sensory_predictor`)
3. **Recurrent Spiking Latent Dynamics with ALIF Memory Hierarchy** (`SpikingLatentDynamics`, `MultiTimescaleMemory`, `ALIFCell`)
4. **Action-Conditioned Latent Attractor Predictor (M1)** (`LatentPredictor`)
5. **Articulated Joint Kinematic Physical Decoder** (`PhysicalDecoder`)
6. **Latent Terminal Cost-to-Go Critic Network** (`LatentCritic`)

In addition, an exact differentiable kinematic solver (`DifferentiableForwardKinematics`) computes forward end-effector kinematics in Cartesian space.

### Tensor Shape Contracts
- Temporal Event Sequence: `[B, T, C_in, H, W]` where default `C_in = 4`, `H = 32`, `W = 32`.
- Single-Step Event Frame: `[B, C_in, H, W]`.
- Action Tensor: `[B, T, action_dim]` or `[B, action_dim]` where `action_dim = 6`.
- Latent State Vector: `z_t = [q_t, p_t] \in \mathbb{R}^{128}` partitioned into:
  - Coordinate Subspace: `q_t \in \mathbb{R}^{32}` (`q_dim = 32`)
  - Generalized Momentum Subspace: `p_t \in \mathbb{R}^{96}` (`p_dim = 96`)
- Spatial Keypoints: `\mathbb{R}^{32}` (`2 * num_keypoints` with `num_keypoints = 16`).
- Decoded Kinematics: `\mathbb{R}^{12}` (`[q^*, p^*] \in \mathbb{R}^{6} \times \mathbb{R}^{6}`).

---

## 2. Component Specifications & Mathematical Formulations

### 2.1 Dual-Camera Visual Fusion Event Encoder (`spwm/models/encoder.py`)

#### Purpose
Ingests dual-stream event frames consisting of static global workspace events and egocentric wrist-mounted events. Fuses multi-view representations, computes feature heatmaps, and extracts continuous 2D coordinates of spatial centers of mass (keypoints) via a differentiable spatial softmax bottleneck.

#### Layer Configuration & Parameters
- **Input Channels**: `4` (Channels 0–1: static global camera ON/OFF polarities; Channels 2–3: egocentric wrist camera ON/OFF polarities).
- **Global / Primary Convolutional Backbone (`conv`)**:
  1. `Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1)` $\rightarrow$ Output `[B, 16, 16, 16]` (592 params)
  2. `GroupNorm(num_groups=4, num_channels=16)` (32 params)
  3. `SiLU()`
  4. `Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)` $\rightarrow$ Output `[B, 32, 8, 8]` (4,640 params)
  5. `GroupNorm(num_groups=4, num_channels=32)` (64 params)
  6. `SiLU()`
  7. `Conv2d(in_channels=32, out_channels=16, kernel_size=1)` $\rightarrow$ Output `[B, 16, 8, 8]` (528 params)
- **Egocentric Wrist Branch (`wrist_stem`)**:
  1. `Conv2d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1)` $\rightarrow$ Output `[B, 16, 16, 16]` (304 params)
  2. `GroupNorm(num_groups=4, num_channels=16)` (32 params)
  3. `SiLU()`
  4. `Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)` $\rightarrow$ Output `[B, 32, 8, 8]` (4,640 params)
  5. `GroupNorm(num_groups=4, num_channels=32)` (64 params)
  6. `SiLU()`
- **Fusion Head (`fusion_head`)**:
  - `Conv2d(in_channels=32, out_channels=16, kernel_size=1)` $\rightarrow$ Output `[B, 16, 8, 8]` (528 params)
  - Fusion operation: $\Phi_{\text{fused}} = \text{conv}(x) + \text{fusion\_head}(\text{wrist\_stem}(x_{2:4}))$.
- **Spatial Softmax Bottleneck (`SpatialSoftmax`)**:
  - Grid: `H = 8`, `W = 8`.
  - Normalized coordinate grid: $x \in [-1, 1]$, $y \in [-1, 1]$.
  - Softmax Attention over flattened feature map ($8 \times 8 = 64$):
    $$A_{c, i, j} = \frac{\exp(F_{c, i, j} / \tau)}{\sum_{u, v} \exp(F_{c, u, v} / \tau)}, \quad \tau = 1.0$$
  - Analytical Keypoints:
    $$\mu_{c}^x = \sum_{i, j} A_{c, i, j} \cdot X_{i, j}, \quad \mu_{c}^y = \sum_{i, j} A_{c, i, j} \cdot Y_{i, j}$$
    Concatenated into `[B, 2 * num_keypoints] = [B, 32]`.
- **Projection MLP (`proj`)**:
  1. `Linear(in_features=32, out_features=128)` (4,224 params)
  2. `LayerNorm(128)` (256 params)
  3. `SiLU()`
  - Outputs sensory embedding $s_t \in \mathbb{R}^{128}$ (`encoder_dim = 128`).

---

### 2.2 Predictive Sensory Decoder (`sensory_predictor`)

#### Architecture & Dimensions
- Single linear projection: `Linear(in_features=128, out_features=128, bias=False)` (16,384 params).
- Input: Prior latent state $z_{t-1} \in \mathbb{R}^{128}$.
- Output: Top-down predicted sensory representation $\hat{s}_t \in \mathbb{R}^{128}$.
- Predictive Coding Residual:
  $$\epsilon_t = s_t - \hat{s}_t$$
  where $s_t$ is the feedforward embedding from the EventEncoder and $\epsilon_t \in \mathbb{R}^{128}$ is injected into the recurrent spiking dynamics.

---

### 2.3 Recurrent Spiking Latent Dynamics with ALIF Memory (`spwm/models/dynamics.py`, `memory.py`, `neurons.py`)

#### Purpose
Maintains continuous state assimilation by combining prediction error $\epsilon_t$, previous generalized momentum $p_{t-1}$, previous generalized coordinates $q_{t-1}$, and applied action $u_t$ through a two-timescale Adaptive Leaky Integrate-and-Fire (ALIF) network.

#### Population Partitioning & Parameters
- **Timescale Tiers**: 2 tiers (`timescale_dims = (128, 128)`), total memory dimension $N_{\text{mem}} = 256$.
- **Membrane Decay Constant**: $\beta_{\text{mem}} = 0.80$.
- **Adaptation Decay Constants**: $\beta_{\text{adapt}} = (0.90, 0.985)$ (Fast tier: $\beta_1 = 0.90$; Slow tier: $\beta_2 = 0.985$).
- **Baseline Threshold**: $V_{\text{th0}} = 1.0$.
- **Threshold Adaptation Coupling**: $\gamma = 0.18$.
- **Surrogate Gradient Function**: Arctangent (`atan`) with $\alpha = 2.0$:
  $$S(x) = \Theta(x), \quad \frac{\partial S}{\partial x} = \frac{\alpha / 2}{1 + \left(\frac{\pi}{2} \alpha x\right)^2}$$

#### ALIF Neuron Dynamics (`ALIFCell`)
For each neuron $j$ at time step $t$:
$$V_j(t) = \beta_{\text{mem}} V_j(t-1) + I_j(t) - z_j(t-1) V_{\text{th0}}$$
$$A_j(t) = \beta_{\text{adapt}} A_j(t-1) + z_j(t-1)$$
$$V_{\text{th}, j}(t) = V_{\text{th0}} + \gamma A_j(t)$$
$$z_j(t) = \text{SurrogateSpike}(V_j(t) - V_{\text{th}, j}(t))$$

#### Input Current Integration
$$I_t = W_{\text{in}} \epsilon_t + W_{\text{rec}} p_{t-1} + W_q q_{t-1} + W_u u_t$$
- `input_proj`: `Linear(128, 256, bias=True)` (33,024 params)
- `recurrent_proj`: `Linear(96, 256, bias=False)` (24,576 params)
- `q_proj`: `Linear(32, 256, bias=False)` (8,192 params)
- `action_proj`: `Linear(6, 256, bias=False)` (1,536 params, zero-initialized)

> [!NOTE] Action Projection Initialization & Training Evolution ($W_u$)
> Sebbene $W_u$ sia esplicitamente azzerato all'inizializzazione (`nn.init.zeros_(self.action_proj.weight)` con norma 0.0 vs $\sim 1.34$ di `input_proj`), durante il training di Phase 0 il gradiente retropropagato tramite la dinamica latente ne aumenta la norma portandola a $\approx 1.365$ (rapporto $\approx 1.02$ rispetto a $W_{\text{in}}$ su run completi quali `spwm_v0_10_4`). Questo previene che in closed-loop il modello dipenda unicamente dagli eventi sensoriali ritardati della telecamera, garantendo consapevolezza immediata della coppia motoria applicata.

#### Momentum Subspace Fusion
Analog membrane potentials and discrete spikes are combined:
$$m_{\text{analog}} = \tanh(V_t) \in \mathbb{R}^{256}$$
$$p_t = \text{LayerNorm}(W_{\text{spk}} z_t + W_{\text{mem}} m_{\text{analog}})$$
- `fuse_spikes`: `Linear(256, 96, bias=True)` (24,672 params)
- `fuse_mems`: `Linear(256, 96, bias=False)` (24,576 params)
- `norm`: `LayerNorm(96)` (192 params)
- Exponential Moving Average of spikes:
  $$\bar{z}_t = \lambda_{\text{ema}} \bar{z}_{t-1} + (1 - \lambda_{\text{ema}}) z_t, \quad \lambda_{\text{ema}} = 0.90$$

#### Coordinate Subspace & Anti-Evanescence Gating
Generalized coordinate update maps momentum to coordinate velocity via a direct slice of the first 32 dimensions $v_p = \text{vel}(p_t) = p_t[0:32]$ (strictly aligned with Phase 0 and $\mathcal{L}_{\text{derivative}}$ without parameter projection layers):
- Event Activity: $a_t = \text{mean}(|E_t|)$.
- Dynamic Blending Factor $K_{\text{gate}}$:
  $$a_{\text{eff}} = \max(0, a_t - \theta_{\text{act}}), \quad \theta_{\text{act}} = 0.001$$
  $$K_{\text{gate}} = \left(1 - \exp\left(-\text{softplus}(w_{\text{gate}}) \cdot a_{\text{eff}}\right)\right) \cdot K_{\text{max}}$$
  where $w_{\text{gate}} \in \mathbb{R}^{32}$ initialized to 50.0.
- Step Update:
  $$q_t = q_{t-1} + \Delta t \cdot v_p + K_{\text{gate}} \odot \text{clip}\left(k_t^{\text{sensory}} - q_{t-1}, \|\cdot\|_2 \le \delta_{\text{max}}\right)$$
  During autonomous rollout without sensory keypoints, $q_t = q_{t-1} + \Delta t \cdot v_p$.
- Reconstructed Latent State:
  $$z_t = [q_t, p_t] \in \mathbb{R}^{128}$$

---

### 2.4 Action-Conditioned Latent Attractor Predictor (M1) (`spwm/models/predictors.py`)

#### Purpose
Autonomous forward simulation in latent phase space without raw visual sensory inputs. Unrolls dynamics conditioned on planned motor actions $u_t \in [-1, 1]^6$.

#### Architectural Formulation
Relaxes conservative Hamiltonian dynamics into a learnable symplectic integrator capable of both dissipative damping and active restorative forces:
1. Concatenation: $[z_t, u_t] \in \mathbb{R}^{134}$.
2. Shared Two-Layer MLP Backbone:
   - `Linear(134, 256)` (34,560 params) $\rightarrow$ `LayerNorm(256)` (512 params) $\rightarrow$ `SiLU()`
   - `Linear(256, 256)` (65,792 params) $\rightarrow$ `LayerNorm(256)` (512 params) $\rightarrow$ `SiLU()`
3. Three Specialized Linear Prediction Heads:
   - **Force Head (`force_head`)**: `Linear(256, 96)` $\rightarrow f_p \in \mathbb{R}^{96}$ (24,672 params).
   - **Dissipation Head (`dissipation_head`)**: `Linear(256, 96)` $\rightarrow \gamma_t = \sigma(\text{Linear}(h)) \in (0, 1)^{96}$ (24,672 params, bias initialized to -2.0).
   - **Restorative Coordinate Head (`restorative_q_head`)**: `Linear(256, 32)` $\rightarrow f_q \in \mathbb{R}^{32}$ (8,224 params, weights and bias zero-initialized).

#### Symplectic Integration Update
$$p_{t+1} = (1 - \Delta t \cdot \gamma_t) \odot p_t + f_p$$
$$q_{t+1} = q_t + \Delta t \cdot \left(p_{t+1}[0:32] + f_q\right)$$
$$\hat{z}_{t+1} = [q_{t+1}, p_{t+1}] \in \mathbb{R}^{128}$$

---

### 2.5 Articulated Physical Decoder (`spwm/models/predictors.py`)

#### Purpose
Decodes abstract latent phase space $z_t = [q_t, p_t]$ directly into physical robotic joint angles $q^* \in \mathbb{R}^6$ and joint angular velocities $p^* \in \mathbb{R}^6$.

#### Architecture
- **Joint Position Head (`pos_head`)**: Ingests $q_t \in \mathbb{R}^{32}$:
  1. `Linear(32, 64)` (2,112 params)
  2. `LayerNorm(64)` (128 params)
  3. `SiLU()`
  4. `Linear(64, 6)` (390 params)
  - Output: $\hat{q}^* \in \mathbb{R}^6$.
- **Joint Velocity Head (`vel_head`)**: Ingests $p_t \in \mathbb{R}^{96}$:
  1. `Linear(96, 64)` (6,208 params)
  2. `LayerNorm(64)` (128 params)
  3. `SiLU()`
  4. `Linear(64, 6)` (390 params)
  - Output: $\hat{p}^* \in \mathbb{R}^6$.
- Decoded Kinematics Output:
  $$\hat{k}_t = [\hat{q}^*, \hat{p}^*] \in \mathbb{R}^{12}$$

---

### 2.6 Latent Terminal Value Critic Network (`spwm/models/predictors.py`)

#### Purpose
Estimates the discounted cost-to-go $V(z) \in \mathbb{R}$ directly from latent state $z \in \mathbb{R}^{128}$ for terminal value bootstrapping in MPPI planning.

#### Architecture
- Feedforward MLP (`net`):
  1. `Linear(128, 256)` (33,024 params)
  2. `SiLU()`
  3. `Linear(256, 128)` (32,896 params)
  4. `SiLU()`
  5. `Linear(128, 1)` (129 params)
- Total Parameters: 66,049.

---

### 2.7 Differentiable Forward Kinematics (`spwm/models/kinematics.py`)

#### Purpose
Computes exact 3D Cartesian coordinates $(x, y, z)$ and $4 \times 4$ homogeneous transformation matrices $T_{0:6}$ of the manipulator arm links and end-effector tip from joint angles $q^* \in \mathbb{R}^6$.

#### Robot Arm Kinematic Constants
- Shoulder Mount Height: $1.25\,\text{m}$
- Base Height: $0.05\,\text{m}$
- Turret Height: $0.05\,\text{m}$
- Upper Arm Length: $0.28\,\text{m}$
- Forearm Length: $0.25\,\text{m}$
- Wrist Pitch Length: $0.04\,\text{m}$
- Wrist Yaw Length: $0.04\,\text{m}$
- Wrist Roll Length: $0.06\,\text{m}$
- End-Effector Tip Offset: $(0.0, 0.06, 0.0)$ along Wrist Roll $+Y$.

#### Canonical Joint Axes
- Joint 0 (Shoulder Pan): $(0.0, 1.0, 0.0)$
- Joint 1 (Shoulder Lift): $(1.0, 0.0, 0.0)$
- Joint 2 (Elbow Flex): $(0.5375313, -0.2756374, -0.7969221)$
- Joint 3 (Wrist Pitch): $(0.5375313, -0.2756374, -0.7969221)$
- Joint 4 (Wrist Yaw): $(0.0, 0.0, -1.0)$
- Joint 5 (Wrist Roll): $(0.0, 1.0, 0.0)$

---

## 3. Parameter Count Summary

| Module | Submodule / Layer | Parameter Count |
|---|---|---|
| **EventEncoder** | Convolutional Trunk (`conv`) | 5,856 |
| | Egocentric Wrist Stem (`wrist_stem`) | 5,072 |
| | Cross-View Fusion Head (`fusion_head`) | 528 |
| | Linear Projection (`proj`) | 4,480 |
| **sensory_predictor** | Linear Projection | 16,384 |
| **SpikingLatentDynamics** | Input & Recurrent Projections | 67,328 |
| | Momentum Fusion (`fuse_spikes`, `fuse_mems`, `norm`) | 49,440 |
| | Velocity Projections & Gate Parameter | 11,296 |
| **LatentPredictor (M1)** | Backbone MLP | 101,376 |
| | Output Heads (`force`, `dissipation`, `restorative_q`) | 57,568 |
| **PhysicalDecoder** | Position Head (`pos_head`) | 2,630 |
| | Velocity Head (`vel_head`) | 6,726 |
| **LatentCritic** | Value MLP (`net`) | 66,049 |
| **Total SPWM Parameters** | | **394,733** |

