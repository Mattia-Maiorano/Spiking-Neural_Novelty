# Architecture Specification: Spiking Predictive World Model (SPWM v4.3)

This document provides a comprehensive technical overview of the current architecture, mathematical modeling, neural components, parameters, data pipeline, and training workflows of the **SPWM-v4.3** (Symplectic Phase Space $(q, p)$ with Spike EMA Velocity Integration) codebase.

---

## 1. System Architecture & Conceptual Flow

SPWM is a biologically grounded, forward-only spiking recurrent world model designed to process continuous spatio-temporal event streams, maintain multi-timescale internal dynamics, and perform autonomous multi-step rollouts with $\mathcal{O}(1)$ computational memory complexity (zero Backpropagation Through Time across sequence unrolling).

```
                      ┌──────────────────────────────────────┐
                      │  2D Continuous Dynamical World       │
                      │  (Ball Kinematics: pos, vel, bounce) │
                      └──────────────────┬───────────────────┘
                                         │ Continuous coordinates
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  Event Camera Simulator              │
                      │  (|ΔL| > θ_event -> ON/OFF Polarities│
                      └──────────────────┬───────────────────┘
                                         │ Event Stream: [B, T, 2, 32, 32]
                                         ▼
                      ┌──────────────────────────────────────┐
                      │  Topological SpatialSoftmax Encoder   │
                      │  Conv2D [16, 32] -> Spatial Softmax  │
                      └───────┬──────────────────────┬───────┘
                              │                      │
                   x_t [128]  │                      │ Keypoints: q_sensor ∈ [-1, 1]^32
                              ▼                      │ (Supervised via L_coord)
               Predictive Error Coding               │
               ┌───────────────────────┐             │
               │  ϵ_t = x_t - W_pred z │             │
               └──────────┬────────────┘             │
                          │                          │
                          ▼                          │
        ┌────────────────────────────────────────────┴───────────────────────────┐
        │  Recurrent Spiking Latent Dynamics (Phase Space z = [q, p] ∈ R^128)   │
        │                                                                        │
        │   Generalized Coordinates q ∈ R^32:                                    │
        │     • Observation mode: q_t = q_sensor,t                               │
        │     • Autonomous mode:  q_{t+1} = clamp(q_t + tanh(W_vel · s̄_p), -1, 1)│
        │                                                                        │
        │   Momentum State p ∈ R^96 (ALIF Population):                           │
        │     • Somatic current: I_soma = W_in ϵ_t + W_rec p_{t-1} + W_q q_{t-1} │
        │     • Dual Timescale Memory Hierarchy:                                 │
        │       - Reactive pool (64 units / 50%): β_adapt = 0.90                 │
        │       - Deep context pool (64 units / 50%): β_adapt = 0.985            │
        │     • Intrinsic Homeostatic Adaptive Threshold:                        │
        │       V_th,t = V_th0 + γ · b_t  (γ = 0.18, V_th0 = 1.0)               │
        │       b_t = β_adapt · b_{t-1} + (1 - β_adapt) · s_{t-1}                │
        │     • Spike EMA Buffer: s̄_{p,t} = β_ema · s̄_{p,t-1} + (1 - β_ema) s_t  │
        │     • Latent p fusion: p_t = LayerNorm(W_spk s_t + W_mem tanh(V_t))    │
        └───────────────────────────────────┬────────────────────────────────────┘
                                            │
                                  z_t = [q_t, p_t] ∈ R^128
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
        ┌───────────────────────────┐               ┌───────────────────────────┐
        │     Latent Predictor      │               │   Physical Decoder Probe  │
        │  ẑ_{t+1} = z_t + Δz(z_t)  │               │   x, y   = pos_head(q_t)  │
        │  (Residual MLP Block)     │               │   vx, vy = vel_head(p_t)  │
        └───────────────────────────┘               └───────────────────────────┘
```

---

## 2. Neural Models & Mathematical Formulations

### 2.1. Spiking Neuron: Adaptive Leaky Integrate-and-Fire (ALIF)
Implemented in [`spwm/models/neurons.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/neurons.py).

The core recurrent units are ALIF cells equipped with dynamic homeostatic thresholds:
1. **Membrane Potential Dynamics**:
   $$V_t = \beta_{\text{mem}} V_{t-1} + I_{\text{soma}, t} - V_{\text{th}, 0} \cdot s_{t-1}$$
   - Soft subtractive reset is utilized.
   - $\beta_{\text{mem}} = 0.80$ corresponds to membrane potential decay constant $\exp(-\Delta t / \tau_m)$.

2. **Threshold Adaptation Dynamics**:
   $$b_t = \beta_{\text{adapt}} b_{t-1} + (1 - \beta_{\text{adapt}}) s_{t-1}$$
   $$V_{\text{th}, t} = V_{\text{th}, 0} + \gamma \cdot b_t$$
   - $V_{\text{th}, 0} = 1.0$ (baseline resting threshold).
   - $\gamma = 0.18$ (threshold adaptation coupling gain).

3. **Spike Generation & Surrogate Gradient**:
   $$s_t = \Theta(V_t - V_{\text{th}, t})$$
   During backward passes or eligibility calculations, the Heaviside step $\Theta(\cdot)$ uses an **Arctan surrogate gradient**:
   $$\psi(v) = \frac{\alpha}{2 (1 + (\frac{\pi}{2} \alpha v)^2)}, \quad \text{with } \alpha = 2.0$$

---

### 2.2. Multi-Timescale Spiking Memory Hierarchy
Implemented in [`spwm/models/memory.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/memory.py).

The momentum ALIF population is partitioned into two distinct functional timescale pools:
- **Reactive Pool (Tier 1)**: $\beta_{\text{adapt}} = 0.90$ (fast adaptation, reactive to rapid trajectory perturbations and bounces).
- **Deep Context Pool (Tier 2)**: $\beta_{\text{adapt}} = 0.985$ (slow adaptation, long-horizon temporal context and momentum memory).
- Filtered spike trains are maintained via exponential moving average (EMA):
  $$\bar{s}_{p, t} = \beta_{\text{ema}} \bar{s}_{p, t-1} + (1 - \beta_{\text{ema}}) s_{p, t}, \quad \text{with } \beta_{\text{ema}} = 0.90$$

---

### 2.3. Symplectic Phase Space Decomposition $z = [q, p]$
Implemented in [`spwm/models/latent_dynamics.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/latent_dynamics.py).

The latent state $z \in \mathbb{R}^{128}$ is split into:
1. **Generalized Position / Coordinate Representation ($q \in \mathbb{R}^{32}$)**:
   - Extracted directly from 16 2D spatial keypoints $(x_k, y_k) \in [-1, 1]^{32}$ via the `SpatialSoftmax` sensory encoder.
   - **Observation Mode**: $q_t = q_{\text{sensor}, t}$.
   - **Autonomous Mode (Rollout)**: Evaluates velocity integration:
     $$q_{t+1} = \text{clamp}(q_t + \tanh(W_{\text{vel}} \cdot \bar{s}_{p, t}), -1.0, 1.0)$$
2. **Generalized Momentum / Internal Dynamical State ($p \in \mathbb{R}^{96}$)**:
   - Driven by the ALIF multi-timescale spiking population:
     $$p_t = \text{LayerNorm}(W_{\text{spk}} s_t + W_{\text{mem}} \tanh(V_t))$$
   - Recurrent somatic current combines sensory predictive error, momentum recurrence, and coordinate state:
     $$I_{\text{soma}, t} = W_{\text{in}} \epsilon_t + W_{\text{rec}} p_{t-1} + W_q q_{t-1}$$

---

### 2.4. Topological SpatialSoftmax Event Encoder
Implemented in [`spwm/models/encoder.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/encoder.py).

Processes incoming 2-channel event frames $E_t \in \mathbb{R}^{2 \times 32 \times 32}$:
- **Convolutional Feature Extractor**: Conv2D layers with channel dims `[16, 32]` and stride 2.
- **Spatial Softmax**: Computes normalized expected center-of-mass coordinates for 16 feature keypoints:
  $$q_{\text{sensor}, k} = \sum_{u, v} \text{softmax}_{u, v}(F_k(u, v)) \cdot (u, v)^T \in [-1, 1]^2$$
- **Linear Projection**: Maps flattened convolutional features to sensory embedding $x_t \in \mathbb{R}^{128}$.

---

### 2.5. Latent Predictor & Physical Probes
- **Latent Predictor** ([`spwm/models/predictor.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/predictor.py)):
  $$\hat{z}_{t+1} = z_t + \text{MLP}(z_t)$$
  Two-layer MLP with LayerNorm, GELU activations, and hidden dimension 256.
- **Physical Decoder Probes** ([`spwm/models/world_model.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/world_model.py)):
  - **Position Head**: Linear probe $q_t \in \mathbb{R}^{32} \to (x, y) \in \mathbb{R}^2$.
  - **Velocity Head**: Linear probe $p_t \in \mathbb{R}^{96} \to (v_x, v_y) \in \mathbb{R}^2$.

---

## 3. Learning Algorithm & Loss Formulation

### 3.1. Online Forward-Only $e$-prop Learning
Implemented in [`spwm/learning/trainer.py`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/learning/trainer.py).

The recurrent spiking latent dynamics are updated via forward-only local synaptic plasticity without Backpropagation Through Time (BPTT):
- **Eligibility Traces**: Maintain local presynaptic and postsynaptic activity traces.
- **Learning Signal**:
  $$L_t = \epsilon_t + \lambda_{\text{kin\_feedback}} \cdot \text{Grad}_{\text{probe}}$$
- **Synaptic Weight Updates**:
  $$\Delta W_{\text{in}} = \frac{1}{B \cdot T} L_t^T x_t, \quad \Delta W_{\text{rec}} = \frac{1}{B \cdot T} L_t^T p_{t-1}, \quad \Delta W_q = \frac{1}{B \cdot T} L_t^T q_{t-1}$$
  Applied with local learning rate $\eta_{\text{local}} = 0.0002$ and gradient clipping $[-0.1, 0.1]$.

### 3.2. Total Loss Objective
$$\mathcal{L}_{\text{total}} = \lambda_{\text{pred}} \mathcal{L}_{\text{pred}} + \lambda_{\text{multi}} \mathcal{L}_{\text{multi}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}} + \lambda_{\text{probe}} \mathcal{L}_{\text{probe}} + \lambda_{\text{coord}} \mathcal{L}_{\text{coord}}$$

Where:
- $\mathcal{L}_{\text{pred}} = \frac{1}{T} \sum_{t=1}^{T} \|\hat{z}_t - z_t\|^2$ (1-step next-latent prediction error)
- $\mathcal{L}_{\text{multi}}$: Multi-step rollout consistency loss ($H=3$)
- $\mathcal{L}_{\text{var}} = \max(0, \sigma_{\text{target}} - \text{std}(z))$ (Latent variance collapse prevention)
- $\mathcal{L}_{\text{sparse}} = (\bar{s} - s_{\text{target}})^2$ (Spike rate regularization)
- $\mathcal{L}_{\text{probe}} = \text{MSE}(\hat{y}_{\text{kin}}, y_{\text{kin}})$ (Physical kinematic probe loss)
- $\mathcal{L}_{\text{coord}} = \text{MSE}(q_{\text{sensor}}, \text{coords}_{\text{true}})$ (Topological keypoint supervision)

---

## 4. Default Configuration Reference (`spwm_v4_3.yaml`)

```yaml
project:
  name: "spwm_v4_3"
  seed: 42

environment:
  num_objects: 1
  height: 32
  width: 32
  event_threshold: 0.08

data:
  total_trajectories: 600
  train_split: 0.8
  val_split: 0.1
  sequence_length: 150
  batch_size: 32

model:
  type: "spwm"
  encoder_type: "spatial_softmax"
  num_keypoints: 16
  encoder_dim: 128
  encoder_conv_channels: [16, 32]
  latent_dim: 128        # total = q_dim + p_dim (32 + 96)
  q_dim: 32
  p_dim: 96
  predictor_hidden_dim: 256
  local_lr: 0.0002
  lambda_kin_feedback: 1.0
  ema_decay: 0.9        # EMA decay for spike buffer
  rls_enabled: false

neuron:
  model: "alif"
  threshold: 1.0
  beta_mem: 0.80
  gamma: 0.18
  surrogate: "atan"
  surrogate_alpha: 2.0

memory:
  num_timescales: 2
  timescale_dims: [64, 64]
  betas: [0.90, 0.985]

training:
  epochs: 40
  learning_rate: 0.0002
  probe_lr: 0.0005
  probe_weight_decay: 0.01
  weight_decay: 0.0001
  batch_size: 32
  grad_clip_norm: 1.0

loss:
  lambda_pred: 1.0
  lambda_multi: 0.5
  lambda_var: 0.1
  lambda_sparse: 0.001
  lambda_probe: 2.0
  lambda_coord: 0.15

evaluation:
  save_best_metric: "val_pos_err"
  rollout_horizons: [1, 5, 10, 25, 50]
```
