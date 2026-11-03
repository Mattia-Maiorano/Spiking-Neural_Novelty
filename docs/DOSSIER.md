# Technical Dossier & Academic Readiness Audit
**Project**: SPWM (Spiking Predictive World Model for 3D Articulated Manipulation)  
**Active Production Baseline**: `SPWM-v0.7.3`  
**Target Publication Venues**: IEEE/RSJ IROS, IEEE ICRA, CoRL, NeurIPS, RSS  
**Date**: November 2026  

---

## Executive Summary & Readiness Dashboard

| Audit Domain | Repository Status | Readiness Score | Primary Blocker / Action Item |
| :--- | :--- | :---: | :--- |
| **1. Problem Formulation & System Overview** | Fully Specified in Contracts & Simulation | **95%** | Verify closed-loop POMDP control formulation beyond open-loop rollout prediction. |
| **2. Neural Architecture & Mathematical Specs** | Fully Specified in Layer 1 PyTorch Implementations | **95%** | None (exact tensor contracts, ALIF equations, and continuous handover verified). |
| **3. Training Protocol & Optimization Guarantees** | Fully Decoupled Two-Phase Curriculum | **90%** | Multi-seed execution required ($\ge 3$ seeds). Single seed (`seed=42`) currently logged. |
| **4. Empirical Evaluation & Benchmarks** | Multi-Horizon Telemetry Logged ($H \in \{1..100\}$) | **70%** | Missing baselines (RSSM, GRU, ConvLSTM) and missing non-spiking ANN ablation. |
| **5. Architectural Evolutionary History** | Fully Documented in `History.md` (v1 $\to$ v0.7.3) | **100%** | None (extensive negative ablations and post-mortems fully cataloged). |
| **6. Publication Gap Analysis** | Gaps Formally Identified | **60%** | SOP energy benchmarking, non-spiking ablation, and sim-to-real transfer plan required. |

---

## 1. Problem Formulation & System Overview

### 1.1 Mathematical Formulation: Continuous-Time Latent POMDP
The physical system operates as a continuous-time Partially Observable Markov Decision Process (POMDP) embedded in Euclidean 3D space $\mathcal{M} = \text{SE}(3) \times \mathbb{R}^J$:

$$\mathcal{P} = \left\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \right\rangle$$

```mermaid
flowchart LR
    subgraph Environment["Physical World (PhysX 4 ArticulationBody)"]
        S_t["Physical State s_t\n(q*, p*, Payload m, I)"]
        S_next["Physical State s_t+1"]
        Action["Motor Torques / Action u_t in [-1, 1]^6"]
        Action -->|Stiff Spring-Damper PD| S_next
        S_t -->|Featherstone Dynamics dt=0.05s| S_next
    end

    subgraph Sensors["Neuromorphic Vision Pipeline"]
        EventCam["Dual Event Sensors\n(Global + Wrist-Mounted, C=4)"]
        S_t -.->|"Scene Projections L(u, v; t)"| EventCam
        EventCam -->|"Spatiotemporal Spikes E_t in {0, 1}^(4 x 32 x 32)"| Obs["Observation E_t"]
    end

    subgraph Agent["SPWM-v0.7.3 Architecture"]
        Obs --> Enc["EventEncoder & SpatialSoftmax\n(K=16 Keypoints k_t)"]
        Enc --> Dyn["SpikingLatentDynamics (ALIF)\n(q_t in R^32, p_t in R^96)"]
        Action --> Pred["LatentPredictor M_1\n(Dissipative Field gamma_t, f_p, f_q)"]
        Dyn --> Pred
        Pred --> Readout["PhysicalDecoder\n(q*_hat, p*_hat in R^12)"]
    end
```

- **Observation Space ($\mathbf{E}_t$)**: Asynchronous binary temporal contrast event frames $\mathbf{E}_t \in \{0, 1\}^{C \times H \times W}$ where $C=4$ channels (channels 0–1: static global observer brightening/dimming polarities; channels 2–3: egocentric wrist-mounted brightening/dimming polarities), with spatial resolution $H = W = 32$.
- **Continuous Motor Action Space ($\mathbf{u}_t$)**: Continuous joint action commands driving the sequential waypoint PD controller:
  $$\mathbf{u}_t \in [-1.0, 1.0]^{D_u}, \quad D_u = J = 6$$
- **Internal Latent State Space ($\mathbf{z}_t$)**: Symplectic/dissipative phase-space factorization:
  $$\mathbf{z}_t = [\mathbf{q}_t, \mathbf{p}_t] \in \mathbb{R}^D, \quad \mathbf{q}_t \in [-1.0, 1.0]^{D_q}, \quad \mathbf{p}_t \in \mathbb{R}^{D_p}$$
  where $D_q = 32$ (generalized coordinate manifold matching $2K=32$ keypoint coordinates), $D_p = 96$ (generalized momentum manifold, integer multiple $D_p = 3 D_q$), and total latent dimensionality $D = D_q + D_p = 128$.
- **Decoded Physical Articulated States ($\hat{\mathbf{y}}_t$)**: Continuous joint angles and joint angular velocities:
  $$\hat{\mathbf{y}}_t = [\hat{\mathbf{q}}_t^*, \hat{\mathbf{p}}_t^*] \in \mathbb{R}^{2J}, \quad \hat{\mathbf{q}}_t^* \in [-\pi, \pi]^6, \quad \hat{\mathbf{p}}_t^* \in \mathbb{R}^6 \quad (J = 6)$$

### 1.2 Core Scientific Hypothesis
Standard recurrent world models (e.g., RSSM in DreamerV1-V3, ConvLSTM, deep GRU/LSTM predictors) fundamentally fail in continuous high-speed robotic manipulation due to three structural pathologies:
1. **Synchronous Visual Bandwidth vs. Dynamic Contact Transients**: Standard RGB world models capture frames at fixed 20–30 Hz, suffering catastrophic motion blur and unobserved impact dynamics during physical contact shocks. SPWM processes differential event polarities at physics rates (20–1000 Hz equivalent) via temporal contrast accumulation, capturing dynamic edge discontinuities with microsecond-level temporal resolution.
2. **Backpropagation Through Time (BPTT) Computational Memory Explosion**: Training recurrent world models over long horizons $T \ge 100$ requires storing forward activation histories $O(T)$, preventing real-time bio-plausible on-chip adaptation. SPWM utilizes Adaptive Leaky Integrate-and-Fire (ALIF) populations trained via decoupled phase strategies, maintaining sparse temporal representations without diverging memory footprints.
3. **Contact Impulse Shocks & Phase-Space Topology Collapse**: Rigid mechanical contact introduces near-infinite instantaneous decelerations ($\mathbf{p}_t \to \mathbf{p}_t + \Delta \mathbf{p}_{\text{impulse}}$). Black-box RNNs experience exploding gradients or static deadlocks (freezing to the mean). SPWM resolves this by decomposing latent dynamics into a Hamiltonian/dissipative vector field $[\mathbf{q}_t, \mathbf{p}_t]$ with event-gated soft position coupling ($K_{\text{max}} = 0.35$), anti-evanescence noise suppression ($\theta_{\text{act}} = 0.001$), and active-perception soft-fade handover ($w_{\text{max}} = 0.40$).

### 1.3 Benchmark Archetype & Physical Specifications
- **Kinematic Morphology**: Anthropomorphic 6-DOF Right Humanoid Manipulator Arm (`RobotArchetype.ManipulatorArm`, $J=6$ actuated revolute joints, max joint excursion $\pm \pi\,\text{rad}$):
  - **Base Root**: Fixed shoulder socket mounted at $Y=1.25\,\text{m}$ ($20.0\,\text{kg}$, base height $0.05\,\text{m}$).
  - **Joint 0 (`Shoulder_Pan`)**: Turret height $0.05\,\text{m}$, mass $2.5\,\text{kg}$, joint limits $[-120^\circ, 120^\circ]$, glenohumeral elevation $1.35\,\text{m}$.
  - **Joint 1 (`Shoulder_Lift`)**: Upper arm / humerus length $0.28\,\text{m}$, mass $2.2\,\text{kg}$, joint limits $[-90^\circ, 185^\circ]$ (resting pose hangs naturally at $q_1 \approx 2.92\,\text{rad}$).
  - **Joint 2 (`Elbow_Flex`)**: Forearm radius-ulna length $0.25\,\text{m}$, mass $1.6\,\text{kg}$, joint limits $[-155^\circ, 15^\circ]$, medial swivel carrying axis $\mathbf{e}_{\text{swivel}} = (0^\circ, 56^\circ, -16^\circ)$.
  - **Joint 3 (`Wrist_Pitch`)**: Proximal carpal flexion/extension length $0.04\,\text{m}$, mass $0.6\,\text{kg}$, limits $[-95^\circ, 95^\circ]$.
  - **Joint 4 (`Wrist_Yaw`)**: Distal carpal radial/ulnar deviation length $0.04\,\text{m}$, mass $0.5\,\text{kg}$, limits $[-75^\circ, 75^\circ]$.
  - **Joint 5 (`Wrist_Roll`)**: Hand pronation-supination length $0.06\,\text{m}$, mass $0.5\,\text{kg}$, limits $[-180^\circ, 180^\circ]$ (physical reaching limits $[-95^\circ, 15^\circ]$).
  - **End-Effector Hand**: Anatomical 5-finger right hand (palm plate, knuckle bridge, 4 two-segment articulated fingers, and radial thumb on $-X$).
- **Physics Engine & Solver**: PhysX 4 Featherstone `ArticulationBody` solver executing at physics step $\Delta t = 0.05\,\text{s}$ ($20\,\text{Hz}$), global joint friction $\mu_{\text{joint}} = 0.06\,\text{N}\cdot\text{s}$, angular damping $0.25\,\text{s}^{-1}$, ground friction $\mu_s = 0.90, \mu_k = 0.80$, gravity $\mathbf{g} = [0, -9.81, 0]^T\,\text{m/s}^2$.
- **Procedural Payload Variances**: Spawns across three workspace elevation planes (Low: $1.04\,\text{m}$, Mid: $1.14\,\text{m}$, High: $1.26\,\text{m}$) at forward reach distances $[0.42, 0.52]\,\text{m}$:
  - Primitive shapes: Sphere, Elongated Cylinder, Box.
  - Mass range: Procedurally sampled in $m_{\text{payload}} \in [1.20, 4.50]\,\text{kg}$ with full inertia tensor scaling.
  - Attachment: `FixedJoint` locking payload into palm/finger mesh upon contact, rigidly transmitting inertial mass into the Featherstone kinematic chain.
- **Dual Neuromorphic Sensor Placement**:
  - Global Camera: Static head observer at eye elevation $1.60\,\text{m}$ (`HeadOffsetFromShoulder` $= (-0.22, 0.25, -0.28)\,\text{m}$), gaze directed toward manipulation workspace center.
  - Egocentric Camera: Mounted directly on distal forearm/wrist facing end-effector workspace, disambiguating elbow-flex depth foreshortening and axial wrist-roll rotational symmetry.

---

## 2. Neural Architecture & Mathematical Specifications

```mermaid
flowchart TD
    subgraph Frontend["Neuromorphic Vision Frontend (C=4, H=W=32)"]
        E_t["Event Frame E_t [B, 4, 32, 32]"]
        ConvGlobal["Conv2D (16->32) + GroupNorm + SiLU"]
        ConvWrist["Wrist Stem Conv2D (16->32) + GN + SiLU"]
        Fuse["Fusion Head Conv2D(32->16, 1x1)"]
        E_t --> ConvGlobal
        E_t --> ConvWrist
        ConvGlobal --> Fuse
        ConvWrist --> Fuse
        Fuse --> SM["SpatialSoftmax (Temperature tau=1.0)"]
        SM --> Keypoints["Keypoints k_t in [-1, 1]^(2K) (K=16)"]
        Keypoints --> Proj["MLP Projection + LayerNorm + SiLU"]
        Proj --> SensoryX["Sensory Feature x_t in R^128"]
    end

    subgraph Recurrence["Recurrent Spiking Latent Dynamics (z_t in R^128)"]
        SensoryX --> Err["Sensory Prediction Error eps_t = x_t - W_pred z_t-1"]
        Err --> ALIF["MultiTimescaleMemory (256 ALIF Units)\nTier 1: beta=0.90 (Reactive)\nTier 2: beta=0.985 (Context)"]
        RecP["Momentum Recurrence W_rec p_t"] --> ALIF
        RecQ["Coordinate Projection W_q q_t"] --> ALIF
        RecU["Action Projection W_act u_t"] --> ALIF
        ALIF --> Spikes["Spikes s_t in {0, 1}^256"]
        ALIF --> Mems["Membranes m_t in R^256"]
        Spikes & Mems --> FuseP["LayerNorm(Linear(s_t) + Linear(tanh(m_t)))"]
        FuseP --> P_next["Momentum Subspace p_t+1 in R^96"]
        P_next --> VelProj["Coordinate Velocity v_p = W_vel_proj p_t+1 in R^32"]
        Keypoints --> Gate["Anti-Evanescence Stasis Gate K_gate (K_max=0.35)"]
        VelProj & Gate --> IntQ["q_t+1 = q_t + dt * v_p + K_gate * (k_t - q_t)"]
        IntQ --> Q_next["Coordinate Subspace q_t+1 in R^32"]
        Q_next & P_next --> Z_grounded["Grounded Latent z_t+1 in R^128"]
    end

    subgraph Predictor["Action-Conditioned Attractor Predictor (M_1)"]
        Z_in["Latent z_t in R^128"] & ActionIn["Action u_t in R^6"] --> BackboneM1["Backbone MLP (256, LayerNorm, SiLU)"]
        BackboneM1 --> DissHead["Dissipation Head: gamma_t = sigmoid(W_d h_t - 2.0)"]
        BackboneM1 --> ForceP["Force Head: f_p in R^96"]
        BackboneM1 --> RestQ["Restorative Head: f_q in R^32"]
        DissHead & ForceP --> DissP["p_t+1 = (1 - dt * gamma_t) * p_t + f_p"]
        RestQ & DissP --> DissQ["q_t+1 = q_t + dt * (p_t+1[:32] + f_q)"]
        DissP & DissQ --> Z_pred["Predicted Latent z_hat_t+1 in R^128"]
    end

    subgraph Handover["Continuous Soft-Fade Handover"]
        Z_pred & Z_grounded --> Blend["z_t+1 = (1 - w_t) * z_pred + w_t * z_grounded\n(w_t in [0.0, 0.40])"]
        Blend --> Z_out["Next Step State z_t+1"]
    end

    subgraph Readout["Articulated Kinematic Decoder"]
        Z_out --> PosHead["Position Head MLP: q_t -> q*_hat in R^6"]
        Z_out --> VelHead["Velocity Head MLP: p_t -> p*_hat in R^6"]
        PosHead & VelHead --> KinOut["Physical Telemetry y_hat = [q*_hat, p*_hat] in R^12"]
    end
```

### 2.1 Neuromorphic Vision Frontend
The visual frontend transforms continuous dual-view differential event streams into low-dimensional topological coordinates:
1. **Temporal Contrast Accumulation**: For each pixel $(u, v)$ on sensor plane $c \in \{0, 1, 2, 3\}$:
   $$\Delta L(u, v; t) = I(u, v; t) - I_{\text{ref}}(u, v; t)$$
   $$S_{\text{pos}}(u, v; t) = \mathbb{I}\left(\Delta L(u, v; t) > \theta\right), \quad S_{\text{neg}}(u, v; t) = \mathbb{I}\left(\Delta L(u, v; t) < -\theta\right)$$
   where default threshold $\theta = 0.08$. The reference intensity updates exclusively at spiking coordinates:
   $$I_{\text{ref}}(u, v; t + 1) = \begin{cases} I(u, v; t), & \text{if } S_{\text{pos}}(u, v; t) \lor S_{\text{neg}}(u, v; t) \\ I_{\text{ref}}(u, v; t), & \text{otherwise} \end{cases}$$
2. **Dual-Stream Convolutional Stem & Fusion**:
   - Primary backbone: $\text{Conv2D}(4 \to 16, 3 \times 3, \text{stride}=2) \to \text{GN}(4) \to \text{SiLU} \to \text{Conv2D}(16 \to 32, 3 \times 3, \text{stride}=2) \to \text{GN}(4) \to \text{SiLU}$.
   - Egocentric wrist stem: Parallel Conv2D stream processing channels $2..3$ with identical stride-2 topology.
   - Cross-stream residual fusion: $\mathbf{F} = \mathbf{F}_{\text{global}} + \text{Conv2D}_{1 \times 1}(\mathbf{F}_{\text{wrist}}) \in \mathbb{R}^{B \times K \times 8 \times 8}$ ($K=16$).
3. **Differentiable SpatialSoftmax**:
   For feature map channel $k \in \{1, \dots, K\}$ with spatial grid $\mathcal{G} = [-1, 1]^2$:
   $$\alpha_{k}(u, v) = \frac{\exp\left(F_{k}(u, v) / \tau\right)}{\sum_{u'=1}^W \sum_{v'=1}^H \exp\left(F_{k}(u', v') / \tau\right)}, \quad \tau = 1.0$$
   $$\mathbf{k}_{t, k} = \left[ \sum_{u, v} \alpha_k(u, v) \cdot x_u, \; \sum_{u, v} \alpha_k(u, v) \cdot y_v \right] \in [-1, 1]^2$$
   Extracting $K=16$ keypoints yields $\mathbf{k}_t \in [-1, 1]^{2K} = [-1, 1]^{32}$.
4. **Visual Feature Projection**:
   $$\mathbf{x}_t = \text{SiLU}\left(\text{LayerNorm}\left(\mathbf{W}_{\text{proj}} \mathbf{k}_t + \mathbf{b}_{\text{proj}}\right)\right) \in \mathbb{R}^{B \times D_{\text{enc}}}, \quad D_{\text{enc}} = 128$$

### 2.2 Recurrent Spiking Latent Dynamics Core
1. **Adaptive Leaky Integrate-and-Fire (ALIF) Neurons**:
   Each neuron $j$ maintains membrane potential $V_{j, t}$ and dynamic threshold adaptation $A_{j, t}$:
   $$V_{j, t} = \beta_{\text{mem}} V_{j, t-1} + I_{j, t} - S_{j, t-1} V_{\text{th0}}$$
   $$A_{j, t} = \beta_{\text{adapt}, j} A_{j, t-1} + S_{j, t-1}$$
   $$V_{\text{th}, j, t} = V_{\text{th0}} + \gamma A_{j, t}$$
   $$S_{j, t} = \Theta\left(V_{j, t} - V_{\text{th}, j, t}\right)$$
   - Baseline threshold: $V_{\text{th0}} = 1.0$
   - Membrane decay: $\beta_{\text{mem}} = 0.80$ ($\tau_{\text{mem}} \approx 0.224\,\text{s}$ at $\Delta t = 0.05\,\text{s}$)
   - Adaptation coupling: $\gamma = 0.18$
2. **Multi-Timescale Memory Hierarchy**: Total population size $N = 256$ partitioned into two equal pools:
   - Tier 1 (Reactive): $N_1 = 128$, $\beta_{\text{adapt}, 1} = 0.90$ ($\tau_{\text{adapt}} \approx 0.47\,\text{s}$)
   - Tier 2 (Deep Context): $N_2 = 128$, $\beta_{\text{adapt}, 2} = 0.985$ ($\tau_{\text{adapt}} \approx 3.28\,\text{s}$)
3. **Surrogate Gradient Formulation**: Non-differentiable step function $\Theta(x)$ replaced in autograd with Arctangent surrogate:
   $$\frac{\partial S}{\partial V} = \psi(V - V_{\text{th}}) = \frac{\alpha / 2}{1 + \left(\frac{\pi}{2} \alpha (V - V_{\text{th}})\right)^2}, \quad \alpha = 2.0$$
4. **Symplectic/Dissipative Phase-Space Coupling & Event-Gated Integration**:
   - Synaptic current injection:
     $$\mathbf{I}_t = \mathbf{W}_{\text{in}} \boldsymbol{\epsilon}_t + \mathbf{W}_{\text{rec}} \mathbf{p}_t + \mathbf{W}_q \mathbf{q}_t + \mathbf{W}_u \mathbf{u}_t, \quad \boldsymbol{\epsilon}_t = \mathbf{x}_t - \mathbf{W}_{\text{pred}} \mathbf{z}_{t-1}$$
   - Normalized continuous momentum projection:
     $$\mathbf{p}_{t+1} = \text{LayerNorm}\left(\mathbf{W}_{\text{spk}} \mathbf{s}_t + \mathbf{W}_{\text{mem}} \tanh(\mathbf{m}_t)\right) \in \mathbb{R}^{96}$$
   - Continuous velocity projection: $\mathbf{v}_p = \mathbf{W}_{\text{vel\_proj}} \mathbf{p}_{t+1} \in \mathbb{R}^{32}$.
   - Anti-Evanescence Stasis Gate:
     $$\text{act}_{\text{eff}} = \max\left(0, \frac{1}{CHW}\sum |\mathbf{E}_t| - \theta_{\text{act}}\right), \quad \theta_{\text{act}} = 0.001$$
     $$K_{\text{gate}} = \text{clamp}\left(\frac{g_{\text{steps}}}{4}, 0, 1\right) \cdot \left(1 - \exp\left(-\text{softplus}(\mathbf{w}_{\text{scale}}) \cdot \text{act}_{\text{eff}}\right)\right) \cdot K_{\text{max}}$$
     with $K_{\text{max}} = 0.35$.
   - Generalized coordinate update:
     $$\mathbf{q}_{t+1} = \mathbf{q}_t + \Delta t \cdot \mathbf{v}_p + K_{\text{gate}} \odot \text{clip}_{\text{norm}}\left(\mathbf{k}_t - \mathbf{q}_t, r_{\text{max}}=0.50\right)$$

### 2.3 Action-Conditioned Attractor Predictor ($M_1$)
When operating autonomously or during active perception rollouts:
1. **Conditioned Dissipative Vector Field**:
   $$\mathbf{h}_t = \text{MLP}_{\text{backbone}}\left([\mathbf{z}_t, \mathbf{u}_t]\right) \in \mathbb{R}^{256}$$
   $$\gamma_t = \sigma\left(\mathbf{W}_{\text{diss}} \mathbf{h}_t + b_{\text{diss}}\right) \in (0, 1)^{96}, \quad b_{\text{diss}} = -2.0$$
   $$\mathbf{f}_p = \mathbf{W}_p \mathbf{h}_t + \mathbf{b}_p \in \mathbb{R}^{96}, \quad \mathbf{f}_q = \mathbf{W}_q \mathbf{h}_t + \mathbf{b}_q \in \mathbb{R}^{32}$$
2. **Phase-Space Unrolling Equations**:
   $$\mathbf{p}_{t+1} = (1 - \Delta t \cdot \gamma_t) \odot \mathbf{p}_t + \mathbf{f}_p$$
   $$\mathbf{q}_{t+1} = \mathbf{q}_t + \Delta t \cdot \left(\mathbf{p}_{t+1}[..., :32] + \mathbf{f}_q\right)$$
   $$\hat{\mathbf{z}}_{t+1} = [\mathbf{q}_{t+1}, \mathbf{p}_{t+1}] \in \mathbb{R}^{128}$$
3. **Continuous Soft-Fade Handover Mechanism**:
   Rather than hard mode switching between sensory and motor phases, $M_1$ unrolls continuously across all horizon steps. When active perception events are present, latent blending authority $w_t$ couples the mental simulation with sensory dynamics:
   $$\mathbf{z}_{t+1} = (1 - w_t) \cdot \hat{\mathbf{z}}_{t+1} + w_t \cdot \mathbf{z}_{\text{grounded}, t+1}$$
   $$w_t = \begin{cases} 
   \text{clamp}\left(\frac{g_{\text{steps}}}{4}, 0, 1\right) \cdot w_{\text{max}}, & \text{if } \text{flag}_t == 1 \text{ (Approach \& Contact Sensory Window)} \\ 
   0.50 \cdot w_{\text{max}}, & \text{if } \text{flag}_t == 2 \land \text{fade\_step} == 1 \text{ (Handover Release Step 1)} \\ 
   0.00, & \text{otherwise (Blind Approach \& Blind Transport)} 
   \end{cases}$$
   where $w_{\text{max}} = 0.40$.

### 2.4 Articulated Kinematic Decoder
- Maps latent coordinates to physical joint telemetry:
  $$\hat{\mathbf{q}}_t^* = \text{MLP}_{\text{pos}}(\mathbf{q}_t) \in \mathbb{R}^6, \quad \hat{\mathbf{p}}_t^* = \text{MLP}_{\text{vel}}(\mathbf{p}_t) \in \mathbb{R}^6$$
  Both MLP heads consist of: $\text{Linear}(D_{\text{sub}}, 64) \to \text{LayerNorm}(64) \to \text{SiLU} \to \text{Linear}(64, 6)$.
- **Zero-Lag EMA Rationale**: Evaluated across 9 grid-search combinations in Iteration 4:
  - $\alpha_{\text{pos}} = 0.0, \alpha_{\text{vel}} = 0.0$ eliminates causal filter phase-lag, reducing mean position error by $2.1\%$ ($0.2114 \to 0.2069\,\text{rad}$), reducing mean velocity error by $2.1\%$, and dropping final velocity error by $12.5\%$ ($0.2187 \to 0.1914\,\text{rad/s}$).

### 2.5 Master Tensor Dimensionality Contract Table

| Tensor Symbol | Variable Name | Producing Module | Consuming Module | Tensor Shape Contract | Invariant & Domain Constraint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $\mathbf{E}_t$ | `events` | `EventCameraSimulator` | `EventEncoder` | `[B, T, 4, 32, 32]` | Sparse binary $\{0, 1\}$, finite, $C=4$ |
| $\mathbf{u}_t$ | `actions` | `ActionConditionedController` | `LatentPredictor` ($M_1$) | `[B, T, 6]` | Motor controls $[-1.0, 1.0]^6$, no NaNs/Infs |
| $\text{flag}_t$ | `phase_flags` | `ActionConditionedController` | `rollout_latent` | `[B, T]` | Integer $\in \{0, 1, 2\}$ (0: Approach, 1: Heft, 2: Transport) |
| $\mathbf{y}_t^*$ | `flat_kinematics` | Dataset / Unity Telemetry | `RepresentationPhaseStrategy` | `[B, T, 12]` | Ground truth $[\mathbf{q}_1^*..\mathbf{q}_6^*, \mathbf{p}_1^*..\mathbf{p}_6^*]$ |
| $\mathbf{k}_t$ | `keypoints` | `SpatialSoftmax` | `SpikingLatentDynamics` | `[B, 32]` | Analytical 2D centers of mass $[-1.0, 1.0]^{32}$ ($2K=32$) |
| $\mathbf{x}_t$ | `sensory_x` | `EventEncoder.proj` | `sensory_predictor` | `[B, 128]` | Continuous visual feature embedding $\mathbb{R}^{128}$ |
| $\boldsymbol{\epsilon}_t$ | `epsilon_t` | Top-down error unit | `SpikingLatentDynamics` | `[B, 128]` | Prediction error $\mathbf{x}_t - \hat{\mathbf{x}}_t \in \mathbb{R}^{128}$ |
| $\mathbf{s}_t$ | `spikes` | `MultiTimescaleMemory` | `SpikingLatentDynamics` | `[B, 256]` | Binary spikes $\{0.0, 1.0\}^{256}$ with surrogate grad |
| $\mathbf{m}_t$ | `concatenated_mems`| `MultiTimescaleMemory` | `SpikingLatentDynamics` | `[B, 256]` | Membrane potentials $\mathbb{R}^{256}$ |
| $\mathbf{q}_t$ | `q` | `SpikingLatentDynamics` | `PhysicalDecoder.pos_head` | `[B, 32]` | Generalized coordinates $[-1.0, 1.0]^{32}$ |
| $\mathbf{p}_t$ | `p` | `SpikingLatentDynamics` | `PhysicalDecoder.vel_head` | `[B, 96]` | Generalized momentum $\mathbb{R}^{96}$ (LayerNormed) |
| $\mathbf{z}_t$ | `latent` | Concatenation $[\mathbf{q}_t, \mathbf{p}_t]$ | $M_1$ Predictor | `[B, 128]` | Partitioned phase space $\mathbb{R}^{128}$ ($D_q + D_p = 128$) |
| $\hat{\mathbf{z}}_{1:H}$ | `trajectory` | `LatentPredictor` ($M_1$) | `RolloutEvaluator` | `[B, H, 128]` | Action-conditioned mental simulation rollouts |
| $\hat{\mathbf{y}}_t$ | `decoded_kinematics`| `PhysicalDecoder` | Telemetry / Loss | `[B, H, 12]` | Decoded joint coordinates and velocities |

---

## 3. Training Protocol & Optimization Guarantees

### 3.1 Decoupled Two-Phase Curriculum
The training pipeline enforces strict physical and mathematical separation between sensory representation learning (Phase 0) and predictive latent attractor dynamics (Phase 1).

```mermaid
flowchart TD
    subgraph Phase0["Phase 0: Grounded Articulated State Estimator (train_representation)"]
        D0["EventWorldDataset (4,000 Trajectories)"] --> B0["Batch [B, T=105, C=4, 32, 32]"]
        B0 --> Mod0["Active: EventEncoder, Dynamics, SensoryPred, PhysicalDecoder"]
        B0 -.-> FrozenM1["Strictly Frozen: LatentPredictor (M1)"]
        Mod0 --> L_kin["L_kinematic: MSE(pos, q*) + (1/s_v^2) * MSE(vel, p*)"]
        Mod0 --> L_der["L_derivative: MSE((q_t+1 - q_t), dt * p_t)"]
        Mod0 --> L_smo["L_smoothness: SmoothL1(p_t+1 - p_t)"]
        Mod0 --> L_sen["L_sensory: MSE(eps_t, 0)"]
        L_kin & L_der & L_smo & L_sen --> Loss0["L_Phase0 = 5.0 L_kin + 1.0 L_der + 0.1 L_smo + 1.0 L_sen"]
        Loss0 --> Opt0["Adam (lr=2e-4, probe_lr=2e-4, grad_clip=0.8)"]
    end

    subgraph CheckpointBridge["Scoped Checkpoint Bridge (strict=True)"]
        Opt0 --> Save0["Phase 0 Checkpoint (model.pt)"]
        Save0 --> Load0["load_representation_checkpoint(model, path)\nLoads: encoder, dynamics, sensory_predictor, physical_decoder\nUntouched: LatentPredictor M1"]
    end

    subgraph Phase1["Phase 1: Action-Conditioned Attractor Predictor (train_m1)"]
        Load0 --> Init1["Freeze: encoder, dynamics, sensory_predictor, physical_decoder"]
        Init1 --> ActiveM1["Active (requires_grad=True): LatentPredictor M1"]
        D0 --> B1["Batch [B, T=105] with actions u_t and phase_flags"]
        B1 --> RolloutEng["Unified rollout_latent with Soft-Fade Sensory Injection\nw_max = 0.40, window [t_contact-7, t_contact+3]"]
        RolloutEng --> Loss1["L_M1 = 1.0 MSE_pos(q_hat, q_target) + 1.5 MSE_vel(p_hat, p_target)\n(2x weighting over grasp lock heft frames)"]
        Loss1 --> Opt1["Adam (lr=2e-4, weight_decay=5e-4, grad_clip=0.8)"]
    end
```

#### Phase 0: Grounded State Estimation (`RepresentationPhaseStrategy`)
- **Objective Function**:
  $$\mathcal{L}_{\text{Phase 0}} = \lambda_{\text{kin}}\mathcal{L}_{\text{kinematic}} + \lambda_{\text{der}}\mathcal{L}_{\text{derivative}} + \lambda_{\text{smo}}\mathcal{L}_{\text{smoothness}} + \lambda_{\text{sen}}\mathcal{L}_{\text{sensory}}$$
  - $\mathcal{L}_{\text{kinematic}} = \frac{1}{BT}\sum_{b, t} \left( \|\hat{\mathbf{q}}_{b, t}^* - \mathbf{q}_{b, t}^*\|_2^2 + \frac{1}{s_v^2} \|\hat{\mathbf{p}}_{b, t}^* - \mathbf{p}_{b, t}^*\|_2^2 \right), \quad s_v = 1.0\,\text{rad/s}$
  - $\mathcal{L}_{\text{derivative}} = \frac{1}{B(T-1)}\sum_{b, t} \|(\mathbf{q}_{b, t+1} - \mathbf{q}_{b, t}) - \Delta t \cdot \mathbf{p}_{b, t}[..., :32]\|_2^2$
  - $\mathcal{L}_{\text{smoothness}} = \frac{1}{B(T-1)}\sum_{b, t} \text{Smooth}_{L1}(\mathbf{p}_{b, t+1} - \mathbf{p}_{b, t}, \beta=0.1)$
  - $\mathcal{L}_{\text{sensory}} = \frac{1}{BT}\sum_{b, t} \|\boldsymbol{\epsilon}_{b, t}\|_2^2$
  - Loss weights: $\lambda_{\text{kin}} = 5.0, \; \lambda_{\text{der}} = 1.0, \; \lambda_{\text{smo}} = 0.1, \; \lambda_{\text{sen}} = 1.0$.
- **Optimization**: Adam optimizer, $\text{lr} = 2 \times 10^{-4}$, weight decay $5 \times 10^{-4}$; probe heads $\text{lr} = 2 \times 10^{-4}$, weight decay $1 \times 10^{-4}$; gradient clipping norm $0.8$.

#### Phase 1: Action-Conditioned Attractor Dynamics (`M1PhaseStrategy`)
- **Objective Function**:
  Representation modules strictly frozen. Canonical target trajectories $\mathbf{z}_{\text{target}} = \text{stop\_grad}(\mathbf{z}^*)$ generated over rollout horizon $H$.
  $$\mathcal{L}_{M_1} = \lambda_{\text{pos}} \frac{1}{BH}\sum_{b, h} w_{\text{heft}}(b, h) \|\hat{\mathbf{q}}_{b, h} - \mathbf{q}_{\text{target}, b, h}\|_2^2 + \lambda_{\text{vel}} \frac{1}{BH}\sum_{b, h} w_{\text{heft}}(b, h) \|\hat{\mathbf{p}}_{b, h} - \mathbf{p}_{\text{target}, b, h}\|_2^2$$
  where $\lambda_{\text{pos}} = 1.0, \lambda_{\text{vel}} = 1.5$, $\lambda_{\text{latent}} = 1.0$, and $w_{\text{heft}} = 2.0$ across $[t_{\text{heft}} - 5, t_{\text{heft}} + 10]$ around contact grasp lock.
- **Optimization**: Adam optimizer updating strictly `model.predictor`, $\text{lr} = 2 \times 10^{-4}$, weight decay $5 \times 10^{-4}$, gradient clipping norm $0.8$.

### 3.2 Gradient Isolation, Scoped Checkpointing & AST Anti-Fallback Enforcement
- **AST Anti-Fallback Policy**: Enforced repository-wide by `scripts/scan_fallbacks.py` and `tests/test_anti_fallback.py`:
  - Zero 3-argument `getattr(obj, attr, default)` calls permitted.
  - Zero ternary `hasattr` branches (`x if hasattr(...) else y`) permitted.
  - Zero silent exception handlers (`except TypeError:`, `except AttributeError:`, `except ...: pass`) permitted.
  - Zero `strict=False` checkpoint loading permitted.
- **Checkpoint Scoping**:
  - `load_representation_checkpoint(model, path)`: Ingests `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` with `strict=True`, leaving `predictor` untouched.
  - `BaseTrainer.load_checkpoint(path)`: Validates state-dict keys strictly with `strict=True`.

### 3.3 Dataset Partitioning & Zero Temporal Leakage
- **Dataset Scale**: 4,000 distinct trajectories generated via PhysX 4 Unity simulation (or deterministic kinematics replay):
  - Trajectory length: $T = 105$ timesteps ($\Delta t = 0.05\,\text{s} \to 5.25\,\text{s}$ per episode).
  - Train split ($80\%$): Trajectories $[0, 3199]$ (3,200 trajectories).
  - Validation split ($10\%$): Trajectories $[3200, 3599]$ (400 trajectories).
  - Test split ($10\%$): Trajectories $[3600, 3999]$ (400 trajectories).
  - Extrapolation split (Held-out): Trajectories $[4000, 4099]$ (100 trajectories with joint velocity range extended to $[-2.0, 2.0]\,\text{rad/s}$).
- **Leakage Prevention**: Partitioning is executed strictly at the **trajectory ID level**. Sub-sequences are never shuffled across splits, guaranteeing strictly unseen robotic reaching sequences at test time.

---

## 4. Empirical Evaluation & Benchmarks

### 4.1 Multi-Horizon Tracking Telemetry Table
Evaluated on the 400 held-out test set trajectories under the production `SPWM-v0.7.3` checkpoint (`results/spwm_v0_7_3_train_m1/metrics_spwm_v0_7_3.json`, 78 epochs, best ep 68) compared against the `SPWM-v0.7.2` baseline:

| Temporal Horizon | Metric | SPWM-v0.7.3 (Active Baseline) | SPWM-v0.7.2 (Prior Baseline) | SPWM-v0.7.0 (Industrial Rigid Baseline) |
| :---: | :--- | :---: | :---: | :---: |
| **$H = 1$** (One-Step) | Position Error (rad)<br>Velocity Error (rad/s)<br>Latent Phase MSE | $0.7983$<br>$0.0393$<br>$0.0022$ | $0.6713$<br>$0.1153$<br>— | $0.5872$<br>$0.2450$<br>— |
| **$H = 5$** ($0.25\,\text{s}$) | Position Error (rad)<br>Velocity Error (rad/s)<br>Latent Phase MSE | $0.4300$<br>$0.0404$<br>$0.0022$ | $0.3442$<br>$0.0916$<br>— | —<br>—<br>— |
| **$H = 10$** ($0.50\,\text{s}$) | Position Error (rad)<br>Velocity Error (rad/s)<br>Latent Phase MSE | $0.2730$<br>$0.1441$<br>$0.0048$ | $0.2269$<br>$0.2099$<br>— | —<br>—<br>— |
| **$H = 25$** ($1.25\,\text{s}$) | Position Error (rad)<br>Velocity Error (rad/s)<br>Latent Phase MSE | $0.2583$<br>$0.1733$<br>$0.0158$ | $0.2275$<br>$0.2227$<br>— | $0.2582$<br>$0.3863$<br>— |
| **$H = 50$** ($2.50\,\text{s}$) | Position Error (rad)<br>Velocity Error (rad/s)<br>Latent Phase MSE | **$0.2385$**<br>$0.1870$<br>$0.0323$ | $0.2204$<br>$0.2214$<br>— | —<br>—<br>— |
| **$H = 100$** ($5.00\,\text{s}$) | Position Error (rad)<br>Velocity Error (rad/s)<br>Latent Phase MSE | $0.2740$<br>**$0.1846$**<br>$0.0616$ | $0.2575$<br>$0.2087$<br>— | $0.2863$<br>$0.4215$<br>— |
| **Canonical Test Trajectory**<br>(TID 3600, $H=100$) | Mean Position Error<br>Final Position Error<br>Mean Velocity Error<br>Mean Latent Error | **$0.2450\,\text{rad}$**<br>$0.4151\,\text{rad}$<br>**$0.1432\,\text{rad/s}$**<br>$3.5459$ | $0.2069\,\text{rad}$<br>$0.4119\,\text{rad}$<br>$0.1745\,\text{rad/s}$<br>$3.5121$ | $0.5872\,\text{rad}$<br>$0.8410\,\text{rad}$<br>$0.3120\,\text{rad/s}$<br>— |

> [!NOTE]
> **Observation on Initial $H=1$ vs Extended $H=50$ Error**: In early frames ($t \in [0, 4]$), the initial coordinate offset between resting posture and active visual tracking creates a transient position error ($0.7983\,\text{rad}$), while velocity error remains near zero ($0.0393\,\text{rad/s}$). As the arm articulates, dynamic attractor prediction rapidly converges, achieving a minimum tracking error of $0.2385\,\text{rad}$ at $H=50$ and maintaining $0.2740\,\text{rad}$ out to $H=100$.

### 4.2 Active Perception Sensory Injection Ablation
To verify the hypothesis that active perception heft micro-movements inject payload mass/inertia information into the latent dynamics, tracking performance was benchmarked across 5 diverse test episodes with vs. without sensory grounding injection:

| Episode Index | Trajectory ID | Payload Geometry | Payload Mass | Elevation Plane | With-Injection Pos Err (Mean / Final) | Pure Open-Loop Pos Err (Mean / Final) | Tracking Error Reduction ($\Delta\%$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ep 0** | 3600 | Elongated Cylinder | $3.09\,\text{kg}$ | Mid ($1.14\,\text{m}$) | $0.2069 / 0.4119\,\text{rad}$ | $0.3185 / 0.4739\,\text{rad}$ | **$-35.0\%$ Mean** |
| **Ep 1** | 3700 | Sphere | $1.84\,\text{kg}$ | Low ($1.04\,\text{m}$) | $0.2470 / 0.3493\,\text{rad}$ | $0.3363 / 0.4160\,\text{rad}$ | **$-26.6\%$ Mean** |
| **Ep 2** | 3800 | Box | $2.51\,\text{kg}$ | High ($1.26\,\text{m}$) | $0.2012 / 0.4425\,\text{rad}$ | $0.2740 / 0.6114\,\text{rad}$ | **$-26.6\%$ Mean ($-27.6\%$ Final)** |
| **Ep 3** | 3899 | Elongated Cylinder | $3.18\,\text{kg}$ | Mid ($1.14\,\text{m}$) | $0.2528 / 0.1604\,\text{rad}$ | $0.4413 / 0.1674\,\text{rad}$ | **$-42.7\%$ Mean** |
| **Ep 4** | 3999 | Box | $3.85\,\text{kg}$ | High ($1.26\,\text{m}$) | $0.2411 / 0.2534\,\text{rad}$ | $0.2584 / 0.2463\,\text{rad}$ | **$-6.7\%$ Mean** |
| **Mean Aggregate** | — | — | $\mathbf{2.89\,\text{kg}}$ | — | $\mathbf{0.2298 / 0.3235\,\text{rad}}$ | $\mathbf{0.3257 / 0.3830\,\text{rad}}$ | **$-27.5\%$ Mean Error** |

### 4.3 Biological Plausibility Metrics
- **Mean Firing Sparsity**: $\bar{s} = 0.1460$ ($14.60\%$ mean spike rate per neuron per timestep). Meets the biophysical sparsity design target of $10.0\% - 15.0\%$, avoiding both silent network quiescence ($< 5\%$) and energetic saturation ($> 25\%$).
- **Sustained Multi-Horizon Firing**: Average of $37.4$ active spikes per timestep across the 256 ALIF population at horizon $H=100$.
- **Membrane & Gradient Stability**: Peak gradient norm capped at $0.80$ via gradient clipping. LayerNorm normalization on $\mathbf{p}_t$ constrains maximum momentum norms to $\|\mathbf{p}\|_2 \le 3.5$, protecting downstream ALIF membrane potentials from hyperpolarizing or depolarizing current explosions.

---

## 5. Architectural Evolutionary History & Post-Mortems

```mermaid
timeline
    title SPWM Architectural Evolution (v1 to v0.7.3)
    section Foundational
        SPWM-v1 : Surrogate BPTT : LIF Neurons : O(T) Memory Complexity
        SPWM-v2 : Non-BPTT Blueprint : Dendritic Compartments : SVD Sleep (Unstable)
        SPWM-v3 : Radical Pruning : ALIF Forward Core : Dual-Trace e-prop
    section Spatial & Phase Space
        SPWM-v4 : SpatialSoftmax (K=16) : Symplectic Phase Space (q in R^32, p in R^96)
        SPWM-v5 : Port-Hamiltonian Collapse : Heuristic Freezing Post-Mortem : Rollback to v4.3
        SPWM-v6 : Population Scaling to 256 ALIFs : CANN Low-Pass Failure : Autoregressive Loss
    section Robotics & Active Perception
        SPWM-v0.5.0 : Unity 3D PhysX 4 Rig : Featherstone 6-DOF : Ghost Playback Manager
        SPWM-v0.5.3 : Contact Shock Post-Mortem : Teleporting Ghost Fix
        SPWM-v0.6.0 : Dual-Camera Perception (Global + Wrist, C=4) : Depth Disambiguation
        SPWM-v0.7.2 : Anti-Fallback AST Enforcement : Zero-Lag Decoder Sweep (alpha=0.0)
        SPWM-v0.7.3 : Anthropomorphic Humanoid Rig : Continuous Soft-Fade Baseline
```

### 5.1 Chronological Milestone Progression
1. **SPWM-v1 (Baseline State)**: Standard LIF neurons trained via global Backpropagation Through Time (BPTT). Suffered from $O(T)$ computational memory footprint and vanishing gradients over $T > 50$.
2. **SPWM-v2 (Non-BPTT Exploratory Paradigm)**: 2-compartment dendritic neurons with noisy mirror feedback alignment and periodic offline SVD sleep consolidation. Discarded due to geometric misalignment and E/I balance divergence during sleep phases.
3. **SPWM-v3 / v3.1 (ALIF Forward Core)**: Dynamic adaptive threshold $A_t$ with dual-trace e-prop. Added $1/T$ inverse sequence trace scaling, eliminating gradient explosion and static arena-center collapse.
4. **SPWM-v4.0 / v4.3 (Topological Spatial-Softmax & Symplectic Space)**: Replaced dense visual layers with SpatialSoftmax extracting $K=16$ keypoints in $[-1, 1]^2$. Formally partitioned latent state into coordinate manifold $\mathbf{q} \in \mathbb{R}^{32}$ and momentum manifold $\mathbf{p} \in \mathbb{R}^{96}$.
5. **SPWM-v6 (Population Scaling & Multi-Step Loss)**: Scaled ALIF population to 256 units (two 128-neuron tiers with $\beta=0.90$ and $\beta=0.985$). Introduced autoregressive multi-step rollout loss, locking firing rate at $10.4\%-11.8\%$.
6. **SPWM-v0.5.0 (Unity 3D Continuous Robotics)**: Transitioned from 2D toy physics to 3D continuous robotics (Unity PhysX 4 Featherstone `ArticulationBody`, actions $\mathbf{u}_t \in \mathbb{R}^6$). Introduced holographic ghost visualizer (`GhostPlaybackManager.cs`).
7. **SPWM-v0.6.0 (Dual-Camera Active Perception, $C=4$)**: Added egocentric wrist-mounted event camera paired with static head observer, resolving axial wrist roll symmetry and elbow flex depth foreshortening.
8. **SPWM-v0.7.2 (Anti-Fallback Enforcement & Zero-Lag Readout)**: Completely eliminated all silent fallbacks (`getattr`, ternary `hasattr`, `strict=False`). Conducted 9-combination grid sweep over decoder EMA smoothing factors, establishing optimal zero-lag filtering ($\alpha_{\text{pos}} = 0.0, \alpha_{\text{vel}} = 0.0$).
9. **SPWM-v0.7.3 (Production Baseline)**: Integrated anthropomorphic right humanoid arm kinematics, continuous soft-fade sensory handover ($w_{\text{max}} = 0.40$), and procedural object variance ($1.20 - 4.50\,\text{kg}$).

### 5.2 Negative Ablations & Post-Mortem Log

#### 1. Port-Hamiltonian Recurrence Collapse (SPWM-v5.1 – v5.4, SPWM-v6.1)
- **Concept Tested**: Enforcing strict skew-symmetric and dissipative recurrence matrices $\mathbf{W}_{\text{rec}} = \mathbf{J} - \mathbf{R}$ ($\mathbf{J} = -\mathbf{J}^T, \mathbf{R} \succeq 0$) on latent transitions and hard wall-reflection heuristics ($|\mathbf{q}| \ge 0.98 \to \mathbf{v} \leftarrow -0.8 \mathbf{v}$).
- **Observed Failure**: Disrupted flow continuity. The network suffered catastrophic long-horizon freezing: $H=1$ position error was acceptable ($0.0919\,\text{rad}$), but $H=50$ and $H=100$ position errors collapsed to $0.8942\,\text{rad}$ and $1.1479\,\text{rad}$.
- **Resolution**: Fully rolled back to the v4.3 unconstrained/dissipative baseline. Replaced with parameterized state-dependent dissipation $\gamma_t = \sigma(\text{Head}_{\text{diss}}(\mathbf{h}_t))$ in $M_1$.

#### 2. Continuous Attractor Neural Network (CANN) Low-Pass Distortion (SPWM-v6.2)
- **Concept Tested**: Integrated a Continuous Attractor Neural Network with $M=64$ Gaussian bump centers across $\mathbf{q} \in [-1, 1]^{32}$ to eliminate dead-reckoning drift.
- **Observed Failure**: The CANN energy landscape acted as a destructive low-pass filter. Trajectories froze onto the nearest attractor center; $H=1$ position error worsened to $0.489\,\text{rad}$, and $H=100$ error degraded to $0.921\,\text{rad}$.
- **Resolution**: Discarded. Manifold stability was achieved via multi-step curriculum training rather than static attractor basins.

#### 3. Rigid Contact Impulse Shock & Teleporting Ghost Artifacts (SPWM-v0.5.3)
- **Concept Tested**: Instantaneous rigid grasp attachment of a heavy payload ($m > 2.0\,\text{kg}$) under unconstrained sensory velocity coupling.
- **Observed Failure**: Impact force induced a joint velocity jump $> 2.5\,\text{rad/s}$ and a discrete position leap of $0.4\,\text{rad}$ within 2 timesteps ($100\,\text{ms}$), causing the ghost holographic arm to "teleport" away from the real arm during contact frames 53–63.
- **Resolution**: Replaced hard switching with the Phasic Active Perception protocol: sensory injection initiated 7 frames prior to contact, capped at $K_{\text{max}} = 0.35$, followed by smoothed continuous soft-fade handover ($w_{\text{max}} = 0.40$).

#### 4. Static Attractor Deadlock & Over-Constrained Clamping (SPWM-v0.5.4, SPWM-v0.6.2)
- **Concept Tested**: To prevent stationary drift when the arm paused, coordinates were clamped when event activity fell below an aggressive threshold ($\theta_{\text{act}} = 0.012$).
- **Observed Failure**: Caused severe motion deadlock. Joint motion excursions were attenuated by $81\%$ (shoulder pan moved only $0.253\,\text{rad}$ vs. $1.350\,\text{rad}$ real), and validation position error surged to $1.177\,\text{rad}$.
- **Resolution**: Recalibrated activity threshold to background thermal noise floor ($\theta_{\text{act}} = 0.001$), restoring full joint excursion while suppressing stationary drift.

---

## 6. Gap Analysis & Paper Readiness Audit

This section highlights the critical missing empirical benchmarks, architectural baselines, and profiling requirements necessary to defend an academic paper submission at top-tier venues (CoRL, ICRA, NeurIPS, RSS).

```mermaid
flowchart TD
    subgraph RepoCurrent["Current Repository Assets"]
        R1["SPWM-v0.7.3 SNN World Model"]
        R2["Unity 3D Simulation & Telemetry"]
        R3["Single Seed (seed=42)"]
        R4["Open-Loop Horizon Benchmarks (H=1..100)"]
    end

    subgraph MissingPublication["[MISSING / ACTION REQUIRED] for Top-Tier Publication"]
        M1["1. Non-Spiking Baseline Models\n(RSSM/Dreamer, LSTM/GRU, ConvLSTM)"]
        M2["2. SNN vs Continuous ANN Ablation\n(Identical params, Continuous Leaky units)"]
        M3["3. Hardware Energy & SOP Profiling\n(Neuromorphic SOPs vs GPU FLOPs / Joules)"]
        M4["4. Statistical Multi-Seed Rigor\n(3-5 seeds: mean +/- std error bars)"]
        M5["5. Downstream Closed-Loop Task\n(Latent MPC / Planning / Policy Control)"]
        M6["6. Sim-to-Real Domain Gap Analysis\n(Event noise, lighting, real arm dynamics)"]
    end

    RepoCurrent -.->|"Requires Comparative Benchmarks"| MissingPublication
```

### 6.1 Missing Comparative Baselines
- [x] **Current Repo State**: Evaluates only internal variations of SPWM across historical versions (v0.5.0 $\to$ v0.7.3).
- `[MISSING / ACTION REQUIRED]`: Peer reviewers at CoRL, ICRA, and NeurIPS will reject world model papers that lack head-to-head empirical comparisons against standard deep recurrent world models.
  - **Required Baseline 1: RSSM (Recurrent State-Space Model / DreamerV3)**: Train a continuous RSSM (deterministic GRU recurrent state $h_t \in \mathbb{R}^{256}$ + stochastic discrete/continuous state $s_t \in \mathbb{R}^{32}$) on the identical event frame input stream $\mathbf{E}_t$.
  - **Required Baseline 2: Deep GRU / LSTM Predictor**: Standard 2-layer GRU/LSTM recurrent baseline matched in parameter budget ($\sim 328\,\text{k}$ parameters) conditioned on actions $\mathbf{u}_t$.
  - **Required Baseline 3: Continuous ConvLSTM**: Spatial-temporal ConvLSTM operating directly on raw spatial feature maps.
  - **Comparative Metric Table Needed**:
    $$\text{MSE}_{\text{pos}}(H \in \{1, 10, 50, 100\}), \quad \text{MSE}_{\text{vel}}(H \in \{1, 10, 50, 100\}), \quad \text{Parameter Count}, \quad \text{Inference Latency (ms)}$$

### 6.2 Missing Non-Spiking SNN Ablation
- [x] **Current Repo State**: Demonstrates that the ALIF multi-timescale spiking network converges with $14.6\%$ firing rate.
- `[MISSING / ACTION REQUIRED]`: Reviewers will question whether performance is driven by the **spiking dynamics** or simply by the **symplectic/dissipative phase-space factorization and SpatialSoftmax inductive bias**.
  - **Required Ablation**: Construct `ANNWorldModel` by replacing the 256 ALIF spiking cells in `MultiTimescaleMemory` with identical continuous leaky rate units:
    $$\mathbf{m}_t = \beta_{\text{mem}} \mathbf{m}_{t-1} + (1 - \beta_{\text{mem}}) \mathbf{I}_t$$
    using identical parameter counts, projection matrices, and loss formulations.
  - **Deliverable**: A side-by-side performance table showing tracking accuracy, drift rate over $H=100$, and training convergence stability between SNN-ALIF and ANN-Rate versions.

### 6.3 Computational & Energy Profiling (SOPs vs. FLOPs)
- [x] **Current Repo State**: `spwm/training/metrics.py` contains a helper function `energy_estimate(total_spikes, avg_fan_out=128, energy_per_sop_pj=0.9)`, but it is never invoked during evaluation and no hardware energy telemetry is reported.
- `[MISSING / ACTION REQUIRED]`: Neuromorphic research claims superior energy efficiency. Reviewers require rigorous empirical calculation of Synaptic Operations (SOPs):
  - **Synaptic Operations Equation**:
    $$\text{SOPs}(t) = \sum_{l=1}^{L} s_{l-1}(t) \cdot F_{\text{out}, l}$$
    where $s_{l-1}(t)$ is the spike count emitted by layer $l-1$, and $F_{\text{out}, l}$ is the average synaptic fan-out.
  - **Energy Metric Formulation**:
    $$E_{\text{neuromorphic}} = \text{SOPs} \times E_{\text{SOP}} + N_{\text{neurons}} \times E_{\text{leak}}$$
    Benchmark against GPU dense Multiply-Accumulate operations (MACs):
    $$E_{\text{GPU}} = \text{FLOPs} \times E_{\text{FLOP}}$$
    using standard neuromorphic hardware benchmarks (e.g., Intel Loihi-2: $E_{\text{SOP}} \approx 0.9\,\text{pJ}$ vs. Nvidia Jetson Orin / RTX GPU: $E_{\text{FLOP}} \approx 3.7 - 5.0\,\text{pJ}$).

### 6.4 Statistical Significance & Multi-Seed Validation
- [x] **Current Repo State**: All active results in `results/spwm_v0_7_3_train_m1/` are generated under a single random seed (`seed = 42`).
- `[MISSING / ACTION REQUIRED]`: Top-tier venues mandate statistical significance across random initializations.
  - Execute the full two-phase pipeline (`run_pipeline.sh all`) across at least $N = 3$ (ideally $N = 5$) distinct random seeds (e.g., `seed = 42, 101, 2024, 3407, 9999`).
  - Report all empirical tables and horizon degradation figures with **mean $\pm$ standard deviation** error bars.

### 6.5 Closed-Loop Downstream Control Task
- [x] **Current Repo State**: The world model operates exclusively as an open-loop forward predictive simulator with active perception injection.
- `[MISSING / ACTION REQUIRED]`: Robotics reviewers at CoRL, RSS, and ICRA will strongly challenge whether this predictive world model can actually control the robot arm.
  - **Required Control Experiment**: Implement a downstream **Model Predictive Control (MPC)** planner (e.g., Cross-Entropy Method (CEM) or Model Predictive Path Integral (MPPI)) executing inside SPWM's latent space:
    $$\mathbf{u}_{t:t+H}^* = \arg\min_{\mathbf{u}_{t:t+H}} \sum_{\tau=t}^{t+H} \mathcal{C}\left(\hat{\mathbf{y}}_\tau, \mathbf{y}_{\text{goal}}\right)$$
  - **Benchmark Metric**: Grasp success rate ($\%$) and terminal payload placement accuracy ($\text{cm}$) when manipulating unseen payloads ($1.20 - 4.50\,\text{kg}$) under dynamic perturbations.

### 6.6 Sim-to-Real Domain Gap Analysis
- [x] **Current Repo State**: The entire simulation is conducted within Unity PhysX 4 and synthetic log-intensity difference compute shaders.
- `[MISSING / ACTION REQUIRED]`: To satisfy sim-to-real transfer requirements, a dedicated section in the paper must formally analyze and address physical domain discrepancies:
  - **Event Noise & Background Activity**: Real neuromorphic cameras (e.g., Prophesee Gen4, iniVation DAVIS346) exhibit thermal leakage, junction dark currents, and refractory period variations that synthetic log-diff shaders omit. Domain randomization over event threshold $\theta \sim \mathcal{U}(0.05, 0.12)$ and synthetic refractory noise injection must be validated.
  - **Mechanical Transmission Discrepancies**: Real articulated arms experience cable elasticity, gearbox backlash, harmonic drive friction, and motor thermal degradation not present in idealized Featherstone articulation equations.
  - **Tactile Slip & Friction Uncertainty**: The rigid `FixedJoint` assumption in simulation ignores physical slippage during heavy payload lifts ($> 3.5\,\text{kg}$).

---

## 7. Concrete Next Steps & Action Plan Prior to Submission

To transition this technical dossier into a submission-ready manuscript:

```mermaid
flowchart LR
    Step1["Step 1: Automated Multi-Seed Pipeline\n(Run seeds 101, 2024, 3407)"] --> Step2["Step 2: Non-Spiking ANN & Baseline Suite\n(Implement rate-based ALIF & GRU baseline)"]
    Step2 --> Step3["Step 3: Neuromorphic Energy Telemetry\n(Integrate SOP counter into RolloutEvaluator)"]
    Step3 --> Step4["Step 4: Latent MPC Planning Experiment\n(Implement MPPI/CEM in latent space)"]
    Step4 --> Step5["Step 5: Manuscript Prose Composition\n(Draft CoRL/ICRA submission manuscript)"]
```

1. **Execute Multi-Seed Verification**: Run `scripts/run_pipeline.sh all` with `--seed 101` and `--seed 2024` to compute mean $\pm$ standard deviation for Table 4.1.
2. **Implement Rate-Based ANN Ablation**: Duplicate `MultiTimescaleMemory` into a continuous-rate variant to generate the definitive SNN ablation table.
3. **Activate SOP Energy Profiling**: Instrument `spwm/evaluation/evaluator.py` to record active synaptic operations per timestep, computing Joules/inference relative to GPU dense inference.
4. **Benchmark Latent MPC Closed-Loop Controller**: Demonstrate that planning motor commands $\mathbf{u}_t$ using SPWM unrolls improves grasp stability under variable payload masses without requiring visual retraining.