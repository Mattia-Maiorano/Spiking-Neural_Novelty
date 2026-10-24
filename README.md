# SPWM: Action-Conditioned Spiking Predictive World Model (v0.4.0)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Unity 2022.3 LTS](https://img.shields.io/badge/Unity-2022.3%20LTS-black.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.4.0** is an action-conditioned neuromorphic predictive world model engineered under a **4-Layer Strict Contract Architecture**. It bridges **Unity 3D articulated robotics simulation** (Featherstone multi-body kinematics, joint friction, and CPG/postural attractors) with continuous motor action conditioning ($\mathbf{u}_t$), resolution-adaptive `SpatialSoftmax` event encoding, canonical phase-space decomposition $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t)$, a dual-timescale Adaptive Leaky Integrate-and-Fire (`ALIFCell`) memory core, a strictly decoupled **Phase 0 Grounded Joint State Estimator**, and an **Action-Conditioned Latent Attractor Predictor** ($M_1$).

---

## 1. Architectural Overview

```text
Event Stream [B, T, 2, H, W]       Motor Actions u_t [B, T, D_u]
              │                                  │
              ▼                                  │
EventEncoder + Dynamic SpatialSoftmax            │
   ├─ sensory_x      [B, D_enc]                  │
   └─ keypoints      [B, 2*K] (q_dim = 2*K)      │
              │                                  │
              ▼                                  │
Top-Down Predictive Coding Error:                │
   ϵ_t = x_t - W_pred z_(t-1)                    │
              │                                  │
              ▼                                  │
SpikingLatentDynamics + Dual-Timescale ALIF Core  │
   ├─ Fast ALIF (β_adapt = 0.90)                 │
   ├─ Slow ALIF (β_adapt = 0.985)                │
   └─ Phase-Space State: z_t = [q_t, p_t] ∈ ℝ^D  │
              │                                  │
              └───────────────┬──────────────────┘
                              │
                              ▼
Action-Conditioned Latent Attractor Predictor (M1):
   ├─ Backbone Feature:        h_t = MLP([z_t, u_t])
   ├─ Dissipation Head:        γ_t = σ(W_γ h_t)
   ├─ Force Heads:             f_p = W_p h_t,  f_q = W_q h_t
   ├─ Momentum Update:         p_(t+1) = (1 - Δt · γ_t) p_t + f_p
   ├─ Coordinate Update:       q_(t+1) = q_t + Δt · (p_(t+1)[..., :q_dim] + f_q)
   └─ Next Latent State:       z_(t+1) = [q_(t+1), p_(t+1)]
                              │
                              ▼
Articulated PhysicalDecoder (MLP Heads: decode_pos, decode_vel)
   └─ Decoded Kinematics: [q*, p*] ∈ ℝ^(2 · joint_count)
```

---

## 2. 4-Layer Strict Contract Workspace

The repository enforces a strict unidirectional dependency DAG verified on every test run by [`tests/test_architecture_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_architecture_contracts.py) and [`tests/test_dead_code.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_dead_code.py):

| Layer | Package | Responsibilities | Allowed `spwm.*` Imports |
| :--- | :--- | :--- | :--- |
| **Layer 0** | [`spwm/contracts/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py) | Frozen `@dataclass(frozen=True)` schemas (`ExperimentConfig`), state containers (`ALIFState`, `SPWMState`, `RolloutResult`), runtime tensor validators (`assert_tensor_contract`, `assert_phase_space_contract` checking actions $\mathbf{u}_t$), and `Protocol` interfaces. | None (Zero internal imports) |
| **Layer 1** | [`spwm/models/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py), [`spwm/data/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py), [`sim/`](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Scripts/ArticulatedRobotRig.cs) | Pure neural modules (`ALIFCell`, `MultiTimescaleMemory`, `EventEncoder`, `SpikingLatentDynamics`, `LatentPredictor` with action conditioning, `PhysicalDecoder`, `SPWM`), action-conditioned `rollout_latent`, Unity 3D C# robotics assets, and neuromorphic event sensor simulators. | Layer 0 (`spwm.contracts`) |
| **Layer 2** | [`spwm/training/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/__init__.py) | Composable loss modules (`PredictorLoss`, `SPWMLoss`), isolated $<150$-line phase strategies (`RepresentationPhaseStrategy`, `M1PhaseStrategy`), and `BaseTrainer` with `strict=True` checkpoint scoping. | Layers 0–1 (`spwm.contracts`, `spwm.models`, `spwm.data`) |
| **Layer 3** | [`spwm/evaluation/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/__init__.py) | Action-conditioned multi-horizon rollout evaluation (`RolloutEvaluator`), segmented telemetry resumes (`metrics_resume.py`), and publication figure generation (`visualizer.py`). | Layers 0–2 |

---

## 3. Installation & Verification

Requires Python $\ge 3.11$:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Execute the full unit & architectural contract test suite
pytest -v
```

---

## 4. Two-Phase Representation / Action-Conditioned Predictor Pipeline

### Phase 0 (`train_representation`): Grounded Articulated State Estimator ($t \in [0, T_{\text{obs}}]$)
Pre-trains `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` purely as a grounded joint state estimator over sensory streams (no autoregressive rollouts; predictor inactive):

$$\mathcal{L}_{\text{Phase0}} = \mathcal{L}_{\text{kinematic}} + \mathcal{L}_{\text{derivative}} + \mathcal{L}_{\text{smoothness}} + \mathcal{L}_{\text{sensory}}$$

```bash
python experiments/train.py \
    --mode train_representation \
    --epochs 100 \
    --output-dir results/spwm_v0_4_0_train_representation
```

### Phase 1 (`train_m1`): Action-Conditioned Latent Attractor Predictor ($t \in [0, H-1]$)
Freezes all Phase 0 representation modules (`requires_grad = False`), detaches latent trajectory targets $z^*_{\text{target}} = z^*.\text{detach}()$, and trains only the Latent Predictor $M_1$ autoregressively conditioned on motor actions $\mathbf{u}_{0:H}$:

$$\hat{\mathbf{z}}_{t+1} = M_1(\hat{\mathbf{z}}_t, \mathbf{u}_t)$$
$$\mathcal{L}_{M_1} = \text{MSE}(\hat{\mathbf{z}}_{t+1}, \mathbf{z}^*_{\text{target}, t+1})$$

```bash
python experiments/train.py \
    --mode train_m1 \
    --representation-checkpoint results/spwm_v0_4_0_train_representation/model.pt \
    --epochs 200 \
    --phase-horizon 50 \
    --output-dir results/spwm_v0_4_0_train_m1
```

---

## 5. Evaluation & Scientific Verification

```bash
# Evaluate a trained checkpoint across horizons H in {1, 5, 10, 25, 50, 100}
python experiments/evaluate.py \
    --checkpoint results/spwm_v0_4_0_train_m1/model.pt \
    --config results/spwm_v0_4_0_train_m1/config.yaml

# Run ablation studies
python experiments/ablation.py \
    --checkpoint results/spwm_v0_4_0_train_m1/model.pt \
    --config results/spwm_v0_4_0_train_m1/config.yaml
```

### Execution Confirmation & Validation Criteria

- **Action-Conditioned Attractor Rollout**: Qualitative check verifying continuous, stable phase-space trajectories reproducing CPG walking gaits and postural balancing without explosive divergence.
- **Joint Angle & Velocity Reconstruction**: High accuracy decoding of continuous articulated joint angles $\mathbf{q}^*$ and angular rates $\mathbf{p}^*$ across all $J=8$ degrees of freedom.
- **Multi-Channel & Multi-Metric Consistency**: Low error across all coordinates, balanced ALIF spike rates, and stable decoded physical kinematics.

---

## 6. Comprehensive Documentation Suite

- [**docs/ARCHITECTURE.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/ARCHITECTURE.md): Complete master reference covering the 4-layer workspace, neural models' mathematical & tensor contracts, and the action-conditioned representation/predictor architecture.
- [**docs/WORLD.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORLD.md): Specification of the 3D Unity C# simulation environment, Featherstone `ArticulationBody` physics, CPG limit cycles, and the GPU compute shader event camera bridge.
- [**docs/WORKSPACE_AND_CONTRACTS.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORKSPACE_AND_CONTRACTS.md): Deep dive into Layer 0 contracts, AST anti-fallback guards, and the $<3$-file context boundary rules.
- [**docs/TRAINING_AND_LOSSES.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/TRAINING_AND_LOSSES.md): Granular mathematical specification of `RepresentationPhaseStrategy` and `M1PhaseStrategy`, dissipative vector field learning, and scoped checkpoint loading.
- [**docs/HISTORY.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/HISTORY.md): Project evolution history from v1 through v0.4.0.
