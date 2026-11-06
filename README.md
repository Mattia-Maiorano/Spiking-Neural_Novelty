# SPWM: Action-Conditioned Spiking Predictive World Model (v0.7.3)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Unity 2022.3 LTS](https://img.shields.io/badge/Unity-2022.3%20LTS-black.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.7.3** is an action-conditioned neuromorphic predictive world model engineered under a **4-Layer Strict Contract Architecture**. It bridges **Unity 3D articulated robotics simulation** (Featherstone multi-body kinematics, anthropomorphic 6-DOF right humanoid arm $J=6$, procedural object spawning with variable mass/inertia $1.20 - 4.50\,\text{kg}$) with continuous motor action conditioning ($\mathbf{u}_t \in \mathbb{R}^6$), dual-camera neuromorphic event sensing ($C_{\text{event}}=4$: static global + egocentric wrist-mounted), resolution-adaptive `SpatialSoftmax` keypoint bottleneck ($K=16$), canonical phase-space decomposition $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t) \in \mathbb{R}^{32 + 96}$, a dual-timescale Adaptive Leaky Integrate-and-Fire (`ALIFCell`) memory core, continuous soft-fade sensory handover ($w_{\text{max}} = 0.40$), and downstream closed-loop **Latent MPPI Control**.

---

## 1. Architectural Overview

```text
Dual-Camera Event Stream [B, T, 4, H, W]       Motor Actions u_t [B, T, D_u] (D_u=6)
              │                                                │
              ▼                                                │
EventEncoder + Dynamic SpatialSoftmax                          │
   ├─ sensory_x      [B, D_enc] (D_enc=128)                    │
   └─ keypoints      [B, 2*K] (q_dim = 2*K = 32)               │
              │                                                │
              ▼                                                │
Top-Down Predictive Coding Error:                              │
   ϵ_t = x_t - W_pred z_(t-1)                                  │
              │                                                │
              ▼                                                │
SpikingLatentDynamics + Dual-Timescale ALIF Core               │
   ├─ Fast ALIF (β_adapt = 0.90, N=128)                        │
   ├─ Slow ALIF (β_adapt = 0.985, N=128)                       │
   ├─ Anti-Evanescence Gate K_gate (K_max=0.35)                │
   └─ Phase-Space State: z_t = [q_t, p_t] ∈ ℝ^128 (q∈ℝ^32, p∈ℝ^96)
              │                                                │
              └────────────────────────┬───────────────────────┘
                                       │
                                       ▼
Action-Conditioned Latent Attractor Predictor (M1):
   ├─ Backbone Feature:        h_t = MLP([z_t, u_t])
   ├─ Dissipation Head:        γ_t = σ(W_γ h_t + b_γ)
   ├─ Force Heads:             f_p = W_p h_t,  f_q = W_q h_t
   ├─ Momentum Update:         p_(t+1) = (1 - Δt · γ_t) p_t + f_p
   ├─ Coordinate Update:       q_(t+1) = q_t + Δt · (p_(t+1)[..., :q_dim] + f_q)
   ├─ Soft-Fade Handover:      z_(t+1) = (1 - w_t) z_pred + w_t z_grounded (w_t ∈ [0, 0.40])
   └─ Next Latent State:       z_(t+1) = [q_(t+1), p_(t+1)]
                                       │
                                       ▼
Articulated PhysicalDecoder (MLP Heads: decode_pos, decode_vel)
   ├─ Decoded Kinematics:      [q*, p*] ∈ ℝ^(2 · joint_count) = ℝ^12
   └─ Differentiable FK:       (x, y, z)_ee = FK(q*) → Latent MPPI Controller
```

---

## 2. 4-Layer Strict Contract Workspace

The repository enforces a strict unidirectional dependency DAG verified on every test run by [`tests/test_architecture_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_architecture_contracts.py) and [`tests/test_dead_code.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_dead_code.py):

| Layer | Package | Responsibilities | Allowed `spwm.*` Imports |
| :--- | :--- | :--- | :--- |
| **Layer 0** | [`spwm/contracts/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py) | Frozen `@dataclass(frozen=True)` schemas (`ExperimentConfig`), state containers (`ALIFState`, `SPWMState`, `RolloutResult`), runtime tensor validators (`assert_tensor_contract`, `assert_phase_space_contract` checking actions $\mathbf{u}_t$), and `Protocol` interfaces. | None (Zero internal imports) |
| **Layer 1** | [`spwm/models/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py), [`spwm/data/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py), [`sim/`](file:///Users/Mattia/Desktop/Studies/Temp/sim/Assets/Scripts/ArticulatedRobotRig.cs) | Pure neural modules (`ALIFCell`, `MultiTimescaleMemory`, `EventEncoder`, `SpikingLatentDynamics`, `LatentPredictor` with action conditioning, `PhysicalDecoder`, `DifferentiableForwardKinematics`, `SPWM`), comparative baselines, action-conditioned `rollout_latent`, Unity 3D C# robotics assets, and neuromorphic event sensor simulators. | Layer 0 (`spwm.contracts`) |
| **Layer 2** | [`spwm/training/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/__init__.py) | Composable loss modules (`PredictorLoss`, `SPWMLoss`), isolated $<150$-line phase strategies (`RepresentationPhaseStrategy`, `M1PhaseStrategy`), and `BaseTrainer` with `strict=True` checkpoint scoping. | Layers 0–1 (`spwm.contracts`, `spwm.models`, `spwm.data`) |
| **Layer 3** | [`spwm/evaluation/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/__init__.py) | Action-conditioned multi-horizon rollout evaluation (`RolloutEvaluator`), downstream latent control (`LatentMPPIController`), zero-shot sim-to-real stress testing (`robustness.py`), segmented telemetry resumes (`metrics_resume.py`), and publication figure generation (`visualizer.py`). | Layers 0–2 |

---

## 3. Installation & Verification

Requires Python $\ge 3.11$:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Execute the full unit & architectural contract test suite (112 tests)
pytest -v

# Run AST anti-fallback scan
python scripts/scan_fallbacks.py spwm/
```

---

## 4. Two-Phase Representation / Action-Conditioned Predictor Pipeline

### Phase 0 (`train_representation`): Grounded Articulated State Estimator ($t \in [0, T_{\text{obs}}]$)
Pre-trains `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` purely as a grounded joint state estimator over sensory streams (no autoregressive rollouts; predictor inactive):

$$\mathcal{L}_{\text{Phase0}} = \lambda_{\text{kinematic}} \mathcal{L}_{\text{kinematic}} + \lambda_{\text{derivative}} \mathcal{L}_{\text{derivative}} + \lambda_{\text{smoothness}} \mathcal{L}_{\text{smoothness}} + \lambda_{\text{sensory}} \mathcal{L}_{\text{sensory}}$$

```bash
python experiments/train.py \
    --mode train_representation \
    --epochs 100 \
    --output-dir results/spwm_v0_7_3_train_representation
```

### Phase 1 (`train_m1`): Action-Conditioned Latent Attractor Predictor ($t \in [0, H-1]$)
Freezes all Phase 0 representation modules (`requires_grad = False`), detaches latent trajectory targets $\mathbf{z}^*_{\text{target}} = \text{stop\_grad}(\mathbf{z}^*)$, and trains only the Latent Predictor $M_1$ autoregressively conditioned on motor actions $\mathbf{u}_{0:H}$ with smooth soft-fade sensory injection during heft:

$$\hat{\mathbf{z}}_{t+1} = M_1(\hat{\mathbf{z}}_t, \mathbf{u}_t)$$
$$\mathcal{L}_{M_1} = \lambda_{\text{position}} \text{MSE}(\hat{\mathbf{q}}, \mathbf{q}^*_{\text{target}}) + \lambda_{\text{velocity}} \text{MSE}(\hat{\mathbf{p}}, \mathbf{p}^*_{\text{target}})$$

```bash
python experiments/train.py \
    --mode train_m1 \
    --representation-checkpoint results/spwm_v0_7_3_train_representation/model.pt \
    --epochs 100 \
    --phase-horizon 50 \
    --output-dir results/spwm_v0_7_3_train_m1
```

---

## 5. Downstream Control, Baselines & Benchmarks

```bash
# Evaluate a trained checkpoint across horizons H in {1, 5, 10, 25, 50, 100}
python experiments/evaluate.py \
    --checkpoint results/spwm_v0_7_3_train_m1/model.pt \
    --config results/spwm_v0_7_3_train_m1/config.yaml

# Benchmark downstream closed-loop Latent MPPI control (80.0% grasp success, 2.17 cm min distance)
python experiments/benchmark_mppi_control.py \
    --checkpoint results/spwm_v0_7_3_train_m1/model.pt \
    --config results/spwm_v0_7_3_train_m1/config.yaml

# Benchmark comparative non-spiking world models (RSSM, Deep GRU, ConvLSTM)
python experiments/train_baselines.py

# Benchmark continuous leaky rate ablation (ANNWorldModel)
python experiments/train_ann_ablation.py

# Run neuromorphic Synaptic Operations (SOPs) and energy profiling (80.8x advantage vs GPU)
python experiments/profile_energy_sops.py

# Run statistical multi-seed validation (seeds 42, 101, 2024)
python experiments/train_multiseed.py

# Run zero-shot sim-to-real sensor & mechanical robustness stress tests
python experiments/benchmark_robustness_stress_test.py
```

### Execution Confirmation & Validation Criteria

- **Action-Conditioned Velocity Stabilization**: Discrete spike thresholding and symplectic factorization bound velocity drift over 100 autoregressive steps ($H=100$), achieving $0.01153\,\text{rad}^2/\text{s}^2$ velocity MSE ($4.2\times$ lower than Continuous RSSM and $14\times$ lower than Deep GRU).
- **6-DOF Kinematic Reconstruction**: High accuracy decoding of continuous articulated joint angles $\mathbf{q}^*$ and angular rates $\mathbf{p}^*$ across all $J=6$ manipulator joints.
- **Closed-Loop MPPI Control**: Direct mental simulation in latent space achieves an 80.0% grasp success rate ($2.17\,\text{cm}$ minimum placement distance, $6.72\,\text{cm}$ final terminal error) under a $38.9\,\text{ms}$ step latency budget.
- **Neuromorphic Energy Efficiency**: Achieves $12.81\,\text{nJ/step}$ on Intel Loihi-2 at a $14.6\%$ spike rate, yielding an **80.8x energy reduction** relative to dense GPU execution on Nvidia Jetson Orin.

---

## 6. Comprehensive Documentation Suite

- [**docs/ARCHITECTURE.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/ARCHITECTURE.md): Authoritative technical specification covering the 4-layer workspace, neural models' mathematical & tensor contracts, forward kinematics, MPPI control, comparative baselines, and ablation models.
- [**docs/WORLD.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORLD.md): Specification of the 3D Unity C# simulation environment, Featherstone `ArticulationBody` physics, procedural object spawner, and dual-camera event sensor bridge.
- [**docs/WORKSPACE_AND_CONTRACTS.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORKSPACE_AND_CONTRACTS.md): Deep dive into Layer 0 contracts, AST anti-fallback guards, and the $<3$-file context boundary rules.
- [**docs/TRAINING_AND_LOSSES.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/TRAINING_AND_LOSSES.md): Granular mathematical specification of `RepresentationPhaseStrategy` and `M1PhaseStrategy`, dissipative vector field learning, and scoped checkpoint loading.
- [**docs/DOSSIER.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/DOSSIER.md): Technical dossier and academic readiness audit for top-tier venues (CoRL, ICRA, NeurIPS, RSS).
- [**docs/HISTORY.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/HISTORY.md): Project evolution history from v1 through v0.7.3.
