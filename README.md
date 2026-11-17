# SPWM: Action-Conditioned Spiking Predictive World Model (v0.12.4)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Unity 2022.3 LTS](https://img.shields.io/badge/Unity-2022.3%20LTS-black.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.12.4** is an action-conditioned neuromorphic predictive world model engineered under a **4-Layer Strict Contract Architecture**. It bridges **Unity 3D articulated robotics simulation** (Featherstone multi-body kinematics, anthropomorphic 6-DOF right humanoid arm $J=6$, procedural object spawning with variable mass/inertia $1.20 - 4.50\,\text{kg}$) with continuous motor action conditioning ($\mathbf{u}_t \in \mathbb{R}^6$), dual-camera neuromorphic event sensing ($C_{\text{event}}=4$: static global + egocentric wrist-mounted), resolution-adaptive `SpatialSoftmax` keypoint bottleneck ($K=16$), canonical phase-space decomposition $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t) \in \mathbb{R}^{32 + 96}$, a dual-timescale Adaptive Leaky Integrate-and-Fire (`ALIFCell`) memory core with firing-rate homeostasis, SE(3) + temporal kinematic consistency, continuous soft-fade sensory handover ($w_{\text{max}} = 0.40$), and downstream deploy-ready **Latent MPPI Control** (`392,141` parameters).

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

# Execute the full unit & architectural contract test suite (99 tests)
.venv/bin/pytest -q

# Run AST anti-fallback scan
.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/
```

---

## 4. Two-Phase Representation / Action-Conditioned Predictor Pipeline

### Phase 0 (`train_representation`): Grounded Articulated State Estimator ($t \in [0, T_{\text{obs}}]$)
Pre-trains `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` purely as a grounded joint state estimator over sensory streams with ALIF firing-rate homeostasis ($\lambda_{\text{spike\_balance}} = 0.25$, target $0.12$), Cartesian + SE(3) rotation supervision ($\lambda_{\text{cart}} = 40.0$, $\lambda_{\text{rot}} = 0.5$), and decoded temporal smoothness ($\lambda_{\text{kin\_smooth}} = 0.5$):

$$\mathcal{L}_{\text{Phase0}} = \lambda_{\text{kinematic}} \mathcal{L}_{\text{kinematic}} + \lambda_{\text{derivative}} \mathcal{L}_{\text{derivative}} + \lambda_{\text{smoothness}} \mathcal{L}_{\text{smoothness}} + \lambda_{\text{sensory}} \mathcal{L}_{\text{sensory}} + \lambda_{\text{cart}} \mathcal{L}_{\text{cart}} + \lambda_{\text{rot}} \mathcal{L}_{\text{rot}} + \lambda_{\text{kin\_smooth}} \mathcal{L}_{\text{kin\_smooth}} + \lambda_{\text{spike\_balance}} \mathcal{L}_{\text{spike\_balance}}$$

```bash
.venv/bin/python experiments/train.py \
    --mode train_representation \
    --epochs 160 \
    --output-dir results/spwm_v0_12_4_train_representation
```

### Phase 1 (`train_m1`): Action-Conditioned Latent Attractor Predictor ($t \in [0, H-1]$)
Freezes all Phase 0 representation modules (`requires_grad = False`), detaches latent trajectory targets $\mathbf{z}^*_{\text{target}} = \text{stop\_grad}(\mathbf{z}^*)$, and trains the Latent Predictor $M_1$ + `LatentCritic` across a 4-phase Latch curriculum (open-loop $\to$ AR(1) matched noise $\beta=0.90, \sigma=0.08$ $\to$ DAgger $\to$ MPPI closed-loop):

$$\hat{\mathbf{z}}_{t+1} = M_1(\hat{\mathbf{z}}_t, \mathbf{u}_t)$$
$$\mathcal{L}_{M_1} = \lambda_{\text{position}} \text{MSE}(\hat{\mathbf{q}}, \mathbf{q}^*_{\text{target}}) + \lambda_{\text{velocity}} \text{MSE}(\hat{\mathbf{p}}, \mathbf{p}^*_{\text{target}}) + \mathcal{L}_{\text{SE(3)}} + \mathcal{L}_{\text{kin\_smooth}}$$

```bash
.venv/bin/python experiments/train.py \
    --mode train_m1 \
    --representation-checkpoint results/spwm_v0_12_4_train_representation/model.pt \
    --epochs 240 \
    --phase-horizon 50 \
    --output-dir results/spwm_v0_12_4_train_m1
```

---

## 5. Downstream Control, Baselines & Benchmarks

```bash
# Evaluate a trained checkpoint across horizons H in {1, 5, 10, 25, 50, 100}
.venv/bin/python experiments/evaluate.py \
    --checkpoint results/spwm_v0_12_4_train_m1/model.pt \
    --config results/spwm_v0_12_4_train_m1/config.yaml

# Benchmark downstream closed-loop Latent MPPI control (50 unseen episodes)
.venv/bin/python experiments/benchmark_mppi_control.py \
    --checkpoint results/spwm_v0_12_4_train_m1/model.pt \
    --config results/spwm_v0_12_4_train_m1/config.yaml \
    --output-dir results/mppi_benchmark_v0_12_4

# Benchmark comparative non-spiking world models (RSSM, Deep GRU, ConvLSTM)
.venv/bin/python experiments/train_baselines.py

# Benchmark continuous leaky rate ablation (ANNWorldModel)
.venv/bin/python experiments/train_ann_ablation.py

# Run neuromorphic Synaptic Operations (SOPs) and energy profiling
.venv/bin/python experiments/profile_energy_sops.py

# Run statistical multi-seed validation (seeds 42, 101, 2024)
.venv/bin/python experiments/train_multiseed.py

# Run zero-shot sim-to-real sensor & mechanical robustness stress tests
.venv/bin/python experiments/benchmark_robustness_stress_test.py
```

### Execution Confirmation & Validation Criteria (`SPWM-v0.12.4`)

- **Action-Conditioned Long-Horizon Rollout Stability**: `rollout_mae = 0.05500 rad` (`rollout_score = 0.38069`, `drift_ratio = 2.69x`), with `H=1, 10, 50, 100` joint position errors of `0.02187 / 0.02801 / 0.08352 / 0.11864 rad` (`position_mse = 0.003501` at $H=100$).
- **ALIF Homeostasis & Temporal Smoothness**: Phase 0 ALIF mean spike rate `0.1684` (silent neurons reduced from `44.9%` to `19.1%`); grounded acceleration ratio reduced from `7.79x` to `3.88x` (`kin_fwd_mae = 0.1079`). Closed-loop MPPI ghost rollout (`traj 3622`) achieves `mean_ddq = 0.00426 rad/step^2` (`-64.7%` vs `v0.12.20`), `mean_dddq = 0.00524` (`-68.8%`), `flips = 74` (`-41.7%`), and `mean_latent_error = 1.17 cm`.
- **Deploy-Ready Closed-Loop MPPI Control**: Receding-horizon warm-start, 2-stage $SE(3)$ horizon lookahead, residual-gated active perception, AR(1)-matched noise ($\beta=0.90, \sigma=0.08$), slew-rate clamping (`max_action_delta=0.10 rad/step`), and joint acceleration & limit costs achieve **96.0% grasp success** on the 50-episode benchmark (Low `90.9%`, Mid `100.0%`, High `100.0%`; Light `88.2%`, Medium `100.0%`, Heavy `100.0%`), **`0.67 ± 0.46 cm`** minimum placement accuracy, **`1.207 cm`** terminal error, and **`22.70 ms`** step latency.
- **Documentation**: See [**docs/History.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/History.md) for the full chronological record through `SPWM-v0.12.4`.

---

## 6. Comprehensive Documentation Suite

- [**docs/ARCHITECTURE.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/ARCHITECTURE.md): Authoritative technical specification covering the 4-layer workspace, neural models' mathematical & tensor contracts, forward kinematics, MPPI control, comparative baselines, and ablation models.
- [**docs/WORLD.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORLD.md): Specification of the 3D Unity C# simulation environment, Featherstone `ArticulationBody` physics, procedural object spawner, and dual-camera event sensor bridge.
- [**docs/WORKSPACE_AND_CONTRACTS.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORKSPACE_AND_CONTRACTS.md): Deep dive into Layer 0 contracts, AST anti-fallback guards, and the $<3$-file context boundary rules.
- [**docs/TRAINING_AND_LOSSES.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/TRAINING_AND_LOSSES.md): Granular mathematical specification of `RepresentationPhaseStrategy` and `M1PhaseStrategy`, dissipative vector field learning, and scoped checkpoint loading.
- [**docs/DOSSIER.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/DOSSIER.md): Technical dossier and academic readiness audit for top-tier venues (CoRL, ICRA, NeurIPS, RSS).
- [**docs/HISTORY.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/HISTORY.md): Project evolution history from v1 through v0.7.3.
