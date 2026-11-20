# SPWM: Action-Conditioned Spiking Predictive World Model (v0.12.4)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Unity 6 (6000.5.7f1)](https://img.shields.io/badge/Unity-6000.5.7f1-black.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.12.4** is an action-conditioned neuromorphic predictive world model engineered under a **4-Layer Strict Contract Architecture**. It bridges **Unity 6 (`6000.5.7f1`) 3D articulated robotics simulation** (Featherstone multi-body kinematics, anthropomorphic 6-DOF right humanoid arm $J=6$, procedural object spawning with variable mass/inertia $1.20 - 4.50\,\text{kg}$, rigid `FixedJoint` palm-latch) with continuous motor action conditioning ($\mathbf{u}_t \in \mathbb{R}^6$), dual-camera neuromorphic event sensing ($C_{\text{event}}=4$: static global + egocentric wrist-mounted), resolution-adaptive `SpatialSoftmax` keypoint bottleneck ($K=16$), canonical phase-space decomposition $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t) \in \mathbb{R}^{32 + 96}$, a dual-timescale Adaptive Leaky Integrate-and-Fire (`ALIFCell`) memory core with firing-rate homeostasis, SE(3) + temporal kinematic consistency, open-loop soft-fade ($w_{\max}=0.60$) / closed-loop residual-gated sensory handover ($w_t \in \{0.0, 0.40\eta_t, 0.85\}$), and downstream deploy-ready **Latent MPPI Control** (`392,141` total / `317,900` active parameters).

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
   ├─ Residual Handover (PIL): z_(t+1) = (1 - w_t) z_pred + w_t z_obs (w_t ∈ {0, 0.40 η_t, 0.85})
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

# Execute the full unit & architectural contract test suite (101 tests)
.venv/bin/pytest -q

# Run AST anti-fallback scan
.venv/bin/python scripts/scan_fallbacks.py spwm/ experiments/ scripts/
```

---

## 4. Two-Phase Representation / Action-Conditioned Predictor Pipeline

### Phase 0 (`train_representation`): Grounded Articulated State Estimator ($t \in [0, T_{\text{obs}}]$)
Pre-trains `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` across 3 learning-rate tiers (5 Adam parameter groups) purely as a grounded joint state estimator over sensory streams with ALIF firing-rate homeostasis ($\lambda_{\text{spike\_balance}} = 0.25$, target $0.12$), Cartesian + SE(3) rotation supervision ($w_{\text{cart}} = 40.0$, $w_{\text{rot}} = 0.5$ nested inside $\mathcal{L}_{\text{kinematic}}$), and decoded temporal smoothness ($\lambda_{\text{kin\_smooth}} = 0.5$):

$$\mathcal{L}_{\text{Phase0}} = 5.0\,\mathcal{L}_{\text{kinematic}} + 1.0\,\mathcal{L}_{\text{derivative}} + 0.1\,\mathcal{L}_{\text{smoothness}} + 0.5\,\mathcal{L}_{\text{kin\_smooth}} + 0.25\,\mathcal{L}_{\text{spike\_balance}} + 1.0\,\mathcal{L}_{\text{sensory}}$$

```bash
.venv/bin/python experiments/train.py \
    --config results/spwm_v0_12_4_train_representation/config.yaml \
    --mode train_representation \
    --epochs 160 \
    --output-dir results/spwm_v0_12_4_train_representation
```

### Phase 1 (`train_m1`): Action-Conditioned Latent Attractor Predictor ($H=100$)
Freezes all Phase 0 representation modules (`requires_grad = False`), detaches latent trajectory targets $\mathbf{z}^*_{\text{target}} = \text{stop\_grad}(\mathbf{z}^*)$, and trains the Latent Predictor $M_1$ + `LatentCritic` across 4 irreversible Latch phases: `(1) Warmup Nominal (w_t=0.60)` $\to$ `(2) Sensory Fade-Out (w_t: 0.60 -> 0)` $\to$ `(3) AR(1) Noise Ramp (alpha: 0 -> 0.25)` $\to$ `(4) DAgger Replay (ratio: 0.10 -> 0.50)` with contact-heft weighting (`2.0x` on $\mathbf{q}$, `3.0x` on $\mathbf{p}$):

$$\hat{\mathbf{z}}_{t+1} = M_1(\hat{\mathbf{z}}_t, \mathbf{u}_t), \qquad \mathcal{L}_{\text{Phase1}} = \mathcal{L}_{M_1} + w_{\text{critic}}\,\mathcal{L}_{\text{critic}}$$

```bash
.venv/bin/python experiments/train.py \
    --config results/spwm_v0_12_4_train_m1/config.yaml \
    --mode train_m1 \
    --representation-checkpoint results/spwm_v0_12_4_train_representation/model.pt \
    --epochs 240 \
    --phase-horizon 100 \
    --weighted-sampling \
    --output-dir results/spwm_v0_12_4_train_m1
```

---

## 5. Downstream Control, Baselines & Benchmarks

```bash
# Evaluate a trained checkpoint across horizons H in {1, 5, 10, 25, 50, 100}
.venv/bin/python experiments/evaluate.py \
    --checkpoint results/spwm_v0_12_4_train_m1/model.pt \
    --config results/spwm_v0_12_4_train_m1/config.yaml

# Benchmark downstream closed-loop Physics-in-the-Loop (PIL) Latent MPPI control (50 unseen episodes, t >= t_c)
.venv/bin/python experiments/benchmark_mppi_control.py \
    --checkpoint results/spwm_v0_12_4_train_m1/model.pt \
    --config results/spwm_v0_12_4_train_m1/config.yaml \
    --output-dir results/mppi_benchmark \
    --num-episodes 50 \
    --device cpu \
    --physics-in-loop

# Benchmark comparative non-spiking world models (RSSM, Deep GRU, ConvLSTM)
.venv/bin/python experiments/train_baselines.py --models all --epochs 60 --total-trajectories 4000

# Benchmark continuous leaky rate ablation (ANNWorldModel)
.venv/bin/python experiments/train.py --config results/spwm_v0_12_4_train_m1/config.yaml --model-type ann_rate

# Run neuromorphic Synaptic Operations (SOPs) and energy profiling
.venv/bin/python experiments/profile_energy_sops.py

# Run statistical multi-seed validation (seeds 42, 101, 2024)
.venv/bin/python experiments/train_multiseed.py

# Run zero-shot sim-to-real sensor & mechanical robustness stress tests in PIL
.venv/bin/python experiments/benchmark_robustness_stress_test.py --physics-in-loop --num-mppi-episodes 50 --num-sensor-trajectories 400
```

### Execution Confirmation & Validation Criteria (`SPWM-v0.12.4`)

- **Multi-Horizon Open-Loop Stability ($N=400$, $H=100$, $t_{\text{start}}=4$, 3 Seeds `42, 101, 2024`)**: Joint position MSE `0.00018 ± 0.00011` ($H=1$) → `0.00240 ± 0.00037 rad²` ($H=100$), and lowest long-horizon joint velocity MSE at `0.00971 ± 0.00177 rad²/s²` (`2.16×` lower than Continuous RSSM `0.02101 ± 0.00044`, `11.5×` lower than Deep GRU `0.11191`).
- **Neuromorphic Sparsity & Loihi-2 Energy (3-Seed Mean ± Std)**: Measured ALIF spike rate **`12.15 ± 0.19%`** (`8,150 ± 129` SOPs/step), consuming **`12.22 ± 0.19 nJ/step`** on Loihi-2 (**`4.94 ± 0.08×`** energy reduction vs. `60.36 nJ/step` on Jetson Orin GPU MACs).
- **Closed-Loop Physics-in-the-Loop (PIL) Control (`50` Episodes `[3600, 3649]`, `payload_attached=(t >= t_c)` where $t_c = t_{\text{contact}} - 7$ and $t_{\text{grasp}} = t_c + 7 = t_{\text{contact}}$)**: Receding-horizon warm-start, 2-stage $SE(3)$ lookahead, causal residual-gated active perception during $\phi_t = 1$, AR(1)-matched noise, and slew-rate clamping (`max_action_delta=0.10`) achieve **`86.0%`** grasp success (`43/50`; `< 3.0 cm` 3-step settling, `< 0.05 rad` $\text{SO}(3)$ error, $\tau_{\text{vel}} = \max(1.5, 1.1\|\dot{\mathbf{q}}^*_{t_{\text{grasp}}}\|_2)$) on canonical Seed `42` (`0.86 ± 0.80 cm` min error, `1.57 ± 1.00 cm` terminal error, `23.3 ms/step` unthrottled / `36.7 ms/step` under `OMP_NUM_THREADS=2`), **`58.0 ± 27.1%`** across 3 seeds under original `N=6` Tier-2 selection, and **`60.7 ± 2.3%`** (`11.7×` lower inter-seed std) under `N=25` validation-split (`[3200, 3224]`) selection — outperforming 3-seed `ANN-Rate` (`37.3 ± 9.5%`), `Continuous RSSM` (`15.3 ± 4.6%` Open-Loop PIL; `0.0 ± 0.0%` Online Posterior), `Deep GRU` (`1.3 ± 2.3%` Open-Loop PIL; `0.0 ± 0.0%` Online Obs.), and `Reactive BC` (`0.0%`). Disentangling Stage 4 shows 100% of the `+14.0 pp` gain (`72.0% -> 86.0%`) comes from 2-stage $SE(3)$ lookahead (`Stage 4a`, $w_t \equiv 0$), while the residual gate maintains `0.0%` nominal false triggers across all 3 seeds and activates selectively under mechanical OOD drift (Seed `2024`: `4.0%` episodes / `0.5%` steps). Across 3-seed zero-shot PIL robustness (`[3600, 3649]`), SPWM achieves `58.0 ± 27.1%` (Nominal), `55.3 ± 19.4%` (`1.0%` Noise, `-4.6%` rel), `39.3 ± 33.7%` (`4.80 kg`), `40.7 ± 27.3%` (`0.80×` Friction), and `44.0 ± 28.0%` (`1.20×` Friction), with Seed `42` retaining `90.7%` of nominal success at `4.80 kg` (`86.0% -> 78.0%`, `-9.3%` rel).
- **End-to-End Closed-Loop Energy**: `1.0979 mJ/step` (`21.96 mW` @ 20 Hz) on Jetson Orin + Loihi-2 core (`12.22 ± 0.19 nJ/step` recurrent core), vs. `1.7490 mJ/step` (`Deep GRU`), `2.1236 mJ/step` (`Continuous RSSM`), and `100.18 mJ/step` (`Continuous ConvLSTM`).
- **Documentation**: Consolidated Single Source of Truth in [**docs_verified/README.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/README.md), plus [**figures/main_v0_12_4.tex**](file:///Users/Mattia/Desktop/Studies/Temp/figures/main_v0_12_4.tex) and legacy [**docs/History.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/History.md).

---

## 6. Comprehensive Documentation Suite (`docs_verified/` SSOT)

- [**docs_verified/README.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/README.md): Audit verdict, testing precision progression, and discrepancy reconciliations over legacy `docs/`.
- [**docs_verified/01_SYSTEM_AND_ARCHITECTURE.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/01_SYSTEM_AND_ARCHITECTURE.md): Repo skeleton, 4-layer contracts, Unity `6000.5.7f1` / Python PIL physics, dataset splits, neural equations, and exact parameter census (`392,141` total / `317,900` active).
- [**docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/02_TRAINING_CURRICULUM_AND_MPPI.md): Phase 0 & Phase 1 training strategies, exact losses, 4-phase latch curriculum, open-loop soft-fade vs. closed-loop residual handover, MPPI cost, Tier-2 validation, and training CLIs.
- [**docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs_verified/03_BENCHMARKS_EVIDENCE_AND_REPLICATION.md): Checkpoint provenance & SHA-256 matrix, Tables 1–5, Figs 1–4, validation sweeps, forensic failure decomposition, task-log links, and full replication runbook.
- [**PAPER_CONTEXT.md**](file:///Users/Mattia/Desktop/Studies/Temp/PAPER_CONTEXT.md): Token-compacted compilation of SSOT benchmark JSONs and YAML configs built via [`scripts/build_paper_context.py`](file:///Users/Mattia/Desktop/Studies/Temp/scripts/build_paper_context.py).
