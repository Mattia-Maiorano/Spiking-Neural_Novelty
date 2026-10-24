# SPWM: Spiking Predictive World Model (v0.3.1)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.3.1** is a neuromorphic predictive world model engineered under a **4-Layer Strict Contract Architecture**. It combines resolution-adaptive `SpatialSoftmax` event encoding, canonical phase-space decomposition $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t)$, a dual-timescale Adaptive Leaky Integrate-and-Fire (`ALIFCell`) memory core, a strictly decoupled **Phase 0 Grounded Representation**, and a **Symplectic Latent Predictor** ($M_1$) utilizing semi-implicit Euler integration for conservative, drift-resistant long-horizon autonomous rollouts.

---

## 1. Architectural Overview

```text
Event Stream [B, T, 2, H, W]
              │
              ▼
EventEncoder + Dynamic SpatialSoftmax Keypoint Bottleneck (K keypoints)
   ├─ sensory_x      [B, D_enc]   (Projected visual feature embedding)
   └─ keypoints      [B, 2*K]     (Normalized [-1, 1] expected centers of mass)
              │
              ▼
Top-Down Predictive Coding Error:  ϵ_t = x_t - W_pred z_(t-1)
              │
              ▼
SpikingLatentDynamics + Dual-Timescale ALIF Memory Core
   ├─ Fast ALIF Population (β_adapt = 0.90)  &  Slow ALIF Population (β_adapt = 0.985)
   └─ Symplectic Phase-Space State: z_t = [q_t, p_t] ∈ ℝ^D
        ├─ Geometric Coordinates:   q_t ∈ ℝ^(q_dim)  (q_dim = 2*K)
        └─ Canonical Momentum:      p_t ∈ ℝ^(p_dim)
              │
              ▼
Symplectic Latent Predictor (M1 - Semi-Implicit Euler Integration):
   ├─ Momentum Update:         Δp_t = MLP(z_t)
   ├─ Next Momentum:           p_(t+1) = p_t + Δp_t
   ├─ Coordinate Update:       q_(t+1) = q_t + Δt · p_(t+1)[..., :q_dim]
   └─ Next Latent State:       z_(t+1) = [q_(t+1), p_(t+1)]
              │
              ▼
PhysicalDecoder (Linear Kinematic Readout Heads: decode_pos, decode_vel)
   └─ Decoded Kinematics: [x, y, vx, vy] ∈ ℝ^(4·N)
```

---

## 2. 4-Layer Strict Contract Workspace

The repository enforces a strict unidirectional dependency DAG verified on every test run by [`tests/test_architecture_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_architecture_contracts.py) and [`tests/test_dead_code.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_dead_code.py):

| Layer | Package | Responsibilities | Allowed `spwm.*` Imports |
| :--- | :--- | :--- | :--- |
| **Layer 0** | [`spwm/contracts/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py) | Frozen `@dataclass(frozen=True)` schemas (`ExperimentConfig`), state containers (`ALIFState`, `SPWMState`, `RolloutResult`), runtime tensor validators (`assert_tensor_contract`), and `Protocol` interfaces (`PhaseStrategy`, `PredictorProtocol`). | None (Zero internal imports) |
| **Layer 1** | [`spwm/models/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py), [`spwm/data/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py) | Pure neural modules (`ALIFCell`, `MultiTimescaleMemory`, `EventEncoder`, `SpikingLatentDynamics`, `LatentPredictor`, `PhysicalDecoder`, `SPWM`), unified `rollout_latent`, and synthetic event-camera physics. | Layer 0 (`spwm.contracts`) |
| **Layer 2** | [`spwm/training/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/__init__.py) | Composable loss modules (`SPWMLoss`), isolated $<150$-line phase strategies (`RepresentationPhaseStrategy`, `M1PhaseStrategy`), and `BaseTrainer` with `strict=True` checkpoint scoping. | Layers 0–1 (`spwm.contracts`, `spwm.models`, `spwm.data`) |
| **Layer 3** | [`spwm/evaluation/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/evaluation/__init__.py) | Multi-horizon rollout evaluation (`RolloutEvaluator`), segmented telemetry resumes (`metrics_resume.py`), and publication figure generation (`visualizer.py`). | Layers 0–2 |

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

## 4. Two-Phase Representation / Symplectic Predictor Pipeline

### Phase 0 (`train_representation`): Grounded State Estimator ($t \in [0, T_{\text{obs}}]$)
Pre-trains `encoder`, `dynamics`, `sensory_predictor`, and `physical_decoder` purely as a grounded state estimator over sensory streams (no autoregressive rollouts; predictor inactive):

$$\mathcal{L}_{\text{Phase0}} = \mathcal{L}_{\text{kinematic}} + \mathcal{L}_{\text{derivative}} + \mathcal{L}_{\text{smoothness}} + \mathcal{L}_{\text{sensory}}$$

```bash
python experiments/train.py \
    --mode train_representation \
    --epochs 100 \
    --output-dir results/spwm_v0_3_1_train_representation
```

### Phase 1 (`train_m1`): Symplectic Predictor on Pristine Latent Targets ($t \in [0, H-1]$)
Freezes all Phase 0 representation modules (`requires_grad = False`), detaches latent trajectory targets $z^*_{\text{target}} = z^*.\text{detach}()$, and trains only the Symplectic Predictor $M_1$ autoregressively in pure latent space:

$$\Delta p_t = \text{MLP}(z_t), \quad p_{t+1} = p_t + \Delta p_t, \quad q_{t+1} = q_t + \Delta t \cdot p_{t+1}[:q\_dim]$$
$$\mathcal{L}_{M_1} = \text{MSE}(\hat{z}_{t+1}, z^*_{\text{target}, t+1})$$

```bash
python experiments/train.py \
    --mode train_m1 \
    --representation-checkpoint results/spwm_v0_3_1_train_representation/model.pt \
    --epochs 200 \
    --phase-horizon 50 \
    --output-dir results/spwm_v0_3_1_train_m1
```

---

## 5. Evaluation & Scientific Figures

```bash
# Evaluate a trained checkpoint across horizons H in {1, 5, 10, 25, 50, 100}
python experiments/evaluate.py \
    --checkpoint results/spwm_v0_3_1_train_m1/model.pt \
    --config results/spwm_v0_3_1_train_m1/config.yaml

# Generate publication figures (Figures 1–7)
python experiments/generate_figures.py \
    --results-dir results/ \
    --output-dir results/figures/
```

### Execution Confirmation & Validation Criteria

- **Autonomous Symplectic Rollout (`relay_rollout.png`)**: Qualitative check verifying continuous, stable phase-space trajectories without energy explosion or damping.
- **Rollout Degradation Curve (`fig3_multi_step_degradation.png`)**: Quantitative confirmation that position and velocity errors stay bounded across horizons up to $H=100$.
- **Multi-Channel & Multi-Metric Consistency**: Low error across all coordinates, balanced ALIF spike rates, and stable decoded physical kinematics.

---

## 6. Comprehensive Documentation Suite

- [**docs/Architecture.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/Architecture.md): Complete master reference covering the 4-layer workspace, neural models' mathematical & tensor contracts, and the symplectic representation/predictor architecture.
- [**docs/WORKSPACE_AND_CONTRACTS.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORKSPACE_AND_CONTRACTS.md): Deep dive into Layer 0 contracts, AST anti-fallback guards, and the $<3$-file context boundary rules.
- [**docs/TRAINING_AND_LOSSES.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/TRAINING_AND_LOSSES.md): Granular mathematical specification of `RepresentationPhaseStrategy` and `M1PhaseStrategy`, symplectic Hamiltonian structure, and scoped checkpoint loading.
- [**docs/History.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/History.md): Evolution history across versions.
