# SPWM: Spiking Predictive World Model (v0.2.0)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.2.0** is a neuromorphic predictive world model engineered under a **4-Layer Strict Contract Architecture**. It combines resolution-adaptive `SpatialSoftmax` event encoding, symplectic phase-space decomposition $\mathbf{z}_t = (\mathbf{q}_t, \mathbf{p}_t)$, a dual-timescale Adaptive Leaky Integrate-and-Fire (`ALIFCell`) memory core, and a **Cascaded Temporal Relay** ($M_1 \to M_2 \leftrightarrow M_3$) trained via the **Convergence Funnel** for long-horizon autonomous rollouts.

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
        ├─ Geometric Configuration: q_t = clamp(0.8·q_(t-1) + 0.2·keypoints_t + Δq_t, -1, 1)
        └─ Canonical Momentum:      p_t = 0.99·p_(t-1) + Δp_t
              │
              ▼
Cascaded Temporal Relay (Stateless Residual Predictors, Horizon H):
   ├─ Steps [0, H-1]:      M1 (Nominal On-Manifold Physics)
   ├─ Steps [H, 2H-1]:     M2 (Convergence Funnel Manifold Recovery)
   ├─ Steps [2H, 3H-1]:    M3 (Sustained Post-Recovery Cruising)
   └─ Steps [3H, ∞):       Cyclic Alternation M2 ↔ M3
              │
              ▼
PhysicalDecoder (EMA Filter α = 0.85 + Linear Kinematic Readout Heads)
   └─ Decoded Kinematics: [x, y, vx, vy] ∈ ℝ^(4·N)
```

---

## 2. 4-Layer Strict Contract Workspace

The repository enforces a strict unidirectional dependency DAG verified on every test run by [`tests/test_architecture_contracts.py`](file:///Users/Mattia/Desktop/Studies/Temp/tests/test_architecture_contracts.py):

| Layer | Package | Responsibilities | Allowed `spwm.*` Imports |
| :--- | :--- | :--- | :--- |
| **Layer 0** | [`spwm/contracts/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/contracts/__init__.py) | Frozen `@dataclass(frozen=True)` schemas (`ExperimentConfig`), state containers (`ALIFState`, `SPWMState`, `RolloutResult`), runtime tensor validators (`assert_tensor_contract`), and `Protocol` interfaces (`PhaseStrategy`). | None (Zero internal imports) |
| **Layer 1** | [`spwm/models/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/models/__init__.py), [`spwm/data/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/data/__init__.py) | Pure neural modules (`ALIFCell`, `MultiTimescaleMemory`, `EventEncoder`, `SpikingLatentDynamics`, `CascadedTemporalRelay`, `PhysicalDecoder`, `SPWM`), unified `rollout_latent`, and synthetic event-camera physics. | Layer 0 (`spwm.contracts`) |
| **Layer 2** | [`spwm/training/`](file:///Users/Mattia/Desktop/Studies/Temp/spwm/training/__init__.py) | Composable loss modules (`SPWMLoss`, `ConvergenceFunnelLoss`), isolated $<150$-line phase strategies (`M1PhaseStrategy`, `M2PhaseStrategy`, `M3PhaseStrategy`), and `BaseTrainer` with `strict=True` checkpoint scoping. | Layers 0–1 (`spwm.contracts`, `spwm.models`, `spwm.data`) |
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

## 4. Three-Phase Cascaded Relay Training Pipeline

### Phase 1 (`train_m1`): Representation & Nominal Physics ($t \in [0, H-1]$)
Jointly optimizes `EventEncoder` (including `encoder.proj`), `PhysicalDecoder`, and `M1` while freezing `M2`, `M3`, and `SpikingLatentDynamics`:

```bash
python experiments/train.py \
    --mode train_m1 \
    --epochs 80 \
    --phase-horizon 10 \
    --output-dir results/spwm_v0_2_0_train_m1
```

### Phase 2 (`train_m2`): Convergence Funnel Recovery ($t \in [H, 2H-1] \cup [3H, 4H-1]$)
Loads the Phase 1 checkpoint via `load_m1_checkpoint` (`strict=True`), freezes `M1`, `M3`, `EventEncoder`, and `PhysicalDecoder`, and trains `M2` exclusively to recover from dirty $M_1/M_3$ terminal states along the interpolated **Convergence Funnel**:

```bash
python experiments/train.py \
    --mode train_m2 \
    --m1-checkpoint results/spwm_v0_2_0_train_m1/model.pt \
    --epochs 80 \
    --phase-horizon 10 \
    --m2-num-phases 2 \
    --output-dir results/spwm_v0_2_0_train_m2
```

### Phase 3 (`train_m3`): Sustained Cyclic Relay ($t \in [H, H_{\max}]$)
Loads `M1` and `M2` strictly, initializes `M3` as an exact clone of `M1` (`init_m3_from_m1`), freezes `M1`, and jointly optimizes `M2` and `M3` across alternating relay windows:

```bash
python experiments/train.py \
    --mode train_m3 \
    --m1-checkpoint results/spwm_v0_2_0_train_m1/model.pt \
    --m2-checkpoint results/spwm_v0_2_0_train_m2/model.pt \
    --epochs 80 \
    --phase-horizon 10 \
    --max-rollout-horizon 30 \
    --output-dir results/spwm_v0_2_0_train_m3
```

---

## 5. Evaluation & Scientific Figures

```bash
# Evaluate a trained checkpoint across horizons H in {1, 5, 10, 25, 50, 100}
python experiments/evaluate.py \
    --checkpoint results/spwm_v0_2_0_train_m3/model.pt \
    --config results/spwm_v0_2_0_train_m3/config.yaml

# Generate publication figures (Figures 1–7)
python experiments/generate_figures.py \
    --results-dir results/ \
    --output-dir results/figures/
```

### Execution Confirmation & Validation Criteria

To confirm the execution is actually good:

- **Relay Rollout (`relay_rollout.png`)**: Qualitative sanity check that handoffs ($M_1 \to M_2 \to M_3$) do not produce sudden discontinuities or divergent oscillations on the sample.
- **Rollout Degradation Curve (`fig3_multi_step_degradation.png`)**: Quantitative confirmation that position/velocity error stays low and scales stably across all test samples up to horizon $H=100$.
- **Multi-Channel & Multi-Metric Consistency**: Low error across all channels, balanced spike rates, and low velocity error.


---

## 6. Comprehensive Documentation Suite

- [**Architecture.md**](file:///Users/Mattia/Desktop/Studies/Temp/Architecture.md): Complete master reference covering the 4-layer workspace, neural models' mathematical & tensor contracts, and the $M_1/M_2/M_3$ loss and training pipeline.
- [**docs/WORKSPACE_AND_CONTRACTS.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/WORKSPACE_AND_CONTRACTS.md): Deep dive into Layer 0 contracts, AST anti-fallback guards, and the $<3$-file context boundary rules for small LLMs.
- [**docs/TRAINING_AND_LOSSES.md**](file:///Users/Mattia/Desktop/Studies/Temp/docs/TRAINING_AND_LOSSES.md): Granular mathematical specification of `M1PhaseStrategy`, `M2PhaseStrategy`, `M3PhaseStrategy`, the Convergence Funnel, and scoped `strict=True` checkpoint loading.
- [**History.md**](file:///Users/Mattia/Desktop/Studies/Temp/History.md): Evolution history across versions.
