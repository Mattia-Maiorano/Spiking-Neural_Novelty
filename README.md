# SPWM: Spiking Predictive World Model (v0.1.0)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v0.1.0** is a state-of-the-art neuromorphic architecture combining continuous event encoding via `SpatialSoftmax`, symplectic phase-space decomposition $(q, p)$, a 256-unit dual-timescale ALIF memory core, and an adaptive **Neuromodulatory Gain Field Corrector** for asymptotic drift suppression.

---

## 1. Overview & Architecture

SPWM processes event-camera sensory streams with $\mathcal{O}(1)$ temporal memory scaling and sparse, asynchronous spiking dynamics.

```text
Event Stream [B, T, 2, 32, 32]
              │
              ▼
Continuous Event Encoder + SpatialSoftmax Keypoint Bottleneck (16 keypoints)
              │
              ▼
Predictive Coding Core & Error Routing:  ϵ_t = x_t - W_pred z_(t-1)
              │
              ▼
Symplectic Latent Phase Space: z_t = [q_t, p_t] ∈ ℝ^128
  ├─ Geometric Coordinates: q_t ∈ [-1, 1]^32 (Topological SpatialSoftmax Anchor)
  └─ Canonical Momentum:    p_t ∈ ℝ^96      (ALIF Recurrent Velocity/Context)
              │
              ▼
Fast Predictor (Local Dynamics): Δz_fast = f_θ(z_t)
              │
              ▼
Slow Neuromodulatory Corrector (Gain Field System):
  ├─ 128-unit Slow ALIF Population (integrates context & transitions)
  ├─ Continuous Volume Gain Mask: G_t = tanh(W u) ∈ [-1, 1]^128
  └─ Transition Modulation: Δz_mod = Δz_fast · (1 + α_gain · G_t)
              │
              ▼
Euclidean Clamping & Physical Momentum Dissipation: q_t ∈ [-1, 1]^32
              │
       ┌──────┴──────┐
       ▼             ▼
Physical Probe   Multi-Step Rollout Loss & Asymptotic Stability
(x, y, vx, vy)   (Exposure bias mitigation & drift suppression)
```

---

## 2. Key Features

- **SpatialSoftmax Topological Bottleneck:** Continuous convolutional encoder extracting analytical $(u_k, v_k)$ centers of mass, preserving Euclidean geometry and topological invariants without non-Euclidean latent distortion.
- **Symplectic Latent Decomposition:** Explicit partition of $z = (q, p)$ where coordinates $q \in [-1, 1]^{32}$ track position, and canonical momentum $p \in \mathbb{R}^{96}$ integrates velocity and temporal context via dual-timescale ALIF neurons ($\beta_{\text{fast}} = 0.90$, $\beta_{\text{slow}} = 0.985$).
- **Neuromodulatory Gain Field Corrector:** Slow ALIF population acts as a neuromodulator emitting a continuous gain mask $G_t \in [-1, 1]^{128}$. Hot spots ($>0$) amplify agile dynamic maneuvers, while cold spots ($<0$) dampen compounding drift during out-of-distribution (OOD) trajectories.
- **Two-Stage Decoupled Optimization Pipeline:**
  1. **Stage 1 (`train_predictor`):** Fast predictor is optimized on clean nominal trajectories for one-step reconstruction, variance regularization, and frozen physical probe alignment.
  2. **Stage 2 (`train_corrector`):** Fast predictor is strictly frozen; the gain field corrector is trained on long horizons ($H=25..50$) with stochastic drift injection and burn-in stabilization.

---

## 3. Installation

Ensure you have Python $\ge 3.11$:

```bash
# Clone the repository
git clone https://github.com/Mattia-Maiorano/Spiking-Neural_Novelty.git
cd Spiking-Neural_Novelty

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

Run the unit test suite:

```bash
pytest
```

---

## 4. Three-Stage Training Pipeline

### Stage 1: Train Representation (Vision Bottleneck & Readout Probes)

Trains the `SpatialSoftmax` event encoder and `physical_decoder` kinematic probes from scratch on single-frame / kinematic supervision:

```bash
python experiments/train.py \
    --mode train_representation \
    --epochs 100 \
    --output-dir results/spwm_v0_1_0_train_representation
```

### Stage 2: Train Predictor (Frozen Representation + Nominal Physics)

Loads the pre-trained representation checkpoint, freezes the encoder and probe, and trains the ALIF recurrent predictor / dynamics on multi-step rollouts:

```bash
python experiments/train.py \
    --mode train_predictor \
    --representation-checkpoint results/spwm_v0_1_0_train_representation/model.pt \
    --epochs 100 \
    --output-dir results/spwm_v0_1_0_train_predictor
```
*(Note: `--representation-checkpoint` automatically defaults to `results/{project_name}_train_representation/model.pt` if omitted).*

### Stage 3: Train Corrector (Frozen Representation + Frozen Predictor)

Freezes both the encoder and predictor, loading the predictor checkpoint and optimizing the neuromodulatory gain field corrector under drift perturbations:

```bash
python experiments/train.py \
    --mode train_corrector \
    --predictor-checkpoint results/spwm_v0_1_0_train_predictor/model.pt \
    --epochs 100 \
    --output-dir results/spwm_v0_1_0_train_corrector
```
*(Note: `--predictor-checkpoint` automatically defaults to `results/{project_name}_train_predictor/model.pt` if omitted).*

---

## 5. Evaluation & Rollouts

Evaluate autonomous rollout stability across horizons $H \in \{1, 5, 10, 25, 50, 100\}$ using the run's saved config:

```bash
# Evaluate predictor checkpoint nominally
python experiments/evaluate.py \
    --checkpoint results/spwm_v0_1_0_train_predictor/model_predictor.pt \
    --config results/spwm_v0_1_0_train_predictor/config.yaml

# Evaluate corrector checkpoint with drift injection
python experiments/evaluate.py \
    --checkpoint results/spwm_v0_1_0_train_corrector/model.pt \
    --config results/spwm_v0_1_0_train_corrector/config.yaml \
    --use-corrector
```

---

## 6. Visualization & Scientific Figures

Generate publication-ready figures:

```bash
python experiments/generate_figures.py \
    --results-dir results/ \
    --output-dir results/figures/
```

---

## 7. Documentation & Evolution History

- [Architecture.md](file:///Users/Mattia/Desktop/Studies/Temp/Architecture.md): Complete technical blueprint of SPWM-v0.1.0.
- [History.md](file:///Users/Mattia/Desktop/Studies/Temp/History.md): Historical changelog across iterations v1 through v0.1.0.
- [EXPERIMENTS.md](file:///Users/Mattia/Desktop/Studies/Temp/docs/EXPERIMENTS.md): Empirical research report and performance benchmarks.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
