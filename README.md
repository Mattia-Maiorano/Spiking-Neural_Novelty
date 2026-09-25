# SPWM: Spiking Predictive World Model (v1)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SPWM-v1** is a scientifically rigorous, modular PyTorch implementation of a **Spiking Predictive World Model**. It demonstrates that recurrent spiking neural networks (SNNs) can learn useful continuous predictive latent dynamics from event streams across multiple temporal timescales, operating with high computational sparsity.

---

## 1. Motivation

Standard artificial recurrent world models (e.g. RSSM, World Models, Dreamer) rely on continuous analog activations, non-local backpropagation through time (BPTT), and dense matrix multiplications computed on fixed clock cycles. 

Biological sensory and predictive systems, by contrast:
1. Process asynchronous, event-driven temporal signals (like biological retinas and neuromorphic event cameras).
2. Represent internal state through **sparse spikes and continuous membrane voltage dynamics**.
3. Maintain memory across **heterogeneous timescales** (from millisecond edge-tracking to multi-second momentum).

SPWM-v1 builds a foundational spiking predictive world model designed to prove that temporal spiking dynamics can autonomously imagine future states while retaining extreme sparsity and robustness to drift.

---

## 2. Architecture

```text
Input Environment (Moving Objects Kinematics)
                    │
                    ▼
   Event Camera Simulator (|ΔL| > θ_event)
                    │
       Event Streams [B, T, 2, 32, 32]
                    │
                    ▼
     Spiking Sensory Encoder (Conv-LIF)
                    │
                    ▼
  Multi-Timescale Recurrent Spiking Dynamics
      ┌─────────────────────────────┐
      │   Fast Dynamics (τ_fast)    │ (β = 0.80)
      │   Slow Dynamics (τ_slow)    │ (β = 0.98)
      └──────────────┬──────────────┘
                     │
                     ▼
             Latent State z_t
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
 Latent Predictor          Physical Decoder Probe
 z_hat_(t+1) = f(z_t)      (x, y, vx, vy)
        │                         │
        ▼                         ▼
 Prediction Error          Ground-Truth Kinematics Error
 ε_t = ||z_hat - z||       ||k - k_hat||
        │
        ▼
 Anti-Collapse Variance Loss + Multi-Step Rollout Loss
```

---

## 3. Installation

Ensure you have Python $\ge 3.11$:

```bash
# Clone the repository
git clone https://github.com/your-username/spwm.git
cd spwm

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

Verify the installation and unit test suite:

```bash
pytest tests/ -v
```

---

## 4. Quickstart

### Train SPWM-v1

To train the primary SPWM-v1 model:

```bash
python experiments/train.py \
    --config configs/experiments/spwm_v1.yaml \
    --seed 42 \
    --epochs 15 \
    --output-dir results/spwm_v1
```

This will automatically create:
```text
results/spwm_v1/
├── config.yaml
├── metrics.json
├── model.pt
├── run_metadata.json
├── training_log.csv
└── figures/
    ├── fig1_architecture.png
    ├── fig2_training_curves.png
    └── fig4_spike_raster.png
```

---

## 5. Evaluation & Autonomous Rollouts

Evaluate a trained checkpoint on both in-distribution test trajectories and out-of-distribution velocity extrapolation:

```bash
python experiments/evaluate.py \
    --checkpoint results/spwm_v1/model.pt \
    --config results/spwm_v1/config.yaml
```

Outputs:
- `test_metrics.json`: Rollout MSE and physical probe errors across horizons $H \in \{1, 5, 10, 25\}$.
- `rollout_predictions.npz`: Autonomous rollout trajectories vs ground truth.
- `figures/fig3_rollout_degradation.png`: Prediction degradation curve.

---

## 6. Systematic Ablation Studies

Execute the full 9-condition ablation study matrix:

```bash
python experiments/ablation.py \
    --base-config configs/experiments/spwm_v1.yaml \
    --epochs 8
```

Generates:
- `results/ablations/ablation_summary.json`
- `results/ablations/fig7_ablation_comparison.png`

---

## 7. Comparative Baselines

SPWM-v1 includes 3 benchmark baselines evaluated under identical conditions:
1. **Continuous GRU World Model**:
   ```bash
   python experiments/train.py --config configs/experiments/baseline_ann.yaml --output-dir results/baseline_gru
   ```
2. **Vanilla Recurrent SNN** (Single-timescale):
   ```bash
   python experiments/train.py --config configs/experiments/baseline_snn.yaml --output-dir results/baseline_vanilla_snn
   ```
3. **MLP Dynamics**:
   ```bash
   python experiments/train.py --config configs/experiments/baseline_mlp.yaml --output-dir results/baseline_mlp
   ```

### Compile Publication Figures:

```bash
python experiments/generate_figures.py --results-dir results/ --output-dir results/figures/
```

---

## 8. Key Scientific Findings

1. **Drift Resistance in Autonomous Rollouts**: While continuous GRU models have lower 1-step teacher-forced error, **SPWM-v1 degrades substantially slower over long autonomous horizons** ($H=10$ Rollout MSE: SPWM = `0.1659` vs GRU = `0.2659`).
2. **Multi-Timescale Memory Advantage**: Heterogeneous memory coupling ($\tau_{\text{fast}}, \tau_{\text{slow}}$) outperforms single-timescale SNN by over 18% on test rollout and 38.9% in ablation benchmarks.
3. **Neuromorphic Sparsity**: SPWM achieves predictive world modeling with an average neural spike rate of **17.1%**, operating predominantly silent.

For full empirical tables and methodology, see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## 9. Current Limitations (V1)

- **Non-local learning**: Trained via surrogate-gradient BPTT rather than on-chip local synaptic plasticity.
- **Deterministic Latent Space**: Predicts point estimates rather than full probability densities.
- **Passive World**: Models unactuated particle kinematics; does not yet support action conditioning.

---

## 10. Roadmap

- [x] **V1 — Foundational Spiking World Model**: Multi-timescale memory, event camera simulator, autonomous rollouts, surrogate BPTT.
- [ ] **V2 — True Predictive Coding**: State updates driven by online hierarchical prediction error feedback ($z_t \leftarrow z_t + K \epsilon_t$).
- [ ] **V3 — Local Learning**: Replacing BPTT with forward-only eligibility traces and local three-factor Hebbian plasticity (e-prop).
- [ ] **V4 — Probabilistic World Model**: Latent distribution estimation $p(z_{t+1} \mid z_t) \sim \mathcal{N}(\mu, \sigma^2)$.
- [ ] **V5 — Action-Conditioned Dynamics**: $p(z_{t+1} \mid z_t, a_t)$ for embodied agents.
- [ ] **V6 — Counterfactual Imagination & Planning**: Imagined rollouts for trajectory selection.
- [ ] **V7 — Neuromorphic Hardware Deployment**: Porting core event dynamics to Intel Loihi 2 / SynSense Speck hardware.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
