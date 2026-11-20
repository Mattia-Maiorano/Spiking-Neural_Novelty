# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test (SPWM-v0.12.3)

**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).
Evaluated on frozen production checkpoint: `results/spwm_v0_12_3_train_m1/model.pt`.

## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)

Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \in \{1, 10, 50, 100\}$:

| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions

Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\text{iter}}=3$) over 50 canonical test episodes:

| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) | Ep w_t>0 (%) | Step w_t>0 (%) | Mean w_t>0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 50 | **78.0%** | **0.82 ± 0.77 cm** | 1.58 cm | 0.33 rad/s | 34.3 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **78.0%** | **0.96 ± 0.86 cm** | 1.63 cm | 0.33 rad/s | 37.0 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **66.0%** | **1.21 ± 0.98 cm** | 1.90 cm | 0.46 rad/s | 39.1 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (0.80x Friction)** | 50 | **70.0%** | **1.41 ± 1.27 cm** | 1.95 cm | 0.41 rad/s | 39.4 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (1.20x Friction)** | 50 | **72.0%** | **1.22 ± 1.03 cm** | 1.95 cm | 0.42 rad/s | 28.8 ms | 0.0% | 0.0% | 0.000 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 68.8% | 73.3% | 89.5% | 87.5% | 56.2% | 88.9% |
| **Sensory Noise Injection (1.0%)** | 56.2% | 80.0% | 94.7% | 81.2% | 62.5% | 88.9% |
| **Payload Extrapolation (4.80 kg)** | 62.5% | 53.3% | 78.9% | 68.8% | 37.5% | 88.9% |
| **Actuator Drift (0.80x Friction)** | 56.2% | 53.3% | 94.7% | 56.2% | 62.5% | 88.9% |
| **Actuator Drift (1.20x Friction)** | 62.5% | 66.7% | 84.2% | 68.8% | 62.5% | 83.3% |
