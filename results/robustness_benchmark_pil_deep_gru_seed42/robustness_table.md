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
| **Nominal Baseline** | 50 | **82.0%** | **0.83 ± 0.62 cm** | 1.71 cm | 0.30 rad/s | 58.0 ms | 10.0% | 2.0% | 0.098 |
| **Sensory Noise Injection (1.0%)** | 50 | **72.0%** | **1.84 ± 1.13 cm** | 2.06 cm | 0.37 rad/s | 45.6 ms | 46.0% | 15.8% | 0.165 |
| **Payload Extrapolation (4.80 kg)** | 50 | **14.0%** | **2.68 ± 0.96 cm** | 4.69 cm | 0.31 rad/s | 39.3 ms | 10.0% | 1.8% | 0.314 |
| **Actuator Drift (0.80x Friction)** | 50 | **10.0%** | **2.72 ± 1.00 cm** | 4.76 cm | 0.26 rad/s | 39.7 ms | 10.0% | 2.2% | 0.378 |
| **Actuator Drift (1.20x Friction)** | 50 | **16.0%** | **2.60 ± 1.02 cm** | 4.64 cm | 0.27 rad/s | 40.3 ms | 6.0% | 1.2% | 0.371 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 68.2% | 88.9% | 100.0% | 82.4% | 76.5% | 87.5% |
| **Sensory Noise Injection (1.0%)** | 77.3% | 50.0% | 100.0% | 76.5% | 82.4% | 56.2% |
| **Payload Extrapolation (4.80 kg)** | 22.7% | 11.1% | 0.0% | 0.0% | 23.5% | 18.8% |
| **Actuator Drift (0.80x Friction)** | 18.2% | 5.6% | 0.0% | 0.0% | 17.6% | 12.5% |
| **Actuator Drift (1.20x Friction)** | 22.7% | 16.7% | 0.0% | 0.0% | 29.4% | 18.8% |
