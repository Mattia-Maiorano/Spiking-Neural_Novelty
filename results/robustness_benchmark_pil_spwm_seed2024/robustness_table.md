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
| **Nominal Baseline** | 50 | **32.0%** | **0.94 ± 0.64 cm** | 2.93 cm | 0.47 rad/s | 36.5 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **34.0%** | **0.81 ± 0.57 cm** | 3.06 cm | 0.36 rad/s | 36.5 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **16.0%** | **1.01 ± 0.65 cm** | 2.89 cm | 0.48 rad/s | 36.4 ms | 4.0% | 0.5% | 0.165 |
| **Actuator Drift (0.80x Friction)** | 50 | **16.0%** | **1.13 ± 0.75 cm** | 3.15 cm | 0.41 rad/s | 36.4 ms | 4.0% | 0.5% | 0.229 |
| **Actuator Drift (1.20x Friction)** | 50 | **24.0%** | **1.11 ± 0.67 cm** | 3.09 cm | 0.39 rad/s | 36.5 ms | 4.0% | 0.5% | 0.169 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 13.6% | 16.7% | 100.0% | 35.3% | 17.6% | 43.8% |
| **Sensory Noise Injection (1.0%)** | 18.2% | 38.9% | 60.0% | 41.2% | 11.8% | 50.0% |
| **Payload Extrapolation (4.80 kg)** | 9.1% | 16.7% | 30.0% | 29.4% | 5.9% | 12.5% |
| **Actuator Drift (0.80x Friction)** | 4.5% | 11.1% | 50.0% | 17.6% | 5.9% | 25.0% |
| **Actuator Drift (1.20x Friction)** | 18.2% | 5.6% | 70.0% | 35.3% | 11.8% | 25.0% |
