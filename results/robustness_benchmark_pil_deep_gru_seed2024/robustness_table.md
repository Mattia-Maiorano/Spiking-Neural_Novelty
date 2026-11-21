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
| **Nominal Baseline** | 50 | **76.0%** | **1.46 ± 1.04 cm** | 1.93 cm | 0.22 rad/s | 54.6 ms | 4.0% | 0.5% | 0.020 |
| **Sensory Noise Injection (1.0%)** | 50 | **78.0%** | **1.24 ± 0.90 cm** | 1.60 cm | 0.19 rad/s | 55.3 ms | 24.0% | 7.2% | 0.090 |
| **Payload Extrapolation (4.80 kg)** | 50 | **18.0%** | **4.88 ± 3.68 cm** | 5.19 cm | 0.43 rad/s | 56.9 ms | 76.0% | 21.2% | 0.315 |
| **Actuator Drift (0.80x Friction)** | 50 | **16.0%** | **4.87 ± 3.47 cm** | 5.13 cm | 0.40 rad/s | 57.8 ms | 74.0% | 21.8% | 0.300 |
| **Actuator Drift (1.20x Friction)** | 50 | **28.0%** | **4.63 ± 3.63 cm** | 4.92 cm | 0.37 rad/s | 57.8 ms | 70.0% | 19.8% | 0.299 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 81.8% | 72.2% | 70.0% | 82.4% | 64.7% | 81.2% |
| **Sensory Noise Injection (1.0%)** | 81.8% | 66.7% | 90.0% | 64.7% | 82.4% | 87.5% |
| **Payload Extrapolation (4.80 kg)** | 0.0% | 33.3% | 30.0% | 17.6% | 0.0% | 37.5% |
| **Actuator Drift (0.80x Friction)** | 4.5% | 16.7% | 40.0% | 23.5% | 0.0% | 25.0% |
| **Actuator Drift (1.20x Friction)** | 9.1% | 44.4% | 40.0% | 47.1% | 0.0% | 37.5% |
