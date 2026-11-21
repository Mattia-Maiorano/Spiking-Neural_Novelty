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
| **Nominal Baseline** | 50 | **72.0%** | **1.14 ± 1.02 cm** | 2.06 cm | 0.37 rad/s | 54.4 ms | 4.0% | 0.5% | 0.045 |
| **Sensory Noise Injection (1.0%)** | 50 | **74.0%** | **1.47 ± 0.97 cm** | 1.90 cm | 0.32 rad/s | 54.8 ms | 34.0% | 9.0% | 0.203 |
| **Payload Extrapolation (4.80 kg)** | 50 | **38.0%** | **2.09 ± 1.56 cm** | 3.15 cm | 0.74 rad/s | 56.4 ms | 2.0% | 0.2% | 0.400 |
| **Actuator Drift (0.80x Friction)** | 50 | **54.0%** | **2.12 ± 1.54 cm** | 3.25 cm | 0.66 rad/s | 57.6 ms | 2.0% | 0.2% | 0.400 |
| **Actuator Drift (1.20x Friction)** | 50 | **46.0%** | **2.09 ± 1.50 cm** | 3.02 cm | 0.64 rad/s | 57.9 ms | 2.0% | 0.2% | 0.400 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 68.2% | 61.1% | 100.0% | 88.2% | 64.7% | 62.5% |
| **Sensory Noise Injection (1.0%)** | 63.6% | 77.8% | 90.0% | 76.5% | 82.4% | 62.5% |
| **Payload Extrapolation (4.80 kg)** | 9.1% | 50.0% | 80.0% | 47.1% | 35.3% | 31.2% |
| **Actuator Drift (0.80x Friction)** | 31.8% | 61.1% | 90.0% | 76.5% | 47.1% | 37.5% |
| **Actuator Drift (1.20x Friction)** | 18.2% | 50.0% | 100.0% | 58.8% | 41.2% | 37.5% |
