# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test (SPWM-v0.12.3)

**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).
Evaluated on frozen production checkpoint: `results/spwm_v0_12_3_train_m1/model.pt`.

## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)

Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \in \{1, 10, 50, 100\}$:

| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions

Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\text{iter}}=3$) over 50 canonical test episodes:

| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 50 | **48.0%** | **0.99 ± 0.61 cm** | 2.68 cm | 0.37 rad/s | 25.2 ms |
| **Sensory Noise Injection (1.0%)** | 50 | **32.0%** | **1.17 ± 0.59 cm** | 3.06 cm | 0.39 rad/s | 25.1 ms |
| **Payload Extrapolation (4.80 kg)** | 50 | **44.0%** | **1.52 ± 1.08 cm** | 2.89 cm | 0.49 rad/s | 25.1 ms |
| **Actuator Drift (0.80x Friction)** | 50 | **44.0%** | **1.53 ± 1.06 cm** | 3.03 cm | 0.41 rad/s | 24.8 ms |
| **Actuator Drift (1.20x Friction)** | 50 | **52.0%** | **1.50 ± 1.05 cm** | 2.82 cm | 0.40 rad/s | 25.1 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 45.5% | 33.3% | 80.0% | 29.4% | 58.8% | 56.2% |
| **Sensory Noise Injection (1.0%)** | 22.7% | 22.2% | 70.0% | 23.5% | 47.1% | 25.0% |
| **Payload Extrapolation (4.80 kg)** | 31.8% | 27.8% | 100.0% | 47.1% | 41.2% | 43.8% |
| **Actuator Drift (0.80x Friction)** | 31.8% | 27.8% | 100.0% | 47.1% | 47.1% | 37.5% |
| **Actuator Drift (1.20x Friction)** | 50.0% | 33.3% | 90.0% | 70.6% | 41.2% | 43.8% |
