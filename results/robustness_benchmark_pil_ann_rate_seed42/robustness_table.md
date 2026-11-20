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
| **Nominal Baseline** | 50 | **34.0%** | **0.83 ± 0.47 cm** | 2.75 cm | 0.59 rad/s | 29.4 ms |
| **Sensory Noise Injection (1.0%)** | 50 | **32.0%** | **0.84 ± 0.47 cm** | 3.06 cm | 0.57 rad/s | 31.4 ms |
| **Payload Extrapolation (4.80 kg)** | 50 | **24.0%** | **1.14 ± 0.57 cm** | 3.56 cm | 0.64 rad/s | 29.9 ms |
| **Actuator Drift (0.80x Friction)** | 50 | **32.0%** | **1.17 ± 0.73 cm** | 3.54 cm | 0.64 rad/s | 25.2 ms |
| **Actuator Drift (1.20x Friction)** | 50 | **28.0%** | **1.10 ± 0.72 cm** | 2.80 cm | 0.60 rad/s | 25.2 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 27.3% | 27.8% | 60.0% | 29.4% | 29.4% | 43.8% |
| **Sensory Noise Injection (1.0%)** | 27.3% | 16.7% | 70.0% | 41.2% | 29.4% | 25.0% |
| **Payload Extrapolation (4.80 kg)** | 22.7% | 16.7% | 40.0% | 35.3% | 0.0% | 37.5% |
| **Actuator Drift (0.80x Friction)** | 18.2% | 38.9% | 50.0% | 52.9% | 17.6% | 25.0% |
| **Actuator Drift (1.20x Friction)** | 13.6% | 16.7% | 80.0% | 41.2% | 17.6% | 25.0% |
