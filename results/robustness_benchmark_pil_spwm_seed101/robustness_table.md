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
| **Nominal Baseline** | 50 | **56.0%** | **0.90 ± 0.65 cm** | 2.19 cm | 0.47 rad/s | 35.9 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **60.0%** | **0.90 ± 0.52 cm** | 2.11 cm | 0.54 rad/s | 36.5 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **24.0%** | **1.28 ± 0.81 cm** | 2.84 cm | 0.64 rad/s | 36.4 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (0.80x Friction)** | 50 | **36.0%** | **1.12 ± 0.70 cm** | 2.61 cm | 0.56 rad/s | 36.4 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (1.20x Friction)** | 50 | **32.0%** | **1.22 ± 0.66 cm** | 2.54 cm | 0.52 rad/s | 36.6 ms | 0.0% | 0.0% | 0.000 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 27.3% | 72.2% | 90.0% | 88.2% | 47.1% | 31.2% |
| **Sensory Noise Injection (1.0%)** | 54.5% | 55.6% | 80.0% | 76.5% | 52.9% | 50.0% |
| **Payload Extrapolation (4.80 kg)** | 9.1% | 33.3% | 40.0% | 5.9% | 47.1% | 18.8% |
| **Actuator Drift (0.80x Friction)** | 18.2% | 44.4% | 60.0% | 29.4% | 52.9% | 25.0% |
| **Actuator Drift (1.20x Friction)** | 27.3% | 27.8% | 50.0% | 35.3% | 41.2% | 18.8% |
