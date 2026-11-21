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
| **Nominal Baseline** | 50 | **62.0%** | **0.85 ± 0.59 cm** | 2.02 cm | 0.51 rad/s | 39.8 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **54.0%** | **0.76 ± 0.42 cm** | 2.31 cm | 0.47 rad/s | 32.1 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **30.0%** | **1.20 ± 0.79 cm** | 2.80 cm | 0.73 rad/s | 32.6 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (0.80x Friction)** | 50 | **44.0%** | **1.15 ± 0.72 cm** | 2.65 cm | 0.62 rad/s | 29.7 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (1.20x Friction)** | 50 | **34.0%** | **1.20 ± 0.75 cm** | 2.44 cm | 0.55 rad/s | 30.0 ms | 0.0% | 0.0% | 0.000 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 54.5% | 55.6% | 90.0% | 88.2% | 64.7% | 31.2% |
| **Sensory Noise Injection (1.0%)** | 36.4% | 66.7% | 70.0% | 82.4% | 41.2% | 37.5% |
| **Payload Extrapolation (4.80 kg)** | 18.2% | 27.8% | 60.0% | 41.2% | 29.4% | 18.8% |
| **Actuator Drift (0.80x Friction)** | 40.9% | 38.9% | 60.0% | 82.4% | 29.4% | 18.8% |
| **Actuator Drift (1.20x Friction)** | 31.8% | 33.3% | 40.0% | 41.2% | 52.9% | 6.2% |
