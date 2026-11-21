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
| **Nominal Baseline** | 50 | **62.0%** | **0.79 ± 0.55 cm** | 1.89 cm | 0.46 rad/s | 39.8 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **56.0%** | **0.80 ± 0.48 cm** | 1.78 cm | 0.43 rad/s | 32.0 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **48.0%** | **0.96 ± 0.72 cm** | 2.22 cm | 0.54 rad/s | 32.6 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (0.80x Friction)** | 50 | **46.0%** | **1.08 ± 0.69 cm** | 2.13 cm | 0.50 rad/s | 29.6 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (1.20x Friction)** | 50 | **40.0%** | **0.88 ± 0.50 cm** | 1.95 cm | 0.48 rad/s | 29.9 ms | 0.0% | 0.0% | 0.000 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 50.0% | 55.6% | 100.0% | 76.5% | 58.8% | 50.0% |
| **Sensory Noise Injection (1.0%)** | 36.4% | 66.7% | 80.0% | 58.8% | 52.9% | 56.2% |
| **Payload Extrapolation (4.80 kg)** | 36.4% | 61.1% | 50.0% | 58.8% | 47.1% | 37.5% |
| **Actuator Drift (0.80x Friction)** | 27.3% | 50.0% | 80.0% | 58.8% | 52.9% | 25.0% |
| **Actuator Drift (1.20x Friction)** | 27.3% | 55.6% | 40.0% | 47.1% | 41.2% | 31.2% |
