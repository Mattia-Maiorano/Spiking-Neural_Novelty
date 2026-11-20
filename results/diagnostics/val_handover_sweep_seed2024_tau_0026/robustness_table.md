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
| **Nominal Baseline** | 50 | **46.0%** | **0.83 ± 0.54 cm** | 2.31 cm | 0.39 rad/s | 37.0 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **36.0%** | **0.69 ± 0.46 cm** | 2.56 cm | 0.33 rad/s | 37.1 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **42.0%** | **1.03 ± 0.60 cm** | 2.27 cm | 0.44 rad/s | 42.4 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (0.80x Friction)** | 50 | **38.0%** | **1.03 ± 0.63 cm** | 2.51 cm | 0.36 rad/s | 30.5 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (1.20x Friction)** | 50 | **38.0%** | **1.12 ± 0.65 cm** | 2.26 cm | 0.37 rad/s | 29.7 ms | 0.0% | 0.0% | 0.000 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 25.0% | 20.0% | 84.2% | 18.8% | 62.5% | 55.6% |
| **Sensory Noise Injection (1.0%)** | 31.2% | 20.0% | 52.6% | 12.5% | 68.8% | 27.8% |
| **Payload Extrapolation (4.80 kg)** | 37.5% | 0.0% | 78.9% | 37.5% | 37.5% | 50.0% |
| **Actuator Drift (0.80x Friction)** | 18.8% | 6.7% | 78.9% | 37.5% | 37.5% | 38.9% |
| **Actuator Drift (1.20x Friction)** | 25.0% | 0.0% | 78.9% | 12.5% | 50.0% | 50.0% |
