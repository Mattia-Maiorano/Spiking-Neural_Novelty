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
| **Nominal Baseline** | 50 | **30.0%** | **1.38 ± 1.01 cm** | 3.52 cm | 0.53 rad/s | 26.5 ms |
| **Sensory Noise Injection (1.0%)** | 50 | **26.0%** | **1.11 ± 0.69 cm** | 3.97 cm | 0.55 rad/s | 26.0 ms |
| **Payload Extrapolation (4.80 kg)** | 50 | **12.0%** | **1.42 ± 1.02 cm** | 4.47 cm | 0.70 rad/s | 25.1 ms |
| **Actuator Drift (0.80x Friction)** | 50 | **16.0%** | **1.55 ± 1.00 cm** | 4.66 cm | 0.66 rad/s | 25.1 ms |
| **Actuator Drift (1.20x Friction)** | 50 | **20.0%** | **1.40 ± 0.79 cm** | 4.91 cm | 0.56 rad/s | 25.1 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 18.2% | 38.9% | 40.0% | 58.8% | 23.5% | 6.2% |
| **Sensory Noise Injection (1.0%)** | 27.3% | 27.8% | 20.0% | 52.9% | 23.5% | 0.0% |
| **Payload Extrapolation (4.80 kg)** | 13.6% | 5.6% | 20.0% | 0.0% | 35.3% | 0.0% |
| **Actuator Drift (0.80x Friction)** | 22.7% | 16.7% | 0.0% | 17.6% | 29.4% | 0.0% |
| **Actuator Drift (1.20x Friction)** | 22.7% | 22.2% | 10.0% | 11.8% | 47.1% | 0.0% |
