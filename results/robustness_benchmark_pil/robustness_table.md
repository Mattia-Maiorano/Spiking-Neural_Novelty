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
| **Nominal Baseline** | 50 | **88.0%** | **0.88 ± 0.81 cm** | 1.60 cm | 0.31 rad/s | 26.9 ms |
| **Sensory Noise Injection (1.0%)** | 50 | **72.0%** | **1.04 ± 0.87 cm** | 1.89 cm | 0.29 rad/s | 28.7 ms |
| **Payload Extrapolation (4.80 kg)** | 50 | **78.0%** | **1.27 ± 1.12 cm** | 1.80 cm | 0.38 rad/s | 28.0 ms |
| **Actuator Drift (0.80x Friction)** | 50 | **70.0%** | **1.35 ± 1.26 cm** | 2.09 cm | 0.37 rad/s | 36.6 ms |
| **Actuator Drift (1.20x Friction)** | 50 | **76.0%** | **1.34 ± 1.23 cm** | 1.98 cm | 0.37 rad/s | 44.5 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 90.9% | 77.8% | 100.0% | 88.2% | 100.0% | 75.0% |
| **Sensory Noise Injection (1.0%)** | 68.2% | 66.7% | 90.0% | 82.4% | 88.2% | 43.8% |
| **Payload Extrapolation (4.80 kg)** | 77.3% | 77.8% | 80.0% | 82.4% | 82.4% | 68.8% |
| **Actuator Drift (0.80x Friction)** | 63.6% | 72.2% | 80.0% | 88.2% | 70.6% | 50.0% |
| **Actuator Drift (1.20x Friction)** | 72.7% | 72.2% | 90.0% | 100.0% | 76.5% | 50.0% |
