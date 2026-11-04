# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test (SPWM-v0.7.3)

**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).
Evaluated on frozen production checkpoint: `results/spwm_v0_7_3_train_m1/model.pt`.

## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)

Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \in \{1, 10, 50, 100\}$:

| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 14.6% | 0.10635 | 0.02117 | 0.01425 | 0.01728 | 0.01153 | **+0.3432** | BOUNDED |
| **Threshold Shift (theta = 0.05)** | 14.3% | 0.08997 | 0.01732 | 0.01364 | 0.01694 | 0.01193 | **+0.3707** | BOUNDED |
| **Threshold Shift (theta = 0.12)** | 15.3% | 0.12759 | 0.02620 | 0.01632 | 0.01983 | 0.01190 | **+0.3355** | BOUNDED |
| **Background Activity (0.5%)** | 14.3% | 0.50225 | 0.07066 | 0.07673 | 0.04694 | 0.02880 | **+0.2436** | BOUNDED |
| **Background Activity (1.0%)** | 14.2% | 0.52888 | 0.07616 | 0.09218 | 0.05485 | 0.03542 | **+0.2564** | BOUNDED |
| **Refractory Dropout (10%)** | 14.7% | 0.11683 | 0.02353 | 0.01458 | 0.01734 | 0.01151 | **+0.3280** | BOUNDED |
| **Refractory Dropout (20%)** | 14.9% | 0.12782 | 0.02608 | 0.01509 | 0.01753 | 0.01157 | **+0.3155** | BOUNDED |

## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions

Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\text{iter}}=3$) over 50 canonical test episodes:

| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 50 | **74.0%** | **2.21 ± 2.10 cm** | 6.66 cm | 0.41 rad/s | 37.2 ms |
| **Sensory Noise Injection (1.0%)** | 50 | **78.0%** | **2.16 ± 1.74 cm** | 6.82 cm | 0.36 rad/s | 37.3 ms |
| **Payload Extrapolation (4.80 kg)** | 50 | **18.0%** | **4.91 ± 2.16 cm** | 6.20 cm | 0.55 rad/s | 37.3 ms |
| **Actuator Drift (0.80x Friction)** | 50 | **18.0%** | **5.09 ± 2.07 cm** | 5.89 cm | 0.46 rad/s | 37.2 ms |
| **Actuator Drift (1.20x Friction)** | 50 | **14.0%** | **4.98 ± 2.01 cm** | 5.78 cm | 0.46 rad/s | 37.3 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 81.8% | 55.6% | 90.0% | 82.4% | 82.4% | 56.2% |
| **Sensory Noise Injection (1.0%)** | 86.4% | 55.6% | 100.0% | 94.1% | 82.4% | 56.2% |
| **Payload Extrapolation (4.80 kg)** | 4.5% | 27.8% | 30.0% | 23.5% | 23.5% | 6.2% |
| **Actuator Drift (0.80x Friction)** | 0.0% | 33.3% | 30.0% | 23.5% | 23.5% | 6.2% |
| **Actuator Drift (1.20x Friction)** | 4.5% | 22.2% | 20.0% | 11.8% | 17.6% | 12.5% |
