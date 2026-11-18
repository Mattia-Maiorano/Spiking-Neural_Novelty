# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test (SPWM-v0.12.3)

**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).
Evaluated on frozen production checkpoint: `results/spwm_v0_12_3_train_m1/model.pt`.

## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)

Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \in \{1, 10, 50, 100\}$:

| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 16.9% | 0.00008 | 0.00025 | 0.00184 | 0.00253 | 0.02197 | **+4.6278** | BOUNDED |
| **Threshold Shift (theta = 0.05)** | 16.2% | 0.00010 | 0.00027 | 0.00173 | 0.00245 | 0.02187 | **+4.1182** | BOUNDED |
| **Threshold Shift (theta = 0.12)** | 18.3% | 0.00010 | 0.00027 | 0.00234 | 0.00295 | 0.02440 | **+4.5747** | BOUNDED |
| **Background Activity (0.5%)** | 16.5% | 0.01740 | 0.01506 | 0.01007 | 0.00703 | 0.06218 | **+0.5619** | BOUNDED |
| **Background Activity (1.0%)** | 16.1% | 0.02794 | 0.02713 | 0.01592 | 0.01055 | 0.08352 | **+0.5351** | BOUNDED |
| **Refractory Dropout (10%)** | 17.1% | 0.00009 | 0.00026 | 0.00192 | 0.00259 | 0.02227 | **+4.4305** | BOUNDED |
| **Refractory Dropout (20%)** | 17.4% | 0.00011 | 0.00027 | 0.00203 | 0.00268 | 0.02267 | **+4.2126** | BOUNDED |

## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions

Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\text{iter}}=3$) over 50 canonical test episodes:

| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 50 | **96.0%** | **0.67 ± 0.46 cm** | 1.21 cm | 0.70 rad/s | 32.9 ms |
| **Sensory Noise Injection (1.0%)** | 50 | **82.0%** | **1.06 ± 1.23 cm** | 1.78 cm | 0.77 rad/s | 33.9 ms |
| **Payload Extrapolation (4.80 kg)** | 50 | **74.0%** | **1.57 ± 1.91 cm** | 2.25 cm | 0.78 rad/s | 32.4 ms |
| **Actuator Drift (0.80x Friction)** | 50 | **70.0%** | **1.61 ± 2.07 cm** | 2.36 cm | 0.77 rad/s | 31.2 ms |
| **Actuator Drift (1.20x Friction)** | 50 | **80.0%** | **1.53 ± 1.91 cm** | 2.21 cm | 0.78 rad/s | 31.8 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 90.9% | 100.0% | 100.0% | 88.2% | 100.0% | 100.0% |
| **Sensory Noise Injection (1.0%)** | 77.3% | 77.8% | 100.0% | 94.1% | 88.2% | 62.5% |
| **Payload Extrapolation (4.80 kg)** | 90.9% | 72.2% | 40.0% | 82.4% | 76.5% | 62.5% |
| **Actuator Drift (0.80x Friction)** | 68.2% | 66.7% | 80.0% | 94.1% | 76.5% | 37.5% |
| **Actuator Drift (1.20x Friction)** | 81.8% | 77.8% | 80.0% | 100.0% | 76.5% | 62.5% |
