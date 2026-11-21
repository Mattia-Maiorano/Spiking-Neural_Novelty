# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test (SPWM-v0.12.3)

**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).
Evaluated on frozen production checkpoint: `results/spwm_v0_12_3_train_m1/model.pt`.

## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)

Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \in \{1, 10, 50, 100\}$:

| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 16.9% | 0.00009 | 0.00029 | 0.00235 | 0.00378 | 0.02892 | **+5.4309** | BOUNDED |
| **Threshold Shift (theta = 0.05)** | 16.2% | 0.00010 | 0.00028 | 0.00229 | 0.00375 | 0.02808 | **+5.0901** | BOUNDED |
| **Threshold Shift (theta = 0.12)** | 18.3% | 0.00010 | 0.00030 | 0.00283 | 0.00408 | 0.03212 | **+5.5207** | BOUNDED |
| **Background Activity (0.5%)** | 16.5% | 0.01789 | 0.01840 | 0.01229 | 0.00877 | 0.07508 | **+0.6329** | BOUNDED |
| **Background Activity (1.0%)** | 16.1% | 0.02859 | 0.03298 | 0.01928 | 0.01304 | 0.09369 | **+0.6013** | BOUNDED |
| **Refractory Dropout (10%)** | 17.1% | 0.00010 | 0.00029 | 0.00243 | 0.00385 | 0.02949 | **+5.3545** | BOUNDED |
| **Refractory Dropout (20%)** | 17.4% | 0.00011 | 0.00030 | 0.00249 | 0.00389 | 0.03003 | **+5.1641** | BOUNDED |

## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions

Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\text{iter}}=3$) over 50 canonical test episodes:

| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) | Ep w_t>0 (%) | Step w_t>0 (%) | Mean w_t>0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 50 | **58.0%** | **1.00 ± 0.72 cm** | 2.12 cm | 0.37 rad/s | 32.4 ms | 0.0% | 0.0% | 0.000 |
| **Sensory Noise Injection (1.0%)** | 50 | **54.0%** | **1.20 ± 0.89 cm** | 2.68 cm | 0.33 rad/s | 32.1 ms | 0.0% | 0.0% | 0.000 |
| **Payload Extrapolation (4.80 kg)** | 50 | **56.0%** | **1.76 ± 1.88 cm** | 2.46 cm | 0.56 rad/s | 32.6 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (0.80x Friction)** | 50 | **68.0%** | **1.72 ± 1.85 cm** | 2.38 cm | 0.50 rad/s | 33.1 ms | 0.0% | 0.0% | 0.000 |
| **Actuator Drift (1.20x Friction)** | 50 | **70.0%** | **1.79 ± 1.82 cm** | 2.46 cm | 0.49 rad/s | 33.7 ms | 0.0% | 0.0% | 0.000 |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 40.9% | 61.1% | 90.0% | 41.2% | 82.4% | 50.0% |
| **Sensory Noise Injection (1.0%)** | 50.0% | 33.3% | 100.0% | 52.9% | 64.7% | 43.8% |
| **Payload Extrapolation (4.80 kg)** | 63.6% | 38.9% | 70.0% | 58.8% | 70.6% | 37.5% |
| **Actuator Drift (0.80x Friction)** | 72.7% | 50.0% | 90.0% | 88.2% | 76.5% | 37.5% |
| **Actuator Drift (1.20x Friction)** | 68.2% | 66.7% | 80.0% | 82.4% | 82.4% | 43.8% |
