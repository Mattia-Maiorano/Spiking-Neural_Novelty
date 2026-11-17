# Zero-Shot Sim-to-Real Domain Gap & Robustness Stress Test (SPWM-v0.12.4)

**Mandatory Constraints:** Strictly zero retraining / zero fine-tuning (`torch.no_grad()`, `model.eval()`).
Evaluated on frozen production checkpoint: `results/spwm_v0_12_4_train_m1/model.pt`.

## 1. Neuromorphic Visual Stream Perturbations (Sensor Domain Gap)

Evaluated over the held-out test split (400 trajectories) across autonomous horizons $H \in \{1, 10, 50, 100\}$:

| Perturbation Condition | Spike Rate | Pos MSE (H=1) | Pos MSE (H=10) | Pos MSE (H=50) | Pos MSE (H=100) | Vel MSE (H=100) | Drift Ratio (H=100) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 16.8% | 0.00008 | 0.00032 | 0.00270 | 0.00320 | 0.02516 | **+4.9162** | BOUNDED |
| **Threshold Shift (theta = 0.05)** | 16.2% | 0.00011 | 0.00034 | 0.00254 | 0.00311 | 0.02494 | **+4.2758** | BOUNDED |
| **Threshold Shift (theta = 0.12)** | 18.2% | 0.00010 | 0.00033 | 0.00340 | 0.00366 | 0.02704 | **+4.9029** | BOUNDED |
| **Background Activity (0.5%)** | 16.5% | 0.02028 | 0.01753 | 0.01333 | 0.00893 | 0.07292 | **+0.5689** | BOUNDED |
| **Background Activity (1.0%)** | 16.1% | 0.03009 | 0.02921 | 0.01822 | 0.01188 | 0.08925 | **+0.5365** | BOUNDED |
| **Refractory Dropout (10%)** | 17.1% | 0.00009 | 0.00033 | 0.00286 | 0.00331 | 0.02540 | **+4.7070** | BOUNDED |
| **Refractory Dropout (20%)** | 17.3% | 0.00010 | 0.00033 | 0.00303 | 0.00341 | 0.02571 | **+4.5731** | BOUNDED |

## 2. Downstream Latent MPPI Robustness Under Perturbed Execution Conditions

Batched closed-loop Latent MPPI controller ($K=256, H=20, N_{\text{iter}}=3$) over 50 canonical test episodes:

| Perturbation Condition | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) | Terminal Error (cm) | Terminal Vel (rad/s) | MPPI Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 20 | **100.0%** | **1.01 ± 0.51 cm** | 1.81 cm | 0.62 rad/s | 40.6 ms |
| **Sensory Noise Injection (1.0%)** | 20 | **95.0%** | **1.19 ± 1.07 cm** | 2.31 cm | 0.75 rad/s | 40.7 ms |
| **Payload Extrapolation (4.80 kg)** | 20 | **70.0%** | **2.84 ± 2.56 cm** | 4.37 cm | 0.65 rad/s | 40.9 ms |
| **Actuator Drift (0.80x Friction)** | 20 | **70.0%** | **2.69 ± 2.47 cm** | 3.70 cm | 0.58 rad/s | 40.8 ms |
| **Actuator Drift (1.20x Friction)** | 20 | **60.0%** | **2.65 ± 2.26 cm** | 4.56 cm | 0.65 rad/s | 40.8 ms |

### Stratification of Downstream Performance Across Shifted Conditions

| Perturbation Condition | Low Plane Succ (%) | Mid Plane Succ (%) | High Plane Succ (%) | Light Mass Succ (%) | Med Mass Succ (%) | Heavy Mass Succ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Nominal Baseline** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Sensory Noise Injection (1.0%)** | 100.0% | 87.5% | 100.0% | 100.0% | 100.0% | 83.3% |
| **Payload Extrapolation (4.80 kg)** | 100.0% | 62.5% | 25.0% | 100.0% | 71.4% | 33.3% |
| **Actuator Drift (0.80x Friction)** | 87.5% | 62.5% | 50.0% | 100.0% | 57.1% | 50.0% |
| **Actuator Drift (1.20x Friction)** | 87.5% | 62.5% | 0.0% | 100.0% | 42.9% | 33.3% |
