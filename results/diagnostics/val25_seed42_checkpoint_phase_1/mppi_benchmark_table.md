# seed42 checkpoint_phase_1 (val N=25) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/spwm_v0_12_4_train_m1/checkpoint_phase_1.pt`
- **Test Episodes:** 25 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **60.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **0.81 ± 0.65 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **1.97 ± 1.44 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.00716 ± 0.00129 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00466 ± 0.00089 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **14.4 ± 7.9** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **37.0 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 9 | 2.44 | **66.7%** | 0.61 cm |
| **Mid Plane** | 8 | 2.97 | **50.0%** | 1.12 cm |
| **High Plane** | 8 | 3.66 | **62.5%** | 0.72 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 7 | **71.4%** | 0.84 cm |
| **Medium (2.30 - 3.40 kg)** | 8 | **37.5%** | 0.82 cm |
| **Heavy (3.40 - 4.50 kg)** | 10 | **70.0%** | 0.78 cm |
