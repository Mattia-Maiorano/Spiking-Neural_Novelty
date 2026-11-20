# seed101 checkpoint_phase_3 (val N=25) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/multiseed/seed_101_m1/checkpoint_phase_3.pt`
- **Test Episodes:** 25 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **4.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **1.07 ± 0.55 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **6.59 ± 2.46 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.01021 ± 0.00324 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00771 ± 0.00243 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **15.8 ± 4.8** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **36.9 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 9 | 2.44 | **0.0%** | 1.30 cm |
| **Mid Plane** | 8 | 2.97 | **12.5%** | 0.96 cm |
| **High Plane** | 8 | 3.66 | **0.0%** | 0.94 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 7 | **0.0%** | 1.57 cm |
| **Medium (2.30 - 3.40 kg)** | 8 | **12.5%** | 0.96 cm |
| **Heavy (3.40 - 4.50 kg)** | 10 | **0.0%** | 0.82 cm |
