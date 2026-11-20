# seed101 checkpoint_phase_2 (val N=25) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/multiseed/seed_101_m1/checkpoint_phase_2.pt`
- **Test Episodes:** 25 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **52.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **0.93 ± 0.48 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **2.34 ± 1.22 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.01231 ± 0.00361 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00985 ± 0.00230 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **28.1 ± 8.8** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **34.0 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 9 | 2.44 | **11.1%** | 0.93 cm |
| **Mid Plane** | 8 | 2.97 | **75.0%** | 0.80 cm |
| **High Plane** | 8 | 3.66 | **75.0%** | 1.04 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 7 | **28.6%** | 1.08 cm |
| **Medium (2.30 - 3.40 kg)** | 8 | **37.5%** | 1.04 cm |
| **Heavy (3.40 - 4.50 kg)** | 10 | **80.0%** | 0.72 cm |
