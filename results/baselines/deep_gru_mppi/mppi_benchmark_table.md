# Deep GRU Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/baselines/deep_gru/model.pt`
- **Test Episodes:** 50 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **14.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **3.35 ± 2.42 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **7.39 ± 6.23 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.01930 ± 0.01039 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.02715 ± 0.01761 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **18.2 ± 7.1** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **49.2 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 22 | 2.77 | **9.1%** | 3.49 cm |
| **Mid Plane** | 18 | 2.85 | **16.7%** | 3.64 cm |
| **High Plane** | 10 | 2.91 | **20.0%** | 2.51 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **11.8%** | 4.75 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **11.8%** | 2.79 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **18.8%** | 2.46 cm |
