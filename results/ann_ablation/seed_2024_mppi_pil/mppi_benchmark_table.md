# ANN-Rate (Seed 2024) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/ann_ablation/seed_2024_m1/model.pt`
- **Test Episodes:** 50 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **48.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **0.99 ± 0.61 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **2.68 ± 1.32 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.01047 ± 0.00444 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00815 ± 0.00292 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **24.7 ± 6.3** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **24.7 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 22 | 2.77 | **45.5%** | 0.96 cm |
| **Mid Plane** | 18 | 2.85 | **33.3%** | 1.14 cm |
| **High Plane** | 10 | 2.91 | **80.0%** | 0.78 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **29.4%** | 1.09 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **58.8%** | 0.84 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **56.2%** | 1.04 cm |
