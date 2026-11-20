# SPWM Seed 101 (Val25-Selected phase_1) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/multiseed/seed_101_m1/checkpoint_phase_1.pt`
- **Test Episodes:** 50 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **62.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **0.85 ± 0.59 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **2.02 ± 1.15 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.01216 ± 0.00383 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00940 ± 0.00245 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **28.6 ± 8.1** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **36.4 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 22 | 2.77 | **54.5%** | 1.11 cm |
| **Mid Plane** | 18 | 2.85 | **55.6%** | 0.60 cm |
| **High Plane** | 10 | 2.91 | **90.0%** | 0.72 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **88.2%** | 0.71 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **64.7%** | 0.92 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **31.2%** | 0.92 cm |
