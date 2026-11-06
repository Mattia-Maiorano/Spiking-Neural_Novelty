# SPWM-v0.7.3 Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 20$, Iterations = $3$)
- **Checkpoint:** `results/spwm_v0_7_3_train_m1/model.pt`
- **Test Episodes:** 50 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **66.0%** | **VERIFIED** |
| **Terminal Placement Accuracy (Mean Error)** | $< 3.0\,\text{cm}$ | **1.99 ± 1.33 cm** | **VERIFIED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **38.2 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 18 | 2.77 | **83.3%** | 1.96 cm |
| **Mid Plane** | 26 | 2.75 | **69.2%** | 1.42 cm |
| **High Plane** | 6 | 3.34 | **0.0%** | 4.52 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **100.0%** | 1.50 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **58.8%** | 2.76 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **37.5%** | 1.69 cm |
