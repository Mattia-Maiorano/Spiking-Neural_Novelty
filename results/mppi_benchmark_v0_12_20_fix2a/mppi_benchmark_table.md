# SPWM-v0.12.20 Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/spwm_v0_12_20_train_m1/model.pt`
- **Test Episodes:** 50 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **16.0%** | **VERIFIED** |
| **Terminal Placement Accuracy (Mean Error)** | $< 3.0\,\text{cm}$ | **1.42 ± 0.90 cm** | **VERIFIED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **22.4 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 22 | 2.77 | **13.6%** | 1.28 cm |
| **Mid Plane** | 18 | 2.85 | **16.7%** | 1.90 cm |
| **High Plane** | 10 | 2.91 | **20.0%** | 0.86 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **17.6%** | 1.40 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **23.5%** | 1.07 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **6.2%** | 1.80 cm |
