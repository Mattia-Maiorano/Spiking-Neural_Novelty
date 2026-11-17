# SPWM-v0.12.4 Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/spwm_v0_12_4_train_m1/model.pt`
- **Test Episodes:** 50 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **96.0%** | **VERIFIED** |
| **Terminal Placement Accuracy (Mean Error)** | $< 3.0\,\text{cm}$ | **0.67 ± 0.46 cm** | **VERIFIED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **22.2 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 22 | 2.77 | **90.9%** | 0.82 cm |
| **Mid Plane** | 18 | 2.85 | **100.0%** | 0.65 cm |
| **High Plane** | 10 | 2.91 | **100.0%** | 0.38 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **88.2%** | 0.62 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **100.0%** | 0.74 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **100.0%** | 0.65 cm |
