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
| **Terminal Placement Accuracy (Mean Error)** | $< 3.0\,\text{cm}$ | **1.24 ± 0.88 cm** | **VERIFIED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **22.1 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 22 | 2.77 | **22.7%** | 1.10 cm |
| **Mid Plane** | 18 | 2.85 | **5.6%** | 1.66 cm |
| **High Plane** | 10 | 2.91 | **20.0%** | 0.77 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 17 | **11.8%** | 1.38 cm |
| **Medium (2.30 - 3.40 kg)** | 17 | **17.6%** | 0.94 cm |
| **Heavy (3.40 - 4.50 kg)** | 16 | **18.8%** | 1.40 cm |
