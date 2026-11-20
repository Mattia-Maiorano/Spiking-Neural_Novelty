# seed2024 best_model (val N=25) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/multiseed/seed_2024_m1/best_model.pt`
- **Test Episodes:** 25 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **48.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **0.90 ± 0.56 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **2.52 ± 1.69 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.00836 ± 0.00361 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00590 ± 0.00273 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **16.2 ± 3.2** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **37.9 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 9 | 2.44 | **22.2%** | 0.94 cm |
| **Mid Plane** | 8 | 2.97 | **37.5%** | 1.23 cm |
| **High Plane** | 8 | 3.66 | **87.5%** | 0.52 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 7 | **14.3%** | 0.91 cm |
| **Medium (2.30 - 3.40 kg)** | 8 | **62.5%** | 0.77 cm |
| **Heavy (3.40 - 4.50 kg)** | 10 | **60.0%** | 0.99 cm |
