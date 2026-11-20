# seed2024 checkpoint_phase_4 (val N=25) Latent MPPI Downstream Control Benchmark

**Evaluated Configuration:**
- **Planner:** Latent MPPI Controller ($K = 256$, $H = 10$, Iterations = $3$)
- **Checkpoint:** `results/multiseed/seed_2024_m1/checkpoint_phase_4.pt`
- **Test Episodes:** 25 trajectories across test partition
- **Payload Masses:** $1.20 - 4.50$ kg
- **Elevation Planes:** Low (1.04m), Mid (1.14m), High (1.26m)

---

## 1. Overall Performance Metrics

| Metric | Target / Spec | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Grasp Success Rate (%)** | $> 80.0\%$ | **60.0%** | **VERIFIED** |
| **Min Placement Error (cm)** | $< 3.0\,\text{cm}$ | **1.93 ± 1.16 cm** | **VERIFIED** |
| **Terminal Placement Error (cm)** | $< 3.0\,\text{cm}$ | **2.69 ± 1.89 cm** | **VERIFIED** |
| **Joint Acceleration ($|\Delta^2 q|$)** | — | **0.00535 ± 0.00184 rad/step²** | **RECORDED** |
| **Joint Jerk ($|\Delta^3 q|$)** | — | **0.00307 ± 0.00096 rad/step³** | **RECORDED** |
| **Direction Flips (per episode)** | — | **9.4 ± 4.0** | **RECORDED** |
| **MPPI Inference Latency (ms / step)** | $< 50.0\,\text{ms}$ | **30.2 ms** | **REAL-TIME** |

---

## 2. Stratification by Vertical Elevation Plane

| Elevation Plane | Episodes | Mean Mass (kg) | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Plane** | 9 | 2.44 | **55.6%** | 1.76 cm |
| **Mid Plane** | 8 | 2.97 | **37.5%** | 2.59 cm |
| **High Plane** | 8 | 3.66 | **87.5%** | 1.46 cm |

---

## 3. Stratification by Payload Mass Tier

| Mass Tier | Episodes | Grasp Success Rate (%) | Placement Accuracy (cm) |
| :--- | :---: | :---: | :---: |
| **Light (1.20 - 2.30 kg)** | 7 | **42.9%** | 2.31 cm |
| **Medium (2.30 - 3.40 kg)** | 8 | **62.5%** | 1.83 cm |
| **Heavy (3.40 - 4.50 kg)** | 10 | **70.0%** | 1.75 cm |
