#!/usr/bin/env bash
# Runner for CPU-only PIL MPPI benchmarks, ablations, energy profiling, figures, and verification.
# Suitable for running on an Antigravity Remote Agent or locally without GPU.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

echo "Using Python: $PYTHON_BIN"

# Ensure Seed 42 ann_rate_m1 copy exists if seed_42_m1 is present
if [[ -f "results/ann_ablation/seed_42_m1/model.pt" && ! -f "results/ann_ablation/ann_rate_m1/model.pt" ]]; then
  mkdir -p results/ann_ablation/ann_rate_m1
  cp -r results/ann_ablation/seed_42_m1/* results/ann_ablation/ann_rate_m1/
fi

run_mppi_pil() {
  local model_type="$1"
  local ckpt="$2"
  local out_dir="$3"
  local label="$4"
  shift 4

  if [[ ! -f "$ckpt" ]]; then
    echo "[SKIP] Missing checkpoint: $ckpt"
    return 0
  fi
  if [[ -f "${out_dir}/mppi_benchmark_metrics.json" ]]; then
    echo "[DONE] Already exists: ${out_dir}/mppi_benchmark_metrics.json"
    return 0
  fi

  echo ">>> Running PIL MPPI: $label -> $out_dir"
  "$PYTHON_BIN" experiments/benchmark_mppi_control.py \
    --config results/spwm_v0_12_4_train_m1/config.yaml \
    --model-type "$model_type" \
    --model-label "$label" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --num-episodes 50 \
    --physics-in-loop \
    --device cpu \
    "$@"
}

# --- Part 3.2: 50-Episode Closed-Loop PIL MPPI ---
# SPWM (42, 101, 2024)
run_mppi_pil spwm results/spwm_v0_12_4_train_m1/model.pt results/mppi_benchmark "SPWM (seed 42)"
run_mppi_pil spwm results/multiseed/seed_101_m1/model.pt results/multiseed/seed_101_mppi_pil "SPWM (seed 101)"
run_mppi_pil spwm results/multiseed/seed_2024_m1/model.pt results/multiseed/seed_2024_mppi_pil "SPWM (seed 2024)"

# Continuous RSSM (42, 101, 2024)
run_mppi_pil rssm results/baselines/continuous_rssm/model.pt results/mppi_benchmark/physics_in_loop/rssm "Continuous RSSM (seed 42)"
run_mppi_pil rssm results/multiseed_rssm/seed_101/continuous_rssm/model.pt results/multiseed_rssm/seed_101/mppi_pil "Continuous RSSM (seed 101)"
run_mppi_pil rssm results/multiseed_rssm/seed_2024/continuous_rssm/model.pt results/multiseed_rssm/seed_2024/mppi_pil "Continuous RSSM (seed 2024)"

# ANN-Rate (42, 101, 2024)
run_mppi_pil ann_rate results/ann_ablation/seed_42_m1/model.pt results/ann_ablation/seed_42_mppi_pil "ANN-Rate (seed 42)"
if [[ -f "results/ann_ablation/seed_42_mppi_pil/mppi_benchmark_metrics.json" ]]; then
  mkdir -p results/mppi_benchmark/physics_in_loop/ann_rate
  cp -r results/ann_ablation/seed_42_mppi_pil/* results/mppi_benchmark/physics_in_loop/ann_rate/
fi
run_mppi_pil ann_rate results/ann_ablation/seed_101_m1/model.pt results/ann_ablation/seed_101_mppi_pil "ANN-Rate (seed 101)"
run_mppi_pil ann_rate results/ann_ablation/seed_2024_m1/model.pt results/ann_ablation/seed_2024_mppi_pil "ANN-Rate (seed 2024)"

# Deep GRU (42)
run_mppi_pil deep_gru results/baselines/deep_gru/model.pt results/mppi_benchmark/physics_in_loop/deep_gru "Deep GRU (seed 42)"

# --- Part 3.3: 4-Stage SPWM PIL Ablation (50 Episodes) ---
run_mppi_pil spwm results/spwm_v0_12_4_train_m1/checkpoint_phase_1.pt results/ablations/stage1_base_pil "Stage 1 Base" \
  --disable-smooth-mppi --disable-handover --disable-lookahead
run_mppi_pil spwm results/spwm_v0_12_4_train_m1/checkpoint_phase_1.pt results/ablations/stage2_smooth_mppi_pil "Stage 2 Smooth MPPI" \
  --disable-handover --disable-lookahead
run_mppi_pil spwm results/spwm_v0_12_4_train_m1/model.pt results/ablations/stage3_se3_homeostasis_pil "Stage 3 SE(3) Homeostasis" \
  --disable-handover --disable-lookahead

# --- Part 3.4 & 3.5: Energy Audit, SSOT Robustness & Publication Figures ---
for s_pair in \
  "results/spwm_v0_12_4_train_m1/model.pt:results/energy_profiling" \
  "results/multiseed/seed_101_m1/model.pt:results/multiseed/seed_101_energy" \
  "results/multiseed/seed_2024_m1/model.pt:results/multiseed/seed_2024_energy"; do
  ckpt="${s_pair%%:*}"
  out_dir="${s_pair##*:}"
  if [[ -f "$ckpt" ]]; then
    echo ">>> Energy/SOPs audit: $ckpt -> $out_dir"
    "$PYTHON_BIN" experiments/profile_energy_sops.py \
      --config results/spwm_v0_12_4_train_m1/config.yaml \
      --checkpoint "$ckpt" \
      --output-dir "$out_dir" \
      --device cpu \
      --total-trajectories 4000
  fi
done

echo ">>> SSOT Robustness + Figures 1-4"
"$PYTHON_BIN" scripts/run_ssot_benchmark_and_plot.py \
  --physics-in-loop \
  --mppi-episodes 50 \
  --robustness-episodes 50 \
  --sensor-trajectories 400

echo ">>> Heft Shock Timeseries"
"$PYTHON_BIN" scripts/plot_heft_shock_timeseries.py --traj-id 3622

# --- Part 6: Verification ---
echo ">>> AST Fallback Scan"
"$PYTHON_BIN" scripts/scan_fallbacks.py spwm/ experiments/ scripts/

echo ">>> Running pytest"
"$PYTHON_BIN" -m pytest tests/test_anti_fallback.py tests/test_architecture_contracts.py
