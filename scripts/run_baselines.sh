#!/usr/bin/env bash
# ==============================================================================
# SPWM Comparative Baseline Benchmark Runner
# Execution helper for comparative non-spiking baselines:
#   1. Continuous RSSM
#   2. Deep GRU Predictor
#   3. Continuous ConvLSTM
#
# Automatically resolves active SPWM checkpoints, configs, and metrics
# via the Global Version Solver (spwm.version / spwm.contracts.version).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib

# Activate virtual environment if present
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
fi

# Detect Python interpreter
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    PYTHON_BIN="python3"
fi

# Resolve defaults dynamically from Global Version Solver
DEFAULT_CONFIG="$("${PYTHON_BIN}" -m spwm.version --format m1-config)"
DEFAULT_CHECKPOINT="$("${PYTHON_BIN}" -m spwm.version --format m1-checkpoint)"
DEFAULT_METRICS="$("${PYTHON_BIN}" -m spwm.version --format m1-metrics)"

# Defaults
EPOCHS=18
PATIENCE=6
TOTAL_TRAJECTORIES=1000
BATCH_SIZE=32
DEVICE=""
OUTPUT_DIR="results/baselines"

show_help() {
    cat << EOF
Usage: $(basename "$0") [options] [-- extra args for experiments/train_baselines.py]

Options:
  --epochs <int>              Max training epochs per baseline (default: ${EPOCHS})
  --patience <int>            Early stopping patience (default: ${PATIENCE})
  --total-trajectories <int>  Total trajectories to partition (default: ${TOTAL_TRAJECTORIES})
  --batch-size <int>          Batch size (default: ${BATCH_SIZE})
  --device <cpu|mps|cuda>     Target execution device (default: auto-detected)
  --output-dir <path>         Output results directory (default: ${OUTPUT_DIR})
  --config <path>             Path to SPWM M1 config (default: ${DEFAULT_CONFIG})
  --checkpoint <path>         Path to SPWM M1 checkpoint (default: ${DEFAULT_CHECKPOINT})
  --m1-metrics <path>         Path to SPWM M1 metrics JSON (default: ${DEFAULT_METRICS})
  -h, --help                  Show this help message and exit
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --total-trajectories)
            TOTAL_TRAJECTORIES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            DEFAULT_CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            DEFAULT_CHECKPOINT="$2"
            shift 2
            ;;
        --m1-metrics)
            DEFAULT_METRICS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

CMD=(
    "${PYTHON_BIN}" experiments/train_baselines.py
    --epochs "${EPOCHS}"
    --patience "${PATIENCE}"
    --total-trajectories "${TOTAL_TRAJECTORIES}"
    --batch-size "${BATCH_SIZE}"
    --output-dir "${OUTPUT_DIR}"
    --config "${DEFAULT_CONFIG}"
    --checkpoint "${DEFAULT_CHECKPOINT}"
    --m1-metrics "${DEFAULT_METRICS}"
)

if [[ -n "${DEVICE}" ]]; then
    CMD+=(--device "${DEVICE}")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "=============================================================================="
echo "Executing SPWM Comparative Baselines Benchmark"
echo "Active Version: $("${PYTHON_BIN}" -m spwm.version --format formatted)"
echo "Config:         ${DEFAULT_CONFIG}"
echo "Checkpoint:     ${DEFAULT_CHECKPOINT}"
echo "Output Dir:     ${OUTPUT_DIR}"
echo "=============================================================================="

exec "${CMD[@]}"

