#!/usr/bin/env bash
# ==============================================================================
# SPWM Pipeline Runner
# Execution helper for:
#   1. Training: Representation (Phase 0) and M1 (Phase 1)
#   2. Unity Demo Export: Generate ghost trajectory JSON for Unity
#   3. Simulation Video: Render standalone 6-DOF simulation MP4
#   4. Simulation Data Export: Headless batch generation of trajectories via Unity
# ==============================================================================

set -euo pipefail

# Project root directory (assumed to be executed from root or scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

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

# Detect Unity binary (can be overridden via UNITY_BIN env var)
UNITY_BIN="${UNITY_BIN:-/Applications/Unity/Hub/Editor/6000.5.7f1/Unity.app/Contents/MacOS/Unity}"
if [[ ! -x "${UNITY_BIN}" ]]; then
    # Try finding any installed 6000.* Unity editor on macOS
    DETECTED_UNITY="$(find /Applications/Unity/Hub/Editor -name "Unity" -type f -perm +111 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${DETECTED_UNITY}" && -x "${DETECTED_UNITY}" ]]; then
        UNITY_BIN="${DETECTED_UNITY}"
    fi
fi

# Defaults
DEFAULT_MODEL_VERSION="spwm_v0_6_0"
DEFAULT_EPOCHS=200
DEFAULT_HORIZON=100
DEFAULT_TOTAL_TRAJECTORIES=4000
DEFAULT_EXPORT_DIR="ExportedTrajectories"

# Helper to automatically resolve the latest/best checkpoint directory:
# strictly selects canonical directories ending in '_train_m1' (no trailing words/suffixes)
# and sorts by semver numbers (major, minor, patch) descending.
find_latest_m1_dir() {
    local target_ver="${1:-}"

    "${PYTHON_BIN}" -c "
import os, re, sys

target_ver = sys.argv[1].strip() if len(sys.argv) > 1 else ''
results_dir = 'results'
if not os.path.isdir(results_dir):
    sys.exit(1)

# Canonical pattern: strictly ends with _train_m1 (no extra suffix like _NO_PEEK or _BASIC_RNDM)
pattern = re.compile(r'^spwm_v(\d+)_(\d+)_(\d+)_train_m1$')

candidates = []
for entry in os.listdir(results_dir):
    full_path = os.path.join(results_dir, entry)
    if not os.path.isdir(full_path):
        continue
    m = pattern.match(entry)
    if not m:
        continue
    if not (os.path.isfile(os.path.join(full_path, 'model.pt')) and os.path.isfile(os.path.join(full_path, 'config.yaml'))):
        continue

    version_tuple = tuple(int(x) for x in m.groups())
    version_str = f'spwm_v{m.group(1)}_{m.group(2)}_{m.group(3)}'

    if target_ver and target_ver != version_str:
        continue

    candidates.append((version_tuple, full_path, version_str))

if not candidates and target_ver:
    # If a specific version was requested but no canonical one found, re-search without target_ver filter
    for entry in os.listdir(results_dir):
        full_path = os.path.join(results_dir, entry)
        if not os.path.isdir(full_path):
            continue
        m = pattern.match(entry)
        if not m:
            continue
        if os.path.isfile(os.path.join(full_path, 'model.pt')) and os.path.isfile(os.path.join(full_path, 'config.yaml')):
            candidates.append((tuple(int(x) for x in m.groups()), full_path, f'spwm_v{m.group(1)}_{m.group(2)}_{m.group(3)}'))

if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_dir = candidates[0][1]
    best_ver = candidates[0][2]
    print(f'{best_dir}\t{best_ver}')
    sys.exit(0)

sys.exit(1)
" "${target_ver}"
}

# ------------------------------------------------------------------------------
# Usage & Help
# ------------------------------------------------------------------------------
show_help() {
    cat <<EOF
==============================================================================
SPWM Pipeline Runner
==============================================================================

Usage: $(basename "$0") <command> [options]

Commands:
  sim-data           Run Unity headless to generate dataset trajectories (batchmode, no -nographics).
  train              Train representation (Phase 0) followed by M1 (Phase 1).
  export-unity       Export ghost trajectory JSON for Unity demo playback.
  video              Render simulation video (.mp4) directly from Unity engine output.
  all                Run complete pipeline: train -> export-unity -> video.

Basic Options:
  --trajectories <N>     Number of trajectories to export in sim-data (default: ${DEFAULT_TOTAL_TRAJECTORIES}).
  --export-dir <path>    Output directory for exported dataset (default: ${DEFAULT_EXPORT_DIR}).
  --epochs <N>           Epochs for training (default: ${DEFAULT_EPOCHS}).
  --config <path>        Path to experiment config.yaml (optional for train).
  --checkpoint <path>    Path to model checkpoint (.pt) for export.
  --model-version <ver>  Model version tag (default: ${DEFAULT_MODEL_VERSION}).
  --horizon <H>          Rollout / render horizon steps (default: ${DEFAULT_HORIZON}).
  --json-file <path>     JSON path for video rendering (default: sim/Assets/StreamingAssets/trajectory_ghost.json).
  --output-video <path>  Output MP4 file path (default: results/videos/robot_arm_simulation.mp4).
  --logfile <path>       Log file for Unity headless batch export (default: export_sim.log).
  -h, --help             Show this help message.

Quick Start Examples:
  # 1. Generate 4,000 simulation trajectories using Unity (headless batchmode):
  ./scripts/run_pipeline.sh sim-data
  ./scripts/run_pipeline.sh sim-data --trajectories 4000 --export-dir ExportedTrajectories

  # 2. Train representation (Phase 0) and latent predictor (Phase 1):
  ./scripts/run_pipeline.sh train --epochs 200

  # 3. Export ghost trajectory JSON for Unity demo playback:
  ./scripts/run_pipeline.sh export-unity

  # 4. Render simulation video directly from Unity scene capture:
  ./scripts/run_pipeline.sh video
  ./scripts/run_pipeline.sh video --horizon 150

  # 5. Run train -> export-unity -> video end-to-end:
  ./scripts/run_pipeline.sh all
EOF
}

# ------------------------------------------------------------------------------
# Action 1: Headless Simulation Dataset Generation (Unity Batchmode)
# NOTE: We intentionally DO NOT pass -nographics because Camera.Render()
# in NeuromorphicEventBridge requires a GPU/graphics context to capture frames.
# ------------------------------------------------------------------------------
run_sim_data() {
    local trajectories="${TOTAL_TRAJECTORIES:-$DEFAULT_TOTAL_TRAJECTORIES}"
    local export_dir="${EXPORT_DIR:-$DEFAULT_EXPORT_DIR}"
    local logfile="${LOGFILE:-export_sim.log}"
    local sim_path="${PROJECT_ROOT}/sim"

    if [[ ! -x "${UNITY_BIN}" ]]; then
        echo "Error: Unity binary not found or not executable at '${UNITY_BIN}'." >&2
        echo "Please set UNITY_BIN env variable to your Unity executable path." >&2
        exit 1
    fi

    echo "======================================================================"
    echo " Exporting Synthetic Trajectory Dataset via Unity Headless Batchmode..."
    echo " Unity:         ${UNITY_BIN}"
    echo " Project Path:  ${sim_path}"
    echo " Trajectories:  ${trajectories}"
    echo " Export Dir:    ${export_dir}"
    echo " Log File:      ${logfile}"
    echo " Note:          Running in -batchmode (without -nographics) to allow camera frame rendering."
    echo "======================================================================"

    "${UNITY_BIN}" \
        -batchmode \
        -projectPath "${sim_path}" \
        -executeMethod SPWM.Editor.GhostVisualizerSceneBuilder.HeadlessBatchExport \
        -totalTrajectories "${trajectories}" \
        -exportDir "${export_dir}" \
        -logfile "${logfile}" \
        -quit

    echo ""
    echo ">>> Unity simulation export completed successfully into '${export_dir}'."
}

# ------------------------------------------------------------------------------
# Action 2: Training (Representation + M1)
# ------------------------------------------------------------------------------
run_train() {
    local epochs="${EPOCHS:-$DEFAULT_EPOCHS}"
    local cfg="${CONFIG_PATH:-}"

    echo "======================================================================"
    echo " [1/2] Training Representation Phase (Phase 0)..."
    echo "======================================================================"
    local rep_cmd=("${PYTHON_BIN}" "experiments/train.py" "--mode" "train_representation" "--epochs" "${epochs}")
    if [[ -n "${cfg}" ]]; then
        rep_cmd+=("--config" "${cfg}")
    fi
    "${rep_cmd[@]}"

    echo ""
    echo "======================================================================"
    echo " [2/2] Training M1 Latent Predictor Phase (Phase 1)..."
    echo "======================================================================"
    local m1_cmd=("${PYTHON_BIN}" "experiments/train.py" "--mode" "train_m1" "--epochs" "${epochs}")
    if [[ -n "${cfg}" ]]; then
        m1_cmd+=("--config" "${cfg}")
    fi
    "${m1_cmd[@]}"

    echo ""
    echo ">>> Training phases completed successfully."
}

# ------------------------------------------------------------------------------
# Action 3: Export Ghost Trajectory for Unity
# ------------------------------------------------------------------------------
run_export_unity() {
    local model_ver="${MODEL_VERSION:-$DEFAULT_MODEL_VERSION}"
    local ckpt="${CHECKPOINT_PATH:-}"
    local cfg="${CONFIG_PATH:-}"
    local horizon="${HORIZON:-$DEFAULT_HORIZON}"

    # Auto-resolve latest checkpoint and config if not explicitly provided
    if [[ -z "${ckpt}" || -z "${cfg}" ]]; then
        local detected_output
        if detected_output="$(find_latest_m1_dir "${MODEL_VERSION:-}")"; then
            local detected_dir
            local detected_ver
            detected_dir="$(echo "${detected_output}" | cut -f1)"
            detected_ver="$(echo "${detected_output}" | cut -f2)"
            echo "[export-unity] Detected latest canonical M1 run directory: ${detected_dir}"
            [[ -z "${ckpt}" ]] && ckpt="${detected_dir}/model.pt"
            [[ -z "${cfg}" ]] && cfg="${detected_dir}/config.yaml"
            [[ -z "${MODEL_VERSION:-}" ]] && model_ver="${detected_ver}"
        else
            echo "Error: Could not automatically detect a valid M1 checkpoint directory in results/." >&2
            echo "Please provide --checkpoint <path/to/model.pt> and --config <path/to/config.yaml>." >&2
            exit 1
        fi
    fi

    echo "======================================================================"
    echo " Generating Ghost Trajectory JSON for Unity Demo..."
    echo " Checkpoint: ${ckpt}"
    echo " Config:     ${cfg}"
    echo " Version:    ${model_ver}"
    echo " Horizon:    ${horizon}"
    echo "======================================================================"

    "${PYTHON_BIN}" experiments/export_ghost_trajectory.py \
        --checkpoint "${ckpt}" \
        --config "${cfg}" \
        --model-version "${model_ver}" \
        --horizon "${horizon}"

    echo ""
    echo ">>> Ghost trajectory exported to sim/Assets/StreamingAssets/."
}

# ------------------------------------------------------------------------------
# Action 4: Render Simulation Video (.mp4)
# ------------------------------------------------------------------------------
run_video() {
    local json_file="${JSON_FILE:-sim/Assets/StreamingAssets/trajectory_ghost.json}"
    local output_video="${OUTPUT_VIDEO:-results/videos/robot_arm_simulation.mp4}"
    local horizon="${HORIZON:-$DEFAULT_HORIZON}"

    echo "======================================================================"
    echo " Rendering Unity Simulation Video (.mp4)..."
    echo " Trajectory: ${json_file}"
    if [[ ! -f "${json_file}" ]]; then
        local alt_json="sim/Assets/StreamingAssets/trajectory_ghost_v0_4_0.json"
        if [[ -f "${alt_json}" ]]; then
            echo " Note: '${json_file}' not found, using '${alt_json}'."
            json_file="${alt_json}"
        else
            echo "Error: '${json_file}' not found. Run 'export-unity' first to generate trajectory." >&2
            exit 1
        fi
    fi
    echo " Output:     ${output_video}"
    echo " Horizon:    ${horizon}"
    echo "======================================================================"

    mkdir -p "$(dirname "${output_video}")"

    "${PYTHON_BIN}" "scripts/render_simulation_video.py" \
        "--json-file" "${json_file}" \
        "--output" "${output_video}" \
        "--horizon" "${horizon}"

    echo ""
    echo ">>> Video rendered successfully at: ${output_video}"
}

# ------------------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------------------
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

TOTAL_TRAJECTORIES=""
EXPORT_DIR=""
LOGFILE=""
EPOCHS=""
CONFIG_PATH=""
CHECKPOINT_PATH=""
MODEL_VERSION=""
HORIZON=""
VIDEO_SOURCE=""
JSON_FILE=""
OUTPUT_VIDEO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trajectories|-t)
            TOTAL_TRAJECTORIES="$2"
            shift 2
            ;;
        --export-dir)
            EXPORT_DIR="$2"
            shift 2
            ;;
        --logfile)
            LOGFILE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --model-version)
            MODEL_VERSION="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --source)
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        --json-file)
            JSON_FILE="$2"
            shift 2
            ;;
        --output-video)
            OUTPUT_VIDEO="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
    esac
done

case "${COMMAND}" in
    sim-data|sim_data|export-data|export_data|data)
        run_sim_data
        ;;
    1|train)
        run_train
        ;;
    2|video)
        run_video
        ;;
    3|export-unity|export_unity|unity)
        run_export_unity
        ;;
    all)
        run_train
        run_export_unity
        run_video
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "Error: Unknown command '${COMMAND}'" >&2
        show_help
        exit 1
        ;;
esac

