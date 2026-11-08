#!/usr/bin/env bash
# ==============================================================================
# SPWM Pipeline Runner
# Execution helper for:
#   1. Training: Representation (Phase 0) and M1 (Phase 1)
#   2. Unity Demo Export: Generate ghost trajectory JSON for Unity
#   3. Simulation Video: Render standalone 6-DOF simulation MP4
#   4. Simulation Data Export: Headless batch generation of trajectories via Unity
#   5. Dual-Model Video Comparison: Overlapping dual-arm rendering
# ==============================================================================

set -euo pipefail

# Project root directory (assumed to be executed from root or scripts/)
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
DEFAULT_MODEL_VERSION="$("${PYTHON_BIN}" -m spwm.version --format snake)"
DEFAULT_MODEL_VERSION_B="legacy_nov03"
DEFAULT_EPOCHS=200
DEFAULT_HORIZON=100
DEFAULT_TOTAL_TRAJECTORIES=4000
DEFAULT_EXPORT_DIR="ExportedTrajectories"
DEFAULT_EPISODES=1
DEFAULT_OUTPUT_DIR="results/videos"

# Helper to automatically resolve the latest/best checkpoint directory
find_latest_m1_dir() {
    local target_ver="${1:-}"

    "${PYTHON_BIN}" -c "
import os, re, sys

target_ver = sys.argv[1].strip() if len(sys.argv) > 1 else ''
results_dir = 'results'
if not os.path.isdir(results_dir):
    sys.exit(1)

# Check direct directory match if target_ver provided
if target_ver:
    for candidate_name in [target_ver, f'{target_ver}_train_m1', f'{target_ver}_m1']:
        candidate_path = os.path.join(results_dir, candidate_name)
        if os.path.isdir(candidate_path) and os.path.isfile(os.path.join(candidate_path, 'model.pt')) and os.path.isfile(os.path.join(candidate_path, 'config.yaml')):
            print(f'{candidate_path}\t{target_ver}')
            sys.exit(0)

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

# Helper to automatically resolve the secondary model directory for comparison
find_second_latest_m1_dir() {
    local target_ver="${1:-}"

    "${PYTHON_BIN}" -c "
import os, re, sys

target_ver = sys.argv[1].strip() if len(sys.argv) > 1 else ''
results_dir = 'results'
if not os.path.isdir(results_dir):
    sys.exit(1)

# Check direct directory match if target_ver provided
if target_ver:
    for candidate_name in [target_ver, f'{target_ver}_train_m1', f'{target_ver}_m1']:
        candidate_path = os.path.join(results_dir, candidate_name)
        if os.path.isdir(candidate_path) and os.path.isfile(os.path.join(candidate_path, 'model.pt')) and os.path.isfile(os.path.join(candidate_path, 'config.yaml')):
            print(f'{candidate_path}\t{target_ver}')
            sys.exit(0)

pattern = re.compile(r'^spwm_v(\d+)_(\d+)_(\d+)_train_m1$')

candidates = []
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
    if len(candidates) >= 2:
        print(f'{candidates[1][1]}\t{candidates[1][2]}')
        sys.exit(0)
    else:
        print(f'{candidates[0][1]}\t{candidates[0][2]}')
        sys.exit(0)

sys.exit(1)
" "${target_ver}"
}

# ------------------------------------------------------------------------------
# Default video variants: <target height>:<payload weight>
# Each variant uses a REAL test episode whose target height plane (low/mid/high)
# and payload mass tier (light <2.3 / normal / heavy >=3.4 kg) match, so the
# trajectory is exported per model from that episode and then rendered.
# ------------------------------------------------------------------------------
VIDEO_VARIANTS=("low:normal" "mid:normal" "high:normal" "mid:heavy")

# Args: <label_prefix> <horizon> <out_dir> <ckpt_a> <cfg_a> <ver_a> [<ckpt_b> <cfg_b> <ver_b>]
export_and_render_variants() {
    local prefix="$1" horizon="$2" out_dir="$3" ckpt_a="$4" cfg_a="$5" ver_a="$6"
    local ckpt_b="${7:-}" cfg_b="${8:-}" ver_b="${9:-}"
    local traj_dir="sim/Assets/StreamingAssets/variants"
    local variants_csv
    variants_csv="$(IFS=,; echo "${VIDEO_VARIANTS[*]}")"
    mkdir -p "${out_dir}" "${traj_dir}"

    local export_cmd=(
        "${PYTHON_BIN}" experiments/export_ghost_trajectory.py
        --checkpoint "${ckpt_a}" --config "${cfg_a}" --model-version "${ver_a}"
        --horizon "${horizon}" --variants "${variants_csv}" --output-dir "${traj_dir}"
    )
    local compare=0
    if [[ -n "${ckpt_b}" ]]; then
        compare=1
        export_cmd+=(--checkpoint-b "${ckpt_b}" --config-b "${cfg_b}" --model-version-b "${ver_b}")
    fi
    "${export_cmd[@]}"

    local variant height weight
    for variant in "${VIDEO_VARIANTS[@]}"; do
        height="${variant%%:*}"
        weight="${variant##*:}"
        echo ">>> Rendering ${prefix}_${height}_${weight}"
        local cmd=(
            "${PYTHON_BIN}" "scripts/render_simulation_video.py"
            "--json-file" "${traj_dir}/trajectory_ghost_${height}_${weight}.json"
            "--output" "${out_dir}/${prefix}_${height}_${weight}.mp4"
            "--horizon" "${horizon}"
        )
        [[ "${compare}" -eq 1 ]] && cmd+=("--compare-models")
        "${cmd[@]}"
    done
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
  train-m1           Train M1 (Phase 1) with Two-Tiered Validation using frozen Phase 0 representation.
  baselines          Train and benchmark non-spiking comparative baselines against SPWM.
  export-unity       Export ghost trajectory JSON for Unity demo playback.
  video              Export (per model, from real test episodes matching each case) and render 4
                     labeled videos <model>_<low|mid|high>_<light|normal|heavy>.mp4:
                     low/mid/high + normal, and mid + heavy.
  video-compare      Auto-detect two models and render the 4 labeled variants as
                     comparison_BLUE_<modelA>_RED_<modelB>_<height>_<weight>.mp4.
  video-mppi         Export and render closed-loop Latent MPPI control video
                     (<output-video> or results/videos/<model>_mppi_control.mp4).
  all                Run complete pipeline: train -> export-unity -> video.

Basic Options:
  --trajectories <N>     Number of trajectories to export in sim-data (default: ${DEFAULT_TOTAL_TRAJECTORIES}).
  --export-dir <path>    Output directory for exported dataset (default: ${DEFAULT_EXPORT_DIR}).
  --epochs <N>           Epochs for training (default: ${DEFAULT_EPOCHS}).
  --config, --config-a   Path to model A experiment config.yaml (optional for train).
  --config-b <path>      Path to model B experiment config.yaml for dual comparison.
  --checkpoint, -a       Path to model A checkpoint (.pt) for export.
  --checkpoint-b <path>  Path to model B checkpoint (.pt) for dual-arm comparison.
  --model-version, -a    Model A version tag (default: ${DEFAULT_MODEL_VERSION}).
  --model-version-b      Model B version tag (default: ${DEFAULT_MODEL_VERSION_B}).
  --trajectory-id <ID>   Specific trajectory ID to export/render for MPPI (default: auto-detected best test episode).
  --horizon <H>          Rollout / render horizon steps (default: ${DEFAULT_HORIZON}).
  --episodes <N>         Number of episodes to export / render (default: ${DEFAULT_EPISODES}).
  --multi-episode        Render 5 episodes across different episodes, each with and without injection.
  --compare-injection    Render/export trajectories both with and without ground truth sensory injection.
  --json-file <path>     JSON path for video rendering (default: sim/Assets/StreamingAssets/trajectory_ghost.json).
  --output-video <path>  Render ONE video to this path instead of the 4 default labeled variants.
  --output-dir <path>    Output directory for videos (default: ${DEFAULT_OUTPUT_DIR}).
  --logfile <path>       Log file for Unity headless batch export (default: export_sim.log).
  -h, --help             Show this help message.

Quick Start Examples:
  # 1. Render single-command comparison video between the two latest models:
  ./scripts/run_pipeline.sh video-compare

  # 2. Render dual-arm comparison with explicit models:
  ./scripts/run_pipeline.sh video-compare --checkpoint-a results/m1_A/model.pt --checkpoint-b results/m1_B/model.pt
EOF
}

# ------------------------------------------------------------------------------
# Action 1: Headless Simulation Dataset Generation (Unity Batchmode)
# ------------------------------------------------------------------------------
run_sim_data() {
    local trajectories="${TOTAL_TRAJECTORIES:-$DEFAULT_TOTAL_TRAJECTORIES}"
    local export_dir="${EXPORT_DIR:-$DEFAULT_EXPORT_DIR}"
    local logfile="${LOGFILE:-export_sim.log}"
    local sim_path="${PROJECT_ROOT}/sim"

    if [[ ! -x "${UNITY_BIN}" ]]; then
        echo "Error: Unity binary not found or not executable at '${UNITY_BIN}'." >&2
        exit 1
    fi

    echo "======================================================================"
    echo " Exporting Synthetic Trajectory Dataset via Unity Headless Batchmode..."
    echo "======================================================================"

    "${UNITY_BIN}" \
        -batchmode \
        -projectPath "${sim_path}" \
        -executeMethod SPWM.Editor.GhostVisualizerSceneBuilder.HeadlessBatchExport \
        -totalTrajectories "${trajectories}" \
        -exportDir "${export_dir}" \
        -logfile "${logfile}" \
        -quit

    echo ">>> Unity simulation export completed successfully."
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

    echo "======================================================================"
    echo " [2/2] Training M1 Latent Predictor Phase (Phase 1)..."
    echo "======================================================================"
    local m1_cmd=("${PYTHON_BIN}" "experiments/train.py" "--mode" "train_m1" "--epochs" "${epochs}")
    if [[ -n "${cfg}" ]]; then
        m1_cmd+=("--config" "${cfg}")
    fi
    "${m1_cmd[@]}"

    echo ">>> Training phases completed successfully."
}

# ------------------------------------------------------------------------------
# Action 2b: Training M1 Only (Phase 1 with Two-Tiered Validation)
# ------------------------------------------------------------------------------
run_train_m1() {
    local epochs="${EPOCHS:-100}"
    local cfg="${CONFIG_PATH:-}"
    local from_scratch="${FROM_SCRATCH:-1}"
    local weighted_sampling="${WEIGHTED_SAMPLING:-1}"

    echo "======================================================================"
    echo " Training M1 Latent Predictor Phase (Phase 1)..."
    echo " Two-Tiered Validation: Tier 1 (Open-Loop MSE) + Tier 2 (Closed-Loop MPPI)"
    echo " Prerequisite: Strictly loading Phase 0 representation checkpoint"
    echo "======================================================================"
    local m1_cmd=("${PYTHON_BIN}" "experiments/train.py" "--mode" "train_m1" "--epochs" "${epochs}")
    if [[ -n "${cfg}" ]]; then
        m1_cmd+=("--config" "${cfg}")
    fi
    if [[ "${from_scratch}" -eq 1 ]]; then
        m1_cmd+=("--from-scratch")
    fi
    if [[ "${weighted_sampling}" -eq 1 ]]; then
        m1_cmd+=("--weighted-sampling")
    fi
    "${m1_cmd[@]}"

    echo ">>> M1 training completed successfully."
}

# ------------------------------------------------------------------------------
# Action: Comparative Baselines Benchmark
# ------------------------------------------------------------------------------
run_baselines() {
    exec "${SCRIPT_DIR}/run_baselines.sh" "$@"
}

# ------------------------------------------------------------------------------
# Action 3: Export Ghost Trajectory for Unity
# ------------------------------------------------------------------------------
run_export_unity() {
    local model_ver="${MODEL_VERSION:-$DEFAULT_MODEL_VERSION}"
    local ckpt="${CHECKPOINT_PATH:-}"
    local cfg="${CONFIG_PATH:-}"
    local ckpt_b="${CHECKPOINT_B_PATH:-}"
    local cfg_b="${CONFIG_B_PATH:-}"
    local model_ver_b="${MODEL_VERSION_B:-$DEFAULT_MODEL_VERSION_B}"
    local horizon="${HORIZON:-$DEFAULT_HORIZON}"
    local episodes="${EPISODES:-$DEFAULT_EPISODES}"
    local compare_injection="${COMPARE_INJECTION:-0}"

    if [[ -z "${ckpt}" || -z "${cfg}" ]]; then
        local detected_output
        if detected_output="$(find_latest_m1_dir "${MODEL_VERSION:-}")"; then
            local detected_dir
            local detected_ver
            detected_dir="$(echo "${detected_output}" | cut -f1)"
            detected_ver="$(echo "${detected_output}" | cut -f2)"
            [[ -z "${ckpt}" ]] && ckpt="${detected_dir}/model.pt"
            [[ -z "${cfg}" ]] && cfg="${detected_dir}/config.yaml"
            [[ -z "${MODEL_VERSION:-}" ]] && model_ver="${detected_ver}"
        else
            echo "Error: Could not automatically detect a valid M1 checkpoint." >&2
            exit 1
        fi
    fi

    echo "======================================================================"
    echo " Generating Ghost Trajectory JSON for Unity Demo..."
    echo " Checkpoint A:      ${ckpt}"
    if [[ -n "${ckpt_b}" ]]; then
        echo " Checkpoint B:      ${ckpt_b}"
        echo " Dual-Model Mode:   Enabled"
    fi
    echo " Config A:          ${cfg}"
    echo " Horizon:           ${horizon}"
    echo "======================================================================"

    local export_cmd=(
        "${PYTHON_BIN}" experiments/export_ghost_trajectory.py
        --checkpoint "${ckpt}"
        --config "${cfg}"
        --model-version "${model_ver}"
        --horizon "${horizon}"
        --episodes "${episodes}"
    )

    if [[ "${compare_injection}" -eq 1 || "${episodes}" -gt 1 ]]; then
        export_cmd+=("--compare-injection")
    fi

    if [[ -n "${ckpt_b}" ]]; then
        export_cmd+=("--checkpoint-b" "${ckpt_b}")
        [[ -n "${cfg_b}" ]] && export_cmd+=("--config-b" "${cfg_b}")
        export_cmd+=("--model-version-b" "${model_ver_b}")
    fi

    "${export_cmd[@]}"

    echo ">>> Ghost trajectory export completed."
}

# ------------------------------------------------------------------------------
# Action 4: Render Simulation Video (.mp4)
# ------------------------------------------------------------------------------
run_video() {
    local episodes="${EPISODES:-$DEFAULT_EPISODES}"
    local compare_injection="${COMPARE_INJECTION:-0}"
    local horizon="${HORIZON:-$DEFAULT_HORIZON}"
    local out_dir="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"

    if [[ "${episodes}" -gt 1 || "${compare_injection}" -eq 1 ]]; then
        local target_episodes="${episodes}"
        if [[ "${target_episodes}" -lt 1 ]]; then
            target_episodes=1
        fi

        mkdir -p "${out_dir}"
        echo "======================================================================"
        echo " Rendering Multi-Episode Simulation Videos (.mp4)..."
        echo "======================================================================"

        local missing_json=0
        for ((i=0; i<target_episodes; i++)); do
            for mode in "with_injection" "no_injection"; do
                local expected_json="sim/Assets/StreamingAssets/trajectory_ghost_ep${i}_${mode}.json"
                if [[ ! -f "${expected_json}" ]]; then
                    missing_json=1
                    break 2
                fi
            done
        done

        if [[ "${missing_json}" -eq 1 ]]; then
            echo "[video] Required trajectory JSON files missing. Auto-running export..."
            EPISODES="${target_episodes}" COMPARE_INJECTION=1 HORIZON="${horizon}" run_export_unity
        fi

        local base_prefix
        if [[ -n "${OUTPUT_VIDEO}" ]]; then
            base_prefix="$(dirname "${OUTPUT_VIDEO}")/$(basename "${OUTPUT_VIDEO}" .mp4)"
        else
            base_prefix="${out_dir}/robot_arm_simulation"
        fi

        for ((i=0; i<target_episodes; i++)); do
            for mode in "with_injection" "no_injection"; do
                local json_file="sim/Assets/StreamingAssets/trajectory_ghost_ep${i}_${mode}.json"
                local output_video="${base_prefix}_ep${i}_${mode}.mp4"

                local video_cmd=(
                    "${PYTHON_BIN}" "scripts/render_simulation_video.py"
                    "--json-file" "${json_file}"
                    "--output" "${output_video}"
                    "--horizon" "${horizon}"
                )

                if [[ -n "${CHECKPOINT_B_PATH:-}" ]]; then
                    video_cmd+=("--compare-models")
                fi

                "${video_cmd[@]}"
            done
        done

        echo ">>> All multi-episode videos rendered successfully."
    else
        local json_file="${JSON_FILE:-sim/Assets/StreamingAssets/trajectory_ghost.json}"

        echo "======================================================================"
        echo " Rendering Unity Simulation Video(s) (.mp4)..."
        echo "======================================================================"

        if [[ -n "${OUTPUT_VIDEO}" ]]; then
            if [[ ! -f "${json_file}" ]]; then
                echo "[video] '${json_file}' not found. Auto-running export-unity first..."
                run_export_unity
            fi
            local output_video="${OUTPUT_VIDEO}"
            mkdir -p "$(dirname "${output_video}")"
            local video_cmd=(
                "${PYTHON_BIN}" "scripts/render_simulation_video.py"
                "--json-file" "${json_file}"
                "--output" "${output_video}"
                "--horizon" "${horizon}"
            )
            [[ -n "${CHECKPOINT_B_PATH:-}" ]] && video_cmd+=("--compare-models")
            "${video_cmd[@]}"
            echo ">>> Video rendered successfully at: ${output_video}"
        else
            local ckpt_a="${CHECKPOINT_PATH:-}" cfg_a="${CONFIG_PATH:-}" ver_a="${MODEL_VERSION:-}"
            if [[ -z "${ckpt_a}" || -z "${cfg_a}" ]]; then
                local detected_a dir_a
                if ! detected_a="$(find_latest_m1_dir "${ver_a}")"; then
                    echo "Error: Could not detect a valid M1 checkpoint." >&2
                    exit 1
                fi
                dir_a="$(echo "${detected_a}" | cut -f1)"
                [[ -z "${ckpt_a}" ]] && ckpt_a="${dir_a}/model.pt"
                [[ -z "${cfg_a}" ]] && cfg_a="${dir_a}/config.yaml"
                [[ -z "${ver_a}" ]] && ver_a="$(echo "${detected_a}" | cut -f2)"
            fi
            if [[ -n "${CHECKPOINT_B_PATH:-}" ]]; then
                local ver_b="${MODEL_VERSION_B:-$DEFAULT_MODEL_VERSION_B}"
                export_and_render_variants "comparison_BLUE_${ver_a}_RED_${ver_b}" "${horizon}" "${out_dir}" \
                    "${ckpt_a}" "${cfg_a}" "${ver_a}" "${CHECKPOINT_B_PATH}" "${CONFIG_B_PATH:-}" "${ver_b}"
            else
                export_and_render_variants "${ver_a}" "${horizon}" "${out_dir}" "${ckpt_a}" "${cfg_a}" "${ver_a}"
            fi
            echo ">>> 4 videos rendered successfully in: ${out_dir}"
        fi
    fi
}

# ------------------------------------------------------------------------------
# Action 5: Compare Models Simulation Video (Zero-Config Default Runner)
# ------------------------------------------------------------------------------
run_video_compare() {
    local horizon="${HORIZON:-$DEFAULT_HORIZON}"
    local output_video="${OUTPUT_VIDEO}"
    local json_file="${JSON_FILE:-sim/Assets/StreamingAssets/trajectory_ghost_dual_comparison.json}"

    local ckpt_a="${CHECKPOINT_PATH:-}"
    local cfg_a="${CONFIG_PATH:-}"
    local ver_a="${MODEL_VERSION:-}"

    local ckpt_b="${CHECKPOINT_B_PATH:-}"
    local cfg_b="${CONFIG_B_PATH:-}"
    local ver_b="${MODEL_VERSION_B:-$DEFAULT_MODEL_VERSION_B}"

    # 1. Resolve Model A if missing
    if [[ -z "${ckpt_a}" || -z "${cfg_a}" ]]; then
        local detected_a
        if detected_a="$(find_latest_m1_dir "${ver_a}")"; then
            local dir_a
            dir_a="$(echo "${detected_a}" | cut -f1)"
            [[ -z "${ckpt_a}" ]] && ckpt_a="${dir_a}/model.pt"
            [[ -z "${cfg_a}" ]] && cfg_a="${dir_a}/config.yaml"
            [[ -z "${ver_a}" ]] && ver_a="$(echo "${detected_a}" | cut -f2)"
        else
            echo "Error: Could not automatically detect a valid Model A M1 checkpoint." >&2
            exit 1
        fi
    fi

    # 2. Resolve Model B if missing
    if [[ -z "${ckpt_b}" || -z "${cfg_b}" ]]; then
        local detected_b
        if detected_b="$(find_second_latest_m1_dir "${ver_b}")"; then
            local dir_b
            dir_b="$(echo "${detected_b}" | cut -f1)"
            [[ -z "${ckpt_b}" ]] && ckpt_b="${dir_b}/model.pt"
            [[ -z "${cfg_b}" ]] && cfg_b="${dir_b}/config.yaml"
            [[ -z "${ver_b}" ]] && ver_b="$(echo "${detected_b}" | cut -f2)"
        else
            echo "Error: Could not detect a distinct Model B (needs model.pt AND config.yaml). Pass --checkpoint-b and --config-b explicitly." >&2
            exit 1
        fi
    fi

    echo "======================================================================"
    echo " Dual Model Comparison"
    echo " Model A (BLUE): ${ckpt_a} (${ver_a})"
    echo " Model B (RED):  ${ckpt_b} (${ver_b})"
    echo " Horizon:        ${horizon}"
    echo " Output:         ${OUTPUT_VIDEO:-${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR} (4 labeled variants)}"
    echo "======================================================================"

    if [[ -n "${OUTPUT_VIDEO}" ]]; then
        "${PYTHON_BIN}" experiments/export_ghost_trajectory.py \
            --checkpoint "${ckpt_a}" --config "${cfg_a}" --model-version "${ver_a}" \
            --checkpoint-b "${ckpt_b}" --config-b "${cfg_b}" --model-version-b "${ver_b}" \
            --horizon "${horizon}" --output "${json_file}"
        mkdir -p "$(dirname "${output_video}")"
        "${PYTHON_BIN}" "scripts/render_simulation_video.py" \
            "--json-file" "${json_file}" \
            "--output" "${output_video}" \
            "--horizon" "${horizon}" \
            "--compare-models"
        echo ">>> Dual comparison video rendered successfully at: ${output_video}"
    else
        local out_dir="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
        export_and_render_variants "comparison_BLUE_${ver_a}_RED_${ver_b}" "${horizon}" "${out_dir}" \
            "${ckpt_a}" "${cfg_a}" "${ver_a}" "${ckpt_b}" "${cfg_b}" "${ver_b}"
        echo ">>> 4 dual comparison videos rendered successfully in: ${out_dir}"
    fi
}

# ------------------------------------------------------------------------------
# Action 6: Closed-Loop Latent MPPI Simulation Video
# ------------------------------------------------------------------------------
run_video_mppi() {
    local horizon="${HORIZON:-$DEFAULT_HORIZON}"
    local traj_id="${TRAJECTORY_ID:-3622}"
    local out_dir="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    local output_video="${OUTPUT_VIDEO:-${out_dir}/mppi_control.mp4}"
    local json_file="${JSON_FILE:-sim/Assets/StreamingAssets/trajectory_ghost_mppi.json}"

    local ckpt="${CHECKPOINT_PATH:-}"
    local cfg="${CONFIG_PATH:-}"
    local ver="${MODEL_VERSION:-}"

    if [[ -z "${ckpt}" || -z "${cfg}" ]]; then
        local detected
        if detected="$(find_latest_m1_dir "${ver}")"; then
            local dir_a
            dir_a="$(echo "${detected}" | cut -f1)"
            [[ -z "${ckpt}" ]] && ckpt="${dir_a}/model.pt"
            [[ -z "${cfg}" ]] && cfg="${dir_a}/config.yaml"
            [[ -z "${ver}" ]] && ver="$(echo "${detected}" | cut -f2)"
        else
            echo "Error: Could not automatically detect a valid M1 checkpoint." >&2
            exit 1
        fi
    fi

    echo "======================================================================"
    echo " Rendering Closed-Loop Latent MPPI Simulation Video (.mp4)..."
    echo " Checkpoint:     ${ckpt} (${ver})"
    echo " Config:         ${cfg}"
    echo " Trajectory ID:  ${traj_id}"
    echo " Horizon:        ${horizon}"
    echo " Output Video:   ${output_video}"
    echo "======================================================================"

    mkdir -p "$(dirname "${json_file}")"
    mkdir -p "$(dirname "${output_video}")"

    "${PYTHON_BIN}" experiments/export_mppi_trajectory.py \
        --checkpoint "${ckpt}" \
        --config "${cfg}" \
        --trajectory-id "${traj_id}" \
        --horizon "${horizon}" \
        --output "${json_file}"

    "${PYTHON_BIN}" scripts/render_simulation_video.py \
        --json-file "${json_file}" \
        --output "${output_video}" \
        --horizon "${horizon}"

    echo ">>> Latent MPPI simulation video rendered successfully at: ${output_video}"
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

if [[ "${COMMAND}" == "baselines" || "${COMMAND}" == "baseline" || "${COMMAND}" == "run-baselines" || "${COMMAND}" == "run_baselines" || "${COMMAND}" == "run-baseline" || "${COMMAND}" == "run_baseline" ]]; then
    run_baselines "$@"
    exit 0
fi

TOTAL_TRAJECTORIES=""
EXPORT_DIR=""
LOGFILE=""
EPOCHS=""
CONFIG_PATH=""
CONFIG_B_PATH=""
CHECKPOINT_PATH=""
CHECKPOINT_B_PATH=""
MODEL_VERSION=""
MODEL_VERSION_B=""
HORIZON=""
VIDEO_SOURCE=""
JSON_FILE=""
OUTPUT_VIDEO=""
EPISODES=""
COMPARE_INJECTION=""
OUTPUT_DIR=""
TRAJECTORY_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trajectories|-t)
            TOTAL_TRAJECTORIES="$2"
            shift 2
            ;;
        --trajectory-id)
            TRAJECTORY_ID="$2"
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
        --config|--config-a)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --config-b)
            CONFIG_B_PATH="$2"
            shift 2
            ;;
        --checkpoint|--checkpoint-a|-a)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --checkpoint-b)
            CHECKPOINT_B_PATH="$2"
            shift 2
            ;;
        --model-version|--model-version-a)
            MODEL_VERSION="$2"
            shift 2
            ;;
        --model-version-b)
            MODEL_VERSION_B="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --episodes|-e)
            EPISODES="$2"
            shift 2
            ;;
        --multi-episode|--multi-video)
            EPISODES=5
            COMPARE_INJECTION=1
            shift 1
            ;;
        --compare-injection)
            COMPARE_INJECTION=1
            shift 1
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
        --output-dir)
            OUTPUT_DIR="$2"
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

if [[ -n "${EPISODES}" && "${EPISODES}" -gt 1 ]]; then
    COMPARE_INJECTION=1
fi

case "${COMMAND}" in
    sim-data|sim_data|export-data|export_data|data|exportTrajectories|export-trajectories)
        run_sim_data
        ;;
    1|train)
        run_train
        ;;
    train-m1|train_m1|m1)
        run_train_m1
        ;;
    baselines|baseline|run-baselines|run_baselines|run-baseline|run_baseline)
        run_baselines "$@"
        ;;
    2|video)
        run_video
        ;;
    video-compare|video-comparison|compare-video)
        run_video_compare
        ;;
    video-mppi|mppi-video|mppi)
        run_video_mppi
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
