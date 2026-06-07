#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

MANIFEST="${MANIFEST:-${REPO_ROOT}/manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl}"
REPO_ID="${REPO_ID:-rlbench/selected10_pi05_waypoint_h1}"
SPLIT="${SPLIT:-train}"

: "${OPENPI_DIR:?Set OPENPI_DIR or run scripts/setup_env.sh}"
: "${RGB_ROOT_200:?Set RGB_ROOT_200 to the all200 RGB root}"
: "${RGB_ROOT_400:?Set RGB_ROOT_400 to the all400 RGB root}"
: "${LOWDIM_ROOT_200:?Set LOWDIM_ROOT_200 to the all200 low-dim metadata root}"
: "${LOWDIM_ROOT_400:?Set LOWDIM_ROOT_400 to the all400 low-dim metadata root}"

TASK_ARGS=()
if [[ -n "${TASKS:-}" ]]; then
  for task in ${TASKS}; do
    TASK_ARGS+=(--task "${task}")
  done
fi

OPTIONAL_ARGS=()
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--overwrite)
fi
if [[ "${VALIDATE_IMAGE_PATHS:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--validate-image-paths)
fi
if [[ -n "${MAX_EPISODES:-}" ]]; then
  OPTIONAL_ARGS+=(--max-episodes "${MAX_EPISODES}")
fi
if [[ -n "${MAX_EPISODES_PER_TASK:-}" ]]; then
  OPTIONAL_ARGS+=(--max-episodes-per-task "${MAX_EPISODES_PER_TASK}")
fi

cd "${OPENPI_DIR}"
uv run python -m rlbench_pi05_waypoint.convert_to_lerobot \
  --manifest "${MANIFEST}" \
  --repo-id "${REPO_ID}" \
  --split "${SPLIT}" \
  --rgb-root-200 "${RGB_ROOT_200}" \
  --rgb-root-400 "${RGB_ROOT_400}" \
  --lowdim-root-200 "${LOWDIM_ROOT_200}" \
  --lowdim-root-400 "${LOWDIM_ROOT_400}" \
  --state-mode "${STATE_MODE:-ee_rotvec}" \
  --image-size "${IMAGE_SIZE:-256}" \
  --sample-every-n "${SAMPLE_EVERY_N:-0}" \
  --summary-out "${SUMMARY_OUT:-${REPO_ROOT}/outputs/conversion_${SPLIT}.summary.json}" \
  "${TASK_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}" \
  "$@"

