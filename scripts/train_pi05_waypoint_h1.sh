#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or run scripts/setup_env.sh}"

CONFIG_NAME="${CONFIG_NAME:-pi05_rlbench_waypoint_h1}"
EXP_NAME="${EXP_NAME:-selected10_pi05_waypoint_h1}"

EXTRA_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${WANDB_ENABLED:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-wandb-enabled)
fi

cd "${OPENPI_DIR}"
uv run scripts/train.py "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
