#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or run scripts/setup_env.sh}"

cd "${OPENPI_DIR}"
uv run python -m rlbench_pi05_waypoint.smoke_lerobot_dataset \
  --repo-id "${REPO_ID:-rlbench/selected10_pi05_waypoint_h1}" \
  --num-samples "${NUM_SAMPLES:-4}" \
  "$@"

