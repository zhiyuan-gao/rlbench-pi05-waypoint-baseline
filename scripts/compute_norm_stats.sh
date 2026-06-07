#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or run scripts/setup_env.sh}"

cd "${OPENPI_DIR}"
uv run scripts/compute_norm_stats.py \
  --config-name "${CONFIG_NAME:-pi05_rlbench_waypoint_h1}" \
  "$@"

