#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/setup_env.sh"

: "${OPENPI_DIR:?Set OPENPI_DIR or run scripts/setup_env.sh}"

python "${REPO_ROOT}/tools/patch_openpi_rlbench_config.py" \
  --openpi-dir "${OPENPI_DIR}" \
  --repo-root "${REPO_ROOT}"

