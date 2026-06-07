#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PI05_ROOT="${PI05_ROOT:-${REPO_ROOT}/external/pi05_baseline}"
OPENPI_DIR="${OPENPI_DIR:-${PI05_ROOT}/openpi}"
OPENPI_REPO="${OPENPI_REPO:-https://github.com/Physical-Intelligence/openpi.git}"
OPENPI_REF="${OPENPI_REF:-c23745b5ad24e98f66967ea795a07b2588ed6c79}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PI05_ROOT}/openpi_cache}"
export HF_HOME="${HF_HOME:-${PI05_ROOT}/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI05_ROOT}/lerobot_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PI05_ROOT}/uv_cache}"
export GIT_LFS_SKIP_SMUDGE=1

mkdir -p "${PI05_ROOT}" "${OPENPI_DATA_HOME}" "${HF_HOME}" "${HF_LEROBOT_HOME}" "${UV_CACHE_DIR}"

if [[ ! -d "${OPENPI_DIR}/.git" ]]; then
  git clone --recurse-submodules --shallow-submodules --filter=blob:none --depth=1 \
    "${OPENPI_REPO}" "${OPENPI_DIR}"
fi

git -C "${OPENPI_DIR}" fetch --depth=1 origin "${OPENPI_REF}" || true
git -C "${OPENPI_DIR}" checkout "${OPENPI_REF}"
git -C "${OPENPI_DIR}" submodule update --init --recursive --depth=1

cd "${OPENPI_DIR}"
uv sync
uv pip install -e .

python "${REPO_ROOT}/tools/patch_openpi_rlbench_config.py" \
  --openpi-dir "${OPENPI_DIR}" \
  --repo-root "${REPO_ROOT}"

echo "OpenPI installed at: ${OPENPI_DIR}"
echo "Use: source ${REPO_ROOT}/scripts/setup_env.sh"

