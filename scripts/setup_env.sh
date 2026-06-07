#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PI05_ROOT="${PI05_ROOT:-${REPO_ROOT}/external/pi05_baseline}"
export OPENPI_DIR="${OPENPI_DIR:-${PI05_ROOT}/openpi}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${PI05_ROOT}/openpi_cache}"
export HF_HOME="${HF_HOME:-${PI05_ROOT}/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PI05_ROOT}/lerobot_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PI05_ROOT}/uv_cache}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

