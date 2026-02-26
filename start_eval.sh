#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
CONFIG_PATH="${1:-${CONFIG_PATH:-config.yaml}}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[INFO] uv not found. Installing uv with python3 -m pip --user ..."
  python3 -m pip install --user --upgrade uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] Failed to install/find uv on PATH."
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[INFO] Creating virtual environment at ${VENV_DIR}"
  uv venv "${VENV_DIR}" --python python3
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}"
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[INFO] Installing requirements with uv pip ..."
uv pip install --upgrade pip
uv pip install -r requirements.txt

echo "[INFO] Running evaluation with config: ${CONFIG_PATH}"
python run_eval.py --config "${CONFIG_PATH}"
