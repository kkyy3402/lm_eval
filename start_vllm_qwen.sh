#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_DIR="${LOG_DIR:-logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/vllm.pid}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/vllm.log}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
READY_CHECK_INTERVAL_SEC="${READY_CHECK_INTERVAL_SEC:-2}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] python3 not found."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found. vLLM requires a compatible NVIDIA GPU environment."
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

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" >/dev/null 2>&1; then
    echo "[INFO] vLLM already running (pid=${OLD_PID})."
    echo "[INFO] Log: ${LOG_FILE}"
    exit 0
  fi
fi

uv venv "${VENV_DIR}" --python python3
VENV_PYTHON="${VENV_DIR}/bin/python"

uv pip install --python "${VENV_PYTHON}" --upgrade pip
uv pip install --python "${VENV_PYTHON}" "vllm>=0.5.5"
uv pip install --python "${VENV_PYTHON}" -r requirements.txt

nohup "${VENV_PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  > "${LOG_FILE}" 2>&1 &

NEW_PID="$!"
echo "${NEW_PID}" > "${PID_FILE}"
echo "[INFO] Started vLLM server (pid=${NEW_PID})"
echo "[INFO] Waiting for http://127.0.0.1:${PORT}/v1/models ..."

START_TS="$(date +%s)"
while true; do
  if "${VENV_PYTHON}" - "${PORT}" <<'PY'
import sys
import urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        ok = r.status == 200
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[INFO] vLLM is ready."
    echo "[INFO] Run eval:"
    echo "       source ${VENV_DIR}/bin/activate && python run_eval.py --config config.yaml"
    exit 0
  fi
  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if (( ELAPSED >= READY_TIMEOUT_SEC )); then
    break
  fi
  sleep "${READY_CHECK_INTERVAL_SEC}"
done

echo "[ERROR] vLLM server did not become ready in time."
echo "[INFO] Waited ${READY_TIMEOUT_SEC}s for readiness."
echo "[INFO] Last logs:"
tail -n 100 "${LOG_FILE}" || true
exit 1
