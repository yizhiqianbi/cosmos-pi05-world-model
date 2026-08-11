#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

: "${QWEN_BASE:?Set QWEN_BASE to the Qwen3.5-9B base model path}"
: "${QWEN_ADAPTER_ROOT:?Set QWEN_ADAPTER_ROOT to the proposal/value/reflection adapter root}"
: "${COSMOS_MODEL:?Set COSMOS_MODEL to the Cosmos3-Nano snapshot}"
: "${PI05_CHECKPOINT:?Set PI05_CHECKPOINT to a pi05_libero_long_subgoal checkpoint step}"

HIGH_LEVEL_PORT="${HIGH_LEVEL_PORT:-10090}"
COSMOS_PORT="${COSMOS_PORT:-10091}"
POLICY_PORT="${POLICY_PORT:-8000}"
QWEN_GPU="${QWEN_GPU:-0}"
QWEN_PYTHON="${QWEN_PYTHON:-${REPOSITORY_ROOT}/.venv-qwen/bin/python}"
COSMOS_GPUS="${COSMOS_GPUS:-1,2,3}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${REPOSITORY_ROOT}/external/cosmos-framework/.venv/bin/python}"
COSMOS_MAX_MEMORY_GIB="${COSMOS_MAX_MEMORY_GIB:-0}"
COSMOS_DEPLOY_FRAMES="${COSMOS_DEPLOY_FRAMES:-5}"
PI05_GPUS="${PI05_GPUS:-4}"
HIGH_LEVEL_INTERVAL="${HIGH_LEVEL_INTERVAL:-10}"
LOG_DIR="${LOG_DIR:-outputs/service_logs}"
if [[ ! -x "${QWEN_PYTHON}" ]]; then
  echo "Missing Qwen runtime ${QWEN_PYTHON}; run scripts/setup_qwen_env.sh first." >&2
  exit 2
fi
if [[ ! -x "${COSMOS_PYTHON}" ]]; then
  echo "Missing Cosmos runtime ${COSMOS_PYTHON}; run scripts/cosmos/setup_cosmos_framework.sh first." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"

children=()
cleanup() {
  for pid in "${children[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="${QWEN_GPU}" "${QWEN_PYTHON}" scripts/services/high_level_server.py \
  --local-base "${QWEN_BASE}" \
  --local-adapter-root "${QWEN_ADAPTER_ROOT}" \
  --local-device cuda:0 \
  --port "${HIGH_LEVEL_PORT}" \
  >"${LOG_DIR}/high_level.log" 2>&1 &
children+=("$!")

CUDA_VISIBLE_DEVICES="${COSMOS_GPUS}" "${COSMOS_PYTHON}" scripts/services/cosmos_server.py \
  --model "${COSMOS_MODEL}" \
  --diffusers \
  --diffusers-device-map balanced \
  --diffusers-max-memory-gib "${COSMOS_MAX_MEMORY_GIB}" \
  --diffusers-deploy-num-frames "${COSMOS_DEPLOY_FRAMES}" \
  --port "${COSMOS_PORT}" \
  >"${LOG_DIR}/cosmos.log" 2>&1 &
children+=("$!")

for endpoint in \
  "http://127.0.0.1:${HIGH_LEVEL_PORT}/health" \
  "http://127.0.0.1:${COSMOS_PORT}/health"; do
  for _ in $(seq 1 180); do
    if curl --fail --silent "${endpoint}" >/dev/null; then
      break
    fi
    sleep 2
  done
  curl --fail --silent "${endpoint}" >/dev/null
done

CUDA_VISIBLE_DEVICES="${PI05_GPUS}" uv run python scripts/serve_hierarchical_policy.py \
  --checkpoint-dir "${PI05_CHECKPOINT}" \
  --high-level-endpoint "http://127.0.0.1:${HIGH_LEVEL_PORT}" \
  --cosmos-endpoint "http://127.0.0.1:${COSMOS_PORT}" \
  --high-level-interval "${HIGH_LEVEL_INTERVAL}" \
  --pytorch-device cuda:0 \
  --port "${POLICY_PORT}"
