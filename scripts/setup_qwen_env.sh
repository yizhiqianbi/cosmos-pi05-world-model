#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_VENV="${QWEN_VENV:-${REPOSITORY_ROOT}/.venv-qwen}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -x "${QWEN_VENV}/bin/python" ]]; then
  uv venv --python 3.11 "${QWEN_VENV}"
fi
uv pip install \
  --python "${QWEN_VENV}/bin/python" \
  --index-url "${PYTORCH_INDEX_URL}" \
  torch==2.7.1 \
  torchvision==0.22.1
uv pip install \
  --python "${QWEN_VENV}/bin/python" \
  --requirements "${REPOSITORY_ROOT}/scripts/requirements-qwen.txt"

"${QWEN_VENV}/bin/python" - <<'PY'
import importlib.metadata as metadata

for package in ("torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d"):
    print(f"{package}=={metadata.version(package)}")
PY
