#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_VENV="${LIBERO_EVAL_VENV:-${REPOSITORY_ROOT}/.venv-libero}"

cd "${REPOSITORY_ROOT}"
git submodule update --init --recursive third_party/libero
uv venv --python 3.8 "${EVAL_VENV}"
uv pip sync \
  --python "${EVAL_VENV}/bin/python" \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install \
  --python "${EVAL_VENV}/bin/python" \
  --no-deps \
  -e packages/openpi-client \
  -e third_party/libero

echo "LIBERO evaluator environment is ready: ${EVAL_VENV}"
