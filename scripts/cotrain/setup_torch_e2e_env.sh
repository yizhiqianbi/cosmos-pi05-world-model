#!/usr/bin/env bash
# Build an isolated Python 3.11 environment for the two-model PyTorch graph.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVIRONMENT_DIR="$(realpath -m "${1:-${REPOSITORY_ROOT}/.venv-e2e}")"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required" >&2; exit 1; }
if [[ ! -x "${ENVIRONMENT_DIR}/bin/python" ]]; then
  uv venv --python 3.11 "${ENVIRONMENT_DIR}"
fi

PYTHON="${ENVIRONMENT_DIR}/bin/python"
uv pip install --python "${PYTHON}" -e "${REPOSITORY_ROOT}[e2e]"

TRANSFORMERS_DIR="$("${PYTHON}" - <<'PY'
from pathlib import Path
import transformers
print(Path(transformers.__file__).resolve().parent)
PY
)"
cp -r "${REPOSITORY_ROOT}/src/openpi/models_pytorch/transformers_replace/." "${TRANSFORMERS_DIR}/"

"${PYTHON}" - <<'PY'
import diffusers
import torch
import transformers
from diffusers import Cosmos3OmniPipeline
from transformers.models.siglip import check

assert diffusers.__version__ == "0.39.0"
assert transformers.__version__ == "4.53.2"
assert check.check_whether_transformers_replace_is_installed_correctly()
print(f"torch={torch.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__}")
print(f"Cosmos pipeline={Cosmos3OmniPipeline.__name__}")
PY

echo "End-to-end environment ready: ${ENVIRONMENT_DIR}"
