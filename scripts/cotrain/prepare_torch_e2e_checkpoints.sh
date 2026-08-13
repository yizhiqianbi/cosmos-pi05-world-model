#!/usr/bin/env bash
# Export the completed Cosmos DCP run and convert the completed pi0.5 JAX run.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 COSMOS_FRAMEWORK COSMOS_RUN PI_JAX_CHECKPOINT OUTPUT_ROOT [E2E_PYTHON]" >&2
  exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COSMOS_FRAMEWORK="$(realpath "$1")"
COSMOS_RUN="$(realpath "$2")"
PI_JAX_CHECKPOINT="$(realpath "$3")"
OUTPUT_ROOT="$(realpath -m "$4")"
E2E_PYTHON="${5:-${REPOSITORY_ROOT}/.venv-e2e/bin/python}"
COSMOS_OUTPUT="${OUTPUT_ROOT}/cosmos3_nano_diffusers"
PI_OUTPUT="${OUTPUT_ROOT}/pi05_torch"
COSMOS_SOURCE="${COSMOS_RUN}/diffusers"

test -x "${E2E_PYTHON}"
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${COSMOS_SOURCE}/model_index.json" ]]; then
  "${REPOSITORY_ROOT}/scripts/cosmos/export_world_model.sh" "${COSMOS_FRAMEWORK}" "${COSMOS_RUN}"
fi
test -f "${COSMOS_SOURCE}/model_index.json"
if [[ ! -e "${COSMOS_OUTPUT}" ]]; then
  ln -s "${COSMOS_SOURCE}" "${COSMOS_OUTPUT}"
else
  echo "Keeping existing Cosmos export: ${COSMOS_OUTPUT}"
fi

if [[ ! -f "${PI_OUTPUT}/model.safetensors" ]]; then
  "${E2E_PYTHON}" "${REPOSITORY_ROOT}/examples/convert_jax_model_to_pytorch.py" \
    --checkpoint-dir "${PI_JAX_CHECKPOINT}" \
    --config-name pi05_libero_long_subgoal \
    --output-path "${PI_OUTPUT}" \
    --precision bfloat16
else
  echo "Keeping existing pi0.5 conversion: ${PI_OUTPUT}"
fi

echo "COSMOS_CHECKPOINT=${COSMOS_OUTPUT}"
echo "PI_CHECKPOINT=${PI_OUTPUT}"
