#!/usr/bin/env bash
# Install the exact Cosmos framework revision used by this integration.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRAMEWORK_DIR="$(realpath -m "${1:-${REPOSITORY_ROOT}/external/cosmos-framework}")"
CUDA_GROUP="${CUDA_GROUP:-cu128-train}"
COSMOS_FRAMEWORK_COMMIT="${COSMOS_FRAMEWORK_COMMIT:-4155d61d14b14e05a8cafe2bd796d090fcb5f145}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -d "$FRAMEWORK_DIR/.git" ]]; then
  git clone https://github.com/NVIDIA/cosmos-framework.git "$FRAMEWORK_DIR"
fi

if [[ -n "$(git -C "$FRAMEWORK_DIR" status --porcelain)" ]]; then
  echo "ERROR: refusing to change a dirty Cosmos checkout: $FRAMEWORK_DIR" >&2
  exit 1
fi

git -C "$FRAMEWORK_DIR" fetch origin "$COSMOS_FRAMEWORK_COMMIT"
git -C "$FRAMEWORK_DIR" checkout --detach "$COSMOS_FRAMEWORK_COMMIT"
(
  cd "$FRAMEWORK_DIR"
  uv sync --all-extras --group="$CUDA_GROUP"
)

echo "Cosmos framework ready: $FRAMEWORK_DIR @ $COSMOS_FRAMEWORK_COMMIT ($CUDA_GROUP)"
