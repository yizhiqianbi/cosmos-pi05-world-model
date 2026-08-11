#!/usr/bin/env bash
# Export a trained Cosmos DCP checkpoint to HF safetensors and Diffusers layout.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 FRAMEWORK_DIR RUN_DIR [CHECKPOINT_ITER]" >&2
  exit 2
fi

FRAMEWORK_DIR="$(realpath "$1")"
RUN_DIR="$(realpath "$2")"
CHECKPOINT_ITER="${3:-$(tr -d '[:space:]' < "$RUN_DIR/checkpoints/latest_checkpoint.txt")}"
CHECKPOINT_PATH="$RUN_DIR/checkpoints/$CHECKPOINT_ITER"
COSMOS_PYTHON="$FRAMEWORK_DIR/.venv/bin/python"

test -d "$CHECKPOINT_PATH"
test -f "$RUN_DIR/config.yaml"
test -x "$COSMOS_PYTHON"
(
  cd "$FRAMEWORK_DIR"
  PYTHONPATH=. "$COSMOS_PYTHON" -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --config-file "$RUN_DIR/config.yaml" \
    -o "$RUN_DIR/model"
  PYTHONPATH=. "$COSMOS_PYTHON" -m cosmos_framework.scripts.convert_model_to_diffusers \
    --checkpoint-path "$RUN_DIR/model" \
    -o "$RUN_DIR/diffusers"
)

echo "Cosmos world model exported: $RUN_DIR/diffusers"
