#!/usr/bin/env bash
# Convert Cosmos3-Nano to DCP (once), then run deterministic first-frame I2V SFT.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 FRAMEWORK_DIR MATERIALIZED_COSMOS_DIR BASE_COSMOS_SNAPSHOT WAN_VAE OUTPUT_ROOT" >&2
  exit 2
fi

FRAMEWORK_DIR="$(realpath "$1")"
DATASET_PATH="$(realpath "$2")"
BASE_SNAPSHOT="$(realpath "$3")"
WAN_VAE_PATH="$(realpath "$4")"
OUTPUT_ROOT="$(realpath -m "$5")"
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-$OUTPUT_ROOT/base_dcp/Cosmos3-Nano}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-50012}"
MAX_ITER="${MAX_ITER:-500}"
SAVE_ITER="${SAVE_ITER:-100}"
JOB_NAME="${JOB_NAME:-cosmos_pi05_nano_i2v}"
COMPILE_ENABLED="${COMPILE_ENABLED:-true}"
EMA_ENABLED="${EMA_ENABLED:-true}"
MAX_SEQUENCE_LENGTH="${MAX_SEQUENCE_LENGTH:-45056}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COSMOS_PYTHON="$FRAMEWORK_DIR/.venv/bin/python"
COSMOS_TORCHRUN="$FRAMEWORK_DIR/.venv/bin/torchrun"

test -f "$FRAMEWORK_DIR/examples/toml/sft_config/vision_sft_nano.toml"
test -x "$COSMOS_PYTHON"
test -x "$COSMOS_TORCHRUN"
test -f "$DATASET_PATH/train/video_dataset_file.jsonl"
test -d "$BASE_SNAPSHOT"
test -f "$WAN_VAE_PATH"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$BASE_CHECKPOINT_PATH")"
uv run --project "$REPO_ROOT" python "$REPO_ROOT/scripts/cosmos/validate_dataset.py" "$DATASET_PATH"

if [[ ! -f "$BASE_CHECKPOINT_PATH/checkpoint.json" || ! -d "$BASE_CHECKPOINT_PATH/model" ]]; then
  echo ">>> converting local Cosmos3-Nano snapshot to DCP"
  (
    cd "$FRAMEWORK_DIR"
    PYTHONPATH=. "$COSMOS_PYTHON" -m cosmos_framework.scripts.convert_model_to_dcp \
      -o "$BASE_CHECKPOINT_PATH" \
      --checkpoint-path "$BASE_SNAPSHOT"
  )
fi

TAIL_OVERRIDES=(
  "job.name=$JOB_NAME"
  "trainer.max_iter=$MAX_ITER"
  "checkpoint.save_iter=$SAVE_ITER"
  "model.config.compile.enabled=$COMPILE_ENABLED"
  "model.config.ema.enabled=$EMA_ENABLED"
  "model.config.max_num_tokens_after_packing=$MAX_SEQUENCE_LENGTH"
  "dataloader_train.max_sequence_length=$MAX_SEQUENCE_LENGTH"
  "dataloader_train.dataloader.datasets.video.dataset.num_video_frames=17"
  "dataloader_train.dataloader.datasets.video.dataset.temporal_interval_mode=entire_chunk"
  "dataloader_train.dataloader.datasets.video.dataset.frame_selection_mode=first"
  "dataloader_train.dataloader.datasets.video.dataset.conditioning_config=null"
  "dataloader_train.dataloader.datasets.video.dataset.conditioning_config={1:1.0}"
  "dataloader_train.dataloader.datasets.video.dataset.cfg_dropout_rate=0.0"
)

echo ">>> launching Cosmos3-Nano I2V SFT on $NPROC_PER_NODE GPUs"
(
  cd "$FRAMEWORK_DIR"
  mkdir -p "$OUTPUT_ROOT/logs"
  DATASET_PATH="$DATASET_PATH" \
  BASE_CHECKPOINT_PATH="$BASE_CHECKPOINT_PATH" \
  WAN_VAE_PATH="$WAN_VAE_PATH" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" \
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}" \
  PYTHONPATH=. \
  "$COSMOS_TORCHRUN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
    -m cosmos_framework.scripts.train \
    --sft-toml=examples/toml/sft_config/vision_sft_nano.toml \
    -- "${TAIL_OVERRIDES[@]}" \
    2>&1 | tee "$OUTPUT_ROOT/logs/$JOB_NAME.log"
)
