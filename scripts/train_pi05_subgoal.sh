#!/usr/bin/env bash
# Standalone pi0.5 training on oracle stage-terminal Subgoal Images.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero_long_subgoal}"
REPO_ID="${REPO_ID:-hubin/libero_long_subgoal}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXP_NAME="${EXP_NAME:-pi05_libero_long_subgoal_5ep}"
DATA_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
INFO_PATH="${DATA_HOME}/${REPO_ID}/meta/info.json"
SKIP_NORM_STATS="${SKIP_NORM_STATS:-false}"
RESUME="${RESUME:-false}"

if [[ ! -f "${INFO_PATH}" ]]; then
  echo "Missing LeRobot metadata: ${INFO_PATH}" >&2
  echo "Run examples/libero/convert_long_hdf5_to_lerobot.py first." >&2
  exit 2
fi
if [[ "${REPO_ID}" != "hubin/libero_long_subgoal" ]]; then
  echo "CONFIG_NAME=${CONFIG_NAME} is bound to repo_id=hubin/libero_long_subgoal, got ${REPO_ID}" >&2
  exit 2
fi

read -r TOTAL_FRAMES STEPS STEPS_PER_EPOCH < <(
  uv run python - "${INFO_PATH}" "${EPOCHS}" "${BATCH_SIZE}" <<'PY'
import json
import math
import pathlib
import sys

info = json.loads(pathlib.Path(sys.argv[1]).read_text())
frames = int(info["total_frames"])
epochs = int(sys.argv[2])
batch = int(sys.argv[3])
per_epoch = math.ceil(frames / batch)
print(frames, per_epoch * epochs, per_epoch)
PY
)

echo "Standalone pi0.5 subgoal training: frames=${TOTAL_FRAMES}, epochs=${EPOCHS}, batch=${BATCH_SIZE}, steps=${STEPS}"
if [[ "${SKIP_NORM_STATS}" != "true" ]]; then
  uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi

TRAIN_ARGS=(
  "${CONFIG_NAME}"
  --exp-name "${EXP_NAME}"
  --batch-size "${BATCH_SIZE}"
  --num-train-steps "${STEPS}"
  --save-interval "${STEPS_PER_EPOCH}"
  --keep-period "${STEPS_PER_EPOCH}"
)
if [[ "${RESUME}" == "true" ]]; then
  TRAIN_ARGS+=(--resume)
else
  TRAIN_ARGS+=(--overwrite)
fi
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  uv run scripts/train.py "${TRAIN_ARGS[@]}"
