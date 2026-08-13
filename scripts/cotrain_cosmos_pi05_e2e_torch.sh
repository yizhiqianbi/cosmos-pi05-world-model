#!/usr/bin/env bash
# Launch true differentiable Cosmos3-Nano -> subgoal image -> pi0.5 co-training.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
E2E_PYTHON="${E2E_PYTHON:-${REPOSITORY_ROOT}/.venv-e2e/bin/python}"
: "${COSMOS_CHECKPOINT:?Set COSMOS_CHECKPOINT to the exported Diffusers directory}"
: "${PI_CHECKPOINT:?Set PI_CHECKPOINT to the converted pi0.5 Torch directory}"

LEROBOT_ROOT="${LEROBOT_ROOT:-/public/interns/hubin/world_models/models/libero-data/lerobot-pi05/hubin/libero_long_subgoal}"
COSMOS_DATASET_ROOT="${COSMOS_DATASET_ROOT:-/public/interns/hubin/world_models/models/libero-data/libero_long_full/high_level/cosmos}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPOSITORY_ROOT}/checkpoints/torch_e2e_cosmos_pi05_5ep}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-5}"
COSMOS_TRAINABLE_BLOCKS="${COSMOS_TRAINABLE_BLOCKS:-4}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

"${E2E_PYTHON}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE}" \
  "${REPOSITORY_ROOT}/scripts/cotrain/train_torch_end_to_end.py" \
  --cosmos-checkpoint "${COSMOS_CHECKPOINT}" \
  --pi-checkpoint "${PI_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --lerobot-root "${LEROBOT_ROOT}" \
  --cosmos-dataset-root "${COSMOS_DATASET_ROOT}" \
  --epochs "${EPOCHS}" \
  --local-batch-size "${LOCAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --cosmos-trainable-blocks "${COSMOS_TRAINABLE_BLOCKS}" \
  "$@"
