#!/usr/bin/env bash
# Standalone Cosmos3-Nano first-frame I2V training for terminal subgoals.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

: "${COSMOS_FRAMEWORK_DIR:?Set COSMOS_FRAMEWORK_DIR to the cosmos-framework checkout}"
: "${COSMOS_DATASET:?Set COSMOS_DATASET to the materialized LIBERO stage-video root}"
: "${COSMOS_BASE_SNAPSHOT:?Set COSMOS_BASE_SNAPSHOT to the Cosmos3-Nano snapshot}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to Wan2.2_VAE.pth}"
: "${COSMOS_OUTPUT_ROOT:?Set COSMOS_OUTPUT_ROOT for checkpoints and logs}"

COSMOS_GPUS="${COSMOS_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<<"${COSMOS_GPUS}"
export CUDA_VISIBLE_DEVICES="${COSMOS_GPUS}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
export JOB_NAME="${JOB_NAME:-cosmos3_nano_libero_subgoal}"
export MAX_ITER="${MAX_ITER:-500}"
export SAVE_ITER="${SAVE_ITER:-100}"

exec scripts/cosmos/train_world_model.sh \
  "${COSMOS_FRAMEWORK_DIR}" \
  "${COSMOS_DATASET}" \
  "${COSMOS_BASE_SNAPSHOT}" \
  "${WAN_VAE_PATH}" \
  "${COSMOS_OUTPUT_ROOT}"
