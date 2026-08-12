#!/usr/bin/env bash
# Alternating co-training for Cosmos3-Nano (high level) and pi0.5 (low level).
#
# This is deliberately not advertised as cross-framework end-to-end
# backpropagation.  Cosmos/PyTorch and pi0.5/JAX exchange generated subgoals,
# action-consistency scores and hard-example replay between optimization
# phases, which is reproducible and executable on the existing stack.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

: "${HF_LEROBOT_HOME:?Set HF_LEROBOT_HOME to the local LeRobot root}"
: "${COSMOS_FRAMEWORK_DIR:?Set COSMOS_FRAMEWORK_DIR to cosmos-framework}"
: "${COSMOS_BASE_SNAPSHOT:?Set COSMOS_BASE_SNAPSHOT to Cosmos3-Nano}"
: "${COSMOS_BASE_DCP:?Set COSMOS_BASE_DCP to a model-only Cosmos DCP directory}"
: "${COSMOS_DATASET:?Set COSMOS_DATASET to the original LIBERO Cosmos dataset}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to Wan2.2_VAE.pth}"
: "${PI05_INITIAL_CHECKPOINT:?Set PI05_INITIAL_CHECKPOINT to the standalone pi0.5 checkpoint}"
: "${COSMOS_INITIAL_RUN_DIR:?Set COSMOS_INITIAL_RUN_DIR to the standalone Cosmos run directory}"

ROUNDS="${ROUNDS:-2}"
SAMPLES_PER_ROUND="${SAMPLES_PER_ROUND:-128}"
PI_STEPS_PER_ROUND="${PI_STEPS_PER_ROUND:-540}"
PI_BATCH_SIZE="${PI_BATCH_SIZE:-256}"
GENERATED_SUBGOAL_PROBABILITY="${GENERATED_SUBGOAL_PROBABILITY:-0.5}"
COSMOS_ITERS_PER_ROUND="${COSMOS_ITERS_PER_ROUND:-100}"
HARD_FRACTION="${HARD_FRACTION:-0.25}"
HARD_REPEAT="${HARD_REPEAT:-3}"
COSMOS_GPUS="${COSMOS_GPUS:-0,1,2,3}"
PI05_GPUS="${PI05_GPUS:-4,5,6,7}"
PI05_SCORE_GPU="${PI05_SCORE_GPU:-${PI05_GPUS%%,*}}"
COSMOS_PORT="${COSMOS_PORT:-10091}"
COSMOS_DENOISING_STEPS="${COSMOS_DENOISING_STEPS:-35}"
PI05_DENOISING_STEPS="${PI05_DENOISING_STEPS:-10}"
COSMOS_DEPLOY_FRAMES="${COSMOS_DEPLOY_FRAMES:-5}"
COTRAIN_ROOT="${COTRAIN_ROOT:-outputs/cotrain_cosmos_pi05}"
PI_CHECKPOINT_ROOT="${PI_CHECKPOINT_ROOT:-${COTRAIN_ROOT}/pi_checkpoints}"
RESUME_ROUNDS="${RESUME_ROUNDS:-false}"
DRY_RUN="${DRY_RUN:-false}"
COSMOS_PYTHON="${COSMOS_PYTHON:-${COSMOS_FRAMEWORK_DIR}/.venv/bin/python}"

test -x "${COSMOS_PYTHON}"
test -f "${COSMOS_DATASET}/train/video_dataset_file.jsonl"
test -f "${COSMOS_DATASET}/val/video_dataset_file.jsonl"
test -f "${COSMOS_BASE_DCP}/checkpoint.json"
test -d "${COSMOS_BASE_DCP}/model"
test -f "${WAN_VAE_PATH}"

if [[ "${DRY_RUN}" == "true" ]]; then
  cat <<EOF
Alternating co-training dry run
  rounds:                    ${ROUNDS}
  replay samples/round:      ${SAMPLES_PER_ROUND}
  pi0.5 steps/round:         ${PI_STEPS_PER_ROUND}
  pi0.5 global batch:        ${PI_BATCH_SIZE}
  generated/oracle mix:      ${GENERATED_SUBGOAL_PROBABILITY}
  Cosmos iterations/round:   ${COSMOS_ITERS_PER_ROUND}
  hard fraction/repeat:      ${HARD_FRACTION}/${HARD_REPEAT}
  Cosmos GPUs:               ${COSMOS_GPUS}
  pi0.5 GPUs:                ${PI05_GPUS}
  output:                    ${COTRAIN_ROOT}

Round dataflow:
  exported Cosmos -> generated replay -> pi0.5 mixed-goal update
  -> pi0.5 action-consistency scoring -> Cosmos hard-example SFT -> export
EOF
  exit 0
fi

mkdir -p "${COTRAIN_ROOT}" "${PI_CHECKPOINT_ROOT}"
service_pid=""
stop_cosmos_service() {
  if [[ -n "${service_pid}" ]] && kill -0 "${service_pid}" 2>/dev/null; then
    kill "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
  fi
  service_pid=""
}
trap stop_cosmos_service EXIT INT TERM

latest_cosmos_checkpoint() {
  local run_dir="$1"
  local latest_file="${run_dir}/checkpoints/latest_checkpoint.txt"
  test -f "${latest_file}"
  local checkpoint_name
  checkpoint_name="$(tr -d '[:space:]' <"${latest_file}")"
  test -d "${run_dir}/checkpoints/${checkpoint_name}/model"
  realpath "${run_dir}/checkpoints/${checkpoint_name}"
}

latest_pi_checkpoint() {
  local experiment_dir="$1"
  local latest
  latest="$(find "${experiment_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | awk '/^[0-9]+$/' | sort -n | tail -1)"
  [[ -n "${latest}" ]]
  test -d "${experiment_dir}/${latest}/params"
  realpath "${experiment_dir}/${latest}"
}

ensure_cosmos_export() {
  local run_dir="$1"
  if [[ ! -f "${run_dir}/diffusers/modular_model_index.json" ]]; then
    scripts/cosmos/export_world_model.sh "${COSMOS_FRAMEWORK_DIR}" "${run_dir}"
  fi
}

start_cosmos_service() {
  local model_dir="$1"
  local log_path="$2"
  CUDA_VISIBLE_DEVICES="${COSMOS_GPUS}" "${COSMOS_PYTHON}" scripts/services/cosmos_server.py \
    --model "${model_dir}" \
    --diffusers \
    --diffusers-device-map balanced \
    --diffusers-deploy-num-frames "${COSMOS_DEPLOY_FRAMES}" \
    --output-root "${COTRAIN_ROOT}/cosmos_generations" \
    --port "${COSMOS_PORT}" \
    >"${log_path}" 2>&1 &
  service_pid="$!"
  for _ in $(seq 1 450); do
    if curl --fail --silent "http://127.0.0.1:${COSMOS_PORT}/health" >/dev/null; then
      return
    fi
    if ! kill -0 "${service_pid}" 2>/dev/null; then
      tail -n 100 "${log_path}" >&2
      return 1
    fi
    sleep 2
  done
  echo "Cosmos service did not become healthy" >&2
  return 1
}

current_pi="$(realpath "${PI05_INITIAL_CHECKPOINT}")"
current_cosmos_run="$(realpath "${COSMOS_INITIAL_RUN_DIR}")"

for round in $(seq 1 "${ROUNDS}"); do
  round_tag="$(printf '%03d' "${round}")"
  round_dir="${COTRAIN_ROOT}/round_${round_tag}"
  if [[ -e "${round_dir}" && "${RESUME_ROUNDS}" != "true" ]]; then
    echo "Round directory exists; set RESUME_ROUNDS=true to continue: ${round_dir}" >&2
    exit 2
  fi
  mkdir -p "${round_dir}/logs"
  replay_dir="${round_dir}/replay"
  replay_manifest="${replay_dir}/replay.jsonl"
  replay_summary="${replay_dir}/replay_summary.json"
  scored_replay="${round_dir}/scored_replay.jsonl"
  hard_dataset="${round_dir}/cosmos_hard_dataset"
  pi_exp="cotrain_round_${round_tag}"
  pi_exp_dir="${PI_CHECKPOINT_ROOT}/pi05_libero_long_subgoal/${pi_exp}"
  cosmos_output="${round_dir}/cosmos_train"
  cosmos_job="cosmos_pi05_cotrain_round_${round_tag}"
  next_cosmos_run="${cosmos_output}/cosmos3/sft/${cosmos_job}"

  echo ">>> round ${round_tag}: generate Cosmos replay"
  if [[ ! -f "${replay_summary}" ]]; then
    ensure_cosmos_export "${current_cosmos_run}"
    start_cosmos_service "${current_cosmos_run}/diffusers" "${round_dir}/logs/cosmos_service.log"
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" uv run python scripts/cotrain/generate_subgoal_replay.py \
      --endpoint "http://127.0.0.1:${COSMOS_PORT}" \
      --cosmos-dataset "${COSMOS_DATASET}" \
      --output-dir "${replay_dir}" \
      --max-samples "${SAMPLES_PER_ROUND}" \
      --seed "${round}" \
      --denoising-steps "${COSMOS_DENOISING_STEPS}"
    stop_cosmos_service
  fi

  echo ">>> round ${round_tag}: update pi0.5 on oracle/generated goal mixture"
  pi_checkpoint_count=0
  if [[ -d "${pi_exp_dir}" ]]; then
    pi_checkpoint_count="$(find "${pi_exp_dir}" -mindepth 2 -maxdepth 2 -type d -name params | wc -l)"
  fi
  if [[ "${pi_checkpoint_count}" -eq 0 ]]; then
    PI_ROUND_EXTRA_ARGS=()
    if [[ -d "${pi_exp_dir}" ]]; then
      PI_ROUND_EXTRA_ARGS+=(--overwrite)
    fi
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
    CUDA_VISIBLE_DEVICES="${PI05_GPUS}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}" \
      uv run python scripts/cotrain/train_pi05_round.py \
        --base-checkpoint "${current_pi}" \
        --generated-subgoal-dir "${replay_dir}" \
        --generated-subgoal-probability "${GENERATED_SUBGOAL_PROBABILITY}" \
        --exp-name "${pi_exp}" \
        --checkpoint-base-dir "${PI_CHECKPOINT_ROOT}" \
        --num-train-steps "${PI_STEPS_PER_ROUND}" \
        --batch-size "${PI_BATCH_SIZE}" \
        --seed "${round}" \
        "${PI_ROUND_EXTRA_ARGS[@]}" \
        2>&1 | tee "${round_dir}/logs/pi05_train.log"
  fi
  current_pi="$(latest_pi_checkpoint "${pi_exp_dir}")"

  echo ">>> round ${round_tag}: score Cosmos goals with updated pi0.5"
  if [[ ! -f "${scored_replay}" ]]; then
    HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
    CUDA_VISIBLE_DEVICES="${PI05_SCORE_GPU}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
      uv run python scripts/cotrain/score_subgoal_replay.py \
        --checkpoint-dir "${current_pi}" \
        --replay-manifest "${replay_manifest}" \
        --output-path "${scored_replay}" \
        --denoising-steps "${PI05_DENOISING_STEPS}" \
        --seed "${round}" \
        2>&1 | tee "${round_dir}/logs/pi05_score.log"
  fi

  echo ">>> round ${round_tag}: build pi0.5-hard Cosmos replay dataset"
  if [[ ! -f "${hard_dataset}/cotrain_dataset.json" ]]; then
    uv run python scripts/cotrain/prepare_cosmos_replay_dataset.py \
      --source-dataset "${COSMOS_DATASET}" \
      --scored-replay "${scored_replay}" \
      --output-dir "${hard_dataset}" \
      --hard-fraction "${HARD_FRACTION}" \
      --hard-repeat "${HARD_REPEAT}"
  fi

  echo ">>> round ${round_tag}: update Cosmos on hard stage transitions"
  if [[ ! -f "${next_cosmos_run}/checkpoints/latest_checkpoint.txt" ]]; then
    previous_dcp="$(latest_cosmos_checkpoint "${current_cosmos_run}")"
    bootstrap_dcp="${round_dir}/cosmos_bootstrap_dcp"
    if [[ ! -d "${bootstrap_dcp}" ]]; then
      mkdir -p "${bootstrap_dcp}"
      cp "${COSMOS_BASE_DCP}/checkpoint.json" "${bootstrap_dcp}/checkpoint.json"
      ln -s "${previous_dcp}/model" "${bootstrap_dcp}/model"
    fi
    IFS=',' read -r -a cosmos_gpu_array <<<"${COSMOS_GPUS}"
    CUDA_VISIBLE_DEVICES="${COSMOS_GPUS}" \
    BASE_CHECKPOINT_PATH="${bootstrap_dcp}" \
    NPROC_PER_NODE="${#cosmos_gpu_array[@]}" \
    JOB_NAME="${cosmos_job}" \
    MAX_ITER="${COSMOS_ITERS_PER_ROUND}" \
    SAVE_ITER="${COSMOS_ITERS_PER_ROUND}" \
    COMPILE_ENABLED=false \
    EMA_ENABLED=false \
      scripts/cosmos/train_world_model.sh \
        "${COSMOS_FRAMEWORK_DIR}" \
        "${hard_dataset}" \
        "${COSMOS_BASE_SNAPSHOT}" \
        "${WAN_VAE_PATH}" \
        "${cosmos_output}"
  fi
  ensure_cosmos_export "${next_cosmos_run}"
  current_cosmos_run="$(realpath "${next_cosmos_run}")"
  echo ">>> round ${round_tag} complete: pi0.5=${current_pi} Cosmos=${current_cosmos_run}/diffusers"
done

echo "Alternating co-training complete."
echo "Final pi0.5 checkpoint: ${current_pi}"
echo "Final Cosmos model: ${current_cosmos_run}/diffusers"
