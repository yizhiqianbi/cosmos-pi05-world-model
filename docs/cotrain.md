# Cosmos3-Nano × pi0.5 staged and alternating training

The repository supports three explicit training jobs:

1. standalone pi0.5 visual-subgoal training;
2. standalone Cosmos3-Nano stage-transition training;
3. alternating co-training through generated-goal replay and low-level
   executability feedback.

The third job is a real high/low-level data loop, but it is not described as
end-to-end backpropagation. Cosmos training runs in PyTorch/Cosmos Framework,
while pi0.5 training runs in JAX/openpi. Gradients do not cross that framework
boundary.

## 1. Standalone pi0.5 visual-subgoal training

The LeRobot dataset contains the current agent view, wrist view, state, action
chunk and the terminal agent-view image of the current semantic stage. The
third native pi0.5 vision slot consumes that terminal image:

```text
base_0_rgb        = current agent view
left_wrist_0_rgb  = current wrist view
right_wrist_0_rgb = stage-terminal Subgoal Image
```

Run exact five-epoch training with:

```bash
HF_LEROBOT_HOME=/public/interns/hubin/world_models/models/libero-data/lerobot-pi05 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
EPOCHS=5 BATCH_SIZE=256 \
  scripts/train_pi05_subgoal.sh
```

`scripts/train_pi05_libero_long_5ep.sh` remains as a compatibility alias. The
trainer computes normalization statistics unless `SKIP_NORM_STATS=true` is
set, and saves once per dataset epoch.

## 2. Standalone Cosmos3-Nano Subgoal training

Each training example is a semantic-stage video. The first frame is clean I2V
conditioning, the caption specifies the full task and current subtask, and the
remaining frames show execution to the terminal state. Training uses 17 frames
spanning the stage clip.

```bash
COSMOS_FRAMEWORK_DIR=/public/interns/hubin/world_models/src/cosmos-framework \
COSMOS_DATASET=/public/interns/hubin/world_models/models/libero-data/libero_long_full/high_level/cosmos \
COSMOS_BASE_SNAPSHOT=/public/interns/hubin/world_models/models/Cosmos3-Nano \
BASE_CHECKPOINT_PATH=/public/interns/hubin/world_models/models/tau0-cosmos-smoke/base_dcp/Cosmos3-Nano \
WAN_VAE_PATH=/public/interns/hubin/world_models/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
COSMOS_OUTPUT_ROOT=/public/interns/hubin/world_models/models/libero-data/checkpoints/cosmos_subgoal \
COSMOS_GPUS=0,1,2,3 MAX_ITER=500 SAVE_ITER=100 \
  scripts/train_cosmos_subgoal.sh
```

After training, export the selected DCP checkpoint to the Diffusers service
layout:

```bash
scripts/cosmos/export_world_model.sh \
  /public/interns/hubin/world_models/src/cosmos-framework \
  /path/to/output/cosmos3/sft/cosmos3_nano_libero_subgoal
```

## 3. Alternating co-training

One round has five phases:

```text
current observation + subtask
              │
              ▼
       Cosmos generated goal
              │
              ├──────────────┐
              ▼              │
pi0.5 oracle/generated mix   │
              │              │
              ▼              │
pi0.5 action consistency     │
              │              │
              ▼              │
hard transition replay ──────┘
              │
              ▼
       next Cosmos SFT
```

For sampled stage episodes, Cosmos generates an actual deployment-format
subgoal. pi0.5 then trains on a configurable mixture of oracle and generated
goals. The updated pi0.5 predicts a ten-step action chunk for each generated
goal using fixed denoising noise. The score combines:

```text
executable_score = exp(-normalized expert-action MSE)
visual_score     = exp(-4 × terminal-image L1)
quality_score    = 0.7 × executable_score + 0.3 × visual_score
```

Low-quality transitions are oversampled in the next Cosmos SFT phase. This is
alternating policy-aware hard-example mining: the low level learns to consume
the high level's real output distribution, and the high level receives the
low level's executability feedback.

Run a configuration-only check first:

```bash
HF_LEROBOT_HOME=/path/to/lerobot \
COSMOS_FRAMEWORK_DIR=/path/to/cosmos-framework \
COSMOS_BASE_SNAPSHOT=/path/to/Cosmos3-Nano \
COSMOS_BASE_DCP=/path/to/base_dcp/Cosmos3-Nano \
COSMOS_DATASET=/path/to/libero_long/high_level/cosmos \
WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth \
PI05_INITIAL_CHECKPOINT=/path/to/pi05/standalone/latest-step \
COSMOS_INITIAL_RUN_DIR=/path/to/cosmos/standalone/run \
DRY_RUN=true scripts/cotrain_cosmos_pi05.sh
```

Then remove `DRY_RUN=true`:

```bash
HF_LEROBOT_HOME=/path/to/lerobot \
COSMOS_FRAMEWORK_DIR=/path/to/cosmos-framework \
COSMOS_BASE_SNAPSHOT=/path/to/Cosmos3-Nano \
COSMOS_BASE_DCP=/path/to/base_dcp/Cosmos3-Nano \
COSMOS_DATASET=/path/to/libero_long/high_level/cosmos \
WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth \
PI05_INITIAL_CHECKPOINT=/path/to/pi05/standalone/latest-step \
COSMOS_INITIAL_RUN_DIR=/path/to/cosmos/standalone/run \
ROUNDS=2 SAMPLES_PER_ROUND=128 \
PI_STEPS_PER_ROUND=540 PI_BATCH_SIZE=256 \
GENERATED_SUBGOAL_PROBABILITY=0.5 \
COSMOS_ITERS_PER_ROUND=100 \
COSMOS_GPUS=0,1,2,3 PI05_GPUS=4,5,6,7 \
  scripts/cotrain_cosmos_pi05.sh
```

The orchestrator is resumable at phase boundaries with
`RESUME_ROUNDS=true`. Each round keeps its generated images, score JSONL,
pi0.5 checkpoint, hard Cosmos dataset, Cosmos DCP and exported model. It never
modifies the original LeRobot or Cosmos dataset.

## Main files

- `scripts/train_pi05_subgoal.sh`: standalone low-level trainer.
- `scripts/train_cosmos_subgoal.sh`: standalone high-level trainer.
- `scripts/cotrain_cosmos_pi05.sh`: complete alternating orchestrator.
- `scripts/cotrain/generate_subgoal_replay.py`: resident Cosmos replay client.
- `scripts/cotrain/train_pi05_round.py`: generated/oracle mixed pi0.5 update.
- `scripts/cotrain/score_subgoal_replay.py`: policy-aware replay scoring.
- `scripts/cotrain/prepare_cosmos_replay_dataset.py`: hard-example Cosmos dataset.
