# Cosmos3-Nano × π0.5 architecture

## Design decision

The low-level Tau-0 implementation has been removed from the new runtime.
π0.5 is loaded, trained, normalized, and served by upstream openpi. The new
code only adds a LIBERO visual-goal transform and a hierarchical wrapper.

| Concern | Previous implementation | This repository |
|---|---|---|
| Low-level VLA | Tau-0/Qwen VLA | Official openpi π0.5 flow matching |
| Action horizon | 30 | 10, with 5-step receding-horizon execution |
| Visual inputs | Agent + wrist; optional custom subgoal slot | π0.5 native three-image input |
| Subgoal slot | Checkpoint-specific camera key | `right_wrist_0_rgb` |
| State | 8-D EEF pose and gripper | Same 8-D state, padded by openpi |
| Action | 7-D LIBERO OSC delta | Same 7-D action, padded/unpadded by openpi |
| Normalization | Tau-specific stats | openpi quantile normalization assets |
| Serving | Custom MessagePack HTTP | Standard openpi websocket policy |

## Inference sequence

For sequence `t`:

1. The evaluator sends agent view, wrist view, 8-D state, full task,
   `episode_id`, and `sequence_id`.
2. Qwen generates one executable proposal plus updated memory and token-level
   confidence statistics.
3. Adaptive TTC routes low-confidence proposals to beam search. Beam search
   repeats Proposal → Cosmos prediction → Value scoring and keeps global Top-B
   branches. Reflection commits one immediate executable subtask.
4. Even on the Fast path, Cosmos is called once for the committed immediate
   subtask. A missing or invalid subgoal is a hard error, not a text-only
   fallback.
5. π0.5 receives:

   ```text
   base_0_rgb        = current agent view
   left_wrist_0_rgb  = current wrist view
   right_wrist_0_rgb = Cosmos subgoal image
   state             = xyz + axis-angle + two gripper positions
   prompt            = full task + current executable subtask
   ```

6. π0.5 performs the configured flow-matching denoising steps and returns a
   `10 × 7` action chunk. The evaluator executes five actions and replans.
7. The new observation is fed back into the same episode state. Stale sequence
   results are rejected by the orchestrator.

## Training data

`convert_long_hdf5_to_lerobot.py` converts all ten `libero_10` HDF5 task files.
Gripper transitions provide semantic stage boundaries; equal spacing is used
only when a demonstration lacks a reliable transition contract. Each semantic
stage is a standalone episode so an action target never crosses from one
high-level command into another.

Every stage frame stores the stage-terminal agent view as `subgoal_image`.
This is supervised future-image conditioning, not a copy of the current image.
At deployment Cosmos predicts an image with the same semantic meaning.

The default dataset repository is `hubin/libero_long_subgoal` below
`HF_LEROBOT_HOME`. The corresponding openpi config is
`pi05_libero_long_subgoal`.

## Five-epoch training

The wrapper reads `total_frames` from LeRobot metadata and computes:

```text
steps_per_epoch = ceil(total_frames / global_batch_size)
total_steps     = 5 × steps_per_epoch
```

It first runs openpi normalization-stat computation, then full fine-tuning from
`gs://openpi-assets/checkpoints/pi05_base/params`. The default global batch is
256 and must be divisible by the number of visible JAX devices.

## Service topology

Three services are intentionally isolated:

- Qwen role service: proposal, value, and reflection adapters on port 10090.
- Cosmos service: official Cosmos Framework worker behind HTTP on port 10091.
- Hierarchical π0.5 websocket policy on port 8000.

`run_hierarchical_stack.sh` owns these processes and terminates its children on
exit. The π0.5 server requires both external endpoints and will not start with
an empty Cosmos endpoint. `QWEN_GPU`, `COSMOS_GPUS`, and `PI05_GPUS` isolate
CUDA visibility so JAX π0.5 does not reserve memory on the planner GPUs.

Qwen3.5 also has its own `.venv-qwen`: its Transformers 5.5 requirement is
incompatible with openpi's pinned Transformers 4.53 runtime. The Cosmos
Framework and LIBERO simulator remain independently isolated as well.

The LIBERO simulator runs in `.venv-libero`, separate from the Python 3.11
training/server environment. `setup_libero_eval_env.sh` reproduces the
upstream Python 3.8 dependency set, and `run_libero_eval.sh` injects only the
LIBERO and websocket client source trees before launching evaluation.

## Evaluation protocol

`eval_hierarchical.py` uses the official LIBERO task suite and its 50 initial
states per task. A complete run is 10 tasks × 50 episodes. Each episode writes
an individual JSON result containing route, router confidence, selected subtask,
beam diagnostics, subgoal provenance, action range, step count, and success.
The evaluator can resume by skipping completed episode JSON files.

## Current status

The repository structure, transforms, planner integration, services, data
converter, five-epoch launcher, and evaluator are implemented. A new π0.5
subgoal-conditioned Long checkpoint still has to be trained before the full
stack can produce a meaningful success rate; the previous Tau checkpoint is
deliberately incompatible and is never loaded here.
