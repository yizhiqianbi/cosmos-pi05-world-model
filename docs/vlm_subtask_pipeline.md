# VLM-Based LIBERO-Long Subtask Pipeline

This pipeline replaces task-specific gripper heuristics with open-weight VLM annotations while retaining deterministic validation and dataset generation.

## System design

For each demonstration, `annotate.py` uniformly samples up to 24 timesteps and presents paired agent/wrist views to a semantic VLM. Gripper transitions and low-motion points are included only as weak temporal hints. The model returns imperative subtask labels, target objects, destinations, rough intervals, completion evidence, and confidence.

A second temporal-grounding service performs two passes. First, the default `TencentARC/TimeLens-8B` backend examines up to 48 frames spanning the complete trajectory, including neighborhoods around every gripper transition. It independently relocates each transition between adjacent semantic subtasks. Second, it reads every original 20 FPS frame within ±24 frames of that global result and reports previous-completion, next-onset, contact, and release frames. It receives end-effector speed, action magnitude, and gripper state, but visible object-state change remains the primary signal.

The global and frame-dense responses, confidence, semantic-to-grounded shift, and coarse-to-fine delta are all retained. Low temporal confidence produces `temporal_status=needs_review`. For WAM supervision, the pipeline selects a low-motion `subgoal_frame` from the final ten frames inside each refined stage instead of blindly using `end_frame - 1`.

The validator requires boundaries to use supplied frames, cover the complete trajectory, and remain consecutive and non-overlapping. Invalid responses become `needs_review`; the pipeline never silently invents equal-length stages. The JSONL is resumable and retains coarse and refinement responses for auditability.

Spatial enrichment is a separate recoverable stage. `enrich_spatial.py` runs Grounding DINO on stage start, midpoint, and completion frames. With a SAM2 checkpoint it also stores masks as uncompressed COCO-style RLE. Missing target objects produce `spatial_status=needs_review`.

`materialize.py` converts reviewed annotations into two aligned outputs:

- one LeRobot episode per stage, with the terminal agent-view frame as `subgoal_image`;
- one 61-frame, 256×256 Cosmos video per stage, with matching UUIDs such as `episode_002_014_stage_01`.

## Run a local open-source VLM

The production default is `Qwen/Qwen3-VL-32B-Thinking`, selected for its native long-video understanding, spatial
reasoning, and temporal localization. `Qwen/Qwen3.5-35B-A3B` is a useful independent verifier when enough GPU memory
is available. Serve the primary model with an OpenAI-compatible runtime:

```bash
vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --port 8000 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt '{"image": 24}'
```

Serve `TencentARC/TimeLens-8B` on port 8001 using its official Transformers inference stack or an OpenAI-compatible wrapper. Keeping endpoints independent allows semantic and temporal models to use different GPU pools.

The repository includes the official native-video wrapper. Run it from an environment containing `transformers==4.57.1` and `qwen-vl-utils[decord]==0.0.14`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.vlm_subtasks.serve_timelens \
  --model-path /path/to/TimeLens-8B --port 8001
```

Start with a small audit batch:

```bash
uv run python -m scripts.vlm_subtasks.annotate \
  --input-dir /public/interns/hubin/world_models/models/libero-data/libero_10 \
  --output artifacts/vlm_subtasks_smoke.jsonl \
  --max-tasks 1 --max-demos 5 --max-frames 16
```

The temporal defaults are `--temporal-base-url http://127.0.0.1:8001 --temporal-model TencentARC/TimeLens-8B --temporal-frames 48 --refine-radius 24 --refine-stride 1 --stability-window 10 --max-subgoal-stability 0.75 --min-temporal-confidence 0.6`. A stage without a sufficiently stable terminal frame is rejected even when its boundary confidence is high.

For parallel annotation, split the sorted task list with `--task-start` and `--max-tasks`, give each worker its own semantic/temporal service pair, and write separate JSONL files. Merge and evaluate shards with:

```bash
uv run python -m scripts.vlm_subtasks.evaluate_annotations \
  artifacts/run/shard*/annotations.jsonl \
  --output-dir artifacts/run/merged
```

Inspect every `needs_review` record and a stratified sample of accepted records. Corrections should edit `stages`, set `status` to `accepted`, and preserve `source_hdf5`, `demo`, and `trajectory_id`. Manual boundaries may use any valid original frame index; document the visual state-change evidence in the stage record.

Optionally add spatial grounding in an environment containing Transformers and Grounding DINO. Install SAM2 and provide its checkpoint to enable masks:

```bash
uv run python -m scripts.vlm_subtasks.enrich_spatial \
  --annotations artifacts/vlm_subtasks.jsonl \
  --output artifacts/vlm_subtasks_spatial.jsonl \
  --sam2-checkpoint /path/to/sam2.1_hiera_large.pt
```

Materialize only after review:

```bash
HF_LEROBOT_HOME=/public/interns/hubin/world_models/models/libero-data/lerobot-pi05 \
uv run python -m scripts.vlm_subtasks.materialize \
  --annotations artifacts/vlm_subtasks.jsonl \
  --lerobot-repo-id hubin/libero_long_vlm_subtasks \
  --cosmos-output /public/interns/hubin/world_models/models/libero-data/libero_long_vlm/cosmos
```

Use the enriched JSONL and add `--require-spatial` when boxes/masks are mandatory. Cosmos records retain the complete annotation under `subtask_annotation`; standard loaders ignore this extra field, while WAM variants can consume it.

Then run `uv run python scripts/cosmos/validate_dataset.py <cosmos-output> --split train --split val`. Do not use `--allow-partial` for final training data. Recommended quality metrics are boundary agreement on a manually labeled subset, invalid-response rate, stage-length distribution, and downstream subgoal-conditioned policy success.
