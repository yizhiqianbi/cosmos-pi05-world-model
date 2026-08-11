# LIBERO Long A/B inference — 2026-08-12

This is a one-episode engineering smoke test, not a benchmark success rate.
Both runs use `libero_10` task 0, initial state 0, a 520-step limit, ten π0.5
flow-matching denoising steps, and five executed actions per low-level query.

Task: `put both the alphabet soup and the tomato sauce in the basket`.

| Run | Checkpoint and conditioning | Result | Steps | Policy queries | Wall time |
|---|---|---:|---:|---:|---:|
| Native π0.5 | Official `pi05_libero`, agent + wrist images | success | 269 | 54 | 46.36 s |
| Cosmos bootstrap | Same official weights, plus Qwen proposal and Cosmos image in the native third vision slot | failure | 520 | 104 | 189.43 s |

The Cosmos run used a high-level interval of ten low-level queries: 11 real
Qwen/Cosmos decisions and 93 subgoal reuses. Every decision was valid and used
the Fast route; there were no degraded results or service errors. Qwen
consistently proposed `pick up the alphabet soup`.

Visual inspection explains the failure. The un-finetuned Cosmos3-Nano output
largely preserves the current observation instead of showing a completed
`pick up the alphabet soup` state. The official π0.5 LIBERO checkpoint was also
never trained to consume that image as a visual goal. The robot approaches and
eventually knocks the object sideways, but never completes the first subtask,
so the high-level planner does not advance to the tomato sauce.

Artifacts:

- Native trace: `outputs/ab_libero_long/native_task00_ep00/episodes/task_00_episode_00.json`
- Native success video: `outputs/ab_libero_long/native_task00_ep00/videos/task_00_episode_00_success.mp4`
- Cosmos trace: `outputs/ab_libero_long/cosmos_task00_ep00_full_hl10/episodes/task_00_episode_00.json`
- Cosmos failure video: `outputs/ab_libero_long/cosmos_task00_ep00_full_hl10/videos/task_00_episode_00_failure.mp4`
- A/B contact sheet: `outputs/ab_libero_long/inspection/ab_contact_sheet.jpg`

The next valid experiment is to fine-tune Cosmos on LIBERO stage-terminal
images and train `pi05_libero_long_subgoal` for five epochs. Reporting the
bootstrap failure as the final Cosmos-π0.5 score would be misleading.
