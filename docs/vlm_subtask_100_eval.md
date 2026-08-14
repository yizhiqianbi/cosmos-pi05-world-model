# VLM Subtask Annotation: 100-Sample Pilot

This pilot annotated 100 LIBERO-Long demonstrations: ten demos from each of the ten `libero_10` tasks. Four independent pipelines used eight H200 GPUs. Each pipeline paired Qwen3.5-9B semantic segmentation with TimeLens-8B native-video grounding.

## Throughput

- Parallel wall time: 655.1 seconds (10.9 minutes)
- Effective throughput: 549.6 trajectories/hour
- Accepted-sample latency: 20.4 seconds mean, 21.9 seconds median, 27.2 seconds P95
- Semantic inference: 14.0 seconds median, 16.3 seconds P95
- Temporal inference: 6.8 seconds median, 11.1 seconds P95

The four shards achieved 159–236 trajectories/hour individually. Unequal task complexity and shard sizes limited linear scaling.

## Quality Gates

The pipeline automatically accepted 61/100 trajectories. It routed 39 to review rather than applying heuristic fallback:

- 13 semantic responses did not exactly cover the trajectory;
- 12 TimeLens stage spans could not form monotonic boundaries;
- 14 refined stages lacked a sufficiently stable terminal subgoal.

Acceptance varied by task from 40% to 90%. The two-stage book-to-caddy task reached 90%; multi-object basket tasks ranged from 40% to 70%. Among accepted annotations, TimeLens moved semantic boundaries by a median of 31 frames and P95 of 91.9 frames, confirming that coarse multi-image timestamps are not suitable as final WAM supervision.

## Interpretation

The throughput is sufficient for full-dataset annotation, but a 61% unattended acceptance rate is not sufficient for fully automatic release. Accepted records can proceed to visual audit and materialization. Review records should be repaired by constrained re-querying or human correction; they must not silently fall back to equal spacing or task-specific boundaries.

Raw merged annotations and machine-readable evaluation files are stored under `artifacts/libero_vlm_100_parallel/merged/` and remain intentionally excluded from Git.
