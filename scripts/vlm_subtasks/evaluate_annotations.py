#!/usr/bin/env python3
"""Aggregate sharded VLM subtask annotations and report quality/throughput metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.vlm_subtasks.core import write_jsonl


def percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.quantile(values, q))


def failure_reason(item: dict) -> str:
    error = str(item.get("error", ""))
    if "TimeLens spans do not yield ordered stages" in error:
        return "timelens_non_monotonic_spans"
    if "stages must cover the complete trajectory" in error:
        return "semantic_incomplete_coverage"
    if "stages must" in error or "stage " in error:
        return "semantic_schema_validation"
    if error:
        return error.split(":", 1)[0]
    if any(stage.get("subgoal_status") == "needs_review" for stage in item.get("stages", [])):
        return "unstable_subgoal"
    return "other_review"


def evaluate(inputs: list[Path], output_dir: Path) -> dict:
    records = []
    shard_summaries = []
    for path in inputs:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        summary_path = path.with_suffix(".summary.json")
        if summary_path.is_file():
            shard_summaries.append(json.loads(summary_path.read_text()))
    records.sort(key=lambda item: item["trajectory_id"])
    if len({item["trajectory_id"] for item in records}) != len(records):
        raise ValueError("duplicate trajectory ids across shards")
    accepted = [item for item in records if item.get("status") == "accepted"]
    durations = [float(item["total_seconds"]) for item in accepted if "total_seconds" in item]
    semantic = [float(item["semantic_seconds"]) for item in records if "semantic_seconds" in item]
    temporal = [float(item["temporal_seconds"]) for item in accepted if "temporal_seconds" in item]
    boundary_shifts = [
        abs(int(boundary["global_grounded_boundary"]) - int(boundary["semantic_rough_boundary"]))
        for item in accepted
        for boundary in item.get("boundary_details", [])
    ]
    errors = Counter(failure_reason(item) for item in records if item.get("status") != "accepted")
    per_task: dict[str, dict] = defaultdict(lambda: {"total": 0, "accepted": 0, "needs_review": 0, "stages": []})
    for item in records:
        task = str(item["task"])
        per_task[task]["total"] += 1
        per_task[task]["accepted" if item.get("status") == "accepted" else "needs_review"] += 1
        if item.get("stages"):
            per_task[task]["stages"].append(len(item["stages"]))
    per_task_report = {
        task: {
            **{key: value for key, value in stats.items() if key != "stages"},
            "acceptance_rate": stats["accepted"] / stats["total"],
            "mean_stages": None if not stats["stages"] else float(np.mean(stats["stages"])),
        }
        for task, stats in per_task.items()
    }
    wall = max((float(item["elapsed_seconds"]) for item in shard_summaries), default=0.0)
    report = {
        "samples": len(records),
        "accepted": len(accepted),
        "needs_review": len(records) - len(accepted),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "parallel_shards": len(inputs),
        "parallel_wall_seconds": wall,
        "effective_trajectories_per_hour": 0.0 if wall <= 0 else len(records) * 3600 / wall,
        "accepted_latency_seconds": {
            "mean": None if not durations else float(np.mean(durations)),
            "median": percentile(durations, 0.5),
            "p95": percentile(durations, 0.95),
        },
        "semantic_seconds": {"median": percentile(semantic, 0.5), "p95": percentile(semantic, 0.95)},
        "temporal_seconds": {"median": percentile(temporal, 0.5), "p95": percentile(temporal, 0.95)},
        "accepted_boundary_shift_frames": {
            "median": percentile(boundary_shifts, 0.5),
            "p95": percentile(boundary_shifts, 0.95),
            "max": None if not boundary_shifts else max(boundary_shifts),
        },
        "failure_types": dict(errors),
        "per_task": per_task_report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "annotations.jsonl", records)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = [
        "# LIBERO VLM 100-Sample Evaluation",
        "",
        f"- Samples: {report['samples']}",
        f"- Accepted: {report['accepted']} ({report['acceptance_rate']:.1%})",
        f"- Needs review: {report['needs_review']}",
        f"- Parallel wall time: {wall:.1f}s",
        f"- Effective throughput: {report['effective_trajectories_per_hour']:.1f} trajectories/hour",
        "",
        "| Task | Accepted | Total | Rate | Mean stages |",
        "|---|---:|---:|---:|---:|",
    ]
    rows.extend(
        f"| {task} | {stats['accepted']} | {stats['total']} | {stats['acceptance_rate']:.0%} | {stats['mean_stages'] or 0:.1f} |"
        for task, stats in per_task_report.items()
    )
    (output_dir / "evaluation.md").write_text("\n".join(rows) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.inputs, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
