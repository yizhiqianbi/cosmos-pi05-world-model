#!/usr/bin/env python3
"""Annotate LIBERO HDF5 demonstrations with open-source VLM subtask boundaries."""

from __future__ import annotations

import argparse
from itertools import pairwise
import json
import os
from pathlib import Path
import time

import h5py
import imageio.v3 as iio
import numpy as np

from scripts.vlm_subtasks.core import OpenAICompatibleVLM
from scripts.vlm_subtasks.core import candidate_events
from scripts.vlm_subtasks.core import sample_indices
from scripts.vlm_subtasks.core import validate_stages
from scripts.vlm_subtasks.core import write_jsonl
from scripts.vlm_subtasks.temporal import NativeTimeLensClient
from scripts.vlm_subtasks.temporal import apply_refined_boundaries
from scripts.vlm_subtasks.temporal import contextual_grounding_query
from scripts.vlm_subtasks.temporal import global_grounding_indices
from scripts.vlm_subtasks.temporal import local_robot_signals
from scripts.vlm_subtasks.temporal import local_window_indices
from scripts.vlm_subtasks.temporal import select_stable_subgoal_frame
from scripts.vlm_subtasks.temporal import spans_to_boundaries


def task_name(handle: h5py.File, path: Path) -> str:
    info = json.loads(str(handle["data"].attrs.get("problem_info", "{}")))
    return str(info.get("language_instruction") or path.stem.replace("_", " "))


def annotate(args: argparse.Namespace) -> dict:
    started_at = time.perf_counter()
    output = args.output.expanduser().resolve()
    existing: dict[str, dict] = {}
    if output.exists() and args.resume:
        existing = {
            json.loads(line)["trajectory_id"]: json.loads(line)
            for line in output.read_text().splitlines()
            if line.strip()
        }
    client = OpenAICompatibleVLM(
        args.base_url, args.model, api_key=os.getenv("VLM_API_KEY", ""), timeout_s=args.timeout
    )
    temporal_client = OpenAICompatibleVLM(
        args.temporal_base_url,
        args.temporal_model,
        api_key=os.getenv("TEMPORAL_VLM_API_KEY", os.getenv("VLM_API_KEY", "")),
        timeout_s=args.timeout,
    )
    native_temporal_client = NativeTimeLensClient(
        args.temporal_base_url, timeout_s=args.timeout, video_fps=args.temporal_video_fps
    )
    video_cache = (args.video_cache or output.parent / "trajectory_videos").expanduser().resolve()
    video_cache.mkdir(parents=True, exist_ok=True)
    records = list(existing.values())
    all_files = sorted(args.input_dir.expanduser().resolve().glob("*.hdf5"))
    files = all_files[args.task_start : args.task_start + args.max_tasks]
    for task_index, path in enumerate(files, start=args.task_start):
        with h5py.File(path, "r") as handle:
            task = task_name(handle, path)
            demos = sorted(handle["data"], key=lambda name: int(name.rsplit("_", 1)[-1]))[: args.max_demos]
            for demo_name in demos:
                trajectory_id = f"episode_{task_index:03d}_{int(demo_name.rsplit('_', 1)[-1]):03d}"
                if trajectory_id in existing:
                    continue
                group = handle["data"][demo_name]
                frames = group["obs/agentview_rgb"]
                indices = sample_indices(len(frames), args.max_frames)
                events = candidate_events(np.asarray(group["actions"]), np.asarray(group["obs/ee_pos"]), indices)
                allowed = {0, len(frames), *indices}
                record = {
                    "schema_version": "cosmos-pi05.vlm-subtasks.v1",
                    "trajectory_id": trajectory_id,
                    "source_hdf5": str(path),
                    "demo": demo_name,
                    "task": task,
                    "num_frames": len(frames),
                    "sampled_frames": indices,
                    "candidate_events": events,
                    "model": args.model,
                }
                try:
                    semantic_started = time.perf_counter()
                    wrist = group["obs/eye_in_hand_rgb"]
                    paired_views = [
                        np.concatenate([np.asarray(frames[i]), np.asarray(wrist[i])], axis=1) for i in indices
                    ]
                    parsed, raw = client.segment(
                        task=task, num_frames=len(frames), indices=indices, images=paired_views, events=events
                    )
                    record["raw_response"] = raw
                    stages = validate_stages(parsed, num_frames=len(frames), allowed_boundaries=allowed)
                    record["semantic_seconds"] = time.perf_counter() - semantic_started
                    temporal_started = time.perf_counter()
                    if args.temporal_protocol == "native-video":
                        video_path = video_cache / f"{trajectory_id}.mp4"
                        if not video_path.is_file():
                            iio.imwrite(video_path, np.asarray(frames), fps=args.source_fps, codec="libx264")
                        spans = []
                        for stage_index in range(len(stages)):
                            query = contextual_grounding_query(task, stages, stage_index)
                            spans.append(native_temporal_client.ground(str(video_path), query))
                        refined_boundaries = spans_to_boundaries(spans, len(frames), args.source_fps)
                        boundary_details = [
                            {
                                "semantic_rough_boundary": stages[index].end_frame,
                                "global_grounded_boundary": refined_boundaries[index],
                                "boundary_frame": refined_boundaries[index],
                                "previous_span": spans[index],
                                "next_span": spans[index + 1],
                                "temporal_protocol": "native-video",
                                "temporal_confidence": None,
                            }
                            for index in range(len(refined_boundaries))
                        ]
                        record["timelens_spans"] = spans
                    else:
                        refined_boundaries, boundary_details = _ground_with_chat_images(
                            args, temporal_client, task, stages, group, frames, wrist
                        )
                    record["boundary_details"] = boundary_details
                    record["partial_refined_boundaries"] = refined_boundaries
                    stages = apply_refined_boundaries(stages, refined_boundaries, len(frames))
                    record["temporal_seconds"] = time.perf_counter() - temporal_started
                    stage_records = []
                    all_images = np.asarray(frames)
                    for stage in stages:
                        subgoal_frame, stability = select_stable_subgoal_frame(
                            stage.start_frame,
                            stage.end_frame,
                            np.asarray(group["actions"]),
                            np.asarray(group["obs/ee_pos"]),
                            all_images,
                            window=args.stability_window,
                        )
                        stage_records.append(
                            {
                                **stage.__dict__,
                                "subgoal_frame": subgoal_frame,
                                "subgoal_stability": stability,
                                "subgoal_status": (
                                    "accepted"
                                    if stability["stability_score"] <= args.max_subgoal_stability
                                    else "needs_review"
                                ),
                            }
                        )
                    temporal_status = (
                        "accepted"
                        if all(item["subgoal_status"] == "accepted" for item in stage_records)
                        else "needs_review"
                    )
                    record.update(
                        {
                            "status": "accepted" if temporal_status == "accepted" else "needs_review",
                            "temporal_status": temporal_status,
                            "temporal_protocol": args.temporal_protocol,
                            "stages": stage_records,
                            "total_seconds": time.perf_counter() - semantic_started,
                        }
                    )
                except Exception as exc:  # Preserve failures for review/resume; never invent labels.
                    record.update({"status": "needs_review", "error": f"{type(exc).__name__}: {exc}"})
                records.append(record)
                write_jsonl(output, records)
    elapsed = time.perf_counter() - started_at
    completed_now = max(0, len(records) - len(existing))
    durations = [float(item["total_seconds"]) for item in records if "total_seconds" in item]
    summary = {
        "output": str(output),
        "total": len(records),
        "accepted": sum(item["status"] == "accepted" for item in records),
        "needs_review": sum(item["status"] != "accepted" for item in records),
        "elapsed_seconds": elapsed,
        "completed_this_run": completed_now,
        "trajectories_per_hour": 0.0 if elapsed <= 0 else completed_now * 3600 / elapsed,
        "mean_trajectory_seconds": 0.0 if not durations else float(np.mean(durations)),
        "median_trajectory_seconds": 0.0 if not durations else float(np.median(durations)),
        "p95_trajectory_seconds": 0.0 if not durations else float(np.quantile(durations, 0.95)),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _ground_with_chat_images(
    args: argparse.Namespace,
    temporal_client: OpenAICompatibleVLM,
    task: str,
    stages: list,
    group: h5py.Group,
    frames: h5py.Dataset,
    wrist: h5py.Dataset,
) -> tuple[list[int], list[dict]]:
    refined_boundaries = []
    boundary_details = []
    for previous, following in pairwise(stages):
        global_indices = global_grounding_indices(np.asarray(group["actions"]), max_frames=args.temporal_frames)
        global_views = [np.concatenate([np.asarray(frames[i]), np.asarray(wrist[i])], axis=1) for i in global_indices]
        global_signals = local_robot_signals(
            np.asarray(group["actions"]), np.asarray(group["obs/ee_pos"]), global_indices
        )
        grounded_boundary, global_detail, global_raw = temporal_client.refine_boundary(
            task=task,
            previous_subtask=previous.label,
            next_subtask=following.label,
            indices=global_indices,
            images=global_views,
            signals=global_signals,
        )
        local_indices = local_window_indices(
            grounded_boundary, len(frames), radius=args.refine_radius, stride=args.refine_stride
        )
        local_views = [np.concatenate([np.asarray(frames[i]), np.asarray(wrist[i])], axis=1) for i in local_indices]
        signals = local_robot_signals(np.asarray(group["actions"]), np.asarray(group["obs/ee_pos"]), local_indices)
        boundary, detail, refine_raw = temporal_client.refine_boundary(
            task=task,
            previous_subtask=previous.label,
            next_subtask=following.label,
            indices=local_indices,
            images=local_views,
            signals=signals,
        )
        refined_boundaries.append(boundary)
        confidence = min(float(global_detail.get("confidence", 0.0)), float(detail.get("confidence", 0.0)))
        boundary_details.append(
            {
                **detail,
                "semantic_rough_boundary": previous.end_frame,
                "global_grounded_boundary": grounded_boundary,
                "global_response": global_detail,
                "global_raw_response": global_raw,
                "raw_response": refine_raw,
                "temporal_confidence": confidence,
                "coarse_to_fine_delta": abs(boundary - grounded_boundary),
            }
        )
    return refined_boundaries, boundary_details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-32B-Thinking")
    parser.add_argument("--temporal-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--temporal-model", default="TencentARC/TimeLens-8B")
    parser.add_argument("--temporal-protocol", choices=("native-video", "chat-images"), default="native-video")
    parser.add_argument("--video-cache", type=Path)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--temporal-video-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-demos", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temporal-frames", type=int, default=48)
    parser.add_argument("--refine-radius", type=int, default=24)
    parser.add_argument("--refine-stride", type=int, default=1)
    parser.add_argument("--stability-window", type=int, default=10)
    parser.add_argument("--max-subgoal-stability", type=float, default=0.75)
    parser.add_argument("--min-temporal-confidence", type=float, default=0.6)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    print(json.dumps(annotate(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
