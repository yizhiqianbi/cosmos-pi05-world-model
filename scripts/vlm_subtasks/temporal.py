from __future__ import annotations

from itertools import pairwise
import json
from typing import Any
import urllib.request

import numpy as np

from scripts.vlm_subtasks.core import Stage


class NativeTimeLensClient:
    """Client for the official native-video TimeLens grounding endpoint."""

    def __init__(self, base_url: str, *, timeout_s: float = 300.0, video_fps: float = 10.0) -> None:
        self.url = base_url.rstrip("/") + "/ground"
        self.timeout_s = timeout_s
        self.video_fps = video_fps

    def ground(self, video_path: str, query: str) -> dict[str, Any]:
        body = {"video_path": video_path, "query": query, "fps": self.video_fps}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read())
        start, end = float(result["start_seconds"]), float(result["end_seconds"])
        if end <= start or start < 0:
            raise ValueError(f"invalid TimeLens span: {start}, {end}")
        return {**result, "start_seconds": start, "end_seconds": end}


def contextual_grounding_query(task: str, stages: list[Stage], index: int) -> str:
    stage = stages[index]
    context = [f"During the successful task '{task}', the robot executes '{stage.label}'"]
    if index > 0:
        context.append(f"after completing '{stages[index - 1].label}'")
    if index + 1 < len(stages):
        context.append(f"before beginning '{stages[index + 1].label}'")
    if stage.target_object:
        context.append(f"with the target object '{stage.target_object}'")
    if stage.destination:
        context.append(f"toward the destination '{stage.destination}'")
    context.append(f"The visible completion evidence is: {stage.evidence}")
    return ", ".join(context)


def spans_to_boundaries(spans: list[dict[str, Any]], num_frames: int, source_fps: float) -> list[int]:
    if not spans:
        raise ValueError("TimeLens returned no stage spans")
    clipped = [
        (
            int(np.clip(round(float(span["start_seconds"]) * source_fps), 0, num_frames - 1)),
            int(np.clip(round(float(span["end_seconds"]) * source_fps), 1, num_frames)),
        )
        for span in spans
    ]
    boundaries = [round((clipped[index][1] + clipped[index + 1][0]) / 2) for index in range(len(clipped) - 1)]
    if any(right <= left for left, right in pairwise([0, *boundaries, num_frames])):
        raise ValueError(f"TimeLens spans do not yield ordered stages: {clipped}")
    return boundaries


def local_window_indices(center: int, num_frames: int, *, radius: int = 16, stride: int = 2) -> list[int]:
    if num_frames < 2 or radius < 1 or stride < 1:
        raise ValueError("invalid temporal window configuration")
    left = max(1, center - radius)
    right = min(num_frames - 1, center + radius)
    indices = list(range(left, right + 1, stride))
    if center not in indices and left <= center <= right:
        indices.append(center)
    return sorted(set(indices))


def global_grounding_indices(actions: np.ndarray, max_frames: int = 48) -> list[int]:
    """Cover the full trajectory while retaining frames around gripper events."""
    actions = np.asarray(actions)
    if len(actions) < 2 or max_frames < 4:
        raise ValueError("trajectory must contain at least two frames and max_frames must be at least four")
    uniform = np.linspace(1, len(actions) - 1, min(max_frames, len(actions) - 1)).round().astype(int)
    gripper = np.where(actions[:, -1] >= 0, 1, -1)
    changes = np.flatnonzero(gripper[1:] != gripper[:-1]) + 1
    event_frames = {
        int(np.clip(index + offset, 1, len(actions) - 1)) for index in changes for offset in (-2, -1, 0, 1, 2)
    }
    merged = sorted(set(uniform.tolist()) | event_frames)
    if len(merged) <= max_frames:
        return merged
    mandatory = sorted(event_frames)
    remaining = max(0, max_frames - len(mandatory))
    sampled_uniform = np.linspace(0, len(uniform) - 1, remaining).round().astype(int) if remaining else []
    return sorted(set(mandatory) | {int(uniform[index]) for index in sampled_uniform})


def local_robot_signals(actions: np.ndarray, ee_pos: np.ndarray, indices: list[int]) -> list[dict[str, Any]]:
    actions = np.asarray(actions)
    ee_pos = np.asarray(ee_pos)
    gripper = np.where(actions[:, -1] >= 0, 1, -1)
    speed = np.r_[0.0, np.linalg.norm(np.diff(ee_pos, axis=0), axis=-1)]
    return [
        {
            "frame": index,
            "gripper": "open" if gripper[index] > 0 else "closed",
            "gripper_transition": bool(index > 0 and gripper[index] != gripper[index - 1]),
            "ee_speed": round(float(speed[index]), 6),
            "action_norm": round(float(np.linalg.norm(actions[index, :-1])), 6),
        }
        for index in indices
    ]


def apply_refined_boundaries(stages: list[Stage], boundaries: list[int], num_frames: int) -> list[Stage]:
    if len(boundaries) != len(stages) - 1:
        raise ValueError("one refined boundary is required between each pair of stages")
    bounds = [0, *boundaries, num_frames]
    if any(right <= left for left, right in pairwise(bounds)):
        raise ValueError("refined boundaries must be strictly increasing")
    return [
        Stage(
            stage.label,
            bounds[index],
            bounds[index + 1],
            stage.evidence,
            stage.confidence,
            stage.target_object,
            stage.destination,
        )
        for index, stage in enumerate(stages)
    ]


def select_stable_subgoal_frame(
    start: int,
    end: int,
    actions: np.ndarray,
    ee_pos: np.ndarray,
    images: np.ndarray,
    *,
    window: int = 10,
) -> tuple[int, dict[str, float]]:
    """Select a low-motion terminal frame without crossing the stage boundary."""
    if end <= start or window < 1:
        raise ValueError("invalid stage interval or stability window")
    actions = np.asarray(actions)
    ee_pos = np.asarray(ee_pos)
    images = np.asarray(images, dtype=np.float32)
    left = max(start, end - window)
    candidates = np.arange(left, end, dtype=np.int64)
    speed = np.r_[0.0, np.linalg.norm(np.diff(ee_pos, axis=0), axis=-1)]
    visual = np.r_[0.0, np.mean(np.abs(np.diff(images, axis=0)), axis=(1, 2, 3)) / 255.0]
    action_norm = np.linalg.norm(actions[:, :-1], axis=-1)

    def normalize(values: np.ndarray) -> np.ndarray:
        selected = values[candidates]
        scale = float(np.ptp(selected))
        return np.zeros_like(selected) if scale < 1e-8 else (selected - selected.min()) / scale

    scores = normalize(speed) + normalize(visual) + 0.5 * normalize(action_norm)
    best_offset = int(np.argmin(scores))
    frame = int(candidates[best_offset])
    return frame, {
        "stability_score": float(scores[best_offset]),
        "ee_speed": float(speed[frame]),
        "visual_delta": float(visual[frame]),
        "action_norm": float(action_norm[frame]),
    }
