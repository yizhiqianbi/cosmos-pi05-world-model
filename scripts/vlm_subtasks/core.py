from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from itertools import pairwise
import json
from pathlib import Path
import re
from typing import Any
import urllib.request

import numpy as np
from PIL import Image

SYSTEM_PROMPT = """You annotate successful LIBERO robot demonstrations. Return one JSON object only, with no reasoning
before it. Split the trajectory into atomic, sequential, physically executable subtasks. Each subtask must end in an
externally observable task-state change: toggling/opening/closing an articulated object, acquiring an object in the
gripper, or placing an object at its destination. Never create approach, reach, move, adjust, release-only, wait, or
reset subtasks; absorb those frames into the semantic stage they support. A grasp and a later placement must be
separate stages. Boundaries must be chosen from the supplied frame indices. Do not invent actions or merge unrelated
object interactions. Intervals must exactly cover [0,num_frames), so the first start is 0 and final end is num_frames.
Schema: {"stages":[{"label":"imperative phrase","start_frame":0,"end_frame":42,
"target_object":"object being manipulated","destination":"target region or empty string",
"evidence":"visible completion cue","confidence":0.0}],"trajectory_summary":"..."}.
Intervals are half-open [start_frame,end_frame), consecutive, non-overlapping, and cover [0,num_frames)."""


@dataclass(frozen=True)
class Stage:
    label: str
    start_frame: int
    end_frame: int
    evidence: str
    confidence: float
    target_object: str = ""
    destination: str = ""


def sample_indices(num_frames: int, max_frames: int) -> list[int]:
    if num_frames < 1 or max_frames < 2:
        raise ValueError("num_frames must be positive and max_frames must be at least 2")
    count = min(num_frames, max_frames)
    return sorted(set(np.linspace(0, num_frames - 1, count).round().astype(int).tolist()))


def candidate_events(actions: np.ndarray, ee_pos: np.ndarray, sampled: list[int]) -> list[dict[str, Any]]:
    """Compute model hints; these are candidates, never authoritative boundaries."""
    gripper = np.where(np.asarray(actions)[:, -1] >= 0, 1, -1)
    changes = set((np.flatnonzero(gripper[1:] != gripper[:-1]) + 1).astype(int).tolist())
    speed = np.linalg.norm(np.diff(np.asarray(ee_pos), axis=0), axis=-1)
    threshold = float(np.quantile(speed, 0.2)) if len(speed) else 0.0
    pauses = set((np.flatnonzero(speed <= threshold) + 1).astype(int).tolist())
    result = []
    for index in sampled:
        tags = []
        if any(abs(index - value) <= 2 for value in changes):
            tags.append("gripper_transition")
        if any(abs(index - value) <= 2 for value in pauses):
            tags.append("low_motion")
        if tags:
            result.append({"frame": index, "signals": tags})
    return result


def encode_image(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("VLM response contains no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


def validate_stages(raw: dict[str, Any], *, num_frames: int, allowed_boundaries: set[int]) -> list[Stage]:
    items = raw.get("stages")
    if not isinstance(items, list) or not items:
        raise ValueError("stages must be a non-empty list")
    stages: list[Stage] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"stage {index} is not an object")
        start, end = int(item["start_frame"]), int(item["end_frame"])
        label = " ".join(str(item.get("label", "")).split())
        if not label:
            raise ValueError(f"stage {index} has an empty label")
        if start not in allowed_boundaries or end not in allowed_boundaries:
            raise ValueError(f"stage {index} boundary is not from the supplied indices: {start}, {end}")
        stages.append(
            Stage(
                label,
                start,
                end,
                str(item.get("evidence", "")),
                float(item.get("confidence", 0.0)),
                str(item.get("target_object", "")),
                str(item.get("destination", "")),
            )
        )
    if stages[0].start_frame != 0 or stages[-1].end_frame != num_frames:
        raise ValueError("stages must cover the complete trajectory")
    for left, right in pairwise(stages):
        if left.end_frame != right.start_frame:
            raise ValueError("stages must be consecutive")
    if any(stage.end_frame <= stage.start_frame for stage in stages):
        raise ValueError("all stages must have positive length")
    if any(not 0.0 <= stage.confidence <= 1.0 for stage in stages):
        raise ValueError("confidence must be in [0, 1]")
    return stages


class OpenAICompatibleVLM:
    """Client for local vLLM/SGLang servers hosting an open-weight VLM."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "", timeout_s: float = 300.0) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model, self.api_key, self.timeout_s = model, api_key, timeout_s

    def _chat(self, system: str, content: list[dict[str, Any]], *, max_tokens: int = 1200) -> tuple[dict, str]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            result = json.loads(response.read())
        raw = str(result["choices"][0]["message"]["content"])
        return parse_json_object(raw), raw

    def segment(
        self, *, task: str, num_frames: int, indices: list[int], images: list[np.ndarray], events: list[dict]
    ) -> tuple[dict, str]:
        content: list[dict[str, Any]] = []
        for index, image in zip(indices, images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": f"FRAME {index}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "task_instruction": task,
                        "num_frames": num_frames,
                        "allowed_start_frames": indices,
                        "allowed_end_frames": [*indices[1:], num_frames],
                        "candidate_events": events,
                    }
                ),
            }
        )
        return self._chat(SYSTEM_PROMPT, content)

    def refine_boundary(
        self,
        *,
        task: str,
        previous_subtask: str,
        next_subtask: str,
        indices: list[int],
        images: list[np.ndarray],
        signals: list[dict[str, Any]],
    ) -> tuple[int, dict[str, Any], str]:
        system = """Locate the transition between two consecutive robot subtasks. Inspect every labeled frame.
Return JSON only: {"boundary_frame": int, "previous_completion_frame": int, "next_onset_frame": int,
"contact_frame": int|null, "release_frame": int|null, "evidence": str, "confidence": number}.
All frame values must be selected from allowed_frames. boundary_frame is the first frame assigned to the next subtask.
Use visible object state change as primary evidence and robot/gripper signals only as supporting evidence."""
        content: list[dict[str, Any]] = []
        for index, image in zip(indices, images, strict=True):
            content.extend(
                [
                    {"type": "text", "text": f"FRAME {index}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "task": task,
                        "previous_subtask": previous_subtask,
                        "next_subtask": next_subtask,
                        "allowed_frames": indices,
                        "robot_signals": signals,
                    }
                ),
            }
        )
        parsed, raw = self._chat(system, content, max_tokens=512)
        boundary = int(parsed["boundary_frame"])
        if boundary not in indices:
            raise ValueError(f"refined boundary {boundary} was not an allowed frame")
        return boundary, parsed, raw


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
