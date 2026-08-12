from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def image_to_uint8(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0)
    image = image.astype(np.uint8)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    return np.ascontiguousarray(image)


def split_task_prompt(prompt: str) -> tuple[str, str]:
    lines = [line.strip() for line in str(prompt).splitlines() if line.strip()]
    full = next((line.removeprefix("Full task:").strip() for line in lines if line.startswith("Full task:")), "")
    subtask = next(
        (
            line.removeprefix("Current executable subtask:").strip()
            for line in lines
            if line.startswith("Current executable subtask:")
        ),
        "",
    )
    if not full or not subtask:
        raise ValueError(f"Invalid LIBERO stage prompt: {prompt!r}")
    return full, subtask


def compile_cosmos_prompt(task: str, subtask: str) -> str:
    return json.dumps(
        {
            "task": task,
            "subtask": subtask,
            "embodiment": "Franka Panda",
            "camera": "fixed LIBERO agent-view camera",
            "objective": "simulate the visible execution and end at the completed physical state",
            "constraints": [
                "preserve object identity and scene layout unless changed by the subtask",
                "do not add objects that are absent from the input image",
                "show the terminal state clearly in the final frame",
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def load_cosmos_uuid_index(dataset_root: Path) -> tuple[dict[int, str], dict[str, str]]:
    """Map LeRobot stage episode ids to Cosmos UUIDs and train/val splits.

    Both materializers enumerate task, demonstration and semantic stage in the
    same order. Sorting the structured Cosmos UUIDs therefore reproduces the
    LeRobot episode order without relying on filesystem enumeration order.
    """

    split_by_uuid: dict[str, str] = {}
    for split in ("train", "val"):
        path = dataset_root / split / "video_dataset_file.jsonl"
        for record in read_jsonl(path):
            uuid = str(record["uuid"])
            if uuid in split_by_uuid:
                raise ValueError(f"Duplicate Cosmos UUID across splits: {uuid}")
            split_by_uuid[uuid] = split
    ordered = sorted(split_by_uuid)
    return dict(enumerate(ordered)), split_by_uuid


def resolve_pi_checkpoint(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "params").is_dir():
        return root
    steps = (
        sorted(
            (candidate for candidate in root.iterdir() if candidate.is_dir() and candidate.name.isdigit()),
            key=lambda candidate: int(candidate.name),
        )
        if root.is_dir()
        else []
    )
    if steps and (steps[-1] / "params").is_dir():
        return steps[-1]
    raise FileNotFoundError(f"No openpi checkpoint with params/ found at {root}")
