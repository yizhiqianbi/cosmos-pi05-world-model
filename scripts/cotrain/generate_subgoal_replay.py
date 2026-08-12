#!/usr/bin/env python3
"""Generate one Cosmos subgoal image for selected LIBERO stage episodes."""

from __future__ import annotations

import base64
import dataclasses
from io import BytesIO
import json
from pathlib import Path
import random
import sys
from typing import Any
import urllib.error
import urllib.request

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cotrain.common import compile_cosmos_prompt
from scripts.cotrain.common import image_to_uint8
from scripts.cotrain.common import load_cosmos_uuid_index
from scripts.cotrain.common import read_jsonl
from scripts.cotrain.common import split_task_prompt
from scripts.cotrain.common import write_jsonl


@dataclasses.dataclass
class Args:
    endpoint: str
    cosmos_dataset: Path
    output_dir: Path
    repo_id: str = "hubin/libero_long_subgoal"
    split: str = "train"
    max_samples: int = 128
    batch_size: int = 4
    seed: int = 0
    denoising_steps: int = 35
    mode: str = "image"
    num_frames: int = 5
    fps: int = 10
    resolution: int = 256
    timeout_s: float = 1800.0
    dry_run: bool = False


def _png_base64(image: object) -> str:
    buffer = BytesIO()
    Image.fromarray(image_to_uint8(image)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _post(endpoint: str, requests: list[dict[str, Any]], timeout_s: float) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/world/predict",
        data=json.dumps({"model": "cosmos3-nano-cotrain", "profile": "deploy", "requests": requests}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Cosmos endpoint returned HTTP {exc.code}: {detail[:2000]}") from exc
    outcomes = body.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(requests):
        raise RuntimeError(f"Cosmos endpoint returned an invalid batch: {body}")
    return outcomes


def generate(args: Args) -> dict[str, Any]:
    if args.max_samples <= 0 or args.batch_size <= 0:
        raise ValueError("max_samples and batch_size must be positive")
    if args.split not in {"train", "val", "all"}:
        raise ValueError("split must be train, val, or all")
    if args.mode not in {"image", "video"}:
        raise ValueError("mode must be image or video")

    episode_to_uuid, split_by_uuid = load_cosmos_uuid_index(args.cosmos_dataset.resolve())
    dataset = LeRobotDataset(args.repo_id)
    if len(episode_to_uuid) != dataset.num_episodes:
        raise ValueError(f"Cosmos/LeRobot stage count mismatch: {len(episode_to_uuid)} != {dataset.num_episodes}")
    candidates = [
        episode for episode, uuid in episode_to_uuid.items() if args.split == "all" or split_by_uuid[uuid] == args.split
    ]
    random.Random(args.seed).shuffle(candidates)
    selected = sorted(candidates[: args.max_samples])
    manifest_path = args.output_dir / "replay.jsonl"
    completed = (
        {int(record["episode_index"]): record for record in read_jsonl(manifest_path)}
        if manifest_path.is_file()
        else {}
    )

    requests: list[dict[str, Any]] = []
    request_meta: list[dict[str, Any]] = []
    planned = []
    for episode_index in selected:
        if episode_index in completed and Path(completed[episode_index]["generated_subgoal"]).is_file():
            continue
        dataset_index = int(dataset.episode_data_index["from"][episode_index])
        row = dataset[dataset_index]
        task, subtask = split_task_prompt(str(row["task"]))
        uuid = episode_to_uuid[episode_index]
        service_request = {
            "request_id": f"cotrain:{uuid}",
            "initial_image_png_b64": _png_base64(row["image"]),
            "prompt": compile_cosmos_prompt(task, subtask),
            "seed": args.seed + episode_index,
            "mode": args.mode,
            "resolution": args.resolution,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "denoising_steps": args.denoising_steps,
            "generate_audio": False,
        }
        meta = {
            "episode_index": episode_index,
            "dataset_index": dataset_index,
            "cosmos_uuid": uuid,
            "split": split_by_uuid[uuid],
            "task": task,
            "subtask": subtask,
            "prompt": service_request["prompt"],
        }
        if args.dry_run:
            planned.append({**meta, "request": {k: v for k, v in service_request.items() if "b64" not in k}})
            continue
        requests.append(service_request)
        request_meta.append(meta)
        if len(requests) < args.batch_size:
            continue
        _flush(args, requests, request_meta, completed, manifest_path)
        requests, request_meta = [], []
    if requests:
        _flush(args, requests, request_meta, completed, manifest_path)

    if args.dry_run:
        return {"dry_run": True, "selected": len(selected), "pending": len(planned), "examples": planned[:3]}
    summary = {
        "dry_run": False,
        "selected": len(selected),
        "completed": sum(episode in completed for episode in selected),
        "manifest": str(manifest_path.resolve()),
        "overlay_dir": str(args.output_dir.resolve()),
    }
    if summary["completed"] != summary["selected"]:
        raise RuntimeError(f"Replay generation is incomplete: {summary}")
    (args.output_dir / "replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _flush(
    args: Args,
    requests: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    manifest_path: Path,
) -> None:
    outcomes = _post(args.endpoint, requests, args.timeout_s)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for meta, outcome in zip(metadata, outcomes, strict=True):
        if not isinstance(outcome, dict) or not outcome.get("valid", True):
            raise RuntimeError(f"Cosmos failed for {meta['cosmos_uuid']}: {outcome}")
        encoded = outcome.get("terminal_image_png_b64")
        if not encoded:
            raise RuntimeError(f"Cosmos returned no terminal image for {meta['cosmos_uuid']}")
        target = args.output_dir / f"episode_{int(meta['episode_index']):06d}.png"
        target.write_bytes(base64.b64decode(str(encoded), validate=True))
        completed[int(meta["episode_index"])] = {
            **meta,
            "generated_subgoal": str(target.resolve()),
            "latency_ms": outcome.get("latency_ms"),
            "worker_metadata": outcome.get("worker_metadata", {}),
        }
    write_jsonl(manifest_path, [completed[index] for index in sorted(completed)])


if __name__ == "__main__":
    print(json.dumps(generate(tyro.cli(Args)), indent=2, ensure_ascii=False))
