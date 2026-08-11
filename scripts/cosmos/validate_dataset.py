#!/usr/bin/env python3
"""Validate a materialized Cosmos3 video SFT split before an expensive launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def probe_video(path: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = (int(part) for part in stream["avg_frame_rate"].split("/"))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": numerator / denominator,
    }


def validate_split(split_dir: Path) -> int:
    metadata_path = split_dir / "video_dataset_file.jsonl"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Cosmos metadata: {metadata_path}")
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Cosmos metadata is empty: {metadata_path}")

    uuids: set[str] = set()
    for line_number, record in enumerate(records, 1):
        required = {"uuid", "duration", "width", "height", "vision_path", "t2w_windows"}
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"{metadata_path}:{line_number}: missing keys {missing}")
        if record["uuid"] in uuids:
            raise ValueError(f"{metadata_path}:{line_number}: duplicate uuid {record['uuid']!r}")
        uuids.add(record["uuid"])
        video_path = Path(record["vision_path"])
        if not video_path.is_absolute():
            video_path = split_dir / video_path
        if not video_path.is_file():
            raise FileNotFoundError(f"{metadata_path}:{line_number}: missing video {video_path}")
        info = probe_video(video_path)
        if info["frames"] < 61:
            raise ValueError(
                f"{video_path}: {info['frames']} frames; official Cosmos3 loader filters windows below 61 frames"
            )
        if (info["width"], info["height"]) != (int(record["width"]), int(record["height"])):
            raise ValueError(f"{video_path}: probed dimensions disagree with JSONL")
        for window in record["t2w_windows"]:
            if int(window["end_frame"]) - int(window["start_frame"]) + 1 < 61:
                raise ValueError(f"{video_path}: a t2w window contains fewer than 61 frames")
            if not window.get("caption_json") and not str(window.get("caption", "")).strip():
                raise ValueError(f"{video_path}: a t2w window has no caption")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Materialized cosmos root, containing train/video_dataset_file.jsonl")
    parser.add_argument("--split", action="append", choices=["train", "val"], help="Default: validate train")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    total = 0
    for split in args.split or ["train"]:
        split_dir = root / split
        count = validate_split(split_dir)
        total += count
        print(f"validated split={split} samples={count} path={split_dir}")
    print(f"Cosmos3 dataset valid: samples={total}")


if __name__ == "__main__":
    main()
