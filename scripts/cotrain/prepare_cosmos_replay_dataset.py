#!/usr/bin/env python3
"""Build a Cosmos SFT dataset that oversamples pi0.5-hard subgoals."""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
import shutil
import sys

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cotrain.common import read_jsonl
from scripts.cotrain.common import write_jsonl


@dataclasses.dataclass
class Args:
    source_dataset: Path
    scored_replay: Path
    output_dir: Path
    hard_fraction: float = 0.25
    hard_repeat: int = 3
    overwrite: bool = False
    dry_run: bool = False


def prepare(args: Args) -> dict:
    source = args.source_dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not 0.0 < args.hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in (0, 1]")
    if args.hard_repeat < 1:
        raise ValueError("hard_repeat must be at least one")
    train = read_jsonl(source / "train" / "video_dataset_file.jsonl")
    val = read_jsonl(source / "val" / "video_dataset_file.jsonl")
    by_uuid = {str(record["uuid"]): record for record in train}
    scored = [
        record
        for record in read_jsonl(args.scored_replay.expanduser().resolve())
        if record.get("split") == "train" and str(record.get("cosmos_uuid")) in by_uuid
    ]
    if not scored:
        raise ValueError("Scored replay has no samples from the Cosmos train split")
    hard_count = max(1, math.ceil(len(scored) * args.hard_fraction))
    hard = sorted(scored, key=lambda record: float(record["quality_score"]))[:hard_count]
    hard_uuids = [str(record["cosmos_uuid"]) for record in hard]

    augmented = list(train)
    for uuid in hard_uuids:
        source_record = by_uuid[uuid]
        augmented.extend(
            {**source_record, "uuid": f"{uuid}__pi_hard_{copy_index:02d}"} for copy_index in range(1, args.hard_repeat)
        )
    summary = {
        "source": str(source),
        "output": str(output),
        "base_train_samples": len(train),
        "validation_samples": len(val),
        "scored_samples": len(scored),
        "hard_samples": len(hard),
        "hard_repeat": args.hard_repeat,
        "augmented_train_samples": len(augmented),
        "hard_uuids": hard_uuids,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        return summary
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    for split in ("train", "val"):
        split_dir = output / split
        split_dir.mkdir(parents=True)
        os.symlink(source / split / "videos", split_dir / "videos", target_is_directory=True)
    write_jsonl(output / "train" / "video_dataset_file.jsonl", augmented)
    write_jsonl(output / "val" / "video_dataset_file.jsonl", val)
    (output / "cotrain_dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(tyro.cli(Args)), indent=2))
