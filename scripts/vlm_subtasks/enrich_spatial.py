#!/usr/bin/env python3
"""Add Grounding DINO boxes and optional SAM2 masks to accepted subtask annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.vlm_subtasks.core import write_jsonl
from scripts.vlm_subtasks.spatial import GroundingDinoDetector
from scripts.vlm_subtasks.spatial import Sam2Segmenter
from scripts.vlm_subtasks.spatial import ground_frame


def enrich(args: argparse.Namespace) -> dict:
    records = [json.loads(line) for line in args.annotations.read_text().splitlines() if line.strip()]
    detector = GroundingDinoDetector(args.detector_model, device=args.device)
    segmenter = (
        Sam2Segmenter(args.sam2_config, args.sam2_checkpoint, device=args.device) if args.sam2_checkpoint else None
    )
    enriched = missing = 0
    for record in records:
        if record.get("status") != "accepted":
            continue
        with h5py.File(record["source_hdf5"], "r") as handle:
            frames = handle["data"][record["demo"]]["obs/agentview_rgb"]
            for stage in record["stages"]:
                indices = sorted(
                    {
                        int(stage["start_frame"]),
                        (int(stage["start_frame"]) + int(stage["end_frame"]) - 1) // 2,
                        int(stage["end_frame"]) - 1,
                    }
                )
                stage["spatial_grounding"] = [
                    {
                        "frame": index,
                        **ground_frame(
                            np.asarray(frames[index]),
                            target_object=str(stage.get("target_object", "")),
                            destination=str(stage.get("destination", "")),
                            detector=detector,
                            segmenter=segmenter,
                        ),
                    }
                    for index in indices
                ]
                found = any(frame.get("target", {}).get("status") == "found" for frame in stage["spatial_grounding"])
                if stage.get("target_object") and not found:
                    stage["spatial_status"] = "needs_review"
                    missing += 1
                else:
                    stage["spatial_status"] = "accepted"
                enriched += 1
    write_jsonl(args.output, records)
    summary = {
        "output": str(args.output.resolve()),
        "records": len(records),
        "enriched_stages": enriched,
        "spatial_needs_review": missing,
        "sam2_enabled": segmenter is not None,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    print(json.dumps(enrich(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
