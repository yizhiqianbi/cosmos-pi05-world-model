#!/usr/bin/env python3
"""Materialize reviewed VLM annotations into aligned pi0.5 and Cosmos datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import imageio.v3 as iio
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image

from scripts.vlm_subtasks.core import write_jsonl


def output_indices(start: int, end: int, count: int = 61) -> np.ndarray:
    if start < 0 or end <= start or count < 2:
        raise ValueError("invalid stage interval or output frame count")
    return np.linspace(start, end - 1, count).round().astype(np.int64)


def split_for(trajectory_id: str, validation_percent: int) -> str:
    if not 0 <= validation_percent < 100:
        raise ValueError("validation_percent must be in [0, 100)")
    bucket = int(hashlib.sha256(trajectory_id.encode()).hexdigest()[:8], 16) % 100
    return "val" if bucket < validation_percent else "train"


def resize_video(frames: np.ndarray, size: int = 256) -> np.ndarray:
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR)) for frame in frames]
    )


def cosmos_record(
    uuid: str,
    subtask: str,
    task: str,
    *,
    fps: int = 12,
    frames: int = 61,
    annotation: dict | None = None,
) -> dict:
    duration = frames / fps
    caption = f"Franka executes {subtask} for {task}."
    record = {
        "uuid": uuid,
        "duration": duration,
        "width": 256,
        "height": 256,
        "nb_frames": frames,
        "framerate": fps,
        "vision_path": f"videos/{uuid}.mp4",
        "t2w_windows": [
            {
                "start_frame": 0,
                "end_frame": frames - 1,
                "temporal_interval": 1,
                "caption_json": {
                    "subjects": [{"description": "a Franka Panda robot and LIBERO objects", "action": subtask}],
                    "background_setting": "the fixed LIBERO tabletop workspace",
                    "cinematography": {
                        "camera_motion": "static",
                        "framing": "agent-view camera",
                        "camera_angle": "fixed",
                    },
                    "actions": [{"time": f"0:00-0:{int(duration):02d}", "description": subtask}],
                    "temporal_caption": caption,
                    "resolution": {"H": 256, "W": 256},
                    "aspect_ratio": "16,16",
                    "duration": f"{int(duration)}s",
                    "fps": fps,
                },
                "caption": caption,
            }
        ],
    }
    if annotation is not None:
        record["subtask_annotation"] = annotation
    return record


def _state(group: h5py.Group, index: int) -> np.ndarray:
    return np.concatenate(
        [group["obs/ee_pos"][index], group["obs/ee_ori"][index], group["obs/gripper_states"][index]]
    ).astype(np.float32)


def _make_lerobot(repo_id: str, *, overwrite: bool) -> tuple[LeRobotDataset, Path]:
    root = Path(HF_LEROBOT_HOME).resolve()
    output = (root / repo_id).resolve()
    if root not in output.parents:
        raise ValueError("lerobot repo id escapes HF_LEROBOT_HOME")
    if output.exists():
        if not overwrite:
            raise FileExistsError(output)
        shutil.rmtree(output)
    features = {
        "image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
        "subgoal_image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="franka_panda",
        fps=20,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    ), output


def materialize(args: argparse.Namespace) -> dict:
    annotations = [json.loads(line) for line in args.annotations.read_text().splitlines() if line.strip()]
    rejected = [item["trajectory_id"] for item in annotations if item.get("status") != "accepted"]
    if rejected and not args.allow_partial:
        raise ValueError(f"{len(rejected)} trajectories need review; first: {rejected[0]}")
    accepted = sorted(
        (item for item in annotations if item.get("status") == "accepted"), key=lambda x: x["trajectory_id"]
    )
    if not accepted:
        raise ValueError("no accepted annotations")
    if args.require_spatial:
        missing_spatial = [
            f"{item['trajectory_id']}:{index}"
            for item in accepted
            for index, stage in enumerate(item["stages"])
            if stage.get("spatial_status") != "accepted"
        ]
        if missing_spatial:
            raise ValueError(
                f"{len(missing_spatial)} stages lack accepted spatial grounding; first: {missing_spatial[0]}"
            )
    cosmos_root = args.cosmos_output.expanduser().resolve()
    if cosmos_root.exists():
        if not args.overwrite:
            raise FileExistsError(cosmos_root)
        shutil.rmtree(cosmos_root)
    for split in ("train", "val"):
        (cosmos_root / split / "videos").mkdir(parents=True)
    lerobot, lerobot_root = _make_lerobot(args.lerobot_repo_id, overwrite=args.overwrite)
    manifests: dict[str, list[dict]] = {"train": [], "val": []}
    episodes = frames_written = 0
    for item in accepted:
        with h5py.File(item["source_hdf5"], "r") as handle:
            group = handle["data"][item["demo"]]
            agent = group["obs/agentview_rgb"]
            wrist = group["obs/eye_in_hand_rgb"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            split = split_for(item["trajectory_id"], args.validation_percent)
            for stage_index, stage in enumerate(item["stages"]):
                start, end = int(stage["start_frame"]), int(stage["end_frame"])
                uuid = f"{item['trajectory_id']}_stage_{stage_index:02d}"
                subgoal_frame = int(stage.get("subgoal_frame", end - 1))
                if not start <= subgoal_frame < end:
                    raise ValueError(f"{uuid}: subgoal_frame {subgoal_frame} is outside [{start}, {end})")
                terminal = np.asarray(agent[subgoal_frame], dtype=np.uint8)
                prompt = f"Full task: {item['task']}\nCurrent executable subtask: {stage['label']}"
                for index in range(start, end):
                    lerobot.add_frame(
                        {
                            "image": np.asarray(agent[index]),
                            "wrist_image": np.asarray(wrist[index]),
                            "subgoal_image": terminal,
                            "state": _state(group, index),
                            "actions": actions[index],
                            "task": prompt,
                        }
                    )
                    frames_written += 1
                lerobot.save_episode()
                clip = resize_video(np.asarray(agent)[output_indices(start, end)])
                iio.imwrite(cosmos_root / split / "videos" / f"{uuid}.mp4", clip, fps=12, codec="libx264")
                manifests[split].append(
                    cosmos_record(uuid, stage["label"], item["task"], annotation={**stage, "task": item["task"]})
                )
                episodes += 1
    for split, records in manifests.items():
        write_jsonl(cosmos_root / split / "video_dataset_file.jsonl", records)
    summary = {
        "schema_version": "cosmos-pi05.vlm-materialized.v1",
        "annotations": str(args.annotations.resolve()),
        "lerobot_root": str(lerobot_root),
        "cosmos_root": str(cosmos_root),
        "trajectories": len(accepted),
        "episodes": episodes,
        "frames": frames_written,
        "train_stages": len(manifests["train"]),
        "val_stages": len(manifests["val"]),
        "excluded_needs_review": len(rejected),
    }
    (cosmos_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--lerobot-repo-id", required=True)
    parser.add_argument("--cosmos-output", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-spatial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    print(json.dumps(materialize(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
