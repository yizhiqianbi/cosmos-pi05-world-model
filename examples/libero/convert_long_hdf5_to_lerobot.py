#!/usr/bin/env python3
"""Convert all LIBERO-Long HDF5 demos into pi0.5 visual-subgoal data.

Each semantic stage becomes a separate LeRobot episode.  Every frame receives
the terminal agent-view image of that stage as ``subgoal_image``.  Training and
runtime therefore share the same three-image contract; runtime replaces the
demonstration terminal image with a Cosmos3-Nano prediction.
"""

from __future__ import annotations

import dataclasses
import hashlib
from itertools import pairwise
import json
from pathlib import Path
import shutil

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

FPS = 20
MIN_STAGE_LENGTH = 10
GRIPPER_MIN_DWELL = 15

STAGE_PLANS: dict[str, tuple[tuple[str, ...], str]] = {
    "turn on the stove and put the moka pot on it": (
        ("turn on the stove", "pick up the moka pot", "place the moka pot on the stove"),
        "stove_then_object",
    ),
    "put the black bowl in the bottom drawer of the cabinet and close it": (
        (
            "open the bottom drawer and pick up the black bowl",
            "put the black bowl in the bottom drawer",
            "close the bottom drawer",
        ),
        "single_object_then_close",
    ),
    "put the yellow and white mug in the microwave and close it": (
        (
            "open the microwave and pick up the yellow and white mug",
            "put the yellow and white mug in the microwave",
            "close the microwave",
        ),
        "single_object_then_close",
    ),
    "put both moka pots on the stove": (
        (
            "pick up the first moka pot",
            "place the first moka pot on the stove",
            "pick up the second moka pot",
            "place the second moka pot on the stove",
        ),
        "two_objects",
    ),
    "put both the alphabet soup and the cream cheese box in the basket": (
        (
            "pick up the alphabet soup",
            "put the alphabet soup in the basket",
            "pick up the cream cheese box",
            "put the cream cheese box in the basket",
        ),
        "two_objects",
    ),
    "put both the alphabet soup and the tomato sauce in the basket": (
        (
            "pick up the alphabet soup",
            "put the alphabet soup in the basket",
            "pick up the tomato sauce",
            "put the tomato sauce in the basket",
        ),
        "two_objects",
    ),
    "put both the cream cheese box and the butter in the basket": (
        (
            "pick up the cream cheese box",
            "put the cream cheese box in the basket",
            "pick up the butter",
            "put the butter in the basket",
        ),
        "two_objects",
    ),
    "put the white mug on the left plate and put the yellow and white mug on the right plate": (
        (
            "pick up the white mug",
            "put the white mug on the left plate",
            "pick up the yellow and white mug",
            "put the yellow and white mug on the right plate",
        ),
        "two_objects",
    ),
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": (
        (
            "pick up the white mug",
            "put the white mug on the plate",
            "pick up the chocolate pudding",
            "put the chocolate pudding to the right of the plate",
        ),
        "two_objects",
    ),
    "pick up the book and place it in the back compartment of the caddy": (
        ("pick up the book", "place the book in the back compartment of the caddy"),
        "single_object",
    ),
}


@dataclasses.dataclass
class Args:
    input_dir: Path
    repo_id: str = "hubin/libero_long_subgoal"
    overwrite: bool = False
    max_tasks: int | None = None
    max_demos_per_task: int | None = None


def _task_name(handle: h5py.File, path: Path) -> str:
    problem = json.loads(str(handle["data"].attrs["problem_info"]))
    return str(problem.get("language_instruction") or path.stem.removesuffix("_demo").replace("_", " "))


def _gripper_changes(actions: np.ndarray) -> list[int]:
    signs = np.where(np.asarray(actions)[:, -1] >= 0.0, 1, -1)
    signs = signs.copy()
    while len(signs):
        changes = (np.flatnonzero(signs[1:] != signs[:-1]) + 1).astype(int).tolist()
        bounds = [0, *changes, len(signs)]
        short = next(
            (i for i, (left, right) in enumerate(pairwise(bounds)) if right - left < GRIPPER_MIN_DWELL),
            None,
        )
        if short is None or len(bounds) == 2:
            break
        left, right = bounds[short], bounds[short + 1]
        replacement = signs[right] if short == 0 else signs[left - 1]
        signs[left:right] = replacement
    return (np.flatnonzero(signs[1:] != signs[:-1]) + 1).astype(int).tolist()


def _stage_bounds(task: str, actions: np.ndarray) -> tuple[tuple[str, ...], list[int]]:
    stages, mode = STAGE_PLANS[task]
    length = len(actions)
    changes = _gripper_changes(actions)
    if mode == "stove_then_object" and len(changes) >= 3:
        tail = changes[-4:]
        selected = tail[1:3] if len(tail) == 4 else tail[-2:]
    elif mode == "single_object_then_close" and len(changes) >= 2:
        selected = changes[-2:]
    elif mode == "two_objects" and len(changes) >= 3:
        selected = changes[-4:][:3]
    elif mode == "single_object" and changes:
        selected = [changes[-1]]
    else:
        selected = []
    if len(selected) != len(stages) - 1:
        selected = np.linspace(0, length, len(stages) + 1).round().astype(int).tolist()[1:-1]
    selected = [min(length - 1, max(1, int(value))) for value in selected]
    bounds = [0, *selected, length]
    if any(right - left < MIN_STAGE_LENGTH for left, right in pairwise(bounds)):
        bounds = np.linspace(0, length, len(stages) + 1).round().astype(int).tolist()
    return stages, bounds


def _state(group: h5py.Group, index: int) -> np.ndarray:
    return np.concatenate(
        [group["obs/ee_pos"][index], group["obs/ee_ori"][index], group["obs/gripper_states"][index]],
        axis=0,
    ).astype(np.float32)


def _safe_output(repo_id: str, *, overwrite: bool) -> Path:
    if repo_id.startswith("/") or ".." in Path(repo_id).parts:
        raise ValueError("repo_id must be a relative Hugging Face-style id")
    root = Path(HF_LEROBOT_HOME).resolve()
    output = (root / repo_id).resolve()
    if root not in output.parents:
        raise ValueError(f"Refusing output outside HF_LEROBOT_HOME: {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    return output


def convert(args: Args) -> dict:
    files = sorted(args.input_dir.expanduser().resolve().glob("*.hdf5"))
    if args.max_tasks is not None:
        files = files[: args.max_tasks]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found below {args.input_dir}")
    output = _safe_output(args.repo_id, overwrite=args.overwrite)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="franka_panda",
        fps=FPS,
        features={
            "image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "subgoal_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episodes = frames = 0
    source_fingerprint = hashlib.sha256()
    for path in files:
        source_fingerprint.update(path.name.encode())
        with h5py.File(path, "r") as handle:
            task = _task_name(handle, path)
            if task not in STAGE_PLANS:
                raise KeyError(f"No semantic stage plan for {task!r}")
            demos = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))
            if args.max_demos_per_task is not None:
                demos = demos[: args.max_demos_per_task]
            for demo_name in demos:
                group = handle["data"][demo_name]
                actions = np.asarray(group["actions"], dtype=np.float32)
                stages, bounds = _stage_bounds(task, actions)
                for subtask, start, end in zip(stages, bounds[:-1], bounds[1:], strict=True):
                    subgoal = np.asarray(group["obs/agentview_rgb"][max(start, end - 1)], dtype=np.uint8)
                    prompt = f"Full task: {task}\nCurrent executable subtask: {subtask}"
                    for index in range(start, end):
                        dataset.add_frame(
                            {
                                "image": np.asarray(group["obs/agentview_rgb"][index], dtype=np.uint8),
                                "wrist_image": np.asarray(group["obs/eye_in_hand_rgb"][index], dtype=np.uint8),
                                "subgoal_image": subgoal,
                                "state": _state(group, index),
                                "actions": actions[index],
                                "task": prompt,
                            }
                        )
                        frames += 1
                    dataset.save_episode()
                    episodes += 1

    manifest = {
        "schema_version": "cosmos-pi05.libero-long.v1",
        "repo_id": args.repo_id,
        "output": str(output),
        "tasks": len(files),
        "episodes": episodes,
        "frames": frames,
        "source_fingerprint": source_fingerprint.hexdigest(),
        "camera_contract": ["agentview", "wrist", "subgoal"],
        "action_contract": "libero_osc_pose_delta_7d",
        "subgoal_contract": "terminal agent-view image of current semantic stage",
    }
    (output / "cosmos_pi05_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(convert(tyro.cli(Args)), indent=2))
