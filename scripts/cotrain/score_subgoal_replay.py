#!/usr/bin/env python3
"""Score Cosmos goals with pi0.5 action consistency and visual fidelity."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
from PIL import Image
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cotrain.common import image_to_uint8
from scripts.cotrain.common import read_jsonl
from scripts.cotrain.common import resolve_pi_checkpoint
from scripts.cotrain.common import write_jsonl


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    replay_manifest: Path
    output_path: Path
    repo_id: str = "hubin/libero_long_subgoal"
    config_name: str = "pi05_libero_long_subgoal"
    denoising_steps: int = 10
    action_weight: float = 0.7
    visual_weight: float = 0.3
    seed: int = 0
    max_samples: int | None = None
    dry_run: bool = False


def score(args: Args) -> dict:
    if args.denoising_steps <= 0:
        raise ValueError("denoising_steps must be positive")
    if args.action_weight < 0.0 or args.visual_weight < 0.0 or args.action_weight + args.visual_weight <= 0.0:
        raise ValueError("score weights must be non-negative and not both zero")
    checkpoint = resolve_pi_checkpoint(args.checkpoint_dir)
    replay = read_jsonl(args.replay_manifest.expanduser().resolve())
    if args.max_samples is not None:
        replay = replay[: args.max_samples]
    if not replay:
        raise ValueError("Replay manifest contains no samples")
    for record in replay:
        if not Path(record["generated_subgoal"]).is_file():
            raise FileNotFoundError(record["generated_subgoal"])
    if args.dry_run:
        return {
            "dry_run": True,
            "checkpoint": str(checkpoint),
            "samples": len(replay),
            "output": str(args.output_path.expanduser().resolve()),
        }

    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as training_config

    config = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        sample_kwargs={"num_steps": args.denoising_steps},
    )
    metadata = LeRobotDatasetMetadata(args.repo_id)
    horizon = config.model.action_horizon
    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps={"actions": [step / metadata.fps for step in range(horizon)]},
    )
    asset_id = config.data.create(config.assets_dirs, config.model).asset_id
    if asset_id is None:
        raise ValueError("pi0.5 config has no normalization asset id")
    stats = normalize.load(checkpoint / "assets" / asset_id)["actions"]
    action_scale = np.maximum(np.asarray(stats.q99) - np.asarray(stats.q01), 1e-3)
    action_weight = args.action_weight / (args.action_weight + args.visual_weight)
    visual_weight = 1.0 - action_weight

    scored = []
    for record in replay:
        dataset_index = int(record["dataset_index"])
        row = dataset[dataset_index]
        generated = np.asarray(Image.open(record["generated_subgoal"]).convert("RGB"), dtype=np.uint8)
        observation = {
            "observation/image": image_to_uint8(row["image"]),
            "observation/wrist_image": image_to_uint8(row["wrist_image"]),
            "observation/subgoal_image": generated,
            "observation/state": np.asarray(row["state"], dtype=np.float32),
            "prompt": str(row["task"]),
        }
        noise = (
            np.random.default_rng(args.seed + int(record["episode_index"]))
            .standard_normal((horizon, config.model.action_dim))
            .astype(np.float32)
        )
        prediction = np.asarray(policy.infer(observation, noise=noise)["actions"], dtype=np.float32)
        expert = np.asarray(row["actions"], dtype=np.float32)[..., :7]
        prediction = prediction[: len(expert), :7]
        normalized_error = (prediction - expert) / action_scale
        action_mse = float(np.mean(np.square(normalized_error)))
        executable_score = math.exp(-min(action_mse, 50.0))

        oracle = image_to_uint8(row["subgoal_image"])
        if generated.shape[:2] != oracle.shape[:2]:
            generated_for_score = np.asarray(
                Image.fromarray(generated).resize((oracle.shape[1], oracle.shape[0]), Image.Resampling.LANCZOS)
            )
        else:
            generated_for_score = generated
        visual_l1 = float(np.mean(np.abs(generated_for_score.astype(np.float32) - oracle)) / 255.0)
        visual_score = math.exp(-4.0 * visual_l1)
        quality = action_weight * executable_score + visual_weight * visual_score
        scored.append(
            {
                **record,
                "pi_checkpoint": str(checkpoint),
                "action_mse_normalized": action_mse,
                "executable_score": executable_score,
                "visual_l1": visual_l1,
                "visual_score": visual_score,
                "quality_score": quality,
                "predicted_actions": prediction.tolist(),
                "expert_actions": expert.tolist(),
            }
        )

    write_jsonl(args.output_path.expanduser().resolve(), scored)
    qualities = np.asarray([record["quality_score"] for record in scored])
    return {
        "dry_run": False,
        "checkpoint": str(checkpoint),
        "samples": len(scored),
        "quality_mean": float(qualities.mean()),
        "quality_min": float(qualities.min()),
        "quality_max": float(qualities.max()),
        "output": str(args.output_path.expanduser().resolve()),
    }


if __name__ == "__main__":
    print(json.dumps(score(tyro.cli(Args)), indent=2))
