#!/usr/bin/env python3
"""Fine-tune pi0.5 for one alternating round using Cosmos replay goals."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cotrain.common import resolve_pi_checkpoint


@dataclasses.dataclass
class Args:
    base_checkpoint: str
    generated_subgoal_dir: Path
    exp_name: str
    config_name: str = "pi05_libero_long_subgoal"
    num_train_steps: int = 540
    batch_size: int = 256
    generated_subgoal_probability: float = 0.5
    seed: int = 0
    checkpoint_base_dir: Path = Path("checkpoints")
    overwrite: bool = False
    dry_run: bool = False


def _config(args: Args):
    from openpi.training import config as training_config
    from openpi.training import weight_loaders

    if args.num_train_steps <= 0 or args.batch_size <= 0:
        raise ValueError("num_train_steps and batch_size must be positive")
    if not 0.0 <= args.generated_subgoal_probability <= 1.0:
        raise ValueError("generated_subgoal_probability must be in [0, 1]")
    overlay = args.generated_subgoal_dir.expanduser().resolve()
    replay_manifest = overlay / "replay.jsonl"
    if not replay_manifest.is_file():
        raise FileNotFoundError(f"Missing Cosmos replay manifest: {replay_manifest}")

    base_checkpoint = resolve_pi_checkpoint(args.base_checkpoint)
    base = training_config.get_config(args.config_name)
    if not isinstance(base.data, training_config.LeRobotLiberoSubgoalDataConfig):
        raise TypeError(f"{args.config_name} is not a LIBERO visual-subgoal config")
    data = dataclasses.replace(
        base.data,
        generated_subgoal_dir=str(overlay),
        generated_subgoal_probability=args.generated_subgoal_probability,
        generated_subgoal_seed=args.seed,
        # Replay intentionally covers only a sampled subset. Other episodes
        # retain their oracle terminal image.
        generated_subgoal_strict=False,
    )
    save_interval = args.num_train_steps
    return dataclasses.replace(
        base,
        data=data,
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        save_interval=save_interval,
        keep_period=save_interval,
        log_interval=min(50, save_interval),
        seed=args.seed,
        checkpoint_base_dir=str(args.checkpoint_base_dir.expanduser().resolve()),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(base_checkpoint / "params")),
        overwrite=args.overwrite,
        resume=False,
        wandb_enabled=False,
        policy_metadata={
            **(base.policy_metadata or {}),
            "training_stage": "alternating-cotrain",
            "base_checkpoint": str(base_checkpoint),
            "generated_subgoal_dir": str(overlay),
            "generated_subgoal_probability": args.generated_subgoal_probability,
        },
    )


def main(args: Args) -> None:
    config = _config(args)
    summary = {
        "config": config.name,
        "exp_name": config.exp_name,
        "checkpoint_dir": str(config.checkpoint_dir),
        "base_checkpoint": str(config.weight_loader),
        "generated_subgoal_dir": str(args.generated_subgoal_dir.expanduser().resolve()),
        "generated_subgoal_probability": args.generated_subgoal_probability,
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    from scripts import train

    train.main(config)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (config.checkpoint_dir / "cotrain_round.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
