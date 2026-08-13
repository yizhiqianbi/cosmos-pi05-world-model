#!/usr/bin/env python3
"""Merge an end-to-end adapter with both frozen base checkpoints."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import shutil

import safetensors.torch
import torch
import tyro

from cosmos_pi05.cotrain.cosmos_diffusers import DifferentiableCosmosI2V
from cosmos_pi05.cotrain.torch_joint import TorchCosmosPi05JointModel
from cosmos_pi05.cotrain.torch_joint import load_trainable_adapter
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as openpi_config


@dataclasses.dataclass(frozen=True)
class Args:
    cosmos_checkpoint: Path
    pi_checkpoint: Path
    adapter_checkpoint: Path
    output_dir: Path
    config_name: str = "pi05_libero_long_subgoal"


def _pi_weights(path: Path) -> Path:
    return path if path.is_file() else path / "model.safetensors"


def export(args: Args) -> None:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    config = openpi_config.get_config(args.config_name)
    pi_config = dataclasses.replace(config.model, dtype="bfloat16", pytorch_compile_mode=None)
    policy = PI0Pytorch(pi_config)
    safetensors.torch.load_model(policy, _pi_weights(args.pi_checkpoint), device="cpu")
    world_model = DifferentiableCosmosI2V.from_pretrained(str(args.cosmos_checkpoint), dtype=torch.bfloat16)
    joint = TorchCosmosPi05JointModel(world_model, policy)
    adapter = (
        args.adapter_checkpoint / "trainable.safetensors"
        if args.adapter_checkpoint.is_dir()
        else args.adapter_checkpoint
    )
    load_trainable_adapter(joint, str(adapter))

    cosmos_output = args.output_dir / "cosmos3_nano"
    pi_output = args.output_dir / "pi05"
    cosmos_output.mkdir(parents=True)
    pi_output.mkdir(parents=True)
    world_model._pipeline_helpers.save_pretrained(cosmos_output)  # noqa: SLF001
    safetensors.torch.save_model(policy, pi_output / "model.safetensors")
    source_assets = args.pi_checkpoint / "assets"
    if source_assets.is_dir():
        shutil.copytree(source_assets, pi_output / "assets")
    (pi_output / "config.json").write_text(
        json.dumps(
            {
                "config_name": args.config_name,
                "action_dim": pi_config.action_dim,
                "action_horizon": pi_config.action_horizon,
                "dtype": pi_config.dtype,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "joint_manifest.json").write_text(
        json.dumps(
            {
                "cosmos": "cosmos3_nano",
                "pi05": "pi05",
                "adapter_source": str(adapter.resolve()),
                "subgoal_slot": "right_wrist_0_rgb",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Merged inference checkpoints written to {args.output_dir}")


if __name__ == "__main__":
    export(tyro.cli(Args))
