#!/usr/bin/env python3
"""True end-to-end PyTorch co-training for Cosmos3-Nano and pi0.5.

Unlike the replay/alternating trainer, this program decodes Cosmos' predicted
clean latent into a subgoal tensor and passes that tensor directly to pi0.5.
There is no detach, image file, NumPy conversion, or second optimizer step in
between, so the action loss updates both the VLA and selected Cosmos blocks.
"""

from __future__ import annotations

from contextlib import nullcontext
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Literal

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tyro

from cosmos_pi05.cotrain.cosmos_diffusers import DifferentiableCosmosI2V
from cosmos_pi05.cotrain.libero_joint_dataset import JointBatch
from cosmos_pi05.cotrain.libero_joint_dataset import LiberoCosmosPi05Dataset
from cosmos_pi05.cotrain.libero_joint_dataset import collate_joint
from cosmos_pi05.cotrain.torch_joint import TorchCosmosPi05JointModel
from cosmos_pi05.cotrain.torch_joint import load_trainable_adapter
from openpi.models import model as openpi_model
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as openpi_config


@dataclasses.dataclass(frozen=True)
class Args:
    cosmos_checkpoint: Path
    pi_checkpoint: Path
    output_dir: Path
    lerobot_root: Path
    cosmos_dataset_root: Path
    config_name: str = "pi05_libero_long_subgoal"
    split: Literal["train", "val", "all"] = "train"
    epochs: int = 5
    max_steps: int = 0
    local_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    seed: int = 7
    cosmos_lr: float = 1e-6
    pi_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    cosmos_trainable_blocks: int = 4
    pi_trainable: Literal["action-expert", "all"] = "action-expert"
    action_loss_weight: float = 1.0
    world_model_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 0.1
    generated_subgoal_start: float = 0.25
    generated_subgoal_warmup_steps: int = 500
    resolution: int = 256
    video_frames: int = 17
    fps: float = 12.0
    log_interval: int = 10
    save_interval: int = 250
    resume: bool = False


def _setup_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    return world_size, rank, local_rank, device


def _set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def _resolve_safetensors(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file() and path.suffix == ".safetensors":
        return path
    candidate = path / "model.safetensors"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No model.safetensors found at {path}")


def _set_cosmos_trainable(model: DifferentiableCosmosI2V, last_n_blocks: int) -> None:
    model.requires_grad_(requires_grad=False)
    layers = model.transformer.layers
    if not 1 <= last_n_blocks <= len(layers):
        raise ValueError(f"cosmos_trainable_blocks must be in [1, {len(layers)}]")
    for layer in layers[-last_n_blocks:]:
        layer.requires_grad_(requires_grad=True)
    model.transformer.norm_moe_gen.requires_grad_(requires_grad=True)
    model.transformer.proj_out.requires_grad_(requires_grad=True)
    model.vae.requires_grad_(requires_grad=False)


def _set_pi_trainable(model: PI0Pytorch, mode: str) -> None:
    if mode == "all":
        model.requires_grad_(requires_grad=True)
        return
    model.requires_grad_(requires_grad=False)
    prefixes = (
        "paligemma_with_expert.gemma_expert.",
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
    )
    for name, parameter in model.named_parameters():
        # GemmaForCausalLM registers a vocabulary lm_head, but pi0.5 consumes
        # hidden states directly and never calls that head.  Leaving it
        # trainable wastes 263M optimizer parameters and breaks DDP's reducer.
        if name.startswith(prefixes) and not name.startswith("paligemma_with_expert.gemma_expert.lm_head."):
            parameter.requires_grad_(requires_grad=True)


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def _move_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = tensor.to(device=device, non_blocking=True)
    if tensor.is_floating_point() and tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor


def _move_tree(value, device: torch.device):
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    return _move_tensor(value, device)


def _prepare_batch(batch: JointBatch, device: torch.device):
    policy = _move_tree(batch.policy, device)
    actions = policy.pop("actions").float()
    observation = openpi_model.Observation.from_dict(policy)
    target_video = batch.target_video.to(device=device, dtype=torch.float32, non_blocking=True)
    return observation, actions, target_video


def _generated_weight(args: Args, step: int) -> float:
    if args.generated_subgoal_warmup_steps <= 0:
        return 1.0
    progress = min(step / args.generated_subgoal_warmup_steps, 1.0)
    return args.generated_subgoal_start + (1.0 - args.generated_subgoal_start) * progress


def _cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))))


def _trainable_state(model: TorchCosmosPi05JointModel) -> dict[str, torch.Tensor]:
    # Clone tensors so safetensors does not reject tied storage and so GPU
    # memory can be released as each entry is materialized.
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _latest_checkpoint(output_dir: Path) -> Path:
    marker = output_dir / "latest.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Cannot resume: {marker} does not exist")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    checkpoint = output_dir / str(payload["checkpoint"])
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _save_checkpoint(
    model: TorchCosmosPi05JointModel,
    optimizer: torch.optim.Optimizer,
    args: Args,
    *,
    step: int,
    epoch: int,
) -> None:
    destination = args.output_dir / f"step_{step:08d}"
    destination.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_file(_trainable_state(model), destination / "trainable.safetensors")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    state = {"step": step, "epoch": epoch, "checkpoint": destination.name}
    (destination / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    (destination / "args.json").write_text(
        json.dumps(dataclasses.asdict(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    temporary = args.output_dir / "latest.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_dir / "latest.json")
    logging.info("Saved trainable Cosmos+pi0.5 adapter: %s", destination)


def _build_models(args: Args, device: torch.device) -> TorchCosmosPi05JointModel:
    config = openpi_config.get_config(args.config_name)
    pi_config = dataclasses.replace(config.model, dtype="bfloat16", pytorch_compile_mode=None)
    policy = PI0Pytorch(pi_config)
    safetensors.torch.load_model(policy, _resolve_safetensors(args.pi_checkpoint), device="cpu")
    policy.gradient_checkpointing_enable()
    _set_pi_trainable(policy, args.pi_trainable)

    world_model = DifferentiableCosmosI2V.from_pretrained(
        str(args.cosmos_checkpoint.expanduser().resolve()),
        dtype=torch.bfloat16,
        resolution=args.resolution,
        num_frames=args.video_frames,
        fps=args.fps,
    )
    world_model.enable_gradient_checkpointing()
    _set_cosmos_trainable(world_model, args.cosmos_trainable_blocks)
    model = TorchCosmosPi05JointModel(
        world_model,
        policy,
        action_loss_weight=args.action_loss_weight,
        world_model_loss_weight=args.world_model_loss_weight,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        generated_subgoal_weight=args.generated_subgoal_start,
    )
    return model.to(device)


def train(args: Args) -> None:
    world_size, rank, local_rank, device = _setup_distributed()
    _setup_logging(rank)
    _set_seed(args.seed, rank)
    if device.type != "cuda":
        raise RuntimeError("Full Cosmos3-Nano + pi0.5 co-training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if rank == 0:
        if args.output_dir.exists() and not args.resume:
            raise FileExistsError(f"Output exists; use --resume or a new --output-dir: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    config = openpi_config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = LiberoCosmosPi05Dataset(
        repo_id=str(data_config.repo_id),
        repo_root=args.lerobot_root,
        data_config=data_config,
        model_config=config.model,
        cosmos_dataset_root=args.cosmos_dataset_root,
        split=args.split,
        video_frames=args.video_frames,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_joint,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        drop_last=True,
    )

    logging.info("Loading both base models on rank %d", rank)
    model = _build_models(args, device)
    world_total, world_trainable = _count_parameters(model.world_model)
    pi_total, pi_trainable = _count_parameters(model.policy)
    logging.info(
        "Parameters: Cosmos %.2fB total / %.2fB trainable; pi0.5 %.2fB total / %.2fB trainable",
        world_total / 1e9,
        world_trainable / 1e9,
        pi_total / 1e9,
        pi_trainable / 1e9,
    )
    cosmos_parameters = [parameter for parameter in model.world_model.parameters() if parameter.requires_grad]
    pi_parameters = [parameter for parameter in model.policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": cosmos_parameters, "lr": args.cosmos_lr, "base_lr": args.cosmos_lr},
            {"params": pi_parameters, "lr": args.pi_lr, "base_lr": args.pi_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    step = 0
    first_epoch = 0
    if args.resume:
        checkpoint = _latest_checkpoint(args.output_dir)
        load_trainable_adapter(model, str(checkpoint / "trainable.safetensors"))
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=False))
        state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
        step = int(state["step"])
        first_epoch = int(state["epoch"])
        logging.info("Resumed %s at optimizer step %d", checkpoint, step)

    distributed_model: torch.nn.Module = model
    if world_size > 1:
        distributed_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps)
    epoch_steps = args.epochs * updates_per_epoch
    total_steps = min(epoch_steps, args.max_steps) if args.max_steps > 0 else epoch_steps
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()

    for epoch in range(first_epoch, args.epochs):
        reached_step_limit = False
        sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(loader):
            group_start = micro_step - micro_step % args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(loader) - group_start)
            synchronize = micro_step % args.gradient_accumulation_steps == group_size - 1
            sync_context = nullcontext() if synchronize or world_size == 1 else distributed_model.no_sync()
            observation, actions, target_video = _prepare_batch(batch, device)
            generated_weight = _generated_weight(args, step)
            with sync_context:
                output = distributed_model(
                    observation,
                    actions,
                    target_video,
                    batch.prompts,
                    generated_subgoal_weight=generated_weight,
                )
                log_bridge = rank == 0 and synchronize and (step == 0 or (step + 1) % args.log_interval == 0)
                if log_bridge:
                    output.generated_subgoal.retain_grad()
                (output.loss / group_size).backward()

            if not synchronize:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_([*cosmos_parameters, *pi_parameters], args.max_grad_norm)
            for group in optimizer.param_groups:
                group["lr"] = _cosine_lr(float(group["base_lr"]), step, total_steps, args.warmup_steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if rank == 0 and (step == 1 or step % args.log_interval == 0):
                bridge_grad = output.generated_subgoal.grad
                bridge_norm = float(bridge_grad.float().norm()) if bridge_grad is not None else float("nan")
                elapsed = time.monotonic() - started
                memory = torch.cuda.max_memory_allocated(device) / 2**30
                logging.info(
                    "epoch=%d step=%d/%d loss=%.5f action=%.5f wm=%.5f recon=%.5f "
                    "mix=%.3f bridge_grad=%.3e grad=%.3f peak=%.1fGiB elapsed=%.0fs",
                    epoch + 1,
                    step,
                    total_steps,
                    float(output.loss.detach()),
                    float(output.action_loss.detach()),
                    float(output.world_model_loss.detach()),
                    float(output.reconstruction_loss.detach()),
                    generated_weight,
                    bridge_norm,
                    float(grad_norm),
                    memory,
                    elapsed,
                )
            if step % args.save_interval == 0:
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    _save_checkpoint(model, optimizer, args, step=step, epoch=epoch)
                if dist.is_initialized():
                    dist.barrier()

            if args.max_steps > 0 and step >= args.max_steps:
                reached_step_limit = True
                break

        if rank == 0 and step % args.save_interval != 0:
            _save_checkpoint(model, optimizer, args, step=step, epoch=epoch if reached_step_limit else epoch + 1)
        if dist.is_initialized():
            dist.barrier()
        if reached_step_limit:
            break

    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    try:
        train(tyro.cli(Args))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
