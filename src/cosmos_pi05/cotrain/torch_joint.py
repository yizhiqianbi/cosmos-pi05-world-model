"""A single PyTorch autograd graph spanning Cosmos and pi0.5.

The world model returns a differentiable terminal image.  That image replaces
pi0.5's third camera slot without a detach or a PIL/NumPy round trip.  The pi0.5
action flow-matching loss can therefore update the trainable Cosmos blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


@dataclass
class WorldModelForwardOutput:
    """Differentiable output of one Cosmos training step."""

    subgoal_image: torch.Tensor
    flow_matching_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sigma: torch.Tensor


@dataclass
class JointForwardOutput:
    """All losses and bridge tensors needed by the trainer."""

    loss: torch.Tensor
    action_loss: torch.Tensor
    world_model_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    generated_subgoal: torch.Tensor
    policy_subgoal: torch.Tensor
    sigma: torch.Tensor


def _replace_subgoal(observation: Any, subgoal: torch.Tensor, slot: str) -> Any:
    images = dict(observation.images)
    if slot not in images:
        raise KeyError(f"pi0.5 observation has no subgoal slot {slot!r}; found {sorted(images)}")
    images[slot] = subgoal

    # Flax struct dataclasses expose ``replace`` while ordinary dataclasses
    # commonly do not.  Keeping both paths makes the bridge independently
    # testable without importing JAX.
    replace = getattr(observation, "replace", None)
    if callable(replace):
        return replace(images=images)

    from dataclasses import replace as dataclass_replace

    return dataclass_replace(observation, images=images)


class TorchCosmosPi05JointModel(nn.Module):
    """Compose a differentiable Cosmos I2V module and the torch pi0.5 VLA."""

    def __init__(
        self,
        world_model: nn.Module,
        policy: nn.Module,
        *,
        action_loss_weight: float = 1.0,
        world_model_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 0.1,
        generated_subgoal_weight: float = 1.0,
        subgoal_slot: str = "right_wrist_0_rgb",
    ) -> None:
        super().__init__()
        if not 0.0 <= generated_subgoal_weight <= 1.0:
            raise ValueError("generated_subgoal_weight must be in [0, 1]")
        self.world_model = world_model
        self.policy = policy
        self.action_loss_weight = float(action_loss_weight)
        self.world_model_loss_weight = float(world_model_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.generated_subgoal_weight = float(generated_subgoal_weight)
        self.subgoal_slot = subgoal_slot

    def forward(
        self,
        observation: Any,
        actions: torch.Tensor,
        target_video: torch.Tensor,
        prompts: list[str],
        *,
        generated_subgoal_weight: float | None = None,
    ) -> JointForwardOutput:
        if target_video.ndim != 5:
            raise ValueError(f"target_video must be [B,C,T,H,W], got {tuple(target_video.shape)}")
        current_image = observation.images["base_0_rgb"]
        oracle_subgoal = observation.images[self.subgoal_slot]

        wm: WorldModelForwardOutput = self.world_model(current_image, target_video, prompts)
        generated = wm.subgoal_image
        if generated.shape[-2:] != oracle_subgoal.shape[-2:]:
            generated = F.interpolate(
                generated,
                size=oracle_subgoal.shape[-2:],
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        generated = generated.to(dtype=oracle_subgoal.dtype)

        mix = self.generated_subgoal_weight if generated_subgoal_weight is None else float(generated_subgoal_weight)
        if not 0.0 <= mix <= 1.0:
            raise ValueError("generated_subgoal_weight must be in [0, 1]")
        policy_subgoal = mix * generated + (1.0 - mix) * oracle_subgoal
        policy_observation = _replace_subgoal(observation, policy_subgoal, self.subgoal_slot)

        action_losses = self.policy(policy_observation, actions)
        if not isinstance(action_losses, torch.Tensor):
            action_losses = torch.stack(list(action_losses))
        action_loss = action_losses.float().mean()
        total = (
            self.action_loss_weight * action_loss
            + self.world_model_loss_weight * wm.flow_matching_loss
            + self.reconstruction_loss_weight * wm.reconstruction_loss
        )
        return JointForwardOutput(
            loss=total,
            action_loss=action_loss,
            world_model_loss=wm.flow_matching_loss,
            reconstruction_loss=wm.reconstruction_loss,
            generated_subgoal=generated,
            policy_subgoal=policy_subgoal,
            sigma=wm.sigma,
        )


def load_trainable_adapter(model: nn.Module, checkpoint: str) -> None:
    """Load a co-training adapter without materializing the frozen base weights."""

    state = safetensors.torch.load_file(checkpoint, device="cpu")
    current = dict(model.named_parameters())
    unexpected = sorted(set(state) - set(current))
    if unexpected:
        raise ValueError(f"Unexpected adapter tensors: {unexpected[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = current[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
