from __future__ import annotations

from dataclasses import dataclass

import safetensors.torch
import torch
from torch import nn

from cosmos_pi05.cotrain.torch_joint import TorchCosmosPi05JointModel
from cosmos_pi05.cotrain.torch_joint import WorldModelForwardOutput
from cosmos_pi05.cotrain.torch_joint import load_trainable_adapter


@dataclass
class _Observation:
    images: dict[str, torch.Tensor]


class _TinyWorldModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.25))

    def forward(self, current, target_video, prompts):
        del prompts
        generated = current * self.gain + target_video[:, :, -1] * (1.0 - self.gain)
        zero = generated.sum() * 0.0
        return WorldModelForwardOutput(generated, zero, zero, torch.full((current.shape[0],), 0.5))


class _TinyPolicy(nn.Module):
    def forward(self, observation, actions):
        prediction = observation.images["right_wrist_0_rgb"].mean(dim=(1, 2, 3))
        target = actions.mean(dim=(1, 2))
        return (prediction - target).square()


def _inputs():
    current = torch.ones(2, 3, 8, 8)
    oracle = torch.zeros_like(current)
    observation = _Observation(images={"base_0_rgb": current, "left_wrist_0_rgb": current, "right_wrist_0_rgb": oracle})
    target = torch.zeros(2, 3, 5, 8, 8)
    actions = torch.ones(2, 4, 2)
    return observation, actions, target


def test_action_loss_backpropagates_into_world_model():
    world = _TinyWorldModel()
    model = TorchCosmosPi05JointModel(
        world,
        _TinyPolicy(),
        world_model_loss_weight=0.0,
        reconstruction_loss_weight=0.0,
        generated_subgoal_weight=1.0,
    )
    observation, actions, target = _inputs()
    output = model(observation, actions, target, ["a", "b"])
    output.loss.backward()
    assert world.gain.grad is not None
    assert world.gain.grad.abs().item() > 0.0


def test_oracle_only_mix_blocks_action_gradient_to_world_model():
    world = _TinyWorldModel()
    model = TorchCosmosPi05JointModel(
        world,
        _TinyPolicy(),
        world_model_loss_weight=0.0,
        reconstruction_loss_weight=0.0,
        generated_subgoal_weight=0.0,
    )
    observation, actions, target = _inputs()
    output = model(observation, actions, target, ["a", "b"])
    output.loss.backward()
    assert world.gain.grad is not None
    assert world.gain.grad.abs().item() == 0.0


def test_trainable_adapter_round_trip(tmp_path):
    world = _TinyWorldModel()
    model = TorchCosmosPi05JointModel(world, _TinyPolicy())
    path = tmp_path / "adapter.safetensors"
    safetensors.torch.save_file({"world_model.gain": torch.tensor(0.75)}, path)
    load_trainable_adapter(model, str(path))
    assert world.gain.item() == 0.75
