"""LIBERO transforms for a natively subgoal-image-conditioned pi0.5 policy.

The base and left-wrist slots contain the current robot observation. The
otherwise unused right-wrist slot contains the desired near-future agent-view
image produced by Cosmos. Keeping all three inputs as first-class vision
tokens avoids lossy captions and uses pi0.5's native three-camera contract.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import einops
import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class GeneratedSubgoalOverlay(transforms.DataTransformFn):
    """Replace an oracle stage goal with a generated goal during training.

    The transform belongs to ``DataConfig.repack_transforms`` rather than the
    policy input transforms.  Consequently it is active only for LeRobot
    training rows and never tries to read an episode id during deployment.
    Missing generated goals fall back to the oracle image unless ``strict`` is
    requested.  The hash-based mixture is deterministic across dataloader
    workers and epochs.
    """

    directory: str
    probability: float = 1.0
    seed: int = 0
    strict: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("Generated-subgoal probability must be in [0, 1]")

    @staticmethod
    def _scalar(value: object, name: str) -> int:
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"{name} must contain one scalar, got {array.shape}")
        return int(array.reshape(()))

    def _selected(self, episode_index: int, frame_index: int) -> bool:
        if self.probability <= 0.0:
            return False
        if self.probability >= 1.0:
            return True
        # Integer mixing avoids process-specific Python hash randomization.
        value = (episode_index * 1_103_515_245 + frame_index * 12_345 + self.seed) & 0xFFFFFFFF
        return value / 2**32 < self.probability

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        episode_index = self._scalar(result.pop("_episode_index"), "episode_index")
        frame_index = self._scalar(result.pop("_frame_index"), "frame_index")
        if not self._selected(episode_index, frame_index):
            return result

        path = Path(self.directory) / f"episode_{episode_index:06d}.png"
        if not path.is_file():
            if self.strict:
                raise FileNotFoundError(f"Missing generated subgoal for selected row: {path}")
            return result
        with Image.open(path) as image:
            result["observation/subgoal_image"] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return result


def make_libero_subgoal_example() -> dict:
    """Return a valid unbatched inference example for smoke tests."""

    return {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/subgoal_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "Full task: put the mug on the plate\nCurrent executable subtask: pick up the mug",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot decodes images as C,H,W floats in [0, 1]. Runtime images
        # arrive as H,W,C uint8. Normalize both into the latter contract.
        image = np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected an image with 3 dimensions, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class LiberoSubgoalInputs(transforms.DataTransformFn):
    """Map LIBERO current observations and a visual subgoal into pi0.5."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        subgoal_image = _parse_image(data["observation/subgoal_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": subgoal_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = str(data["prompt"])
        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoSubgoalOutputs(transforms.DataTransformFn):
    """Return the seven-dimensional LIBERO OSC action."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
