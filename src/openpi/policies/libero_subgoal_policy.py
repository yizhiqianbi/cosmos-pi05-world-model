"""LIBERO transforms for a natively subgoal-image-conditioned pi0.5 policy.

The base and left-wrist slots contain the current robot observation. The
otherwise unused right-wrist slot contains the desired near-future agent-view
image produced by Cosmos. Keeping all three inputs as first-class vision
tokens avoids lossy captions and uses pi0.5's native three-camera contract.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


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
