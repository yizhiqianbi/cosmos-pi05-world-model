"""Pair pi0.5 LIBERO rows with their Cosmos stage-video supervision."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

import av
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import torch

from openpi import transforms as openpi_transforms
from openpi.training import data_loader as openpi_data

_STAGE_PATTERN = re.compile(r"episode_(?P<task>\d+)_(?P<demo>\d+)_stage_(?P<stage>\d+)$")


@dataclass(frozen=True)
class CosmosStageRecord:
    episode_index: int
    split: str
    uuid: str
    video_path: Path
    prompt: str


@dataclass
class JointBatch:
    policy: dict[str, Any]
    target_video: torch.Tensor
    prompts: list[str]
    episode_indices: torch.Tensor


def _stage_sort_key(uuid: str) -> tuple[int, int, int]:
    match = _STAGE_PATTERN.fullmatch(uuid)
    if match is None:
        raise ValueError(f"Unexpected Cosmos stage uuid: {uuid!r}")
    return tuple(int(match.group(name)) for name in ("task", "demo", "stage"))


def _future_indices(frame_index: int, episode_length: int, video_length: int, output_frames: int) -> np.ndarray:
    if episode_length < 1 or video_length < 1 or output_frames < 1:
        raise ValueError("episode_length, video_length and output_frames must be positive")
    progress = 0.0 if episode_length == 1 else min(max(frame_index / (episode_length - 1), 0.0), 1.0)
    start = round(progress * (video_length - 1))
    return np.linspace(start, video_length - 1, output_frames).round().astype(np.int64)


def _frame_indices(episode_lengths: Mapping[int, int], selected_episodes: list[int]) -> list[int]:
    """Convert global episode ids into indices of the unfiltered LeRobot table."""

    selected = set(selected_episodes)
    indices: list[int] = []
    offset = 0
    for episode_index in range(len(episode_lengths)):
        length = episode_lengths[episode_index]
        if episode_index in selected:
            indices.extend(range(offset, offset + length))
        offset += length
    return indices


class CosmosStageIndex:
    def __init__(self, root: str | Path, expected_episodes: int) -> None:
        root = Path(root).expanduser().resolve()
        unsorted: list[tuple[str, str, Path, str]] = []
        for split in ("train", "val"):
            manifest = root / split / "video_dataset_file.jsonl"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            for line in manifest.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                window = record["t2w_windows"][0]
                unsorted.append(
                    (
                        split,
                        str(record["uuid"]),
                        (root / split / str(record["vision_path"])).resolve(),
                        str(window["caption"]),
                    )
                )
        unsorted.sort(key=lambda item: _stage_sort_key(item[1]))
        if len(unsorted) != expected_episodes:
            raise ValueError(f"Cosmos/LeRobot episode mismatch: {len(unsorted)} != {expected_episodes}")
        self.records = [
            CosmosStageRecord(index, split, uuid, video_path, prompt)
            for index, (split, uuid, video_path, prompt) in enumerate(unsorted)
        ]
        missing = [str(record.video_path) for record in self.records if not record.video_path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} Cosmos videos; first: {missing[0]}")

    def episodes(self, split: str) -> list[int]:
        if split == "all":
            return list(range(len(self.records)))
        return [record.episode_index for record in self.records if record.split == split]

    def __getitem__(self, episode_index: int) -> CosmosStageRecord:
        return self.records[episode_index]


class LiberoCosmosPi05Dataset(torch.utils.data.Dataset):
    """Transformed pi0.5 samples plus future stage clips for Cosmos SFT."""

    def __init__(
        self,
        *,
        repo_id: str,
        repo_root: str | Path | None,
        data_config: Any,
        model_config: Any,
        cosmos_dataset_root: str | Path,
        split: str = "train",
        video_frames: int = 17,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val, or all")
        metadata = LeRobotDatasetMetadata(repo_id, root=repo_root)
        self.stage_index = CosmosStageIndex(cosmos_dataset_root, metadata.total_episodes)
        episodes = self.stage_index.episodes(split)
        episode_metadata = metadata.episodes.values() if isinstance(metadata.episodes, Mapping) else metadata.episodes
        self._episode_lengths = {int(item["episode_index"]): int(item["length"]) for item in episode_metadata}
        self._indices = _frame_indices(self._episode_lengths, episodes)
        self._raw = LeRobotDataset(
            repo_id,
            root=repo_root,
            delta_timestamps={
                key: [step / metadata.fps for step in range(model_config.action_horizon)]
                for key in data_config.action_sequence_keys
            },
        )
        self._raw = openpi_data.TransformedDataset(self._raw, [openpi_transforms.PromptFromLeRobotTask(metadata.tasks)])
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            raise ValueError("pi0.5 normalization stats are required for joint training")
        self._policy_transform = openpi_transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                openpi_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self.video_frames = int(video_frames)

    def __len__(self) -> int:
        return len(self._indices)

    @staticmethod
    @lru_cache(maxsize=8)
    def _decode_video(path: str) -> torch.Tensor:
        with av.open(path) as container:
            frames = [torch.from_numpy(frame.to_ndarray(format="rgb24")) for frame in container.decode(video=0)]
        if not frames:
            raise ValueError(f"Cosmos clip has no frames: {path}")
        return torch.stack(frames).permute(0, 3, 1, 2).contiguous()  # [T,C,H,W], uint8

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self._raw[self._indices[index]]
        episode_index = int(torch.as_tensor(raw["episode_index"]).item())
        frame_index = int(torch.as_tensor(raw["frame_index"]).item())
        stage = self.stage_index[episode_index]
        video = self._decode_video(str(stage.video_path))
        indices = _future_indices(
            frame_index,
            self._episode_lengths[episode_index],
            video.shape[0],
            self.video_frames,
        )
        target = video[torch.from_numpy(indices)].permute(1, 0, 2, 3).float() / 127.5 - 1.0
        policy = self._policy_transform(raw)
        policy = {
            key: policy[key]
            for key in (
                "image",
                "image_mask",
                "state",
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "actions",
            )
        }
        return {
            "policy": policy,
            "target_video": target,
            "prompt": stage.prompt,
            "episode_index": episode_index,
        }


def _collate_tree(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, Mapping):
        return {key: _collate_tree([item[key] for item in items]) for key in first}
    if isinstance(first, torch.Tensor):
        return torch.stack(items)
    return torch.stack([torch.as_tensor(np.asarray(item)) for item in items])


def collate_joint(items: list[dict[str, Any]]) -> JointBatch:
    return JointBatch(
        policy=_collate_tree([item["policy"] for item in items]),
        target_video=torch.stack([item["target_video"] for item in items]),
        prompts=[str(item["prompt"]) for item in items],
        episode_indices=torch.tensor([int(item["episode_index"]) for item in items], dtype=torch.long),
    )
