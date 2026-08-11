"""Cosmos3-Nano subtask-conditioned image/video world-model adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json

from .http_utils import ModelServiceError
from .http_utils import base64_png_to_image
from .http_utils import image_to_base64_png
from .http_utils import post_json
from .types import HighLevelObservation
from .types import PredictedOutcome
from .types import ProposalResult


@dataclass(frozen=True)
class CosmosInferenceProfile:
    name: str
    mode: str = "image"
    resolution: int = 256
    # Video profiles use at least 24 frames in the official Cosmos3 Framework;
    # image profiles use a single image2image sample and ignore this field.
    num_frames: int = 1
    fps: int = 24
    denoising_steps: int = 35


DEFAULT_COSMOS_PROFILES = {
    # 35 is the official Cosmos Framework default for image2image/image2video.
    "deploy": CosmosInferenceProfile("deploy", mode="image", denoising_steps=35),
    "research": CosmosInferenceProfile("research", mode="video", num_frames=24, fps=10, denoising_steps=35),
}


def compile_cosmos_prompt(context: HighLevelObservation, subtask: str) -> str:
    """Compile deterministic structured vision-generation conditioning.

    The initial image supplies scene appearance.  The text only states the
    intended physical transition, preventing a runtime captioning model from
    hallucinating scene attributes that are not visible.
    """

    payload = {
        "task": context.task_instruction,
        "subtask": subtask,
        "embodiment": str(context.metadata.get("embodiment", "AGIBOT G1")),
        "camera": "fixed robot head camera",
        "objective": "simulate the visible execution and end at the completed physical state",
        "constraints": [
            "preserve object identity and scene layout unless changed by the subtask",
            "do not add objects that are absent from the input image",
            "show the terminal state clearly in the final frame",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class Cosmos3NanoHttpWorldModel:
    """Batch client for a separately deployed Cosmos generator service.

    Service contract (the deployment profile is image2image; the research
    profile is image2video)::

        POST /v1/world/predict
        {"requests": [{"initial_image_png_b64", "prompt", "mode", ...}]}
        -> {"outcomes": [{"terminal_image_png_b64", "video_uri", ...}]}
    """

    def __init__(
        self,
        endpoint: str,
        *,
        version: str = "cosmos3-nano",
        timeout_s: float = 180.0,
        profiles: dict[str, CosmosInferenceProfile] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.version = version
        self.timeout_s = timeout_s
        self.profiles = profiles or DEFAULT_COSMOS_PROFILES

    def predict_many(
        self,
        requests: Sequence[tuple[HighLevelObservation, ProposalResult]],
        *,
        seeds: Sequence[int],
        profile: str,
    ) -> list[PredictedOutcome]:
        if len(requests) != len(seeds):
            raise ValueError("requests and seeds must have equal length")
        if profile not in self.profiles:
            raise KeyError(f"Unknown Cosmos profile {profile!r}; available={sorted(self.profiles)}")
        cfg = self.profiles[profile]
        service_requests = []
        for (context, proposal), seed in zip(requests, seeds, strict=True):
            service_requests.append(
                {
                    "request_id": f"{context.episode_id}:{context.sequence_id}:{proposal.sample_index}",
                    "initial_image_png_b64": image_to_base64_png(context.head_image),
                    "prompt": compile_cosmos_prompt(context, proposal.subtask),
                    "seed": int(seed),
                    "mode": cfg.mode,
                    "resolution": cfg.resolution,
                    "num_frames": cfg.num_frames,
                    "fps": cfg.fps,
                    "denoising_steps": cfg.denoising_steps,
                    "generate_audio": False,
                }
            )
        try:
            response = post_json(
                f"{self.endpoint}/v1/world/predict",
                {"model": self.version, "profile": profile, "requests": service_requests},
                timeout_s=self.timeout_s,
            )
        except ModelServiceError as exc:
            return [PredictedOutcome.invalid(str(exc), seed=seed, profile=profile) for seed in seeds]
        raw_outcomes = response.get("outcomes")
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(requests):
            error = (
                f"Cosmos service returned {len(raw_outcomes) if isinstance(raw_outcomes, list) else 'invalid'} outcomes"
            )
            return [PredictedOutcome.invalid(error, seed=seed, profile=profile) for seed in seeds]

        outcomes = []
        for raw, seed in zip(raw_outcomes, seeds, strict=True):
            if not isinstance(raw, dict) or not raw.get("valid", True):
                outcomes.append(
                    PredictedOutcome.invalid(
                        str(raw.get("error", "invalid Cosmos response"))
                        if isinstance(raw, dict)
                        else "invalid response",
                        seed=seed,
                        profile=profile,
                    )
                )
                continue
            terminal_b64 = raw.get("terminal_image_png_b64")
            if not terminal_b64:
                outcomes.append(PredictedOutcome.invalid("missing terminal frame", seed=seed, profile=profile))
                continue
            outcomes.append(
                PredictedOutcome(
                    terminal_image=base64_png_to_image(terminal_b64),
                    video_uri=raw.get("video_uri"),
                    valid=True,
                    seed=seed,
                    profile=profile,
                    latency_ms=raw.get("latency_ms"),
                    metadata={k: v for k, v in raw.items() if k not in {"terminal_image_png_b64"}},
                )
            )
        return outcomes
