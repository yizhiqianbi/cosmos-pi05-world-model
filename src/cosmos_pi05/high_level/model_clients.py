"""HTTP clients for proposal, value, and reflective Qwen adapters."""

from __future__ import annotations

from collections.abc import Sequence

from .http_utils import image_to_base64_png
from .http_utils import post_json
from .types import HighLevelObservation
from .types import PredictedOutcome
from .types import ProposalResult
from .types import SearchBranch
from .types import TokenConfidence
from .types import ValueResult


def _context_payload(context: HighLevelObservation) -> dict:
    return {
        "episode_id": context.episode_id,
        "sequence_id": context.sequence_id,
        "task_instruction": context.task_instruction,
        "previous_subtask": context.previous_subtask,
        "memory": context.memory,
        "images_png_b64": {key: image_to_base64_png(value) for key, value in context.images.items()},
        "metadata": dict(context.metadata),
    }


class ProposalHttpModel:
    def __init__(self, endpoint: str, *, version: str = "proposal", timeout_s: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.version = version
        self.timeout_s = timeout_s

    def generate_many(
        self,
        contexts: Sequence[HighLevelObservation],
        *,
        samples_per_context: int,
        seeds: Sequence[Sequence[int]],
    ) -> list[list[ProposalResult]]:
        response = post_json(
            f"{self.endpoint}/v1/proposal/generate",
            {
                "adapter": self.version,
                "samples_per_context": samples_per_context,
                "contexts": [
                    {**_context_payload(context), "seeds": list(group)}
                    for context, group in zip(contexts, seeds, strict=True)
                ],
            },
            timeout_s=self.timeout_s,
        )
        groups = response.get("results")
        if not isinstance(groups, list):
            raise ValueError("Proposal service response has no results list")
        return [
            [
                ProposalResult(
                    thought=str(item.get("thought", "")),
                    memory=str(item.get("memory", "")),
                    subtask=str(item["subtask"]),
                    confidence=TokenConfidence(
                        tuple(float(x) for x in item.get("generated_token_probabilities", ())),
                        tuple(float(x) for x in item.get("memory_logit_margins", ())),
                    ),
                    raw_text=str(item.get("raw_text", "")),
                    sample_index=int(item.get("sample_index", idx)),
                    seed=int(item.get("seed", 0)),
                    degraded=bool(item.get("degraded", False)),
                    error=str(item["error"]) if item.get("error") else None,
                )
                for idx, item in enumerate(group)
            ]
            for group in groups
        ]


class ValueHttpModel:
    def __init__(self, endpoint: str, *, version: str = "value", timeout_s: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.version = version
        self.timeout_s = timeout_s

    def score_many(
        self,
        requests: Sequence[tuple[str, ProposalResult, PredictedOutcome]],
    ) -> list[ValueResult]:
        response = post_json(
            f"{self.endpoint}/v1/value/score",
            {
                "adapter": self.version,
                "requests": [
                    {
                        "task_instruction": task,
                        "candidate_subtask": proposal.subtask,
                        "terminal_image_png_b64": image_to_base64_png(outcome.terminal_image),
                    }
                    for task, proposal, outcome in requests
                ],
            },
            timeout_s=self.timeout_s,
        )
        return [
            ValueResult(float(item["score"]), str(item.get("level", "")), item.get("metadata", {}))
            for item in response.get("results", [])
        ]


class ReflectionHttpModel:
    def __init__(self, endpoint: str, *, version: str = "reflection", timeout_s: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.version = version
        self.timeout_s = timeout_s

    def reflect(self, context: HighLevelObservation, beam: Sequence[SearchBranch]) -> str:
        response = post_json(
            f"{self.endpoint}/v1/reflection/generate",
            {
                "adapter": self.version,
                "context": _context_payload(context),
                "beam": [branch.summary() for branch in beam],
            },
            timeout_s=self.timeout_s,
        )
        subtask = str(response.get("subtask", "")).strip()
        if not subtask:
            raise ValueError("Reflection service returned an empty subtask")
        return subtask
