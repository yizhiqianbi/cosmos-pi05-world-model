"""Core, transport-neutral types for the hierarchical pi0.5 controller.

These dataclasses intentionally contain no FastAPI, ROS, Transformers, or
Cosmos imports.  They are the contract shared by offline evaluation, search,
HTTP serving, and the optional ROS 2 bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class TokenConfidence:
    """Confidence statistics collected from one proposal generation."""

    generated_token_probabilities: tuple[float, ...]
    memory_logit_margins: tuple[float, ...]

    @property
    def mean_probability(self) -> float:
        values = self.generated_token_probabilities
        return sum(values) / len(values) if values else 0.0

    @property
    def mean_memory_margin(self) -> float:
        values = self.memory_logit_margins
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class HighLevelObservation:
    """Observation-aligned context consumed by the high-level policy."""

    episode_id: str
    sequence_id: int
    timestamp_s: float
    task_instruction: str
    previous_subtask: str = ""
    memory: str = ""
    images: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def head_image(self) -> Any:
        # LIBERO calls its fixed external camera ``agentview``; treat it as
        # the high-level/head view while preserving the robot-camera names.
        for key in (
            "head",
            "top_head",
            "observation.images.head",
            "observation.images.top_head",
            "agentview",
            "observation.images.agentview",
        ):
            if key in self.images:
                return self.images[key]
        raise KeyError(f"High-level observation has no head camera; available={list(self.images)}")

    def with_branch_state(self, *, memory: str, previous_subtask: str, head_image: Any) -> HighLevelObservation:
        return HighLevelObservation(
            episode_id=self.episode_id,
            sequence_id=self.sequence_id,
            timestamp_s=self.timestamp_s,
            task_instruction=self.task_instruction,
            previous_subtask=previous_subtask,
            memory=memory,
            images={"head": head_image},
            metadata={**self.metadata, "imagined": True},
        )


@dataclass(frozen=True)
class ProposalResult:
    thought: str
    memory: str
    subtask: str
    confidence: TokenConfidence
    raw_text: str = ""
    sample_index: int = 0
    seed: int = 0
    degraded: bool = False
    error: str | None = None


@dataclass(frozen=True)
class PredictedOutcome:
    """A Cosmos-generated consequence for one proposed subtask."""

    terminal_image: Any | None
    video_uri: str | None = None
    valid: bool = True
    error: str | None = None
    seed: int = 0
    profile: str = "deploy"
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def invalid(cls, error: str, *, seed: int = 0, profile: str = "deploy") -> PredictedOutcome:
        return cls(terminal_image=None, valid=False, error=error, seed=seed, profile=profile)


@dataclass(frozen=True)
class ValueResult:
    score: float
    level: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchBranch:
    branch_id: str
    parent_id: str | None
    depth: int
    sample_index: int
    path: tuple[str, ...]
    memory: str
    local_score: float
    cumulative_score: float
    outcome: PredictedOutcome | None
    proposal: ProposalResult | None = None
    # For depth>1 the branch's ``outcome`` is the terminal image after the
    # last imagined subtask.  The low-level policy, however, must pursue the
    # first subtask from the real observation.  Keep that first prediction
    # explicitly so a deep search can never feed a second-step image to VLA.
    immediate_outcome: PredictedOutcome | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "sample_index": self.sample_index,
            "path": list(self.path),
            "memory": self.memory,
            "local_score": self.local_score,
            "cumulative_score": self.cumulative_score,
            "video_uri": self.outcome.video_uri if self.outcome else None,
            "outcome_valid": bool(self.outcome and self.outcome.valid),
            "outcome_error": self.outcome.error if self.outcome else None,
            "has_immediate_subgoal": bool(
                self.immediate_outcome
                and self.immediate_outcome.valid
                and self.immediate_outcome.terminal_image is not None
            ),
        }


@dataclass(frozen=True)
class SearchResult:
    beam: tuple[SearchBranch, ...]
    expanded_nodes: int
    invalid_nodes: int


@dataclass(frozen=True)
class RouterDecision:
    use_ttc: bool
    mean_probability: float
    mean_memory_margin: float
    probability_threshold: float
    memory_margin_threshold: float


@dataclass(frozen=True)
class HighLevelDecision:
    episode_id: str
    sequence_id: int
    committed_subtask: str
    memory: str
    route: str
    router: RouterDecision
    beam: tuple[SearchBranch, ...] = ()
    degraded: bool = False
    error: str | None = None
    latency_ms: float | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)
    # Transport intentionally omits the raw image.  The in-process controller
    # injects it into the low-level VLA payload; JSON/MessagePack traces expose
    # only presence/provenance unless an endpoint explicitly encodes pixels.
    subgoal_image: Any | None = None
    subgoal_source: str | None = None

    def as_dict(self, *, include_beam: bool = True) -> dict[str, Any]:
        payload = {
            "episode_id": self.episode_id,
            "sequence_id": self.sequence_id,
            "committed_subtask": self.committed_subtask,
            "memory": self.memory,
            "route": self.route,
            "degraded": self.degraded,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "model_versions": dict(self.model_versions),
            "has_subgoal_image": self.subgoal_image is not None,
            "subgoal_source": self.subgoal_source,
            "router": {
                "use_ttc": self.router.use_ttc,
                "mean_probability": self.router.mean_probability,
                "mean_memory_margin": self.router.mean_memory_margin,
                "probability_threshold": self.router.probability_threshold,
                "memory_margin_threshold": self.router.memory_margin_threshold,
            },
        }
        if include_beam:
            payload["beam"] = [branch.summary() for branch in self.beam]
        return payload


def ensure_unique_branch_ids(branches: Sequence[SearchBranch]) -> None:
    ids = [branch.branch_id for branch in branches]
    if len(ids) != len(set(ids)):
        raise ValueError("Search produced duplicate branch ids")
