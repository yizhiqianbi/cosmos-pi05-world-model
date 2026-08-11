"""Adaptive TTC router from proposal token-confidence statistics."""

from __future__ import annotations

from dataclasses import dataclass

from .types import ProposalResult
from .types import RouterDecision


@dataclass(frozen=True)
class RouterThresholds:
    probability: float
    memory_margin: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability threshold must be in [0, 1]")


def route_proposal(proposal: ProposalResult, thresholds: RouterThresholds) -> RouterDecision:
    """Implement Eq. 20: TTC when either confidence statistic is too low."""

    mean_probability = proposal.confidence.mean_probability
    mean_memory_margin = proposal.confidence.mean_memory_margin
    use_ttc = mean_probability <= thresholds.probability or mean_memory_margin <= thresholds.memory_margin
    return RouterDecision(
        use_ttc=use_ttc,
        mean_probability=mean_probability,
        mean_memory_margin=mean_memory_margin,
        probability_threshold=thresholds.probability,
        memory_margin_threshold=thresholds.memory_margin,
    )
