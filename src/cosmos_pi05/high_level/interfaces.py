"""Protocols implemented by high-level model and world-model adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import HighLevelObservation
from .types import PredictedOutcome
from .types import ProposalResult
from .types import SearchBranch
from .types import ValueResult


class ProposalModel(Protocol):
    version: str

    def generate_many(
        self,
        contexts: Sequence[HighLevelObservation],
        *,
        samples_per_context: int,
        seeds: Sequence[Sequence[int]],
    ) -> list[list[ProposalResult]]: ...


class WorldModel(Protocol):
    version: str

    def predict_many(
        self,
        requests: Sequence[tuple[HighLevelObservation, ProposalResult]],
        *,
        seeds: Sequence[int],
        profile: str,
    ) -> list[PredictedOutcome]: ...


class ValueModel(Protocol):
    version: str

    def score_many(
        self,
        requests: Sequence[tuple[str, ProposalResult, PredictedOutcome]],
    ) -> list[ValueResult]: ...


class ReflectionModel(Protocol):
    version: str

    def reflect(self, context: HighLevelObservation, beam: Sequence[SearchBranch]) -> str: ...
