"""Episode-aware hierarchical controller with stale-result protection."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
import threading
import time

from .interfaces import ProposalModel
from .interfaces import ReflectionModel
from .router import RouterThresholds
from .router import route_proposal
from .search import WorldModelBeamSearch
from .search import deterministic_seed
from .types import HighLevelDecision
from .types import HighLevelObservation
from .types import PredictedOutcome


@dataclass
class EpisodeState:
    episode_id: str
    latest_sequence_id: int = -1
    memory: str = ""
    committed_subtask: str = ""
    decision: HighLevelDecision | None = None
    pending: Future | None = None
    updated_at_s: float = 0.0


class HierarchicalOrchestrator:
    """Runs direct proposal immediately and TTC in a background worker."""

    def __init__(
        self,
        proposal: ProposalModel,
        reflection: ReflectionModel,
        search: WorldModelBeamSearch,
        thresholds_for_task: Callable[[str], RouterThresholds],
        *,
        workers: int = 1,
        require_subgoal_image: bool = False,
    ) -> None:
        self.proposal = proposal
        self.reflection = reflection
        self.search = search
        self.thresholds_for_task = thresholds_for_task
        self.require_subgoal_image = bool(require_subgoal_image)
        self._episodes: dict[str, EpisodeState] = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cosmos-pi05-high-level")

    def reset(self, episode_id: str) -> EpisodeState:
        with self._lock:
            old = self._episodes.pop(episode_id, None)
            if old and old.pending:
                old.pending.cancel()
            state = EpisodeState(episode_id=episode_id, updated_at_s=time.time())
            self._episodes[episode_id] = state
            return state

    def submit(self, observation: HighLevelObservation) -> HighLevelDecision | None:
        with self._lock:
            state = self._episodes.setdefault(observation.episode_id, EpisodeState(observation.episode_id))
            if observation.sequence_id <= state.latest_sequence_id:
                raise ValueError(
                    f"Stale observation sequence {observation.sequence_id}; latest={state.latest_sequence_id}"
                )
            state.latest_sequence_id = observation.sequence_id
            if state.pending:
                state.pending.cancel()
            aligned = HighLevelObservation(
                episode_id=observation.episode_id,
                sequence_id=observation.sequence_id,
                timestamp_s=observation.timestamp_s,
                task_instruction=observation.task_instruction,
                previous_subtask=state.committed_subtask or observation.previous_subtask,
                memory=state.memory or observation.memory,
                images=observation.images,
                metadata=observation.metadata,
            )

        direct = self.proposal.generate_many(
            [aligned],
            samples_per_context=1,
            seeds=[[deterministic_seed(aligned.episode_id, aligned.sequence_id, "direct")]],
        )[0][0]
        router = route_proposal(direct, self.thresholds_for_task(aligned.task_instruction))
        # A deterministic fallback is deliberately committed directly.  It is
        # already a safe first clause, and sending it through TTC would turn a
        # model-formatting failure into an unnecessary Cosmos dependency.
        if direct.degraded:
            router = replace(router, use_ttc=False)
        fast_subgoal = (
            self._predict_subgoal(aligned, direct) if self.require_subgoal_image and not router.use_ttc else None
        )

        with self._lock:
            state = self._episodes[aligned.episode_id]
            if state.latest_sequence_id != aligned.sequence_id:
                return state.decision
            # The direct proposal is the only persistent-memory update even if TTC follows.
            state.memory = direct.memory
            if not router.use_ttc:
                subgoal = fast_subgoal
                subgoal_valid = bool(subgoal and subgoal.valid and subgoal.terminal_image is not None)
                decision = HighLevelDecision(
                    episode_id=aligned.episode_id,
                    sequence_id=aligned.sequence_id,
                    committed_subtask=direct.subtask,
                    memory=direct.memory,
                    route="degraded_fast" if direct.degraded else "fast",
                    router=router,
                    degraded=direct.degraded or (self.require_subgoal_image and not subgoal_valid),
                    error=(
                        direct.error
                        if direct.error
                        else subgoal.error
                        if subgoal is not None and not subgoal_valid
                        else None
                    ),
                    model_versions={
                        "proposal": self.proposal.version,
                        **({"world": self.search.world.version} if self.require_subgoal_image else {}),
                    },
                    subgoal_image=subgoal.terminal_image if subgoal_valid else None,
                    subgoal_source="cosmos_direct" if subgoal_valid else None,
                )
                state.committed_subtask = direct.subtask
                state.decision = decision
                state.pending = None
                state.updated_at_s = time.time()
                return decision

            reflection_context = replace(aligned, memory=direct.memory)
            state.pending = self._pool.submit(
                self._run_ttc,
                aligned,
                reflection_context,
                direct.subtask,
                direct,
                router,
            )
            state.pending.add_done_callback(
                lambda future, episode_id=aligned.episode_id, sequence_id=aligned.sequence_id: self._publish_ttc(
                    episode_id, sequence_id, future
                )
            )
            return state.decision

    def get_state(self, episode_id: str) -> EpisodeState | None:
        with self._lock:
            return self._episodes.get(episode_id)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _run_ttc(
        self, search_context, reflection_context, direct_subtask, direct_proposal, router
    ) -> HighLevelDecision:
        started = time.perf_counter()
        try:
            result = self.search.run(search_context)
            valid_beam = tuple(branch for branch in result.beam if branch.outcome and branch.outcome.valid)
            if not valid_beam:
                return HighLevelDecision(
                    episode_id=search_context.episode_id,
                    sequence_id=search_context.sequence_id,
                    committed_subtask=direct_subtask,
                    memory=reflection_context.memory,
                    route="ttc_degraded",
                    router=router,
                    beam=result.beam,
                    degraded=True,
                    error="all Cosmos branches invalid",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            committed = self.reflection.reflect(reflection_context, valid_beam)
            subgoal_image = None
            subgoal_source = None
            if self.require_subgoal_image:
                # A depth-2 branch ends at the second imagined state.  Match
                # Reflection's immediate command against path[0] and use the
                # preserved first-step outcome, never branch.outcome.
                matching = [
                    branch
                    for branch in valid_beam
                    if branch.path
                    and branch.path[0].strip().casefold() == committed.strip().casefold()
                    and branch.immediate_outcome is not None
                    and branch.immediate_outcome.valid
                    and branch.immediate_outcome.terminal_image is not None
                ]
                if matching:
                    selected = max(matching, key=lambda item: item.cumulative_score)
                    subgoal_image = selected.immediate_outcome.terminal_image
                    subgoal_source = "cosmos_beam_immediate"
                else:
                    # Reflection may safely repair/rephrase a candidate.  In
                    # that case generate an image aligned with the actual
                    # committed command from the real observation.
                    repaired = replace(direct_proposal, subtask=committed)
                    outcome = self._predict_subgoal(search_context, repaired)
                    if outcome.valid and outcome.terminal_image is not None:
                        subgoal_image = outcome.terminal_image
                        subgoal_source = "cosmos_reflection_refresh"
            return HighLevelDecision(
                episode_id=search_context.episode_id,
                sequence_id=search_context.sequence_id,
                committed_subtask=committed,
                memory=reflection_context.memory,
                route="ttc",
                router=router,
                beam=result.beam,
                degraded=self.require_subgoal_image and subgoal_image is None,
                error=(
                    "no valid Cosmos image for the committed immediate subtask"
                    if self.require_subgoal_image and subgoal_image is None
                    else None
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
                model_versions={
                    "proposal": self.proposal.version,
                    "world": self.search.world.version,
                    "value": self.search.value.version,
                    "reflection": self.reflection.version,
                },
                subgoal_image=subgoal_image,
                subgoal_source=subgoal_source,
            )
        except Exception as exc:
            return HighLevelDecision(
                episode_id=search_context.episode_id,
                sequence_id=search_context.sequence_id,
                committed_subtask=direct_subtask,
                memory=reflection_context.memory,
                route="ttc_degraded",
                router=router,
                degraded=True,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    def _predict_subgoal(self, context, proposal):
        seed = deterministic_seed(context.episode_id, context.sequence_id, "committed_subgoal")
        try:
            outcomes = self.search.world.predict_many(
                [(context, proposal)],
                seeds=[seed],
                profile=self.search.config.world_profile,
            )
            if len(outcomes) != 1:
                raise ValueError("World model returned a different number of committed subgoals")
            return outcomes[0]
        except Exception as exc:
            return PredictedOutcome.invalid(str(exc), seed=seed, profile=self.search.config.world_profile)

    def _publish_ttc(self, episode_id: str, sequence_id: int, future: Future) -> None:
        if future.cancelled():
            return
        decision = future.result()
        with self._lock:
            state = self._episodes.get(episode_id)
            if state is None or state.latest_sequence_id != sequence_id:
                return
            state.decision = decision
            state.committed_subtask = decision.committed_subtask
            state.pending = None
            state.updated_at_s = time.time()
