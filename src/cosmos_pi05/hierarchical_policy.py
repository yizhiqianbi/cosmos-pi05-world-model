"""A pi0.5 policy wrapper driven by Qwen planning and Cosmos visual subgoals."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from cosmos_pi05.high_level.cosmos import Cosmos3NanoHttpWorldModel
from cosmos_pi05.high_level.model_clients import ProposalHttpModel
from cosmos_pi05.high_level.model_clients import ReflectionHttpModel
from cosmos_pi05.high_level.model_clients import ValueHttpModel
from cosmos_pi05.high_level.orchestrator import HierarchicalOrchestrator
from cosmos_pi05.high_level.router import RouterThresholds
from cosmos_pi05.high_level.search import SearchConfig
from cosmos_pi05.high_level.search import WorldModelBeamSearch
from cosmos_pi05.high_level.types import HighLevelDecision
from cosmos_pi05.high_level.types import HighLevelObservation


def low_level_prompt(task_instruction: str, subtask: str) -> str:
    """Text contract used identically by training and inference."""

    return f"Full task: {task_instruction.strip()}\nCurrent executable subtask: {subtask.strip()}"


def build_orchestrator(
    *,
    high_level_endpoint: str,
    cosmos_endpoint: str,
    probability_threshold: float = 0.65,
    memory_margin_threshold: float = 1.0,
    branching_factor: int = 3,
    beam_width: int = 2,
    depth: int = 2,
    world_profile: str = "deploy",
    timeout_s: float = 180.0,
) -> HierarchicalOrchestrator:
    """Build the complete Proposal -> TTC/Beam -> Cosmos planner."""

    proposal = ProposalHttpModel(high_level_endpoint, timeout_s=timeout_s)
    value = ValueHttpModel(high_level_endpoint, timeout_s=timeout_s)
    reflection = ReflectionHttpModel(high_level_endpoint, timeout_s=timeout_s)
    world = Cosmos3NanoHttpWorldModel(cosmos_endpoint, timeout_s=timeout_s)
    search = WorldModelBeamSearch(
        proposal,
        world,
        value,
        SearchConfig(
            branching_factor=branching_factor,
            beam_width=beam_width,
            depth=depth,
            world_profile=world_profile,
        ),
    )
    thresholds = RouterThresholds(probability_threshold, memory_margin_threshold)
    return HierarchicalOrchestrator(
        proposal,
        reflection,
        search,
        lambda _task: thresholds,
        workers=1,
        # Unlike the legacy text-only Long policy, this pi0.5 route refuses to
        # execute unless the selected immediate subtask has a visual goal.
        require_subgoal_image=True,
    )


class HierarchicalPi05Policy(_base_policy.BasePolicy):
    """Expose the hierarchical stack through openpi's ordinary policy API.

    Required observation keys are the standard LIBERO keys plus
    ``episode_id`` and ``sequence_id``.  A sequence is observation-aligned:
    exactly one high-level decision and one pi0.5 action chunk are committed
    for each sequence number.
    """

    def __init__(
        self,
        low_level_policy: _base_policy.BasePolicy,
        orchestrator: HierarchicalOrchestrator,
        *,
        decision_timeout_s: float = 180.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._low_level = low_level_policy
        self._orchestrator = orchestrator
        self._decision_timeout_s = float(decision_timeout_s)
        self._lock = threading.RLock()
        self._known_episodes: set[str] = set()
        self._metadata = {
            "architecture": "qwen3.5+cosmos3-nano+pi0.5",
            "requires_subgoal_image": True,
            "schema_version": "cosmos-pi05.v1",
            **(metadata or {}),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def reset_episode(self, episode_id: str) -> None:
        with self._lock:
            self._orchestrator.reset(episode_id)
            self._known_episodes.add(episode_id)

    @override
    def reset(self) -> None:
        # The websocket protocol has no episode identifier on reset.  Runtime
        # callers should send sequence_id=0; infer() then resets that episode.
        pass

    @override
    def infer(self, obs: dict) -> dict:
        episode_id = str(obs.get("episode_id", "")).strip()
        if not episode_id:
            raise ValueError("episode_id is required for hierarchical inference")
        sequence_id = int(obs.get("sequence_id", -1))
        if sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        task = str(obs.get("task_instruction") or obs.get("prompt") or "").strip()
        if not task:
            raise ValueError("task_instruction or prompt is required")

        agent_image = np.asarray(obs["observation/image"], dtype=np.uint8)
        wrist_image = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"LIBERO state must have shape (8,), got {state.shape}")

        with self._lock:
            if sequence_id == 0 or episode_id not in self._known_episodes:
                self.reset_episode(episode_id)
            context = HighLevelObservation(
                episode_id=episode_id,
                sequence_id=sequence_id,
                timestamp_s=float(obs.get("timestamp_s", time.time())),
                task_instruction=task,
                previous_subtask=str(obs.get("previous_subtask", "")),
                memory=str(obs.get("memory", "")),
                images={"agentview": agent_image, "wrist": wrist_image},
                metadata={"embodiment": "Franka Panda", "environment": "LIBERO"},
            )
            decision = self._orchestrator.submit(context)
            decision = self._wait_for_current_decision(episode_id, sequence_id, decision)

            if decision.degraded:
                raise RuntimeError(f"High-level decision degraded: {decision.error or decision.route}")
            if decision.subgoal_image is None:
                raise RuntimeError("Planner committed a subtask without a Cosmos subgoal image")

            low_obs = {
                "observation/image": agent_image,
                "observation/wrist_image": wrist_image,
                "observation/subgoal_image": np.asarray(decision.subgoal_image, dtype=np.uint8),
                "observation/state": state,
                "prompt": low_level_prompt(task, decision.committed_subtask),
            }
            result = dict(self._low_level.infer(low_obs))
            actions = np.asarray(result.get("actions"), dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"pi0.5 returned invalid LIBERO action shape {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("pi0.5 returned non-finite actions")
            result.update(
                {
                    "actions": actions,
                    "episode_id": episode_id,
                    "sequence_id": sequence_id,
                    "committed_subtask": decision.committed_subtask,
                    "decision": decision.as_dict(),
                    "subgoal_image": np.asarray(decision.subgoal_image, dtype=np.uint8),
                }
            )
            return result

    def _wait_for_current_decision(
        self,
        episode_id: str,
        sequence_id: int,
        decision: HighLevelDecision | None,
    ) -> HighLevelDecision:
        if decision is not None and decision.sequence_id == sequence_id:
            return decision
        deadline = time.monotonic() + self._decision_timeout_s
        while time.monotonic() < deadline:
            state = self._orchestrator.get_state(episode_id)
            if state is not None and state.decision is not None and state.decision.sequence_id == sequence_id:
                return state.decision
            time.sleep(0.02)
        raise TimeoutError(f"No high-level decision for {episode_id=} {sequence_id=}")
