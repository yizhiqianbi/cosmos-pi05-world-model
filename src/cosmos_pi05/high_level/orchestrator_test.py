from __future__ import annotations

# ruff: noqa: PT009
import time
import unittest

import numpy as np

from cosmos_pi05.high_level.orchestrator import HierarchicalOrchestrator
from cosmos_pi05.high_level.router import RouterThresholds
from cosmos_pi05.high_level.types import HighLevelObservation
from cosmos_pi05.high_level.types import PredictedOutcome
from cosmos_pi05.high_level.types import ProposalResult
from cosmos_pi05.high_level.types import SearchBranch
from cosmos_pi05.high_level.types import SearchResult
from cosmos_pi05.high_level.types import TokenConfidence


class SequenceProposal:
    version = "proposal-test"

    def generate_many(self, contexts, *, samples_per_context, seeds):
        groups = []
        for context in contexts:
            confidence = (
                TokenConfidence((0.2,), (0.1,)) if context.sequence_id == 1 else TokenConfidence((0.99,), (9.0,))
            )
            groups.append(
                [
                    ProposalResult(
                        "",
                        f"memory-{context.sequence_id}",
                        f"direct-{context.sequence_id}",
                        confidence,
                        sample_index=0,
                    )
                ]
            )
        return groups


class SlowSearch:
    world = type("World", (), {"version": "world-test"})()
    value = type("Value", (), {"version": "value-test"})()

    def run(self, context):
        time.sleep(0.08)
        outcome = PredictedOutcome(np.zeros((2, 2, 3), dtype=np.uint8))
        branch = SearchBranch("b", "root", 1, 0, ("imagined",), "branch-memory", 1.0, 1.0, outcome)
        return SearchResult((branch,), 1, 0)


class Reflection:
    version = "reflection-test"

    def reflect(self, context, beam):
        return "reflected-old"


class DirectWorld:
    version = "world-direct-test"

    def predict_many(self, requests, *, seeds, profile):
        return [
            PredictedOutcome(
                np.full((3, 4, 3), 73, dtype=np.uint8),
                seed=seed,
                profile=profile,
            )
            for seed in seeds
        ]


class DegradedProposal:
    version = "proposal-degraded"

    def generate_many(self, contexts, *, samples_per_context, seeds):
        return [
            [
                ProposalResult(
                    "",
                    "",
                    "pick up the object",
                    TokenConfidence((), ()),
                    degraded=True,
                    error="empty model subtask",
                )
            ]
            for _ in contexts
        ]


class OrchestratorTest(unittest.TestCase):
    def test_fast_route_generates_native_low_level_subgoal_when_required(self):
        search = SlowSearch()
        search.world = DirectWorld()
        search.config = type("Config", (), {"world_profile": "deploy"})()
        orchestrator = HierarchicalOrchestrator(
            SequenceProposal(),
            Reflection(),
            search,
            lambda _: RouterThresholds(0.5, 1.0),
            require_subgoal_image=True,
        )
        try:
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            decision = orchestrator.submit(HighLevelObservation("ep", 2, 0.0, "task", images={"head": image}))
            self.assertEqual(decision.route, "fast")
            self.assertEqual(decision.subgoal_source, "cosmos_direct")
            self.assertEqual(decision.subgoal_image.shape, (3, 4, 3))
            self.assertTrue(decision.as_dict()["has_subgoal_image"])
            self.assertNotIn("subgoal_image", decision.as_dict())
        finally:
            orchestrator.close()

    def test_degraded_proposal_commits_without_ttc(self):
        orchestrator = HierarchicalOrchestrator(
            DegradedProposal(), Reflection(), SlowSearch(), lambda _: RouterThresholds(0.5, 1.0)
        )
        try:
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            decision = orchestrator.submit(HighLevelObservation("ep", 1, 0.0, "task", images={"head": image}))
            self.assertEqual(decision.route, "degraded_fast")
            self.assertTrue(decision.degraded)
            self.assertEqual(decision.committed_subtask, "pick up the object")
            self.assertIsNone(decision.beam or None)
        finally:
            orchestrator.close()

    def test_stale_background_ttc_cannot_overwrite_newer_fast_decision(self):
        orchestrator = HierarchicalOrchestrator(
            SequenceProposal(), Reflection(), SlowSearch(), lambda _: RouterThresholds(0.5, 1.0)
        )
        try:
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            first = HighLevelObservation("ep", 1, 0.0, "task", images={"head": image})
            second = HighLevelObservation("ep", 2, 0.1, "task", images={"head": image})
            self.assertIsNone(orchestrator.submit(first))
            decision = orchestrator.submit(second)
            self.assertEqual(decision.committed_subtask, "direct-2")
            time.sleep(0.12)
            state = orchestrator.get_state("ep")
            self.assertEqual(state.latest_sequence_id, 2)
            self.assertEqual(state.committed_subtask, "direct-2")
            self.assertEqual(state.memory, "memory-2")
        finally:
            orchestrator.close()


if __name__ == "__main__":
    unittest.main()
