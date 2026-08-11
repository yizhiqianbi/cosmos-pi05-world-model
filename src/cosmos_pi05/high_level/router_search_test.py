from __future__ import annotations

from typing import ClassVar

# ruff: noqa: PT009
import unittest

import numpy as np

from cosmos_pi05.high_level.cosmos import DEFAULT_COSMOS_PROFILES
from cosmos_pi05.high_level.router import RouterThresholds
from cosmos_pi05.high_level.router import route_proposal
from cosmos_pi05.high_level.search import SearchConfig
from cosmos_pi05.high_level.search import WorldModelBeamSearch
from cosmos_pi05.high_level.search import deterministic_seed
from cosmos_pi05.high_level.types import HighLevelObservation
from cosmos_pi05.high_level.types import PredictedOutcome
from cosmos_pi05.high_level.types import ProposalResult
from cosmos_pi05.high_level.types import TokenConfidence
from cosmos_pi05.high_level.types import ValueResult


def proposal(name: str, index: int = 0) -> ProposalResult:
    return ProposalResult("think", f"memory:{name}", name, TokenConfidence((0.9,), (2.0,)), sample_index=index)


class ProposalMock:
    version = "proposal-test"

    def generate_many(self, contexts, *, samples_per_context, seeds):
        groups = []
        for context in contexts:
            prefix = context.previous_subtask
            names = ["A", "B", "bad"] if not prefix else [f"{prefix}1", f"{prefix}2", "bad"]
            groups.append([proposal(name, idx) for idx, name in enumerate(names[:samples_per_context])])
        return groups


class WorldMock:
    version = "world-test"

    def predict_many(self, requests, *, seeds, profile):
        return [
            PredictedOutcome.invalid("synthetic failure", seed=seed, profile=profile)
            if item.subtask == "bad"
            else PredictedOutcome(np.full((2, 2, 3), len(item.subtask), np.uint8), seed=seed, profile=profile)
            for (_, item), seed in zip(requests, seeds, strict=True)
        ]


class InvalidWorldMock:
    version = "invalid-world-test"

    def predict_many(self, requests, *, seeds, profile):
        return [PredictedOutcome.invalid("all invalid", seed=seed, profile=profile) for seed in seeds]


class ValueMock:
    version = "value-test"
    scores: ClassVar[dict[str, float]] = {"A": 0.9, "B": 0.8, "A1": 0.1, "A2": 0.0, "B1": 0.7, "B2": 0.6}

    def score_many(self, requests):
        return [ValueResult(self.scores[item.subtask]) for _, item, _ in requests]


class RouterSearchTest(unittest.TestCase):
    def test_libero_agentview_is_accepted_as_high_level_head(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        context = HighLevelObservation("episode", 0, 0.0, "task", images={"agentview": image})
        self.assertIs(context.head_image, image)

    def test_cosmos_deploy_profile_is_terminal_image_and_research_is_video(self):
        self.assertEqual(DEFAULT_COSMOS_PROFILES["deploy"].mode, "image")
        self.assertEqual(DEFAULT_COSMOS_PROFILES["deploy"].num_frames, 1)
        self.assertEqual(DEFAULT_COSMOS_PROFILES["research"].mode, "video")
        self.assertGreaterEqual(DEFAULT_COSMOS_PROFILES["research"].num_frames, 24)

    def test_router_uses_or_boundary(self):
        low_probability = ProposalResult("", "", "x", TokenConfidence((0.5,), (5.0,)))
        low_margin = ProposalResult("", "", "x", TokenConfidence((0.9,), (0.5,)))
        confident = ProposalResult("", "", "x", TokenConfidence((0.9,), (5.0,)))
        thresholds = RouterThresholds(0.6, 1.0)
        self.assertTrue(route_proposal(low_probability, thresholds).use_ttc)
        self.assertTrue(route_proposal(low_margin, thresholds).use_ttc)
        self.assertFalse(route_proposal(confident, thresholds).use_ttc)

    def test_beam_search_is_global_and_cumulative(self):
        context = HighLevelObservation("episode", 3, 0.0, "task", images={"head": np.zeros((2, 2, 3))})
        search = WorldModelBeamSearch(
            ProposalMock(), WorldMock(), ValueMock(), SearchConfig(branching_factor=3, beam_width=2, depth=2)
        )
        result = search.run(context)
        self.assertEqual([branch.path for branch in result.beam], [("B", "B1"), ("B", "B2")])
        self.assertAlmostEqual(result.beam[0].cumulative_score, 1.5)
        self.assertEqual(result.expanded_nodes, 9)
        self.assertEqual(result.invalid_nodes, 3)
        self.assertEqual(result.beam[0].memory, "memory:B1")
        # The final outcome belongs to the second semantic step, while the
        # low-level controller must receive the first imagined state.
        self.assertEqual(int(result.beam[0].outcome.terminal_image[0, 0, 0]), len("B1"))
        self.assertEqual(int(result.beam[0].immediate_outcome.terminal_image[0, 0, 0]), len("B"))

    def test_seed_is_reproducible_and_keyed(self):
        first = deterministic_seed("ep", 1, "branch", 2)
        self.assertEqual(first, deterministic_seed("ep", 1, "branch", 2))
        self.assertNotEqual(first, deterministic_seed("ep", 1, "branch", 3))

    def test_all_invalid_frontier_stops_without_reexpansion(self):
        context = HighLevelObservation("episode", 3, 0.0, "task", images={"head": np.zeros((2, 2, 3))})
        search = WorldModelBeamSearch(
            ProposalMock(), InvalidWorldMock(), ValueMock(), SearchConfig(branching_factor=3, beam_width=2, depth=2)
        )
        result = search.run(context)
        self.assertEqual(result.expanded_nodes, 3)
        self.assertEqual(result.invalid_nodes, 3)
        self.assertTrue(all(not branch.outcome.valid for branch in result.beam))


if __name__ == "__main__":
    unittest.main()
