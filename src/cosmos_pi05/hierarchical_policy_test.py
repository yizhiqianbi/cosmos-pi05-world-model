import numpy as np
import pytest

from cosmos_pi05.hierarchical_policy import HierarchicalPi05Policy
from cosmos_pi05.hierarchical_policy import low_level_prompt
from cosmos_pi05.high_level.types import HighLevelDecision
from cosmos_pi05.high_level.types import RouterDecision


class _LowLevel:
    def __init__(self):
        self.last_observation = None

    def infer(self, observation):
        self.last_observation = observation
        return {"actions": np.zeros((10, 7), dtype=np.float32)}

    def reset(self):
        pass


class _Orchestrator:
    def __init__(self, *, subgoal=True):
        self.subgoal = subgoal
        self.resets = []

    def reset(self, episode_id):
        self.resets.append(episode_id)

    def submit(self, observation):
        return HighLevelDecision(
            episode_id=observation.episode_id,
            sequence_id=observation.sequence_id,
            committed_subtask="pick up the mug",
            memory="",
            route="fast",
            router=RouterDecision(
                use_ttc=False,
                mean_probability=0.9,
                mean_memory_margin=2.0,
                probability_threshold=0.65,
                memory_margin_threshold=1.0,
            ),
            subgoal_image=np.full((32, 32, 3), 123, dtype=np.uint8) if self.subgoal else None,
            subgoal_source="cosmos_direct" if self.subgoal else None,
        )

    def get_state(self, _episode_id):
        return None


def _observation():
    return {
        "episode_id": "episode-1",
        "sequence_id": 0,
        "task_instruction": "put the mug on the plate",
        "observation/image": np.zeros((32, 32, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((32, 32, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
    }


def test_hierarchical_policy_feeds_cosmos_image_to_pi05():
    low = _LowLevel()
    orchestrator = _Orchestrator()
    policy = HierarchicalPi05Policy(low, orchestrator)

    result = policy.infer(_observation())

    assert result["actions"].shape == (10, 7)
    assert result["decision"]["subgoal_source"] == "cosmos_direct"
    assert orchestrator.resets == ["episode-1"]
    np.testing.assert_array_equal(
        low.last_observation["observation/subgoal_image"],
        np.full((32, 32, 3), 123, dtype=np.uint8),
    )
    assert low.last_observation["prompt"] == low_level_prompt("put the mug on the plate", "pick up the mug")


def test_hierarchical_policy_rejects_missing_subgoal():
    policy = HierarchicalPi05Policy(_LowLevel(), _Orchestrator(subgoal=False))
    with pytest.raises(RuntimeError, match="without a Cosmos subgoal image"):
        policy.infer(_observation())
