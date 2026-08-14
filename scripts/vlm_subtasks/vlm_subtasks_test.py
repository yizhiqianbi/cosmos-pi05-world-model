import numpy as np
import pytest

from scripts.vlm_subtasks.core import candidate_events
from scripts.vlm_subtasks.core import parse_json_object
from scripts.vlm_subtasks.core import sample_indices
from scripts.vlm_subtasks.core import validate_stages
from scripts.vlm_subtasks.materialize import cosmos_record
from scripts.vlm_subtasks.materialize import output_indices
from scripts.vlm_subtasks.materialize import split_for
from scripts.vlm_subtasks.spatial import Detection
from scripts.vlm_subtasks.spatial import binary_mask_rle
from scripts.vlm_subtasks.spatial import select_detection
from scripts.vlm_subtasks.temporal import apply_refined_boundaries
from scripts.vlm_subtasks.temporal import contextual_grounding_query
from scripts.vlm_subtasks.temporal import global_grounding_indices
from scripts.vlm_subtasks.temporal import local_robot_signals
from scripts.vlm_subtasks.temporal import local_window_indices
from scripts.vlm_subtasks.temporal import select_stable_subgoal_frame
from scripts.vlm_subtasks.temporal import spans_to_boundaries


def test_validate_vlm_stages_requires_exact_coverage():
    raw = {
        "stages": [
            {"label": "pick up mug", "start_frame": 0, "end_frame": 10, "confidence": 0.9},
            {"label": "place mug", "start_frame": 10, "end_frame": 21, "confidence": 0.8},
        ]
    }
    stages = validate_stages(raw, num_frames=21, allowed_boundaries={0, 10, 21})
    assert [stage.label for stage in stages] == ["pick up mug", "place mug"]
    raw["stages"][1]["start_frame"] = 11
    with pytest.raises(ValueError, match="supplied indices|consecutive"):
        validate_stages(raw, num_frames=21, allowed_boundaries={0, 10, 21})


def test_sampling_events_and_json_parsing():
    assert sample_indices(5, 3) == [0, 2, 4]
    actions = np.zeros((6, 7))
    actions[:3, -1] = -1
    actions[3:, -1] = 1
    events = candidate_events(actions, np.zeros((6, 3)), list(range(6)))
    assert any("gripper_transition" in event["signals"] for event in events)
    assert parse_json_object('```json\n{"stages": []}\n```') == {"stages": []}


def test_materialization_helpers_are_deterministic_and_cosmos_compatible():
    indices = output_indices(4, 17)
    assert len(indices) == 61
    assert indices[0] == 4
    assert indices[-1] == 16
    assert split_for("episode_001_002", 10) == split_for("episode_001_002", 10)
    record = cosmos_record("episode_001_002_stage_00", "pick up mug", "put mug on plate")
    assert record["nb_frames"] == 61
    assert record["t2w_windows"][0]["end_frame"] == 60


def test_dense_temporal_refinement_preserves_stage_semantics():
    stages = validate_stages(
        {
            "stages": [
                {"label": "grasp mug", "start_frame": 0, "end_frame": 10, "confidence": 0.9},
                {"label": "place mug", "start_frame": 10, "end_frame": 20, "confidence": 0.8},
            ]
        },
        num_frames=20,
        allowed_boundaries={0, 10, 20},
    )
    refined = apply_refined_boundaries(stages, [12], 20)
    assert refined[0].end_frame == 12
    assert refined[1].start_frame == 12
    assert 10 in local_window_indices(10, 20, radius=4, stride=2)
    actions = np.zeros((20, 7))
    actions[:10, -1] = -1
    actions[10:, -1] = 1
    signals = local_robot_signals(actions, np.zeros((20, 3)), [9, 10])
    assert signals[1]["gripper_transition"]


def test_global_grounding_keeps_gripper_events_and_stable_subgoal_is_in_stage():
    actions = np.zeros((100, 7))
    actions[:40, -1] = -1
    indices = global_grounding_indices(actions, max_frames=24)
    assert 40 in indices
    assert len(indices) <= 24
    ee_pos = np.linspace(0, 1, 300).reshape(100, 3)
    ee_pos[95:] = ee_pos[95]
    images = np.zeros((100, 4, 4, 3), dtype=np.uint8)
    subgoal, detail = select_stable_subgoal_frame(50, 100, actions, ee_pos, images, window=10)
    assert 90 <= subgoal < 100
    assert detail["stability_score"] >= 0.0


def test_native_timelens_spans_form_boundaries_with_context():
    stages = validate_stages(
        {
            "stages": [
                {"label": "turn on stove", "start_frame": 0, "end_frame": 100, "confidence": 0.9},
                {"label": "pick up pot", "start_frame": 100, "end_frame": 200, "confidence": 0.9},
                {"label": "place pot", "start_frame": 200, "end_frame": 272, "confidence": 0.9},
            ]
        },
        num_frames=272,
        allowed_boundaries={0, 100, 200, 272},
    )
    query = contextual_grounding_query("turn on stove and place pot", stages, 1)
    assert "after completing 'turn on stove'" in query
    assert "before beginning 'place pot'" in query
    spans = [
        {"start_seconds": 4.0, "end_seconds": 6.0},
        {"start_seconds": 6.0, "end_seconds": 11.0},
        {"start_seconds": 11.0, "end_seconds": 13.0},
    ]
    assert spans_to_boundaries(spans, 272, 20.0) == [120, 220]


def test_spatial_selection_and_mask_encoding():
    detection = select_detection(
        [
            Detection("mug", 0.4, (1, 2, 3, 4)),
            Detection("mug", 0.9, (5, 6, 7, 8)),
        ]
    )
    assert detection is not None
    assert detection.score == 0.9
    rle = binary_mask_rle(np.array([[0, 1], [1, 1]], dtype=bool))
    assert rle["size"] == [2, 2]
    assert sum(rle["counts"]) == 4
