import numpy as np

from cosmos_pi05.cotrain.libero_joint_dataset import _frame_indices
from cosmos_pi05.cotrain.libero_joint_dataset import _future_indices
from cosmos_pi05.cotrain.libero_joint_dataset import _stage_sort_key


def test_stage_sort_key_matches_converter_order():
    values = ["episode_010_001_stage_02", "episode_002_011_stage_01", "episode_002_003_stage_04"]
    assert sorted(values, key=_stage_sort_key) == [
        "episode_002_003_stage_04",
        "episode_002_011_stage_01",
        "episode_010_001_stage_02",
    ]


def test_future_indices_start_at_row_progress_and_end_at_goal():
    indices = _future_indices(frame_index=50, episode_length=101, video_length=61, output_frames=17)
    assert indices[0] == 30
    assert indices[-1] == 60
    assert np.all(indices[1:] >= indices[:-1])


def test_global_episode_subset_maps_to_unfiltered_frame_indices():
    assert _frame_indices({0: 2, 1: 3, 2: 1, 3: 2}, [1, 3]) == [2, 3, 4, 6, 7]
