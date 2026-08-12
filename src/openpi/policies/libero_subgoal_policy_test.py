import numpy as np
from PIL import Image

from openpi.models import model as _model
from openpi.policies import libero_subgoal_policy


def test_three_images_are_native_pi05_slots():
    agent = np.full((12, 16, 3), 11, dtype=np.uint8)
    wrist = np.full((3, 12, 16), 0.25, dtype=np.float32)
    subgoal = np.full((12, 16, 3), 33, dtype=np.uint8)
    transform = libero_subgoal_policy.LiberoSubgoalInputs(model_type=_model.ModelType.PI05)

    result = transform(
        {
            "observation/image": agent,
            "observation/wrist_image": wrist,
            "observation/subgoal_image": subgoal,
            "observation/state": np.arange(8, dtype=np.float32),
            "actions": np.zeros((10, 7), dtype=np.float32),
            "prompt": "test",
        }
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], agent)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], np.full((12, 16, 3), 63, dtype=np.uint8))
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], subgoal)
    assert all(bool(value) for value in result["image_mask"].values())
    assert result["state"].shape == (8,)
    assert result["actions"].shape == (10, 7)


def test_outputs_remove_openpi_action_padding():
    output = libero_subgoal_policy.LiberoSubgoalOutputs()({"actions": np.zeros((10, 32), dtype=np.float32)})
    assert output["actions"].shape == (10, 7)


def test_generated_subgoal_overlay_replaces_selected_episode(tmp_path):
    generated = np.full((8, 9, 3), 77, dtype=np.uint8)
    Image.fromarray(generated).save(tmp_path / "episode_000012.png")
    transform = libero_subgoal_policy.GeneratedSubgoalOverlay(str(tmp_path), probability=1.0, strict=True)

    output = transform(
        {
            "_episode_index": np.asarray(12),
            "_frame_index": np.asarray(3),
            "observation/subgoal_image": np.zeros((8, 9, 3), dtype=np.uint8),
        }
    )

    np.testing.assert_array_equal(output["observation/subgoal_image"], generated)
    assert "_episode_index" not in output
    assert "_frame_index" not in output


def test_generated_subgoal_overlay_can_fall_back_to_oracle(tmp_path):
    oracle = np.full((4, 5, 3), 19, dtype=np.uint8)
    transform = libero_subgoal_policy.GeneratedSubgoalOverlay(str(tmp_path), probability=1.0, strict=False)
    output = transform(
        {
            "_episode_index": np.asarray(2),
            "_frame_index": np.asarray(1),
            "observation/subgoal_image": oracle,
        }
    )
    np.testing.assert_array_equal(output["observation/subgoal_image"], oracle)
