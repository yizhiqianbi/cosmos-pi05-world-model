import json

from scripts.cotrain.common import image_to_uint8
from scripts.cotrain.common import split_task_prompt
from scripts.cotrain.common import write_jsonl
from scripts.cotrain.prepare_cosmos_replay_dataset import Args
from scripts.cotrain.prepare_cosmos_replay_dataset import prepare
from scripts.cotrain.score_subgoal_replay import Args as ScoreArgs
from scripts.cotrain.score_subgoal_replay import score
from scripts.cotrain.train_pi05_round import Args as TrainRoundArgs
from scripts.cotrain.train_pi05_round import _config


def _record(uuid: str) -> dict:
    return {
        "uuid": uuid,
        "duration": 5.0,
        "width": 256,
        "height": 256,
        "vision_path": f"videos/{uuid}.mp4",
        "t2w_windows": [{"start_frame": 0, "end_frame": 60, "caption": uuid}],
    }


def test_prompt_and_image_contract():
    task, subtask = split_task_prompt("Full task: put mug down\nCurrent executable subtask: pick up mug")
    assert task == "put mug down"
    assert subtask == "pick up mug"
    assert image_to_uint8([[[0.5]], [[0.5]], [[0.5]]]).shape == (1, 1, 3)


def test_hard_replay_has_unique_oversample_ids(tmp_path):
    source = tmp_path / "source"
    (source / "train" / "videos").mkdir(parents=True)
    (source / "val" / "videos").mkdir(parents=True)
    train = [_record("episode_000_000_stage_00"), _record("episode_000_000_stage_01")]
    val = [_record("episode_000_001_stage_00")]
    write_jsonl(source / "train" / "video_dataset_file.jsonl", train)
    write_jsonl(source / "val" / "video_dataset_file.jsonl", val)
    scores = tmp_path / "scores.jsonl"
    write_jsonl(
        scores,
        [
            {"cosmos_uuid": train[0]["uuid"], "split": "train", "quality_score": 0.1},
            {"cosmos_uuid": train[1]["uuid"], "split": "train", "quality_score": 0.9},
        ],
    )
    output = tmp_path / "output"

    summary = prepare(
        Args(
            source_dataset=source,
            scored_replay=scores,
            output_dir=output,
            hard_fraction=0.5,
            hard_repeat=3,
        )
    )

    records = [json.loads(line) for line in (output / "train" / "video_dataset_file.jsonl").read_text().splitlines()]
    assert summary["hard_uuids"] == [train[0]["uuid"]]
    assert len(records) == 4
    assert len({record["uuid"] for record in records}) == 4
    assert (output / "train" / "videos").is_symlink()


def test_pi_round_and_scorer_dry_run_contracts(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    generated = replay_dir / "episode_000000.png"
    generated.write_bytes(b"generated-image-placeholder")
    replay_manifest = replay_dir / "replay.jsonl"
    write_jsonl(
        replay_manifest,
        [{"episode_index": 0, "dataset_index": 0, "generated_subgoal": str(generated)}],
    )

    config = _config(
        TrainRoundArgs(
            base_checkpoint=str(checkpoint),
            generated_subgoal_dir=replay_dir,
            exp_name="round_test",
            num_train_steps=3,
            batch_size=4,
            dry_run=True,
        )
    )
    assert config.data.generated_subgoal_dir == str(replay_dir.resolve())
    assert config.num_train_steps == 3
    summary = score(
        ScoreArgs(
            checkpoint_dir=str(checkpoint),
            replay_manifest=replay_manifest,
            output_path=tmp_path / "scores.jsonl",
            dry_run=True,
        )
    )
    assert summary["samples"] == 1
