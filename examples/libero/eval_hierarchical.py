#!/usr/bin/env python3
"""Evaluate the complete Qwen -> Cosmos -> pi0.5 stack on LIBERO-Long."""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
from pathlib import Path
import time

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tyro

DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 50
    max_steps: int = 520
    wait_steps: int = 10
    replan_steps: int = 5
    resize_size: int = 224
    seed: int = 7
    output_dir: Path = Path("outputs/libero_long_hierarchical")
    save_videos: bool = True
    resume: bool = True


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] * 2.0 * math.acos(quat[3]) / denominator, dtype=np.float32)


def _state(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]],
        axis=0,
    ).astype(np.float32)


def _image(image: np.ndarray, size: int) -> np.ndarray:
    # LIBERO stores demonstrations with this camera orientation.
    image = np.ascontiguousarray(np.asarray(image)[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, size, size))


def _prepare_libero_config() -> None:
    """Avoid LIBERO's interactive first-import prompt in headless evaluation."""
    if "LIBERO_CONFIG_PATH" in os.environ:
        return
    config_dir = REPOSITORY_ROOT / "outputs" / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    benchmark_root = REPOSITORY_ROOT / "third_party" / "libero" / "libero" / "libero"
    config_file.write_text(
        json.dumps(
            {
                "benchmark_root": str(benchmark_root),
                "bddl_files": str(benchmark_root / "bddl_files"),
                "init_states": str(benchmark_root / "init_files"),
                "datasets": str(benchmark_root.parent / "datasets"),
                "assets": str(benchmark_root / "assets"),
            },
            indent=2,
        )
        + "\n"
    )


def _environment(task, *, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _episode_path(output: Path, task_id: int, episode_idx: int) -> Path:
    return output / "episodes" / f"task_{task_id:02d}_episode_{episode_idx:02d}.json"


def evaluate(args: Args) -> dict:
    _prepare_libero_config()
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise RuntimeError(
            "LIBERO evaluation dependencies are isolated from the training environment. "
            "Run scripts/setup_libero_eval_env.sh once, then use scripts/run_libero_eval.sh."
        ) from exc

    if args.task_suite_name != "libero_10":
        raise ValueError("This evaluator is intentionally scoped to LIBERO-Long / libero_10")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "episodes").mkdir(exist_ok=True)
    if args.save_videos:
        (args.output_dir / "videos").mkdir(exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if not metadata.get("requires_subgoal_image"):
        raise RuntimeError("Connected server is not the Cosmos-subgoal pi0.5 policy")

    records: list[dict] = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        instruction = str(task.language)
        initial_states = suite.get_task_init_states(task_id)
        if args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"Requested {args.num_trials_per_task} trials but task {task_id} has {len(initial_states)} states"
            )
        env = _environment(task, resolution=256, seed=args.seed)
        try:
            for episode_idx in range(args.num_trials_per_task):
                target = _episode_path(args.output_dir, task_id, episode_idx)
                if args.resume and target.is_file():
                    records.append(json.loads(target.read_text()))
                    continue

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(DUMMY_ACTION)

                episode_id = f"libero-long-{task_id:02d}-{episode_idx:02d}"
                action_plan: collections.deque[np.ndarray] = collections.deque()
                trace: list[dict] = []
                replay: list[np.ndarray] = []
                sequence_id = 0
                success = False
                started = time.monotonic()
                step = 0

                while step < args.max_steps and not success:
                    agent = _image(obs["agentview_image"], args.resize_size)
                    wrist = _image(obs["robot0_eye_in_hand_image"], args.resize_size)
                    replay.append(agent)
                    if not action_plan:
                        result = client.infer(
                            {
                                "episode_id": episode_id,
                                "sequence_id": sequence_id,
                                "timestamp_s": time.time(),
                                "task_instruction": instruction,
                                "observation/image": agent,
                                "observation/wrist_image": wrist,
                                "observation/state": _state(obs),
                                "prompt": instruction,
                            }
                        )
                        actions = np.asarray(result["actions"], dtype=np.float32)
                        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                            raise ValueError(f"Invalid action chunk {actions.shape}")
                        if len(actions) < args.replan_steps:
                            raise ValueError(
                                f"Policy horizon {len(actions)} is shorter than replan_steps={args.replan_steps}"
                            )
                        decision = dict(result.get("decision", {}))
                        trace.append(
                            {
                                "sequence_id": sequence_id,
                                "step": step,
                                "committed_subtask": result.get("committed_subtask"),
                                "route": decision.get("route"),
                                "router": decision.get("router"),
                                "subgoal_source": decision.get("subgoal_source"),
                                "has_subgoal_image": bool(decision.get("has_subgoal_image")),
                                "degraded": bool(decision.get("degraded", False)),
                                "error": decision.get("error"),
                                "beam": decision.get("beam", []),
                                "actions_shape": list(actions.shape),
                                "action_min": float(actions.min()),
                                "action_max": float(actions.max()),
                            }
                        )
                        action_plan.extend(actions[: args.replan_steps])
                        sequence_id += 1

                    obs, _, done, _ = env.step(action_plan.popleft().tolist())
                    step += 1
                    success = bool(done or env.check_success())

                record = {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "episode_id": episode_id,
                    "instruction": instruction,
                    "success": success,
                    "steps": step,
                    "policy_queries": len(trace),
                    "elapsed_s": time.monotonic() - started,
                    "trace": trace,
                }
                target.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                records.append(record)
                if args.save_videos and replay:
                    suffix = "success" if success else "failure"
                    imageio.mimwrite(
                        args.output_dir / "videos" / f"task_{task_id:02d}_episode_{episode_idx:02d}_{suffix}.mp4",
                        replay,
                        fps=20,
                    )
                logging.info(
                    "task=%d episode=%d success=%s steps=%d aggregate=%d/%d",
                    task_id,
                    episode_idx,
                    success,
                    step,
                    sum(int(item["success"]) for item in records),
                    len(records),
                )
        finally:
            env.close()

    by_task = {}
    for task_id in range(suite.n_tasks):
        subset = [record for record in records if record["task_id"] == task_id]
        by_task[str(task_id)] = {
            "instruction": str(suite.get_task(task_id).language),
            "successes": sum(int(record["success"]) for record in subset),
            "episodes": len(subset),
            "success_rate": sum(int(record["success"]) for record in subset) / len(subset) if subset else 0.0,
        }
    successes = sum(int(record["success"]) for record in records)
    summary = {
        "schema_version": "cosmos-pi05.libero-eval.v1",
        "suite": args.task_suite_name,
        "successes": successes,
        "episodes": len(records),
        "success_rate": successes / len(records) if records else 0.0,
        "by_task": by_task,
        "server_metadata": metadata,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    print(json.dumps(evaluate(tyro.cli(Args)), indent=2, ensure_ascii=False))
