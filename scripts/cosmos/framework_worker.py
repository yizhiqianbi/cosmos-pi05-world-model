#!/usr/bin/env python3
"""Run one or more Cosmos3-Nano image/video requests through the official framework.

This worker is called by ``scripts/services/cosmos_server.py``.  The
framework environment is intentionally separate from openpi's environment, so
the worker launches the official ``cosmos_framework.scripts.inference`` entry
point in the configured Cosmos virtualenv and returns a terminal frame.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMEWORK_ROOT = REPOSITORY_ROOT / "external" / "cosmos-framework"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--framework-root",
        default=os.environ.get("COSMOS_FRAMEWORK_ROOT", str(DEFAULT_FRAMEWORK_ROOT)),
    )
    parser.add_argument(
        "--framework-python",
        default=os.environ.get(
            "COSMOS_FRAMEWORK_PYTHON",
            str(DEFAULT_FRAMEWORK_ROOT / ".venv" / "bin" / "python"),
        ),
    )
    parser.add_argument("--gpus", default=os.environ.get("COSMOS_GPUS", "0,1,2"))
    return parser.parse_args()


def _safe_sample_name(request_id: str, index: int) -> str:
    value = "".join(char if char.isalnum() or char in "-_" else "_" for char in request_id).strip("_")
    return f"sample_{index:03d}_{value[:80] or 'request'}"


def _load_requests(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        return raw
    raise ValueError("request file must contain one request object or a list of request objects")


def main() -> None:
    args = _parse_args()
    requests = _load_requests(Path(args.request))
    if not requests:
        raise ValueError("request file contains no requests")
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    samples: list[tuple[dict, str, str, int]] = []
    for index, request in enumerate(requests):
        name = _safe_sample_name(str(request.get("request_id", "cosmos_request")), index)
        mode = str(request.get("mode", "image")).lower()
        if mode not in {"image", "video"}:
            raise ValueError(f"Unsupported Cosmos mode {mode!r}; expected image or video")
        # Cosmos Framework requires at least 24 frames for 256p video. Its
        # resolved sample may add one conditioning frame, as expected for I2V.
        num_frames = max(24, int(request.get("num_frames", 24))) if mode == "video" else 1
        resolution = str(request.get("resolution", "256"))
        sample = {
            "name": name,
            "model_mode": "image2image" if mode == "image" else "image2video",
            "vision_path": str(Path(request["initial_image"]).resolve()),
            "prompt": str(request["prompt"]),
            "negative_prompt": "blurry, distorted, deformed, low quality, flickering, extra objects",
            "resolution": resolution,
            "aspect_ratio": "1,1",
            "num_steps": int(request.get("denoising_steps", 35)),
            "guidance": 6.0,
            "shift": 10.0,
            "seed": int(request.get("seed", 0)),
            "enable_sound": False,
        }
        if mode == "video":
            sample.update({"num_frames": num_frames, "fps": max(10, int(request.get("fps", 10)))})
        samples.append((request, name, mode, num_frames))
        request["_framework_sample"] = sample
    input_path = output_root / ("framework_input.json" if len(samples) == 1 else "framework_input.jsonl")
    if len(samples) == 1:
        input_path.write_text(json.dumps(samples[0][0]["_framework_sample"], indent=2) + "\n", encoding="utf-8")
    else:
        input_path.write_text(
            "".join(json.dumps(item[0]["_framework_sample"], separators=(",", ":")) + "\n" for item in samples),
            encoding="utf-8",
        )
    framework_output = output_root / "framework_output"
    framework_output.mkdir(parents=True, exist_ok=True)

    gpu_list = [item for item in str(args.gpus).split(",") if item.strip()]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    env["PYTHONPATH"] = args.framework_root
    command = [
        str(Path(args.framework_python).with_name("torchrun")),
        "--standalone",
        "--nproc_per_node",
        str(len(gpu_list)),
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset",
        "throughput",
        "--dp-shard-size",
        str(len(gpu_list)),
        "--dp-replicate-size",
        "1",
        "--cp-size",
        "1",
        "--cfgp-size",
        "1",
        "--no-use-torch-compile",
        "--no-use-cuda-graphs",
        "--no-guardrails",
        "--input-files",
        str(input_path),
        "--output-dir",
        str(framework_output),
        "--checkpoint-path",
        str(Path(args.model).resolve()),
        "--seed",
        str(int(requests[0].get("seed", 0))),
    ]
    completed = subprocess.run(command, cwd=args.framework_root, env=env, check=False, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Cosmos Framework exited with code {completed.returncode}")

    from PIL import Image

    results = []
    for index, (request, name, mode, num_frames) in enumerate(samples):
        sample_dir = framework_output / name
        terminal_path = output_root / f"terminal-{index:03d}.png"
        if mode == "image":
            image_path = sample_dir / "vision.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Cosmos Framework produced no image: {image_path}")
            Image.open(image_path).convert("RGB").save(terminal_path)
            output_media = None
        else:
            video_path = sample_dir / "vision.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"Cosmos Framework produced no video: {video_path}")
            extract = subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-sseof",
                    "-0.1",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-y",
                    str(terminal_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if extract.returncode != 0 or not terminal_path.is_file():
                raise RuntimeError(f"terminal frame extraction failed: {extract.stderr[-1000:]}")
            output_media = video_path
        results.append(
            {
                "request_id": str(request.get("request_id", name)),
                "terminal_image": str(terminal_path),
                "video": str(output_media) if output_media else None,
                "metadata": {
                    "backend": "cosmos-framework",
                    "model": str(Path(args.model).resolve()),
                    "mode": mode,
                    "num_frames": num_frames,
                    "num_steps": int(request.get("denoising_steps", 35)),
                },
            }
        )
    (output_root / "batch_result.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if len(results) == 1:
        (output_root / "result.json").write_text(json.dumps(results[0], indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
