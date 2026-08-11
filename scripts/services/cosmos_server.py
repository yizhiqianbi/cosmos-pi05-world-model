"""Batch HTTP adapter for local Cosmos3-Nano image/video inference.

The NVIDIA inference stack is isolated from pi0.5's Python environment. Use its
vLLM-Omni synchronous endpoint directly, or provide a custom worker command
that writes ``terminal.png`` and optionally ``video.mp4``/``result.json``.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from fastapi import HTTPException


def _safe_request_dir(output_root: Path, request_id: str) -> Path:
    safe_id = "".join(char if char.isalnum() or char in "-_" else "_" for char in request_id)[:100]
    run_dir = output_root / f"{int(time.time() * 1000)}-{safe_id}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    return run_dir


def _multipart_body(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = f"----cosmos-pi05-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="input_reference"; filename="{file_path.name}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


class VllmOmniCosmosBackend:
    """Direct adapter for the release-tested vLLM-Omni synchronous I2V API.

    vLLM-Omni exposes a video endpoint, so image2image requests must use the
    bundled official-framework backend instead.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        model_path: str | Path,
        output_root: str | Path,
        timeout_s: float = 900.0,
        ffmpeg: str = "ffmpeg",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_path = str(Path(model_path).resolve())
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.ffmpeg = ffmpeg
        negative_path = Path(self.model_path) / "assets" / "negative_prompt.json"
        self.negative_prompt = negative_path.read_text(encoding="utf-8") if negative_path.is_file() else ""

    @staticmethod
    def _target_size(image_path: Path, short_side: int) -> tuple[int, int]:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        if width >= height:
            target_height = short_side
            target_width = max(16, round(width / height * short_side / 16) * 16)
        else:
            target_width = short_side
            target_height = max(16, round(height / width * short_side / 16) * 16)
        return target_width, target_height

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        request_id = str(request.get("request_id") or uuid.uuid4())
        if str(request.get("mode", "video")).lower() == "image":
            return {
                "valid": False,
                "request_id": request_id,
                "error": "vLLM-Omni sync endpoint is video-only; use --framework-worker for Cosmos image2image",
            }
        run_dir = _safe_request_dir(self.output_root, request_id)
        source_path = run_dir / "initial-source.png"
        source_path.write_bytes(base64.b64decode(str(request["initial_image_png_b64"]), validate=True))
        resolution = int(request.get("resolution", 256))
        width, height = self._target_size(source_path, resolution)
        input_path = run_dir / "initial.png"
        with Image.open(source_path) as image:
            image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(input_path)
        fields = {
            "prompt": str(request["prompt"]),
            "negative_prompt": self.negative_prompt,
            "size": f"{width}x{height}",
            "num_frames": str(int(request.get("num_frames", 17))),
            "fps": str(int(request.get("fps", 4))),
            "num_inference_steps": str(int(request.get("denoising_steps", 35))),
            "guidance_scale": "6.0",
            "max_sequence_length": "4096",
            "flow_shift": "10.0",
            "seed": str(int(request.get("seed", 0))),
            "extra_params": json.dumps(
                {"use_resolution_template": False, "use_duration_template": False, "guardrails": True}
            ),
        }
        body, boundary = _multipart_body(fields, input_path)
        http_request = urllib.request.Request(
            f"{self.endpoint}/v1/videos/sync",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "video/mp4"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                video = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {"valid": False, "error": f"vLLM-Omni HTTP {exc.code}: {detail[:2000]}"}
        video_path = run_dir / "video.mp4"
        video_path.write_bytes(video)
        terminal_path = run_dir / "terminal.png"
        completed = subprocess.run(
            [
                self.ffmpeg,
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
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not terminal_path.is_file():
            return {"valid": False, "error": f"terminal-frame extraction failed: {completed.stderr[-1000:]}"}
        return {
            "valid": True,
            "request_id": request_id,
            "terminal_image_png_b64": base64.b64encode(terminal_path.read_bytes()).decode("ascii"),
            "video_uri": video_path.as_uri(),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "worker_metadata": {"backend": "vllm-omni", "size": f"{width}x{height}"},
        }


class DiffusersCosmosBackend:
    """Resident Cosmos3-Nano backend using ``Cosmos3OmniPipeline``.

    The Diffusers Cosmos3 pipeline does not condition text-to-image generation
    on an input image when ``num_frames == 1``.  For the deploy profile we
    therefore run the shortest useful image-conditioned rollout (five pixel
    frames: one clean conditioning frame plus one temporally-compressed future
    latent) and return its last frame as the terminal subgoal image.  This is a
    real image-conditioned 35-step Cosmos rollout, not an unconditioned T2I
    approximation.

    The pipeline is kept resident so a controller episode does not reload the
    35 GB checkpoint for every search candidate.  ``device_map=balanced`` also
    lets Accelerate place the model across the CUDA devices visible to this
    process, which is useful on a shared node.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        output_root: str | Path,
        device_map: str = "balanced",
        max_memory_gib: int = 0,
        deploy_num_frames: int = 5,
    ) -> None:
        from diffusers import Cosmos3OmniPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        import torch

        self.model_path = str(Path(model_path).resolve())
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.deploy_num_frames = max(5, int(deploy_num_frames))
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": device_map,
            "sound_tokenizer": None,
            "enable_safety_checker": False,
        }
        if max_memory_gib > 0:
            load_kwargs["max_memory"] = {
                **dict.fromkeys(range(torch.cuda.device_count()), f"{max_memory_gib}GiB"),
                "cpu": "512GiB",
            }
        self.pipeline = Cosmos3OmniPipeline.from_pretrained(self.model_path, **load_kwargs)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config, flow_shift=10.0)
        self._torch = torch
        negative_path = Path(self.model_path) / "assets" / "negative_prompt.json"
        if negative_path.is_file():
            raw_negative = json.loads(negative_path.read_text(encoding="utf-8"))
            self.negative_prompt = json.dumps(raw_negative, ensure_ascii=False)
        else:
            self.negative_prompt = "blurry, distorted, deformed, low quality, extra objects"

    @staticmethod
    def _square_image(encoded: str, resolution: int):
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
        return image.resize((resolution, resolution), Image.Resampling.LANCZOS)

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        from diffusers.utils import export_to_video

        request_id = str(request.get("request_id") or uuid.uuid4())
        run_dir = _safe_request_dir(self.output_root, request_id)
        resolution = max(128, int(request.get("resolution", 256)))
        initial = self._square_image(str(request["initial_image_png_b64"]), resolution)
        initial_path = run_dir / "initial.png"
        initial.save(initial_path)
        requested_mode = str(request.get("mode", "image")).lower()
        if requested_mode not in {"image", "video"}:
            return {"valid": False, "request_id": request_id, "error": f"unsupported mode {requested_mode!r}"}
        num_frames = (
            self.deploy_num_frames
            if requested_mode == "image"
            else max(self.deploy_num_frames, int(request.get("num_frames", 24)))
        )
        fps = max(1, int(request.get("fps", 10)))
        steps = int(request.get("denoising_steps", 35))
        seed = int(request.get("seed", 0))
        started = time.perf_counter()
        try:
            result = self.pipeline(
                prompt=str(request["prompt"]),
                negative_prompt=self.negative_prompt,
                image=initial,
                num_frames=num_frames,
                height=resolution,
                width=resolution,
                fps=float(fps),
                num_inference_steps=steps,
                guidance_scale=6.0,
                generator=self._torch.Generator(device="cpu").manual_seed(seed),
                enable_sound=False,
                output_type="pil",
                enable_safety_check=False,
            )
            frames = list(result.video)
            if not frames:
                raise RuntimeError("Cosmos3 returned no frames")
            terminal_path = run_dir / "terminal.png"
            frames[-1].convert("RGB").save(terminal_path)
            video_path = None
            if requested_mode == "video":
                video_path = run_dir / "video.mp4"
                export_to_video(frames, str(video_path), fps=fps)
        except Exception as exc:
            return {"valid": False, "request_id": request_id, "error": f"Diffusers Cosmos3 failed: {exc}"}
        elapsed_ms = (time.perf_counter() - started) * 1000
        metadata = {
            "backend": "diffusers-cosmos3",
            "requested_mode": requested_mode,
            "effective_mode": "image2video-terminal",
            "num_frames": num_frames,
            "num_steps": steps,
            "resolution": resolution,
            "seed": seed,
        }
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "terminal_image": str(terminal_path),
                    "video": str(video_path) if video_path else None,
                    "latency_ms": elapsed_ms,
                    "metadata": metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "valid": True,
            "request_id": request_id,
            "terminal_image_png_b64": base64.b64encode(terminal_path.read_bytes()).decode("ascii"),
            "video_uri": video_path.as_uri() if video_path else None,
            "latency_ms": elapsed_ms,
            "worker_metadata": metadata,
        }

    def generate_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Cosmos3OmniPipeline currently documents one sample per call. Keep the
        # model resident and evaluate the search batch deterministically in
        # request order.
        return [self.generate(request) for request in requests]


class CommandCosmosBackend:
    def __init__(
        self,
        command: str,
        *,
        model_path: str,
        output_root: str | Path,
        timeout_s: float = 900.0,
    ) -> None:
        self.command = shlex.split(command)
        if not self.command:
            raise ValueError("worker command cannot be empty")
        self.model_path = str(Path(model_path).resolve())
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = str(request.get("request_id") or uuid.uuid4())
        run_dir = _safe_request_dir(self.output_root, request_id)
        image_path = run_dir / "initial.png"
        image_path.write_bytes(base64.b64decode(str(request["initial_image_png_b64"]), validate=True))
        worker_request = {
            **{key: value for key, value in request.items() if key != "initial_image_png_b64"},
            "initial_image": str(image_path),
            "generate_audio": False,
        }
        request_path = run_dir / "request.json"
        request_path.write_text(json.dumps(worker_request, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        completed = subprocess.run(
            [
                *self.command,
                "--model",
                self.model_path,
                "--request",
                str(request_path),
                "--output-dir",
                str(run_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "worker failed")[-2000:]
            return {"valid": False, "error": detail, "request_id": request_id}

        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        terminal_path = Path(result.get("terminal_image", run_dir / "terminal.png"))
        if not terminal_path.is_absolute():
            terminal_path = run_dir / terminal_path
        if not terminal_path.is_file():
            return {"valid": False, "error": "worker produced no terminal image", "request_id": request_id}
        video_value = result.get("video")
        video_path = Path(video_value) if video_value else run_dir / "video.mp4"
        if not video_path.is_absolute():
            video_path = run_dir / video_path
        return {
            "valid": True,
            "request_id": request_id,
            "terminal_image_png_b64": base64.b64encode(terminal_path.read_bytes()).decode("ascii"),
            "video_uri": video_path.as_uri() if video_path.is_file() else None,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "worker_metadata": result.get("metadata", {}),
        }

    def generate_many(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run a request batch in one framework process/model load.

        Beam search naturally submits all candidates for one depth together.
        Keeping that batch intact avoids reloading the 35G Cosmos checkpoint
        once per candidate.
        """

        if not requests:
            return []
        run_dir = _safe_request_dir(self.output_root, "batch")
        worker_requests = []
        for index, request in enumerate(requests):
            image_path = run_dir / f"initial-{index:03d}.png"
            image_path.write_bytes(base64.b64decode(str(request["initial_image_png_b64"]), validate=True))
            worker_requests.append(
                {
                    **{key: value for key, value in request.items() if key != "initial_image_png_b64"},
                    "initial_image": str(image_path),
                    "generate_audio": False,
                }
            )
        request_path = run_dir / "requests.json"
        request_path.write_text(json.dumps(worker_requests, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [
                    *self.command,
                    "--model",
                    self.model_path,
                    "--request",
                    str(request_path),
                    "--output-dir",
                    str(run_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            error = f"Cosmos worker timed out after {self.timeout_s:.1f}s"
            return [{"valid": False, "request_id": item.get("request_id"), "error": error} for item in requests]
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "worker failed")[-2000:]
            return [{"valid": False, "request_id": item.get("request_id"), "error": detail} for item in requests]
        result_path = run_dir / "batch_result.json"
        try:
            results = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            error = f"worker produced invalid batch_result.json: {exc}"
            return [{"valid": False, "request_id": item.get("request_id"), "error": error} for item in requests]
        if not isinstance(results, list) or len(results) != len(requests):
            error = f"worker produced {len(results) if isinstance(results, list) else 'invalid'} batch results"
            return [{"valid": False, "request_id": item.get("request_id"), "error": error} for item in requests]
        elapsed_ms = (time.perf_counter() - started) * 1000
        outcomes = []
        for request, result in zip(requests, results, strict=True):
            if not isinstance(result, dict):
                outcomes.append({"valid": False, "error": "invalid worker result"})
                continue
            terminal_path = Path(result.get("terminal_image", ""))
            if not terminal_path.is_absolute():
                terminal_path = run_dir / terminal_path
            if not terminal_path.is_file():
                outcomes.append(
                    {
                        "valid": False,
                        "request_id": request.get("request_id"),
                        "error": "worker produced no terminal image",
                    }
                )
                continue
            video_value = result.get("video")
            video_path = Path(video_value) if video_value else run_dir / "video.mp4"
            if not video_path.is_absolute():
                video_path = run_dir / video_path
            outcomes.append(
                {
                    "valid": True,
                    "request_id": str(request.get("request_id") or result.get("request_id") or uuid.uuid4()),
                    "terminal_image_png_b64": base64.b64encode(terminal_path.read_bytes()).decode("ascii"),
                    "video_uri": video_path.as_uri() if video_path.is_file() else None,
                    "latency_ms": elapsed_ms,
                    "worker_metadata": result.get("metadata", {}),
                }
            )
        return outcomes


def build_app(backend: Any, *, max_parallel: int = 1) -> FastAPI:
    app = FastAPI()
    pool = ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="cosmos3-nano")

    @app.post("/v1/world/predict")
    async def predict(body: dict[str, Any]):
        requests = body.get("requests")
        if not isinstance(requests, list):
            raise HTTPException(status_code=400, detail="requests must be a list")
        try:
            if hasattr(backend, "generate_many"):
                future = pool.submit(backend.generate_many, [dict(item) for item in requests])
                outcomes = future.result()
            else:
                futures = [pool.submit(backend.generate, dict(item)) for item in requests]
                outcomes = [future.result() for future in futures]
            return {"outcomes": outcomes}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_path": backend.model_path, "max_parallel": max_parallel}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Cosmos3-Nano snapshot")
    backend_group = parser.add_mutually_exclusive_group(required=True)
    backend_group.add_argument("--vllm-omni-url", help="Cosmos vLLM-Omni base URL, for example http://127.0.0.1:8000")
    backend_group.add_argument("--worker-command", help="Quoted command for a custom NVIDIA inference wrapper")
    backend_group.add_argument(
        "--diffusers",
        action="store_true",
        help="Load Cosmos3OmniPipeline once in this process and return terminal frames from short I2V rollouts",
    )
    backend_group.add_argument(
        "--framework-worker",
        action="store_true",
        help="Use the bundled official Cosmos Framework worker (no vLLM-Omni required)",
    )
    parser.add_argument("--cosmos-gpus", default="0,1,2", help="CUDA devices for --framework-worker")
    parser.add_argument("--diffusers-device-map", default="balanced")
    parser.add_argument("--diffusers-max-memory-gib", type=int, default=0)
    parser.add_argument("--diffusers-deploy-num-frames", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/cosmos3_nano_generations")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10091)
    args = parser.parse_args()
    if args.vllm_omni_url:
        backend = VllmOmniCosmosBackend(
            args.vllm_omni_url,
            model_path=args.model,
            output_root=args.output_root,
            timeout_s=args.timeout_s,
        )
    elif args.diffusers:
        backend = DiffusersCosmosBackend(
            args.model,
            output_root=args.output_root,
            device_map=args.diffusers_device_map,
            max_memory_gib=args.diffusers_max_memory_gib,
            deploy_num_frames=args.diffusers_deploy_num_frames,
        )
    elif args.framework_worker:
        os.environ["COSMOS_GPUS"] = args.cosmos_gpus
        worker = Path(__file__).resolve().parents[2] / "scripts/cosmos/framework_worker.py"
        backend = CommandCosmosBackend(
            f"{sys.executable} {worker}",
            model_path=args.model,
            output_root=args.output_root,
            timeout_s=args.timeout_s,
        )
    elif args.worker_command:
        backend = CommandCosmosBackend(
            args.worker_command,
            model_path=args.model,
            output_root=args.output_root,
            timeout_s=args.timeout_s,
        )
    else:
        parser.error("one of --vllm-omni-url, --worker-command, or --framework-worker is required")
    import uvicorn

    uvicorn.run(build_app(backend, max_parallel=args.max_parallel), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
