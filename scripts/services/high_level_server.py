"""Serve proposal/value/reflection LoRA roles for the hierarchical pi0.5 stack.

The upstream must expose an OpenAI-compatible ``/v1/chat/completions`` API.
All three model names resolve to LoRA adapters sharing one Qwen3.5-9B base.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastapi import FastAPI
from fastapi import HTTPException

from cosmos_pi05.high_level.confidence import proposal_confidence
from cosmos_pi05.high_level.prompts import PROPOSAL_SYSTEM_PROMPT
from cosmos_pi05.high_level.prompts import REFLECTION_SYSTEM_PROMPT
from cosmos_pi05.high_level.prompts import VALUE_SYSTEM_PROMPT
from cosmos_pi05.high_level.prompts import compact_json
from cosmos_pi05.high_level.prompts import fallback_subtask
from cosmos_pi05.high_level.prompts import parse_json_object


class OpenAICompatibleRoles:
    def __init__(
        self,
        base_url: str,
        *,
        proposal_model: str,
        value_model: str,
        reflection_model: str,
        api_key: str = "",
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models = {
            "proposal": proposal_model,
            "value": value_model,
            "reflection": reflection_model,
        }
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat(
        self,
        role: str,
        system: str,
        payload: dict[str, Any],
        images: dict[str, str],
        *,
        seed: int = 0,
        logprobs: bool = False,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for _, encoded in sorted(images.items()):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )
        content.append({"type": "text", "text": compact_json(payload)})
        request_body: dict[str, Any] = {
            "model": self.models[role],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "seed": int(seed),
            "max_tokens": 384,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if logprobs:
            request_body.update({"logprobs": True, "top_logprobs": 2})
        raw = json.dumps(request_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=raw,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"upstream returned HTTP {exc.code}: {detail[:1000]}") from exc
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("upstream returned no choices")
        return choices[0]


class LocalTransformersRoles:
    """Serve all three LoRA roles from one local Qwen3.5 base.

    This backend is intended for a single-machine research run when vLLM is
    unavailable.  The base weights are loaded once and the three adapters are
    switched under a lock, so proposal/value/reflection keep the exact same
    multimodal preprocessing and model contract as the training code.
    """

    def __init__(
        self,
        model_path: str,
        adapter_root: str,
        *,
        device: str = "cuda",
        max_new_tokens: int = 384,
    ) -> None:
        from peft import PeftModel
        import torch
        from transformers import AutoModelForImageTextToText
        from transformers import AutoProcessor

        self._torch = torch
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        base = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        root = Path(adapter_root).resolve()
        self.model = PeftModel.from_pretrained(base, str(root / "proposal" / "adapter"), adapter_name="proposal")
        self.model.load_adapter(str(root / "value" / "adapter"), adapter_name="value")
        self.model.load_adapter(str(root / "reflection" / "adapter"), adapter_name="reflection")
        self.model.eval()
        self.models = {
            "proposal": "local://proposal",
            "value": "local://value",
            "reflection": "local://reflection",
        }
        self._lock = threading.Lock()

    @staticmethod
    def _image(encoded: str):
        from PIL import Image

        value = str(encoded)
        if value.startswith("data:"):
            value = value.split(",", 1)[-1]
        return Image.open(BytesIO(base64.b64decode(value))).convert("RGB")

    def chat(
        self,
        role: str,
        system: str,
        payload: dict[str, Any],
        images: dict[str, str],
        *,
        seed: int = 0,
        logprobs: bool = False,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        torch = self._torch
        slots = [{"type": "image"} for _ in sorted(images)]
        slots.append({"type": "text", "text": compact_json(payload)})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": slots},
        ]
        image_values = [self._image(images[key]) for key in sorted(images)]
        try:
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[rendered], images=image_values, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        batch = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in batch.items()}
        with self._lock, torch.inference_mode():
            self.model.set_adapter(role)
            if seed:
                torch.manual_seed(int(seed))
            output = self.model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=bool(temperature > 0.0),
                temperature=max(float(temperature), 1e-3),
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        generated = output[0][batch["input_ids"].shape[-1] :]
        raw = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        # Transformers generation does not expose vLLM token logprobs.  Use a
        # conservative synthetic confidence for valid local generations; an
        # empty/invalid proposal is handled by the explicit degraded fallback.
        return {
            "message": {"content": raw},
            "logprobs": {
                "content": [
                    {
                        "token": "<local-generation>",
                        "logprob": 0.0,
                        "top_logprobs": [{"logprob": 0.0}, {"logprob": -2.0}],
                    }
                ]
            },
        }


def build_app(backend: OpenAICompatibleRoles) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/proposal/generate")
    async def proposal(body: dict[str, Any]):
        samples = int(body.get("samples_per_context", 1))
        groups = []
        try:
            for context in body.get("contexts", []):
                seeds = [int(value) for value in context.get("seeds", [])]
                if len(seeds) != samples:
                    raise ValueError("each proposal context must contain samples_per_context seeds")
                images = dict(context.get("images_png_b64") or {})
                user = {
                    "task_instruction": context.get("task_instruction", ""),
                    "previous_subtask": context.get("previous_subtask", ""),
                    "execution_memory": context.get("memory", ""),
                    "metadata": context.get("metadata", {}),
                }
                results = []
                for sample_index, seed in enumerate(seeds):
                    choice = backend.chat(
                        "proposal",
                        PROPOSAL_SYSTEM_PROMPT,
                        user,
                        images,
                        seed=seed,
                        logprobs=True,
                        temperature=0.7 if samples > 1 else 0.0,
                    )
                    raw_text = str((choice.get("message") or {}).get("content", ""))
                    parse_error = ""
                    try:
                        parsed = parse_json_object(raw_text)
                    except ValueError as exc:
                        parsed = {}
                        parse_error = str(exc)
                    memory = str(parsed.get("memory", "")).strip()
                    subtask = str(parsed.get("subtask", "")).strip()
                    degraded = False
                    error = parse_error
                    if not subtask:
                        subtask = fallback_subtask(
                            str(context.get("task_instruction", "")),
                            str(context.get("previous_subtask", "")),
                        )
                        if not subtask:
                            raise ValueError(error or "proposal returned an empty subtask and fallback was empty")
                        degraded = True
                        error = error or "proposal returned an empty subtask"
                    probabilities, margins = proposal_confidence(choice, raw_text, memory)
                    if degraded:
                        probabilities, margins = ([], [])
                    results.append(
                        {
                            "thought": str(parsed.get("thought", "")),
                            "memory": memory,
                            "subtask": subtask,
                            "generated_token_probabilities": probabilities,
                            "memory_logit_margins": margins,
                            "raw_text": raw_text,
                            "sample_index": sample_index,
                            "seed": seed,
                            "degraded": degraded,
                            "error": error,
                        }
                    )
                groups.append(results)
            return {"results": groups}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/value/score")
    async def value(body: dict[str, Any]):
        results = []
        try:
            for item in body.get("requests", []):
                choice = backend.chat(
                    "value",
                    VALUE_SYSTEM_PROMPT,
                    {
                        "task_instruction": item.get("task_instruction", ""),
                        "candidate_subtask": item.get("candidate_subtask", ""),
                    },
                    {"terminal": str(item["terminal_image_png_b64"])},
                )
                parsed = parse_json_object(str((choice.get("message") or {}).get("content", "")))
                score = max(-1.0, min(1.0, float(parsed["score"])))
                results.append(
                    {
                        "score": score,
                        "level": str(parsed.get("level", "")),
                        "metadata": {"rationale": str(parsed.get("rationale", ""))},
                    }
                )
            return {"results": results}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/reflection/generate")
    async def reflection(body: dict[str, Any]):
        context = dict(body.get("context") or {})
        images = dict(context.pop("images_png_b64", {}) or {})
        try:
            choice = backend.chat(
                "reflection",
                REFLECTION_SYSTEM_PROMPT,
                {"context": context, "beam": body.get("beam", [])},
                images,
            )
            parsed = parse_json_object(str((choice.get("message") or {}).get("content", "")))
            subtask = str(parsed.get("subtask", "")).strip()
            if not subtask:
                raise ValueError("reflection returned an empty subtask")
            return {"subtask": subtask}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": backend.models}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url", help="OpenAI-compatible Qwen server base URL")
    source.add_argument("--local-base", help="Local Qwen3.5-9B base for Transformers serving")
    parser.add_argument("--local-adapter-root", help="Root containing proposal/value/reflection/adapter")
    parser.add_argument("--local-device", default="cuda")
    parser.add_argument("--local-max-new-tokens", type=int, default=384)
    parser.add_argument("--proposal-model", default="cosmos-pi05-proposal")
    parser.add_argument("--value-model", default="cosmos-pi05-value")
    parser.add_argument("--reflection-model", default="cosmos-pi05-reflection")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10090)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    if args.local_base:
        if not args.local_adapter_root:
            parser.error("--local-adapter-root is required with --local-base")
        backend = LocalTransformersRoles(
            args.local_base,
            args.local_adapter_root,
            device=args.local_device,
            max_new_tokens=args.local_max_new_tokens,
        )
    else:
        backend = OpenAICompatibleRoles(
            args.base_url,
            proposal_model=args.proposal_model,
            value_model=args.value_model,
            reflection_model=args.reflection_model,
            api_key=os.environ.get(args.api_key_env, ""),
            timeout_s=args.timeout_s,
        )
    import uvicorn

    uvicorn.run(build_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
