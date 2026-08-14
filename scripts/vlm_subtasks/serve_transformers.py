#!/usr/bin/env python3
"""Serve one local open-weight VLM through a minimal OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from PIL import Image
import torch
from transformers import AutoModelForMultimodalLM
from transformers import AutoProcessor
import uvicorn


def decode_data_image(url: str) -> Image.Image:
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("only base64 data image URLs are supported")
    return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


class LocalVLM:
    def __init__(self, model_path: str, *, max_new_tokens: int = 1200) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, body: dict[str, Any]) -> str:
        messages = []
        for message in body.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, list):
                converted = []
                for item in content:
                    if item.get("type") == "image_url":
                        converted.append({"type": "image", "image": decode_data_image(item["image_url"]["url"])})
                    elif item.get("type") == "text":
                        converted.append({"type": "text", "text": str(item.get("text", ""))})
                content = converted
            messages.append({"role": str(message.get("role", "user")), "content": content})
        try:
            batch = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            batch = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
            )
        batch = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in batch.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **batch,
                max_new_tokens=min(int(body.get("max_tokens", self.max_new_tokens)), self.max_new_tokens),
                do_sample=False,
            )
        return self.processor.decode(output[0, batch["input_ids"].shape[1] :], skip_special_tokens=True)


def create_app(backend: LocalVLM, served_model: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": served_model}

    @app.post("/v1/chat/completions")
    def completions(body: dict[str, Any]) -> dict[str, Any]:
        requested = str(body.get("model", ""))
        if requested and requested not in {served_model, "local"}:
            raise HTTPException(status_code=404, detail=f"model not served: {requested}")
        try:
            content = backend.generate(body)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return {
            "id": "local-vlm",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    args = parser.parse_args()
    uvicorn.run(
        create_app(LocalVLM(args.model_path, max_new_tokens=args.max_new_tokens), args.served_model),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
