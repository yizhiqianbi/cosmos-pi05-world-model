#!/usr/bin/env python3
"""Serve TimeLens with its official native-video temporal-grounding protocol."""

from __future__ import annotations

import argparse
import re
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from qwen_vl_utils import process_vision_info
import torch
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
import uvicorn

PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)


def extract_span(text: str) -> tuple[float, float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", text.lower())
    if match is None:
        raise ValueError(f"no timestamp span in response: {text!r}")
    start, end = float(match.group(1)), float(match.group(2))
    if end <= start:
        raise ValueError(f"invalid timestamp span: {start}, {end}")
    return start, end


class TimeLensBackend:
    def __init__(self, model_path: str) -> None:
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, padding_side="left", do_resize=False, trust_remote_code=True
        )

    def ground(self, video_path: str, query: str, *, fps: float = 10.0) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 64 * 32 * 32,
                        "total_pixels": 14336 * 32 * 32,
                        "fps": fps,
                    },
                    {"type": "text", "text": PROMPT.format(query)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        videos, metadata = zip(*videos, strict=True)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=list(videos),
            video_metadata=list(metadata),
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to("cuda")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=128
            )
        answer = self.processor.batch_decode(
            [output[0, len(inputs.input_ids[0]) :]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        start, end = extract_span(answer)
        return {"query": query, "answer": answer, "start_seconds": start, "end_seconds": end}


def create_app(backend: TimeLensBackend, model: str) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model, "protocol": "native-video"}

    @app.post("/ground")
    def ground(body: dict[str, Any]) -> dict[str, Any]:
        try:
            return backend.ground(str(body["video_path"]), str(body["query"]), fps=float(body.get("fps", 10.0)))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="TencentARC/TimeLens-8B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(create_app(TimeLensBackend(args.model_path), args.served_model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
