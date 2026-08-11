"""Small JSON/HTTP helpers for independently deployed model services."""

from __future__ import annotations

import base64
import io
import json
from typing import Any
import urllib.error
import urllib.request


class ModelServiceError(RuntimeError):
    pass


def image_to_base64_png(image: Any) -> str:
    if isinstance(image, str):
        return image
    if isinstance(image, bytes | bytearray | memoryview):
        return base64.b64encode(bytes(image)).decode("ascii")
    try:
        import numpy as np
        from PIL import Image

        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if arr.max(initial=0) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:
        raise TypeError(f"Unsupported image payload type {type(image).__name__}: {exc}") from exc


def base64_png_to_image(payload: str) -> Any:
    raw = base64.b64decode(payload)
    try:
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        return raw


def post_json(url: str, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ModelServiceError(f"POST {url} failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ModelServiceError(f"POST {url} returned {type(result).__name__}, expected object")
    return result
