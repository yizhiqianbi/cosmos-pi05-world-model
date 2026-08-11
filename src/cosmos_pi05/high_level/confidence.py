"""Token-confidence extraction shared by serving and offline calibration."""

from __future__ import annotations

import json
import math
from typing import Any


def proposal_confidence(choice: dict[str, Any], raw_text: str, memory: str) -> tuple[list[float], list[float]]:
    tokens = (choice.get("logprobs") or {}).get("content") or []
    probabilities: list[float] = []
    memory_margins: list[float] = []
    fallback_margins: list[float] = []

    memory_literal = json.dumps(memory, ensure_ascii=False)
    memory_start = raw_text.find(memory_literal)
    memory_end = memory_start + len(memory_literal) if memory_start >= 0 else -1
    offset = 0
    for item in tokens:
        token = str(item.get("token", ""))
        token_start, token_end = offset, offset + len(token)
        offset = token_end
        logprob = float(item.get("logprob", -math.inf))
        probabilities.append(math.exp(logprob) if math.isfinite(logprob) else 0.0)
        alternatives = sorted(
            (float(value.get("logprob", -math.inf)) for value in item.get("top_logprobs", [])),
            reverse=True,
        )
        if len(alternatives) >= 2:
            margin = alternatives[0] - alternatives[1]
            fallback_margins.append(margin)
            if memory_start >= 0 and token_end > memory_start and token_start < memory_end:
                memory_margins.append(margin)
    return probabilities, memory_margins or fallback_margins
