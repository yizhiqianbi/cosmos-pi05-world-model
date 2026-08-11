"""Stable prompt and structured-output contracts for the three Qwen adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

from .types import HighLevelObservation
from .types import SearchBranch

PROPOSAL_SYSTEM_PROMPT = """You are the proposal policy for a hierarchical robot controller.
Use the visible robot observation and execution memory to select exactly one immediate,
physically executable subtask. Return one JSON object with string fields thought, memory,
and subtask. Memory must be a concise corrected record of completed progress; never record
an imagined action as completed."""

VALUE_SYSTEM_PROMPT = """You are a robot outcome value model. Compare the terminal image
against the task and candidate subtask. Return one JSON object with score in [-1, 1], level
in {failed, unchanged, partial, complete}, and a short rationale."""

REFLECTION_SYSTEM_PROMPT = """You are the reflection policy for a hierarchical robot
controller. Compare the simulated beam branches and select or repair the single safest
immediate subtask. Return one JSON object with string field subtask."""


def proposal_user_payload(context: HighLevelObservation) -> dict[str, Any]:
    return {
        "task_instruction": context.task_instruction,
        "previous_subtask": context.previous_subtask,
        "execution_memory": context.memory,
        "metadata": dict(context.metadata),
    }


def value_user_payload(task: str, candidate_subtask: str) -> dict[str, Any]:
    return {"task_instruction": task, "candidate_subtask": candidate_subtask}


def reflection_user_payload(context: HighLevelObservation, beam: Sequence[SearchBranch]) -> dict[str, Any]:
    return {
        **proposal_user_payload(context),
        "beam": [branch.summary() for branch in beam],
    }


def compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model response, tolerating one Markdown JSON fence."""

    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response does not contain a JSON object") from error
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def fallback_subtask(task: str, previous_subtask: str = "") -> str:
    """Extract a safe first executable clause when a role emits no subtask.

    This is intentionally deterministic and conservative.  It is not a second
    planner: it only splits common two-stage LIBERO instructions (for example,
    ``pick up X and place it on Y``) so the controller can keep the robot in a
    valid degraded mode while preserving the raw model error in the decision.
    """

    text = re.sub(r"\s+", " ", str(task)).strip(" \t\r\n.,;")
    if not text:
        return ""

    previous = re.sub(r"\s+", " ", str(previous_subtask)).strip(" \t\r\n.,;")
    if previous and text.lower().startswith(previous.lower()):
        remainder = text[len(previous) :].lstrip(" ,;:")
        remainder = re.sub(r"^(?:and then|then|and)\s+", "", remainder, flags=re.IGNORECASE)
        if remainder:
            text = remainder

    # Avoid splitting spatial descriptions such as "between A and B".  These
    # markers denote the transition to the second action in LIBERO language.
    markers = (
        r"\s+and\s+(?:then\s+)?(?:place|put|set|move|open|close|turn|release)\b",
        r"\s+then\s+(?:place|put|set|move|open|close|turn|release)\b",
        r",\s*(?:then|afterwards)\s+",
    )
    for marker in markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match and match.start() > 0:
            prefix = text[: match.start()].strip(" \t\r\n.,;")
            if prefix:
                return prefix
    return text
