"""Validate the tagged public Tool-Star demonstration format."""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolValidation:
    valid: bool
    errors: tuple[str, ...]
    tool_calls: int
    final_answer: str | None


_PAIR_TAGS = ("think", "search", "python", "result", "answer")


def _last_boxed(text: str) -> str | None:
    """Return the last brace-balanced ``\\boxed{...}`` payload."""

    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i].strip()
    return None


def extract_final_answer(text: str) -> str | None:
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if not answers:
        return None
    return _last_boxed(answers[-1]) or answers[-1].strip() or None


def validate_transcript(
    text: str, *, max_tool_calls: int = 3, require_final: bool = True
) -> ProtocolValidation:
    errors: list[str] = []
    lower = text.lower()
    for tag in _PAIR_TAGS:
        opens = lower.count(f"<{tag}>")
        closes = lower.count(f"</{tag}>")
        if opens != closes:
            errors.append(f"unbalanced {tag} tags: {opens} open/{closes} close")

    tokens = list(
        re.finditer(
            r"<(search|python|result|answer)>|</(search|python|result|answer)>", lower
        )
    )
    sequence: list[str] = []
    for token in tokens:
        if token.group(1):
            sequence.append(token.group(1))

    tool_calls = sequence.count("search") + sequence.count("python")
    if tool_calls > max_tool_calls:
        errors.append(f"tool budget exceeded: {tool_calls}>{max_tool_calls}")

    pending = False
    final_seen = False
    for item in sequence:
        if item in {"search", "python"}:
            if pending:
                errors.append("tool action not followed by result before next action")
            if final_seen:
                errors.append("tool action appears after final answer")
            pending = True
        elif item == "result":
            if not pending:
                errors.append("orphan result tag")
            pending = False
        elif item == "answer":
            if pending:
                errors.append("final answer appears before tool result")
            if final_seen:
                errors.append("multiple final answers")
            final_seen = True
    if pending:
        errors.append("last tool action has no result")
    answer = extract_final_answer(text)
    if require_final and answer is None:
        errors.append("missing complete final answer")
    return ProtocolValidation(not errors, tuple(errors), tool_calls, answer)
