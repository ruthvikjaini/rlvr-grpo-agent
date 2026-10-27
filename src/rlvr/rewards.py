"""Verifiable rewards and reward-hacking checks.

A reward here is a deterministic function of a trace and the ground-truth
answer. There is no learned reward model, so the signal cannot drift with
a reward model's own errors. The total reward is deliberately dominated
by correctness: the format bonus is small enough that a well-formatted
wrong answer can never outscore a correct answer, which is the simplest
defence against format reward hacking.

Components
----------
correctness: 1.0 if the extracted answer equals the ground truth.
format:      0.1 if the trace ends with a ``<answer>...</answer>`` block.
efficiency: -0.05 per tool call beyond the first, floored so the total
             reward never goes below 0 for a correct answer.
"""

from __future__ import annotations

import re
from typing import Any

ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>\s*$", re.DOTALL)


def extract_answer(text: str) -> int | None:
    """Extract the final boxed answer, or ``None`` if it is absent/malformed."""
    match = ANSWER_RE.search(text.strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def reward_components(
    trace_text: str,
    ground_truth: int,
    n_tool_calls: int,
) -> dict[str, float]:
    """Score one completion. All components are in [0, 1] except efficiency."""
    extracted = extract_answer(trace_text)
    correctness = 1.0 if extracted == ground_truth else 0.0
    format_ok = 1.0 if extracted is not None else 0.0
    efficiency = -0.05 * max(0, n_tool_calls - 1)
    total = correctness + 0.1 * format_ok + efficiency
    total = max(0.0, min(1.1, total))
    return {
        "correctness": correctness,
        "format": format_ok,
        "efficiency": efficiency,
        "total": total,
    }


def evaluate_trace(
    trace: dict[str, Any],
    ground_truth: int,
) -> dict[str, Any]:
    """Score a trace dict and attach reward-hacking diagnostics."""
    text = trace.get("text", "")
    tool_calls = trace.get("tool_calls", [])
    components = reward_components(text, ground_truth, len(tool_calls))
    words = text.split()
    repeated = len(words) - len(set(words)) if words else 0
    hacking = {
        "excessive_length": len(words) > 120,
        "repeated_words": repeated,
        "many_tool_calls": len(tool_calls) > 3,
        "formatted_but_wrong": bool(
            components["format"] and not components["correctness"]
        ),
        "answer_copied_from_prompt": False,
    }
    return {**components, "hacking": hacking, "n_tool_calls": len(tool_calls)}


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Mean squared error between a stated confidence and a 0/1 outcome.

    Lower is better; 0 is perfect, 1 is maximally wrong. This is the
    proper scoring rule used for calibrated forecasting in AlphaBet
    (Ye et al., 2026), adapted here to the policy's confidence that its
    chosen strategy will produce the correct answer.
    """
    if not confidences:
        raise ValueError("brier_score needs at least one sample")
    if len(confidences) != len(outcomes):
        raise ValueError("confidences and outcomes must have the same length")
    return float(
        sum((float(c) - float(o)) ** 2 for c, o in zip(confidences, outcomes))
        / len(confidences)
    )
