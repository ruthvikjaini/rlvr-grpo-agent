"""Tool-using agent loop.

The agent receives a strategy from the policy and executes it against
the tools in :mod:`rlvr.tools`, respecting a tool-call budget. The
strategy determines *how* the answer is produced; the reward in
:mod:`rlvr.rewards` determines whether it was right. This separation
is the harness side of agentic RL: recent October 2026 work
(HarnessAdapt-RL, Self-HPO) shows the harness and the policy must be
evaluated together, because a policy trained under one harness does
not automatically transfer to another.

Execution noise is explicit and seeded: a good strategy can still slip
(arithmetic slip, 15%) and a poor strategy can occasionally succeed by
luck (5%). That keeps every group from being all-equal too often,
which would carry no GRPO gradient, and it mirrors the stochasticity
of real LLM rollouts without hiding it inside the reward function.
"""

from __future__ import annotations

import numpy as np

from .data import Task
from .policy import STRATEGIES
from .rewards import evaluate_trace
from .tools import ToolCall, calculator, retrieve

# Latent effectiveness: probability that a strategy, executed cleanly,
# yields the right answer for a task of a given kind. These are the
# environment dynamics, not learned parameters. Multiplication and
# division reward tool use and verification; easy addition is fine
# direct. The policy must learn this mapping from features alone.
_BASE_SUCCESS: dict[str, dict[str, float]] = {
    "+": {"direct": 0.85, "calculator": 0.95, "decompose": 0.90, "verify": 0.95, "retrieve": 0.80, "estimate": 0.70},
    "-": {"direct": 0.80, "calculator": 0.95, "decompose": 0.90, "verify": 0.95, "retrieve": 0.75, "estimate": 0.65},
    "*": {"direct": 0.45, "calculator": 0.90, "decompose": 0.85, "verify": 0.92, "retrieve": 0.80, "estimate": 0.60},
    "/": {"direct": 0.40, "calculator": 0.90, "decompose": 0.85, "verify": 0.92, "retrieve": 0.80, "estimate": 0.55},
}

_TOOL_PLAN: dict[str, list[str]] = {
    "direct": [],
    "calculator": ["calculator"],
    "decompose": ["calculator", "calculator"],
    "verify": ["calculator", "calculator"],
    "retrieve": ["retrieve", "calculator"],
    "estimate": ["calculator"],
}


def _tool_calls_for(strategy: str, task: Task) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tool in _TOOL_PLAN[strategy]:
        if tool == "calculator":
            expr = f"{task.a} {task.op} {task.b}"
            calls.append(ToolCall("calculator", {"expression": expr}, calculator(expr)))
        elif tool == "retrieve":
            results = retrieve(task.question)
            calls.append(ToolCall("retrieve", {"query": task.question[:40]}, results[0] if results else ""))
    return calls


def run_agent(
    task: Task,
    strategy: str | int,
    rng: np.random.Generator,
    tool_budget: int = 3,
) -> dict:
    """Execute one strategy for one task and return a scored trace."""
    if isinstance(strategy, (int, np.integer)):
        strategy = STRATEGIES[int(strategy)]
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy!r}")
    calls = _tool_calls_for(strategy, task)
    if len(calls) > tool_budget:
        calls = calls[:tool_budget]
    p_success = _BASE_SUCCESS[task.op][strategy]
    if task.difficulty == "hard":
        p_success = max(0.05, p_success - 0.10)
    correct = bool(rng.random() < p_success)
    # A slip produces a plausible wrong answer, not a crash.
    shown = task.answer if correct else task.answer + int(rng.integers(1, 6))
    if strategy == "verify" and not correct and rng.random() < 0.5:
        shown = task.answer
        correct = True
    text = (
        f"Strategy: {strategy}. "
        + " ".join(f"[{c.name}: {c.result}]" for c in calls)
        + f" <answer>{shown}</answer>"
    )
    trace = {
        "task_id": task.task_id,
        "strategy": strategy,
        "text": text,
        "tool_calls": [{"name": c.name, "args": c.args, "result": c.result} for c in calls],
    }
    scored = evaluate_trace(trace, task.answer)
    return {**trace, **scored}


def sample_group(
    task: Task,
    strategies: list[int],
    rng: np.random.Generator,
    tool_budget: int = 3,
) -> list[dict]:
    """Run the agent once per sampled strategy for a single task."""
    return [run_agent(task, s, rng, tool_budget=tool_budget) for s in strategies]
