"""Held-out evaluation: greedy accuracy, pass@k, calibration, hacking checks."""

from __future__ import annotations

import numpy as np

from .agent import run_agent, sample_group
from .data import Task
from .policy import MLPPolicy
from .rewards import brier_score


def evaluate(
    policy: MLPPolicy,
    tasks: list[Task],
    group_size: int = 8,
    seed: int = 123,
    tool_budget: int = 3,
) -> dict:
    """Evaluate on held-out tasks the policy never trained on."""
    rng = np.random.default_rng(seed)
    features = np.array([t.features() for t in tasks], dtype=float)
    greedy_actions = policy.greedy(features)
    probs = policy.probs(features)
    confidences: list[float] = []
    outcomes: list[int] = []
    for task, action, prob_row in zip(tasks, greedy_actions, probs):
        trace = run_agent(task, int(action), rng, tool_budget=tool_budget)
        confidences.append(float(prob_row[action]))
        outcomes.append(int(trace["correctness"]))

    # pass@k: fresh groups of `group_size` sampled completions per task.
    rng2 = np.random.default_rng(seed + 1)
    hits = 0
    hacking_flags = 0
    total_traces = 0
    for task in tasks:
        tiled = np.tile(np.array([task.features()], dtype=float), (group_size, 1))
        sampled = [int(a) for a in policy.sample(tiled, rng2)]
        group = sample_group(task, sampled, rng2, tool_budget)
        if any(t["correctness"] for t in group):
            hits += 1
        for trace in group:
            total_traces += 1
            if trace["hacking"]["formatted_but_wrong"] or trace["hacking"]["many_tool_calls"]:
                hacking_flags += 1
    return {
        "n_tasks": len(tasks),
        "greedy_accuracy": float(np.mean(outcomes)) if outcomes else 0.0,
        f"pass@{group_size}": float(hits / len(tasks)) if tasks else 0.0,
        "brier_score": brier_score(confidences, outcomes),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "hacking_flag_rate": float(hacking_flags / max(total_traces, 1)),
        "eval_group_size": group_size,
    }


def baseline_evaluate(tasks: list[Task], seed: int = 123) -> dict:
    """Uniform-random strategy choice, for the before/after comparison."""
    rng = np.random.default_rng(seed)
    outcomes = []
    for task in tasks:
        action = int(rng.integers(0, 6))
        trace = run_agent(task, action, rng)
        outcomes.append(int(trace["correctness"]))
    return {"greedy_accuracy": float(np.mean(outcomes)), "n_tasks": len(tasks)}
