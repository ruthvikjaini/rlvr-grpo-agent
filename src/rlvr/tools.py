"""Deterministic, side-effect-free tools for the agent loop.

The calculator evaluates only the four arithmetic operators on integers,
using an explicit parser instead of ``eval``. The retrieval tool searches
a small in-memory knowledge base of arithmetic facts; it is a stand-in
for a real retriever (vector store, API) and keeps the whole system
runnable offline while preserving the tool-use interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_KB = [
    ("multiplication", "Multiplication is repeated addition: a * b adds a to itself b times."),
    ("division", "Division splits a total into equal parts: a / b is the size of each part."),
    ("subtraction", "Subtraction finds the difference between two numbers."),
    ("addition", "Addition combines two quantities into a total."),
    ("inverse", "Check a multiplication by dividing the product by one factor."),
    ("estimate", "Estimate first: round to the nearest ten, then compute exactly."),
]


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    result: str


def calculator(expression: str) -> str:
    """Evaluate ``<int> <op> <int>`` for +, -, *, /. No other input is valid."""
    match = re.fullmatch(r"\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*", expression)
    if match is None:
        raise ValueError(f"unsupported expression: {expression!r}")
    a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    if op == "/":
        if b == 0:
            raise ValueError("division by zero")
        if a % b != 0:
            raise ValueError("calculator only returns exact integer results")
        return str(a // b)
    raise ValueError(f"unsupported operator: {op!r}")


def retrieve(query: str, top_k: int = 2) -> list[str]:
    """Keyword retrieval over the local knowledge base."""
    query_words = {w.strip(".,?") for w in query.lower().split()}
    scored = []
    for title, text in _KB:
        overlap = len(query_words & {w for w in text.lower().split()})
        scored.append((overlap, title, text))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [text for _, _, text in scored[:top_k]]


TOOLS = {"calculator": calculator, "retrieve": retrieve}
