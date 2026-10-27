"""Data engineering: deterministic task generation, validation, and storage.

The task set is arithmetic word problems whose answers are computed, never
labelled by hand, so the verifier in :mod:`rlvr.rewards` is exact. A task is
identified by ``(a, b, op, template_id)``; that key is what the split uses
to guarantee no prompt leaks between train, validation, and test.

An optional JSONL input in the same schema as the generated records can be
supplied instead of (or merged with) generated tasks, e.g. a GSM8K-style
export with ``question`` / ``answer`` fields. External records are
re-validated with the same checks and are rejected, not silently fixed,
when their stated answer does not match a recomputed answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

OPS = ("+", "-", "*", "/")

_TEMPLATES = [
    "A warehouse has {a} boxes. Each box holds {b} items. How many items are there in total? Answer with a single number.",
    "Maya buys {a} packs with {b} stickers in each pack. How many stickers does she have? Answer with a single number.",
    "A truck travels {a} kilometres in {b} equal legs. How many kilometres is each leg? Answer with a single number.",
    "There are {a} apples. {b} are eaten. How many apples remain? Answer with a single number.",
    "A baker makes {a} cakes and sells {b} of them. How many cakes are left? Answer with a single number.",
    "Each of {a} teams has {b} players. How many players are there altogether? Answer with a single number.",
]


@dataclass(frozen=True)
class Task:
    """One verifiable reasoning task."""

    task_id: str
    question: str
    answer: int
    a: int
    b: int
    op: str
    template_id: int
    difficulty: str

    def key(self) -> tuple[int, int, str, int]:
        return (self.a, self.b, self.op, self.template_id)

    def features(self) -> list[float]:
        """Numeric features the policy sees. No answer information leaks."""
        return [
            math.log1p(abs(self.a)) / 5.0,
            math.log1p(abs(self.b)) / 5.0,
            1.0 if self.op == "+" else 0.0,
            1.0 if self.op == "-" else 0.0,
            1.0 if self.op == "*" else 0.0,
            1.0 if self.op == "/" else 0.0,
            1.0 if self.difficulty == "hard" else 0.0,
            1.0 if abs(self.a) >= 100 or abs(self.b) >= 20 else 0.0,
        ]


def _answer(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a // b
    raise ValueError(f"unknown op: {op!r}")


def _difficulty(a: int, b: int, op: str) -> str:
    magnitude = abs(_answer(a, b, op))
    if magnitude >= 500:
        return "hard"
    if magnitude >= 100:
        return "medium"
    return "easy"


def generate_tasks(n: int, seed: int = 42) -> list[Task]:
    """Generate ``n`` unique, validated tasks deterministically."""
    rng = random.Random(seed)
    seen: set[tuple[int, int, str, int]] = set()
    tasks: list[Task] = []
    attempts = 0
    while len(tasks) < n and attempts < n * 50:
        attempts += 1
        op = rng.choice(OPS)
        # Division only fits the "equal legs" wording (template 2);
        # every other op uses the remaining templates, so no op is
        # starved by rejection sampling.
        if op == "/":
            template_id = 2
        else:
            template_id = rng.choice([i for i in range(len(_TEMPLATES)) if i != 2])
        if op == "+":
            a, b = rng.randint(12, 240), rng.randint(7, 120)
        elif op == "-":
            a, b = rng.randint(30, 300), rng.randint(5, 120)
        elif op == "*":
            a, b = rng.randint(6, 48), rng.randint(3, 24)
        else:
            b = rng.randint(3, 18)
            a = b * rng.randint(4, 30)
        key = (a, b, op, template_id)
        if key in seen:
            continue
        seen.add(key)
        question = _TEMPLATES[template_id].format(a=a, b=b)
        tasks.append(
            Task(
                task_id=f"task-{len(tasks):04d}",
                question=question,
                answer=_answer(a, b, op),
                a=a,
                b=b,
                op=op,
                template_id=template_id,
                difficulty=_difficulty(a, b, op),
            )
        )
    if len(tasks) < n:
        raise RuntimeError(f"could only generate {len(tasks)} unique tasks")
    return tasks


def validate_task(task: Task) -> list[str]:
    """Return a list of validation errors; empty means the task is valid."""
    errors: list[str] = []
    if task.op not in OPS:
        errors.append(f"unknown op {task.op!r}")
    elif task.answer != _answer(task.a, task.b, task.op):
        errors.append("answer does not match recomputed result")
    if task.op == "/" and task.b == 0:
        errors.append("division by zero")
    if task.op == "/" and task.a % task.b != 0:
        errors.append("non-integer division result")
    if not task.question or "{a}" in task.question or "{b}" in task.question:
        errors.append("question is empty or still contains a placeholder")
    if str(task.a) not in task.question or str(task.b) not in task.question:
        errors.append("question does not contain both operands")
    return errors


def split_tasks(
    tasks: Sequence[Task],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Task]]:
    """Stratified-by-op split with no overlapping task keys across splits."""
    rng = random.Random(seed)
    by_op: dict[str, list[Task]] = {}
    for task in tasks:
        by_op.setdefault(task.op, []).append(task)
    splits: dict[str, list[Task]] = {"train": [], "validation": [], "test": []}
    for op_tasks in by_op.values():
        shuffled = op_tasks[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits["train"].extend(shuffled[:n_train])
        splits["validation"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])
    # Keys must be disjoint by construction; assert rather than assume.
    keys = {name: {t.key() for t in ts} for name, ts in splits.items()}
    assert not (keys["train"] & keys["validation"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["validation"] & keys["test"])
    return splits


def dataset_manifest(tasks: Sequence[Task]) -> dict:
    """Hash-addressed manifest: one record per task, plus split-free totals."""
    records = []
    for task in tasks:
        payload = json.dumps(asdict(task), sort_keys=True).encode()
        records.append(
            {
                "task_id": task.task_id,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "op": task.op,
                "difficulty": task.difficulty,
            }
        )
    return {
        "n_tasks": len(tasks),
        "by_op": {op: sum(1 for t in tasks if t.op == op) for op in OPS},
        "by_difficulty": {
            d: sum(1 for t in tasks if t.difficulty == d)
            for d in ("easy", "medium", "hard")
        },
        "records": records,
    }


def write_sqlite(tasks: Sequence[Task], path: str | Path) -> Path:
    """Store validated tasks in SQLite (standard library only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks "
            "(task_id TEXT PRIMARY KEY, question TEXT, answer INTEGER, "
            "a INTEGER, b INTEGER, op TEXT, template_id INTEGER, difficulty TEXT)"
        )
        conn.execute("DELETE FROM tasks")
        for task in tasks:
            errors = validate_task(task)
            if errors:
                raise ValueError(f"invalid task {task.task_id}: {errors}")
            conn.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                (task.task_id, task.question, task.answer, task.a, task.b,
                 task.op, task.template_id, task.difficulty),
            )
        conn.commit()
    finally:
        conn.close()
    return path


def load_tasks(path: str | Path) -> list[Task]:
    """Load tasks back from the SQLite store, in task_id order."""
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT task_id, question, answer, a, b, op, template_id, difficulty "
            "FROM tasks ORDER BY task_id"
        ).fetchall()
    finally:
        conn.close()
    return [Task(*row) for row in rows]


def tasks_from_jsonl(path: str | Path) -> list[Task]:
    """Load external JSONL records, re-validating every answer.

    Expected fields per line: ``question``, ``answer``, ``a``, ``b``, ``op``.
    Records that fail validation raise ``ValueError`` instead of being
    silently corrected.
    """
    tasks: list[Task] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task = Task(
                task_id=record.get("task_id", f"ext-{line_no:04d}"),
                question=record["question"],
                answer=int(record["answer"]),
                a=int(record["a"]),
                b=int(record["b"]),
                op=record["op"],
                template_id=int(record.get("template_id", 0)),
                difficulty=record.get(
                    "difficulty", _difficulty(int(record["a"]), int(record["b"]), record["op"])
                ),
            )
            errors = validate_task(task)
            if errors:
                raise ValueError(f"{path}:{line_no}: {errors}")
            tasks.append(task)
    return tasks


def build_dataset(
    n: int = 300,
    seed: int = 42,
    extra_jsonl: str | Path | None = None,
) -> dict[str, list[Task]]:
    """Generate, validate, dedupe, and split the full task set."""
    tasks = generate_tasks(n, seed=seed)
    if extra_jsonl is not None:
        extra = tasks_from_jsonl(extra_jsonl)
        seen = {t.key() for t in tasks}
        tasks.extend(t for t in extra if t.key() not in seen)
    for task in tasks:
        errors = validate_task(task)
        if errors:
            raise ValueError(f"invalid task {task.task_id}: {errors}")
    if len({t.key() for t in tasks}) != len(tasks):
        raise ValueError("duplicate task keys survived generation")
    return split_tasks(tasks, seed=seed)


def write_jsonl(tasks: Iterable[Task], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    return path
