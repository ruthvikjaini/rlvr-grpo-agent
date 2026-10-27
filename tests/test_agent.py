import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rlvr.agent import run_agent
from rlvr.data import Task


class TestAgent(unittest.TestCase):
    def test_agent_trace_is_scored(self):
        rng = np.random.default_rng(0)
        task = Task("t1", "A truck travels 20 kilometres in 4 equal legs. How many kilometres is each leg? Answer with a single number.", 5, 20, 4, "/", 2, "easy")
        trace = run_agent(task, "calculator", rng, tool_budget=3)
        self.assertIn("total", trace)
        self.assertEqual(trace["strategy"], "calculator")
        self.assertEqual(len(trace["tool_calls"]), 1)

    def test_tool_budget_is_respected(self):
        rng = np.random.default_rng(0)
        task = Task("t2", "Each of 6 teams has 4 players. How many players are there altogether? Answer with a single number.", 24, 6, 4, "*", 5, "easy")
        trace = run_agent(task, "decompose", rng, tool_budget=1)
        self.assertLessEqual(len(trace["tool_calls"]), 1)

    def test_unknown_strategy_rejected(self):
        rng = np.random.default_rng(0)
        task = Task("t3", "x", 1, 1, 1, "+", 0, "easy")
        with self.assertRaises(ValueError):
            run_agent(task, "teleport", rng)


if __name__ == "__main__":
    unittest.main()
