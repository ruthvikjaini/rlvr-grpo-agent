import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from rlvr.data import build_dataset, generate_tasks, split_tasks, validate_task
from rlvr.grpo import compute_advantages, grpo_loss
from rlvr.policy import MLPPolicy
from rlvr.rewards import brier_score, extract_answer, reward_components
from rlvr.tools import calculator, retrieve


class TestAdvantages(unittest.TestCase):
    def test_no_std_is_mean_centred(self):
        r = np.array([[1.0, 0.0, 1.0, 0.0]])
        adv = compute_advantages(r, use_std=False)
        np.testing.assert_allclose(adv, [[0.5, -0.5, 0.5, -0.5]])

    def test_std_variant(self):
        r = np.array([[1.0, 0.0, 1.0, 0.0]])
        adv = compute_advantages(r, use_std=True)
        # std (population) of [1,0,1,0] is 0.5, so advantages are +-1.
        np.testing.assert_allclose(adv, [[1.0, -1.0, 1.0, -1.0]])

    def test_all_equal_group_has_zero_advantage(self):
        r = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        for use_std in (True, False):
            np.testing.assert_allclose(compute_advantages(r, use_std=use_std), 0.0)

    def test_grpo_loss_clipping(self):
        rewards = np.array([[1.0, 0.0, 1.0, 0.0]])
        old = np.log(np.array([[0.25, 0.25, 0.25, 0.25]]))
        # New policy puts all mass on the first (correct) completion.
        new = np.log(np.array([[0.9, 0.033, 0.033, 0.034]]))
        ref = old.copy()
        out = grpo_loss(new, old, ref, rewards, use_std=False, epsilon=0.2)
        self.assertIn("loss", out)
        self.assertGreater(out["frac_clipped"], 0.0)
        self.assertAlmostEqual(out["mean_reward"], 0.5)


class TestRewards(unittest.TestCase):
    def test_extract_answer(self):
        self.assertEqual(extract_answer("work <answer>42</answer>"), 42)
        self.assertIsNone(extract_answer("no answer here"))
        self.assertIsNone(extract_answer("<answer>abc</answer>"))

    def test_correct_beats_formatted_wrong(self):
        good = reward_components("x <answer>42</answer>", 42, 1)
        bad = reward_components("x <answer>41</answer>", 42, 1)
        self.assertGreater(good["total"], bad["total"])
        self.assertEqual(bad["correctness"], 0.0)

    def test_brier(self):
        self.assertAlmostEqual(brier_score([1.0], [1]), 0.0)
        self.assertAlmostEqual(brier_score([0.0], [1]), 1.0)
        self.assertAlmostEqual(brier_score([0.5, 0.5], [1, 0]), 0.25)


class TestTools(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator("12 * 7"), "84")
        self.assertEqual(calculator("20 / 4"), "5")
        with self.assertRaises(ValueError):
            calculator("import os")

    def test_retrieve_returns_strings(self):
        self.assertTrue(retrieve("multiplication facts"))


class TestData(unittest.TestCase):
    def test_generate_and_validate(self):
        tasks = generate_tasks(50, seed=1)
        self.assertEqual(len(tasks), 50)
        self.assertEqual(len({t.key() for t in tasks}), 50)
        for task in tasks:
            self.assertEqual(validate_task(task), [])

    def test_split_has_no_leakage(self):
        tasks = generate_tasks(120, seed=2)
        splits = split_tasks(tasks, seed=2)
        total = sum(len(v) for v in splits.values())
        self.assertEqual(total, len(tasks))
        keys = {name: {t.key() for t in ts} for name, ts in splits.items()}
        self.assertFalse(keys["train"] & keys["test"])
        self.assertFalse(keys["train"] & keys["validation"])

    def test_build_dataset(self):
        splits = build_dataset(n=120, seed=3)
        self.assertTrue(all(len(v) > 0 for v in splits.values()))


class TestPolicy(unittest.TestCase):
    def test_probs_sum_to_one(self):
        policy = MLPPolicy(seed=0)
        probs = policy.probs(np.zeros((4, 8)))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_training_step_changes_weights(self):
        policy = MLPPolicy(seed=0, learning_rate=0.1)
        before = policy.W2.copy()
        features = np.random.default_rng(0).normal(size=(8, 8))
        policy.update_from_advantages(
            features, np.zeros(8, dtype=int), np.ones(8)
        )
        self.assertFalse(np.allclose(before, policy.W2))

    def test_kl_is_zero_at_reference(self):
        policy = MLPPolicy(seed=0)
        self.assertAlmostEqual(policy.kl_to_reference(np.zeros((3, 8))), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
