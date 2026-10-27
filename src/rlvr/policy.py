"""A small NumPy MLP policy over solution strategies.

The policy maps task features to a distribution over six strategies.
It is deliberately tiny: the point of this repository is the GRPO
machinery (grouping, advantages, KL, clipping), not a large model.
The same machinery is what TRL's GRPOTrainer applies at LLM scale;
see ``rlvr/scaleup/trl_grpo.py`` for that mapping.

Strategies
----------
0 direct      answer without a tool call
1 calculator  one calculator call, then answer
2 decompose   split the problem, two calculator calls
3 verify      compute, then check with the inverse operation
4 retrieve    look up a fact first, then calculate
5 estimate    estimate, then compute exactly
"""

from __future__ import annotations

import numpy as np

STRATEGIES = ["direct", "calculator", "decompose", "verify", "retrieve", "estimate"]
N_STRATEGIES = len(STRATEGIES)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class MLPPolicy:
    """Two-layer tanh MLP with manual backpropagation (NumPy only)."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 16,
        n_actions: int = N_STRATEGIES,
        seed: int = 0,
        learning_rate: float = 0.05,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.5, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0.0, 0.5, size=(hidden_dim, n_actions))
        self.b2 = np.zeros(n_actions)
        self.lr = learning_rate
        self._ref = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
        self._cache: dict[str, np.ndarray] = {}

    # -- forward ---------------------------------------------------------
    def logits(self, features: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(features, dtype=float))
        h_pre = x @ self.W1 + self.b1
        h = np.tanh(h_pre)
        z = h @ self.W2 + self.b2
        self._cache = {"x": x, "h_pre": h_pre, "h": h, "z": z}
        return z

    def probs(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.logits(features))

    def log_probs(self, features: np.ndarray) -> np.ndarray:
        return np.log(self.probs(features) + 1e-12)

    def sample(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probs = self.probs(features)
        actions = np.array(
            [rng.choice(len(p), p=p) for p in probs], dtype=int
        )
        return actions

    def greedy(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.probs(features), axis=-1)

    def entropy(self, features: np.ndarray) -> float:
        p = self.probs(features)
        return float(np.mean(-np.sum(p * np.log(p + 1e-12), axis=-1)))

    def kl_to_reference(self, features: np.ndarray) -> float:
        """Mean KL(policy || frozen reference) over a batch of features."""
        W1, b1, W2, b2 = self._ref
        x = np.atleast_2d(np.asarray(features, dtype=float))
        ref_logits = np.tanh(x @ W1 + b1) @ W2 + b2
        ref = _softmax(ref_logits)
        p = self.probs(features)
        return float(np.mean(np.sum(p * (np.log(p + 1e-12) - np.log(ref + 1e-12)), axis=-1)))

    def snapshot_reference(self) -> None:
        self._ref = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    # -- update ----------------------------------------------------------
    def apply_gradients(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        d_logits: np.ndarray,
    ) -> None:
        """One SGD step from per-example dL/dlogits.

        ``d_logits`` holds one gradient row per example. This method
        averages over the batch (dividing by ``n`` once) so the step
        size does not depend on the batch size.
        """
        cache = self._cache
        x, h, h_pre = cache["x"], cache["h"], cache["h_pre"]
        n = x.shape[0]
        dW2 = h.T @ d_logits / n
        db2 = d_logits.mean(axis=0)
        dh = d_logits @ self.W2.T
        dh_pre = dh * (1.0 - np.tanh(h_pre) ** 2)
        dW1 = x.T @ dh_pre / n
        db1 = dh_pre.mean(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def update_from_advantages(
        self,
        features: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        beta_kl: float = 0.0,
    ) -> dict[str, float]:
        """GRPO-style update: reinforce sampled actions by their advantage.

        The gradient of ``-A * log p(a)`` with respect to the logits is
        ``-A * (one_hot(a) - p)``. The KL term uses the exact categorical
        gradient ``p * (log p - log p_ref - KL)`` per row.
        """
        probs = self.probs(features)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(actions)), actions] = 1.0
        d_logits = -advantages[:, None] * (one_hot - probs)
        if beta_kl > 0.0:
            W1, b1, W2, b2 = self._ref
            x = np.atleast_2d(np.asarray(features, dtype=float))
            ref = _softmax(np.tanh(x @ W1 + b1) @ W2 + b2)
            log_ratio = np.log(probs + 1e-12) - np.log(ref + 1e-12)
            kl_rows = np.sum(probs * log_ratio, axis=-1, keepdims=True)
            d_logits += beta_kl * probs * (log_ratio - kl_rows)
        # Per-example gradients; apply_gradients averages over the batch.
        self.apply_gradients(features, actions, d_logits)
        return {"mean_advantage": float(np.mean(advantages))}

    # -- persistence -----------------------------------------------------
    def save(self, path: str) -> str:
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        return path

    @classmethod
    def load(cls, path: str, **kwargs) -> "MLPPolicy":
        data = np.load(path)
        policy = cls(**kwargs)
        policy.W1 = data["W1"]
        policy.b1 = data["b1"]
        policy.W2 = data["W2"]
        policy.b2 = data["b2"]
        policy.snapshot_reference()
        return policy
