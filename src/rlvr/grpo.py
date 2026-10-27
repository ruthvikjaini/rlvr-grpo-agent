"""GRPO: group-relative advantages and the clipped surrogate objective.

References (verified October 2026, see README for full citations):

* Shao et al., DeepSeekMath / GRPO (2024): group-relative advantage
  ``(R - mean(R)) / std(R)`` with no learned critic.
* Liu et al., "Understanding R1-Zero-Like Training" / Dr. GRPO
  (arXiv:2503.20783): removing the std term and the per-response length
  term removes a difficulty bias and a length bias.
* Ye et al., AlphaBet (arXiv:2610.22684, submitted 15 Oct 2026):
  std normalisation can distort the reporting target under multi-class
  proper scores; the no-std variant preserves the expected-score
  gradient up to a positive factor.
* Yu et al., DAPO (2025): clip-higher and dynamic sampling; all-equal
  groups carry no gradient and are flagged here as ``frac_zero_std``.

This module is framework-free so the maths can be unit-tested exactly.
``compute_advantages`` and ``grpo_loss`` operate on reward matrices of
shape ``(batch, group)``.
"""

from __future__ import annotations

import numpy as np


def compute_advantages(
    rewards: np.ndarray,
    use_std: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Group-relative advantages for each prompt's sampled group.

    With ``use_std=True`` this is vanilla GRPO; with ``use_std=False``
    it is the Dr. GRPO / AlphaBet no-std variant. Groups whose rewards
    are all equal get exactly zero advantage either way.
    """
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 2:
        raise ValueError("rewards must have shape (batch, group)")
    mean = r.mean(axis=1, keepdims=True)
    if use_std:
        std = r.std(axis=1, keepdims=True)
        advantages = (r - mean) / (std + eps)
        # A group with zero variance must give zero advantage, not a
        # tiny epsilon-amplified signal.
        advantages = np.where(std < 1e-12, 0.0, advantages)
    else:
        advantages = r - mean
    return advantages


def grpo_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    log_probs_ref: np.ndarray,
    rewards: np.ndarray,
    use_std: bool = True,
    epsilon: float = 0.2,
    epsilon_high: float | None = None,
    beta: float = 0.0,
) -> dict[str, float]:
    """Clipped GRPO surrogate loss (to be minimised) and diagnostics.

    ``log_probs_*`` are per-completion summed log-probabilities of shape
    ``(batch, group)``. The ratio is clipped PPO-style; ``epsilon_high``
    implements DAPO's clip-higher when set. The KL term is the per-sample
    KL estimate ``log p_new - log p_ref`` averaged over the batch.
    """
    advantages = compute_advantages(rewards, use_std=use_std)
    ratio = np.exp(np.asarray(log_probs_new) - np.asarray(log_probs_old))
    lo = 1.0 - epsilon
    hi = 1.0 + (epsilon_high if epsilon_high is not None else epsilon)
    clipped = np.clip(ratio, lo, hi)
    surrogate = np.minimum(ratio * advantages, clipped * advantages)
    kl = float(np.mean(np.asarray(log_probs_new) - np.asarray(log_probs_ref)))
    loss = float(-np.mean(surrogate) + beta * kl)
    return {
        "loss": loss,
        "mean_reward": float(np.mean(rewards)),
        "mean_advantage": float(np.mean(np.abs(advantages))),
        "mean_ratio": float(np.mean(ratio)),
        "frac_clipped": float(np.mean((ratio < lo) | (ratio > hi))),
        "kl": kl,
        "frac_zero_std": float(
            np.mean(np.asarray(rewards).std(axis=1) < 1e-12)
        ),
    }
