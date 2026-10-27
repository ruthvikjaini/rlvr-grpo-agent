"""GRPO training loop over the NumPy policy.

One step: sample a batch of prompts, sample ``group_size`` strategies
per prompt from the current policy, run the agent, score with the
verifier, form group-relative advantages, and update. The reference
policy is snapshotted at the start of training and used only for the
KL term; with ``beta_kl=0`` (the TRL GRPOTrainer default) no reference
is needed at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .agent import sample_group
from .data import Task, build_dataset
from .grpo import compute_advantages, grpo_loss
from .policy import MLPPolicy


@dataclass
class TrainConfig:
    steps: int = 80
    batch_size: int = 24
    group_size: int = 8
    learning_rate: float = 0.05
    use_std: bool = False
    beta_kl: float = 0.0
    epsilon: float = 0.2
    seed: int = 42
    n_tasks: int = 300
    tool_budget: int = 3
    log_every: int = 10


def train(
    config: TrainConfig,
    tasks: list[Task] | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Train from scratch and return a history dict with per-step metrics."""
    rng = np.random.default_rng(config.seed)
    if tasks is None:
        tasks = build_dataset(n=config.n_tasks, seed=config.seed)["train"]
    features_all = np.array([t.features() for t in tasks], dtype=float)
    policy = MLPPolicy(learning_rate=config.learning_rate, seed=config.seed)
    policy.snapshot_reference()
    history: list[dict] = []
    for step in range(1, config.steps + 1):
        idx = rng.choice(len(tasks), size=config.batch_size, replace=False)
        batch_tasks = [tasks[i] for i in idx]
        batch_features = features_all[idx]
        # Sample G strategies per prompt.
        probs = policy.probs(batch_features)
        sampled = np.array(
            [[rng.choice(len(p), p=p) for _ in range(config.group_size)] for p in probs]
        )
        rewards = np.zeros((config.batch_size, config.group_size))
        logp_old = np.zeros_like(rewards)
        for i, task in enumerate(batch_tasks):
            traces = sample_group(task, sampled[i].tolist(), rng, config.tool_budget)
            rewards[i] = [t["total"] for t in traces]
            logp_old[i] = [np.log(probs[i, a] + 1e-12) for a in sampled[i]]
        advantages = compute_advantages(rewards, use_std=config.use_std)
        # Flatten for the policy update, keeping (prompt, group) order.
        flat_features = np.repeat(batch_features, config.group_size, axis=0)
        flat_actions = sampled.reshape(-1)
        flat_adv = advantages.reshape(-1)
        # Diagnostics use the same loss the LLM-scale trainer uses.
        diagnostics = grpo_loss(
            logp_old, logp_old, logp_old, rewards,
            use_std=config.use_std, epsilon=config.epsilon, beta=config.beta_kl,
        )
        policy.update_from_advantages(
            flat_features, flat_actions, flat_adv, beta_kl=config.beta_kl
        )
        record = {
            "step": step,
            "mean_reward": float(rewards.mean()),
            "greedy_accuracy_proxy": float((rewards.max(axis=1) > 0.5).mean()),
            "entropy": policy.entropy(batch_features),
            "kl_to_reference": policy.kl_to_reference(batch_features),
            "frac_zero_std": diagnostics["frac_zero_std"],
        }
        history.append(record)
        if step % config.log_every == 0 or step == 1:
            print(
                f"step {step:3d}  reward {record['mean_reward']:.3f}  "
                f"entropy {record['entropy']:.3f}  kl {record['kl_to_reference']:.4f}  "
                f"zero-std groups {record['frac_zero_std']:.2f}"
            )
    result = {"config": vars(config), "history": history}
    if checkpoint_path is not None:
        policy.save(str(checkpoint_path))
        result["checkpoint"] = str(checkpoint_path)
    result["final_policy"] = policy
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the GRPO policy (NumPy, CPU).")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--use-std", action="store_true")
    parser.add_argument("--beta-kl", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/train_history.json")
    parser.add_argument("--checkpoint", default="results/policy.npz")
    args = parser.parse_args(argv)
    config = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, group_size=args.group_size,
        learning_rate=args.learning_rate, use_std=args.use_std,
        beta_kl=args.beta_kl, seed=args.seed,
    )
    result = train(config, checkpoint_path=args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {"config": result["config"], "history": result["history"]}
    out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    print(f"wrote {out} and {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
