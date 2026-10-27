"""Ablation runner: std normalisation, group size, and KL, measured honestly.

Each configuration trains from the same seed on the same split and is
evaluated on the same held-out test tasks with the same evaluation
seed. Differences therefore reflect the configuration, not a change
of data or evaluation. The headline comparison is the one the
October 2026 literature debates: vanilla GRPO (std on) versus the
Dr. GRPO / AlphaBet no-std variant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import build_dataset
from .evaluate import evaluate
from .train import TrainConfig, train


@dataclass
class AblationConfig:
    name: str
    use_std: bool
    group_size: int
    beta_kl: float


DEFAULT_CONFIGS = [
    AblationConfig("std-on_G8_KL0", use_std=True, group_size=8, beta_kl=0.0),
    AblationConfig("std-off_G8_KL0", use_std=False, group_size=8, beta_kl=0.0),
    AblationConfig("std-off_G4_KL0", use_std=False, group_size=4, beta_kl=0.0),
    AblationConfig("std-off_G8_KL0.04", use_std=False, group_size=8, beta_kl=0.04),
]


def run_ablation(
    configs: list[AblationConfig] | None = None,
    steps: int = 40,
    seed: int = 42,
) -> dict:
    configs = configs or DEFAULT_CONFIGS
    splits = build_dataset(n=300, seed=seed)
    results: list[dict] = []
    for cfg in configs:
        train_cfg = TrainConfig(
            steps=steps, group_size=cfg.group_size, use_std=cfg.use_std,
            beta_kl=cfg.beta_kl, seed=seed,
        )
        trained = train(train_cfg, tasks=splits["train"])
        policy = trained["final_policy"]
        metrics = evaluate(policy, splits["test"], group_size=8, seed=123)
        history = trained["history"]
        results.append(
            {
                "name": cfg.name,
                "use_std": cfg.use_std,
                "group_size": cfg.group_size,
                "beta_kl": cfg.beta_kl,
                "final_train_reward": history[-1]["mean_reward"],
                "first_train_reward": history[0]["mean_reward"],
                **{k: v for k, v in metrics.items() if k != "eval_group_size"},
                "eval_group_size": 8,
            }
        )
        print(f"{cfg.name}: test accuracy {metrics['greedy_accuracy']:.3f} "
              f"pass@8 {metrics['pass@8']:.3f} brier {metrics['brier_score']:.3f}")
    return {"seed": seed, "steps": steps, "results": results}


def plot_results(results: list[dict], out_path: str | Path) -> Path:
    """Bar chart of held-out accuracy by configuration, as SVG.

    Uses matplotlib when available; otherwise writes a hand-built SVG
    with the same numbers, so the plot never depends on an optional
    dependency in CI.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in results]
    values = [r["greedy_accuracy"] for r in results]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, values, color="#4C72B0")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Held-out greedy accuracy")
        ax.set_title("Ablation: held-out accuracy by GRPO configuration")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}",
                    ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_path, format="svg")
        plt.close(fig)
        return out_path
    except Exception:
        pass
    # Fallback: plain SVG, no external dependency.
    width, height = 720, 320
    bar_w = 110
    gap = 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="16">Ablation: held-out accuracy by GRPO configuration</text>',
    ]
    for i, (name, value) in enumerate(zip(names, values)):
        x = 30 + i * (bar_w + gap)
        bar_h = int(value * 220)
        y = 280 - bar_h
        parts.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" fill="#4C72B0"/>')
        parts.append(f'<text x="{x + bar_w // 2}" y="{y - 6}" font-family="sans-serif" font-size="13" text-anchor="middle">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w // 2}" y="300" font-family="sans-serif" font-size="11" text-anchor="middle">{name}</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the GRPO ablation.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/ablation.json")
    parser.add_argument("--plot", default="results/ablation.svg")
    args = parser.parse_args(argv)
    data = run_ablation(steps=args.steps, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    plot_results(data["results"], args.plot)
    print(f"wrote {out} and {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
