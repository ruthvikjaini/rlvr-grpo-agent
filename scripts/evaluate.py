import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rlvr.data import build_dataset
from rlvr.evaluate import evaluate
from rlvr.policy import MLPPolicy


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved policy on held-out tasks.")
    parser.add_argument("--checkpoint", default="results/policy.npz")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="results/eval.json")
    args = parser.parse_args()
    splits = build_dataset(n=300, seed=42)
    policy = MLPPolicy.load(args.checkpoint)
    metrics = evaluate(policy, splits["test"], seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
