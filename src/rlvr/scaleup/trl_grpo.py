"""Scale-up reference: the same ideas with TRL / vLLM at LLM scale.

This file is documentation-as-code. It is not imported by the NumPy
training loop and is not run in CI, because it requires a GPU, a
model download, and the ``trl`` / ``datasets`` / ``vllm`` packages.
It exists so the mapping from this repository's from-scratch GRPO to
the production stack is exact, not hand-waved.

Verified against the TRL GRPOTrainer usage current in October 2026
(see README citations):

* ``num_generations`` is the group size G.
* ``scale_rewards`` controls the group-std division: ``True`` is
  vanilla GRPO, ``False`` is the Dr. GRPO / AlphaBet no-std variant.
* ``beta`` is the KL coefficient; TRL's default is 0.0.
* ``loss_type="dapo"`` selects DAPO-style token-level aggregation
  and clip-higher via ``epsilon_high``.
* Reward functions receive completions plus dataset columns and
  return one float per completion. The verifier in ``rlvr.rewards``
  is the same contract, with a deterministic checker instead of a
  learned reward model.
"""

from __future__ import annotations


def build_trl_example() -> str:
    return '''\
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from rlvr.rewards import extract_answer


def reward_correct(completions, answer, **kwargs):
    # Verifiable reward: 1.0 iff the extracted answer matches.
    return [
        1.0 if extract_answer(c) == int(a) else 0.0
        for c, a in zip(completions, answer)
    ]


args = GRPOConfig(
    output_dir="grpo-qwen",
    num_generations=8,       # G: completions per prompt
    scale_rewards=False,     # no-std variant (Dr. GRPO / AlphaBet)
    beta=0.0,                # KL term off, as in current TRL defaults
    loss_type="dapo",
    epsilon_high=0.28,
    use_vllm=True,
    vllm_mode="colocate",
    learning_rate=1e-6,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    reward_funcs=reward_correct,
    args=args,
    train_dataset=load_dataset("my/math-prompts", split="train"),
)
trainer.train()
'''
