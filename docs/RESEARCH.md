# Research notes — 27 October 2026

Research was completed before any code was written. Each claim below
was checked against a live source on the day; URLs are copied
verbatim from the search results.

## Verified claims used in the build

### 1. GRPO std normalisation and proper scores — AlphaBet

* Source: Ye et al., arXiv:2610.22684, submitted 15 October 2026.
  https://arxiv.org/abs/2610.22684v1
* Verified: the paper introduces an agentic RL framework that trains
  one LLM policy to search, reason, and report probabilities across
  multiple pre-resolution dates, scored by a proper scoring rule.
  It states that GRPO's std normalisation "can distort the reporting
  target under multi-class proper scores", and that the no-std
  variant preserves the expected-score gradient up to a positive
  factor (mean-centring retained, std removed).
* Verified numbers: Brier 0.731 → 0.626 on Forecast-Dojo and
  0.612 → 0.528 on FutureX, on GLM-4.5-Air.
* Used in this repository: the `use_std` flag in
  `compute_advantages`, and the Brier-score evaluation. The
  direction of the effect was *not* assumed to transfer to a
  binary-correctness reward; the ablation measures it here.

### 2. Dr. GRPO — removing std and length normalisation

* Source: Liu et al., arXiv:2503.20783.
  https://arxiv.org/html/2503.20783
* Verified: Dr. GRPO removes the `1/|o_i|` length term and the
  group-std term, arguing the former rewards long incorrect
  responses and the latter over-weights near-solved / near-failed
  prompts (difficulty bias). The resulting objective recovers a
  PPO-style objective with a Monte-Carlo baseline.
* Used: same `use_std=False` path; length normalisation is not
  applied anywhere in this codebase (completions are single
  strategy choices, so per-token aggregation does not arise).

### 3. DAPO — clip-higher, dynamic sampling, token-level loss

* Source: October 2026 summary of Yu et al. (DAPO, 2025).
  https://www.aicassindra.com/blogs/gpu/gpu_grpo.html
* Verified: clip-higher decouples the clip range (upper bound
  wider, e.g. 0.28), dynamic sampling drops all-equal groups, and
  token-level aggregation weighs tokens equally. Many recipes set
  the KL coefficient to zero for verifiable-reward training.
* Used: `epsilon_high` in `grpo_loss`, `frac_zero_std` diagnostics,
  and the KL-off default. This is a secondary source; the DAPO
  paper itself should be read before citing DAPO results.

### 4. SageMaker Training Sessions — generally available 22 Oct 2026

* Source: AWS Machine Learning Blog, published 22 Oct 2026.
  https://aws.amazon.com/blogs/machine-learning/announcing-new-ways-to-customize-models-with-sagemaker-ai-without-managing-gpus/
* Verified: training sessions for SageMaker AI serverless model
  customization are generally available; the developer writes the
  post-training loop (custom loss, reward signals, multi-turn RL)
  while SageMaker AI handles GPU allocation, distributed execution,
  fault recovery, and checkpoint storage. Billing is per token
  processed and the meter stops while the reward function scores
  a batch. Launch models included `gpt-oss-20b`, `Qwen3.6-27B`,
  and `Qwen3.8-27B`.
* Corroborated by the AWS Weekly Roundup of 26 Oct 2026, which
  repeats the GA date and the per-token / meter-stopped billing.
* Used: the sketch in `src/rlvr/scaleup/sagemaker_loop.py` only.
  No AWS resources were created for this project and no AWS
  claims were tested live.

### 5. TRL GRPOTrainer — current usage

* Source: October 2026 write-up.
  https://ilirivezaj.com/blog/grpo-rlvr-post-training
* Verified: `GRPOTrainer` takes a model, a dataset of prompts, and
  reward callables; `num_generations` is the group size;
  `scale_rewards=True` keeps the std division and `False` drops
  it; `beta=0.0` is the KL-off default; `use_vllm` with
  `vllm_mode` colocate/server controls rollout placement.
* Caveat: this is a secondary source describing the author's
  verification against current docs. TRL's API has moved quickly
  in 2026 (including deprecation of `trl vllm-serve` in favour of
  `vllm serve`, noted in a separate October 2026 write-up).
  The scale-up file is therefore labelled reference-only and
  must be re-checked against the installed TRL release.

### 6. Harness co-training — HarnessAdapt-RL and Self-HPO

* Sources:
  https://aiweekly.co/alerts/harnessadapt-rl-lifts-9b-agent-126-pts-on-swe-bench-multilingual
  https://explainx.ai/blog/self-hpo-harness-policy-optimization-coding-agents-microsoft-research-2026
* Verified: HarnessAdapt-RL (posted 21 Oct 2026) co-trains the
  policy with mechanism-level skills and evolved tools/hooks,
  reporting +9.8 / +12.6 / +6.4 pass@1 points over a GRPO
  baseline on SWE-bench Verified / Multilingual / Pro with a 9B
  model, and a 7.7-point cost from removing evolved tools at
  deployment. Self-HPO (Microsoft Research, Oct 2026) reports
  57.8% → 64.8% average resolve rate by letting the agent rewrite
  its harness and then training with RL.
* Used: the design decision to version and freeze the harness
  (tool budget, trace format, verifier) within each comparison,
  and the warning that policies do not transfer across harnesses
  for free. These are secondary reports of the papers; the
  papers themselves are the primary sources.

## Claims deliberately not used

* Unverified vendor benchmark claims from news aggregators
  (e.g. new model releases in the same week) were not used:
  they are not needed for this project and were not checked
  against primary sources.
* No claim is made that the toy results here generalise to LLM
  scale. The ablation is reported with its sample size (n=48)
  and with the opposite-direction Brier finding discussed
  honestly in the README.
