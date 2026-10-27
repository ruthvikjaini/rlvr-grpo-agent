"""Scale-up sketch: a custom GRPO loop on SageMaker Training Sessions.

SageMaker AI serverless model customization training sessions became
generally available on 22 October 2026 (AWS Machine Learning Blog).
The developer writes the training loop in Python; SageMaker AI runs
GPU allocation, distributed execution, fault recovery, and checkpoint
storage, and billing is per token processed, with the meter stopped
while the developer's reward code scores a batch.

The loop below is a sketch, not executed code: it requires the
SageMaker Python SDK, an IAM role, an S3 output location, and a
supported model (e.g. ``gpt-oss-20b``). It mirrors the public
programming model described by AWS: three clients (service, training,
sampling), with ``sample()``, ``forward_backward()``, and
``optim_step()`` composing one GRPO step. The reward function is the
same deterministic verifier used in ``rlvr.rewards``.
"""

from __future__ import annotations


def sagemaker_grpo_sketch() -> str:
    return '''\
# Requires: sagemaker SDK, an IAM role, an S3 output path, a GPU session.
# from sagemaker.train import ServiceClient, TrainingClient, SamplingClient

# service = ServiceClient(...)
# session = service.create_training_session(model="gpt-oss-20b", ...)
# training = TrainingClient(session)
# sampling = SamplingClient(session)

# for batch in prompts:
#     completions = sampling.sample(batch, num_generations=8)
#     rewards = [verifier(c, gold) for c, gold in zip(completions, batch)]
#     advantages = group_advantages(rewards, use_std=False)  # no-std variant
#     training.forward_backward(completions, advantages)
#     training.optim_step()
#     # Meter stops while verifier() runs in the developer environment.
'''
