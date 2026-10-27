"""RLVR Agent: verifiable-reward GRPO for tool-using reasoning agents.

Core package. The NumPy implementation in this package runs on CPU with no
GPU and no deep-learning framework. The optional scale-up scripts under
``rlvr.scaleup`` show the same reward and grouping ideas expressed with
TRL / vLLM and with SageMaker Training Sessions.
"""

from .data import Task, build_dataset, load_tasks
from .grpo import compute_advantages, grpo_loss
from .policy import MLPPolicy
from .rewards import evaluate_trace, reward_components

__all__ = [
    "Task",
    "build_dataset",
    "load_tasks",
    "compute_advantages",
    "grpo_loss",
    "MLPPolicy",
    "evaluate_trace",
    "reward_components",
]
__version__ = "0.1.0"
