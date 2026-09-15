"""CLAP-sized joint generation/retrieval training from random initialization."""

from joint_scratch.config import ScratchModelConfig, estimate_parameter_count
from joint_scratch.model import ScratchJointModel, Task

__all__ = [
    "ScratchJointModel",
    "ScratchModelConfig",
    "Task",
    "estimate_parameter_count",
]
