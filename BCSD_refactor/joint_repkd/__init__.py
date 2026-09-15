"""Two-stage representation-distillation pretraining for the joint student.

Implements the supervisor's "Joint-Scratch Distillation Objectives":

* Stage 1: causal NTP (assembly translation) plus final-layer hidden-state
  distillation from a frozen teacher, both under a causal attention mask.
* Stage 2: bidirectional MLM on token-masked assembly plus hidden-state
  distillation, the teacher seeing the clean assembly bidirectionally.

The student is the joint-scratch ``ScratchJointModel`` (causal LM + bidirectional
MLM).  The teacher is the frozen EBM-full Nova model serving ``last_hidden_state``
through :class:`joint_distill.teacher.FullEBMTeacher.hidden_states`.
"""

from joint_repkd.balance import GradientBalance, teacher_lambda
from joint_repkd.data import NovaRetrievalCollator, RepKDCollator
from joint_repkd.losses import hidden_state_distillation_loss

__all__ = [
    "GradientBalance",
    "NovaRetrievalCollator",
    "RepKDCollator",
    "hidden_state_distillation_loss",
    "teacher_lambda",
]
