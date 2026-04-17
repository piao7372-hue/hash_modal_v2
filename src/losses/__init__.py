from .balance_loss import compute_balance_loss
from .induced_targets import build_induced_targets
from .pair_loss import compute_pair_loss
from .quantization_loss import compute_quantization_loss
from .relation_predictions import build_relation_predictions
from .semantic_loss import compute_semantic_loss
from .total_loss import LossConfig, compute_total_loss

__all__ = [
    "LossConfig",
    "build_induced_targets",
    "build_relation_predictions",
    "compute_balance_loss",
    "compute_pair_loss",
    "compute_quantization_loss",
    "compute_semantic_loss",
    "compute_total_loss",
]
