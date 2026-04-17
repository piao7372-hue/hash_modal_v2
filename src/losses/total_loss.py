from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor

from .balance_loss import compute_balance_loss
from .induced_targets import build_induced_targets
from .pair_loss import compute_pair_loss
from .quantization_loss import compute_quantization_loss
from .semantic_loss import compute_semantic_loss


@dataclass(frozen=True)
class LossConfig:
    alpha: float
    lambda_sem: float
    lambda_pair: float
    lambda_q: float
    lambda_bal: float
    bce_eps: float


def compute_total_loss(
    P_IT: Tensor,
    P_II: Tensor,
    P_TT: Tensor,
    S: Tensor,
    H_I: Tensor,
    H_T: Tensor,
    config: LossConfig,
) -> Dict[str, Tensor]:
    induced_targets = build_induced_targets(S=S, eps=config.bce_eps)
    semantic_terms = compute_semantic_loss(
        P_IT=P_IT,
        P_II=P_II,
        P_TT=P_TT,
        S=S,
        S_II_star=induced_targets["S_II_star"],
        S_TT_star=induced_targets["S_TT_star"],
        alpha=config.alpha,
        eps=config.bce_eps,
    )
    L_pair = compute_pair_loss(H_I=H_I, H_T=H_T)
    L_q = compute_quantization_loss(H_I=H_I, H_T=H_T)
    L_bal = compute_balance_loss(H_I=H_I, H_T=H_T)
    loss_total = (
        (config.lambda_sem * semantic_terms["L_sem"])
        + (config.lambda_pair * L_pair)
        + (config.lambda_q * L_q)
        + (config.lambda_bal * L_bal)
    )
    if not torch.isfinite(loss_total):
        raise RuntimeError("Total loss contains NaN or Inf.")
    return {
        **semantic_terms,
        "S_II_star": induced_targets["S_II_star"],
        "S_TT_star": induced_targets["S_TT_star"],
        "L_pair": L_pair,
        "L_q": L_q,
        "L_bal": L_bal,
        "loss_total": loss_total,
    }


__all__ = ["LossConfig", "compute_total_loss"]
