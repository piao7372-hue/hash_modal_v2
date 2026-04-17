from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def _bounded_probability(name: str, value: Tensor, eps: float) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2D batch matrix.")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")
    if torch.any((value < 0.0) | (value > 1.0)):
        raise RuntimeError(f"{name} contains values outside [0, 1].")
    return value.clamp(min=eps, max=1.0 - eps)


def compute_semantic_loss(
    P_IT: Tensor,
    P_II: Tensor,
    P_TT: Tensor,
    S: Tensor,
    S_II_star: Tensor,
    S_TT_star: Tensor,
    alpha: float,
    eps: float,
) -> Dict[str, Tensor]:
    if alpha < 0.0:
        raise ValueError("alpha must be >= 0.")
    if P_IT.shape != S.shape:
        raise ValueError("P_IT and S must have identical shapes.")
    if P_II.shape != S_II_star.shape:
        raise ValueError("P_II and S_II_star must have identical shapes.")
    if P_TT.shape != S_TT_star.shape:
        raise ValueError("P_TT and S_TT_star must have identical shapes.")
    P_IT_bounded = _bounded_probability("P_IT", P_IT, eps)
    P_II_bounded = _bounded_probability("P_II", P_II, eps)
    P_TT_bounded = _bounded_probability("P_TT", P_TT, eps)
    S_bounded = _bounded_probability("S", S, eps)
    S_II_star_bounded = _bounded_probability("S_II_star", S_II_star, eps)
    S_TT_star_bounded = _bounded_probability("S_TT_star", S_TT_star, eps)
    L_IT = F.binary_cross_entropy(P_IT_bounded, S_bounded)
    L_II = F.binary_cross_entropy(P_II_bounded, S_II_star_bounded)
    L_TT = F.binary_cross_entropy(P_TT_bounded, S_TT_star_bounded)
    L_sem = L_IT + (alpha * 0.5 * (L_II + L_TT))
    for name, value in {"L_IT": L_IT, "L_II": L_II, "L_TT": L_TT, "L_sem": L_sem}.items():
        if not torch.isfinite(value):
            raise RuntimeError(f"{name} contains NaN or Inf.")
    return {"L_IT": L_IT, "L_II": L_II, "L_TT": L_TT, "L_sem": L_sem}


__all__ = ["compute_semantic_loss"]
