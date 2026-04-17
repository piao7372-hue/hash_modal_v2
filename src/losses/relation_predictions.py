from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _normalize_rows(name: str, value: Tensor, eps: float) -> Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be batch-first [batch, dim].")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")
    norms = torch.linalg.norm(value, dim=1, keepdim=True)
    if torch.any(norms <= eps):
        raise RuntimeError(f"{name} contains a zero-norm row.")
    return value / norms


def _cosine_to_probability(left: Tensor, right: Tensor, eps: float) -> Tensor:
    left_normalized = _normalize_rows("left", left, eps)
    right_normalized = _normalize_rows("right", right, eps)
    similarity = left_normalized @ right_normalized.transpose(0, 1)
    probability = torch.clamp((similarity + 1.0) * 0.5, min=0.0, max=1.0)
    if not torch.isfinite(probability).all():
        raise RuntimeError("Relation probability contains NaN or Inf.")
    return probability


def build_relation_predictions(H_I: Tensor, H_T: Tensor, eps: float) -> Dict[str, Tensor]:
    if H_I.shape != H_T.shape:
        raise ValueError("H_I and H_T shapes must match.")
    P_IT = _cosine_to_probability(H_I, H_T, eps)
    P_II = _cosine_to_probability(H_I, H_I, eps)
    P_TT = _cosine_to_probability(H_T, H_T, eps)
    return {"P_IT": P_IT, "P_II": P_II, "P_TT": P_TT}


__all__ = ["build_relation_predictions"]
