from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _normalize_supervision_profiles(name: str, S_value: Tensor, eps: float) -> Tensor:
    if S_value.ndim != 2:
        raise ValueError(f"{name} must be a 2D batch matrix.")
    if S_value.shape[0] != S_value.shape[1]:
        raise ValueError(f"{name} must be square inside the current batch.")
    if not torch.isfinite(S_value).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")
    if torch.any((S_value < 0.0) | (S_value > 1.0)):
        raise RuntimeError(f"{name} contains values outside [0, 1].")
    norms = torch.linalg.norm(S_value, dim=1, keepdim=True)
    if torch.any(norms <= eps):
        raise RuntimeError(f"{name} contains a zero supervision profile.")
    return S_value / norms


def build_induced_targets(S: Tensor, eps: float) -> Dict[str, Tensor]:
    image_profiles = _normalize_supervision_profiles("S", S, eps)
    text_profiles = _normalize_supervision_profiles("S_transposed", S.transpose(0, 1), eps)
    S_II_star = torch.clamp(image_profiles @ image_profiles.transpose(0, 1), min=0.0, max=1.0)
    S_TT_star = torch.clamp(text_profiles @ text_profiles.transpose(0, 1), min=0.0, max=1.0)
    if not torch.isfinite(S_II_star).all() or not torch.isfinite(S_TT_star).all():
        raise RuntimeError("Induced targets contain NaN or Inf.")
    return {"S_II_star": S_II_star, "S_TT_star": S_TT_star}


__all__ = ["build_induced_targets"]
