from __future__ import annotations

import torch
from torch import Tensor


def compute_quantization_loss(H_I: Tensor, H_T: Tensor) -> Tensor:
    if H_I.shape != H_T.shape:
        raise ValueError("H_I and H_T must have identical shapes.")
    if not torch.isfinite(H_I).all() or not torch.isfinite(H_T).all():
        raise RuntimeError("H_I or H_T contains NaN or Inf.")
    target_I = torch.ones_like(H_I)
    target_T = torch.ones_like(H_T)
    L_q = 0.5 * (torch.mean((H_I.abs() - target_I) ** 2) + torch.mean((H_T.abs() - target_T) ** 2))
    if not torch.isfinite(L_q):
        raise RuntimeError("L_q contains NaN or Inf.")
    return L_q


__all__ = ["compute_quantization_loss"]
