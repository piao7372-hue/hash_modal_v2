from __future__ import annotations

import torch
from torch import Tensor


def compute_pair_loss(H_I: Tensor, H_T: Tensor) -> Tensor:
    if H_I.shape != H_T.shape:
        raise ValueError("H_I and H_T must have identical shapes.")
    if not torch.isfinite(H_I).all() or not torch.isfinite(H_T).all():
        raise RuntimeError("H_I or H_T contains NaN or Inf.")
    L_pair = torch.mean((H_I - H_T) ** 2)
    if not torch.isfinite(L_pair):
        raise RuntimeError("L_pair contains NaN or Inf.")
    return L_pair


__all__ = ["compute_pair_loss"]
