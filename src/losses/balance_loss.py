from __future__ import annotations

import torch
from torch import Tensor


def compute_balance_loss(H_I: Tensor, H_T: Tensor) -> Tensor:
    if H_I.ndim != 2 or H_T.ndim != 2:
        raise ValueError("Balance loss expects batch-first 2D tensors.")
    if H_I.shape[1] != H_T.shape[1]:
        raise ValueError("H_I and H_T bit dimensions must match.")
    if not torch.isfinite(H_I).all() or not torch.isfinite(H_T).all():
        raise RuntimeError("H_I or H_T contains NaN or Inf.")
    image_bit_mean = torch.mean(H_I, dim=0)
    text_bit_mean = torch.mean(H_T, dim=0)
    L_bal = 0.5 * (torch.mean(image_bit_mean**2) + torch.mean(text_bit_mean**2))
    if not torch.isfinite(L_bal):
        raise RuntimeError("L_bal contains NaN or Inf.")
    return L_bal


__all__ = ["compute_balance_loss"]
