from __future__ import annotations

import torch
from torch import Tensor, nn


class HashBinarize(nn.Module):
    def forward(self, H_value: Tensor) -> Tensor:
        if H_value.ndim != 2:
            raise ValueError("HashBinarize expects a batch-first 2D tensor.")
        if not torch.isfinite(H_value).all():
            raise RuntimeError("H contains NaN or Inf before binarization.")
        B_value = torch.where(H_value >= 0.0, torch.ones_like(H_value), -torch.ones_like(H_value))
        unique_values = torch.unique(B_value)
        if not torch.all((unique_values == -1.0) | (unique_values == 1.0)):
            raise RuntimeError("B contains values outside {-1, +1}.")
        return B_value


__all__ = ["HashBinarize"]
