from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _ensure_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


@dataclass(frozen=True)
class ChebyKANConfig:
    input_dim_image: int
    input_dim_text: int
    d_z: int
    polynomial_order: int
    hidden_dims: tuple[int, ...]
    normalize_inputs: bool
    input_clip_value: float
    eps: float


class ChebyshevBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, polynomial_order: int) -> None:
        super().__init__()
        if polynomial_order < 1:
            raise ValueError("polynomial_order must be >= 1.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.polynomial_order = polynomial_order
        self.projection = nn.Linear(input_dim * (polynomial_order + 1), output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError("ChebyshevBlock expects a batch-first 2D tensor.")
        basis_terms = [torch.ones_like(x), x]
        for _ in range(2, self.polynomial_order + 1):
            basis_terms.append((2.0 * x * basis_terms[-1]) - basis_terms[-2])
        expanded = torch.cat(basis_terms, dim=-1)
        output = self.norm(F.gelu(self.projection(expanded)))
        _ensure_finite("ChebyshevBlock output", output)
        return output


class ModalityChebyKAN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], d_z: int, polynomial_order: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims, d_z]
        for index in range(len(dims) - 1):
            layers.append(ChebyshevBlock(dims[index], dims[index + 1], polynomial_order))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        output = x
        for layer in self.layers:
            output = layer(output)
        _ensure_finite("ModalityChebyKAN output", output)
        return output


class ChebyKAN(nn.Module):
    def __init__(self, config: ChebyKANConfig) -> None:
        super().__init__()
        if config.input_clip_value <= 0.0:
            raise ValueError("input_clip_value must be > 0.")
        self.config = config
        self.image_encoder = ModalityChebyKAN(
            input_dim=config.input_dim_image,
            hidden_dims=config.hidden_dims,
            d_z=config.d_z,
            polynomial_order=config.polynomial_order,
        )
        self.text_encoder = ModalityChebyKAN(
            input_dim=config.input_dim_text,
            hidden_dims=config.hidden_dims,
            d_z=config.d_z,
            polynomial_order=config.polynomial_order,
        )

    def _prepare_inputs(self, name: str, x: Tensor, expected_dim: int) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"{name} must be batch-first [batch, dim].")
        if x.shape[1] != expected_dim:
            raise ValueError(f"{name} feature dim mismatch: {x.shape[1]} != {expected_dim}.")
        _ensure_finite(name, x)
        if self.config.normalize_inputs:
            norms = torch.linalg.norm(x, dim=1, keepdim=True)
            if torch.any(norms <= self.config.eps):
                raise RuntimeError(f"{name} contains a zero-norm row.")
            x = x / norms
        x = torch.clamp(x, min=-self.config.input_clip_value, max=self.config.input_clip_value)
        _ensure_finite(f"{name} after normalization and clipping", x)
        return x

    def forward(self, X_I: Tensor, X_T: Tensor) -> tuple[Tensor, Tensor]:
        X_I = self._prepare_inputs("X_I", X_I, self.config.input_dim_image)
        X_T = self._prepare_inputs("X_T", X_T, self.config.input_dim_text)
        Z_I = self.image_encoder(X_I)
        Z_T = self.text_encoder(X_T)
        if Z_I.shape[1] != self.config.d_z or Z_T.shape[1] != self.config.d_z:
            raise RuntimeError("ChebyKAN output dim does not match configured d_z.")
        _ensure_finite("Z_I", Z_I)
        _ensure_finite("Z_T", Z_T)
        return Z_I, Z_T


__all__ = ["ChebyKAN", "ChebyKANConfig"]
