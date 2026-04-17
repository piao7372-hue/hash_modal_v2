from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _ensure_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


@dataclass(frozen=True)
class SemanticTreeConfig:
    tree_depth: int
    prototype_counts: tuple[int, ...]
    feature_dim: int
    dropout: float
    eps: float


class SemanticTreeLayer(nn.Module):
    def __init__(self, feature_dim: int, prototype_count: int, dropout: float) -> None:
        super().__init__()
        if prototype_count < 1:
            raise ValueError("prototype_count must be >= 1.")
        self.feature_dim = feature_dim
        self.prototype_count = prototype_count
        self.input_projection = nn.Linear(feature_dim, feature_dim)
        self.prototypes = nn.Parameter(torch.randn(prototype_count, feature_dim) * 0.02)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, dict[str, Any]]:
        projected = self.input_projection(x)
        _ensure_finite("SemanticTree projected", projected)
        logits = projected @ self.prototypes.transpose(0, 1)
        logits = logits / (self.feature_dim**0.5)
        attention = torch.softmax(logits, dim=-1)
        injected = attention @ self.prototypes
        output = self.norm(projected + self.dropout(self.output_projection(injected)))
        _ensure_finite("SemanticTree output", output)
        summary = {
            "input_shape": [int(value) for value in x.shape],
            "attention_shape": [int(value) for value in attention.shape],
            "prototype_shape": [int(value) for value in self.prototypes.shape],
            "output_shape": [int(value) for value in output.shape],
        }
        return output, summary


class ModalitySemanticTree(nn.Module):
    def __init__(self, config: SemanticTreeConfig) -> None:
        super().__init__()
        if config.tree_depth not in (2, 3):
            raise ValueError("tree_depth must be 2 or 3 for formal Stage 4.")
        if len(config.prototype_counts) != config.tree_depth:
            raise ValueError("prototype_counts length must match tree_depth.")
        self.layers = nn.ModuleList(
            [
                SemanticTreeLayer(
                    feature_dim=config.feature_dim,
                    prototype_count=prototype_count,
                    dropout=config.dropout,
                )
                for prototype_count in config.prototype_counts
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[dict[str, Any]]]:
        summaries: list[dict[str, Any]] = []
        output = x
        for layer_index, layer in enumerate(self.layers, start=1):
            output, layer_summary = layer(output)
            layer_summary["layer_index"] = layer_index
            summaries.append(layer_summary)
        return output, summaries


class SemanticTree(nn.Module):
    def __init__(self, config: SemanticTreeConfig) -> None:
        super().__init__()
        self.config = config
        self.image_tree = ModalitySemanticTree(config)
        self.text_tree = ModalitySemanticTree(config)

    def forward(
        self,
        Z_I: Tensor,
        Z_T: Tensor,
        return_summary: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Any] | None]:
        if Z_I.ndim != 2 or Z_T.ndim != 2:
            raise ValueError("SemanticTree inputs must be batch-first 2D tensors.")
        if Z_I.shape[1] != self.config.feature_dim or Z_T.shape[1] != self.config.feature_dim:
            raise ValueError("SemanticTree feature dim mismatch.")
        _ensure_finite("Z_I", Z_I)
        _ensure_finite("Z_T", Z_T)
        Y_I, image_summary = self.image_tree(Z_I)
        Y_T, text_summary = self.text_tree(Z_T)
        _ensure_finite("Y_I", Y_I)
        _ensure_finite("Y_T", Y_T)
        summary = None
        if return_summary:
            summary = {
                "tree_depth": self.config.tree_depth,
                "prototype_counts": list(self.config.prototype_counts),
                "image_layers": image_summary,
                "text_layers": text_summary,
            }
        return Y_I, Y_T, summary


__all__ = ["SemanticTree", "SemanticTreeConfig"]
