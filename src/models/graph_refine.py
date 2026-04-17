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
class GraphRefineConfig:
    input_dim: int
    f_dim: int
    hash_bits: int
    k_neighbors: int
    propagation_steps: int
    self_loop_weight: float
    residual_weight: float
    similarity_eps: float


class BatchLocalGraphRefiner(nn.Module):
    def __init__(self, config: GraphRefineConfig) -> None:
        super().__init__()
        if config.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1.")
        if config.propagation_steps < 1:
            raise ValueError("propagation_steps must be >= 1.")
        self.config = config
        self.feature_projection = nn.Linear(config.input_dim, config.f_dim)
        self.hash_projection = nn.Linear(config.f_dim, config.hash_bits)
        self.residual_projection = nn.Linear(config.f_dim, config.hash_bits)
        self.feature_norm = nn.LayerNorm(config.f_dim)

    def _build_graph(self, F_value: Tensor) -> tuple[Tensor, dict[str, Any]]:
        batch_size = int(F_value.shape[0])
        if batch_size <= self.config.k_neighbors:
            raise RuntimeError(
                f"Batch graph is invalid: batch_size={batch_size}, k_neighbors={self.config.k_neighbors}."
            )
        norms = torch.linalg.norm(F_value, dim=1, keepdim=True)
        if torch.any(norms <= self.config.similarity_eps):
            raise RuntimeError("Graph features contain a zero-norm row.")
        normalized = F_value / norms
        similarity = normalized @ normalized.transpose(0, 1)
        _ensure_finite("Graph similarity", similarity)
        masked = similarity.clone()
        masked.fill_diagonal_(-torch.inf)
        top_values, top_indices = torch.topk(masked, k=self.config.k_neighbors, dim=1)
        if not torch.isfinite(top_values).all():
            raise RuntimeError("Graph top-k selection produced invalid values.")
        adjacency = torch.zeros_like(similarity)
        neighbor_weights = torch.softmax(top_values, dim=1)
        adjacency.scatter_(1, top_indices, neighbor_weights)
        adjacency = adjacency + (torch.eye(batch_size, device=adjacency.device) * self.config.self_loop_weight)
        row_sums = adjacency.sum(dim=1, keepdim=True)
        if torch.any(row_sums <= 0.0):
            raise RuntimeError("Graph row sum is non-positive.")
        G_value = adjacency / row_sums
        _ensure_finite("G", G_value)
        summary = {
            "batch_size": batch_size,
            "k_neighbors": self.config.k_neighbors,
            "graph_shape": [batch_size, batch_size],
            "graph_density": float((G_value > 0.0).sum().item() / float(batch_size * batch_size)),
            "row_sum_min": float(G_value.sum(dim=1).min().item()),
            "row_sum_max": float(G_value.sum(dim=1).max().item()),
        }
        return G_value, summary

    def forward(self, Y_value: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
        if Y_value.ndim != 2:
            raise ValueError("Graph refiner expects a batch-first 2D tensor.")
        if Y_value.shape[1] != self.config.input_dim:
            raise ValueError("Graph refiner input dim mismatch.")
        _ensure_finite("Y", Y_value)
        F_value = self.feature_norm(F.gelu(self.feature_projection(Y_value)))
        _ensure_finite("F", F_value)
        G_value, graph_summary = self._build_graph(F_value)
        propagated = F_value
        for _ in range(self.config.propagation_steps):
            propagated = G_value @ propagated
            _ensure_finite("Graph propagated features", propagated)
        hash_from_graph = self.hash_projection(propagated)
        hash_from_residual = self.residual_projection(F_value)
        H_pre = (
            (1.0 - self.config.residual_weight) * hash_from_graph
            + self.config.residual_weight * hash_from_residual
        )
        H_value = torch.tanh(H_pre)
        _ensure_finite("H", H_value)
        summary = {
            **graph_summary,
            "F_shape": [int(value) for value in F_value.shape],
            "H_shape": [int(value) for value in H_value.shape],
            "H_min": float(H_value.min().item()),
            "H_max": float(H_value.max().item()),
        }
        return F_value, G_value, H_value, summary


class GraphRefine(nn.Module):
    def __init__(self, config: GraphRefineConfig) -> None:
        super().__init__()
        self.image_refiner = BatchLocalGraphRefiner(config)
        self.text_refiner = BatchLocalGraphRefiner(config)

    def forward(
        self,
        Y_I: Tensor,
        Y_T: Tensor,
        return_summary: bool = False,
    ) -> dict[str, Tensor | dict[str, Any] | None]:
        F_I, G_I, H_I, image_summary = self.image_refiner(Y_I)
        F_T, G_T, H_T, text_summary = self.text_refiner(Y_T)
        summary = None
        if return_summary:
            summary = {
                "image": image_summary,
                "text": text_summary,
            }
        return {
            "F_I": F_I,
            "F_T": F_T,
            "G_I": G_I,
            "G_T": G_T,
            "H_I": H_I,
            "H_T": H_T,
            "summary": summary,
        }


__all__ = ["GraphRefine", "GraphRefineConfig"]
