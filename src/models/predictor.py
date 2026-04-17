from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from src.losses import build_relation_predictions

from .chebykan import ChebyKAN, ChebyKANConfig
from .graph_refine import GraphRefine, GraphRefineConfig
from .hash_binarize import HashBinarize
from .semantic_tree import SemanticTree, SemanticTreeConfig


@dataclass(frozen=True)
class PredictorConfig:
    relation_eps: float


class FormalHashModel(nn.Module):
    def __init__(
        self,
        chebykan_config: ChebyKANConfig,
        tree_config: SemanticTreeConfig,
        graph_config: GraphRefineConfig,
        predictor_config: PredictorConfig,
    ) -> None:
        super().__init__()
        self.chebykan = ChebyKAN(chebykan_config)
        self.semantic_tree = SemanticTree(tree_config)
        self.graph_refine = GraphRefine(graph_config)
        self.hash_binarize = HashBinarize()
        self.predictor_config = predictor_config

    def forward(self, X_I: Tensor, X_T: Tensor) -> dict[str, Tensor | dict[str, Any]]:
        Z_I, Z_T = self.chebykan(X_I, X_T)
        Y_I, Y_T, tree_summary = self.semantic_tree(Z_I, Z_T, return_summary=True)
        graph_outputs = self.graph_refine(Y_I, Y_T, return_summary=True)
        H_I = graph_outputs["H_I"]
        H_T = graph_outputs["H_T"]
        B_I = self.hash_binarize(H_I)
        B_T = self.hash_binarize(H_T)
        predictions = build_relation_predictions(
            H_I=H_I,
            H_T=H_T,
            eps=self.predictor_config.relation_eps,
        )
        return {
            "Z_I": Z_I,
            "Z_T": Z_T,
            "Y_I": Y_I,
            "Y_T": Y_T,
            "F_I": graph_outputs["F_I"],
            "F_T": graph_outputs["F_T"],
            "G_I": graph_outputs["G_I"],
            "G_T": graph_outputs["G_T"],
            "H_I": H_I,
            "H_T": H_T,
            "B_I": B_I,
            "B_T": B_T,
            "P_IT": predictions["P_IT"],
            "P_II": predictions["P_II"],
            "P_TT": predictions["P_TT"],
            "model_summary": {
                "Z_I_shape": [int(value) for value in Z_I.shape],
                "Z_T_shape": [int(value) for value in Z_T.shape],
                "Y_I_shape": [int(value) for value in Y_I.shape],
                "Y_T_shape": [int(value) for value in Y_T.shape],
                "F_I_shape": [int(value) for value in graph_outputs["F_I"].shape],
                "F_T_shape": [int(value) for value in graph_outputs["F_T"].shape],
                "H_I_shape": [int(value) for value in H_I.shape],
                "H_T_shape": [int(value) for value in H_T.shape],
                "tree": tree_summary,
                "graph": graph_outputs["summary"],
            },
        }


__all__ = ["FormalHashModel", "PredictorConfig"]
