"""Stage 3 formal semantic relation interfaces."""

from .confidence import (
    build_confidence_from_probabilities,
    compute_confidence,
    stable_col_softmax_sparse,
    stable_row_softmax_sparse,
)
from .direct_support import (
    build_cross_modal_candidate_support,
    compute_direct_support,
    validate_feature_inputs,
)
from .final_supervision import (
    build_S,
    build_S_tilde,
    build_final_support,
    restrict_to_final_support,
)
from .semantic_cache import (
    FormalSemanticRelationConfig,
    build_run_summary,
    build_validator_summary,
    load_semantic_relation_config,
    run_formal_semantic_relation,
    write_semantic_cache,
    write_semantic_validator_summary,
)
from .structural_support import (
    build_intra_modal_profiles,
    compute_structural_support,
    sparse_profile_cosine,
)

__all__ = [
    "FormalSemanticRelationConfig",
    "build_S",
    "build_S_tilde",
    "build_confidence_from_probabilities",
    "build_cross_modal_candidate_support",
    "build_final_support",
    "build_intra_modal_profiles",
    "build_run_summary",
    "build_validator_summary",
    "compute_confidence",
    "compute_direct_support",
    "compute_structural_support",
    "load_semantic_relation_config",
    "restrict_to_final_support",
    "run_formal_semantic_relation",
    "sparse_profile_cosine",
    "stable_col_softmax_sparse",
    "stable_row_softmax_sparse",
    "validate_feature_inputs",
    "write_semantic_cache",
    "write_semantic_validator_summary",
]
