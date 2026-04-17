from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, eye

from .ann_utils import HnswBuildParams, build_cosine_index, iter_knn_query


def validate_feature_inputs(
    X_I: np.ndarray,
    X_T: np.ndarray,
    filtered_count: int,
    l2_norm_atol: float,
) -> Tuple[int, int]:
    if X_I.ndim != 2 or X_T.ndim != 2:
        raise ValueError("Stage 3 feature inputs must be rank-2 arrays.")
    if X_I.shape[0] != X_T.shape[0]:
        raise ValueError("X_I.shape[0] must equal X_T.shape[0].")
    if X_I.shape[0] != filtered_count or X_T.shape[0] != filtered_count:
        raise ValueError("Stage 3 feature row count does not match manifest filtered_count.")
    if X_I.dtype != np.float32 or X_T.dtype != np.float32:
        raise ValueError("Stage 3 feature dtype must be float32.")
    if not np.isfinite(X_I).all() or not np.isfinite(X_T).all():
        raise ValueError("Stage 3 feature inputs contain NaN or Inf.")
    X_I_norms = np.linalg.norm(X_I, axis=1)
    X_T_norms = np.linalg.norm(X_T, axis=1)
    if not np.allclose(X_I_norms, np.ones_like(X_I_norms), atol=l2_norm_atol, rtol=0.0):
        raise ValueError("X_I row norms are not close to 1.0.")
    if not np.allclose(X_T_norms, np.ones_like(X_T_norms), atol=l2_norm_atol, rtol=0.0):
        raise ValueError("X_T row norms are not close to 1.0.")
    return int(X_I.shape[0]), int(X_I.shape[1])


def _build_directional_support(
    query_features: np.ndarray,
    index_features: np.ndarray,
    topk: int,
    ann_params: HnswBuildParams,
    batch_size: int,
) -> csr_matrix:
    total_rows = int(query_features.shape[0])
    if topk > int(index_features.shape[0]):
        raise ValueError("topk exceeds the searchable corpus size.")
    index = build_cosine_index(index_features, ann_params)
    flat_indices = np.empty(total_rows * topk, dtype=np.int32)
    cursor = 0
    for _, _, labels, _ in iter_knn_query(index, query_features, k=topk, batch_size=batch_size):
        batch_flat = labels.reshape(-1)
        next_cursor = cursor + int(batch_flat.size)
        flat_indices[cursor:next_cursor] = batch_flat
        cursor = next_cursor
    if cursor != total_rows * topk:
        raise RuntimeError("Directional ANN support size mismatch.")
    data = np.ones(cursor, dtype=np.float32)
    indptr = np.arange(0, cursor + 1, topk, dtype=np.int64)
    matrix = csr_matrix((data, flat_indices, indptr), shape=(total_rows, int(index_features.shape[0])))
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def build_cross_modal_candidate_support(
    X_I: np.ndarray,
    X_T: np.ndarray,
    direct_topk: int,
    ann_params: HnswBuildParams,
    batch_size: int = 2048,
) -> csr_matrix:
    total_rows = int(X_I.shape[0])
    forward = _build_directional_support(
        query_features=X_I,
        index_features=X_T,
        topk=direct_topk,
        ann_params=ann_params,
        batch_size=batch_size,
    )
    reverse = _build_directional_support(
        query_features=X_T,
        index_features=X_I,
        topk=direct_topk,
        ann_params=ann_params,
        batch_size=batch_size,
    )
    diagonal = eye(total_rows, dtype=np.float32, format="csr")
    candidate_support = (forward + reverse.transpose(copy=False) + diagonal).tocsr()
    candidate_support.sum_duplicates()
    candidate_support.sort_indices()
    candidate_support.data = np.ones(candidate_support.nnz, dtype=np.float32)
    if candidate_support.nnz == 0:
        raise RuntimeError("Candidate support is empty.")
    return candidate_support


def compute_direct_support(
    X_I: np.ndarray,
    X_T: np.ndarray,
    support: csr_matrix,
) -> csr_matrix:
    if not isinstance(support, csr_matrix):
        raise TypeError("support must be a csr_matrix.")
    if support.shape != (X_I.shape[0], X_T.shape[0]):
        raise ValueError("support shape does not match X_I/X_T.")
    support.sort_indices()
    data = np.empty(support.nnz, dtype=np.float32)
    for row_index in range(support.shape[0]):
        row_start = int(support.indptr[row_index])
        row_end = int(support.indptr[row_index + 1])
        cols = support.indices[row_start:row_end]
        if cols.size == 0:
            raise RuntimeError("Candidate support contains an empty row.")
        row_vector = X_I[row_index]
        cosine_values = X_T[cols] @ row_vector
        support_values = (1.0 + cosine_values.astype(np.float32, copy=False)) * 0.5
        np.clip(support_values, 0.0, 1.0, out=support_values)
        data[row_start:row_end] = support_values
    return csr_matrix((data, support.indices, support.indptr), shape=support.shape, copy=False)
