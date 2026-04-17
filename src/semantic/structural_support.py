from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from .ann_utils import HnswBuildParams, build_cosine_index, iter_knn_query


def build_intra_modal_profiles(
    features: np.ndarray,
    intra_topk: int,
    ann_params: HnswBuildParams,
    batch_size: int = 2048,
) -> csr_matrix:
    total_rows = int(features.shape[0])
    if intra_topk > total_rows:
        raise ValueError("intra_topk exceeds the searchable corpus size.")
    index = build_cosine_index(features, ann_params)
    flat_indices = np.empty(total_rows * intra_topk, dtype=np.int32)
    flat_values = np.empty(total_rows * intra_topk, dtype=np.float32)
    cursor = 0
    query_k = min(total_rows, intra_topk + 1)
    for start, end, labels, distances in iter_knn_query(index, features, k=query_k, batch_size=batch_size):
        similarities = 1.0 - distances
        local_rows = np.arange(start, end, dtype=np.int32)
        selected_labels = np.empty((end - start, intra_topk), dtype=np.int32)
        selected_sims = np.empty((end - start, intra_topk), dtype=np.float32)
        for batch_row in range(end - start):
            row_id = local_rows[batch_row]
            row_labels = labels[batch_row]
            row_sims = similarities[batch_row].astype(np.float32, copy=False)
            self_hits = np.flatnonzero(row_labels == row_id)
            if self_hits.size == 0:
                if row_labels.shape[0] < intra_topk - 1:
                    raise RuntimeError("Intra-modal ANN profile size is too small to inject self.")
                chosen_labels = np.empty(intra_topk, dtype=np.int32)
                chosen_sims = np.empty(intra_topk, dtype=np.float32)
                chosen_labels[:-1] = row_labels[: intra_topk - 1]
                chosen_sims[:-1] = row_sims[: intra_topk - 1]
                chosen_labels[-1] = row_id
                chosen_sims[-1] = 1.0
            else:
                self_position = int(self_hits[0])
                keep_mask = np.ones(row_labels.shape[0], dtype=bool)
                if row_labels.shape[0] > intra_topk:
                    if self_position >= intra_topk:
                        keep_mask[intra_topk - 1 :] = False
                        keep_mask[self_position] = True
                    else:
                        keep_mask[intra_topk:] = False
                chosen_labels = row_labels[keep_mask]
                chosen_sims = row_sims[keep_mask]
                if chosen_labels.shape[0] != intra_topk:
                    raise RuntimeError("Intra-modal profile selection did not preserve intra_topk.")
            chosen_sims[chosen_labels == row_id] = 1.0
            selected_labels[batch_row] = chosen_labels
            selected_sims[batch_row] = chosen_sims
        batch_size_rows = end - start
        batch_flat_size = batch_size_rows * intra_topk
        next_cursor = cursor + batch_flat_size
        flat_indices[cursor:next_cursor] = selected_labels.reshape(-1)
        flat_values[cursor:next_cursor] = selected_sims.reshape(-1)
        cursor = next_cursor
    if cursor != total_rows * intra_topk:
        raise RuntimeError("Intra-modal profile size mismatch.")
    indptr = np.arange(0, cursor + 1, intra_topk, dtype=np.int64)
    profiles = csr_matrix((flat_values, flat_indices, indptr), shape=(total_rows, total_rows))
    profiles.sum_duplicates()
    profiles.sort_indices()
    return profiles


def sparse_profile_cosine(
    support_cols: np.ndarray,
    product_cols: np.ndarray,
    product_values: np.ndarray,
) -> np.ndarray:
    cosine_values = np.zeros(support_cols.shape[0], dtype=np.float32)
    if support_cols.size == 0 or product_cols.size == 0:
        return cosine_values
    positions = np.searchsorted(product_cols, support_cols)
    valid = positions < product_cols.size
    if not np.any(valid):
        return cosine_values
    matched_positions = positions[valid]
    matched_cols = support_cols[valid]
    exact = product_cols[matched_positions] == matched_cols
    if np.any(exact):
        cosine_values[np.flatnonzero(valid)[exact]] = product_values[matched_positions[exact]]
    return cosine_values


def _row_l2_normalize(matrix: csr_matrix) -> csr_matrix:
    squared_norms = np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1)
    if np.any(squared_norms <= 0.0):
        raise RuntimeError("Sparse profile row norm must be positive.")
    inv_norms = 1.0 / np.sqrt(squared_norms.astype(np.float64))
    return diags(inv_norms.astype(np.float32)) @ matrix


def compute_structural_support(
    candidate_support: csr_matrix,
    image_profiles: csr_matrix,
    text_profiles: csr_matrix,
    batch_size: int = 256,
) -> csr_matrix:
    if not isinstance(candidate_support, csr_matrix):
        raise TypeError("candidate_support must be a csr_matrix.")
    image_profiles = image_profiles.tocsr()
    text_profiles = text_profiles.tocsr()
    candidate_support.sort_indices()
    image_profiles.sort_indices()
    text_profiles.sort_indices()

    normalized_image_profiles = _row_l2_normalize(image_profiles).tocsr()
    normalized_text_profiles_t = _row_l2_normalize(text_profiles).transpose().tocsc()

    values = np.empty(candidate_support.nnz, dtype=np.float32)
    total_rows = int(candidate_support.shape[0])
    for start in range(0, total_rows, batch_size):
        end = min(total_rows, start + batch_size)
        cosine_block = normalized_image_profiles[start:end].dot(normalized_text_profiles_t).tocsr()
        cosine_block.sort_indices()
        for local_row in range(end - start):
            global_row = start + local_row
            support_start = int(candidate_support.indptr[global_row])
            support_end = int(candidate_support.indptr[global_row + 1])
            support_cols = candidate_support.indices[support_start:support_end]

            product_start = int(cosine_block.indptr[local_row])
            product_end = int(cosine_block.indptr[local_row + 1])
            product_cols = cosine_block.indices[product_start:product_end]
            product_vals = cosine_block.data[product_start:product_end].astype(np.float32, copy=False)

            cosine_values = sparse_profile_cosine(support_cols, product_cols, product_vals)
            np.clip(cosine_values, -1.0, 1.0, out=cosine_values)
            support_values = (1.0 + cosine_values) * 0.5
            np.clip(support_values, 0.0, 1.0, out=support_values)
            values[support_start:support_end] = support_values
    return csr_matrix((values, candidate_support.indices, candidate_support.indptr), shape=candidate_support.shape, copy=False)
