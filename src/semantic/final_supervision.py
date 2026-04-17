from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, eye


def assert_same_support(left: csr_matrix, right: csr_matrix) -> None:
    if not isinstance(left, csr_matrix) or not isinstance(right, csr_matrix):
        raise TypeError("Both sparse matrices must be csr_matrix instances.")
    if left.shape != right.shape:
        raise ValueError("Sparse matrix shapes do not match.")
    if not np.array_equal(left.indptr, right.indptr):
        raise ValueError("Sparse matrix row pointers do not match.")
    if not np.array_equal(left.indices, right.indices):
        raise ValueError("Sparse matrix support indices do not match.")


def build_S_tilde(A: csr_matrix, R: csr_matrix, relation_lambda: float) -> csr_matrix:
    assert_same_support(A, R)
    values = relation_lambda * A.data.astype(np.float32) + (1.0 - relation_lambda) * R.data.astype(np.float32)
    np.clip(values, 0.0, 1.0, out=values)
    # S_tilde is the Python identifier for the formal mathematical symbol S̃.
    return csr_matrix((values, A.indices, A.indptr), shape=A.shape, copy=False)


def build_S(C: csr_matrix, S_tilde: csr_matrix) -> csr_matrix:
    assert_same_support(C, S_tilde)
    values = C.data.astype(np.float32) * S_tilde.data.astype(np.float32)
    np.clip(values, 0.0, 1.0, out=values)
    return csr_matrix((values, C.indices, C.indptr), shape=C.shape, copy=False)


def _row_topk_support(matrix: csr_matrix, topk: int) -> csr_matrix:
    if not isinstance(matrix, csr_matrix):
        raise TypeError("_row_topk_support expects csr_matrix input.")
    total_rows = int(matrix.shape[0])
    nnz_upper_bound = total_rows * topk
    indices = np.empty(nnz_upper_bound, dtype=np.int32)
    indptr = np.zeros(total_rows + 1, dtype=np.int64)
    cursor = 0
    for row_index in range(total_rows):
        row_start = int(matrix.indptr[row_index])
        row_end = int(matrix.indptr[row_index + 1])
        row_cols = matrix.indices[row_start:row_end]
        row_vals = matrix.data[row_start:row_end]
        if row_cols.size == 0:
            raise RuntimeError("Final top-k support encountered an empty row.")
        keep_count = min(topk, row_cols.size)
        order = np.lexsort((row_cols, -row_vals))
        selected_cols = np.sort(row_cols[order[:keep_count]].astype(np.int32, copy=False))
        next_cursor = cursor + int(selected_cols.size)
        indices[cursor:next_cursor] = selected_cols
        cursor = next_cursor
        indptr[row_index + 1] = cursor
    data = np.ones(cursor, dtype=np.float32)
    return csr_matrix((data, indices[:cursor], indptr), shape=matrix.shape)


def build_final_support(S: csr_matrix, final_topk: int) -> csr_matrix:
    row_top = _row_topk_support(S.tocsr(), final_topk)
    col_top = _row_topk_support(S.transpose().tocsr(), final_topk).transpose().tocsr()
    diagonal = eye(S.shape[0], dtype=np.float32, format="csr")
    final_support = (row_top + col_top + diagonal).tocsr()
    final_support.sum_duplicates()
    final_support.sort_indices()
    final_support.data = np.ones(final_support.nnz, dtype=np.float32)
    return final_support


def restrict_to_final_support(matrix: csr_matrix, final_support: csr_matrix) -> csr_matrix:
    if matrix.shape != final_support.shape:
        raise ValueError("final_support shape does not match the source matrix.")
    matrix = matrix.tocsr()
    matrix.sort_indices()
    final_support = final_support.tocsr()
    final_support.sort_indices()
    values = np.empty(final_support.nnz, dtype=np.float32)
    for row_index in range(final_support.shape[0]):
        source_start = int(matrix.indptr[row_index])
        source_end = int(matrix.indptr[row_index + 1])
        source_cols = matrix.indices[source_start:source_end]
        source_vals = matrix.data[source_start:source_end]
        target_start = int(final_support.indptr[row_index])
        target_end = int(final_support.indptr[row_index + 1])
        target_cols = final_support.indices[target_start:target_end]
        positions = np.searchsorted(source_cols, target_cols)
        if np.any(positions >= source_cols.size):
            raise RuntimeError("Final support is not a subset of the source support.")
        if not np.array_equal(source_cols[positions], target_cols):
            raise RuntimeError("Final support is not aligned with the source support.")
        values[target_start:target_end] = source_vals[positions].astype(np.float32, copy=False)
    return csr_matrix((values, final_support.indices, final_support.indptr), shape=matrix.shape, copy=False)
