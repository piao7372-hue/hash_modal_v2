from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def _stable_softmax_axis_sparse(matrix: csr_matrix | csc_matrix, tau: float) -> csr_matrix | csc_matrix:
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive.")
    if matrix.nnz == 0:
        raise RuntimeError("Sparse softmax input must not be empty.")
    output = matrix.copy()
    for row_index in range(matrix.shape[0]):
        row_start = int(matrix.indptr[row_index])
        row_end = int(matrix.indptr[row_index + 1])
        segment = matrix.data[row_start:row_end]
        if segment.size == 0:
            raise RuntimeError("Sparse softmax encountered an empty axis slice.")
        logits = segment.astype(np.float64) / tau
        max_logit = float(np.max(logits))
        shifted = logits - max_logit
        exp_shifted = np.exp(shifted)
        sum_exp = float(np.sum(exp_shifted))
        if not np.isfinite(sum_exp) or sum_exp <= 0.0:
            raise RuntimeError("Sparse softmax encountered an invalid denominator.")
        log_denom = max_logit + np.log(sum_exp)
        probabilities = np.exp(logits - log_denom).astype(np.float32)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Sparse softmax produced NaN or Inf.")
        output.data[row_start:row_end] = probabilities
    return output


def stable_row_softmax_sparse(matrix: csr_matrix, tau: float) -> csr_matrix:
    if not isinstance(matrix, csr_matrix):
        raise TypeError("stable_row_softmax_sparse expects csr_matrix input.")
    matrix = matrix.tocsr()
    matrix.sort_indices()
    return _stable_softmax_axis_sparse(matrix, tau).tocsr()


def stable_col_softmax_sparse(matrix: csr_matrix, tau: float) -> csr_matrix:
    if not isinstance(matrix, csr_matrix):
        raise TypeError("stable_col_softmax_sparse expects csr_matrix input.")
    matrix = matrix.tocsr()
    matrix.sort_indices()
    column_major = matrix.tocsc()
    column_major.sort_indices()
    softmax_column_major = _stable_softmax_axis_sparse(column_major, tau)
    return softmax_column_major.tocsr()


def build_confidence_from_probabilities(
    P_I_to_T: csr_matrix,
    P_T_to_I: csr_matrix,
) -> csr_matrix:
    product = P_I_to_T.multiply(P_T_to_I).tocsr()
    if product.nnz == 0:
        raise RuntimeError("Confidence support became empty.")
    product.data = np.sqrt(product.data.astype(np.float64)).astype(np.float32)
    if not np.all((product.data > 0.0) & (product.data <= 1.0)):
        raise ValueError("Confidence values must lie in (0, 1].")
    return product


def compute_confidence(
    S_tilde: csr_matrix,
    tau: float,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    P_I_to_T = stable_row_softmax_sparse(S_tilde, tau)
    P_T_to_I = stable_col_softmax_sparse(S_tilde, tau)
    C = build_confidence_from_probabilities(P_I_to_T, P_T_to_I)
    return P_I_to_T, P_T_to_I, C
