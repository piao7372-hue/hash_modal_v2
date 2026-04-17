from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class HnswBuildParams:
    M: int
    ef_construction: int
    ef_search: int


def require_hnswlib():
    try:
        import hnswlib  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Stage 3 requires hnswlib, but it is not installed in the active Python environment."
        ) from exc
    return hnswlib


def build_cosine_index(features: np.ndarray, params: HnswBuildParams):
    if features.ndim != 2:
        raise ValueError("ANN feature input must be rank-2.")
    if features.dtype != np.float32:
        raise ValueError("ANN feature input dtype must be float32.")
    hnswlib = require_hnswlib()
    index = hnswlib.Index(space="cosine", dim=int(features.shape[1]))
    index.init_index(
        max_elements=int(features.shape[0]),
        ef_construction=int(params.ef_construction),
        M=int(params.M),
    )
    labels = np.arange(features.shape[0], dtype=np.int32)
    index.add_items(features, labels)
    index.set_ef(int(params.ef_search))
    return index


def iter_knn_query(
    index,
    queries: np.ndarray,
    k: int,
    batch_size: int,
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
    if k <= 0:
        raise ValueError("k must be positive for ANN queries.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for ANN queries.")
    total_rows = int(queries.shape[0])
    for start in range(0, total_rows, batch_size):
        end = min(total_rows, start + batch_size)
        labels, distances = index.knn_query(queries[start:end], k=k)
        if labels.shape != (end - start, k):
            raise RuntimeError(
                "ANN labels shape mismatch: {0} != {1}".format(labels.shape, (end - start, k))
            )
        if distances.shape != (end - start, k):
            raise RuntimeError(
                "ANN distances shape mismatch: {0} != {1}".format(
                    distances.shape,
                    (end - start, k),
                )
            )
        if labels.size == 0 or distances.size == 0:
            raise RuntimeError("ANN returned an empty result block.")
        yield start, end, labels.astype(np.int32, copy=False), distances.astype(np.float32, copy=False)
