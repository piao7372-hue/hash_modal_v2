from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass(frozen=True)
class SplitResult:
    seed: int
    ordered_sample_ids: tuple[str, ...]
    permuted_sample_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    retrieval_ids: tuple[str, ...]
    train_ids: tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "counts": {
                "ordered_total": len(self.ordered_sample_ids),
                "query": len(self.query_ids),
                "retrieval": len(self.retrieval_ids),
                "train": len(self.train_ids),
            },
            "query_ids_path_semantics": "query split ids",
            "retrieval_ids_path_semantics": "retrieval split ids",
            "train_ids_path_semantics": "train split ids; must be a subset of retrieval ids",
        }


def build_query_retrieval_train_splits(
    sample_ids: Sequence[str],
    query_size: int,
    train_size: int,
    seed: int,
) -> SplitResult:
    if query_size <= 0:
        raise ValueError("query_size must be positive.")
    if train_size <= 0:
        raise ValueError("train_size must be positive.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique before split construction.")

    ordered_sample_ids = tuple(sorted(sample_ids))
    if len(ordered_sample_ids) <= query_size:
        raise ValueError("sample_ids must be larger than query_size.")

    permuted = list(ordered_sample_ids)
    random.Random(seed).shuffle(permuted)
    permuted_sample_ids = tuple(permuted)

    query_ids = permuted_sample_ids[:query_size]
    retrieval_ids = permuted_sample_ids[query_size:]
    if len(retrieval_ids) < train_size:
        raise ValueError("retrieval_ids must be at least as large as train_size.")

    train_ids = retrieval_ids[:train_size]
    validate_split_relations(query_ids, retrieval_ids, train_ids)

    return SplitResult(
        seed=seed,
        ordered_sample_ids=ordered_sample_ids,
        permuted_sample_ids=permuted_sample_ids,
        query_ids=query_ids,
        retrieval_ids=retrieval_ids,
        train_ids=train_ids,
    )


def validate_split_relations(
    query_ids: Sequence[str],
    retrieval_ids: Sequence[str],
    train_ids: Sequence[str],
) -> None:
    query_set = set(query_ids)
    retrieval_set = set(retrieval_ids)
    train_set = set(train_ids)

    if len(query_ids) != len(query_set):
        raise ValueError("query_ids must be unique.")
    if len(retrieval_ids) != len(retrieval_set):
        raise ValueError("retrieval_ids must be unique.")
    if len(train_ids) != len(train_set):
        raise ValueError("train_ids must be unique.")
    if query_set & retrieval_set:
        raise ValueError("query_ids and retrieval_ids must be disjoint.")
    if not train_set.issubset(retrieval_set):
        raise ValueError("train_ids must be a subset of retrieval_ids.")


def build_split_summary(dataset_name: str, split_result: SplitResult) -> Dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "seed": split_result.seed,
        "counts": {
            "ordered_total": len(split_result.ordered_sample_ids),
            "query": len(split_result.query_ids),
            "retrieval": len(split_result.retrieval_ids),
            "train": len(split_result.train_ids),
        },
        "relations": {
            "query_retrieval_disjoint": True,
            "train_subset_of_retrieval": True,
        },
    }
