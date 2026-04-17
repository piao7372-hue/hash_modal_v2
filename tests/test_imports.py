from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DATASET_ADAPTERS, SUPPORTED_DATASETS
from src.datasets.split_builder import build_query_retrieval_train_splits
from src.features import load_feature_extraction_config, validate_feature_cache


def test_supported_datasets_are_registered() -> None:
    assert SUPPORTED_DATASETS == ("mirflickr25k", "nuswide", "mscoco")
    assert set(DATASET_ADAPTERS.keys()) == set(SUPPORTED_DATASETS)


def test_split_builder_enforces_query_retrieval_train_contract() -> None:
    sample_ids = [f"sample-{index:04d}" for index in range(16)]
    split_result = build_query_retrieval_train_splits(
        sample_ids=sample_ids,
        query_size=2,
        train_size=5,
        seed=0,
    )
    assert len(split_result.query_ids) == 2
    assert len(split_result.retrieval_ids) == 14
    assert len(split_result.train_ids) == 5
    assert set(split_result.query_ids).isdisjoint(split_result.retrieval_ids)
    assert set(split_result.train_ids).issubset(split_result.retrieval_ids)


def test_stage2_interfaces_are_importable() -> None:
    assert callable(load_feature_extraction_config)
    assert callable(validate_feature_cache)
