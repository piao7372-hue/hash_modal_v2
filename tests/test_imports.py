from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DATASET_ADAPTERS, SUPPORTED_DATASETS
from src.datasets.split_builder import build_query_retrieval_train_splits
from src.engine import (
    build_model_output_validator_summary,
    build_training_output_validator_summary,
    load_train_config,
    run_formal_training,
)
from src.losses import LossConfig, build_induced_targets, build_relation_predictions, compute_total_loss
from src.models import (
    ChebyKAN,
    ChebyKANConfig,
    FormalHashModel,
    GraphRefine,
    GraphRefineConfig,
    HashBinarize,
    PredictorConfig,
    SemanticTree,
    SemanticTreeConfig,
)


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


def test_stage4_model_interfaces_are_importable() -> None:
    assert ChebyKAN is not None
    assert ChebyKANConfig is not None
    assert SemanticTree is not None
    assert SemanticTreeConfig is not None
    assert GraphRefine is not None
    assert GraphRefineConfig is not None
    assert HashBinarize is not None
    assert FormalHashModel is not None
    assert PredictorConfig is not None


def test_stage4_loss_interfaces_are_importable() -> None:
    assert callable(build_relation_predictions)
    assert callable(build_induced_targets)
    assert callable(compute_total_loss)
    assert LossConfig is not None


def test_stage4_engine_interfaces_are_importable() -> None:
    assert callable(load_train_config)
    assert callable(run_formal_training)
    assert callable(build_model_output_validator_summary)
    assert callable(build_training_output_validator_summary)


def main() -> int:
    test_supported_datasets_are_registered()
    test_split_builder_enforces_query_retrieval_train_contract()
    test_stage4_model_interfaces_are_importable()
    test_stage4_loss_interfaces_are_importable()
    test_stage4_engine_interfaces_are_importable()
    print("test_imports.py: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
