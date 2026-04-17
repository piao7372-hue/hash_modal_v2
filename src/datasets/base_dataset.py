from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


SPLIT_NAMES = ("train", "query", "retrieval")


@dataclass(frozen=True)
class ProtocolCounts:
    raw_total: int
    filtered_total: int
    query: int
    retrieval: int
    train: int


@dataclass(frozen=True)
class DatasetProtocol:
    dataset_name: str
    sample_granularity: str
    text_source_rule: str
    filter_rule: str
    label_dim_raw: int
    label_dim_output: int
    split_seed: int
    counts: ProtocolCounts
    train_subset_of_retrieval: bool
    query_retrieval_disjoint: bool
    concept_subset_size: Optional[int] = None
    concept_subset_rule: Optional[str] = None
    quantity_closure_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "sample_granularity": self.sample_granularity,
            "text_source_rule": self.text_source_rule,
            "filter_rule": self.filter_rule,
            "label_dim_raw": self.label_dim_raw,
            "label_dim_output": self.label_dim_output,
            "split_seed": self.split_seed,
            "counts": {
                "raw_total": self.counts.raw_total,
                "filtered_total": self.counts.filtered_total,
                "query": self.counts.query,
                "retrieval": self.counts.retrieval,
                "train": self.counts.train,
            },
            "train_subset_of_retrieval": self.train_subset_of_retrieval,
            "query_retrieval_disjoint": self.query_retrieval_disjoint,
            "concept_subset_size": self.concept_subset_size,
            "concept_subset_rule": self.concept_subset_rule,
            "quantity_closure_rule": self.quantity_closure_rule,
        }


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    dataset_name: str
    image_path: str
    text_source: str
    label_vector: Tuple[int, ...]
    raw_index: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")
        if not isinstance(self.dataset_name, str) or not self.dataset_name:
            raise ValueError("dataset_name must be a non-empty string.")
        if not isinstance(self.image_path, str) or not self.image_path:
            raise ValueError("image_path must be a non-empty string.")
        if not isinstance(self.text_source, str):
            raise ValueError("text_source must be a string.")
        if not isinstance(self.raw_index, int) or self.raw_index < 0:
            raise ValueError("raw_index must be a non-negative integer.")
        if not self.label_vector:
            raise ValueError("label_vector must not be empty.")
        if any(value not in (0, 1) for value in self.label_vector):
            raise ValueError("label_vector must contain only 0/1 integers.")

    def to_manifest_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "sample_id": self.sample_id,
            "dataset_name": self.dataset_name,
            "image_path": self.image_path,
            "text_source": self.text_source,
            "label_vector": list(self.label_vector),
            "raw_index": self.raw_index,
            "meta": self.meta,
        }


class BaseDatasetAdapter(ABC):
    """Base contract for Stage 1 preprocessing adapters."""

    required_source_keys: Tuple[str, ...] = ()

    def __init__(
        self,
        project_root: Path,
        preprocess_config: Mapping[str, Any],
        dataset_config: Mapping[str, Any],
    ) -> None:
        self.project_root = project_root
        self.preprocess_config = preprocess_config
        self.dataset_config = dataset_config
        self.dataset_name = str(self.dataset_config["dataset_name"])
        self.raw_root = self._resolve_project_path(str(self.dataset_config["raw_root"]))
        self._prepared = False

    @abstractmethod
    def protocol(self) -> DatasetProtocol:
        """Return the frozen Stage 1 protocol for the dataset."""

    @abstractmethod
    def iter_raw_samples(self) -> Iterable[SampleRecord]:
        """Yield raw samples in the dataset's deterministic raw order."""

    @abstractmethod
    def filter_raw_sample(self, sample: SampleRecord) -> Optional[SampleRecord]:
        """Convert a raw sample into a filtered sample or return None if dropped."""

    def prepare(self) -> None:
        """Load dataset-specific raw assets required to iterate samples."""
        self._prepared = True

    def raw_label_names(self) -> Tuple[str, ...]:
        return tuple()

    def filtered_label_names(self) -> Tuple[str, ...]:
        return self.raw_label_names()

    def protocol_metadata(self) -> Dict[str, Any]:
        return {}

    def dataset_metadata(self) -> Dict[str, Any]:
        return {}

    def describe(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "raw_root": str(self.raw_root),
            "required_source_keys": list(self.required_source_keys),
            "protocol": self.protocol().to_dict(),
            "raw_label_names": list(self.raw_label_names()),
            "filtered_label_names": list(self.filtered_label_names()),
            "protocol_metadata": self.protocol_metadata(),
            "dataset_metadata": self.dataset_metadata(),
        }

    def validate_dataset_config(self) -> None:
        if self.dataset_name != self.protocol().dataset_name:
            raise ValueError(
                "dataset_name mismatch: config={0}, protocol={1}".format(
                    self.dataset_name,
                    self.protocol().dataset_name,
                )
            )

        sources = self.dataset_config["sources"]
        missing_source_keys = [key for key in self.required_source_keys if key not in sources]
        if missing_source_keys:
            raise KeyError(
                "Missing required source keys for {0}: {1}".format(
                    self.dataset_name,
                    missing_source_keys,
                )
            )

        protocol_block = self.dataset_config["protocol"]
        if tuple(protocol_block["split_names"]) != SPLIT_NAMES:
            raise ValueError(
                "{0} split_names must be exactly {1}, got {2}".format(
                    self.dataset_name,
                    SPLIT_NAMES,
                    protocol_block["split_names"],
                )
            )

        frozen_protocol = self.protocol()
        expected_counts = frozen_protocol.counts
        actual_counts = protocol_block["counts"]
        if actual_counts["raw_total"] != expected_counts.raw_total:
            raise ValueError(
                "{0} raw_total mismatch: {1} != {2}".format(
                    self.dataset_name,
                    actual_counts["raw_total"],
                    expected_counts.raw_total,
                )
            )
        if actual_counts["filtered_total"] != expected_counts.filtered_total:
            raise ValueError(
                "{0} filtered_total mismatch: {1} != {2}".format(
                    self.dataset_name,
                    actual_counts["filtered_total"],
                    expected_counts.filtered_total,
                )
            )
        if actual_counts["query"] != expected_counts.query:
            raise ValueError(
                "{0} query mismatch: {1} != {2}".format(
                    self.dataset_name,
                    actual_counts["query"],
                    expected_counts.query,
                )
            )
        if actual_counts["retrieval"] != expected_counts.retrieval:
            raise ValueError(
                "{0} retrieval mismatch: {1} != {2}".format(
                    self.dataset_name,
                    actual_counts["retrieval"],
                    expected_counts.retrieval,
                )
            )
        if actual_counts["train"] != expected_counts.train:
            raise ValueError(
                "{0} train mismatch: {1} != {2}".format(
                    self.dataset_name,
                    actual_counts["train"],
                    expected_counts.train,
                )
            )
        if int(protocol_block["split_seed"]) != frozen_protocol.split_seed:
            raise ValueError(
                "{0} split_seed mismatch: {1} != {2}".format(
                    self.dataset_name,
                    protocol_block["split_seed"],
                    frozen_protocol.split_seed,
                )
            )

    def resolve_required_sources(self) -> Dict[str, Path]:
        sources = self.dataset_config["sources"]
        resolved: Dict[str, Path] = {}
        for key in self.required_source_keys:
            resolved[key] = self._resolve_relative_to_raw_root(str(sources[key]))
        return resolved

    def validate_required_sources_exist(self) -> Dict[str, Path]:
        resolved = self.resolve_required_sources()
        for key, path in resolved.items():
            if not path.exists():
                raise FileNotFoundError(
                    "Required raw source for {0} does not exist: {1} -> {2}".format(
                        self.dataset_name,
                        key,
                        path,
                    )
                )
        return resolved

    def output_dataset_root(self) -> Path:
        processed_root = self._resolve_project_path(
            str(self.preprocess_config["paths"]["processed_root"])
        )
        return processed_root / self.dataset_name

    def build_index_sample_id(self, index_one_based: int, width: int = 6) -> str:
        return "{0}_{1:0{2}d}".format(self.dataset_name, index_one_based, width)

    def ensure_prepared(self) -> None:
        if not self._prepared:
            self.prepare()

    def read_text_lines(self, path: Path) -> Tuple[str, ...]:
        if not path.exists():
            raise FileNotFoundError("Expected text file does not exist: {0}".format(path))
        return tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    def read_integer_lines(self, path: Path) -> Tuple[int, ...]:
        values: list[int] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            values.append(int(line))
        return tuple(values)

    def join_text_tokens(self, tokens: Sequence[str]) -> str:
        return " ".join(token for token in tokens if token)

    def _resolve_relative_to_raw_root(self, value: str) -> Path:
        return (self.raw_root / value).resolve()

    def _resolve_project_path(self, value: str) -> Path:
        return (self.project_root / value).resolve()
