from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .base_dataset import BaseDatasetAdapter, DatasetProtocol, ProtocolCounts, SampleRecord


class NUSWIDEAdapter(BaseDatasetAdapter):
    required_source_keys = (
        "image_root",
        "all_tags_file",
        "all_labels_root",
        "concepts_file",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sources: Dict[str, Path] = {}
        self._raw_label_names: Tuple[str, ...] = tuple()
        self._raw_label_columns: List[bytearray] = []
        self._top10_names: Tuple[str, ...] = tuple()
        self._top10_indices: Tuple[int, ...] = tuple()
        self._top10_positive_counts: Dict[str, int] = {}
        self._sorted_image_paths: List[Path] = []

    def protocol(self) -> DatasetProtocol:
        return DatasetProtocol(
            dataset_name="nuswide",
            sample_granularity="one image + corresponding tags text + 81-dimensional raw label vector",
            text_source_rule="Read the tags from All_Tags.txt in original order and join them with a single space.",
            filter_rule="Count positives for 81 categories, choose top-10 by descending positive count with lexical tie-break, truncate to 10 dimensions, then keep only samples with sum(label_vector) > 0; filtered total must be exactly 186,577.",
            label_dim_raw=81,
            label_dim_output=10,
            split_seed=0,
            counts=ProtocolCounts(
                raw_total=269648,
                filtered_total=186577,
                query=2000,
                retrieval=184577,
                train=5000,
            ),
            train_subset_of_retrieval=True,
            query_retrieval_disjoint=True,
            concept_subset_size=10,
            concept_subset_rule="Sort categories by positive-sample count descending; if counts tie, break ties by lexical category name ascending; keep the first 10.",
        )

    def prepare(self) -> None:
        self.validate_dataset_config()
        self._sources = self.validate_required_sources_exist()

        image_root = self._sources["image_root"]
        all_tags_file = self._sources["all_tags_file"]
        all_labels_root = self._sources["all_labels_root"]
        concepts_file = self._sources["concepts_file"]

        if not image_root.is_dir():
            raise NotADirectoryError("NUS-WIDE image_root must be a directory: {0}".format(image_root))
        if not all_labels_root.is_dir():
            raise NotADirectoryError(
                "NUS-WIDE all_labels_root must be a directory: {0}".format(all_labels_root)
            )
        if not all_tags_file.is_file():
            raise FileNotFoundError("NUS-WIDE all_tags_file must be a file: {0}".format(all_tags_file))
        if not concepts_file.is_file():
            raise FileNotFoundError("NUS-WIDE concepts_file must be a file: {0}".format(concepts_file))

        raw_label_names = tuple(
            line.strip()
            for line in concepts_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if len(raw_label_names) != self.protocol().label_dim_raw:
            raise ValueError(
                "NUS-WIDE Concepts81.txt length mismatch: {0} != {1}".format(
                    len(raw_label_names),
                    self.protocol().label_dim_raw,
                )
            )

        raw_label_columns: List[bytearray] = []
        positive_counts: Dict[str, int] = {}
        expected_total = self.protocol().counts.raw_total
        for label_name in raw_label_names:
            label_path = all_labels_root / "Labels_{0}.txt".format(label_name)
            if not label_path.is_file():
                raise FileNotFoundError(
                    "Missing NUS-WIDE AllLabels file for concept {0}: {1}".format(
                        label_name,
                        label_path,
                    )
                )
            values = bytearray()
            with label_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        raise ValueError(
                            "Blank label line in {0} at line {1}".format(label_path, line_number)
                        )
                    if line not in ("0", "1"):
                        raise ValueError(
                            "Unexpected NUS-WIDE label value {0!r} in {1} at line {2}".format(
                                line,
                                label_path,
                                line_number,
                            )
                        )
                    values.append(int(line))
            if len(values) != expected_total:
                raise ValueError(
                    "NUS-WIDE label length mismatch for {0}: {1} != {2}".format(
                        label_name,
                        len(values),
                        expected_total,
                    )
                )
            raw_label_columns.append(values)
            positive_counts[label_name] = int(sum(values))

        sorted_image_paths = sorted(image_root.glob("*.jpg"), key=lambda path: path.name)
        if len(sorted_image_paths) != expected_total:
            raise ValueError(
                "NUS-WIDE image count mismatch: {0} != {1}".format(
                    len(sorted_image_paths),
                    expected_total,
                )
            )

        top10_ranked = sorted(
            ((positive_counts[name], name, index) for index, name in enumerate(raw_label_names)),
            key=lambda item: (-item[0], item[1]),
        )[:10]
        top10_names = tuple(item[1] for item in top10_ranked)
        top10_indices = tuple(item[2] for item in top10_ranked)

        self._raw_label_names = raw_label_names
        self._raw_label_columns = raw_label_columns
        self._top10_names = top10_names
        self._top10_indices = top10_indices
        self._top10_positive_counts = {name: positive_counts[name] for name in top10_names}
        self._sorted_image_paths = sorted_image_paths
        self._prepared = True

    def raw_label_names(self) -> Tuple[str, ...]:
        return self._raw_label_names

    def filtered_label_names(self) -> Tuple[str, ...]:
        return self._top10_names

    def dataset_metadata(self) -> Dict[str, object]:
        return {
            "raw_label_names": list(self._raw_label_names),
            "top10_concepts": list(self._top10_names),
            "top10_positive_counts": self._top10_positive_counts,
            "top10_rule": self.protocol().concept_subset_rule,
            "image_order_rule": "sort image filenames ascending to align with official line-aligned raw assets",
        }

    def iter_raw_samples(self) -> Iterable[SampleRecord]:
        self.ensure_prepared()
        expected_total = self.protocol().counts.raw_total
        all_tags_file = self._sources["all_tags_file"]

        raw_count = 0
        with all_tags_file.open("r", encoding="utf-8") as handle:
            for raw_index, raw_line in enumerate(handle):
                if raw_index >= len(self._sorted_image_paths):
                    raise ValueError("NUS-WIDE All_Tags.txt has more rows than images.")
                image_path = self._sorted_image_paths[raw_index]
                tag_tokens = raw_line.strip().split()
                text_source = self.join_text_tokens(tag_tokens)
                label_vector = tuple(
                    int(self._raw_label_columns[label_index][raw_index])
                    for label_index in range(len(self._raw_label_names))
                )
                record = SampleRecord(
                    sample_id=self.build_index_sample_id(raw_index + 1, width=6),
                    dataset_name=self.dataset_name,
                    image_path=str(image_path),
                    text_source=text_source,
                    label_vector=label_vector,
                    raw_index=raw_index,
                    meta={
                        "image_filename": image_path.name,
                        "text_source_path": str(all_tags_file),
                        "text_source_line_index": raw_index,
                        "text_source_kind": "all_tags_line",
                    },
                )
                record.validate()
                raw_count += 1
                yield record

        if raw_count != expected_total:
            raise ValueError(
                "NUS-WIDE All_Tags.txt row count mismatch: {0} != {1}".format(
                    raw_count,
                    expected_total,
                )
            )

    def filter_raw_sample(self, sample: SampleRecord) -> Optional[SampleRecord]:
        if len(sample.label_vector) != self.protocol().label_dim_raw:
            raise ValueError(
                "NUS-WIDE raw label dimension mismatch for {0}".format(sample.sample_id)
            )
        filtered_label_vector = tuple(sample.label_vector[index] for index in self._top10_indices)
        if sum(filtered_label_vector) == 0:
            return None
        return SampleRecord(
            sample_id=sample.sample_id,
            dataset_name=sample.dataset_name,
            image_path=sample.image_path,
            text_source=sample.text_source,
            label_vector=filtered_label_vector,
            raw_index=sample.raw_index,
            meta={
                **sample.meta,
                "top10_concepts": list(self._top10_names),
            },
        )
