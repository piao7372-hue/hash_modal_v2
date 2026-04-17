from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .base_dataset import BaseDatasetAdapter, DatasetProtocol, ProtocolCounts, SampleRecord


class MIRFlickr25KAdapter(BaseDatasetAdapter):
    required_source_keys = (
        "image_root",
        "raw_tags_root",
        "annotations_root",
        "annotation_readme_file",
    )

    _LABEL_NAMES: Tuple[str, ...] = (
        "animals",
        "baby",
        "bird",
        "car",
        "clouds",
        "dog",
        "female",
        "flower",
        "food",
        "indoor",
        "lake",
        "male",
        "night",
        "people",
        "plant_life",
        "portrait",
        "river",
        "sea",
        "sky",
        "structures",
        "sunset",
        "transport",
        "tree",
        "water",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._label_index_sets: Dict[str, set[int]] = {}
        self._annotation_positive_counts: Dict[str, int] = {}
        self._raw_tag_token_counts: Dict[str, int] = {}
        self._selected_sample_ids: set[str] = set()
        self._sources: Dict[str, Path] = {}
        self._empty_text_rows = 0

    def protocol(self) -> DatasetProtocol:
        return DatasetProtocol(
            dataset_name="mirflickr25k",
            sample_granularity="one image + corresponding tags text + 24-dimensional label vector",
            text_source_rule="Read the tags file for each sample, preserve original token order, and join tokens with a single space.",
            filter_rule=(
                "Remove empty-text rows, rank by raw_tag_token_count desc, "
                "annotation_positive_count desc, sample_id asc, and keep the top 20,015."
            ),
            label_dim_raw=24,
            label_dim_output=24,
            split_seed=0,
            counts=ProtocolCounts(
                raw_total=25000,
                filtered_total=20015,
                query=2000,
                retrieval=18015,
                train=5000,
            ),
            train_subset_of_retrieval=True,
            query_retrieval_disjoint=True,
        )

    def prepare(self) -> None:
        self.validate_dataset_config()
        self._sources = self.validate_required_sources_exist()

        image_root = self._sources["image_root"]
        tags_root = self._sources["raw_tags_root"]
        annotations_root = self._sources["annotations_root"]
        readme_file = self._sources["annotation_readme_file"]

        if not image_root.is_dir():
            raise NotADirectoryError("MIRFlickr image_root must be a directory: {0}".format(image_root))
        if not tags_root.is_dir():
            raise NotADirectoryError("MIRFlickr raw_tags_root must be a directory: {0}".format(tags_root))
        if not annotations_root.is_dir():
            raise NotADirectoryError(
                "MIRFlickr annotations_root must be a directory: {0}".format(annotations_root)
            )
        if not readme_file.is_file():
            raise FileNotFoundError(
                "MIRFlickr annotation_readme_file must be a file: {0}".format(readme_file)
            )

        image_count = len(list(image_root.glob("im*.jpg")))
        if image_count != self.protocol().counts.raw_total:
            raise ValueError(
                "MIRFlickr image count mismatch: {0} != {1}".format(
                    image_count,
                    self.protocol().counts.raw_total,
                )
            )

        tags_count = len(list(tags_root.glob("tags*.txt")))
        if tags_count != self.protocol().counts.raw_total:
            raise ValueError(
                "MIRFlickr tags count mismatch: {0} != {1}".format(
                    tags_count,
                    self.protocol().counts.raw_total,
                )
            )

        label_index_sets: Dict[str, set[int]] = {}
        for label_name in self._LABEL_NAMES:
            annotation_path = annotations_root / "{0}.txt".format(label_name)
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    "Missing MIRFlickr annotation file for label {0}: {1}".format(
                        label_name,
                        annotation_path,
                    )
                )
            label_indices = set(self.read_integer_lines(annotation_path))
            if not label_indices:
                raise ValueError("MIRFlickr annotation file is empty: {0}".format(annotation_path))
            if min(label_indices) < 1 or max(label_indices) > self.protocol().counts.raw_total:
                raise ValueError(
                    "MIRFlickr annotation indices out of range for label {0}".format(label_name)
                )
            label_index_sets[label_name] = label_indices

        ranking_rows: list[tuple[int, int, str]] = []
        annotation_positive_counts: Dict[str, int] = {}
        raw_tag_token_counts: Dict[str, int] = {}
        empty_text_rows = 0
        for image_number in range(1, self.protocol().counts.raw_total + 1):
            sample_id = self.build_index_sample_id(image_number, width=5)
            tags_path = tags_root / "tags{0}.txt".format(image_number)
            tag_tokens = self.read_text_lines(tags_path)
            raw_tag_token_count = len(tag_tokens)
            annotation_positive_count = sum(
                int(image_number in label_index_sets[label_name]) for label_name in self._LABEL_NAMES
            )
            raw_tag_token_counts[sample_id] = raw_tag_token_count
            annotation_positive_counts[sample_id] = annotation_positive_count
            if raw_tag_token_count == 0:
                empty_text_rows += 1
                continue
            ranking_rows.append(
                (-raw_tag_token_count, -annotation_positive_count, sample_id)
            )

        ranking_rows.sort()
        target_filtered_count = self.protocol().counts.filtered_total
        if len(ranking_rows) < target_filtered_count:
            raise ValueError(
                "MIRFlickr pragmatic selection candidates are insufficient: {0} < {1}".format(
                    len(ranking_rows),
                    target_filtered_count,
                )
            )

        self._label_index_sets = label_index_sets
        self._annotation_positive_counts = annotation_positive_counts
        self._raw_tag_token_counts = raw_tag_token_counts
        self._selected_sample_ids = {
            sample_id for _, _, sample_id in ranking_rows[:target_filtered_count]
        }
        self._empty_text_rows = empty_text_rows
        self._prepared = True

    def raw_label_names(self) -> Tuple[str, ...]:
        return self._LABEL_NAMES

    def protocol_metadata(self) -> Dict[str, object]:
        return {
            "protocol_name": "pragmatic_high_signal_v1",
            "protocol_source": "project_defined_not_ra_literal",
            "target_filtered_count": self.protocol().counts.filtered_total,
            "selection_policy": "nonempty_text_then_rank_by_tag_density_and_label_density",
            "seed": self.protocol().split_seed,
            "empty_text_rows_removed": self._empty_text_rows,
            "final_filtered_count": len(self._selected_sample_ids),
            "query_count": self.protocol().counts.query,
            "retrieval_count": self.protocol().counts.retrieval,
            "train_count": self.protocol().counts.train,
        }

    def dataset_metadata(self) -> Dict[str, object]:
        return {
            "label_names": list(self._LABEL_NAMES),
            "annotation_readme_file": str(self._sources.get("annotation_readme_file", "")),
            "empty_text_rows": self._empty_text_rows,
            "nonempty_text_candidate_count": self.protocol().counts.raw_total - self._empty_text_rows,
        }

    def iter_raw_samples(self) -> Iterable[SampleRecord]:
        self.ensure_prepared()
        image_root = self._sources["image_root"]
        tags_root = self._sources["raw_tags_root"]

        for raw_index in range(self.protocol().counts.raw_total):
            image_number = raw_index + 1
            image_path = image_root / "im{0}.jpg".format(image_number)
            if not image_path.is_file():
                raise FileNotFoundError("Missing MIRFlickr image file: {0}".format(image_path))

            tags_path = tags_root / "tags{0}.txt".format(image_number)
            tag_tokens = self.read_text_lines(tags_path)
            text_source = self.join_text_tokens(tag_tokens)

            label_vector = tuple(
                int(image_number in self._label_index_sets[label_name])
                for label_name in self._LABEL_NAMES
            )
            sample_id = self.build_index_sample_id(image_number, width=5)
            record = SampleRecord(
                sample_id=sample_id,
                dataset_name=self.dataset_name,
                image_path=str(image_path),
                text_source=text_source,
                label_vector=label_vector,
                raw_index=raw_index,
                meta={
                    "image_filename": image_path.name,
                    "text_source_path": str(tags_path),
                    "text_source_kind": "tags_file",
                    "raw_tag_token_count": self._raw_tag_token_counts[sample_id],
                    "annotation_positive_count": self._annotation_positive_counts[sample_id],
                    "text_is_empty": text_source == "",
                },
            )
            record.validate()
            yield record

    def filter_raw_sample(self, sample: SampleRecord) -> Optional[SampleRecord]:
        if len(sample.label_vector) != self.protocol().label_dim_raw:
            raise ValueError(
                "MIRFlickr raw label dimension mismatch for {0}".format(sample.sample_id)
            )
        if sample.text_source == "":
            return None
        if sample.sample_id not in self._selected_sample_ids:
            return None
        return sample
