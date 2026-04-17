from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .base_dataset import BaseDatasetAdapter, DatasetProtocol, ProtocolCounts, SampleRecord


class MSCOCOAdapter(BaseDatasetAdapter):
    required_source_keys = (
        "images_train2014",
        "images_val2014",
        "captions_train2014",
        "captions_val2014",
        "instances_train2014",
        "instances_val2014",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sources: Dict[str, Path] = {}
        self._category_ids: Tuple[int, ...] = tuple()
        self._category_names: Tuple[str, ...] = tuple()
        self._rows: List[Dict[str, object]] = []

    def protocol(self) -> DatasetProtocol:
        return DatasetProtocol(
            dataset_name="mscoco",
            sample_granularity="image-level pair: one image + all captions merged into one text + 80-dimensional multi-hot labels",
            text_source_rule="Sort captions for the same image by annotation id ascending and join them with '. ' into a single string.",
            filter_rule="Do not filter out valid image-level pairs after aggregation; total sample count must be exactly 123,287.",
            label_dim_raw=80,
            label_dim_output=80,
            split_seed=0,
            counts=ProtocolCounts(
                raw_total=123287,
                filtered_total=123287,
                query=2000,
                retrieval=121287,
                train=5000,
            ),
            train_subset_of_retrieval=True,
            query_retrieval_disjoint=True,
            quantity_closure_rule="Use train as a subset of retrieval so that 2,000 query + 121,287 retrieval closes to 123,287 total samples.",
        )

    def prepare(self) -> None:
        self.validate_dataset_config()
        self._sources = self.validate_required_sources_exist()

        image_root_train = self._sources["images_train2014"]
        image_root_val = self._sources["images_val2014"]
        captions_train = self._sources["captions_train2014"]
        captions_val = self._sources["captions_val2014"]
        instances_train = self._sources["instances_train2014"]
        instances_val = self._sources["instances_val2014"]

        for directory in (image_root_train, image_root_val):
            if not directory.is_dir():
                raise NotADirectoryError("MSCOCO image source must be a directory: {0}".format(directory))
        for file_path in (captions_train, captions_val, instances_train, instances_val):
            if not file_path.is_file():
                raise FileNotFoundError("MSCOCO annotation source must be a file: {0}".format(file_path))

        categories_train = self._load_categories(instances_train)
        categories_val = self._load_categories(instances_val)
        if categories_train != categories_val:
            raise ValueError("MSCOCO category definitions differ between train and val instance files.")
        self._category_ids = tuple(category_id for category_id, _ in categories_train)
        self._category_names = tuple(category_name for _, category_name in categories_train)
        if len(self._category_ids) != self.protocol().label_dim_raw:
            raise ValueError(
                "MSCOCO category count mismatch: {0} != {1}".format(
                    len(self._category_ids),
                    self.protocol().label_dim_raw,
                )
            )

        rows: List[Dict[str, object]] = []
        rows.extend(
            self._load_split_rows(
                split_name="train2014",
                image_root=image_root_train,
                captions_path=captions_train,
                instances_path=instances_train,
            )
        )
        rows.extend(
            self._load_split_rows(
                split_name="val2014",
                image_root=image_root_val,
                captions_path=captions_val,
                instances_path=instances_val,
            )
        )
        rows.sort(key=lambda row: row["sample_id"])
        if len(rows) != self.protocol().counts.raw_total:
            raise ValueError(
                "MSCOCO row count mismatch: {0} != {1}".format(
                    len(rows),
                    self.protocol().counts.raw_total,
                )
            )

        self._rows = rows
        self._prepared = True

    def raw_label_names(self) -> Tuple[str, ...]:
        return self._category_names

    def dataset_metadata(self) -> Dict[str, object]:
        return {
            "category_ids": list(self._category_ids),
            "category_names": list(self._category_names),
            "text_joiner": ". ",
        }

    def iter_raw_samples(self) -> Iterable[SampleRecord]:
        self.ensure_prepared()
        category_index = {category_id: idx for idx, category_id in enumerate(self._category_ids)}
        for raw_index, row in enumerate(self._rows):
            captions = sorted(row["captions"], key=lambda item: item[0])
            text_source = ". ".join(caption for _, caption in captions)
            if not text_source:
                raise ValueError("MSCOCO text_source resolved to an empty string for {0}".format(row["sample_id"]))
            label_vector_list = [0] * len(self._category_ids)
            for category_id in row["category_ids"]:
                if category_id not in category_index:
                    raise ValueError("Unknown MSCOCO category id {0}".format(category_id))
                label_vector_list[category_index[category_id]] = 1
            record = SampleRecord(
                sample_id=str(row["sample_id"]),
                dataset_name=self.dataset_name,
                image_path=str(row["image_path"]),
                text_source=text_source,
                label_vector=tuple(label_vector_list),
                raw_index=raw_index,
                meta={
                    "image_id": row["image_id"],
                    "image_filename": row["image_filename"],
                    "source_split": row["source_split"],
                    "text_source_files": [str(row["captions_path"])],
                    "label_source_files": [str(row["instances_path"])],
                    "caption_annotation_ids": [annotation_id for annotation_id, _ in captions],
                    "category_ids": sorted(row["category_ids"]),
                },
            )
            record.validate()
            yield record

    def filter_raw_sample(self, sample: SampleRecord) -> Optional[SampleRecord]:
        if len(sample.label_vector) != self.protocol().label_dim_output:
            raise ValueError(
                "MSCOCO label dimension mismatch for {0}".format(sample.sample_id)
            )
        return sample

    def _load_categories(self, instances_path: Path) -> Tuple[Tuple[int, str], ...]:
        categories = []
        for category in self._iter_json_array(instances_path, "categories"):
            categories.append((int(category["id"]), str(category["name"])))
        categories.sort(key=lambda item: item[0])
        return tuple(categories)

    def _load_split_rows(
        self,
        split_name: str,
        image_root: Path,
        captions_path: Path,
        instances_path: Path,
    ) -> List[Dict[str, object]]:
        caption_map: DefaultDict[int, List[Tuple[int, str]]] = defaultdict(list)
        for annotation in self._iter_json_array(captions_path, "annotations"):
            image_id = int(annotation["image_id"])
            annotation_id = int(annotation["id"])
            caption = str(annotation["caption"]).strip()
            if not caption:
                raise ValueError(
                    "Empty caption found in {0} for annotation {1}".format(
                        captions_path,
                        annotation_id,
                    )
                )
            caption_map[image_id].append((annotation_id, caption))

        category_ids_by_image: DefaultDict[int, set[int]] = defaultdict(set)
        for annotation in self._iter_json_array(instances_path, "annotations"):
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            category_ids_by_image[image_id].add(category_id)

        rows: List[Dict[str, object]] = []
        seen_image_ids: set[int] = set()
        for image in self._iter_json_array(instances_path, "images"):
            image_id = int(image["id"])
            if image_id in seen_image_ids:
                raise ValueError("Duplicate MSCOCO image id detected: {0}".format(image_id))
            seen_image_ids.add(image_id)

            file_name = str(image["file_name"])
            image_path = image_root / file_name
            if not image_path.is_file():
                raise FileNotFoundError("Missing MSCOCO image file: {0}".format(image_path))
            if image_id not in caption_map:
                raise ValueError(
                    "Missing captions for MSCOCO image id {0} in {1}".format(
                        image_id,
                        captions_path,
                    )
                )

            rows.append(
                {
                    "sample_id": "mscoco_{0:012d}".format(image_id),
                    "image_id": image_id,
                    "image_filename": file_name,
                    "image_path": image_path,
                    "captions": caption_map[image_id],
                    "category_ids": set(category_ids_by_image.get(image_id, set())),
                    "source_split": split_name,
                    "captions_path": captions_path,
                    "instances_path": instances_path,
                }
            )

        return rows

    def _iter_json_array(self, path: Path, field_name: str) -> Iterator[Dict[str, object]]:
        decoder = json.JSONDecoder()
        search_token = '"{0}"'.format(field_name)
        with path.open("r", encoding="utf-8") as handle:
            buffer = ""
            array_index: Optional[int] = None

            while array_index is None:
                token_index = buffer.find(search_token)
                if token_index >= 0:
                    bracket_index = buffer.find("[", token_index)
                    if bracket_index >= 0:
                        array_index = bracket_index + 1
                        break
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise ValueError(
                        "Could not find JSON array field {0!r} in {1}".format(field_name, path)
                    )
                buffer += chunk

            while True:
                while True:
                    while array_index < len(buffer) and buffer[array_index] in " \r\n\t,":
                        array_index += 1
                    if array_index < len(buffer):
                        break
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        raise ValueError(
                            "Unexpected end of file while parsing array {0!r} in {1}".format(
                                field_name,
                                path,
                            )
                        )
                    buffer += chunk

                if buffer[array_index] == "]":
                    return

                try:
                    item, next_index = decoder.raw_decode(buffer, array_index)
                except json.JSONDecodeError:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        raise
                    buffer += chunk
                    continue

                yield item
                array_index = next_index

                if array_index > 1024 * 1024:
                    buffer = buffer[array_index:]
                    array_index = 0
