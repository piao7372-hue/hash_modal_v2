from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

from .base_dataset import DatasetProtocol, SampleRecord


@dataclass(frozen=True)
class DatasetOutputPaths:
    dataset_root: Path
    manifest_dir: Path
    manifest_raw_path: Path
    manifest_filtered_path: Path
    manifest_meta_path: Path
    splits_dir: Path
    query_ids_path: Path
    retrieval_ids_path: Path
    train_ids_path: Path
    split_summary_path: Path
    reports_dir: Path
    preprocess_summary_path: Path
    validator_summary_path: Path
    config_snapshot_path: Path


def build_output_paths(processed_root: Path, dataset_name: str) -> DatasetOutputPaths:
    dataset_root = (processed_root / dataset_name).resolve()
    manifest_dir = dataset_root / "manifest"
    splits_dir = dataset_root / "splits"
    reports_dir = dataset_root / "reports"
    return DatasetOutputPaths(
        dataset_root=dataset_root,
        manifest_dir=manifest_dir,
        manifest_raw_path=manifest_dir / "manifest_raw.jsonl",
        manifest_filtered_path=manifest_dir / "manifest_filtered.jsonl",
        manifest_meta_path=manifest_dir / "manifest_meta.json",
        splits_dir=splits_dir,
        query_ids_path=splits_dir / "query_ids.txt",
        retrieval_ids_path=splits_dir / "retrieval_ids.txt",
        train_ids_path=splits_dir / "train_ids.txt",
        split_summary_path=splits_dir / "split_summary.json",
        reports_dir=reports_dir,
        preprocess_summary_path=reports_dir / "preprocess_summary.json",
        validator_summary_path=reports_dir / "validator_summary.json",
        config_snapshot_path=reports_dir / "config_snapshot.json",
    )


def prepare_output_dirs(paths: DatasetOutputPaths) -> None:
    for directory in (
        paths.dataset_root,
        paths.manifest_dir,
        paths.splits_dir,
        paths.reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def stable_json_dumps(payload: Any, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_lines(path: Path, values: Sequence[str]) -> None:
    if any(not value for value in values):
        raise ValueError("Text line outputs must not contain empty ids.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def read_text_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError("Missing expected text file: {0}".format(path))
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not line for line in lines):
        raise ValueError("Found an empty line in {0}".format(path))
    return lines


def write_manifest_jsonl(path: Path, records: Iterable[SampleRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(stable_json_dumps(record.to_manifest_dict()) + "\n")
            line_count += 1
    return line_count


def read_manifest_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError("Missing manifest artifact: {0}".format(path))
    records: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError("Blank JSONL line detected in {0} at line {1}".format(path, line_number))
            records.append(json.loads(line))
    return records


def iter_manifest_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError("Missing manifest artifact: {0}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError("Blank JSONL line detected in {0} at line {1}".format(path, line_number))
            yield json.loads(line)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_meta(
    dataset_name: str,
    protocol: DatasetProtocol,
    raw_record_count: int,
    filtered_record_count: int,
    raw_label_names: Sequence[str],
    filtered_label_names: Sequence[str],
    protocol_metadata: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "manifest_fields": [
            "sample_id",
            "dataset_name",
            "image_path",
            "text_source",
            "label_vector",
            "raw_index",
            "meta",
        ],
        "protocol": protocol.to_dict(),
        "counts": {
            "raw_record_count": raw_record_count,
            "filtered_record_count": filtered_record_count,
        },
        "raw_label_names": list(raw_label_names),
        "filtered_label_names": list(filtered_label_names),
        "protocol_metadata": protocol_metadata,
        "dataset_metadata": dataset_metadata,
    }


def build_preprocess_summary(
    dataset_name: str,
    protocol: DatasetProtocol,
    raw_record_count: int,
    filtered_record_count: int,
    split_counts: Dict[str, int],
    output_paths: DatasetOutputPaths,
    raw_label_names: Sequence[str],
    filtered_label_names: Sequence[str],
    protocol_metadata: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "stage": "stage_1_preprocess",
        "raw_record_count": raw_record_count,
        "filtered_record_count": filtered_record_count,
        "split_counts": split_counts,
        "expected_counts": protocol.to_dict()["counts"],
        "raw_label_names": list(raw_label_names),
        "filtered_label_names": list(filtered_label_names),
        "protocol_metadata": protocol_metadata,
        "dataset_metadata": dataset_metadata,
        "artifacts": {
            "manifest_raw": str(output_paths.manifest_raw_path),
            "manifest_filtered": str(output_paths.manifest_filtered_path),
            "manifest_meta": str(output_paths.manifest_meta_path),
            "query_ids": str(output_paths.query_ids_path),
            "retrieval_ids": str(output_paths.retrieval_ids_path),
            "train_ids": str(output_paths.train_ids_path),
            "split_summary": str(output_paths.split_summary_path),
            "preprocess_summary": str(output_paths.preprocess_summary_path),
            "validator_summary": str(output_paths.validator_summary_path),
            "config_snapshot": str(output_paths.config_snapshot_path),
        },
        "artifact_sha256": {
            "manifest_raw": sha256_of_file(output_paths.manifest_raw_path),
            "manifest_filtered": sha256_of_file(output_paths.manifest_filtered_path),
            "query_ids": sha256_of_file(output_paths.query_ids_path),
            "retrieval_ids": sha256_of_file(output_paths.retrieval_ids_path),
            "train_ids": sha256_of_file(output_paths.train_ids_path),
        },
    }
