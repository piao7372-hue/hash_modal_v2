from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DATASET_ADAPTERS, SUPPORTED_DATASETS
from src.datasets.manifest_builder import (
    build_output_paths,
    iter_manifest_jsonl,
    read_json,
    read_text_lines,
    write_json,
)
from src.datasets.split_builder import (
    build_query_retrieval_train_splits,
    validate_split_relations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 1 preprocessing artifacts against the formal protocol."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "preprocess.yaml"),
        help="Path to the top-level Stage 1 preprocessing config.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name to validate.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write validator_summary.json into the dataset reports directory.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping: {0}".format(path))
    return payload


def load_configs(config_path: Path, dataset_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    preprocess_config = load_yaml(config_path)
    dataset_config_path = (
        PROJECT_ROOT / preprocess_config["datasets"][dataset_name]["config_path"]
    ).resolve()
    dataset_config = load_yaml(dataset_config_path)
    return preprocess_config, dataset_config, dataset_config_path


def _next_json_record(
    iterator: Iterator[Dict[str, Any]],
    manifest_name: str,
    expected_index: int,
) -> Dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError(
            "Manifest {0} ended early at record index {1}".format(manifest_name, expected_index)
        ) from exc


def _ensure_manifest_exhausted(
    iterator: Iterator[Dict[str, Any]],
    manifest_name: str,
) -> None:
    try:
        extra = next(iterator)
    except StopIteration:
        return
    raise ValueError(
        "Manifest {0} contains extra records after expected end; first extra sample_id={1}".format(
            manifest_name,
            extra.get("sample_id"),
        )
    )


def _compare_manifest_record(
    manifest_name: str,
    expected_index: int,
    expected_record: Dict[str, Any],
    actual_record: Dict[str, Any],
) -> None:
    if actual_record != expected_record:
        raise ValueError(
            "Manifest mismatch in {0} at index {1} for sample_id={2}".format(
                manifest_name,
                expected_index,
                expected_record.get("sample_id"),
            )
        )


def _validate_manifest_record_shape(
    record: Dict[str, Any],
    expected_dataset_name: str,
    expected_label_dim: int,
    manifest_name: str,
) -> None:
    required_fields = {
        "sample_id",
        "dataset_name",
        "image_path",
        "text_source",
        "label_vector",
        "raw_index",
        "meta",
    }
    missing_fields = sorted(required_fields - set(record.keys()))
    if missing_fields:
        raise ValueError(
            "Manifest record in {0} missing required fields: {1}".format(
                manifest_name,
                missing_fields,
            )
        )
    if record["dataset_name"] != expected_dataset_name:
        raise ValueError(
            "Unexpected dataset_name in {0}: {1} != {2}".format(
                manifest_name,
                record["dataset_name"],
                expected_dataset_name,
            )
        )
    if not Path(record["image_path"]).is_file():
        raise FileNotFoundError(
            "Manifest image_path does not exist in {0}: {1}".format(
                manifest_name,
                record["image_path"],
            )
        )
    if not isinstance(record["text_source"], str):
        raise ValueError(
            "Manifest text_source must be a string in {0} for sample_id={1}".format(
                manifest_name,
                record["sample_id"],
            )
        )
    if expected_dataset_name != "mirflickr25k" and not record["text_source"].strip():
        raise ValueError(
            "Manifest text_source is empty in {0} for sample_id={1}".format(
                manifest_name,
                record["sample_id"],
            )
        )
    meta = record["meta"]
    if "text_source_path" in meta and not Path(meta["text_source_path"]).exists():
        raise FileNotFoundError(
            "text_source_path does not exist in {0} for sample_id={1}: {2}".format(
                manifest_name,
                record["sample_id"],
                meta["text_source_path"],
            )
        )
    for key in ("text_source_files", "label_source_files"):
        if key in meta:
            for source_path in meta[key]:
                if not Path(source_path).exists():
                    raise FileNotFoundError(
                        "{0} entry does not exist in {1} for sample_id={2}: {3}".format(
                            key,
                            manifest_name,
                            record["sample_id"],
                            source_path,
                        )
                    )
    label_vector = record["label_vector"]
    if len(label_vector) != expected_label_dim:
        raise ValueError(
            "Label dimension mismatch in {0} for sample_id={1}: {2} != {3}".format(
                manifest_name,
                record["sample_id"],
                len(label_vector),
                expected_label_dim,
            )
        )
    if any(value not in (0, 1) for value in label_vector):
        raise ValueError(
            "Label vector contains non-binary values in {0} for sample_id={1}".format(
                manifest_name,
                record["sample_id"],
            )
        )


def validate_preprocess_artifacts(
    project_root: Path,
    preprocess_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    dataset_config_path: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    adapter = DATASET_ADAPTERS[dataset_name](project_root, preprocess_config, dataset_config)
    adapter.validate_dataset_config()
    adapter.prepare()
    protocol = adapter.protocol()

    output_paths = build_output_paths(
        (project_root / preprocess_config["paths"]["processed_root"]).resolve(),
        dataset_name,
    )
    required_paths = [
        output_paths.manifest_raw_path,
        output_paths.manifest_filtered_path,
        output_paths.manifest_meta_path,
        output_paths.query_ids_path,
        output_paths.retrieval_ids_path,
        output_paths.train_ids_path,
        output_paths.split_summary_path,
        output_paths.preprocess_summary_path,
        output_paths.config_snapshot_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing required Stage 1 artifacts for {0}: {1}".format(
                dataset_name,
                missing_paths,
            )
        )

    manifest_meta = read_json(output_paths.manifest_meta_path)
    preprocess_summary = read_json(output_paths.preprocess_summary_path)
    split_summary = read_json(output_paths.split_summary_path)
    config_snapshot = read_json(output_paths.config_snapshot_path)

    if config_snapshot["dataset_config_path"] != str(dataset_config_path):
        raise ValueError("config_snapshot dataset_config_path mismatch.")

    raw_label_names = list(adapter.raw_label_names())
    filtered_label_names = list(adapter.filtered_label_names())
    protocol_metadata = adapter.protocol_metadata()
    dataset_metadata = adapter.dataset_metadata()

    if manifest_meta["raw_label_names"] != raw_label_names:
        raise ValueError("manifest_meta raw_label_names mismatch.")
    if manifest_meta["filtered_label_names"] != filtered_label_names:
        raise ValueError("manifest_meta filtered_label_names mismatch.")
    if preprocess_summary["raw_label_names"] != raw_label_names:
        raise ValueError("preprocess_summary raw_label_names mismatch.")
    if preprocess_summary["filtered_label_names"] != filtered_label_names:
        raise ValueError("preprocess_summary filtered_label_names mismatch.")
    if manifest_meta["protocol_metadata"] != protocol_metadata:
        raise ValueError("manifest_meta protocol_metadata mismatch.")
    if preprocess_summary["protocol_metadata"] != protocol_metadata:
        raise ValueError("preprocess_summary protocol_metadata mismatch.")
    if manifest_meta["dataset_metadata"] != dataset_metadata:
        raise ValueError("manifest_meta dataset_metadata mismatch.")
    if preprocess_summary["dataset_metadata"] != dataset_metadata:
        raise ValueError("preprocess_summary dataset_metadata mismatch.")

    raw_manifest_iter = iter_manifest_jsonl(output_paths.manifest_raw_path)
    filtered_manifest_iter = iter_manifest_jsonl(output_paths.manifest_filtered_path)

    raw_record_count = 0
    filtered_record_count = 0
    raw_empty_text_rows = 0
    filtered_empty_text_rows = 0
    filtered_sample_ids: List[str] = []

    for expected_raw in adapter.iter_raw_samples():
        expected_raw_dict = expected_raw.to_manifest_dict()
        actual_raw_dict = _next_json_record(raw_manifest_iter, "manifest_raw.jsonl", raw_record_count)
        _compare_manifest_record(
            "manifest_raw.jsonl",
            raw_record_count,
            expected_raw_dict,
            actual_raw_dict,
        )
        _validate_manifest_record_shape(
            actual_raw_dict,
            expected_dataset_name=dataset_name,
            expected_label_dim=protocol.label_dim_raw,
            manifest_name="manifest_raw.jsonl",
        )
        if actual_raw_dict["text_source"] == "":
            raw_empty_text_rows += 1
        raw_record_count += 1

        expected_filtered = adapter.filter_raw_sample(expected_raw)
        if expected_filtered is None:
            continue

        expected_filtered_dict = expected_filtered.to_manifest_dict()
        actual_filtered_dict = _next_json_record(
            filtered_manifest_iter,
            "manifest_filtered.jsonl",
            filtered_record_count,
        )
        _compare_manifest_record(
            "manifest_filtered.jsonl",
            filtered_record_count,
            expected_filtered_dict,
            actual_filtered_dict,
        )
        _validate_manifest_record_shape(
            actual_filtered_dict,
            expected_dataset_name=dataset_name,
            expected_label_dim=protocol.label_dim_output,
            manifest_name="manifest_filtered.jsonl",
        )
        if actual_filtered_dict["text_source"] == "":
            filtered_empty_text_rows += 1
        filtered_sample_ids.append(expected_filtered.sample_id)
        filtered_record_count += 1

    _ensure_manifest_exhausted(raw_manifest_iter, "manifest_raw.jsonl")
    _ensure_manifest_exhausted(filtered_manifest_iter, "manifest_filtered.jsonl")

    if raw_record_count != protocol.counts.raw_total:
        raise ValueError(
            "raw_record_count mismatch: {0} != {1}".format(
                raw_record_count,
                protocol.counts.raw_total,
            )
        )
    if filtered_record_count != protocol.counts.filtered_total:
        raise ValueError(
            "filtered_record_count mismatch: {0} != {1}".format(
                filtered_record_count,
                protocol.counts.filtered_total,
            )
        )
    if dataset_name == "mirflickr25k":
        expected_empty_text_rows = int(dataset_metadata["empty_text_rows"])
        if raw_empty_text_rows != expected_empty_text_rows:
            raise ValueError(
                "MIRFlickr empty_text_rows mismatch: {0} != {1}".format(
                    raw_empty_text_rows,
                    expected_empty_text_rows,
                )
            )
        if filtered_empty_text_rows != 0:
            raise ValueError(
                "MIRFlickr filtered manifest must not contain empty text rows."
            )
        if int(protocol_metadata["empty_text_rows_removed"]) != raw_empty_text_rows:
            raise ValueError("MIRFlickr protocol_metadata empty_text_rows_removed mismatch.")
        if int(protocol_metadata["final_filtered_count"]) != filtered_record_count:
            raise ValueError("MIRFlickr protocol_metadata final_filtered_count mismatch.")

    query_ids = read_text_lines(output_paths.query_ids_path)
    retrieval_ids = read_text_lines(output_paths.retrieval_ids_path)
    train_ids = read_text_lines(output_paths.train_ids_path)
    validate_split_relations(query_ids, retrieval_ids, train_ids)

    rebuilt_split = build_query_retrieval_train_splits(
        sample_ids=filtered_sample_ids,
        query_size=protocol.counts.query,
        train_size=protocol.counts.train,
        seed=protocol.split_seed,
    )
    if list(rebuilt_split.query_ids) != query_ids:
        raise ValueError("query_ids do not match the deterministic rerun split.")
    if list(rebuilt_split.retrieval_ids) != retrieval_ids:
        raise ValueError("retrieval_ids do not match the deterministic rerun split.")
    if list(rebuilt_split.train_ids) != train_ids:
        raise ValueError("train_ids do not match the deterministic rerun split.")

    if split_summary["counts"]["query"] != protocol.counts.query:
        raise ValueError("split_summary query count mismatch.")
    if split_summary["counts"]["retrieval"] != protocol.counts.retrieval:
        raise ValueError("split_summary retrieval count mismatch.")
    if split_summary["counts"]["train"] != protocol.counts.train:
        raise ValueError("split_summary train count mismatch.")

    summary: Dict[str, Any] = {
        "stage": preprocess_config["stage_name"],
        "dataset_name": dataset_name,
        "config_validation": "passed",
        "artifact_validation": {
            "image_paths_exist": True,
            "text_sources_exist": True,
            "label_dimensions_correct": True,
            "raw_record_count": raw_record_count,
            "filtered_record_count": filtered_record_count,
            "empty_text_rows": raw_empty_text_rows,
            "filtered_empty_text_rows": filtered_empty_text_rows,
            "query_count": len(query_ids),
            "retrieval_count": len(retrieval_ids),
            "train_count": len(train_ids),
            "query_retrieval_disjoint": True,
            "train_subset_of_retrieval": True,
            "split_internal_uniqueness": True,
        },
        "rerun_consistency": {
            "status": "passed",
            "recomputed_query_count": len(rebuilt_split.query_ids),
            "recomputed_retrieval_count": len(rebuilt_split.retrieval_ids),
            "recomputed_train_count": len(rebuilt_split.train_ids),
        },
        "raw_label_names": raw_label_names,
        "filtered_label_names": filtered_label_names,
        "protocol_metadata": protocol_metadata,
        "dataset_metadata": dataset_metadata,
        "preprocess_config_path": str((project_root / "configs" / "preprocess.yaml").resolve()),
        "dataset_config_path": str(dataset_config_path),
        "artifacts": {
            "manifest_raw": str(output_paths.manifest_raw_path),
            "manifest_filtered": str(output_paths.manifest_filtered_path),
            "manifest_meta": str(output_paths.manifest_meta_path),
            "query_ids": str(output_paths.query_ids_path),
            "retrieval_ids": str(output_paths.retrieval_ids_path),
            "train_ids": str(output_paths.train_ids_path),
            "split_summary": str(output_paths.split_summary_path),
            "preprocess_summary": str(output_paths.preprocess_summary_path),
            "config_snapshot": str(output_paths.config_snapshot_path),
        },
    }
    return summary


def main() -> int:
    args = parse_args()
    preprocess_config_path = Path(args.config).resolve()
    preprocess_config, dataset_config, dataset_config_path = load_configs(
        preprocess_config_path,
        args.dataset,
    )
    summary = validate_preprocess_artifacts(
        project_root=PROJECT_ROOT,
        preprocess_config=preprocess_config,
        dataset_config=dataset_config,
        dataset_config_path=dataset_config_path,
        dataset_name=args.dataset,
    )

    output_paths = build_output_paths(
        (PROJECT_ROOT / preprocess_config["paths"]["processed_root"]).resolve(),
        args.dataset,
    )
    if args.write_summary:
        write_json(output_paths.validator_summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
