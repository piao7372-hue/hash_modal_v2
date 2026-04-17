from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import DATASET_ADAPTERS, SUPPORTED_DATASETS
from src.datasets.manifest_builder import (
    build_manifest_meta,
    build_output_paths,
    build_preprocess_summary,
    prepare_output_dirs,
    stable_json_dumps,
    write_json,
    write_text_lines,
)
from src.datasets.split_builder import build_query_retrieval_train_splits, build_split_summary
from scripts.validate_preprocess import load_configs, validate_preprocess_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal Stage 1 preprocessing pipeline for one dataset."
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
        help="Dataset name to preprocess.",
    )
    return parser.parse_args()


def _write_config_snapshot(
    output_path: Path,
    preprocess_config_path: Path,
    dataset_config_path: Path,
    preprocess_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
) -> None:
    write_json(
        output_path,
        {
            "preprocess_config_path": str(preprocess_config_path),
            "dataset_config_path": str(dataset_config_path),
            "preprocess_config": preprocess_config,
            "dataset_config": dataset_config,
        },
    )


def _write_manifest_pair(
    adapter,
    output_paths,
) -> Tuple[int, int, List[str]]:
    raw_count = 0
    filtered_count = 0
    filtered_sample_ids: List[str] = []

    with output_paths.manifest_raw_path.open("w", encoding="utf-8") as raw_handle, output_paths.manifest_filtered_path.open(
        "w",
        encoding="utf-8",
    ) as filtered_handle:
        for raw_record in adapter.iter_raw_samples():
            raw_handle.write(stable_json_dumps(raw_record.to_manifest_dict()) + "\n")
            raw_count += 1

            filtered_record = adapter.filter_raw_sample(raw_record)
            if filtered_record is None:
                continue

            filtered_handle.write(stable_json_dumps(filtered_record.to_manifest_dict()) + "\n")
            filtered_count += 1
            filtered_sample_ids.append(filtered_record.sample_id)

    return raw_count, filtered_count, filtered_sample_ids


def main() -> int:
    args = parse_args()
    preprocess_config_path = Path(args.config).resolve()
    preprocess_config, dataset_config, dataset_config_path = load_configs(
        preprocess_config_path,
        args.dataset,
    )

    adapter = DATASET_ADAPTERS[args.dataset](PROJECT_ROOT, preprocess_config, dataset_config)
    adapter.validate_dataset_config()
    adapter.prepare()

    output_paths = build_output_paths(
        (PROJECT_ROOT / preprocess_config["paths"]["processed_root"]).resolve(),
        args.dataset,
    )
    prepare_output_dirs(output_paths)

    _write_config_snapshot(
        output_path=output_paths.config_snapshot_path,
        preprocess_config_path=preprocess_config_path,
        dataset_config_path=dataset_config_path,
        preprocess_config=preprocess_config,
        dataset_config=dataset_config,
    )

    raw_count, filtered_count, filtered_sample_ids = _write_manifest_pair(adapter, output_paths)
    protocol = adapter.protocol()
    if raw_count != protocol.counts.raw_total:
        raise ValueError(
            "raw manifest count mismatch for {0}: {1} != {2}".format(
                args.dataset,
                raw_count,
                protocol.counts.raw_total,
            )
        )
    if filtered_count != protocol.counts.filtered_total:
        raise ValueError(
            "filtered manifest count mismatch for {0}: {1} != {2}".format(
                args.dataset,
                filtered_count,
                protocol.counts.filtered_total,
            )
        )

    split_result = build_query_retrieval_train_splits(
        sample_ids=filtered_sample_ids,
        query_size=protocol.counts.query,
        train_size=protocol.counts.train,
        seed=protocol.split_seed,
    )
    write_text_lines(output_paths.query_ids_path, list(split_result.query_ids))
    write_text_lines(output_paths.retrieval_ids_path, list(split_result.retrieval_ids))
    write_text_lines(output_paths.train_ids_path, list(split_result.train_ids))
    write_json(output_paths.split_summary_path, build_split_summary(args.dataset, split_result))

    dataset_metadata = adapter.dataset_metadata()
    protocol_metadata = adapter.protocol_metadata()
    write_json(
        output_paths.manifest_meta_path,
        build_manifest_meta(
            dataset_name=args.dataset,
            protocol=protocol,
            raw_record_count=raw_count,
            filtered_record_count=filtered_count,
            raw_label_names=adapter.raw_label_names(),
            filtered_label_names=adapter.filtered_label_names(),
            protocol_metadata=protocol_metadata,
            dataset_metadata=dataset_metadata,
        ),
    )

    split_counts = {
        "query": len(split_result.query_ids),
        "retrieval": len(split_result.retrieval_ids),
        "train": len(split_result.train_ids),
    }
    write_json(
        output_paths.preprocess_summary_path,
        build_preprocess_summary(
            dataset_name=args.dataset,
            protocol=protocol,
            raw_record_count=raw_count,
            filtered_record_count=filtered_count,
            split_counts=split_counts,
            output_paths=output_paths,
            raw_label_names=adapter.raw_label_names(),
            filtered_label_names=adapter.filtered_label_names(),
            protocol_metadata=protocol_metadata,
            dataset_metadata=dataset_metadata,
        ),
    )

    validator_summary = validate_preprocess_artifacts(
        project_root=PROJECT_ROOT,
        preprocess_config=preprocess_config,
        dataset_config=dataset_config,
        dataset_config_path=dataset_config_path,
        dataset_name=args.dataset,
    )
    write_json(output_paths.validator_summary_path, validator_summary)

    runtime_summary = {
        "stage": preprocess_config["stage_name"],
        "dataset_name": args.dataset,
        "raw_record_count": raw_count,
        "filtered_record_count": filtered_count,
        "query_count": len(split_result.query_ids),
        "retrieval_count": len(split_result.retrieval_ids),
        "train_count": len(split_result.train_ids),
        "raw_label_names": list(adapter.raw_label_names()),
        "filtered_label_names": list(adapter.filtered_label_names()),
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
    }
    print(json.dumps(runtime_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
