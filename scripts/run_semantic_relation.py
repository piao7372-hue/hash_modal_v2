from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import SUPPORTED_DATASETS
from src.semantic import (
    build_run_summary,
    load_semantic_relation_config,
    run_formal_semantic_relation,
    write_semantic_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal Stage 3 semantic relation pipeline for one dataset."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "semantic_relation.yaml"),
        help="Path to the top-level Stage 3 semantic relation config.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_semantic_relation_config(PROJECT_ROOT, config_path)
    artifacts = run_formal_semantic_relation(config, args.dataset)
    write_semantic_cache(config, args.dataset, artifacts)
    print(json.dumps(build_run_summary(config, args.dataset, artifacts), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
