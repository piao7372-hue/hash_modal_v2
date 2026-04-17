from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import SUPPORTED_DATASETS
from src.features import load_feature_extraction_config, run_formal_feature_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal Stage 2 feature extraction pipeline for one dataset."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "feature_extraction.yaml"),
        help="Path to the top-level Stage 2 feature extraction config.",
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
    config = load_feature_extraction_config(PROJECT_ROOT, config_path)
    summary = run_formal_feature_extraction(config, args.dataset)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
