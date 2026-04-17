from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import SUPPORTED_DATASETS, build_model_output_validator_summary, load_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal Stage 4 model outputs for one training run.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train.yaml"),
        help="Path to the top-level Stage 4 training config.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name associated with the run.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Existing Stage 4 run name under outputs/train/<dataset>/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_train_config(PROJECT_ROOT, Path(args.config).resolve())
    summary = build_model_output_validator_summary(config=config, dataset_name=args.dataset, run_name=args.run_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if bool(summary["validator_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
