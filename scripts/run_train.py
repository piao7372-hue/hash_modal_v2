from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import SUPPORTED_DATASETS, load_train_config, run_formal_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal Stage 4 training pipeline for one dataset.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train.yaml"),
        help="Path to the top-level Stage 4 training config.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset name to train on.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit run name. If omitted, a timestamped formal run name is generated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_train_config(PROJECT_ROOT, Path(args.config).resolve())
    summary = run_formal_training(config=config, dataset_name=args.dataset, run_name=args.run_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
