from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.data_contract import DATASET_PATH, format_summary, inspect_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the local sepsis dataset.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to the local dataset CSV. Default: {DATASET_PATH}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = inspect_csv(args.csv)
    print(format_summary(summary))
    return 1 if summary.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
