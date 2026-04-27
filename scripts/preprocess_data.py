from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.data_contract import DATASET_PATH
from cdss_project.preprocessing import format_dry_run_summary, run_preprocessing_dry_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run professor-style preprocessing.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to the local dataset CSV. Default: {DATASET_PATH}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize preprocessing without writing output files.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        print("Only --dry-run is supported for now; no output files were written.", file=sys.stderr)
        return 2

    summary = run_preprocessing_dry_run(
        args.csv,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(format_dry_run_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
