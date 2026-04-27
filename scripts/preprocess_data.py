from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.data_contract import DATASET_PATH
from cdss_project.preprocessing import (
    format_dry_run_summary,
    format_write_summary,
    run_preprocessing_dry_run,
    write_preprocessed_csv,
)

DEFAULT_OUTPUT_PATH = Path("data/processed/mimic_sepsis_preprocessed.csv")


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
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Processed CSV output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        summary = run_preprocessing_dry_run(
            args.csv,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(format_dry_run_summary(summary))
        return 0

    try:
        summary = write_preprocessed_csv(
            args.csv,
            args.output,
            test_size=args.test_size,
            random_state=args.random_state,
            overwrite=args.overwrite,
        )
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(format_write_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
