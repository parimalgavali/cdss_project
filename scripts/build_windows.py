from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from cdss_project.windows import (
    WINDOW_SIZE,
    build_windows,
    format_window_summary,
    save_windows,
    summarize_windows,
)

DEFAULT_INPUT_PATH = Path("data/processed/mimic_sepsis_preprocessed.csv")
DEFAULT_OUTPUT_PATH = Path("data/processed/windows_w6.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 6-hour sliding windows with horizons.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Preprocessed hourly CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Compressed .npz output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument(
        "--min-observed-hours",
        type=int,
        default=0,
        help="Drop windows with fewer non-padding hours (needs is_padding). Default: 0 (keep all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize windows without writing output files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(
            f"Preprocessed CSV not found at {args.input}. Run scripts/preprocess_data.py first.",
            file=sys.stderr,
        )
        return 2
    if not args.dry_run and args.output.exists() and not args.overwrite:
        print(f"Output file already exists: {args.output}. Use --overwrite.", file=sys.stderr)
        return 2

    dataset = build_windows(
        pd.read_csv(args.input),
        window_size=args.window_size,
        min_observed_hours=args.min_observed_hours,
    )
    print(format_window_summary(summarize_windows(dataset)))
    if args.dry_run:
        print("No output files were written.")
        return 0

    save_windows(dataset, args.output, overwrite=args.overwrite)
    print(f"Output file: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
