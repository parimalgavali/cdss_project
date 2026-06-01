from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.windowing import (
    build_expanding_windows,
    build_fixed_windows,
    build_window_summary,
    format_window_summary,
    summarize_window_plan,
    write_window_dataset,
)

DEFAULT_INPUT_PATH = Path("data/processed/mimic_sepsis_preprocessed.csv")
DEFAULT_OUTPUT_DIR = Path("data/windows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready time-series windows.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the preprocessed CSV. Default: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--window-type",
        choices=("fixed", "expanding"),
        default="fixed",
        help="Windowing strategy to build.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=6,
        help="Fixed window length in hours. Used when --window-type fixed.",
    )
    parser.add_argument(
        "--min-window-length",
        type=int,
        default=3,
        help="Minimum expanding window length in hours. Used when --window-type expanding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to data/windows/<window config>.npz.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and summarize windows without writing output files.",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="For fixed windows, predict from the first hour and pad early windows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Reading input CSV: {args.csv}", flush=True)
    frame = pd.read_csv(args.csv)

    if args.dry_run:
        print("Counting windows without writing output...", flush=True)
        if args.window_type == "fixed":
            summary = summarize_window_plan(
                frame,
                window_type="fixed",
                window_length=args.window_length,
                include_partial=args.include_partial,
            )
        else:
            summary = summarize_window_plan(
                frame,
                window_type="expanding",
                min_window_length=args.min_window_length,
            )
        print(format_window_summary(summary))
        return 0

    if args.window_type == "fixed":
        if args.include_partial:
            print(f"Building padded fixed {args.window_length}-hour windows...", flush=True)
        else:
            print(f"Building fixed {args.window_length}-hour windows...", flush=True)
        dataset = build_fixed_windows(
            frame,
            window_length=args.window_length,
            include_partial=args.include_partial,
        )
        suffix = "padded_windows" if args.include_partial else "windows"
        default_output = DEFAULT_OUTPUT_DIR / f"fixed_{args.window_length}h_{suffix}.npz"
        output_path = args.output or default_output
        summary = build_window_summary(
            frame,
            dataset,
            window_length=args.window_length,
            output_path=output_path,
        )
    else:
        print(f"Building expanding windows from {args.min_window_length} hours...", flush=True)
        dataset = build_expanding_windows(frame, min_window_length=args.min_window_length)
        default_output = DEFAULT_OUTPUT_DIR / f"expanding_min_{args.min_window_length}h_windows.npz"
        output_path = args.output or default_output
        summary = build_window_summary(
            frame,
            dataset,
            min_window_length=args.min_window_length,
            output_path=output_path,
        )

    try:
        print(f"Writing window dataset: {output_path}", flush=True)
        write_window_dataset(dataset, output_path, overwrite=args.overwrite)
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return 2

    print(format_window_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
