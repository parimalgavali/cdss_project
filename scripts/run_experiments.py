from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.experiments import (
    load_window_dataset,
    run_classical_experiment,
    run_neural_experiment,
    subset_window_dataset_by_stays,
)

WINDOW_FILENAMES = {
    "fixed_3h_padded": "fixed_3h_padded_windows.npz",
    "fixed_6h_padded": "fixed_6h_padded_windows.npz",
    "expanding_min_1h": "expanding_min_1h_windows.npz",
}

MODELS = ("logistic_regression", "random_forest", "lstm", "gru")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model/window experiments.")
    parser.add_argument("--all", action="store_true", help="Run all configured windows and models.")
    parser.add_argument("--windows", nargs="+", choices=sorted(WINDOW_FILENAMES), default=None)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None)
    parser.add_argument("--window-root", type=Path, default=Path("data/windows"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--smoke", action="store_true", help="Run on a small stay subset for testing.")
    parser.add_argument("--smoke-stays", type=int, default=40)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    window_names = args.windows or list(WINDOW_FILENAMES)
    model_names = args.models or list(MODELS)
    if not args.all and args.windows is None and args.models is None:
        print("Nothing selected. Use --all or provide --windows/--models.", file=sys.stderr)
        return 2

    for window_name in window_names:
        window_path = args.window_root / WINDOW_FILENAMES[window_name]
        if not window_path.exists():
            print(f"Missing window file: {window_path}", file=sys.stderr)
            return 2
        print(f"Loading window dataset: {window_path}", flush=True)
        dataset = load_window_dataset(window_path)
        if args.smoke:
            dataset = subset_window_dataset_by_stays(dataset, max_stays=args.smoke_stays)
            print(f"Smoke mode: using {dataset.metadata['stay_id'].nunique()} stays", flush=True)

        for model_name in model_names:
            output_dir = args.output_root / window_name / model_name
            print(f"Training {model_name} on {window_name}", flush=True)
            if model_name in {"logistic_regression", "random_forest"}:
                run_classical_experiment(
                    dataset,
                    model_name=model_name,
                    output_dir=output_dir,
                    random_state=args.random_state,
                    max_train_windows=args.max_train_windows,
                )
            else:
                run_neural_experiment(
                    dataset,
                    model_name=model_name,
                    output_dir=output_dir,
                    epochs=1 if args.smoke else args.epochs,
                    batch_size=args.batch_size,
                    random_state=args.random_state,
                    max_train_windows=args.max_train_windows,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
