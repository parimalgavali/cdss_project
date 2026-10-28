from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from cdss_project.baselines import (
    MODEL_NAMES,
    aggregate_windows,
    choose_validation_fold,
    run_baseline,
)
from cdss_project.windows import load_windows

DEFAULT_WINDOWS_PATH = Path("data/processed/windows_w6.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/baselines")
SHORT_NAMES = {"lr": "logistic_regression", "rf": "random_forest"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LR/RF baselines on aggregated windows and report horizon-wise metrics."
    )
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Results go to <output-dir>/test_fold_<k>/. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*SHORT_NAMES, *MODEL_NAMES],
        default=["lr", "rf"],
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=None,
        help="Pick the threshold reaching this validation sensitivity instead of max F1.",
    )
    parser.add_argument(
        "--alert-persistence",
        type=int,
        default=1,
        help="Windows (hours) an alert must hold before it counts for EDA. Default: 1 (first crossing).",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=20)
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Keep the model fitted without the validation fold instead of refitting on all training stays.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.windows.exists():
        print(
            f"Windows file not found at {args.windows}. Run scripts/build_windows.py first.",
            file=sys.stderr,
        )
        return 2

    dataset = load_windows(args.windows)
    if dataset.fold is None or dataset.split is None:
        print("Windows file has no fold/split arrays; rebuild it.", file=sys.stderr)
        return 2
    test_fold, _ = choose_validation_fold(dataset.fold, dataset.split.astype(str))
    run_dir = args.output_dir / f"test_fold_{test_fold}"
    metrics_path = run_dir / "metrics_by_horizon.csv"
    if metrics_path.exists() and not args.overwrite:
        print(f"Results already exist in {run_dir}. Use --overwrite.", file=sys.stderr)
        return 2

    features = aggregate_windows(dataset.X, dataset.feature_columns)
    model_names = [SHORT_NAMES.get(name, name) for name in dict.fromkeys(args.models)]

    results = []
    for model_name in model_names:
        print(f"Training {model_name} ...", flush=True)
        model_kwargs = (
            {
                "rf_n_estimators": args.rf_n_estimators,
                "rf_min_samples_leaf": args.rf_min_samples_leaf,
            }
            if model_name == "random_forest"
            else {}
        )
        results.append(
            run_baseline(
                dataset,
                model_name,
                target_sensitivity=args.target_sensitivity,
                alert_persistence=args.alert_persistence,
                refit=not args.no_refit,
                features=features,
                **model_kwargs,
            )
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.concat([result.metrics for result in results], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    for result in results:
        result.predictions.to_csv(run_dir / f"{result.model_name}_predictions.csv.gz", index=False)
        result.alerts.to_csv(run_dir / f"{result.model_name}_stay_alerts.csv", index=False)
    summary = {
        "windows_file": str(args.windows),
        "test_fold": results[0].test_fold,
        "validation_fold": results[0].validation_fold,
        "feature_count": len(features.names),
        "alert_persistence": args.alert_persistence,
        "models": {
            result.model_name: {
                "threshold": result.threshold,
                "params": result.params,
                "overall": result.metrics.iloc[0].drop(["model", "horizon"]).to_dict(),
                "early_detection": result.early_detection,
            }
            for result in results
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    columns = ["model", "horizon", "auroc", "auprc", "f1", "sensitivity", "specificity"]
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for result in results:
        print(f"{result.model_name} threshold={result.threshold:.4f} early_detection={result.early_detection}")
    print(f"Results written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
