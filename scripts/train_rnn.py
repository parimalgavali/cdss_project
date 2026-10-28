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

from cdss_project.baselines import choose_validation_fold
from cdss_project.evaluation import early_detection_advantage
from cdss_project.rnn import CELL_TYPES, TrainConfig, run_rnn
from cdss_project.windows import load_windows

DEFAULT_WINDOWS_PATH = Path("data/processed/windows_w6.npz")
DEFAULT_OUTPUT_DIR = Path("outputs/rnn")
DEFAULT_BASELINE_DIR = Path("outputs/baselines")


def parse_args() -> argparse.Namespace:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train LSTM/GRU models on sliding windows and report horizon-wise metrics."
    )
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Results go to <output-dir>/test_fold_<k>/. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Baseline results used for Early Detection Advantage (same test fold).",
    )
    parser.add_argument("--models", nargs="+", choices=CELL_TYPES, default=list(CELL_TYPES))
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device, help="auto, cpu, cuda, ...")
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=None,
        help="Pick the threshold reaching this validation sensitivity instead of max F1.",
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

    config = TrainConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    results = []
    for cell in dict.fromkeys(args.models):
        print(f"Training {cell} ...", flush=True)
        result = run_rnn(dataset, cell, config=config, target_sensitivity=args.target_sensitivity)
        print(
            f"  {result.params['epochs_trained']} epochs, best epoch {result.params['best_epoch']}",
            flush=True,
        )
        results.append(result)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([result.metrics for result in results], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)
    for result in results:
        result.predictions.to_csv(run_dir / f"{result.model_name}_predictions.csv.gz", index=False)
        result.alerts.to_csv(run_dir / f"{result.model_name}_stay_alerts.csv", index=False)
        pd.DataFrame(result.history).to_csv(run_dir / f"{result.model_name}_history.csv", index=False)

    eda_rows = []
    baseline_run_dir = args.baseline_dir / f"test_fold_{test_fold}"
    for reference_path in sorted(baseline_run_dir.glob("*_stay_alerts.csv")):
        reference = pd.read_csv(reference_path)
        reference_name = reference_path.name.removesuffix("_stay_alerts.csv")
        for result in results:
            eda_rows.append(
                {
                    "model": result.model_name,
                    "reference": reference_name,
                    "eda_hours": early_detection_advantage(result.alerts, reference),
                }
            )
    eda = pd.DataFrame(eda_rows, columns=["model", "reference", "eda_hours"])
    if eda_rows:
        eda.to_csv(run_dir / "eda_vs_baselines.csv", index=False)

    summary = {
        "windows_file": str(args.windows),
        "test_fold": test_fold,
        "validation_fold": results[0].validation_fold,
        "models": {
            result.model_name: {
                "threshold": result.threshold,
                "params": result.params,
                "overall": result.metrics.iloc[0].drop(["model", "horizon"]).to_dict(),
                "early_detection": result.early_detection,
            }
            for result in results
        },
        "eda_vs_baselines": eda_rows,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    columns = ["model", "horizon", "auroc", "auprc", "f1", "sensitivity", "specificity"]
    print(metrics[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if eda_rows:
        print(eda.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    else:
        print(f"No baseline alerts in {baseline_run_dir}; EDA skipped.")
    print(f"Results written to {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
