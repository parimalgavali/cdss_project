from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cdss_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics and plots.")
    parser.add_argument("--experiment-root", type=Path, default=Path("outputs/experiments"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.experiment_root
    comparison_dir = root / "comparison"
    plots_dir = comparison_dir / "plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    overall_frames = []
    by_hour_frames = []
    for overall_path in root.glob("*/*/metrics_overall.csv"):
        window_name = overall_path.parents[1].name
        model_name = overall_path.parent.name
        frame = pd.read_csv(overall_path)
        frame.insert(0, "window_name", window_name)
        frame.insert(1, "model_name", model_name)
        overall_frames.append(frame)

    for by_hour_path in root.glob("*/*/metrics_by_hour.csv"):
        window_name = by_hour_path.parents[1].name
        model_name = by_hour_path.parent.name
        frame = pd.read_csv(by_hour_path)
        frame.insert(0, "window_name", window_name)
        frame.insert(1, "model_name", model_name)
        by_hour_frames.append(frame)

    if not overall_frames:
        print(f"No metrics found under {root}", file=sys.stderr)
        return 2

    overall = pd.concat(overall_frames, ignore_index=True)
    by_hour = pd.concat(by_hour_frames, ignore_index=True)
    overall.to_csv(comparison_dir / "all_metrics_overall.csv", index=False)
    by_hour.to_csv(comparison_dir / "all_metrics_by_hour.csv", index=False)

    model_comparison = overall.sort_values(["auprc", "auroc"], ascending=False)
    model_comparison.to_csv(comparison_dir / "model_window_comparison.csv", index=False)

    for metric in ["auroc", "auprc", "f1", "sensitivity", "specificity"]:
        _plot_metric_by_hour(by_hour, metric, plots_dir / f"{metric}_comparison.png")

    print(f"Wrote comparison outputs to {comparison_dir}")
    return 0


def _plot_metric_by_hour(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    for (window_name, model_name), group in frame.groupby(["window_name", "model_name"]):
        group = group.sort_values("prediction_hour_before", ascending=False)
        label = f"{window_name} / {model_name}"
        plt.plot(group["prediction_hour_before"], group[metric], marker="o", linewidth=1, label=label)
    plt.gca().invert_xaxis()
    plt.xlabel("hour_before")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} by prediction hour")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


if __name__ == "__main__":
    sys.exit(main())
