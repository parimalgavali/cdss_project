from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.reporting import (
    COUNT_METRICS,
    PLOT_METRICS,
    load_fold_tables,
    load_observed_subset_metrics,
    markdown_report,
    minimum_reliable_horizon,
    plot_eda,
    plot_metric_by_horizon,
    summarize_across_folds,
    summarize_eda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-fold results into mean ± std tables and plots."
    )
    parser.add_argument(
        "--result-dirs",
        nargs="+",
        type=Path,
        default=[Path("outputs/baselines"), Path("outputs/rnn")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--auroc-threshold", type=float, default=0.80)
    parser.add_argument(
        "--min-observed-hours",
        type=int,
        default=6,
        help="Also score windows with at least this many recorded hours (0 disables).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = args.output_dir / "metrics_mean_std.csv"
    if metrics_path.exists() and not args.overwrite:
        print(f"Summary already exists in {args.output_dir}. Use --overwrite.", file=sys.stderr)
        return 2
    metrics = load_fold_tables(args.result_dirs, "metrics_by_horizon.csv")
    if metrics.empty:
        print("No metrics_by_horizon.csv found under the result dirs.", file=sys.stderr)
        return 2

    summary = summarize_across_folds(metrics)
    reliable = minimum_reliable_horizon(summary, threshold=args.auroc_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_path, index=False)
    reliable.to_csv(args.output_dir / "minimum_reliable_horizon.csv", index=False)
    figures = []
    for metric in PLOT_METRICS:
        figures.append(args.figures_dir / f"{metric}_by_horizon.png")
        plot_metric_by_horizon(
            summary,
            metric,
            args.figures_dir / f"{metric}_by_horizon.png",
            reference_line=args.auroc_threshold if metric == "auroc" else None,
        )

    folds = sorted(metrics["fold"].unique().tolist())
    overall = summary[(summary["horizon"] == "all") & (summary["metric"] == "auroc")]
    print(f"Folds found: {folds}")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(reliable.to_string(index=False))

    eda = load_fold_tables(args.result_dirs, "eda_vs_baselines.csv")
    eda_summary = None
    if eda.empty:
        print("No eda_vs_baselines.csv found; EDA summary skipped.")
    else:
        eda_summary = summarize_eda(eda)
        eda_summary.to_csv(args.output_dir / "eda_mean_std.csv", index=False)
        figures.append(plot_eda(eda_summary, args.figures_dir / "eda_vs_baselines.png"))
        print(eda_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    observed_summary = observed_reliable = None
    if args.min_observed_hours > 0:
        observed = load_observed_subset_metrics(args.result_dirs, args.min_observed_hours)
        if observed.empty:
            print("No predictions with observed_hours found; observed-window summary skipped.")
        else:
            observed_summary = summarize_across_folds(
                observed, metric_columns=(*COUNT_METRICS, *PLOT_METRICS)
            )
            observed_reliable = minimum_reliable_horizon(
                observed_summary, threshold=args.auroc_threshold
            )
            suffix = f"observed_ge{args.min_observed_hours}"
            observed_summary.to_csv(args.output_dir / f"metrics_mean_std_{suffix}.csv", index=False)
            observed_reliable.to_csv(
                args.output_dir / f"minimum_reliable_horizon_{suffix}.csv", index=False
            )
            figures.append(args.figures_dir / f"auroc_by_horizon_{suffix}.png")
            plot_metric_by_horizon(
                observed_summary, "auroc", figures[-1], reference_line=args.auroc_threshold
            )
    report = markdown_report(
        summary,
        reliable,
        eda_summary,
        figures=figures,
        observed_summary=observed_summary,
        observed_reliable=observed_reliable,
        min_observed_hours=args.min_observed_hours,
    )
    (args.output_dir / "results.md").write_text(report)
    print(f"Tables written to {args.output_dir}, figures to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
