"""Cross-fold summaries, minimum reliable horizon, and result plots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cdss_project.evaluation import metrics_by_horizon

PLOT_METRICS: tuple[str, ...] = ("auroc", "auprc", "f1", "sensitivity", "specificity")
COUNT_METRICS: tuple[str, ...] = ("windows", "positives")
_FOLD_DIR = re.compile(r"test_fold_(\d+)$")


def load_fold_tables(result_dirs: Iterable[Path], filename: str) -> pd.DataFrame:
    """Concatenate `<dir>/test_fold_<k>/<filename>` with a `fold` column."""
    frames = []
    for result_dir in result_dirs:
        for fold_dir in sorted(Path(result_dir).glob("test_fold_*")):
            match = _FOLD_DIR.search(fold_dir.name)
            path = fold_dir / filename
            if match and path.exists():
                frames.append(pd.read_csv(path, dtype={"horizon": str}).assign(fold=int(match[1])))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_observed_subset_metrics(
    result_dirs: Iterable[Path], min_observed_hours: int
) -> pd.DataFrame:
    """Recompute each fold's metrics on windows with at least `min_observed_hours` recorded hours.

    Uses the saved `<model>_predictions.csv.gz` and each model's original threshold, so models
    trained on all windows are judged only where the recording pattern carries no signal.
    """
    frames = []
    metrics = load_fold_tables(result_dirs, "metrics_by_horizon.csv")
    if metrics.empty:
        return metrics
    for result_dir in result_dirs:
        for fold_dir in sorted(Path(result_dir).glob("test_fold_*")):
            match = _FOLD_DIR.search(fold_dir.name)
            if not match:
                continue
            fold = int(match[1])
            for model in metrics.loc[metrics["fold"] == fold, "model"].unique():
                path = fold_dir / f"{model}_predictions.csv.gz"
                if not path.exists():
                    continue
                predictions = pd.read_csv(path)
                if "observed_hours" not in predictions.columns:
                    continue
                subset = predictions[predictions["observed_hours"] >= min_observed_hours]
                if subset.empty:
                    continue
                threshold = float(
                    metrics.loc[(metrics["fold"] == fold) & (metrics["model"] == model), "threshold"]
                    .iloc[0]
                )
                table = metrics_by_horizon(subset, threshold=threshold)
                table["horizon"] = table["horizon"].astype(str)
                table.insert(0, "model", model)
                frames.append(table.assign(fold=fold))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_across_folds(
    metrics: pd.DataFrame, *, metric_columns: Sequence[str] = PLOT_METRICS
) -> pd.DataFrame:
    """Long table: model, horizon, metric, mean, std (ddof=1), n_folds."""
    long = metrics.melt(
        id_vars=["model", "horizon", "fold"],
        value_vars=list(metric_columns),
        var_name="metric",
    )
    summary = (
        long.groupby(["model", "horizon", "metric"], sort=False)["value"]
        .agg(mean="mean", std="std", n_folds="count")
        .reset_index()
    )
    return summary


def minimum_reliable_horizon(
    summary: pd.DataFrame, *, metric: str = "auroc", threshold: float = 0.80
) -> pd.DataFrame:
    """Largest h such that the mean metric reaches the threshold at every horizon 1..h (0 if none)."""
    rows = []
    per_horizon = summary[(summary["metric"] == metric) & (summary["horizon"] != "all")]
    for model, group in per_horizon.groupby("model", sort=False):
        means = group.assign(horizon=group["horizon"].astype(int)).set_index("horizon")["mean"]
        reliable = 0
        for horizon in sorted(means.index):
            if horizon != reliable + 1 or means[horizon] < threshold:
                break
            reliable = horizon
        rows.append(
            {"model": model, "metric": metric, "threshold": threshold, "max_reliable_horizon": reliable}
        )
    return pd.DataFrame(rows, columns=["model", "metric", "threshold", "max_reliable_horizon"])


def summarize_eda(eda: pd.DataFrame) -> pd.DataFrame:
    return (
        eda.groupby(["model", "reference"], sort=False)["eda_hours"]
        .agg(mean="mean", std="std", n_folds="count")
        .reset_index()
    )


def plot_metric_by_horizon(
    summary: pd.DataFrame, metric: str, path: Path, *, reference_line: float | None = None
) -> Path:
    data = summary[(summary["metric"] == metric) & (summary["horizon"] != "all")]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for model, group in data.groupby("model", sort=False):
        group = group.assign(horizon=group["horizon"].astype(int)).sort_values("horizon")
        std = group["std"].fillna(0)
        axis.plot(group["horizon"], group["mean"], marker="o", markersize=3, label=model)
        axis.fill_between(group["horizon"], group["mean"] - std, group["mean"] + std, alpha=0.2)
    if reference_line is not None:
        axis.axhline(reference_line, color="grey", linestyle="--", linewidth=1)
    axis.invert_xaxis()
    axis.set_xlabel("Prediction horizon (hours before decision point)")
    axis.set_ylabel(f"{metric.upper()} (mean ± std across folds)")
    axis.grid(alpha=0.3)
    axis.legend()
    return _save(figure, path)


def plot_eda(eda_summary: pd.DataFrame, path: Path) -> Path:
    pivot_mean = eda_summary.pivot(index="model", columns="reference", values="mean")
    pivot_std = eda_summary.pivot(index="model", columns="reference", values="std").fillna(0)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    pivot_mean.plot.bar(yerr=pivot_std, ax=axis, capsize=4, rot=0)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Early Detection Advantage (hours, mean ± std)")
    axis.set_xlabel("Model")
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, path)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_importance(importance: pd.DataFrame, kind: str, path: Path, *, title: str) -> Path:
    data = importance.head(15)
    figure, axis = plt.subplots(figsize=(7, 0.35 * len(data) + 1.5))
    if kind == "feature":
        data = data.iloc[::-1]
        axis.barh(data[kind], data["mean_abs_shap"])
        axis.set_xlabel("Mean |Shapley value| (score)")
    else:
        axis.bar(data[kind], data["mean_abs_shap"])
        axis.set_xlabel("Hour within window (t-0 = most recent)")
        axis.set_ylabel("Mean |Shapley value| (score)")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    return _save(figure, path)


def markdown_report(
    summary: pd.DataFrame,
    reliable: pd.DataFrame,
    eda_summary: pd.DataFrame | None = None,
    *,
    figures: Sequence[Path] = (),
    observed_summary: pd.DataFrame | None = None,
    observed_reliable: pd.DataFrame | None = None,
    min_observed_hours: int | None = None,
) -> str:
    """Results summary for the written report: overall mean ± std, reliable horizon, EDA."""
    overall = summary[summary["horizon"] == "all"]
    n_folds = int(overall["n_folds"].max()) if len(overall) else 0
    lines = [
        "# Results summary",
        "",
        f"Generated by `scripts/summarize_results.py` from {n_folds} test fold(s).",
        "",
        "## Overall test-fold metrics (mean ± std)",
        "",
        _markdown_table(_overall_table(summary)),
        "",
        "## Minimum reliable horizon",
        "",
        "Largest horizon h (hours) with mean AUROC ≥ threshold at every horizon 1..h.",
        "",
        _markdown_table(reliable),
    ]
    if eda_summary is not None and len(eda_summary):
        eda = eda_summary.assign(eda_hours=_mean_std(eda_summary["mean"], eda_summary["std"], 2))
        lines += [
            "",
            "## Early Detection Advantage vs baselines (hours, mean ± std)",
            "",
            _markdown_table(eda[["model", "reference", "eda_hours", "n_folds"]]),
        ]
    if observed_summary is not None and len(observed_summary):
        lines += [
            "",
            f"## Windows with at least {min_observed_hours} recorded hours (mean ± std)",
            "",
            "Same models and thresholds, scored only on windows whose recorded-hour count is at",
            "least the cut-off, so the recording pattern cannot separate the classes (decision 0014).",
            "",
            _markdown_table(_overall_table(observed_summary)),
        ]
        if observed_reliable is not None:
            lines += ["", "Minimum reliable horizon on these windows:", ""]
            lines += [_markdown_table(observed_reliable)]
    if figures:
        lines += ["", "## Figures", ""] + [f"- `{path}`" for path in figures]
    return "\n".join(lines) + "\n"


def _overall_table(summary: pd.DataFrame) -> pd.DataFrame:
    overall = summary[summary["horizon"] == "all"]
    digits = overall["metric"].map(lambda metric: 0 if metric in COUNT_METRICS else 3)
    texts = [
        _mean_std(pd.Series([m]), pd.Series([s]), d)[0]
        for m, s, d in zip(overall["mean"], overall["std"], digits, strict=True)
    ]
    table = overall.assign(text=texts).pivot(index="model", columns="metric", values="text")
    columns = [m for m in (*COUNT_METRICS, *PLOT_METRICS) if m in table.columns]
    return table.reindex(index=overall["model"].unique(), columns=columns).reset_index()


def _mean_std(mean: pd.Series, std: pd.Series, digits: int) -> pd.Series:
    return [
        f"{m:.{digits}f}" + ("" if pd.isna(s) else f" ± {s:.{digits}f}")
        for m, s in zip(mean, std, strict=True)
    ]


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])
