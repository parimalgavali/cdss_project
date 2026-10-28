"""Cross-fold summaries, minimum reliable horizon, and result plots."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PLOT_METRICS: tuple[str, ...] = ("auroc", "auprc", "f1", "sensitivity", "specificity")
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
