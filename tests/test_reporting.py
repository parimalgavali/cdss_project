import pandas as pd
import pytest

from cdss_project.reporting import (
    load_fold_tables,
    minimum_reliable_horizon,
    plot_eda,
    plot_metric_by_horizon,
    summarize_across_folds,
    summarize_eda,
)


def write_fold(root, fold, model, aurocs, eda=None):
    fold_dir = root / f"test_fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"model": model, "horizon": "all", "auroc": sum(aurocs) / len(aurocs)}]
    rows += [{"model": model, "horizon": h, "auroc": a} for h, a in zip((3, 2, 1), aurocs)]
    table = pd.DataFrame(rows)
    for metric in ("auprc", "f1", "sensitivity", "specificity"):
        table[metric] = 0.5
    table.to_csv(fold_dir / "metrics_by_horizon.csv", index=False)
    if eda is not None:
        pd.DataFrame(
            [{"model": model, "reference": "random_forest", "eda_hours": eda}]
        ).to_csv(fold_dir / "eda_vs_baselines.csv", index=False)


@pytest.fixture
def results(tmp_path):
    write_fold(tmp_path / "rnn", 0, "lstm", [0.70, 0.82, 0.90], eda=2.0)
    write_fold(tmp_path / "rnn", 1, "lstm", [0.74, 0.80, 0.92], eda=4.0)
    write_fold(tmp_path / "baselines", 0, "random_forest", [0.60, 0.70, 0.85])
    return tmp_path


def test_summary_mean_std_and_reliable_horizon(results):
    metrics = load_fold_tables([results / "baselines", results / "rnn"], "metrics_by_horizon.csv")
    assert sorted(metrics["fold"].unique()) == [0, 1]

    summary = summarize_across_folds(metrics).set_index(["model", "horizon", "metric"])

    assert summary.loc[("lstm", "2", "auroc"), "mean"] == pytest.approx(0.81)
    assert summary.loc[("lstm", "2", "auroc"), "std"] == pytest.approx(0.0141421, abs=1e-6)
    assert summary.loc[("lstm", "2", "auroc"), "n_folds"] == 2
    assert summary.loc[("random_forest", "1", "auroc"), "n_folds"] == 1

    reliable = minimum_reliable_horizon(summary.reset_index()).set_index("model")
    assert reliable.loc["lstm", "max_reliable_horizon"] == 2
    assert reliable.loc["random_forest", "max_reliable_horizon"] == 1
    strict = minimum_reliable_horizon(summary.reset_index(), threshold=0.95)
    assert strict["max_reliable_horizon"].tolist() == [0, 0]


def test_eda_summary_and_plots(results, tmp_path):
    eda = summarize_eda(load_fold_tables([results / "rnn"], "eda_vs_baselines.csv"))
    assert eda.loc[0, "mean"] == 3.0 and eda.loc[0, "n_folds"] == 2

    metrics = load_fold_tables([results / "rnn"], "metrics_by_horizon.csv")
    summary = summarize_across_folds(metrics)
    figure = plot_metric_by_horizon(summary, "auroc", tmp_path / "fig" / "auroc.png", reference_line=0.8)
    bars = plot_eda(eda, tmp_path / "fig" / "eda.png")
    assert figure.stat().st_size > 0 and bars.stat().st_size > 0


def test_load_fold_tables_empty_when_nothing_found(tmp_path):
    assert load_fold_tables([tmp_path], "metrics_by_horizon.csv").empty


def test_markdown_report_contains_tables(results):
    from cdss_project.reporting import markdown_report

    metrics = load_fold_tables([results / "baselines", results / "rnn"], "metrics_by_horizon.csv")
    summary = summarize_across_folds(metrics)
    eda = summarize_eda(load_fold_tables([results / "rnn"], "eda_vs_baselines.csv"))

    report = markdown_report(summary, minimum_reliable_horizon(summary), eda)

    assert "from 2 test fold(s)" in report
    assert "| lstm | 0.813 ± 0.009 |" in report
    assert "| random_forest | 0.717 |" in report
    assert "| lstm | random_forest | 3.00 ± 1.41 | 2 |" in report


def test_observed_subset_metrics_use_saved_predictions_and_threshold(tmp_path):
    from cdss_project.reporting import load_observed_subset_metrics, markdown_report

    fold_dir = tmp_path / "baselines" / "test_fold_0"
    fold_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"model": "random_forest", "horizon": "all", "threshold": 0.5, "auroc": 0.99}]
    ).to_csv(fold_dir / "metrics_by_horizon.csv", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 1, 2, 2, 3, 3],
            "horizon": [2, 1, 2, 1, 2, 1],
            "y": [1, 1, 0, 0, 1, 0],
            "score": [0.9, 0.2, 0.1, 0.6, 0.8, 0.3],
            "observed_hours": [6, 2, 6, 6, 6, 1],
        }
    ).to_csv(fold_dir / "random_forest_predictions.csv.gz", index=False)

    subset = load_observed_subset_metrics([tmp_path / "baselines"], 6)

    overall = subset[subset["horizon"] == "all"].iloc[0]
    assert overall["windows"] == 4 and overall["positives"] == 2
    assert overall["threshold"] == 0.5
    assert overall["sensitivity"] == pytest.approx(1.0)
    assert overall["specificity"] == pytest.approx(0.5)
    summary = summarize_across_folds(subset)
    report = markdown_report(
        summary, minimum_reliable_horizon(summary), observed_summary=summary, min_observed_hours=6
    )
    assert "## Windows with at least 6 recorded hours" in report
