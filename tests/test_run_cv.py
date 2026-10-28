import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from cdss_project.synthetic import make_synthetic_frame

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_run_cv():
    spec = importlib.util.spec_from_file_location("run_cv", SCRIPTS / "run_cv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_lists_every_step_per_fold(tmp_path, capsys):
    run_cv = load_run_cv()

    code = run_cv.main(["--csv", str(tmp_path / "x.csv"), "--folds", "0", "3", "--explain", "--dry-run"])

    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("$ ")]
    scripts = [Path(line.split()[2]).name for line in lines]
    per_fold = [
        "preprocess_data.py",
        "build_windows.py",
        "train_baselines.py",
        "train_rnn.py",
        "explain_rnn.py",
    ]
    assert scripts == [*per_fold, *per_fold, "summarize_results.py"]
    assert "--test-fold 3" in lines[5]
    assert "--save-model" in lines[3]


def test_alert_persistence_reaches_both_training_steps(tmp_path, capsys):
    run_cv = load_run_cv()

    run_cv.main(
        ["--csv", str(tmp_path / "x.csv"), "--folds", "0", "--alert-persistence", "3", "--dry-run"]
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("$ ")]
    training = [line for line in lines if "train_baselines.py" in line or "train_rnn.py" in line]
    assert len(training) == 2
    assert all("--alert-persistence 3" in line for line in training)


def test_skip_rnn_drops_rnn_steps(tmp_path, capsys):
    run_cv = load_run_cv()

    run_cv.main(["--csv", str(tmp_path / "x.csv"), "--folds", "1", "--skip-rnn", "--dry-run"])

    out = capsys.readouterr().out
    assert "train_rnn.py" not in out
    assert "outputs/rnn" not in out


def test_full_pipeline_on_synthetic_data(tmp_path):
    csv = tmp_path / "synthetic.csv"
    make_synthetic_frame(150, positive_rate=0.3, signal=2.0, seed=0).to_csv(csv, index=False)
    outputs = tmp_path / "outputs"

    command = [
        sys.executable, str(SCRIPTS / "run_cv.py"),
        "--csv", str(csv),
        "--n-folds", "3", "--folds", "0", "1",
        "--work-dir", str(tmp_path / "work"),
        "--output-root", str(outputs),
        "--model-root", str(tmp_path / "models"),
        "--figures-dir", str(tmp_path / "figures"),
        "--rf-n-estimators", "10",
        "--max-epochs", "1", "--device", "cpu",
        "--explain", "--n-windows", "4",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    summary = pd.read_csv(outputs / "summary" / "metrics_mean_std.csv")
    assert set(summary["model"]) == {"logistic_regression", "random_forest", "lstm", "gru"}
    assert (outputs / "summary" / "results.md").exists()
    for fold in (0, 1):
        assert json.loads((outputs / "rnn" / f"test_fold_{fold}" / "run_summary.json").read_text())
        assert (tmp_path / "models" / "rnn" / f"test_fold_{fold}" / "lstm.pt").exists()
        assert any((outputs / "explain" / f"test_fold_{fold}").glob("*_importance.csv"))

    resumed = subprocess.run(
        [*command, "--resume"], capture_output=True, text=True, check=False
    )
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert resumed.stdout.count("already complete, skipping") == 2
