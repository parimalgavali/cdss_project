"""Run the whole cross-validation pipeline: every test fold, then the summary.

For each test fold k this runs preprocess_data.py -> build_windows.py ->
train_baselines.py -> train_rnn.py (-> explain_rnn.py), then
summarize_results.py once over all folds. The per-fold processed CSV and
windows file are scratch files that are overwritten for every fold, because
imputation statistics are refit on each fold's training stays.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.data_contract import DATASET_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=DATASET_PATH)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--folds", type=int, nargs="+", default=None, help="Test folds to run. Default: all."
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--work-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--min-observed-hours", type=int, default=0)
    parser.add_argument("--baseline-models", nargs="+", default=["lr", "rf"])
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--rnn-models", nargs="+", default=["lstm", "gru"])
    parser.add_argument("--skip-rnn", action="store_true", help="Run the baselines only.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, ...")
    parser.add_argument("--target-sensitivity", type=float, default=None)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--explain", action="store_true", help="Run explain_rnn.py per fold (implies --save-model)."
    )
    parser.add_argument("--n-windows", type=int, default=200, help="Windows to explain per fold.")
    parser.add_argument("--auroc-threshold", type=float, default=0.80)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folds whose result directories already have run_summary.json.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing results.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    args = parser.parse_args(argv)
    if args.folds is None:
        args.folds = list(range(args.n_folds))
    bad = [k for k in args.folds if not 0 <= k < args.n_folds]
    if bad:
        parser.error(f"--folds must be in [0, {args.n_folds - 1}], got {bad}")
    if args.explain and args.skip_rnn:
        parser.error("--explain needs the RNN models; drop --skip-rnn")
    if args.explain:
        args.save_model = True
    return args


def _optional(flag: str, value: object) -> list[str]:
    return [] if value is None else [flag, str(value)]


def fold_commands(args: argparse.Namespace, fold: int) -> list[list[str]]:
    python = sys.executable
    processed = args.work_dir / "cv_preprocessed.csv"
    windows = args.work_dir / "cv_windows_w6.npz"
    baseline_dir = args.output_root / "baselines"
    overwrite = ["--overwrite"] if args.overwrite else []
    commands = [
        [
            python, str(SCRIPTS / "preprocess_data.py"),
            "--csv", str(args.csv), "--output", str(processed),
            "--n-folds", str(args.n_folds), "--test-fold", str(fold),
            "--random-state", str(args.random_state), "--overwrite",
        ],
        [
            python, str(SCRIPTS / "build_windows.py"),
            "--input", str(processed), "--output", str(windows),
            "--min-observed-hours", str(args.min_observed_hours), "--overwrite",
        ],
        [
            python, str(SCRIPTS / "train_baselines.py"),
            "--windows", str(windows), "--output-dir", str(baseline_dir),
            "--models", *args.baseline_models,
            "--rf-n-estimators", str(args.rf_n_estimators),
            *_optional("--target-sensitivity", args.target_sensitivity),
            *overwrite,
        ],
    ]
    if not args.skip_rnn:
        commands.append(
            [
                python, str(SCRIPTS / "train_rnn.py"),
                "--windows", str(windows),
                "--output-dir", str(args.output_root / "rnn"),
                "--baseline-dir", str(baseline_dir),
                "--model-dir", str(args.model_root / "rnn"),
                "--models", *args.rnn_models,
                *_optional("--max-epochs", args.max_epochs),
                *_optional("--batch-size", args.batch_size),
                *_optional("--device", args.device),
                *_optional("--target-sensitivity", args.target_sensitivity),
                *(["--save-model"] if args.save_model else []),
                *overwrite,
            ]
        )
    if args.explain:
        commands.append(
            [
                python, str(SCRIPTS / "explain_rnn.py"),
                "--windows", str(windows),
                "--model-dir", str(args.model_root / "rnn"),
                "--models", *args.rnn_models,
                "--output-dir", str(args.output_root / "explain"),
                "--figures-dir", str(args.figures_dir),
                "--n-windows", str(args.n_windows),
                *overwrite,
            ]
        )
    return commands


def summary_command(args: argparse.Namespace) -> list[str]:
    result_dirs = [args.output_root / "baselines"]
    if not args.skip_rnn:
        result_dirs.append(args.output_root / "rnn")
    return [
        sys.executable, str(SCRIPTS / "summarize_results.py"),
        "--result-dirs", *map(str, result_dirs),
        "--output-dir", str(args.output_root / "summary"),
        "--figures-dir", str(args.figures_dir),
        "--auroc-threshold", str(args.auroc_threshold),
        "--overwrite",
    ]


def fold_is_complete(args: argparse.Namespace, fold: int) -> bool:
    names = ["baselines"] if args.skip_rnn else ["baselines", "rnn"]
    return all(
        (args.output_root / name / f"test_fold_{fold}" / "run_summary.json").exists()
        for name in names
    )


def run(command: list[str], *, dry_run: bool) -> int:
    print("$ " + shlex.join(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not args.csv.exists():
        print(f"Dataset not found at {args.csv}.", file=sys.stderr)
        return 2
    if not args.dry_run:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for fold in args.folds:
        if args.resume and fold_is_complete(args, fold):
            print(f"== test fold {fold}: already complete, skipping", flush=True)
            continue
        print(f"== test fold {fold} ({len(args.folds)} folds requested)", flush=True)
        for command in fold_commands(args, fold):
            code = run(command, dry_run=args.dry_run)
            if code != 0:
                print(f"Step failed with exit code {code} on test fold {fold}.", file=sys.stderr)
                return code
    print("== summary", flush=True)
    code = run(summary_command(args), dry_run=args.dry_run)
    if code == 0 and not args.dry_run:
        minutes = (time.monotonic() - started) / 60
        print(f"Done in {minutes:.1f} min. Report: {args.output_root / 'summary' / 'results.md'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
