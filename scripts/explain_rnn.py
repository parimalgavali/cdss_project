from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pandas as pd

from cdss_project.baselines import choose_validation_fold
from cdss_project.explain import (
    event_shapley,
    feature_groups,
    feature_shapley,
    mean_encoded_window,
)
from cdss_project.reporting import plot_importance
from cdss_project.rnn import CELL_TYPES, load_rnn, predict_scores
from cdss_project.windows import load_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TimeSHAP-style event and feature explanations for saved LSTM/GRU models."
    )
    parser.add_argument("--windows", type=Path, default=Path("data/processed/windows_w6.npz"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/rnn"))
    parser.add_argument("--models", nargs="+", choices=CELL_TYPES, default=list(CELL_TYPES))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/explain"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--n-windows", type=int, default=200, help="Test windows to explain.")
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None, help="Only explain these horizons."
    )
    parser.add_argument(
        "--all-stays", action="store_true", help="Sample all test stays, not only sepsis-positive."
    )
    parser.add_argument("--n-permutations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.windows.exists():
        print(f"Windows file not found at {args.windows}.", file=sys.stderr)
        return 2
    dataset = load_windows(args.windows)
    if dataset.fold is None or dataset.split is None:
        print("Windows file has no fold/split arrays; rebuild it.", file=sys.stderr)
        return 2
    split = dataset.split.astype(str)
    test_fold, _ = choose_validation_fold(dataset.fold, split)
    model_dir = args.model_dir / f"test_fold_{test_fold}"
    run_dir = args.output_dir / f"test_fold_{test_fold}"
    if (run_dir / f"{args.models[0]}_feature_importance.csv").exists() and not args.overwrite:
        print(f"Results already exist in {run_dir}. Use --overwrite.", file=sys.stderr)
        return 2

    candidates = split == "test"
    if not args.all_stays:
        candidates &= dataset.y == 1
    if args.horizons:
        candidates &= np.isin(dataset.horizon, args.horizons)
    rng = np.random.default_rng(args.seed)
    pool = np.flatnonzero(candidates)
    chosen = np.sort(rng.choice(pool, size=min(args.n_windows, len(pool)), replace=False))
    if len(chosen) == 0:
        print("No test windows match the selection.", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)

    for cell in dict.fromkeys(args.models):
        if not (model_dir / f"{cell}.pt").exists():
            print(f"No saved {cell} in {model_dir}; run train_rnn.py --save-model.", file=sys.stderr)
            return 2
        model, encoder = load_rnn(model_dir, cell)
        device = next(model.parameters()).device

        def predict(X: np.ndarray, model=model, device=device) -> np.ndarray:
            return predict_scores(model, X.astype(np.float32), device=device)

        baseline = mean_encoded_window(encoder, dataset.X[split == "train"])
        X = encoder.transform(dataset.X[chosen])
        groups = feature_groups(encoder, dataset.feature_columns)
        print(f"Explaining {len(chosen)} {cell} windows ...", flush=True)
        events = event_shapley(predict, X, baseline)
        features = feature_shapley(
            predict, X, baseline, groups, n_permutations=args.n_permutations, seed=args.seed
        )

        meta = pd.DataFrame(
            {
                "stay_id": dataset.stay_id[chosen],
                "horizon": dataset.horizon[chosen],
                "y": dataset.y[chosen],
                "score": predict(X),
                "baseline_score": float(predict(baseline[None])[0]),
            }
        )
        n_steps = X.shape[1]
        event_names = [f"t-{n_steps - 1 - step}" for step in range(n_steps)]
        feature_names = [name for name, _ in groups]
        pd.concat([meta, pd.DataFrame(events, columns=event_names)], axis=1).to_csv(
            run_dir / f"{cell}_event_shap.csv", index=False
        )
        pd.concat([meta, pd.DataFrame(features, columns=feature_names)], axis=1).to_csv(
            run_dir / f"{cell}_feature_shap.csv", index=False
        )
        for kind, values, names in (
            ("event", events, event_names),
            ("feature", features, feature_names),
        ):
            importance = pd.DataFrame(
                {kind: names, "mean_abs_shap": np.abs(values).mean(axis=0), "mean_shap": values.mean(axis=0)}
            )
            if kind == "feature":
                importance = importance.sort_values("mean_abs_shap", ascending=False)
            importance.to_csv(run_dir / f"{cell}_{kind}_importance.csv", index=False)
            plot_importance(
                importance,
                kind,
                args.figures_dir / f"{cell}_{kind}_importance_fold{test_fold}.png",
                title=f"{cell.upper()} {kind} importance (mean |Shapley|)",
            )
        top = pd.read_csv(run_dir / f"{cell}_feature_importance.csv").head(10)
        print(top.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Explanations written to {run_dir}, figures to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
