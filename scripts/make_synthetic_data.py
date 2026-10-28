"""Write a synthetic CSV that follows the real data contract (smoke tests only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cdss_project.synthetic import make_synthetic_frame

DEFAULT_OUTPUT_PATH = Path("data/synthetic/synthetic_sepsis.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-stays", type=int, default=2000)
    parser.add_argument("--positive-rate", type=float, default=0.17)
    parser.add_argument("--max-padding-hours", type=int, default=24)
    parser.add_argument("--missing-rate", type=float, default=0.4)
    parser.add_argument(
        "--signal", type=float, default=1.0, help="Drift (in SD) of positive stays at hour 1."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        print(f"{args.output} already exists. Use --overwrite.", file=sys.stderr)
        return 2
    frame = make_synthetic_frame(
        args.n_stays,
        positive_rate=args.positive_rate,
        max_padding_hours=args.max_padding_hours,
        missing_rate=args.missing_rate,
        signal=args.signal,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame):,} rows ({args.n_stays:,} stays) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
