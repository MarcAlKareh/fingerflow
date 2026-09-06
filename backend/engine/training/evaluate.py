"""Match rate of a weight file against human fingerings.

    python -m engine.training.evaluate --pig-dir PIG/FingeringFiles \
        --weights weights_learned.json [--all]

Reports the general match rate (over every annotator file) and the
highest match rate (best annotator per piece), as in Nakamura, Saito
and Yoshii, "Statistical learning and estimation of piano fingering"
(Information Sciences, 2020). With ``--all`` the whole dataset is used
instead of the held-out test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from ..weights import Weights
from .pig import load_pig_dir, split_by_piece
from .train import evaluate, prepare_all


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pig-dir", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="Weights JSON (default: built-in)")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--span-cm", type=float, default=21.0)
    parser.add_argument("--hands", default="both", choices=["both", "right", "left"])
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--all", action="store_true", help="Evaluate on every sequence, not just the test split")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    hands = ("right", "left") if args.hands == "both" else (args.hands,)
    sequences = load_pig_dir(args.pig_dir, hands=hands)
    if args.limit:
        sequences = sequences[: args.limit]
    if not args.all:
        _, sequences = split_by_piece(sequences, args.test_fraction)
    if not sequences:
        print("No sequences to evaluate", file=sys.stderr)
        return 1
    prepared = prepare_all(sequences, args.span_cm, args.cache, args.workers)
    weights = Weights.load(args.weights) if args.weights else Weights.default()
    metrics = evaluate(prepared, weights)
    for hand in hands:
        subset = [p for p in prepared if p.hand == hand]
        if subset:
            m = evaluate(subset, weights)
            print(f"{hand:5s}: general {m['general_match_rate']:.3f}  highest {m['highest_match_rate']:.3f}  ({m['notes']} notes)")
    print(f"all  : general {metrics['general_match_rate']:.3f}  highest {metrics['highest_match_rate']:.3f}  ({metrics['notes']} notes, {metrics['sequences']} sequences)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
