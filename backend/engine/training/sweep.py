"""Find which individual weights are miscalibrated.

Cross-validation says how good the model is; it does not say which of the 99
weights is wrong. This does, by the crudest reliable method: take one weight,
scale it up and down, and re-measure. A weight whose best multiplier is far
from 1.0 is mis-set, and because every feature has a physical meaning the
result reads as a statement about piano technique rather than a number.

    python -m engine.training.sweep --pig-dir <PIG>/FingeringFiles \\
        --cache .feature_cache/timing-aware --weights weights_learned.json

No retraining happens, so each measurement is one decode over the evaluation
set. Sweeping the default seven features takes a few minutes.

It also prints a finger-usage table: how often the model reaches for each
finger against how often pianists did. That is the diagnostic that motivated
this tool, since a persistent shortfall on one finger points straight at the
weights that discourage it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import SECOND_ORDER_FEATURES, STATE_FEATURES, TRANSITION_FEATURES
from ..viterbi import decode
from ..weights import Weights
from .benchmark import split_by_annotator_count
from .pig import load_pig_dir
from .train import PreparedSequence, evaluate, predicted_fingers, prepare_all

# The weights that bear on which finger gets chosen, which is where the
# cross-validation confusion table pointed.
DEFAULT_FEATURES = [
    "use_f4", "use_f5", "weak_fast", "three_four", "rule_345",
    "four_black", "four_black_three_white",
]
DEFAULT_MULTIPLIERS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
ALL_FEATURES = STATE_FEATURES + TRANSITION_FEATURES + SECOND_ORDER_FEATURES


def scaled(weights: Weights, feature: str, multiplier: float) -> Weights:
    values = weights.to_dict()
    if feature not in values:
        raise KeyError(f"unknown feature {feature!r}")
    values[feature] = values[feature] * multiplier
    return Weights.from_dict(values)


def finger_usage(prepared: Sequence[PreparedSequence], weights: Weights) -> Tuple[Counter, Counter]:
    """How often each finger is chosen by the model, and by the annotators."""
    model_counts: Counter = Counter()
    human_counts: Counter = Counter()
    for prep in prepared:
        prediction = predicted_fingers(prep, decode(prep.tensors, weights).path)
        for note_id, gold in prep.gold_fingers.items():
            chosen = prediction.get(note_id)
            if chosen is None:
                continue
            model_counts[chosen] += 1
            human_counts[gold] += 1
    return model_counts, human_counts


def print_usage(model_counts: Counter, human_counts: Counter) -> None:
    total = sum(human_counts.values()) or 1
    print("\nHow often each finger is used")
    print(f"  {'finger':>8s}{'model':>10s}{'pianists':>11s}{'difference':>13s}{'as % of notes':>15s}")
    for finger in (1, 2, 3, 4, 5):
        diff = model_counts[finger] - human_counts[finger]
        print(f"  {finger:>8d}{model_counts[finger]:>10d}{human_counts[finger]:>11d}"
              f"{diff:>+13d}{diff / total * 100:>14.2f}%")
    print("  A negative difference means the model avoids that finger more than pianists do.")


def sweep(
    prepared: Sequence[PreparedSequence],
    base: Weights,
    features: Sequence[str],
    multipliers: Sequence[float],
) -> Dict[str, List[Tuple[float, float]]]:
    results: Dict[str, List[Tuple[float, float]]] = {}
    baseline = evaluate(prepared, base)["general_match_rate"]
    print(f"baseline general match rate {baseline * 100:.2f}%\n", file=sys.stderr)

    for feature in features:
        current = base.to_dict().get(feature)
        if current is None:
            print(f"  skipping unknown feature {feature}", file=sys.stderr)
            continue
        row: List[Tuple[float, float]] = []
        for multiplier in multipliers:
            score = evaluate(prepared, scaled(base, feature, multiplier))["general_match_rate"]
            row.append((multiplier, score))
            print(f"  {feature:24s} x{multiplier:<5.2f} -> {score * 100:6.2f}%", file=sys.stderr)
        results[feature] = row
    return results, baseline


def print_sweep(results: Dict[str, List[Tuple[float, float]]], baseline: float, multipliers: Sequence[float]) -> None:
    print("\n" + "=" * 78)
    print("Weight sweep: general match rate at each multiplier")
    print("=" * 78)
    header = f"  {'feature':26s}" + "".join(f"{f'x{m:g}':>9s}" for m in multipliers) + f"{'best':>8s}{'gain':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    improvements: List[Tuple[float, str, float]] = []
    for feature, row in results.items():
        line = f"  {feature:26s}"
        for _, score in row:
            line += f"{score * 100:9.2f}"
        best_mult, best_score = max(row, key=lambda item: item[1])
        gain = (best_score - baseline) * 100
        line += f"{best_mult:8.2f}{gain:+8.2f}"
        print(line)
        if gain > 0.01:
            improvements.append((gain, feature, best_mult))

    print(f"\n  baseline (everything at x1) {baseline * 100:.2f}%")
    if improvements:
        improvements.sort(reverse=True)
        print("\n  Worth changing, largest gain first:")
        for gain, feature, multiplier in improvements:
            direction = "lower" if multiplier < 1 else "raise"
            print(f"    {direction} {feature} to x{multiplier:g}  ({gain:+.2f} points)")
        print("\n  These are single-weight changes measured one at a time, so the gains do not")
        print("  simply add up. Apply the largest, re-run, and see what is left.")
    else:
        print("\n  No single weight change helps, so the weights are locally well calibrated.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pig-dir", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="Weights JSON (default: built-in)")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--span-cm", type=float, default=21.0)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES),
                        help="Comma-separated feature names, or 'all'")
    parser.add_argument("--multipliers", default=",".join(str(m) for m in DEFAULT_MULTIPLIERS))
    parser.add_argument("--test-min-annotators", type=int, default=4,
                        help="Evaluate on pieces with at least this many annotators (0 for everything)")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N sequences, for a quick look")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    sequences = load_pig_dir(args.pig_dir)
    if args.test_min_annotators:
        _, sequences = split_by_annotator_count(sequences, args.test_min_annotators)
    if args.limit:
        sequences = sequences[: args.limit]
    if not sequences:
        print("No sequences to evaluate", file=sys.stderr)
        return 1

    print(f"evaluating on {len(sequences)} hand sequences over "
          f"{len({s.piece for s in sequences})} pieces", file=sys.stderr)
    prepared = prepare_all(sequences, args.span_cm, args.cache, args.workers)
    base = Weights.load(args.weights) if args.weights else Weights.default()

    model_counts, human_counts = finger_usage(prepared, base)
    print_usage(model_counts, human_counts)

    features = ALL_FEATURES if args.features.strip() == "all" else [
        f.strip() for f in args.features.split(",") if f.strip()
    ]
    multipliers = [float(m) for m in args.multipliers.split(",")]
    print(f"\nsweeping {len(features)} feature(s) at {len(multipliers)} multipliers, "
          f"{len(features) * len(multipliers)} evaluations\n", file=sys.stderr)

    results, baseline = sweep(prepared, base, features, multipliers)
    print_sweep(results, baseline, multipliers)

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "baseline": baseline,
            "usage": {"model": dict(model_counts), "human": dict(human_counts)},
            "sweep": {k: [{"multiplier": m, "general": s} for m, s in v] for k, v in results.items()},
        }, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
