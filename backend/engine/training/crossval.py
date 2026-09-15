"""Cross-validation and per-piece diagnostics.

A single train/test split gives one number over one arbitrary 20% of the
repertoire. Five-fold cross-validation tests every piece exactly once, which
is both a more robust estimate and, more usefully, a per-piece score showing
*where* the model is weak rather than only how weak it is on average.

    python -m engine.training.crossval --pig-dir <PIG>/FingeringFiles \\
        --cache .feature_cache --workers 4 --folds 5

Three things come out.

**Fold spread.** The same measurement five times over. If the folds disagree
wildly, one split was never going to be trustworthy.

**Scores split by how many pianists fingered each piece.** General match rate
is not comparable across those groups: a piece fingered by one person allows
a near-perfect score, while a piece fingered by six who disagree does not.
Mixing them produces a number that means nothing, which is the trap the first
benchmark run fell into. They are reported separately here.

**Per-piece scores and a confusion table.** Which pieces the model handles
worst, and across the whole dataset, which finger it reaches for when a human
reached for another. The confusion table is the actionable part: a systematic
"model plays 4 where pianists play 3" points at a specific weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..viterbi import decode
from ..weights import Weights
from .benchmark import annotators_per_piece
from .pig import PigSequence, load_pig_dir
from .train import PreparedSequence, evaluate, predicted_fingers, prepare_all, train

BUCKETS = [(1, 1, "1 annotator"), (2, 3, "2-3 annotators"), (4, 99, "4+ annotators")]


def stratified_folds(sequences: Sequence[PigSequence], folds: int, seed: int = 7) -> List[set]:
    """Split pieces into folds, keeping the annotator-count mix even.

    Without stratification one fold can land most of the multi-annotator
    pieces and score far lower than the others for a reason that has nothing
    to do with the model.
    """
    import random

    counts = annotators_per_piece(sequences)
    by_bucket: Dict[str, List[str]] = defaultdict(list)
    for piece, n in counts.items():
        label = next(name for lo, hi, name in BUCKETS if lo <= n <= hi)
        by_bucket[label].append(piece)

    rng = random.Random(seed)
    assignment: List[set] = [set() for _ in range(folds)]
    for label in sorted(by_bucket):
        pieces = sorted(by_bucket[label])
        rng.shuffle(pieces)
        for i, piece in enumerate(pieces):
            assignment[i % folds].add(piece)
    return assignment


def bucket_of(n_annotators: int) -> str:
    return next(name for lo, hi, name in BUCKETS if lo <= n_annotators <= hi)


@dataclass
class PieceResult:
    piece: str
    annotators: int
    notes: int
    general: float
    highest: float
    soft: float
    confusions: Counter = field(default_factory=Counter)


def score_pieces(prepared: Sequence[PreparedSequence], weights: Weights) -> List[PieceResult]:
    """Per-piece metrics and per-piece model-versus-human finger confusions."""
    by_piece: Dict[str, List[PreparedSequence]] = defaultdict(list)
    for prep in prepared:
        by_piece[prep.piece].append(prep)

    results: List[PieceResult] = []
    for piece, preps in by_piece.items():
        metrics = evaluate(preps, weights)
        confusions: Counter = Counter()
        notes = 0
        for prep in preps:
            prediction = predicted_fingers(prep, decode(prep.tensors, weights).path)
            for note_id, gold in prep.gold_fingers.items():
                model = prediction.get(note_id)
                if model is None:
                    continue
                notes += 1
                if model != gold:
                    confusions[(gold, model)] += 1
        results.append(PieceResult(
            piece=piece,
            annotators=len({p.annotator for p in preps}),
            notes=notes,
            general=metrics["general_match_rate"],
            highest=metrics["highest_match_rate"],
            soft=metrics["soft_match_rate"],
            confusions=confusions,
        ))
    return results


def aggregate(results: Sequence[PieceResult]) -> Dict[str, Dict[str, float]]:
    """Note-weighted metrics overall and per annotator-count bucket."""
    out: Dict[str, Dict[str, float]] = {}
    groups: Dict[str, List[PieceResult]] = defaultdict(list)
    for r in results:
        groups[bucket_of(r.annotators)].append(r)
        groups["all"].append(r)
    for label, items in groups.items():
        notes = sum(r.notes for r in items)
        if not notes:
            continue
        out[label] = {
            "general": sum(r.general * r.notes for r in items) / notes,
            "highest": sum(r.highest * r.notes for r in items) / notes,
            "soft": sum(r.soft * r.notes for r in items) / notes,
            "notes": notes,
            "pieces": len(items),
        }
    return out


def run_crossval(
    sequences: Sequence[PigSequence],
    span_cm: float,
    cache: Optional[Path],
    workers: int,
    folds: int,
    epochs: int,
    lr: float,
    rho: float,
    l2_to_prior: float,
    seed: int = 7,
) -> Tuple[List[PieceResult], List[Dict[str, float]]]:
    prepared_all = prepare_all(list(sequences), span_cm, cache, workers)
    by_piece: Dict[str, List[PreparedSequence]] = defaultdict(list)
    for prep in prepared_all:
        by_piece[prep.piece].append(prep)

    fold_pieces = stratified_folds(sequences, folds, seed)
    all_results: List[PieceResult] = []
    fold_summaries: List[Dict[str, float]] = []

    for index, held_out in enumerate(fold_pieces, start=1):
        train_preps = [p for p in prepared_all if p.piece not in held_out]
        test_preps = [p for p in prepared_all if p.piece in held_out]
        if not test_preps:
            continue
        print(f"\nfold {index}/{folds}: train {len(train_preps)} sequences, "
              f"test {len(test_preps)} sequences over {len(held_out)} pieces", file=sys.stderr)
        weights = train(train_preps, Weights.default(), epochs=epochs, lr=lr, rho=rho,
                        l2_to_prior=l2_to_prior, verbose=False)
        results = score_pieces(test_preps, weights)
        all_results.extend(results)
        summary = aggregate(results).get("all", {})
        summary["fold"] = index
        fold_summaries.append(summary)
        print(f"  general {summary.get('general', 0) * 100:.2f}  "
              f"highest {summary.get('highest', 0) * 100:.2f}  "
              f"soft {summary.get('soft', 0) * 100:.2f}", file=sys.stderr)
    return all_results, fold_summaries


def print_report(results: Sequence[PieceResult], folds: Sequence[Dict[str, float]], worst: int = 15) -> None:
    print("\n" + "=" * 72)
    print("Fold spread")
    print("=" * 72)
    generals = [f["general"] * 100 for f in folds if "general" in f]
    for f in folds:
        print(f"  fold {int(f['fold'])}: general {f['general']*100:6.2f}  "
              f"highest {f['highest']*100:6.2f}  soft {f['soft']*100:6.2f}  "
              f"({f['pieces']} pieces, {f['notes']} notes)")
    if generals:
        print(f"\n  mean {np.mean(generals):.2f}, standard deviation {np.std(generals):.2f}, "
              f"range {min(generals):.2f} to {max(generals):.2f}")

    print("\n" + "=" * 72)
    print("By how many pianists fingered the piece")
    print("=" * 72)
    print("  These are not comparable with each other. One annotator allows a near-perfect")
    print("  score; six disagreeing annotators do not. Only the 4+ row is comparable with")
    print("  the published benchmark.\n")
    summary = aggregate(results)
    print(f"  {'group':18s}{'general':>9s}{'highest':>9s}{'soft':>9s}{'pieces':>9s}{'notes':>9s}")
    for _, _, label in BUCKETS:
        if label in summary:
            m = summary[label]
            print(f"  {label:18s}{m['general']*100:8.2f}%{m['highest']*100:8.2f}%"
                  f"{m['soft']*100:8.2f}%{m['pieces']:9d}{m['notes']:9d}")
    if "all" in summary:
        m = summary["all"]
        print(f"  {'all':18s}{m['general']*100:8.2f}%{m['highest']*100:8.2f}%"
              f"{m['soft']*100:8.2f}%{m['pieces']:9d}{m['notes']:9d}")

    print("\n" + "=" * 72)
    print(f"The {worst} pieces the model handles worst")
    print("=" * 72)
    ranked = sorted((r for r in results if r.notes >= 50), key=lambda r: r.general)
    print(f"  {'piece':10s}{'annot':>7s}{'notes':>8s}{'general':>9s}{'soft':>8s}   most common error")
    for r in ranked[:worst]:
        top = r.confusions.most_common(1)
        detail = f"human {top[0][0][0]} -> model {top[0][0][1]}, {top[0][1]}x" if top else ""
        print(f"  {r.piece:10s}{r.annotators:7d}{r.notes:8d}{r.general*100:8.2f}%"
              f"{r.soft*100:7.2f}%   {detail}")

    print("\n" + "=" * 72)
    print("Where the model disagrees with pianists, across everything")
    print("=" * 72)
    total: Counter = Counter()
    for r in results:
        total.update(r.confusions)
    grand = sum(total.values())
    print(f"  {grand} disagreements in total. The most systematic:\n")
    print(f"  {'human plays':>12s}{'model plays':>13s}{'count':>9s}{'share':>8s}")
    for (gold, model), count in total.most_common(12):
        print(f"  {gold:>12d}{model:>13d}{count:>9d}{count / grand * 100:7.1f}%")

    # Which fingers the model over- and under-uses overall.
    over: Counter = Counter()
    under: Counter = Counter()
    for (gold, model), count in total.items():
        over[model] += count
        under[gold] += count
    print(f"\n  {'finger':>8s}{'model reaches for it':>22s}{'pianists wanted it':>20s}{'net':>8s}")
    for finger in (1, 2, 3, 4, 5):
        net = over[finger] - under[finger]
        print(f"  {finger:>8d}{over[finger]:>22d}{under[finger]:>20d}{net:>+8d}")
    print("\n  A large positive net means the model over-uses that finger and a weight for it")
    print("  is probably too low; a large negative net means the opposite.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pig-dir", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--l2-to-prior", type=float, default=0.01)
    parser.add_argument("--span-cm", type=float, default=21.0)
    parser.add_argument("--worst", type=int, default=15, help="How many weak pieces to list")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    sequences = load_pig_dir(args.pig_dir)
    if args.limit:
        sequences = sequences[: args.limit]
    if not sequences:
        print(f"No fingering files found under {args.pig_dir}", file=sys.stderr)
        return 1

    pieces = len({s.piece for s in sequences})
    print(f"{pieces} pieces, {len(sequences)} hand sequences, "
          f"{sum(len(s.hand_data) for s in sequences)} notes", file=sys.stderr)
    print(f"{args.folds}-fold cross-validation, every piece tested exactly once", file=sys.stderr)

    results, folds = run_crossval(
        sequences, args.span_cm, args.cache, args.workers, args.folds,
        args.epochs, args.lr, args.rho, args.l2_to_prior,
    )
    print_report(results, folds, worst=args.worst)

    if args.json_out:
        payload = {
            "folds": folds,
            "pieces": [
                {"piece": r.piece, "annotators": r.annotators, "notes": r.notes,
                 "general": r.general, "highest": r.highest, "soft": r.soft,
                 "confusions": {f"{g}->{m}": c for (g, m), c in r.confusions.items()}}
                for r in results
            ],
            "summary": aggregate(results),
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFull per-piece results written to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
