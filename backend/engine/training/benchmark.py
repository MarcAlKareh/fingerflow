"""One command from a PIG folder to a comparable set of numbers.

    python -m engine.training.benchmark --pig-dir path/to/PIG/FingeringFiles \\
        --cache .feature_cache --workers 4

Reports the general, highest and soft match rates for each hand and for both
combined, before and after training, and runs the timing ablation.

**The timing ablation.** PIG's note times come from real performances, so the
dataset carries tempo and articulation that the published HMM baselines
largely ignore. This engine consumes them through the speed factor, the
legato coupling and the Fitts pressure term. To measure what that is worth,
the whole benchmark is run twice: once on the real times, and once on the
same notes re-timed as uniform notes at a fixed tempo with no gaps, which
destroys the timing information while leaving pitch order untouched. The
difference between the two is the value of knowing how fast the music goes.

Published reference points on this benchmark, both hands combined
(Ramoneda et al. 2022, Table 2):

    HMM1               61.77   HMM2  63.78   HMM3  63.63
    ArLSTMThumb-f      65.34   ArGNNThumb-s  66.84
    Human agreement    71.40

Two pianists agree with each other only 71.4% of the time, so that is the
ceiling, not 100%. HMM2 is the closest published relative of this engine, a
second-order model over hand-crafted features, which makes 63 to 64 the
honest target.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..weights import Weights
from .pig import PigSequence, load_pig_dir, split_by_piece
from .train import evaluate, prepare_all, train

FLAT_IOI_SEC = 0.5
REFERENCE = {
    "HMM1": 61.77,
    "HMM2 (closest published relative)": 63.78,
    "HMM3": 63.63,
    "ArLSTMThumb-f": 65.34,
    "ArGNNThumb-s (best published)": 66.84,
    "Human agreement (ceiling)": 71.40,
}


def flatten_timing(sequences: Sequence[PigSequence], ioi: float = FLAT_IOI_SEC) -> List[PigSequence]:
    """Re-time every sequence as uniform, fully legato notes.

    Notes struck together stay together; each distinct onset becomes one step
    of ``ioi`` seconds and every note lasts exactly until the next onset. The
    pitch sequence and the chord structure survive; tempo, rubato and
    articulation do not.
    """
    out: List[PigSequence] = []
    for seq in sequences:
        onsets = sorted({round(float(n["start_time_sec"]), 4) for n in seq.hand_data})
        index = {onset: i for i, onset in enumerate(onsets)}
        notes = []
        for note in seq.hand_data:
            step = index[round(float(note["start_time_sec"]), 4)]
            notes.append({**note, "start_time_sec": step * ioi, "duration_sec": ioi})
        out.append(PigSequence(
            piece=seq.piece, annotator=seq.annotator, hand=seq.hand, path=seq.path,
            hand_data=notes, fingers=list(seq.fingers),
        ))
    return out


def annotators_per_piece(sequences: Sequence[PigSequence]) -> Dict[str, int]:
    """How many distinct annotators fingered each piece."""
    seen: Dict[str, set] = {}
    for seq in sequences:
        seen.setdefault(seq.piece, set()).add(seq.annotator)
    return {piece: len(people) for piece, people in seen.items()}


def describe_annotator_coverage(sequences: Sequence[PigSequence], label: str, stream=sys.stderr) -> float:
    """Print the annotator histogram and return the mean annotators per piece.

    This matters more than it looks. General match rate pools agreement over
    every annotator file, so on a piece fingered by one person a perfect score
    is reachable, while on a piece fingered by eight people who disagree with
    each other no single answer can match them all. A test set of mostly
    single-annotator pieces therefore scores far higher than the published
    one, which was deliberately built from the multi-annotator pieces.
    """
    counts = annotators_per_piece(sequences)
    if not counts:
        return 0.0
    histogram: Dict[int, int] = {}
    for n in counts.values():
        histogram[n] = histogram.get(n, 0) + 1
    summary = ", ".join(f"{pieces} piece(s) with {n}" for n, pieces in sorted(histogram.items()))
    mean = sum(counts.values()) / len(counts)
    print(f"  {label}: {summary}  (mean {mean:.2f} annotators per piece)", file=stream)
    return mean


def split_by_annotator_count(sequences: Sequence[PigSequence], min_annotators: int):
    """Test on the pieces several pianists fingered, train on the rest.

    This reproduces the protocol of the published benchmark, whose test set is
    the composer-specific subsets where every piece carries at least four
    independent fingerings.
    """
    counts = annotators_per_piece(sequences)
    test_pieces = {piece for piece, n in counts.items() if n >= min_annotators}
    train = [s for s in sequences if s.piece not in test_pieces]
    test = [s for s in sequences if s.piece in test_pieces]
    return train, test


@dataclass
class Row:
    name: str
    metrics: Dict[str, Dict[str, float]]


def _evaluate_all(prepared, weights: Weights) -> Dict[str, Dict[str, float]]:
    out = {}
    for hand in ("right", "left"):
        subset = [p for p in prepared if p.hand == hand]
        if subset:
            out[hand] = evaluate(subset, weights)
    out["both"] = evaluate(prepared, weights)
    return out


def _fmt(metrics: Optional[Dict[str, float]]) -> str:
    if not metrics:
        return "      .      .      ."
    return (f"{metrics['general_match_rate'] * 100:7.2f}"
            f"{metrics['highest_match_rate'] * 100:7.2f}"
            f"{metrics['soft_match_rate'] * 100:7.2f}")


def print_table(rows: Sequence[Row], stream=sys.stdout) -> None:
    header = f"{'':38s}" + "".join(f"{h:^21s}" for h in ("right hand", "left hand", "both"))
    sub = f"{'':38s}" + "".join(f"{'gen':>7s}{'high':>7s}{'soft':>7s}" for _ in range(3))
    print(header, file=stream)
    print(sub, file=stream)
    print("-" * len(sub), file=stream)
    for row in rows:
        line = f"{row.name:38s}"
        for hand in ("right", "left", "both"):
            line += _fmt(row.metrics.get(hand))
        print(line, file=stream)
    print("-" * len(sub), file=stream)
    print("\nPublished reference, both hands, general match rate:", file=stream)
    for name, value in REFERENCE.items():
        print(f"  {name:38s}{value:7.2f}", file=stream)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pig-dir", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=None, help="Feature cache directory, strongly recommended")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--l2-to-prior", type=float, default=0.01)
    parser.add_argument("--span-cm", type=float, default=21.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--test-pieces", type=Path, default=None,
                        help="File with one piece id per line, to reproduce a published split")
    parser.add_argument("--test-min-annotators", type=int, default=0,
                        help="Test on pieces fingered by at least this many pianists (use 4 to "
                             "match the published protocol). Overrides --test-fraction.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--save-weights", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    sequences = load_pig_dir(args.pig_dir)
    if args.limit:
        sequences = sequences[: args.limit]
    if not sequences:
        print(f"No *_fingering.txt files found under {args.pig_dir}", file=sys.stderr)
        return 1

    if args.test_pieces and args.test_pieces.is_file():
        wanted = {line.strip() for line in args.test_pieces.read_text().splitlines() if line.strip()}
        train_seqs = [s for s in sequences if s.piece not in wanted]
        test_seqs = [s for s in sequences if s.piece in wanted]
        split_note = f"explicit list of {len(wanted)} pieces"
    elif args.test_min_annotators:
        train_seqs, test_seqs = split_by_annotator_count(sequences, args.test_min_annotators)
        split_note = f"pieces with at least {args.test_min_annotators} annotators (published protocol)"
    else:
        train_seqs, test_seqs = split_by_piece(sequences, args.test_fraction)
        split_note = f"random {args.test_fraction:.0%} of pieces (NOT the published protocol)"

    pieces = len({s.piece for s in sequences})
    notes = sum(len(s.hand_data) for s in sequences)
    print(f"{pieces} pieces, {len(sequences)} hand sequences, {notes} annotated notes", file=sys.stderr)
    print(f"train {len({s.piece for s in train_seqs})} pieces / test {len({s.piece for s in test_seqs})} pieces"
          f"  [test set: {split_note}]", file=sys.stderr)
    describe_annotator_coverage(train_seqs, "train")
    test_mean = describe_annotator_coverage(test_seqs, "test ")
    if test_mean and test_mean < 2.5 and not args.test_min_annotators and not args.test_pieces:
        print("\n  WARNING: the test pieces carry few annotators each, so these numbers are NOT\n"
              "  comparable with the published references below. Those were measured on the\n"
              "  multi-annotator subset, where no single answer can satisfy everyone and the\n"
              "  human ceiling is 71.4%. Re-run with --test-min-annotators 4 to compare fairly.\n",
              file=sys.stderr)

    rows: List[Row] = []
    results: Dict[str, Dict] = {}

    variants = [("timing-aware", sequences, train_seqs, test_seqs)]
    if not args.skip_ablation:
        variants.append(("timing-blind", None, flatten_timing(train_seqs), flatten_timing(test_seqs)))

    for name, _all, tr, te in variants:
        cache = (args.cache / name) if args.cache else None
        print(f"\n[{name}] preparing features ...", file=sys.stderr)
        prepared_train = prepare_all(tr, args.span_cm, cache, args.workers)
        prepared_test = prepare_all(te, args.span_cm, cache, args.workers)

        untrained = _evaluate_all(prepared_test, Weights.default())
        rows.append(Row(f"{name}, hand-tuned weights", untrained))

        print(f"[{name}] training ...", file=sys.stderr)
        learned = train(
            prepared_train, Weights.default(), epochs=args.epochs, lr=args.lr,
            rho=args.rho, l2_to_prior=args.l2_to_prior, prepared_dev=prepared_test,
        )
        trained = _evaluate_all(prepared_test, learned)
        rows.append(Row(f"{name}, learned weights", trained))
        results[name] = {"untrained": untrained, "trained": trained}

        if name == "timing-aware" and args.save_weights:
            learned.save(args.save_weights)
            print(f"[{name}] weights saved to {args.save_weights}", file=sys.stderr)

    print()
    print_table(rows)

    if "timing-blind" in results:
        aware = results["timing-aware"]["trained"]["both"]["general_match_rate"] * 100
        blind = results["timing-blind"]["trained"]["both"]["general_match_rate"] * 100
        delta = aware - blind
        print(f"\nTiming ablation: real performance timing is worth {delta:+.2f} points of "
              f"general match rate ({aware:.2f} against {blind:.2f}).")
        if delta > 0.5:
            print("Timing information helps, which is the biomechanical claim FingerFlow rests on.")
        elif delta < -0.5:
            print("Timing information hurts here. Worth checking the speed and Fitts weights "
                  "before reading anything into it.")
        else:
            print("No meaningful difference. The timing features are not yet earning their place; "
                  "inspect their learned weights before drawing a conclusion.")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nFull metrics written to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
