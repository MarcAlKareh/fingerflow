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
    parser.add_argument("--show-diff", type=int, nargs="?", const=40, default=0,
                        help="List up to N notes where the model and the annotation differ")
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

    print(f"{'':6s}{'general':>9s}{'highest':>9s}{'soft':>9s}   notes")
    for hand in hands:
        subset = [p for p in prepared if p.hand == hand]
        if subset:
            m = evaluate(subset, weights)
            print(f"{hand:6s}{m['general_match_rate']*100:8.2f}%{m['highest_match_rate']*100:8.2f}%"
                  f"{m['soft_match_rate']*100:8.2f}%   {m['notes']}")
    print(f"{'all':6s}{metrics['general_match_rate']*100:8.2f}%{metrics['highest_match_rate']*100:8.2f}%"
          f"{metrics['soft_match_rate']*100:8.2f}%   {metrics['notes']} "
          f"({metrics['sequences']} sequences, {metrics['pieces']} piece-hands)")

    if args.show_diff:
        _show_differences(prepared, weights, args.show_diff)
    return 0


def _show_differences(prepared, weights: Weights, limit: int) -> None:
    """List the notes the model fingered differently from the annotator."""
    from ..keyboard import midi_to_name
    from ..viterbi import decode
    from .train import predicted_fingers

    print("\nNotes where the model and the annotation differ:")
    shown = 0
    for prep in prepared:
        if shown >= limit:
            break
        prediction = predicted_fingers(prep, decode(prep.tensors, weights).path)
        onset_of = {}
        pitch_of = {}
        for ev in prep.tensors.events:
            for note in ev.sounding:
                onset_of[note.note_id] = note.onset
                pitch_of[note.note_id] = note.midi
        rows = [
            (onset_of.get(nid, 0.0), pitch_of.get(nid, 0), gold, prediction[nid])
            for nid, gold in sorted(prep.gold_fingers.items())
            if nid in prediction and prediction[nid] != gold
        ]
        if not rows:
            continue
        print(f"\n  {prep.piece} ({prep.hand} hand, annotator {prep.annotator}): "
              f"{len(rows)} of {len(prep.gold_fingers)} notes")
        for onset, pitch, gold, model in sorted(rows)[: max(0, limit - shown)]:
            print(f"    {onset:7.2f}s  {midi_to_name(pitch):<5s}  you {gold}   model {model}")
            shown += 1
    if shown == 0:
        print("  none, the model agrees with every annotated note")


if __name__ == "__main__":
    sys.exit(main())
