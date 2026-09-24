"""Finger a whole score, both hands, chords included.

``explain_passage.py`` answers a different question. It takes one hand and
shows the near-misses for a passage you are stuck on, and it prints one note
per event, which is the right thing for a single line and the wrong thing for
a chord. This script is for the other case: you have a real piece, it has two
staves and it has chords, and you want to see the fingering written under the
notes the way an editor would print it.

    python scripts/finger_score.py --musicxml chopin.mxl --span-cm 23

Each measure is printed twice over, note names above and fingers below, right
hand first. A chord appears as one column with its notes joined by ``+``,
lowest note first, and its fingers in the same order.

    python scripts/finger_score.py --musicxml chopin.mxl --bars 1-4 \\
        --span-right-cm 23 --span-left-cm 23.6 --weights backend/weights_learned.json

Tempo comes from the score. Pass ``--bpm`` to finger the piece at the speed
you will actually practise it at, which is the speed that should decide the
fingering. ``--csv`` writes every note with its measure, offset and finger,
which is the form to keep if you want to compare two runs later.

A note the engine could not finger is printed as ``.``. That happens only
when more than five notes are struck at one instant in one hand, where the
outer five are kept and the rest reported as unplayable by that hand alone.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engine.fingering import FingeringResult, assign_fingering  # noqa: E402
from engine.keyboard import midi_to_name  # noqa: E402
from engine.weights import Weights  # noqa: E402
from omr.parser import parse_musicxml  # noqa: E402

HANDS = ("right", "left")


def parse_bars(text: Optional[str]) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    if "-" in text:
        lo, hi = text.split("-", 1)
        return int(lo), int(hi)
    only = int(text)
    return only, only


def measure_of(result: FingeringResult, note_id: int, lookup: Dict[int, dict]) -> int:
    note = lookup.get(note_id)
    return int(note["measure"]) if note and note.get("measure") is not None else 0


def columns(result: FingeringResult, lookup: Dict[int, dict]) -> List[Tuple[int, str, str]]:
    """One (measure, names, fingers) column per onset event."""
    out: List[Tuple[int, str, str]] = []
    for ev in result.events:
        notes = sorted(ev.new_notes, key=lambda n: n.midi)
        names = "+".join(midi_to_name(n.midi) for n in notes)
        fingers = "+".join(
            str(result.fingers[n.note_id]) if result.fingers.get(n.note_id) else "."
            for n in notes
        )
        out.append((measure_of(result, notes[0].note_id, lookup), names, fingers))
    return out


def print_hand(label: str, cols: Sequence[Tuple[int, str, str]], width: int) -> None:
    if not cols:
        return
    cells = [(max(len(n), len(f)) + 2, n, f) for _, n, f in cols]
    line_n, line_f = f"  {label} ", f"  {' ' * len(label)} "
    indent = len(line_n)
    for w, name, finger in cells:
        if len(line_n) + w > width and len(line_n) > indent:
            print(line_n.rstrip())
            print(line_f.rstrip())
            line_n, line_f = " " * indent, " " * indent
        line_n += name.rjust(w)
        line_f += finger.rjust(w)
    print(line_n.rstrip())
    print(line_f.rstrip())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--musicxml", type=Path, required=True)
    parser.add_argument("--hand", default="both", choices=["both", "right", "left"])
    parser.add_argument("--span-cm", type=float, default=21.0,
                        help="Thumb-to-little-finger spread, used for both hands")
    parser.add_argument("--span-right-cm", type=float, default=None)
    parser.add_argument("--span-left-cm", type=float, default=None)
    parser.add_argument("--bpm", type=float, default=None,
                        help="Override the score tempo; finger it at practice speed")
    parser.add_argument("--goal", default=None, choices=["beginner", "expression", "speed"])
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--no-thumb-on-black", action="store_true",
                        help="Keep the thumb off black keys where any fingering allows it")
    parser.add_argument("--bars", default=None, help="Measure range to print, e.g. 1-4")
    parser.add_argument("--width", type=int, default=100, help="Console width to wrap at")
    parser.add_argument("--csv", type=Path, default=None, help="Write every note to this file")
    args = parser.parse_args(argv)

    data = parse_musicxml(args.musicxml, tempo_bpm=args.bpm)
    weights = Weights.load(args.weights).with_preset(args.goal) if args.weights \
        else Weights.default(args.goal)
    spans = {"right": args.span_right_cm or args.span_cm,
             "left": args.span_left_cm or args.span_cm}
    wanted = HANDS if args.hand == "both" else (args.hand,)
    bars = parse_bars(args.bars)

    print(f"{args.musicxml.name}: staff split {data['hand_split']}, "
          f"score tempo {data['score_tempo_bpm']:g} bpm"
          + (f", fingered at {data['tempo_bpm']:g} bpm" if args.bpm else ""))

    results: Dict[str, FingeringResult] = {}
    lookups: Dict[str, Dict[int, dict]] = {}
    for hand in wanted:
        notes = data[f"{hand}_hand"]
        lookups[hand] = {n["note_id"]: n for n in notes}
        if not notes:
            print(f"{hand} hand: no notes on this staff")
            continue
        results[hand] = assign_fingering(
            notes, hand=hand, hand_span_cm=spans[hand], weights=weights,
            avoid_thumb_on_black=args.no_thumb_on_black, explain=False)
        res = results[hand]
        chords = sum(1 for ev in res.events if len(ev.new_notes) > 1)
        print(f"{hand:>5} hand: {len(notes)} notes in {len(res.events)} events "
              f"({chords} chords), span {spans[hand]:g} cm, cost {res.total_cost:.2f}"
              + (f", {len(res.unfingered_note_ids)} unplayable" if res.unfingered_note_ids else ""))

    by_hand = {h: columns(r, lookups[h]) for h, r in results.items()}
    measures = sorted({m for cols in by_hand.values() for m, _, _ in cols})
    if bars:
        measures = [m for m in measures if bars[0] <= m <= bars[1]]

    for measure in measures:
        print(f"\nbar {measure}")
        for hand in wanted:
            cols = [c for c in by_hand.get(hand, []) if c[0] == measure]
            print_hand("R" if hand == "right" else "L", cols, args.width)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["hand", "measure", "offset_ql", "onset_sec", "pitch", "note", "finger"])
            for hand, res in results.items():
                for note in data[f"{hand}_hand"]:
                    writer.writerow([hand, note["measure"], note["offset_ql"],
                                     round(note["start_time_sec"], 4), note["pitch"],
                                     midi_to_name(note["pitch"]),
                                     res.fingers.get(note["note_id"]) or ""])
        print(f"\nWritten to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
