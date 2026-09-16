"""Choose which passages are worth a human judgement.

Annotating whole pieces at random is a poor use of a pianist's time: most
notes have one obvious finger and teach the model nothing. Two queries find
the passages that actually carry information.

**Uncertainty.** :func:`engine.viterbi.event_margins` gives, for each event,
the extra cost of fingering it any other way. Where that margin is small the
model is close to indifferent between two fingerings, so which one a pianist
actually chooses is genuinely informative. These passages need no existing
annotation, so they work on any score.

**Confident disagreement.** Where several annotators agree with each other
and the model disagrees with all of them, the model is not merely expressing
a preference, it is wrong about something. These are the highest-value
passages of all, but they need a piece that already carries two or more
independent fingerings, so in practice they come from PIG.

Both produce :class:`Passage` objects that
:func:`engine.training.musescore.export_review` can write out for MuseScore.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..events import build_events, new_note_fingers, notes_from_hand_data
from ..features import FeatureTensors
from ..hand import hand_profile
from ..keyboard import midi_to_name
from ..viterbi import decode, event_margins
from ..weights import Weights

DEFAULT_WINDOW = 10
DEFAULT_SPAN_CM = 21.0


@dataclass
class Passage:
    source: str
    hand: str
    reason: str
    score: float
    notes: List[Dict] = field(default_factory=list)
    model_fingers: List[Optional[int]] = field(default_factory=list)
    gold_fingers: List[Optional[int]] = field(default_factory=list)
    detail: str = ""

    @property
    def label(self) -> str:
        return f"{self.source} {self.hand[0].upper()}H"

    def describe(self) -> str:
        pitches = " ".join(midi_to_name(int(n["pitch"])) for n in self.notes)
        model = "".join(str(f) if f else "?" for f in self.model_fingers)
        line = f"[{self.reason}] {self.label} score={self.score:.2f}\n  {pitches}\n  model {model}"
        if any(self.gold_fingers):
            gold = "".join(str(f) if f else "?" for f in self.gold_fingers)
            line += f"\n  human {gold}"
        if self.detail:
            line += f"\n  {self.detail}"
        return line


def _decode_hand(hand_data: Sequence[Dict], hand: str, weights: Weights, span_cm: float):
    profile = hand_profile(hand, span_cm)
    events = build_events(notes_from_hand_data(hand_data), profile)
    tensors = FeatureTensors(events, profile)
    result = decode(tensors, weights)
    return events, tensors, result


def _window_notes(events, path, lo: int, hi: int) -> Tuple[List[Dict], List[Optional[int]]]:
    notes: List[Dict] = []
    fingers: List[Optional[int]] = []
    for k in range(lo, hi):
        ev = events[k]
        assign = ev.assignments[path[k]]
        for note, finger in new_note_fingers(ev, assign):
            notes.append({
                "note_id": note.note_id,
                "pitch": note.midi,
                "start_time_sec": note.onset,
                "duration_sec": note.duration,
            })
            fingers.append(int(finger))
    return notes, fingers


def _greedy_windows(scores: np.ndarray, window: int, top_k: int, ascending: bool) -> List[Tuple[int, int, float]]:
    """Non-overlapping windows centred on the most interesting events."""
    order = np.argsort(scores if ascending else -scores)
    taken: List[Tuple[int, int, float]] = []
    claimed = np.zeros(len(scores), dtype=bool)
    half = max(1, window // 2)
    for centre in order:
        value = float(scores[centre])
        if not np.isfinite(value):
            continue
        if not ascending and value <= 0:
            break
        lo = max(0, int(centre) - half)
        hi = min(len(scores), lo + window)
        lo = max(0, hi - window)
        if claimed[lo:hi].any():
            continue
        claimed[lo:hi] = True
        taken.append((lo, hi, value))
        if len(taken) >= top_k:
            break
    return taken


def uncertain_passages(
    hand_data: Sequence[Dict],
    hand: str,
    source: str = "score",
    weights: Optional[Weights] = None,
    span_cm: float = DEFAULT_SPAN_CM,
    window: int = DEFAULT_WINDOW,
    top_k: int = 5,
    goal: Optional[str] = None,
) -> List[Passage]:
    """Passages where a different fingering would cost the model almost nothing."""
    if not hand_data:
        return []
    w = (weights or Weights.default()).with_preset(goal) if weights else Weights.default(goal)
    events, tensors, result = _decode_hand(hand_data, hand, w, span_cm)
    if len(events) < 2:
        return []
    margins = event_margins(tensors, w, result.path)

    passages: List[Passage] = []
    for lo, hi, value in _greedy_windows(margins, window, top_k, ascending=True):
        notes, fingers = _window_notes(events, result.path, lo, hi)
        if not notes:
            continue
        tightest = int(lo + int(np.argmin(margins[lo:hi])))
        passages.append(Passage(
            source=source, hand=hand, reason="uncertain", score=value,
            notes=notes, model_fingers=fingers,
            gold_fingers=[None] * len(notes),
            detail=(f"cheapest alternative costs only {value:.2f} more, "
                    f"at event {tightest - lo + 1} of this window"),
        ))
    return passages


def _consensus(annotations: Sequence[Dict[Tuple[float, int], int]], min_annotators: int = 2):
    """Notes where at least ``min_annotators`` annotators all chose the same finger."""
    votes: Dict[Tuple[float, int], List[int]] = defaultdict(list)
    for table in annotations:
        for key, finger in table.items():
            votes[key].append(finger)
    agreed: Dict[Tuple[float, int], int] = {}
    for key, fingers in votes.items():
        if len(fingers) >= min_annotators and len(set(fingers)) == 1:
            agreed[key] = fingers[0]
    return agreed


def disagreement_passages(
    sequences: Sequence,
    weights: Optional[Weights] = None,
    span_cm: float = DEFAULT_SPAN_CM,
    window: int = DEFAULT_WINDOW,
    top_k_per_piece: int = 2,
    min_annotators: int = 2,
) -> List[Passage]:
    """Passages where the annotators agree with each other but not with the model.

    ``sequences`` is a list of :class:`engine.training.pig.PigSequence`.
    """
    w = weights or Weights.default()
    grouped: Dict[Tuple[str, str], List] = defaultdict(list)
    for seq in sequences:
        grouped[(seq.piece, seq.hand)].append(seq)

    passages: List[Passage] = []
    for (piece, hand), group in sorted(grouped.items()):
        if len(group) < min_annotators:
            continue
        tables = []
        for seq in group:
            tables.append({
                (round(float(n["start_time_sec"]), 4), int(n["pitch"])): int(f)
                for n, f in zip(seq.hand_data, seq.fingers)
            })
        agreed = _consensus(tables, min_annotators)
        if not agreed:
            continue

        reference = group[0]
        events, tensors, result = _decode_hand(reference.hand_data, hand, w, span_cm)
        if len(events) < 2:
            continue

        mismatch = np.zeros(len(events))
        for k, ev in enumerate(events):
            assign = ev.assignments[result.path[k]]
            for note, finger in new_note_fingers(ev, assign):
                key = (round(float(note.onset), 4), int(note.midi))
                gold = agreed.get(key)
                if gold is not None and gold != int(finger):
                    mismatch[k] += 1.0

        if not mismatch.any():
            continue
        density = np.convolve(mismatch, np.ones(window), mode="same")
        for lo, hi, value in _greedy_windows(density, window, top_k_per_piece, ascending=False):
            notes, fingers = _window_notes(events, result.path, lo, hi)
            if not notes:
                continue
            gold_fingers = [
                agreed.get((round(float(n["start_time_sec"]), 4), int(n["pitch"])))
                for n in notes
            ]
            differing = sum(
                1 for g, f in zip(gold_fingers, fingers) if g is not None and g != f
            )
            passages.append(Passage(
                source=piece, hand=hand, reason="disagreement", score=float(value),
                notes=notes, model_fingers=fingers, gold_fingers=gold_fingers,
                detail=(f"{differing} note(s) where {len(group)} annotators agree "
                        f"with each other and the model differs"),
            ))
    passages.sort(key=lambda p: -p.score)
    return passages


def to_review_passages(passages: Sequence[Passage]):
    """Group per-hand passages into the two-staff form the exporter wants."""
    from .musescore import ReviewPassage

    out = []
    for index, passage in enumerate(passages, start=1):
        right = passage.hand == "right"
        out.append(ReviewPassage(
            label=f"{index}. {passage.label}",
            right_notes=passage.notes if right else [],
            right_fingers=passage.model_fingers if right else [],
            left_notes=[] if right else passage.notes,
            left_fingers=[] if right else passage.model_fingers,
            note=passage.detail,
        ))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    unc = sub.add_parser("uncertain", help="Flag passages where the model is nearly indifferent")
    unc.add_argument("source", type=Path, nargs="+", help="MusicXML file(s)")
    unc.add_argument("--top-k", type=int, default=5, help="Passages per hand per file")

    dis = sub.add_parser("disagree", help="Flag passages where annotators agree and the model does not")
    dis.add_argument("--pig-dir", type=Path, required=True)
    dis.add_argument("--top-k", type=int, default=2, help="Passages per piece per hand")
    dis.add_argument("--limit", type=int, default=0)

    for p in (unc, dis):
        p.add_argument("--weights", type=Path, default=None)
        p.add_argument("--span-cm", type=float, default=DEFAULT_SPAN_CM)
        p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
        p.add_argument("--out", type=Path, default=None, help="Write a MusicXML review file here")
    args = parser.parse_args(argv)

    weights = Weights.load(args.weights) if args.weights else Weights.default()
    passages: List[Passage] = []

    if args.mode == "uncertain":
        from omr.parser import parse_musicxml

        for source in args.source:
            data = parse_musicxml(source)
            for hand in ("right", "left"):
                notes = [{k: v for k, v in n.items() if k != "finger"} for n in data.get(f"{hand}_hand", [])]
                passages += uncertain_passages(
                    notes, hand, source=source.stem, weights=weights,
                    span_cm=args.span_cm, window=args.window, top_k=args.top_k,
                )
    else:
        from .pig import load_pig_dir

        sequences = load_pig_dir(args.pig_dir)
        if args.limit:
            sequences = sequences[: args.limit]
        passages = disagreement_passages(
            sequences, weights=weights, span_cm=args.span_cm,
            window=args.window, top_k_per_piece=args.top_k,
        )

    if not passages:
        print("No passages flagged.", file=sys.stderr)
        return 0

    for passage in passages:
        print(passage.describe())
        print()

    if args.out:
        from .musescore import export_review

        export_review(to_review_passages(passages), args.out)
        print(f"{len(passages)} passage(s) written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
