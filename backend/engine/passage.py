"""A passage of music, and the machinery for fingering parts of it under constraints.

Several tools need the same three things: turn a handful of notes into
feature tensors, ask what the engine would do, and ask what it would do if
a particular group of notes were fingered a particular way. That last one
is the interesting one. Constraining a segment and re-optimising everything
around it gives a *complete* alternative fingering rather than a note-level
substitution, so two candidates can be compared by a single number.

A passage can be described entirely by data (pitches, onsets, durations,
hand and span), which is what :class:`PassageSpec` is for. That matters
because a recorded human judgement about a passage should stay meaningful
after the MusicXML file it came from has been moved, re-exported or lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .events import Event, build_events, new_note_fingers, notes_from_hand_data
from .features import FeatureTensors
from .hand import HandProfile, hand_profile
from .keyboard import midi_to_name, name_to_midi
from .viterbi import decode, path_cost
from .weights import Weights

# Finite, so that a banned assignment stays distinguishable from a
# physically infeasible one when reading a decode back.
BAN = 1.0e6

# A chord event has no single "the finger", so it cannot be constrained by
# finger number. Those events are marked with this sentinel.
CHORD = 0


@dataclass
class PassageSpec:
    """Everything needed to rebuild a passage, with no reference to any file."""

    hand: str
    span_cm: float
    notes: List[Dict[str, float]]
    label: str = ""

    def to_dict(self) -> dict:
        return {"hand": self.hand, "span_cm": self.span_cm,
                "notes": self.notes, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict) -> "PassageSpec":
        return cls(hand=data["hand"], span_cm=float(data["span_cm"]),
                   notes=[dict(n) for n in data["notes"]], label=data.get("label", ""))

    @classmethod
    def from_note_names(cls, text: str, hand: str = "right", span_cm: float = 21.0,
                        bpm: float = 60.0, note_value: int = 8, legato: bool = True,
                        label: str = "") -> "PassageSpec":
        pitches = [name_to_midi(t) for t in text.replace(",", " ").split() if t]
        if len(pitches) < 2:
            raise ValueError("A passage needs at least two notes")
        ioi = (4.0 / note_value) * 60.0 / bpm
        notes = [{"pitch": int(p), "start_time_sec": i * ioi,
                  "duration_sec": ioi if legato else ioi * 0.5}
                 for i, p in enumerate(pitches)]
        return cls(hand=hand, span_cm=span_cm, notes=notes,
                   label=label or f"{len(pitches)} notes at {bpm:g} bpm")

    @classmethod
    def from_musicxml(cls, path: Path, hand: str = "right", span_cm: float = 21.0,
                      tempo_bpm: Optional[float] = None, label: str = "") -> "PassageSpec":
        from omr.parser import parse_musicxml  # lazy: engine does not depend on omr

        parsed = parse_musicxml(path, tempo_bpm=tempo_bpm)
        staff = parsed[f"{hand}_hand"]
        if not staff:
            other = "left" if hand == "right" else "right"
            raise ValueError(f"No notes on the {hand} staff; the file has "
                             f"{len(parsed[f'{other}_hand'])} on the {other}")
        notes = [{"pitch": int(n["pitch"]),
                  "start_time_sec": round(float(n["start_time_sec"]), 6),
                  "duration_sec": round(float(n["duration_sec"]), 6)} for n in staff]
        return cls(hand=hand, span_cm=span_cm, notes=notes,
                   label=label or Path(path).name)

    def build(self) -> "Passage":
        return Passage(self)


class Passage:
    """A built passage: events, feature tensors and constrained decoding."""

    def __init__(self, spec: PassageSpec):
        self.spec = spec
        self.hand = spec.hand
        self.span_cm = spec.span_cm
        self.data = [{"note_id": i, **note} for i, note in enumerate(spec.notes)]
        self.profile: HandProfile = hand_profile(spec.hand, spec.span_cm)
        self.events: List[Event] = build_events(notes_from_hand_data(self.data), self.profile)
        self.tensors = FeatureTensors(self.events, self.profile)
        self._options = _finger_options(self.events)

    def __len__(self) -> int:
        return len(self.events)

    @property
    def names(self) -> List[str]:
        return [midi_to_name(ev.new_notes[0].midi) if len(ev.new_notes) == 1 else "chord"
                for ev in self.events]

    def options(self, index: int) -> List[int]:
        """Fingers available for the note struck at event ``index`` (0-based)."""
        return sorted(k for k in self._options[index] if k != CHORD)

    def fingers(self, path: Sequence[int]) -> List[int]:
        """The finger on each struck note along a path; ``CHORD`` where ambiguous."""
        out: List[int] = []
        for ev, a in zip(self.events, path):
            pairs = new_note_fingers(ev, ev.assignments[a])
            out.append(pairs[0][1] if len(pairs) == 1 else CHORD)
        return out

    def best(self, weights: Weights) -> Tuple[List[int], float]:
        result = decode(self.tensors, weights)
        return result.path, path_cost(self.tensors, weights, result.path)

    def constrained(self, weights: Weights, fixed: Dict[int, int]) -> Optional[Tuple[List[int], float]]:
        """Cheapest complete path in which event ``k`` uses finger ``fixed[k]``.

        Returns ``None`` when no such path exists, rather than a path that
        quietly violates the constraint.
        """
        augment: List[Optional[np.ndarray]] = [None] * len(self.events)
        for k, finger in fixed.items():
            allowed = self._options[k].get(finger, [])
            if not allowed:
                return None
            penalty = np.full(len(self.events[k].assignments), -BAN)
            penalty[allowed] = 0.0
            augment[k] = penalty  # subtracted from the state cost, so -BAN adds BAN
        path = decode(self.tensors, weights, augment=augment).path
        for k, finger in fixed.items():
            if path[k] not in self._options[k].get(finger, []):
                return None
        return path, path_cost(self.tensors, weights, path)

    def path_features(self, path: Sequence[int]):
        return self.tensors.path_features(list(path))


def _finger_options(events: Sequence[Event]) -> List[Dict[int, List[int]]]:
    """Per event, which assignment indices give each finger to the struck note."""
    table: List[Dict[int, List[int]]] = []
    for ev in events:
        by_finger: Dict[int, List[int]] = {}
        for index, assign in enumerate(ev.assignments):
            pairs = new_note_fingers(ev, assign)
            key = pairs[0][1] if len(pairs) == 1 else CHORD
            by_finger.setdefault(key, []).append(index)
        table.append(by_finger)
    return table


def parse_segment(text: str, n_events: int, max_len: int = 8) -> Tuple[int, int]:
    """``"9-13"`` or ``"9"`` to a 0-based inclusive range."""
    lo_text, _, hi_text = text.partition("-")
    hi_text = hi_text or lo_text
    lo, hi = int(lo_text) - 1, int(hi_text) - 1
    if not (0 <= lo <= hi < n_events):
        raise ValueError(f"segment must lie inside 1-{n_events}")
    if hi - lo + 1 > max_len:
        raise ValueError(f"a segment of more than {max_len} notes has too many fingerings to enumerate")
    return lo, hi


def parse_fingering(text: str) -> List[int]:
    """``"3-4-3-2-1"`` or ``"34321"`` to ``[3, 4, 3, 2, 1]``."""
    cleaned = text.strip().replace(" ", "")
    parts = cleaned.split("-") if "-" in cleaned else list(cleaned)
    fingers = [int(p) for p in parts if p]
    if not fingers or any(f < 1 or f > 5 for f in fingers):
        raise ValueError(f"{text!r} is not a fingering: use digits 1-5, e.g. 3-4-3-2-1")
    return fingers


def format_fingering(fingers: Sequence[int]) -> str:
    return "-".join(str(f) for f in fingers)
