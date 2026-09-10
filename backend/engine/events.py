"""Note events for one hand: onsets, chords, held notes and timing.

The fingering engine does not work on a flat list of notes. It works on
*events*: sets of notes struck at the same instant. Each event also
knows which earlier notes are still sounding (held) at its onset, so a
finger that is holding a key cannot be re-used, and it knows how much
time separates it from the previous event, which is what makes tempo
matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

from .hand import FINGERS, HandProfile

ONSET_TOLERANCE_SEC = 1e-4
HELD_TOLERANCE_SEC = 1e-4
MAX_SOUNDING_NOTES = 5

# Legato coupling: two consecutive notes are fully coupled when the
# first is released within LEGATO_GAP_SEC of the second's onset. Beyond
# that the coupling decays with time constant GAP_TAU_SEC, because the
# hand can relocate during a rest.
LEGATO_GAP_SEC = 0.03
GAP_TAU_SEC = 0.25


@dataclass(frozen=True)
class Note:
    note_id: int
    midi: int
    onset: float
    duration: float

    @property
    def release(self) -> float:
        return self.onset + max(self.duration, 0.0)


Assignment = Tuple[int, ...]


@dataclass
class Event:
    index: int
    onset: float
    new_notes: Tuple[Note, ...]
    held_notes: Tuple[Note, ...]
    sounding: Tuple[Note, ...]
    ioi: float
    gap: float
    coupling: float
    assignments: List[Assignment] = field(default_factory=list)
    # For each new note (by position in ``new_notes``): index into the
    # previous event's ``new_notes`` of the melodically nearest note.
    pairing: Tuple[Optional[int], ...] = ()
    # For each held note (by position in ``sounding``): its position in
    # the previous event's ``sounding``.
    held_map: Tuple[Tuple[int, int], ...] = ()
    dropped: Tuple[Note, ...] = ()

    @property
    def is_chord(self) -> bool:
        return len(self.new_notes) > 1

    def sounding_index(self, note_id: int) -> int:
        for idx, note in enumerate(self.sounding):
            if note.note_id == note_id:
                return idx
        raise KeyError(note_id)


def coupling_from_gap(gap: float) -> float:
    if gap <= LEGATO_GAP_SEC:
        return 1.0
    return math.exp(-(gap - LEGATO_GAP_SEC) / GAP_TAU_SEC)


def notes_from_hand_data(hand_data: Sequence[Dict]) -> List[Note]:
    notes: List[Note] = []
    for idx, item in enumerate(hand_data):
        notes.append(
            Note(
                note_id=int(item.get("note_id", idx)),
                midi=int(item["pitch"]),
                onset=float(item["start_time_sec"]),
                duration=float(item.get("duration_sec", 0.0)),
            )
        )
    return notes


def enumerate_assignments(count: int, hand: HandProfile) -> List[Assignment]:
    """All monotone finger assignments for ``count`` simultaneously sounding notes.

    Notes are ordered by ascending pitch. The right hand uses ascending
    finger numbers, the left hand descending ones.
    """
    if count == 0:
        return [()]
    if count > len(FINGERS):
        return []
    result: List[Assignment] = []
    for combo in combinations(FINGERS, count):
        result.append(tuple(combo) if hand.is_right else tuple(reversed(combo)))
    return result


def build_events(notes: Sequence[Note], hand: HandProfile) -> List[Event]:
    """Group notes into onset events with held-note tracking."""
    ordered = sorted(notes, key=lambda n: (n.onset, n.midi))
    groups: List[List[Note]] = []
    for note in ordered:
        if groups and abs(note.onset - groups[-1][0].onset) <= ONSET_TOLERANCE_SEC:
            groups[-1].append(note)
        else:
            groups.append([note])

    events: List[Event] = []
    prev: Optional[Event] = None
    for index, group in enumerate(groups):
        onset = group[0].onset
        new_notes = tuple(sorted(group, key=lambda n: n.midi))
        dropped: Tuple[Note, ...] = ()
        if len(new_notes) > MAX_SOUNDING_NOTES:
            # More simultaneous notes than fingers: keep the outer notes
            # the hand can reach and report the rest as unfingerable.
            if hand.is_right:
                dropped = new_notes[: len(new_notes) - MAX_SOUNDING_NOTES]
                new_notes = new_notes[len(new_notes) - MAX_SOUNDING_NOTES :]
            else:
                dropped = new_notes[MAX_SOUNDING_NOTES:]
                new_notes = new_notes[:MAX_SOUNDING_NOTES]

        held: List[Note] = []
        if prev is not None:
            for note in prev.sounding:
                if note.release > onset + HELD_TOLERANCE_SEC:
                    held.append(note)
        # Never more than five sounding notes: release the oldest held
        # notes first (they are the ones most likely to be pedalled).
        room = MAX_SOUNDING_NOTES - len(new_notes)
        if len(held) > room:
            held.sort(key=lambda n: n.onset)
            held = held[len(held) - room :] if room > 0 else []
        held_notes = tuple(sorted(held, key=lambda n: n.midi))

        sounding = tuple(sorted(held_notes + new_notes, key=lambda n: n.midi))

        if prev is None:
            ioi = math.inf
            gap = math.inf
            coupling = 0.0
        else:
            ioi = onset - prev.onset
            last_release = max(n.release for n in prev.new_notes)
            gap = onset - last_release
            coupling = coupling_from_gap(gap)

        pairing: List[Optional[int]] = []
        if prev is not None:
            for note in new_notes:
                best = None
                best_key = None
                for pidx, pnote in enumerate(prev.new_notes):
                    key = (abs(pnote.midi - note.midi), -pnote.midi if hand.is_right else pnote.midi)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = pidx
                pairing.append(best)
        else:
            pairing = [None] * len(new_notes)

        held_map: List[Tuple[int, int]] = []
        if prev is not None:
            for sidx, note in enumerate(sounding):
                if any(h.note_id == note.note_id for h in held_notes):
                    held_map.append((sidx, prev.sounding_index(note.note_id)))

        event = Event(
            index=index,
            onset=onset,
            new_notes=new_notes,
            held_notes=held_notes,
            sounding=sounding,
            ioi=ioi,
            gap=gap,
            coupling=coupling,
            assignments=enumerate_assignments(len(sounding), hand),
            pairing=tuple(pairing),
            held_map=tuple(held_map),
            dropped=dropped,
        )
        events.append(event)
        prev = event
    return events


def transition_feasible(prev: Event, prev_assign: Assignment, cur: Event, cur_assign: Assignment) -> bool:
    """Held notes must keep the finger that struck them."""
    for cur_idx, prev_idx in cur.held_map:
        if cur_assign[cur_idx] != prev_assign[prev_idx]:
            return False
    return True


def new_note_fingers(event: Event, assign: Assignment) -> List[Tuple[Note, int]]:
    """(note, finger) for the notes struck at this event."""
    result = []
    new_ids = {n.note_id for n in event.new_notes}
    for note, finger in zip(event.sounding, assign):
        if note.note_id in new_ids:
            result.append((note, finger))
    return result
