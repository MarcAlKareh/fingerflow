"""MusicXML to per-hand note events, in real seconds.

Improvements over the first version of this parser:

* Tempo is honoured. Every metronome mark in the score (including
  tempo changes mid-piece) is integrated into a piecewise map from
  quarter-length offsets to seconds. A ``tempo_bpm`` override lets the
  user ask for the fingering at the tempo they actually intend to play
  at, which matters most for beginners practising slowly.
* Tied notes are merged, so a tie is one strike with a long duration
  rather than two strikes.
* Grace notes get a short real duration and are placed just before the
  note they decorate instead of on top of it.
* Every note keeps its measure number and quarter-length offset so the
  frontend can draw fingerings back onto the score.
* Staff-to-hand mapping is explicit and falls back to a middle-C split
  when the OMR output has a single staff.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import music21 as m21

DEFAULT_TEMPO_BPM = 120.0
GRACE_NOTE_SEC = 0.08
LEFT_HAND_ID_OFFSET = 1000


@dataclass(frozen=True)
class TempoSegment:
    start_ql: float
    end_ql: float
    quarter_bpm: float


class TempoMap:
    """Piecewise-constant tempo, integrated to convert offsets to seconds."""

    def __init__(self, segments: Sequence[TempoSegment]):
        if not segments:
            segments = [TempoSegment(0.0, math.inf, DEFAULT_TEMPO_BPM)]
        self.segments = sorted(segments, key=lambda s: s.start_ql)
        self._starts_sec: List[float] = []
        elapsed = 0.0
        for seg in self.segments:
            self._starts_sec.append(elapsed)
            if math.isfinite(seg.end_ql):
                elapsed += (seg.end_ql - seg.start_ql) * 60.0 / seg.quarter_bpm

    @property
    def initial_bpm(self) -> float:
        return self.segments[0].quarter_bpm

    def scaled(self, factor: float) -> "TempoMap":
        return TempoMap([
            TempoSegment(s.start_ql, s.end_ql, s.quarter_bpm * factor) for s in self.segments
        ])

    def seconds(self, offset_ql: float) -> float:
        offset_ql = max(0.0, float(offset_ql))
        for start_sec, seg in zip(self._starts_sec, self.segments):
            if offset_ql < seg.end_ql or not math.isfinite(seg.end_ql):
                return start_sec + (offset_ql - seg.start_ql) * 60.0 / seg.quarter_bpm
        last_sec, last = self._starts_sec[-1], self.segments[-1]
        return last_sec + (offset_ql - last.start_ql) * 60.0 / last.quarter_bpm

    def bpm_at(self, offset_ql: float) -> float:
        for seg in self.segments:
            if seg.start_ql <= offset_ql < seg.end_ql:
                return seg.quarter_bpm
        return self.segments[-1].quarter_bpm


def _mark_quarter_bpm(mark: m21.tempo.MetronomeMark) -> Optional[float]:
    try:
        bpm = mark.getQuarterBPM()
    except Exception:  # noqa: BLE001
        bpm = None
    if bpm is None:
        bpm = mark.number
    if bpm is None or not math.isfinite(float(bpm)) or float(bpm) <= 0:
        return None
    return float(bpm)


def build_tempo_map(score: m21.stream.Score) -> TempoMap:
    """Collect metronome marks from every part into a tempo map."""
    marks: List[Tuple[float, float]] = []
    for mark in score.recurse().getElementsByClass(m21.tempo.MetronomeMark):
        bpm = _mark_quarter_bpm(mark)
        if bpm is None:
            continue
        try:
            offset = float(mark.getOffsetInHierarchy(score))
        except Exception:  # noqa: BLE001
            offset = float(mark.offset)
        marks.append((offset, bpm))

    if not marks:
        return TempoMap([TempoSegment(0.0, math.inf, DEFAULT_TEMPO_BPM)])

    # Keep the last mark at each offset, sorted by offset.
    by_offset: Dict[float, float] = {}
    for offset, bpm in sorted(marks):
        by_offset[round(offset, 6)] = bpm
    offsets = sorted(by_offset)
    if offsets[0] > 0:
        by_offset[0.0] = by_offset[offsets[0]]
        offsets.insert(0, 0.0)
    segments = []
    for i, start in enumerate(offsets):
        end = offsets[i + 1] if i + 1 < len(offsets) else math.inf
        segments.append(TempoSegment(start, end, by_offset[start]))
    return TempoMap(segments)


def _measure_number(element: m21.base.Music21Object) -> Optional[int]:
    measure = element.getContextByClass(m21.stream.Measure)
    if measure is None:
        return None
    return int(measure.number) if measure.number is not None else None


def _fingerings(element: m21.note.NotRest) -> List[Optional[int]]:
    """Finger numbers written on a note or chord, in ascending pitch order.

    MusicXML carries one ``<fingering>`` per note, including inside a chord;
    music21 collects a chord's fingerings onto the chord object in the order
    its notes appear, which publishers and MuseScore write ascending. Where
    the count does not match the number of pitches the fingerings cannot be
    matched to pitches reliably, so they are all dropped: a partially
    annotated passage is fine for training, a wrongly aligned one is not.
    """
    marks: List[Optional[int]] = []
    for art in element.articulations:
        if not isinstance(art, m21.articulations.Fingering):
            continue
        value = art.fingerNumber
        try:
            number = int(str(value).strip().split("-")[0].split("_")[0])
        except (TypeError, ValueError):
            number = None
        marks.append(number if number is not None and 1 <= number <= 5 else None)

    n_pitches = len(element.pitches)
    if not marks:
        return [None] * n_pitches
    if len(marks) != n_pitches:
        return [None] * n_pitches
    return marks


def extract_hand_data(
    part: m21.stream.Stream,
    tempo: TempoMap,
    start_id: int = 0,
    score: Optional[m21.stream.Score] = None,
) -> List[Dict[str, Any]]:
    """Note events of one staff/part in seconds."""
    try:
        merged = part.stripTies(inPlace=False)
    except Exception:  # noqa: BLE001
        merged = part
    flat = merged.flatten()

    events: List[Dict[str, Any]] = []
    pending_grace: List[Dict[str, Any]] = []
    note_id = start_id

    for element in flat.notes:
        if score is not None:
            try:
                offset_ql = float(element.getOffsetInHierarchy(score))
            except Exception:  # noqa: BLE001
                offset_ql = float(element.offset)
        else:
            offset_ql = float(element.offset)
        duration_ql = float(element.duration.quarterLength)
        is_grace = bool(element.duration.isGrace) or (duration_ql == 0.0)
        start_sec = tempo.seconds(offset_ql)
        end_sec = tempo.seconds(offset_ql + duration_ql) if duration_ql > 0 else start_sec
        measure = _measure_number(element)

        if isinstance(element, (m21.chord.Chord, m21.note.Note)):
            order = sorted(range(len(element.pitches)), key=lambda i: element.pitches[i].midi)
            pitches = [element.pitches[i].midi for i in order]
            marks = _fingerings(element)
            fingers = [marks[i] if i < len(marks) else None for i in order]
        else:
            continue

        base = {
            "offset_ql": offset_ql,
            "duration_ql": duration_ql,
            "measure": measure,
            "grace": is_grace,
        }
        if is_grace:
            for midi, finger in zip(pitches, fingers):
                pending_grace.append({
                    "note_id": note_id, "pitch": int(midi), "start_time_sec": start_sec,
                    "duration_sec": GRACE_NOTE_SEC, "finger": finger, **base,
                })
                note_id += 1
            continue

        if pending_grace:
            # Place the grace notes just before this principal note.
            n = len(pending_grace)
            for i, grace in enumerate(pending_grace):
                grace["start_time_sec"] = start_sec - GRACE_NOTE_SEC * (n - i)
                events.append(grace)
            pending_grace = []

        for midi, finger in zip(pitches, fingers):
            events.append({
                "note_id": note_id, "pitch": int(midi), "start_time_sec": start_sec,
                "duration_sec": max(end_sec - start_sec, 0.0), "finger": finger, **base,
            })
            note_id += 1

    # Grace notes with nothing after them: keep them where they are.
    events.extend(pending_grace)
    events.sort(key=lambda e: (e["start_time_sec"], e["pitch"]))
    return events


def _split_by_middle_c(notes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    right = [dict(n) for n in notes if n["pitch"] >= 60]
    left = [dict(n) for n in notes if n["pitch"] < 60]
    for i, n in enumerate(right):
        n["note_id"] = i
    for i, n in enumerate(left):
        n["note_id"] = LEFT_HAND_ID_OFFSET + i
    return right, left


def parse_musicxml(mxl_path: Path, tempo_bpm: Optional[float] = None) -> Dict[str, Any]:
    """Ingest a MusicXML file and return per-hand note events.

    ``tempo_bpm`` overrides the score's initial tempo; later tempo
    changes are scaled by the same factor.
    """
    mxl_path = Path(mxl_path)
    if not mxl_path.exists():
        raise FileNotFoundError(f"Cannot find MusicXML file at {mxl_path}")
    try:
        score = m21.converter.parse(str(mxl_path))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse MusicXML: {e}") from e
    if not isinstance(score, m21.stream.Score):
        wrapper = m21.stream.Score()
        wrapper.insert(0, score)
        score = wrapper

    tempo = build_tempo_map(score)
    score_bpm = tempo.initial_bpm
    if tempo_bpm is not None and tempo_bpm > 0:
        tempo = tempo.scaled(float(tempo_bpm) / score_bpm)

    parts = list(score.parts)
    hand_split = "grand_staff"
    if len(parts) >= 2:
        right = extract_hand_data(parts[0], tempo, start_id=0, score=score)
        left = extract_hand_data(parts[1], tempo, start_id=LEFT_HAND_ID_OFFSET, score=score)
        if len(parts) > 2:
            hand_split = "grand_staff_extra_parts_ignored"
    elif len(parts) == 1:
        everything = extract_hand_data(parts[0], tempo, start_id=0, score=score)
        right, left = _split_by_middle_c(everything)
        hand_split = "middle_c_fallback"
    else:
        right, left = [], []
        hand_split = "no_parts"

    return {
        "tempo_bpm": tempo.initial_bpm,
        "score_tempo_bpm": score_bpm,
        "tempo_segments": [
            {"start_ql": s.start_ql, "end_ql": (None if not math.isfinite(s.end_ql) else s.end_ql), "bpm": s.quarter_bpm}
            for s in tempo.segments
        ],
        "hand_split": hand_split,
        "right_hand": right,
        "left_hand": left,
    }
