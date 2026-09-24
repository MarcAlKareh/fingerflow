"""Write engine fingerings back onto the original MusicXML for display."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import music21 as m21


def _finger_lookup(notes: Iterable[Dict[str, Any]]) -> Dict[Tuple[float, int], int]:
    lookup: Dict[Tuple[float, int], int] = {}
    for note in notes:
        finger = note.get("finger")
        if finger is None:
            continue
        offset = round(float(note.get("offset_ql", note.get("start_time_sec", 0.0))), 4)
        lookup[(offset, int(note["pitch"]))] = int(finger)
    return lookup


def _offset_ql(element: m21.base.Music21Object, score: m21.stream.Score) -> float:
    try:
        return float(element.getOffsetInHierarchy(score))
    except Exception:  # noqa: BLE001
        return float(element.offset)


def _annotate_element(
    element: m21.note.NotRest,
    score: m21.stream.Score,
    lookup: Dict[Tuple[float, int], int],
    placement: str,
) -> None:
    element.articulations = [
        art
        for art in element.articulations
        if not isinstance(art, m21.articulations.Fingering)
    ]
    if not isinstance(element, (m21.note.Note, m21.chord.Chord)):
        return

    offset = round(_offset_ql(element, score), 4)
    pitches = sorted((p.midi for p in element.pitches))
    for midi in pitches:
        finger = lookup.get((offset, int(midi)))
        if finger is None:
            continue
        mark = m21.articulations.Fingering(finger)
        mark.placement = placement
        element.articulations.append(mark)


def _annotate_part(
    part: m21.stream.Stream,
    score: m21.stream.Score,
    notes: Iterable[Dict[str, Any]],
    placement: str,
) -> None:
    lookup = _finger_lookup(notes)
    if not lookup:
        return
    for element in part.flatten().notes:
        _annotate_element(element, score, lookup, placement)


def compact_piano_layout(score: m21.stream.Score) -> None:
    """Keep both hands on one grand staff and stop OSMD wrapping each bar."""
    parts = list(score.parts)
    if len(parts) >= 2 and not list(score.getElementsByClass(m21.layout.StaffGroup)):
        group = m21.layout.StaffGroup(parts[:2], name="Piano", symbol="brace")
        group.barTogether = True
        score.insert(0, group)

    for part in parts:
        for measure in part.getElementsByClass(m21.stream.Measure):
            for layout in list(measure.getElementsByClass(m21.layout.SystemLayout)):
                measure.remove(layout)
            for layout in list(measure.getElementsByClass(m21.layout.StaffLayout)):
                measure.remove(layout)


def write_fingered_musicxml(
    source: Path,
    music_data: Dict[str, Any],
    dest: Path,
) -> Path:
    """Copy Audiveris MusicXML and stamp engine fingerings onto the notes."""
    source = Path(source)
    dest = Path(dest)
    score = m21.converter.parse(str(source))
    if not isinstance(score, m21.stream.Score):
        wrapper = m21.stream.Score()
        wrapper.insert(0, score)
        score = wrapper

    parts = list(score.parts)
    right = music_data.get("right_hand") or []
    left = music_data.get("left_hand") or []

    if len(parts) >= 2:
        _annotate_part(parts[0], score, right, "above")
        _annotate_part(parts[1], score, left, "below")
    elif len(parts) == 1:
        combined = _finger_lookup(right)
        combined.update(_finger_lookup(left))
        for element in parts[0].flatten().notes:
            offset = round(_offset_ql(element, score), 4)
            pitches = sorted((p.midi for p in element.pitches)) if hasattr(element, "pitches") else []
            placement = "above" if pitches and pitches[-1] >= 60 else "below"
            _annotate_element(element, score, combined, placement)

    compact_piano_layout(score)
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = Path(str(score.write("musicxml", fp=str(dest))))
    if written.resolve() != dest.resolve() and written.is_file():
        dest.write_bytes(written.read_bytes())
    return dest


# Used by the API after fingering.
__all__ = ["compact_piano_layout", "write_fingered_musicxml"]
