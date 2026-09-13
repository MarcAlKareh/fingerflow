"""Bridge between MuseScore and the training pipeline.

Kevin annotates in MuseScore, which stores a finger number on each note and
exports it in MusicXML as ``<notations><technical><fingering>``. This module
carries those annotations into PIG format so they sit alongside the public
dataset, and carries model output the other way so a flagged passage can be
opened in MuseScore, tried at the piano, corrected and exported back.

The loop is:

    1. ``export_review`` writes flagged passages with the model's fingering.
    2. Open in MuseScore, play them, fix what is wrong, export MusicXML.
    3. ``convert`` turns the corrected file into a PIG fingering file.
    4. ``engine.training.train`` consumes it with the rest of the data.

PIG's own format is one note per line:

    id  onset  offset  pitch  onset_vel  offset_vel  channel  finger

with channel 0 and a positive finger for the right hand, channel 1 and a
negative finger for the left.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import music21 as m21

from ..keyboard import midi_to_name
from omr.parser import DEFAULT_TEMPO_BPM, parse_musicxml

PIG_HEADER = "//Version: PianoFingering_v170101"
# Rhythmic grid used when rebuilding notation from note times, fine enough
# for triplets and sixteenths without producing unreadable tuplet chains.
REVIEW_GRID = Fraction(1, 12)
REVIEW_MAX_QL = Fraction(8)


@dataclass
class FingeredHand:
    hand: str
    notes: List[Dict]           # engine input dicts, no 'finger' key
    fingers: List[Optional[int]]

    @property
    def annotated_count(self) -> int:
        return sum(1 for f in self.fingers if f is not None)

    @property
    def coverage(self) -> float:
        return self.annotated_count / len(self.fingers) if self.fingers else 0.0


def load_fingered_musicxml(path: Path, tempo_bpm: Optional[float] = None) -> Dict[str, FingeredHand]:
    """Read a MusicXML file and split out the notes and their written fingerings."""
    data = parse_musicxml(Path(path), tempo_bpm=tempo_bpm)
    out: Dict[str, FingeredHand] = {}
    for hand in ("right", "left"):
        raw = data.get(f"{hand}_hand") or []
        notes = [{k: v for k, v in n.items() if k != "finger"} for n in raw]
        fingers = [n.get("finger") for n in raw]
        out[hand] = FingeredHand(hand=hand, notes=notes, fingers=fingers)
    out["_meta"] = data  # type: ignore[assignment]
    return out


def pig_lines(hands: Sequence[FingeredHand], comment: str = "") -> List[str]:
    """Render annotated notes as PIG-format lines, skipping unannotated ones."""
    rows: List[Tuple[float, float, int, int, int]] = []
    for fh in hands:
        channel = 0 if fh.hand == "right" else 1
        sign = 1 if fh.hand == "right" else -1
        for note, finger in zip(fh.notes, fh.fingers):
            if finger is None:
                continue
            onset = float(note["start_time_sec"])
            offset = onset + float(note.get("duration_sec", 0.0) or 0.0)
            rows.append((onset, offset, int(note["pitch"]), channel, sign * int(finger)))
    rows.sort(key=lambda r: (r[0], r[2]))

    lines = [PIG_HEADER]
    if comment:
        lines.append(f"//{comment}")
    for idx, (onset, offset, pitch, channel, finger) in enumerate(rows):
        lines.append(
            f"{idx}\t{onset:.6f}\t{offset:.6f}\t{midi_to_name(pitch)}\t80\t80\t{channel}\t{finger}"
        )
    return lines


def convert(
    source: Path,
    out_path: Path,
    tempo_bpm: Optional[float] = None,
    comment: str = "",
) -> Dict[str, float]:
    """Convert one fingered MusicXML file to a PIG fingering file."""
    hands = load_fingered_musicxml(Path(source), tempo_bpm=tempo_bpm)
    right, left = hands["right"], hands["left"]
    lines = pig_lines([right, left], comment=comment or f"from {Path(source).name}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "right_notes": len(right.notes),
        "right_annotated": right.annotated_count,
        "right_coverage": right.coverage,
        "left_notes": len(left.notes),
        "left_annotated": left.annotated_count,
        "left_coverage": left.coverage,
        "written": right.annotated_count + left.annotated_count,
    }


# ---------------------------------------------------------------------------
# The other direction: model output back into MuseScore
# ---------------------------------------------------------------------------

def _quantise(seconds: float, bpm: float) -> Fraction:
    """Seconds to a readable quarter-length on the review grid."""
    ql = Fraction(seconds * bpm / 60.0).limit_denominator(48)
    steps = max(1, round(ql / REVIEW_GRID))
    value = REVIEW_GRID * steps
    return min(value, REVIEW_MAX_QL)


def _build_part(
    notes: Sequence[Dict],
    fingers: Sequence[Optional[int]],
    bpm: float,
    clef: m21.clef.Clef,
) -> m21.stream.Part:
    part = m21.stream.Part()
    part.append(clef)
    if not notes:
        part.append(m21.note.Rest(quarterLength=4))
        return part

    grouped: Dict[float, List[Tuple[Dict, Optional[int]]]] = {}
    for note, finger in zip(notes, fingers):
        grouped.setdefault(round(float(note["start_time_sec"]), 4), []).append((note, finger))

    onsets = sorted(grouped)
    origin = onsets[0]
    cursor = Fraction(0)
    for onset in onsets:
        items = sorted(grouped[onset], key=lambda it: it[0]["pitch"])
        target = _quantise(onset - origin, bpm)
        if target > cursor:
            part.append(m21.note.Rest(quarterLength=float(target - cursor)))
            cursor = target
        duration = max(
            _quantise(max(float(it[0].get("duration_sec", 0.0) or 0.0) for it in items), bpm),
            REVIEW_GRID,
        )
        if len(items) == 1:
            element = m21.note.Note(int(items[0][0]["pitch"]))
        else:
            element = m21.chord.Chord([int(it[0]["pitch"]) for it in items])
        element.quarterLength = float(duration)
        for _, finger in items:
            if finger is not None:
                element.articulations.append(m21.articulations.Fingering(int(finger)))
        if len(items) > 1 and len(element.articulations) not in (0, len(items)):
            # Partial chord fingering cannot be aligned to pitches on re-import.
            element.articulations = []
        part.append(element)
        cursor += duration
    return part


@dataclass
class ReviewPassage:
    label: str
    right_notes: List[Dict]
    right_fingers: List[Optional[int]]
    left_notes: List[Dict]
    left_fingers: List[Optional[int]]
    note: str = ""


def export_review(passages: Sequence[ReviewPassage], out_path: Path, bpm: float = DEFAULT_TEMPO_BPM) -> Path:
    """Write flagged passages, with the model's fingering, as one MusicXML file.

    Each passage becomes its own short section with a rehearsal mark, so the
    file can be opened in MuseScore and read through at the piano. Rhythms are
    rebuilt from note times on a 1/12-quarter grid, so they are legible rather
    than a faithful transcription of the original engraving.
    """
    score = m21.stream.Score()
    rh = m21.stream.Part(id="RH")
    lh = m21.stream.Part(id="LH")
    rh.append(m21.clef.TrebleClef())
    lh.append(m21.clef.BassClef())
    rh.append(m21.tempo.MetronomeMark(number=bpm))

    for passage in passages:
        for part, notes, fingers, clef in (
            (rh, passage.right_notes, passage.right_fingers, m21.clef.TrebleClef()),
            (lh, passage.left_notes, passage.left_fingers, m21.clef.BassClef()),
        ):
            section = _build_part(notes, fingers, bpm, clef)
            first = True
            for element in section.notesAndRests:
                if first and part is rh:
                    mark = m21.expressions.RehearsalMark(passage.label)
                    part.append(mark)
                    first = False
                part.append(element)
            part.append(m21.bar.Barline("double"))

    score.insert(0, rh)
    score.insert(0, lh)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out_path))
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert fingered MusicXML exported from MuseScore into PIG format.",
    )
    parser.add_argument("source", type=Path, nargs="+", help="MusicXML file(s) exported from MuseScore")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the PIG fingering files")
    parser.add_argument("--annotator", default="kh", help="Annotator tag used in the output filename")
    parser.add_argument("--tempo-bpm", type=float, default=None, help="Override the score tempo")
    args = parser.parse_args(argv)

    total = 0
    for index, source in enumerate(args.source, start=1):
        piece = source.stem.replace(" ", "_")
        out_path = args.out_dir / f"{piece}-{args.annotator}_fingering.txt"
        stats = convert(source, out_path, tempo_bpm=args.tempo_bpm)
        total += int(stats["right_annotated"] + stats["left_annotated"])
        print(
            f"{source.name}: RH {stats['right_annotated']}/{stats['right_notes']} "
            f"({stats['right_coverage']:.0%}), LH {stats['left_annotated']}/{stats['left_notes']} "
            f"({stats['left_coverage']:.0%}) -> {out_path}",
            file=sys.stderr,
        )
        if stats["right_notes"] and stats["right_coverage"] == 0 and stats["left_coverage"] == 0:
            print(
                "  no fingerings found. In MuseScore the numbers must be Fingering marks "
                "(Palettes > Fingering, or added with the Add > Text menu), not free-floating "
                "staff text, or they are not written to MusicXML.",
                file=sys.stderr,
            )
    print(f"{total} annotated notes written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
