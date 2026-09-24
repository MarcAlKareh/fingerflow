"""Read noteheads and page coordinates from an Audiveris .omr book."""

from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from .parser import DEFAULT_TEMPO_BPM, LEFT_HAND_ID_OFFSET

# Audiveris staff pitch 0 is the middle line.
# Treble middle line is B4; bass middle line is D3.
_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
_TREBLE_CENTER = (6, 4)  # B4
_BASS_CENTER = (1, 3)  # D3
_SHARP_COUNT_LETTERS = {
    1: {"F"},
    2: {"F", "C"},
    3: {"F", "C", "G"},
    4: {"F", "C", "G", "D"},
    5: {"F", "C", "G", "D", "A"},
    6: {"F", "C", "G", "D", "A", "E"},
    7: {"F", "C", "G", "D", "A", "E", "B"},
}


def _letter_midi(letter: str, octave: int, alter: int) -> int:
    semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    return 12 * (octave + 1) + semitone + alter


def _staff_pos_to_letter(center: tuple[int, int], position: int) -> tuple[str, int]:
    letter_index, octave = center
    index = letter_index - position
    octave += index // 7
    letter_index = index % 7
    if letter_index < 0:
        letter_index += 7
        octave -= 1
    return _LETTERS[letter_index], octave


def _parse_fraction(value: str | None, default: float = 0.0) -> float:
    if not value:
        return default
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den)
    return float(value)


def parse_omr_heads(
    omr_path: Path,
    image_path: Path | None = None,
    *,
    tempo_bpm: Optional[float] = None,
) -> dict[str, Any] | None:
    """Build engine note events with page coordinates from a .omr book."""
    omr_path = Path(omr_path)
    if not omr_path.is_file():
        return None
    try:
        with zipfile.ZipFile(omr_path) as archive:
            names = [name for name in archive.namelist() if name.endswith("sheet#1.xml")]
            if not names:
                names = [name for name in archive.namelist() if name.endswith(".xml") and "sheet" in name]
            if not names:
                return None
            root = ET.fromstring(archive.read(names[0]))
    except (zipfile.BadZipFile, ET.ParseError, OSError):
        return None

    picture = root.find("picture")
    width = float(picture.get("width", 0) or 0) if picture is not None else 0.0
    height = float(picture.get("height", 0) or 0) if picture is not None else 0.0
    if image_path and Path(image_path).is_file():
        try:
            import cv2
            import numpy as np

            data = np.fromfile(str(image_path), dtype=np.uint8)
            gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                height, width = float(gray.shape[0]), float(gray.shape[1])
        except Exception:  # noqa: BLE001
            pass
    if width <= 0 or height <= 0:
        return None

    stacks: list[dict[str, Any]] = []
    for stack in root.iter("stack"):
        sid = int(stack.get("id", "0") or 0)
        left = float(stack.get("left", 0) or 0)
        right = float(stack.get("right", 0) or 0)
        slots = []
        for slot in stack.findall("slot"):
            slots.append(
                {
                    "x": left + float(slot.get("x-offset", 0) or 0),
                    "time": _parse_fraction(slot.get("time-offset"), 0.0),
                }
            )
        stacks.append({"id": sid, "left": left, "right": right, "slots": slots})
    stacks.sort(key=lambda item: item["left"])
    if not stacks:
        return None

    fifths = len(list(root.iter("key-alter")))
    if fifths > 7:
        fifths = 4
    sharp_letters = _SHARP_COUNT_LETTERS.get(fifths, set())

    clefs: list[tuple[float, int, str]] = []
    for clef in root.iter("clef"):
        kind = (clef.get("kind") or "").upper()
        if kind not in {"TREBLE", "BASS"}:
            continue
        staff = int(clef.get("staff", "1") or 1)
        bounds = clef.find("bounds")
        x = float(bounds.get("x", 0) or 0) if bounds is not None else 0.0
        clefs.append((x, staff, kind))
    if not any(staff == 1 and kind == "TREBLE" for _, staff, kind in clefs):
        clefs.append((0.0, 1, "TREBLE"))
    if not any(staff == 2 and kind == "BASS" for _, staff, kind in clefs):
        clefs.append((0.0, 2, "BASS"))

    def clef_for(staff: int, x: float) -> str:
        chosen = "TREBLE" if staff == 1 else "BASS"
        for cx, cst, kind in sorted(clefs):
            if cst == staff and cx <= x:
                chosen = kind
        return chosen

    def stack_for(x: float) -> dict[str, Any] | None:
        for stack in stacks:
            if stack["left"] <= x < stack["right"]:
                return stack
        return stacks[-1] if stacks else None

    def slot_time(stack: dict[str, Any], x: float) -> float:
        slots = stack["slots"]
        if not slots:
            return 0.0
        return min(slots, key=lambda slot: abs(slot["x"] - x))["time"]

    bpm = float(tempo_bpm) if tempo_bpm and tempo_bpm > 0 else DEFAULT_TEMPO_BPM
    ql_to_sec = 60.0 / bpm

    right: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    for head in root.iter("head"):
        pitch_attr = head.get("pitch")
        if pitch_attr is None:
            continue
        staff = int(head.get("staff", "1") or 1)
        bounds = head.find("bounds")
        if bounds is None:
            continue
        x = float(bounds.get("x", 0) or 0) + float(bounds.get("w", 0) or 0) / 2.0
        y = float(bounds.get("y", 0) or 0) + float(bounds.get("h", 0) or 0) / 2.0
        position = int(float(pitch_attr))
        kind = clef_for(staff, x)
        center = _TREBLE_CENTER if kind == "TREBLE" else _BASS_CENTER
        letter, octave = _staff_pos_to_letter(center, position)
        alter = 1 if letter in sharp_letters else 0
        midi = _letter_midi(letter, octave, alter)
        stack = stack_for(x)
        measure = int(stack["id"]) if stack else None
        time_ql = slot_time(stack, x) if stack else 0.0
        measure_start = 0.0
        if stack:
            index = next((i for i, item in enumerate(stacks) if item["id"] == stack["id"]), 0)
            # Incomplete first bar still starts at 0; later bars are 4/4.
            measure_start = 0.0 if index == 0 else sum(4.0 for _ in stacks[:index])
        offset_ql = measure_start + time_ql
        event = {
            "pitch": midi,
            "start_time_sec": offset_ql * ql_to_sec,
            "duration_sec": 0.25 * ql_to_sec,
            "offset_ql": round(offset_ql, 4),
            "duration_ql": 0.25,
            "measure": measure,
            "grace": False,
            "x": round(x / width, 5),
            "y": round(y / height, 5),
        }
        if staff == 1:
            right.append(event)
        else:
            left.append(event)

    def finish(events: list[dict[str, Any]], start_id: int) -> list[dict[str, Any]]:
        events.sort(key=lambda item: (item["offset_ql"], item["pitch"]))
        for index, event in enumerate(events):
            event["note_id"] = start_id + index
        return events

    right = finish(right, 0)
    left = finish(left, LEFT_HAND_ID_OFFSET)
    if not right and not left:
        return None

    return {
        "tempo_bpm": bpm,
        "score_tempo_bpm": bpm,
        "tempo_segments": [{"start_ql": 0.0, "end_ql": None, "bpm": bpm}],
        "hand_split": "omr_heads",
        "right_hand": right,
        "left_hand": left,
        "image": {"width": width, "height": height, "source": "processed"},
    }


def find_book_omr(output_dir: Path) -> Path | None:
    """The full-page .omr, not per-measure recovery books."""
    output_dir = Path(output_dir)
    candidates = [
        path
        for path in output_dir.glob("*.omr")
        if "measure_omr" not in path.parts
    ]
    return sorted(candidates)[0] if candidates else None
