"""Loader for the PIG piano fingering dataset (Nakamura, Saito, Yoshii).

Dataset page: https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/
Each ``FingeringFiles/<piece>-<annotator>_fingering.txt`` is a tab
separated table with one note per line:

    note_id  onset_sec  offset_sec  pitch  onset_vel  offset_vel  channel  finger

Pitch is spelled (C4, F#3, Bb5). Channel 0 is the right hand and 1 the
left hand; finger numbers are positive for the right hand and negative
for the left. A finger substitution is written like ``1_2`` (strike
with 1, replace by 2 while held); the striking finger is used here.
Lines starting with ``//`` are comments.

Check the README shipped with the dataset in case a later release
changes the column layout; the parser below is deliberately tolerant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..keyboard import name_to_midi

FILENAME_RE = re.compile(r"^(?P<piece>\d+)-(?P<annotator>\d+)_fingering\.txt$", re.IGNORECASE)


@dataclass
class PigSequence:
    piece: str
    annotator: str
    hand: str
    path: Path
    hand_data: List[Dict] = field(default_factory=list)   # engine input format
    fingers: List[int] = field(default_factory=list)      # gold, aligned with hand_data


def parse_finger(token: str) -> Optional[int]:
    """'3' -> 3, '-2' -> 2, '1_2' -> 1, '-1_-2' -> 1; None if unparseable."""
    token = token.strip()
    if not token:
        return None
    first = token.split("_")[0]
    try:
        value = abs(int(first))
    except ValueError:
        return None
    return value if 1 <= value <= 5 else None


def parse_pig_file(path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Return (right_hand_notes, left_hand_notes) with a 'finger' key each."""
    rows: List[Tuple[Dict, bool, int]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        cols = re.split(r"\s+", line)
        if len(cols) < 8:
            continue
        try:
            note_id = int(cols[0])
            onset = float(cols[1])
            offset = float(cols[2])
            midi = name_to_midi(cols[3])
            channel = int(cols[6])
        except ValueError:
            continue
        token = cols[7].strip()
        finger = parse_finger(token)
        if finger is None:
            continue
        note = {
            "note_id": note_id,
            "pitch": midi,
            "start_time_sec": onset,
            "duration_sec": max(offset - onset, 0.0),
            "finger": finger,
        }
        rows.append((note, token.startswith("-"), channel))

    # PIG marks the left hand with negative finger numbers. If a file
    # carries no signs at all, fall back to the channel column.
    use_sign = any(negative for _, negative, _ in rows)
    right: List[Dict] = []
    left: List[Dict] = []
    for note, negative, channel in rows:
        hand_is_left = negative if use_sign else channel == 1
        (left if hand_is_left else right).append(note)
    right.sort(key=lambda n: (n["start_time_sec"], n["pitch"]))
    left.sort(key=lambda n: (n["start_time_sec"], n["pitch"]))
    return right, left


def load_pig_dir(directory: Path, hands: Iterable[str] = ("right", "left")) -> List[PigSequence]:
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*_fingering.txt"))
    sequences: List[PigSequence] = []
    for path in files:
        match = FILENAME_RE.match(path.name)
        piece = match.group("piece") if match else path.stem
        annotator = match.group("annotator") if match else "0"
        right, left = parse_pig_file(path)
        for hand, notes in (("right", right), ("left", left)):
            if hand not in hands or not notes:
                continue
            hand_data = [{k: v for k, v in n.items() if k != "finger"} for n in notes]
            sequences.append(
                PigSequence(
                    piece=piece, annotator=annotator, hand=hand, path=path,
                    hand_data=hand_data, fingers=[n["finger"] for n in notes],
                )
            )
    return sequences


def split_by_piece(sequences: List[PigSequence], test_fraction: float = 0.2, seed: int = 7):
    """Deterministic train/test split with whole pieces kept together."""
    import random

    pieces = sorted({s.piece for s in sequences})
    rng = random.Random(seed)
    rng.shuffle(pieces)
    n_test = int(round(len(pieces) * test_fraction))
    test_pieces = set(pieces[:n_test])
    train = [s for s in sequences if s.piece not in test_pieces]
    test = [s for s in sequences if s.piece in test_pieces]
    return train, test
