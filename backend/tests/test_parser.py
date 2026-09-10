"""Parser tests: tempo map, ties, chords, grace notes, hand split, override."""

import math
from pathlib import Path

import music21 as m21
import pytest

from omr.parser import TempoMap, TempoSegment, build_tempo_map, parse_musicxml


def make_score(path: Path) -> None:
    score = m21.stream.Score()
    rh = m21.stream.Part(id="RH")
    lh = m21.stream.Part(id="LH")

    m1 = m21.stream.Measure(number=1)
    m1.insert(0, m21.tempo.MetronomeMark(number=90))
    m1.append(m21.meter.TimeSignature("4/4"))
    for name in ["C4", "D4", "E4", "F4"]:
        m1.append(m21.note.Note(name, quarterLength=1))

    m2 = m21.stream.Measure(number=2)
    m2.append(m21.chord.Chord(["C4", "E4", "G4"], quarterLength=1))
    m2.append(m21.note.Rest(quarterLength=1))
    tied = m21.note.Note("G4", quarterLength=2)
    tied.tie = m21.tie.Tie("start")
    m2.append(tied)

    m3 = m21.stream.Measure(number=3)
    m3.insert(0, m21.tempo.MetronomeMark(number=120))
    tied_end = m21.note.Note("G4", quarterLength=1)
    tied_end.tie = m21.tie.Tie("stop")
    m3.append(tied_end)
    m3.append(m21.note.Note("G4").getGrace())
    m3.append(m21.note.Note("A4", quarterLength=1))
    m3.append(m21.note.Note("B4", quarterLength=2))
    rh.append([m1, m2, m3])

    for i in range(3):
        m = m21.stream.Measure(number=i + 1)
        if i == 0:
            m.append(m21.meter.TimeSignature("4/4"))
        m.append(m21.note.Note("C3", quarterLength=2))
        m.append(m21.note.Note("G2", quarterLength=2))
        lh.append(m)

    score.insert(0, rh)
    score.insert(0, lh)
    score.write("musicxml", fp=str(path))


@pytest.fixture(scope="module")
def score_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("scores") / "test.musicxml"
    make_score(path)
    return path


def test_tempo_map_integrates_changes():
    tm = TempoMap([TempoSegment(0, 8, 90), TempoSegment(8, math.inf, 120)])
    assert math.isclose(tm.seconds(4), 4 * 60 / 90)
    assert math.isclose(tm.seconds(8), 8 * 60 / 90)
    assert math.isclose(tm.seconds(10), 8 * 60 / 90 + 2 * 60 / 120)
    scaled = tm.scaled(0.5)
    assert math.isclose(scaled.seconds(10), 2 * (8 * 60 / 90 + 2 * 60 / 120))


def test_parse_uses_real_tempo_and_changes(score_path):
    data = parse_musicxml(score_path)
    assert data["hand_split"] == "grand_staff"
    assert data["tempo_bpm"] == 90
    assert [s["bpm"] for s in data["tempo_segments"]] == [90, 120]
    rh = data["right_hand"]
    # Quarter notes at 90 BPM last 2/3 s.
    assert math.isclose(rh[0]["duration_sec"], 60 / 90)
    assert math.isclose(rh[1]["start_time_sec"], 60 / 90)
    # Bar 3 is at 120 BPM: the final half note lasts 1 s.
    last = rh[-1]
    assert last["pitch"] == 71 and math.isclose(last["duration_sec"], 1.0)
    assert last["measure"] == 3


def test_chord_notes_share_onset(score_path):
    rh = parse_musicxml(score_path)["right_hand"]
    chord = [n for n in rh if n["measure"] == 2 and not n["grace"]][:3]
    assert sorted(n["pitch"] for n in chord) == [60, 64, 67]
    assert len({n["start_time_sec"] for n in chord}) == 1


def test_tie_is_merged_into_one_strike(score_path):
    rh = parse_musicxml(score_path)["right_hand"]
    gs = [n for n in rh if n["pitch"] == 67 and not n["grace"] and n["measure"] in (2, 3)]
    # Only the chord's G (measure 2) and the tied G (one event) should remain.
    tied = [n for n in gs if n["duration_ql"] >= 3]
    assert len(tied) == 1
    # 2 quarters at 90 BPM plus 1 quarter at 120 BPM.
    assert math.isclose(tied[0]["duration_sec"], 2 * 60 / 90 + 60 / 120, rel_tol=1e-6)


def test_grace_note_precedes_principal(score_path):
    rh = parse_musicxml(score_path)["right_hand"]
    grace = [n for n in rh if n["grace"]]
    assert len(grace) == 1
    principal = [n for n in rh if n["pitch"] == 69][0]
    assert grace[0]["start_time_sec"] < principal["start_time_sec"]
    assert grace[0]["duration_sec"] > 0


def test_tempo_override_scales_everything(score_path):
    base = parse_musicxml(score_path)
    slow = parse_musicxml(score_path, tempo_bpm=45)
    assert slow["tempo_bpm"] == 45 and slow["score_tempo_bpm"] == 90
    assert [s["bpm"] for s in slow["tempo_segments"]] == [45, 60]
    assert math.isclose(slow["right_hand"][-1]["start_time_sec"], 2 * base["right_hand"][-1]["start_time_sec"])


def test_left_hand_ids_are_offset(score_path):
    data = parse_musicxml(score_path)
    assert all(n["note_id"] >= 1000 for n in data["left_hand"])
    assert all(n["note_id"] < 1000 for n in data["right_hand"])


def test_single_staff_falls_back_to_middle_c_split(tmp_path):
    part = m21.stream.Part()
    m = m21.stream.Measure(number=1)
    m.append(m21.meter.TimeSignature("4/4"))
    for name in ["C3", "G3", "E4", "G4"]:
        m.append(m21.note.Note(name, quarterLength=1))
    part.append(m)
    score = m21.stream.Score()
    score.insert(0, part)
    path = tmp_path / "single.musicxml"
    score.write("musicxml", fp=str(path))
    data = parse_musicxml(path)
    assert data["hand_split"] == "middle_c_fallback"
    assert [n["pitch"] for n in data["right_hand"]] == [64, 67]
    assert [n["pitch"] for n in data["left_hand"]] == [48, 55]
