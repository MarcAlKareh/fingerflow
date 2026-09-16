"""Tests for the annotation pipeline: margins, passage selection, MuseScore bridge."""

import itertools
import math
import random
from pathlib import Path

import music21 as m21
import numpy as np
import pytest

from engine.events import build_events, notes_from_hand_data
from engine.features import FeatureTensors
from engine.hand import hand_profile
from engine.training.benchmark import flatten_timing
from engine.training.musescore import (
    ReviewPassage, convert, export_review, load_fingered_musicxml, pig_lines,
)
from engine.training.pig import PigSequence, load_pig_dir
from engine.training.select import disagreement_passages, uncertain_passages, to_review_passages
from engine.training.train import evaluate, prepare_all
from engine.viterbi import decode, event_best_costs, event_margins
from engine.weights import Weights

FIXTURES = Path(__file__).parent / "fixtures" / "pig"


def line(pitches, ioi=0.4, start_id=0):
    return [
        {"note_id": start_id + i, "pitch": p, "start_time_sec": i * ioi, "duration_sec": ioi}
        for i, p in enumerate(pitches)
    ]


def write_fingered_score(path, rh, lh=(), tempo=100):
    """Write a two-staff MusicXML with fingerings, as MuseScore would export."""
    score = m21.stream.Score()
    treble = m21.stream.Part(id="P1")
    bass = m21.stream.Part(id="P2")
    treble.append(m21.clef.TrebleClef())
    bass.append(m21.clef.BassClef())
    treble.append(m21.tempo.MetronomeMark(number=tempo))
    for entry in rh:
        pitches, fingers = entry
        if len(pitches) == 1:
            el = m21.note.Note(pitches[0], quarterLength=1)
        else:
            el = m21.chord.Chord(list(pitches), quarterLength=1)
        for f in fingers:
            if f is not None:
                el.articulations.append(m21.articulations.Fingering(f))
        treble.append(el)
    for entry in lh:
        pitches, fingers = entry
        el = (m21.note.Note(pitches[0], quarterLength=1) if len(pitches) == 1
              else m21.chord.Chord(list(pitches), quarterLength=1))
        for f in fingers:
            if f is not None:
                el.articulations.append(m21.articulations.Fingering(f))
        bass.append(el)
    if not lh:
        bass.append(m21.note.Rest(quarterLength=len(rh)))
    score.insert(0, treble)
    score.insert(0, bass)
    score.write("musicxml", fp=str(path))
    return path


# ---------------------------------------------------------------------------
# Forward-backward margins
# ---------------------------------------------------------------------------

def test_event_best_costs_matches_brute_force():
    rng = random.Random(11)
    w = Weights.default()
    checked = 0
    for _ in range(30):
        n = rng.randint(2, 5)
        pitches = [rng.randint(55, 79)]
        for _ in range(n - 1):
            pitches.append(max(48, min(84, pitches[-1] + rng.choice([-12, -7, -5, -3, -1, 1, 2, 3, 5, 7, 12]))))
        hand = rng.choice(["right", "left"])
        ioi = rng.choice([0.15, 0.5])
        data = [{"note_id": i, "pitch": p, "start_time_sec": i * ioi, "duration_sec": ioi}
                for i, p in enumerate(pitches)]
        prof = hand_profile(hand, 21.0)
        events = build_events(notes_from_hand_data(data), prof)
        tensors = FeatureTensors(events, prof)
        # Only compare where no held-note corner forces the decoder's relaxation,
        # since that relaxation legitimately reaches states brute force cannot.
        if any(tensors.feasible[k] is not None and (~tensors.feasible[k].any(axis=0)).any()
               for k in range(1, len(events))):
            continue
        best = event_best_costs(tensors, w)
        brute = [np.full(len(e.assignments), np.inf) for e in events]
        for path in itertools.product(*[range(len(e.assignments)) for e in events]):
            if not all(tensors.feasible[k] is None or tensors.feasible[k][path[k - 1], path[k]]
                       for k in range(1, len(path))):
                continue
            fs, ft, fq = tensors.path_features(path)
            cost = float(fs @ w.state + ft @ w.trans + fq @ w.second)
            for k, a in enumerate(path):
                brute[k][a] = min(brute[k][a], cost)
        for k in range(len(events)):
            finite = np.isfinite(brute[k])
            assert np.allclose(best[k][finite], brute[k][finite], atol=1e-9)
        checked += 1
    assert checked >= 15


def test_margins_are_non_negative_and_zero_free():
    data = line([60, 62, 64, 65, 67, 69, 71, 72], ioi=0.5)
    prof = hand_profile("right", 21.0)
    events = build_events(notes_from_hand_data(data), prof)
    tensors = FeatureTensors(events, prof)
    w = Weights.default()
    result = decode(tensors, w)
    margins = event_margins(tensors, w, result.path)
    assert len(margins) == len(events)
    assert (margins >= -1e-9).all()
    # Changing any single event of a standard scale should cost something.
    assert (margins > 0).all()


# ---------------------------------------------------------------------------
# MuseScore bridge
# ---------------------------------------------------------------------------

def test_reads_fingerings_from_musicxml(tmp_path):
    path = write_fingered_score(
        tmp_path / "s.musicxml",
        rh=[(["C4"], [1]), (["D4"], [2]), (["E4"], [3])],
        lh=[(["C3", "E3", "G3"], [5, 3, 1])],
    )
    hands = load_fingered_musicxml(path)
    assert [n["pitch"] for n in hands["right"].notes] == [60, 62, 64]
    assert hands["right"].fingers == [1, 2, 3]
    assert [n["pitch"] for n in hands["left"].notes] == [48, 52, 55]
    assert hands["left"].fingers == [5, 3, 1]
    assert hands["right"].coverage == 1.0


def test_partial_annotation_is_preserved(tmp_path):
    path = write_fingered_score(
        tmp_path / "p.musicxml",
        rh=[(["C4"], [1]), (["D4"], []), (["E4"], [3])],
    )
    hands = load_fingered_musicxml(path)
    assert hands["right"].fingers == [1, None, 3]
    assert hands["right"].annotated_count == 2
    assert hands["right"].coverage == pytest.approx(2 / 3)


def test_misaligned_chord_fingering_is_dropped_not_guessed(tmp_path):
    # Two fingerings on a three-note chord cannot be matched to pitches.
    path = write_fingered_score(tmp_path / "c.musicxml", rh=[(["C4", "E4", "G4"], [1, 3])])
    hands = load_fingered_musicxml(path)
    assert hands["right"].fingers == [None, None, None]


def test_convert_round_trips_through_pig_loader(tmp_path):
    path = write_fingered_score(
        tmp_path / "piece.musicxml",
        rh=[(["C4"], [1]), (["D4"], [2]), (["E4"], [3]), (["F4"], [1])],
        lh=[(["C3"], [5]), (["D3"], [4]), (["E3"], [3]), (["F3"], [2])],
        tempo=120,
    )
    out = tmp_path / "pig" / "piece-kh_fingering.txt"
    stats = convert(path, out)
    assert stats["right_annotated"] == 4 and stats["left_annotated"] == 4
    assert out.is_file()

    sequences = load_pig_dir(tmp_path / "pig")
    by_hand = {s.hand: s for s in sequences}
    assert by_hand["right"].fingers == [1, 2, 3, 1]
    assert by_hand["left"].fingers == [5, 4, 3, 2]
    assert [n["pitch"] for n in by_hand["right"].hand_data] == [60, 62, 64, 65]
    # 120 BPM means a quarter note lasts half a second.
    assert by_hand["right"].hand_data[1]["start_time_sec"] == pytest.approx(0.5)


def test_unannotated_notes_are_omitted_from_pig_output(tmp_path):
    path = write_fingered_score(tmp_path / "q.musicxml", rh=[(["C4"], [1]), (["D4"], []), (["E4"], [3])])
    hands = load_fingered_musicxml(path)
    lines = pig_lines([hands["right"], hands["left"]])
    data_lines = [ln for ln in lines if not ln.startswith("//")]
    assert len(data_lines) == 2
    assert all("\t" in ln for ln in data_lines)


def test_export_review_writes_readable_fingered_musicxml(tmp_path):
    notes = line([60, 62, 64, 65], ioi=0.5)
    passage = ReviewPassage(
        label="1. test RH", right_notes=notes, right_fingers=[1, 2, 3, 1],
        left_notes=[], left_fingers=[],
    )
    out = export_review([passage], tmp_path / "review.musicxml")
    raw = out.read_text()
    assert "<fingering" in raw
    # And it survives being read back, which is what closes the annotation loop.
    hands = load_fingered_musicxml(out)
    assert hands["right"].fingers[:4] == [1, 2, 3, 1]
    assert [n["pitch"] for n in hands["right"].notes][:4] == [60, 62, 64, 65]


# ---------------------------------------------------------------------------
# Passage selection
# ---------------------------------------------------------------------------

def test_uncertain_passages_are_returned_in_order_of_indifference():
    data = line([60, 62, 64, 65, 67, 69, 71, 72, 71, 69, 67, 65], ioi=0.35)
    passages = uncertain_passages(data, "right", source="scale", window=5, top_k=3)
    assert passages
    assert all(p.reason == "uncertain" for p in passages)
    assert passages == sorted(passages, key=lambda p: p.score)
    assert all(p.notes and len(p.notes) == len(p.model_fingers) for p in passages)
    assert all(f in (1, 2, 3, 4, 5) for p in passages for f in p.model_fingers)


def test_uncertain_passages_do_not_overlap():
    data = line(list(range(60, 84)), ioi=0.3)
    passages = uncertain_passages(data, "right", window=6, top_k=4)
    spans = sorted((p.notes[0]["start_time_sec"], p.notes[-1]["start_time_sec"]) for p in passages)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start > end


def test_disagreement_needs_annotators_to_agree_with_each_other():
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    data = line(pitches, ioi=0.35)
    agreeing = [
        PigSequence(piece="900", annotator=a, hand="right", path=Path(f"900-{a}_fingering.txt"),
                    hand_data=[dict(n) for n in data], fingers=[2, 3, 4, 5, 4, 3, 2, 1])
        for a in ("1", "2")
    ]
    found = disagreement_passages(agreeing, window=5, top_k_per_piece=2)
    assert found, "two annotators agreeing on an unusual fingering should be flagged"
    assert all(p.reason == "disagreement" for p in found)
    assert any(p.gold_fingers != p.model_fingers for p in found)

    # A single annotator is only an opinion, so nothing is flagged.
    assert disagreement_passages(agreeing[:1], window=5) == []

    # Annotators who contradict each other have no consensus to disagree with.
    conflicting = [
        PigSequence(piece="901", annotator="1", hand="right", path=Path("901-1_fingering.txt"),
                    hand_data=[dict(n) for n in data], fingers=[2, 3, 4, 5, 4, 3, 2, 1]),
        PigSequence(piece="901", annotator="2", hand="right", path=Path("901-2_fingering.txt"),
                    hand_data=[dict(n) for n in data], fingers=[1, 2, 3, 1, 2, 3, 4, 5]),
    ]
    assert disagreement_passages(conflicting, window=5) == []


def test_selected_passages_export_for_review(tmp_path):
    data = line([60, 62, 64, 65, 67, 69, 71, 72], ioi=0.35)
    passages = uncertain_passages(data, "right", source="scale", window=4, top_k=2)
    out = export_review(to_review_passages(passages), tmp_path / "r.musicxml")
    assert out.is_file() and out.stat().st_size > 0
    assert "<fingering" in out.read_text()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def test_flatten_timing_keeps_pitches_and_chords_but_removes_tempo():
    data = [
        {"note_id": 0, "pitch": 60, "start_time_sec": 0.0, "duration_sec": 0.11},
        {"note_id": 1, "pitch": 64, "start_time_sec": 0.0, "duration_sec": 0.11},
        {"note_id": 2, "pitch": 67, "start_time_sec": 0.13, "duration_sec": 1.9},
        {"note_id": 3, "pitch": 72, "start_time_sec": 2.4, "duration_sec": 0.2},
    ]
    seq = PigSequence(piece="1", annotator="1", hand="right", path=Path("x"),
                      hand_data=data, fingers=[1, 3, 5, 5])
    flat = flatten_timing([seq])[0]
    assert [n["pitch"] for n in flat.hand_data] == [60, 64, 67, 72]
    assert flat.fingers == [1, 3, 5, 5]
    # The two simultaneous notes stay simultaneous.
    assert flat.hand_data[0]["start_time_sec"] == flat.hand_data[1]["start_time_sec"]
    # Onsets become a uniform grid and every gap disappears.
    onsets = sorted({n["start_time_sec"] for n in flat.hand_data})
    assert onsets == [0.0, 0.5, 1.0]
    assert all(n["duration_sec"] == 0.5 for n in flat.hand_data)


def test_soft_match_rate_rewards_agreeing_with_any_annotator():
    sequences = [s for s in load_pig_dir(FIXTURES) if s.piece == "001"]
    prepared = prepare_all(sequences, 21.0, None)
    metrics = evaluate(prepared, Weights.default())
    assert metrics["soft_match_rate"] >= metrics["general_match_rate"]
    assert 0.0 <= metrics["general_match_rate"] <= 1.0
    assert metrics["pieces"] >= 1


def test_filename_parsing_handles_both_pig_and_our_own_names(tmp_path):
    from engine.training.pig import FILENAME_RE

    cases = {
        "001-1_fingering.txt": ("001", "1"),
        "chopin_op10_no1-kh_fingering.txt": ("chopin_op10_no1", "kh"),
        "bach-invention-8-marc_fingering.txt": ("bach-invention-8", "marc"),
    }
    for name, (piece, annotator) in cases.items():
        match = FILENAME_RE.match(name)
        assert match, name
        assert (match.group("piece"), match.group("annotator")) == (piece, annotator)


def test_own_annotations_evaluate_without_pig(tmp_path):
    """The whole no-dataset path: annotate, convert, get a match rate."""
    from engine.training.train import prepare_all

    path = write_fingered_score(
        tmp_path / "my_piece.musicxml",
        rh=[(["C4"], [1]), (["E4"], [2]), (["G4"], [3]), (["C5"], [5])],
        lh=[(["C3"], [5]), (["G3"], [1])],
        tempo=92,
    )
    out = tmp_path / "mine" / "my_piece-kh_fingering.txt"
    convert(path, out)

    sequences = load_pig_dir(tmp_path / "mine")
    assert {s.piece for s in sequences} == {"my_piece"}
    assert {s.annotator for s in sequences} == {"kh"}

    prepared = prepare_all(sequences, 23.0, None)
    metrics = evaluate(prepared, Weights.default())
    assert metrics["notes"] == 6
    assert 0.0 <= metrics["general_match_rate"] <= 1.0


def test_macos_zip_debris_is_ignored(tmp_path):
    """The PIG release is zipped on macOS and ships a __MACOSX shadow tree."""
    from engine.training.pig import find_fingering_files, is_macos_junk

    real = tmp_path / "PianoFingeringDataset_v1.2" / "FingeringFiles"
    junk = tmp_path / "__MACOSX" / "PianoFingeringDataset_v1.2" / "FingeringFiles"
    real.mkdir(parents=True)
    junk.mkdir(parents=True)
    for name in ("001-1_fingering.txt", "001-2_fingering.txt"):
        (real / name).write_text(
            "//Version: PianoFingering_v170101\n"
            "0\t0.0\t0.5\tC4\t80\t80\t0\t1\n"
            "1\t0.5\t1.0\tD4\t80\t80\t0\t2\n",
            encoding="utf-8",
        )
        (junk / f"._{name}").write_bytes(b"\x00\x05\x16\x07binary resource fork")

    found = find_fingering_files(tmp_path)
    assert len(found) == 2
    assert all(not is_macos_junk(p) for p in found)
    assert all("__MACOSX" not in p.parts for p in found)

    sequences = load_pig_dir(tmp_path)
    assert len(sequences) == 2
    assert {s.piece for s in sequences} == {"001"}
    assert all(s.fingers == [1, 2] for s in sequences)


def test_annotator_coverage_and_published_split():
    """The test set must be chosen by annotator count to match the literature."""
    from engine.training.benchmark import annotators_per_piece, split_by_annotator_count

    def seq(piece, annotator, hand="right"):
        return PigSequence(piece=piece, annotator=annotator, hand=hand,
                           path=Path(f"{piece}-{annotator}_fingering.txt"),
                           hand_data=line([60, 62, 64]), fingers=[1, 2, 3])

    sequences = (
        [seq("solo", "1")]                                      # 1 annotator
        + [seq("duo", a) for a in ("1", "2")]                   # 2 annotators
        + [seq("crowd", a) for a in ("1", "2", "3", "4", "5")]  # 5 annotators
    )
    assert annotators_per_piece(sequences) == {"solo": 1, "duo": 2, "crowd": 5}

    train, test = split_by_annotator_count(sequences, min_annotators=4)
    assert {s.piece for s in test} == {"crowd"}
    assert {s.piece for s in train} == {"solo", "duo"}

    # With a lower bar the multi-annotator pieces all move into the test set.
    train, test = split_by_annotator_count(sequences, min_annotators=2)
    assert {s.piece for s in test} == {"crowd", "duo"}
    assert {s.piece for s in train} == {"solo"}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _piece_sequences(piece, n_annotators, hand="right"):
    return [
        PigSequence(piece=piece, annotator=str(a), hand=hand,
                    path=Path(f"{piece}-{a}_fingering.txt"),
                    hand_data=line([60, 62, 64, 65]), fingers=[1, 2, 3, 1])
        for a in range(1, n_annotators + 1)
    ]


def test_folds_cover_every_piece_exactly_once():
    from engine.training.crossval import stratified_folds

    sequences = []
    for i in range(20):
        sequences += _piece_sequences(f"p{i:02d}", 1 if i < 12 else (2 if i < 16 else 4))
    folds = stratified_folds(sequences, folds=5)
    assert len(folds) == 5
    everything = [piece for fold in folds for piece in fold]
    assert len(everything) == len(set(everything)) == 20
    for a in range(5):
        for b in range(a + 1, 5):
            assert not (folds[a] & folds[b]), "folds must not overlap"


def test_folds_are_stratified_by_annotator_count():
    from engine.training.crossval import bucket_of, stratified_folds

    sequences = []
    for i in range(20):
        sequences += _piece_sequences(f"p{i:02d}", 1 if i < 10 else 4)
    counts = {f"p{i:02d}": (1 if i < 10 else 4) for i in range(20)}
    folds = stratified_folds(sequences, folds=5)
    # Each fold should hold two easy and two hard pieces, not four of one kind.
    for fold in folds:
        buckets = [bucket_of(counts[p]) for p in fold]
        assert buckets.count("1 annotator") == 2
        assert buckets.count("4+ annotators") == 2


def test_aggregate_is_note_weighted_and_split_by_bucket():
    from engine.training.crossval import PieceResult, aggregate

    results = [
        PieceResult("a", annotators=1, notes=100, general=0.90, highest=0.90, soft=0.90),
        PieceResult("b", annotators=6, notes=300, general=0.70, highest=0.80, soft=0.95),
    ]
    summary = aggregate(results)
    assert summary["1 annotator"]["general"] == pytest.approx(0.90)
    assert summary["4+ annotators"]["general"] == pytest.approx(0.70)
    # 100 notes at 0.90 and 300 at 0.70 average to 0.75, not 0.80.
    assert summary["all"]["general"] == pytest.approx(0.75)
    assert summary["all"]["notes"] == 400 and summary["all"]["pieces"] == 2


def test_score_pieces_reports_confusions():
    from engine.training.crossval import score_pieces
    from engine.training.train import prepare_all

    # A fingering no pianist would use, so the model is certain to differ.
    odd = PigSequence(piece="odd", annotator="1", hand="right", path=Path("odd-1_fingering.txt"),
                      hand_data=line([60, 62, 64, 65, 67, 69, 71, 72]),
                      fingers=[5, 4, 3, 2, 1, 2, 3, 4])
    prepared = prepare_all([odd], 21.0, None)
    results = score_pieces(prepared, Weights.default())
    assert len(results) == 1
    result = results[0]
    assert result.piece == "odd" and result.annotators == 1 and result.notes == 8
    assert 0.0 <= result.general <= 1.0
    assert sum(result.confusions.values()) == round((1 - result.general) * result.notes)
    assert all(1 <= g <= 5 and 1 <= m <= 5 and g != m for g, m in result.confusions)


# ---------------------------------------------------------------------------
# Weight sweep
# ---------------------------------------------------------------------------

def _fifths_fingered_12345():
    """Passages a pianist fingers 1-2-3-4-5, so avoiding 4 or 5 costs match rate."""
    return [
        PigSequence(piece=f"p{i}", annotator="1", hand="right",
                    path=Path(f"p{i}-1_fingering.txt"),
                    hand_data=line([s, s + 2, s + 4, s + 5, s + 7]),
                    fingers=[1, 2, 3, 4, 5])
        for i, s in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
    ]


def test_sweep_finds_a_deliberately_broken_weight():
    from engine.training.sweep import scaled, sweep
    from engine.training.train import prepare_all
    from engine.weights import DEFAULT_WEIGHTS

    prepared = prepare_all(_fifths_fingered_12345(), 21.0, None)
    broken = dict(DEFAULT_WEIGHTS)
    broken["use_f4"] = 5.0                      # absurdly discouraging finger 4
    weights = Weights.from_dict(broken)

    results, baseline = sweep(prepared, weights, ["use_f4", "three_four"], [0.0, 1.0, 2.0])

    by_multiplier = dict(results["use_f4"])
    assert by_multiplier[1.0] == pytest.approx(baseline)
    assert by_multiplier[0.0] > baseline, "lowering the broken weight must help"
    assert by_multiplier[2.0] < baseline, "raising it further must hurt"

    # An unrelated weight should not look like the culprit.
    unrelated = dict(results["three_four"])
    assert unrelated[0.0] == pytest.approx(baseline)


def test_finger_usage_shows_the_shortfall():
    from engine.training.sweep import finger_usage
    from engine.training.train import prepare_all
    from engine.weights import DEFAULT_WEIGHTS

    prepared = prepare_all(_fifths_fingered_12345(), 21.0, None)
    broken = dict(DEFAULT_WEIGHTS)
    broken["use_f4"] = 5.0
    model_counts, human_counts = finger_usage(prepared, Weights.from_dict(broken))

    assert sum(model_counts.values()) == sum(human_counts.values()) == 40
    assert human_counts[4] == 8
    assert model_counts[4] < human_counts[4], "the model should visibly avoid finger 4"


def test_scaled_changes_only_the_named_weight():
    from engine.training.sweep import scaled

    base = Weights.default()
    changed = scaled(base, "use_f4", 2.0)
    before, after = base.to_dict(), changed.to_dict()
    assert after["use_f4"] == pytest.approx(before["use_f4"] * 2)
    for name, value in before.items():
        if name != "use_f4":
            assert after[name] == pytest.approx(value)
    with pytest.raises(KeyError):
        scaled(base, "not_a_feature", 2.0)
