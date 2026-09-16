"""Passage constraints and preference learning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.passage import PassageSpec, format_fingering, parse_fingering, parse_segment
from engine.training.preferences import (
    Preference, append_preference, drift, fit, leave_one_out, load_preferences,
    resolve, resolve_all, satisfied,
)
from engine.weights import Weights

TURN = "C6 D6 C6 A#5 G#5"
DESCENT = "F5 G5 F5 D#5 D5 C5 A#4"


def spec(notes: str = TURN, **kw) -> PassageSpec:
    return PassageSpec.from_note_names(notes, hand="right", span_cm=23.0, **kw)


def test_spec_round_trips_through_json():
    original = spec(label="turn")
    restored = PassageSpec.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.notes == original.notes
    assert restored.hand == original.hand
    assert restored.span_cm == original.span_cm
    assert restored.label == "turn"


def test_parse_helpers():
    assert parse_fingering("3-4-3-2-1") == [3, 4, 3, 2, 1]
    assert parse_fingering("34321") == [3, 4, 3, 2, 1]
    assert format_fingering([3, 4]) == "3-4"
    assert parse_segment("9-13", 27) == (8, 12)
    assert parse_segment("9", 27) == (8, 8)
    with pytest.raises(ValueError):
        parse_fingering("3-9")
    with pytest.raises(ValueError):
        parse_segment("9-40", 27)
    with pytest.raises(ValueError):
        parse_segment("1-20", 27)  # too long to enumerate


def test_constrained_decode_honours_the_constraint():
    passage = spec().build()
    weights = Weights.default()
    best_path, best_cost = passage.best(weights)
    for finger in passage.options(0):
        got = passage.constrained(weights, {0: finger})
        assert got is not None
        path, cost = got
        assert passage.fingers(path)[0] == finger
        assert cost >= best_cost - 1e-9
    # The unconstrained optimum is reproduced by constraining to its own choice.
    got = passage.constrained(weights, {0: passage.fingers(best_path)[0]})
    assert got is not None and got[1] == pytest.approx(best_cost)


def test_constrained_decode_is_the_true_optimum_for_that_constraint():
    """Brute force over a short passage, to check the ban trick does not leak."""
    import itertools

    passage = spec("C5 D5 E5").build()
    weights = Weights.default()
    options = [passage.options(k) for k in range(len(passage))]
    for first in options[0]:
        want = min(
            (passage.constrained(weights, dict(enumerate(combo)))[1]
             for combo in itertools.product(*options) if combo[0] == first),
        )
        got = passage.constrained(weights, {0: first})
        assert got is not None
        assert got[1] == pytest.approx(want, abs=1e-6)


def make_pref(notes: str = TURN, prefer="3-4-3-2-1", over="2-4-3-2-1", **kw) -> Preference:
    return Preference(source=spec(notes, **kw), segment=(1, 5),
                      prefer=parse_fingering(prefer), over=parse_fingering(over))


def test_resolve_rejects_a_wrong_length_fingering():
    pref = Preference(source=spec(), segment=(1, 5), prefer=[3, 4, 3], over=[2, 4, 3, 2, 1])
    with pytest.raises(ValueError, match="3 fingers"):
        resolve(pref, Weights.default())


def test_fitting_one_preference_flips_it():
    base = Weights.default()
    comparisons = resolve_all([make_pref()], base)
    before = comparisons[0].cost_gap(base)
    adapted = fit(comparisons, base)
    after = comparisons[0].cost_gap(adapted)
    assert after < before
    assert satisfied(comparisons, adapted) == [True]


def test_fitting_leaves_the_prior_alone_when_it_already_agrees():
    base = Weights.default()
    passage = spec().build()
    path, _ = passage.best(base)
    chosen = passage.fingers(path)
    other = [f for f in passage.options(0) if f != chosen[0]][0]
    pref = Preference(source=spec(), segment=(1, 5),
                      prefer=chosen, over=[other] + chosen[1:])
    comparisons = resolve_all([pref], base)
    assert satisfied(comparisons, base) == [True]
    adapted = fit(comparisons, base, margin=0.0)
    distance, _ = drift(base, adapted)
    assert distance < 1e-6


def test_l2_to_prior_limits_the_drift():
    base = Weights.default()
    comparisons = resolve_all([make_pref()], base)
    loose, _ = drift(base, fit(comparisons, base, l2_to_prior=0.0))
    tight, _ = drift(base, fit(comparisons, base, l2_to_prior=0.5))
    assert tight < loose


def test_leave_one_out_generalises_across_the_same_shape():
    """Two passages sharing a shape: each should predict the other."""
    base = Weights.default()
    prefs = [
        make_pref(TURN, "3-4-3-2-1", "2-4-3-2-1", label="turn A"),
        make_pref("F5 G5 F5 D#5 D5", "3-4-3-2-1", "2-4-3-2-1", label="turn B"),
    ]
    comparisons = resolve_all(prefs, base)
    held = leave_one_out(comparisons, base)
    assert all(held)


def test_preferences_round_trip_through_a_file(tmp_path: Path):
    path = tmp_path / "prefs.jsonl"
    append_preference(path, make_pref(label="one"))
    append_preference(path, make_pref(DESCENT, "3-4-3-2-1-3-2", "2-4-3-2-1-3-2", label="two"))
    loaded = load_preferences(path)
    assert [p.source.label for p in loaded] == ["one", "two"]
    assert loaded[0].prefer == [3, 4, 3, 2, 1]
    assert loaded[0].recorded  # stamped on write


def test_loading_a_bad_line_names_the_line(tmp_path: Path):
    path = tmp_path / "prefs.jsonl"
    path.write_text('{"not": "a preference"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="prefs.jsonl:1"):
        load_preferences(path)


def test_comments_and_blank_lines_are_skipped(tmp_path: Path):
    path = tmp_path / "prefs.jsonl"
    append_preference(path, make_pref())
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n# a note to myself\n")
    assert len(load_preferences(path)) == 1
