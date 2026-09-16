"""Behavioural tests for the fingering engine.

These check the canonical fingerings every piano method agrees on, the
constraints (held notes, chords) and the sensitivities the model is
meant to have (tempo, hand size, articulation).
"""

import math
from pathlib import Path

import pytest

from engine import HandProfile, Keyboard, STANDARD_KEYBOARD, Weights, assign_fingering, fingers_for_pitches
from engine.events import build_events, enumerate_assignments, notes_from_hand_data
from engine.fingering import cost_breakdown
from engine.hand import hand_profile
from engine.keyboard import midi_to_name, name_to_midi

C_MAJOR_UP = [60, 62, 64, 65, 67, 69, 71, 72]
C_MAJOR_2OCT = C_MAJOR_UP + [74, 76, 77, 79, 81, 83, 84]


def line(pitches, ioi=0.5, dur=None, start_id=0):
    d = ioi if dur is None else dur
    return [
        {"note_id": start_id + i, "pitch": p, "start_time_sec": i * ioi, "duration_sec": d}
        for i, p in enumerate(pitches)
    ]


def chord(pitches, onset, dur, start_id):
    return [
        {"note_id": start_id + i, "pitch": p, "start_time_sec": onset, "duration_sec": dur}
        for i, p in enumerate(pitches)
    ]


# ---------------------------------------------------------------------------
# Keyboard geometry
# ---------------------------------------------------------------------------

def test_keyboard_dimensions():
    kb = STANDARD_KEYBOARD
    assert math.isclose(kb.octave_mm, 164.5)
    assert math.isclose(kb.white_width_mm, 23.5)
    assert math.isclose(kb.signed_distance_mm(60, 72), 164.5)            # octave
    assert math.isclose(kb.signed_distance_mm(64, 65), 23.5)             # E to F: one white key
    assert math.isclose(kb.signed_distance_mm(60, 61), kb.semitone_mm)   # C to C#: rear spacing
    assert kb.is_black(61) and not kb.is_black(60)
    assert midi_to_name(60) == "C4" and name_to_midi("F#3") == 54 and name_to_midi("Bb5") == 82


def test_narrow_keyboard_scales_distances():
    kb = Keyboard(octave_mm=140.0)
    assert math.isclose(kb.signed_distance_mm(60, 72), 140.0)


# ---------------------------------------------------------------------------
# Hand model
# ---------------------------------------------------------------------------

def test_hand_span_scales_bounds():
    small = hand_profile("right", 18.0)
    large = hand_profile("right", 24.0)
    assert large.bounds_mm(1, 5).max_prac > small.bounds_mm(1, 5).max_prac
    # A tenth (C4 to E5, 211.5 mm) is beyond practical for the small hand only.
    assert small.bounds_mm(1, 5).max_prac < 211.5 < large.bounds_mm(1, 5).max_prac


def test_left_hand_mirrors_natural_direction():
    rh = hand_profile("right", 21.0)
    lh = hand_profile("left", 21.0)
    # Thumb on C4, finger 5 on G4: natural for the right hand, crossed for the left.
    assert rh.natural_span_mm(1, 60, 5, 67) > 0
    assert lh.natural_span_mm(1, 60, 5, 67) < 0
    assert lh.natural_span_mm(5, 48, 1, 55) > 0


def test_assignment_enumeration_is_monotone():
    rh = hand_profile("right", 21.0)
    lh = hand_profile("left", 21.0)
    assert enumerate_assignments(3, rh) == [
        (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5), (1, 4, 5), (2, 3, 4), (2, 3, 5), (2, 4, 5), (3, 4, 5),
    ]
    assert enumerate_assignments(2, lh)[0] == (2, 1)
    assert enumerate_assignments(6, rh) == []


# ---------------------------------------------------------------------------
# Canonical fingerings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hand, pitches, expected", [
    ("right", C_MAJOR_UP, [1, 2, 3, 1, 2, 3, 4, 5]),
    ("right", C_MAJOR_UP[::-1], [5, 4, 3, 2, 1, 3, 2, 1]),
    ("left", C_MAJOR_UP, [5, 4, 3, 2, 1, 3, 2, 1]),
    ("left", C_MAJOR_UP[::-1], [1, 2, 3, 1, 2, 3, 4, 5]),
    ("right", [62, 64, 66, 67, 69, 71, 73, 74], [1, 2, 3, 1, 2, 3, 4, 5]),   # D major
    ("right", [71, 73, 75, 76, 78, 80, 82, 83], [1, 2, 3, 1, 2, 3, 4, 5]),   # B major
    ("left", [71, 73, 75, 76, 78, 80, 82, 83], [5, 4, 3, 2, 1, 3, 2, 1]),
])
def test_scales(hand, pitches, expected):
    assert fingers_for_pitches(pitches, hand) == expected


def test_two_octave_scale_right_hand():
    assert fingers_for_pitches(C_MAJOR_2OCT, "right") == [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4, 5]


def test_arpeggios():
    assert fingers_for_pitches([60, 64, 67, 72], "right") == [1, 2, 3, 5]
    assert fingers_for_pitches([48, 52, 55, 60], "left") == [5, 4, 2, 1]
    assert fingers_for_pitches([60, 64, 67, 72, 76, 79, 84], "right") == [1, 2, 3, 1, 2, 3, 5]


def test_alberti_bass_keeps_hand_still():
    assert fingers_for_pitches([48, 55, 52, 55] * 2, "left", ioi_sec=0.25) == [5, 1, 3, 1] * 2


def test_single_note_and_empty():
    assert fingers_for_pitches([60], "right") == [1]
    assert fingers_for_pitches([60], "left") in ([1], [2], [3])
    assert assign_fingering([], "right").finger_list == []


# ---------------------------------------------------------------------------
# Chords and held notes
# ---------------------------------------------------------------------------

def chord_fingers(pitches, hand, **kw):
    return assign_fingering(chord(pitches, 0.0, 1.0, 0), hand, **kw).finger_list


def test_triads():
    assert chord_fingers([60, 64, 67], "right") == [1, 3, 5]
    assert chord_fingers([48, 52, 55], "left") == [5, 3, 1]
    assert chord_fingers([64, 67, 72], "right") == [1, 2, 5]      # first inversion
    assert chord_fingers([67, 72, 76], "right") == [1, 3, 5]      # second inversion
    assert chord_fingers([60, 64, 67, 72], "right") == [1, 2, 3, 5]
    assert chord_fingers([48, 52, 55, 60], "left") == [5, 4, 2, 1]


def test_chord_fingers_are_distinct_and_monotone():
    fingers = chord_fingers([60, 62, 64, 65, 67], "right")
    assert fingers == [1, 2, 3, 4, 5]
    fingers = chord_fingers([60, 64, 67, 72, 76], "right", hand_span_cm=24)
    assert len(set(fingers)) == 5 and fingers == sorted(fingers)


def test_more_than_five_notes_reports_unfingered():
    result = assign_fingering(chord([60, 62, 64, 65, 67, 69], 0.0, 1.0, 0), "right")
    assert len(result.unfingered_note_ids) == 1
    assert result.fingers[0] is None            # lowest note dropped for the right hand
    assert all(result.fingers[i] is not None for i in range(1, 6))


def test_held_note_keeps_its_finger_and_blocks_reuse():
    data = [{"note_id": 0, "pitch": 60, "start_time_sec": 0.0, "duration_sec": 2.0}]
    data += line([64, 67, 72], ioi=0.5, start_id=1)
    for n in data[1:]:
        n["start_time_sec"] += 0.5
    result = assign_fingering(data, "right")
    events = result.events
    assert events[1].held_notes and events[1].held_notes[0].note_id == 0
    assert result.fingers[0] == 1
    assert result.fingers[0] not in {result.fingers[1], result.fingers[2], result.fingers[3]}
    assert result.finger_list == [1, 2, 3, 5]


def test_held_bass_forces_monotone_assignment():
    # Left hand holds C2 with 5; the notes above must use fingers < 5.
    data = [{"note_id": 0, "pitch": 36, "start_time_sec": 0.0, "duration_sec": 3.0}]
    data += [{"note_id": i + 1, "pitch": p, "start_time_sec": 0.5 * (i + 1), "duration_sec": 0.5}
             for i, p in enumerate([43, 48, 52])]
    result = assign_fingering(data, "left")
    assert result.fingers[0] == 5
    assert all(result.fingers[i] < 5 for i in (1, 2, 3))


# ---------------------------------------------------------------------------
# Sensitivities: tempo, hand size, articulation, goal
# ---------------------------------------------------------------------------

def test_fast_repeated_notes_change_fingers():
    slow = fingers_for_pitches([60] * 6, "right", ioi_sec=0.5)
    fast = fingers_for_pitches([60] * 6, "right", ioi_sec=0.1)
    assert len(set(slow)) == 1
    assert len(set(fast)) >= 2


def test_octave_leaps_cost_more_when_fast():
    def cost(ioi):
        data = line([60, 72, 60, 72], ioi=ioi, dur=ioi * 0.5)
        return assign_fingering(data, "right").total_cost
    assert cost(1.0) < cost(0.5) < cost(0.25) < cost(0.125)
    assert assign_fingering(line([60, 72, 60, 72], ioi=0.125, dur=0.06), "right").finger_list == [1, 5, 1, 5]


def test_small_hand_pays_for_a_tenth():
    data = chord([60, 76], 0.0, 1.0, 0)
    small = cost_breakdown(data, "right", [1, 5], hand_span_cm=18.0)
    large = cost_breakdown(data, "right", [1, 5], hand_span_cm=24.0)
    assert small["total"] > large["total"]
    assert small.get("chord_prac_out", 0.0) > 0.0
    assert large.get("chord_prac_out", 0.0) == 0.0


def test_rest_decouples_span_constraint():
    # C4 then C5 legato: connecting finger pair is stretched. With a long rest
    # between them the hand simply relocates and the transition cost drops.
    legato = line([60, 72], ioi=0.5)
    rest = line([60, 72], ioi=2.0, dur=0.3)
    c_legato = cost_breakdown(legato, "right", [1, 1])
    c_rest = cost_breakdown(rest, "right", [1, 1])
    assert c_legato["same_finger_diff_pitch_coupled"] > c_rest.get("same_finger_diff_pitch_coupled", 0.0)


def test_goal_presets_change_weights_not_validity():
    expr = Weights.default("expression")
    speed = Weights.default("speed")
    base = Weights.default()
    assert expr.to_dict()["fitts_pressure"] < base.to_dict()["fitts_pressure"] < speed.to_dict()["fitts_pressure"]
    for goal in ("expression", "speed", None):
        assert fingers_for_pitches(C_MAJOR_UP, "right", goal=goal) == [1, 2, 3, 1, 2, 3, 4, 5]


def test_explanations_and_cost_breakdown_agree():
    data = line(C_MAJOR_UP)
    result = assign_fingering(data, "right")
    assert any(result.explanations[i] for i in range(8))
    breakdown = cost_breakdown(data, "right", result.finger_list)
    assert math.isclose(breakdown["total"], result.total_cost, rel_tol=1e-6)
    # A legal but worse fingering must cost more than the optimum.
    worse = cost_breakdown(data, "right", [1, 2, 1, 2, 1, 2, 3, 4])
    assert worse["total"] > result.total_cost


def test_weights_roundtrip(tmp_path):
    w = Weights.default()
    w.save(tmp_path / "w.json")
    loaded = Weights.load(tmp_path / "w.json")
    assert loaded.to_dict() == w.to_dict()


def test_illegal_fingering_is_rejected_in_breakdown():
    data = chord([60, 64, 67], 0.0, 1.0, 0)
    with pytest.raises(ValueError):
        cost_breakdown(data, "right", [3, 2, 1])


# ---------------------------------------------------------------------------
# Decoder optimality against brute force
# ---------------------------------------------------------------------------

def test_viterbi_matches_brute_force():
    import itertools
    import random

    from engine.events import build_events, notes_from_hand_data
    from engine.features import FeatureTensors
    from engine.viterbi import decode

    rng = random.Random(11)
    w = Weights.default()
    for _ in range(25):
        n = rng.randint(1, 5)
        pitches = [rng.randint(55, 79)]
        for _ in range(n - 1):
            pitches.append(max(48, min(84, pitches[-1] + rng.choice([-12, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 12]))))
        hand = rng.choice(["right", "left"])
        ioi = rng.choice([0.1, 0.3, 0.6])
        data = [{"note_id": i, "pitch": p, "start_time_sec": i * ioi, "duration_sec": ioi * rng.choice([0.5, 1.0, 2.0])}
                for i, p in enumerate(pitches)]
        # Occasionally make it polyphonic: a second note on the first event, held.
        if rng.random() < 0.4 and n >= 2:
            data.append({"note_id": n, "pitch": pitches[0] + rng.choice([4, 7, -5]), "start_time_sec": 0.0, "duration_sec": ioi * 2.5})
        prof = hand_profile(hand, 21.0)
        events = build_events(notes_from_hand_data(data), prof)
        tensors = FeatureTensors(events, prof)
        result = decode(tensors, w)
        best = math.inf
        for path in itertools.product(*[range(len(e.assignments)) for e in events]):
            feasible = all(
                tensors.feasible[k] is None or tensors.feasible[k][path[k - 1], path[k]]
                for k in range(1, len(path))
            )
            if not feasible:
                continue
            fs, ft, fq = tensors.path_features(path)
            best = min(best, float(fs @ w.state + ft @ w.trans + fq @ w.second))
        assert math.isclose(best, result.cost, rel_tol=1e-9, abs_tol=1e-9)


def test_large_hands_can_still_play_close_intervals():
    """Reach scales with hand size; squeeze does not.

    Scaling the minimum bounds up with span made a 23 cm hand unable to play a
    chromatic semitone with fingers 2 and 3, which every pianist does.
    """
    semitone = STANDARD_KEYBOARD.semitone_mm
    small, reference, large = (hand_profile("right", cm) for cm in (17.0, 21.0, 25.0))

    # Squeeze: never worse than the reference hand, looser for a small one.
    assert small.bounds_mm(2, 3).min_prac < reference.bounds_mm(2, 3).min_prac
    assert large.bounds_mm(2, 3).min_prac == pytest.approx(reference.bounds_mm(2, 3).min_prac)
    for profile in (small, reference, large):
        assert profile.bounds_mm(2, 3).min_prac <= semitone + 1e-9, (
            "adjacent fingers must always manage neighbouring keys"
        )

    # Reach: still grows with the hand.
    assert small.bounds_mm(1, 5).max_prac < reference.bounds_mm(1, 5).max_prac
    assert reference.bounds_mm(1, 5).max_prac < large.bounds_mm(1, 5).max_prac

    # Crossings are reach, not squeeze, so they still scale.
    assert abs(small.bounds_mm(1, 3).min_prac) < abs(large.bounds_mm(1, 3).min_prac)


def test_chromatic_semitone_costs_nothing_extra_for_a_large_hand():
    from engine.fingering import cost_breakdown

    data = line([68, 67], ioi=0.3)          # G#5 then G5, played 3 then 2
    for span in (19.0, 21.0, 23.0, 25.0):
        breakdown = cost_breakdown(data, "right", [3, 2], hand_span_cm=span)
        assert breakdown.get("prac_in", 0.0) == 0.0, f"{span} cm hand called a semitone impossible"
        assert breakdown.get("comf_in", 0.0) == 0.0, f"{span} cm hand called a semitone cramped"
