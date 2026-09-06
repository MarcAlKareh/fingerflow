"""Biomechanical feature functions.

The cost of a fingering is a weighted sum of features:

    cost(path) = sum_k [ w_s . phi_s(k, s_k)
                       + w_t . phi_t(k, s_{k-1}, s_k)
                       + w_2 . phi_2(k, s_{k-2}, s_{k-1}, s_k) ]

where s_k is the finger assignment of event k. Three families:

* state features   phi_s: depend on one event and its assignment
                   (weak fingers, black-key rules, chord spans);
* transition       phi_t: depend on two consecutive assignments
                   (signed spans, crossings, hand shifts with Fitts's
                   law time pressure, finger repetition, legato coupling);
* second-order     phi_2: depend on three consecutive assignments
                   (3-4-5 rule, thumb zigzags, position changes).

Every feature is non-negative and expressed in physically meaningful
units (semitone-equivalents of distance, tenths of a second, counts), so
the default weights are interpretable and the vector can be learned
from human fingerings by a structured perceptron (see training/).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .events import Assignment, Event, new_note_fingers
from .hand import BLACK_KEY_REACH_CREDIT_ST, FINGERS, HandProfile, SpanBounds

# ---------------------------------------------------------------------------
# Tunable physical constants
# ---------------------------------------------------------------------------

# Fitts's law (Shannon form): MT = A + B * log2(1 + D / W).
FITTS_A_SEC = 0.05
FITTS_B_SEC_PER_BIT = 0.10

# Inter-onset interval at which "speed" features start to bite. A quarter
# note at 120 BPM lasts 0.5 s; the speed factor is log2(0.5 / IOI),
# clipped to [0, 4].
SPEED_REF_IOI_SEC = 0.5
SPEED_MAX = 4.0

# Same-finger repetition on one key becomes hard below this interval.
REPEAT_REF_IOI_SEC = 0.2

# The hand-centre estimate is noisy by about a semitone when the fingers
# are compressed or spread within one position, so small displacements
# are ignored; above SHIFT_THRESHOLD_MM a displacement is a real
# position change.
SHIFT_DEAD_ZONE_MM = 14.0
SHIFT_THRESHOLD_MM = 35.0


def speed_factor(ioi: float) -> float:
    if not math.isfinite(ioi) or ioi <= 0:
        return 0.0
    return max(0.0, min(SPEED_MAX, math.log2(SPEED_REF_IOI_SEC / ioi)))


def fitts_time_sec(distance_mm: float, width_mm: float) -> float:
    if distance_mm <= 0:
        return 0.0
    return FITTS_A_SEC + FITTS_B_SEC_PER_BIT * math.log2(1.0 + distance_mm / width_mm)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

STATE_FEATURES: List[str] = [
    "use_f1", "use_f2", "use_f3", "use_f4", "use_f5",
    "weak_fast",
    "thumb_black", "two_black", "four_black", "five_black",
    "chord_rel_out", "chord_comf_out", "chord_prac_out",
    "chord_rel_in", "chord_comf_in", "chord_prac_in",
]

TRANSITION_FEATURES: List[str] = [
    "repeat_same_finger", "repeat_fast", "repeat_change",
    "same_finger_diff_pitch", "same_finger_diff_pitch_coupled",
    "rel_out_thumb", "rel_out_other", "rel_in_thumb", "rel_in_other",
    "comf_out", "comf_in", "prac_out", "prac_in",
    "cross_1_2", "cross_1_3", "cross_1_4", "cross_1_5", "cross_amount",
    "cross_other", "cross_other_amount",
    "cross_fast", "cross_thumb_on_black", "cross_other_on_black",
    "three_four", "four_black_three_white",
    "depth_change",
    "shift_mm", "shift_big", "fitts_pressure",
] + [
    f"pair_{a}{b}_{d}" for a in FINGERS for b in FINGERS for d in ("up", "down")
]

SECOND_ORDER_FEATURES: List[str] = [
    "rule_345", "thumb_zigzag", "pos_change_count", "pos_change_size",
]

S_IDX: Dict[str, int] = {name: i for i, name in enumerate(STATE_FEATURES)}
T_IDX: Dict[str, int] = {name: i for i, name in enumerate(TRANSITION_FEATURES)}
Q_IDX: Dict[str, int] = {name: i for i, name in enumerate(SECOND_ORDER_FEATURES)}

N_STATE = len(STATE_FEATURES)
N_TRANS = len(TRANSITION_FEATURES)
N_SECOND = len(SECOND_ORDER_FEATURES)

# Human-readable labels for explanations shown to users.
FEATURE_LABELS: Dict[str, str] = {
    "use_f4": "uses the weak 4th finger",
    "use_f5": "uses the weak 5th finger",
    "weak_fast": "weak finger at speed",
    "thumb_black": "thumb on a black key",
    "two_black": "2nd finger on a black key",
    "five_black": "5th finger on a black key",
    "four_black": "4th finger on a black key",
    "repeat_change": "changes finger on a repeated key",
    "cross_amount": "wide thumb pass",
    "chord_rel_out": "chord wider than relaxed",
    "chord_comf_out": "chord stretch beyond comfortable",
    "chord_prac_out": "chord stretch beyond practical",
    "chord_rel_in": "chord fingers bunched",
    "chord_comf_in": "chord fingers cramped",
    "chord_prac_in": "chord fingers impossibly cramped",
    "repeat_same_finger": "same finger repeats the key",
    "repeat_fast": "fast repetition with one finger",
    "same_finger_diff_pitch": "same finger moves to a new key",
    "same_finger_diff_pitch_coupled": "same finger reused in legato",
    "rel_out_thumb": "stretch beyond relaxed (thumb pair)",
    "rel_out_other": "stretch beyond relaxed",
    "rel_in_thumb": "fingers closer than relaxed (thumb pair)",
    "rel_in_other": "fingers closer than relaxed",
    "comf_out": "stretch beyond comfortable",
    "comf_in": "fingers cramped",
    "prac_out": "stretch beyond practical reach",
    "prac_in": "fingers impossibly cramped",
    "cross_1_2": "thumb passes 2nd finger",
    "cross_1_3": "thumb passes 3rd finger",
    "cross_1_4": "thumb passes 4th finger",
    "cross_1_5": "thumb passes 5th finger",
    "cross_other": "crossing without the thumb",
    "cross_fast": "crossing at speed",
    "cross_thumb_on_black": "thumb passes onto a black key",
    "three_four": "3rd and 4th fingers in succession",
    "four_black_three_white": "4 on black next to 3 on white",
    "depth_change": "hand moves in or out of the keys",
    "shift_mm": "hand shifts position",
    "shift_big": "large hand shift",
    "fitts_pressure": "shift too fast for the time available",
    "rule_345": "3-4-5 in succession",
    "thumb_zigzag": "thumb crosses back and forth",
    "pos_change_count": "position change",
    "pos_change_size": "large position change",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span_tiers(distance_mm: float, bounds: SpanBounds, semitone_mm: float) -> Tuple[float, float, float, float, float, float]:
    """Tiered hinge amounts (rel_out, comf_out, prac_out, rel_in, comf_in, prac_in).

    Each tier measures how far the span intrudes into that zone, in
    semitone-equivalents, capped at the width of the zone so the tiers
    add up to a convex piecewise-linear penalty.
    """
    d = distance_mm
    rel_out = max(0.0, min(d, bounds.max_comf) - bounds.max_rel)
    comf_out = max(0.0, min(d, bounds.max_prac) - bounds.max_comf)
    prac_out = max(0.0, d - bounds.max_prac)
    rel_in = max(0.0, bounds.min_rel - max(d, bounds.min_comf))
    comf_in = max(0.0, bounds.min_comf - max(d, bounds.min_prac))
    prac_in = max(0.0, bounds.min_prac - d)
    return (
        rel_out / semitone_mm,
        comf_out / semitone_mm,
        prac_out / semitone_mm,
        rel_in / semitone_mm,
        comf_in / semitone_mm,
        prac_in / semitone_mm,
    )


def hand_travel_mm(hand: HandProfile, f_prev: int, midi_prev: int, f_cur: int, midi_cur: int) -> float:
    """Distance the hand must relocate between two notes, beyond finger reach.

    With the same finger the hand covers the whole distance. With two
    different fingers the pair can span up to its comfortable bound
    (outward, or the comfortable crossing bound inward) without moving
    the hand; only the excess is hand travel.
    """
    kb = hand.keyboard
    distance = abs(kb.signed_distance_mm(midi_prev, midi_cur))
    if f_prev == f_cur:
        return max(0.0, distance - SHIFT_DEAD_ZONE_MM)
    d = hand.natural_span_mm(f_prev, midi_prev, f_cur, midi_cur)
    bounds = hand.bounds_mm(f_prev, f_cur)
    reach = bounds.max_comf if d >= 0 else abs(min(bounds.min_comf, 0.0))
    return max(0.0, distance - reach)


def _is_crossing(hand: HandProfile, f_a: int, midi_a: int, f_b: int, midi_b: int) -> bool:
    if f_a == f_b or midi_a == midi_b:
        return False
    return hand.natural_span_mm(f_a, midi_a, f_b, midi_b) < 0


# ---------------------------------------------------------------------------
# State features
# ---------------------------------------------------------------------------

def state_features(event: Event, assign: Assignment, hand: HandProfile) -> np.ndarray:
    phi = np.zeros(N_STATE, dtype=np.float64)
    kb = hand.keyboard
    speed = speed_factor(event.ioi)

    for note, finger in new_note_fingers(event, assign):
        phi[S_IDX[f"use_f{finger}"]] += 1.0
        if finger in (4, 5):
            phi[S_IDX["weak_fast"]] += speed
        if kb.is_black(note.midi):
            if finger == 1:
                phi[S_IDX["thumb_black"]] += 1.0
            elif finger == 2:
                phi[S_IDX["two_black"]] += 1.0
            elif finger == 4:
                phi[S_IDX["four_black"]] += 1.0
            elif finger == 5:
                phi[S_IDX["five_black"]] += 1.0

    # Spans between every pair of simultaneously sounding notes.
    sounding = event.sounding
    if len(sounding) > 1:
        for i in range(len(sounding)):
            for j in range(i + 1, len(sounding)):
                fa, fb = assign[i], assign[j]
                d = hand.natural_span_mm(fa, sounding[i].midi, fb, sounding[j].midi)
                tiers = _span_tiers(d, hand.bounds_mm(fa, fb), kb.semitone_mm)
                phi[S_IDX["chord_rel_out"]] += tiers[0]
                phi[S_IDX["chord_comf_out"]] += tiers[1]
                phi[S_IDX["chord_prac_out"]] += tiers[2]
                phi[S_IDX["chord_rel_in"]] += tiers[3]
                phi[S_IDX["chord_comf_in"]] += tiers[4]
                phi[S_IDX["chord_prac_in"]] += tiers[5]
    return phi


# ---------------------------------------------------------------------------
# Transition features
# ---------------------------------------------------------------------------

def _paired_notes(prev: Event, prev_assign: Assignment, cur: Event, cur_assign: Assignment):
    """Yield (prev_note, prev_finger, cur_note, cur_finger) for each struck note."""
    prev_struck = new_note_fingers(prev, prev_assign)
    cur_struck = new_note_fingers(cur, cur_assign)
    for j, (note, finger) in enumerate(cur_struck):
        pidx = cur.pairing[j] if j < len(cur.pairing) else None
        if pidx is None or pidx >= len(prev_struck):
            continue
        pnote, pfinger = prev_struck[pidx]
        yield pnote, pfinger, note, finger


def transition_features(
    prev: Event, prev_assign: Assignment, cur: Event, cur_assign: Assignment, hand: HandProfile
) -> np.ndarray:
    phi = np.zeros(N_TRANS, dtype=np.float64)
    kb = hand.keyboard
    c = cur.coupling
    ioi = cur.ioi
    speed = speed_factor(ioi)

    for pnote, pf, note, f in _paired_notes(prev, prev_assign, cur, cur_assign):
        rising = note.midi > pnote.midi
        if note.midi != pnote.midi:
            phi[T_IDX[f"pair_{pf}{f}_{'up' if rising else 'down'}"]] += 1.0

        if note.midi == pnote.midi:
            # Repeated key: either the same finger strikes again or the
            # finger is changed on the key. No span is involved.
            if f == pf:
                phi[T_IDX["repeat_same_finger"]] += 1.0
                if math.isfinite(ioi) and ioi > 0:
                    phi[T_IDX["repeat_fast"]] += max(0.0, math.log2(REPEAT_REF_IOI_SEC / ioi))
            else:
                phi[T_IDX["repeat_change"]] += 1.0
        elif f == pf:
            phi[T_IDX["same_finger_diff_pitch"]] += 1.0
            phi[T_IDX["same_finger_diff_pitch_coupled"]] += c
        else:
            d = hand.natural_span_mm(pf, pnote.midi, f, note.midi)
            bounds = hand.bounds_mm(pf, f)
            thumb = 1 in (pf, f)
            other = f if pf == 1 else pf
            thumb_midi = note.midi if f == 1 else pnote.midi
            other_midi = pnote.midi if f == 1 else note.midi
            if thumb and d >= 0 and kb.is_black(other_midi) and not kb.is_black(thumb_midi):
                d += BLACK_KEY_REACH_CREDIT_ST.get(other, 0.0) * kb.semitone_mm * hand.scale
            tiers = list(_span_tiers(d, bounds, kb.semitone_mm))
            if d < 0 and thumb:
                # A thumb pass is a rotation, not a compression: the
                # relaxed-zone term does not apply, the crossing terms do.
                tiers[3] = 0.0
            phi[T_IDX["rel_out_thumb" if thumb else "rel_out_other"]] += c * tiers[0]
            phi[T_IDX["comf_out"]] += c * tiers[1]
            phi[T_IDX["prac_out"]] += c * tiers[2]
            phi[T_IDX["rel_in_thumb" if thumb else "rel_in_other"]] += c * tiers[3]
            phi[T_IDX["comf_in"]] += c * tiers[4]
            phi[T_IDX["prac_in"]] += c * tiers[5]

            if d < 0:
                # Crossed fingers.
                if thumb:
                    phi[T_IDX[f"cross_1_{other}"]] += c
                    phi[T_IDX["cross_amount"]] += c * (-d) / kb.semitone_mm
                    if kb.is_black(thumb_midi):
                        phi[T_IDX["cross_thumb_on_black"]] += c
                    if kb.is_black(other_midi):
                        phi[T_IDX["cross_other_on_black"]] += c
                else:
                    phi[T_IDX["cross_other"]] += c
                    phi[T_IDX["cross_other_amount"]] += c * (-d) / kb.semitone_mm
                phi[T_IDX["cross_fast"]] += c * speed

            if {pf, f} == {3, 4}:
                phi[T_IDX["three_four"]] += 1.0
            if (f == 4 and kb.is_black(note.midi) and pf == 3 and not kb.is_black(pnote.midi)) or (
                pf == 4 and kb.is_black(pnote.midi) and f == 3 and not kb.is_black(note.midi)
            ):
                phi[T_IDX["four_black_three_white"]] += 1.0

        phi[T_IDX["depth_change"]] += abs(kb.depth_mm(note.midi) - kb.depth_mm(pnote.midi)) / 100.0

    # Whole-hand shift between the two events (smoothness prior).
    prev_pos = hand.hand_position_mm((n.midi, fg) for n, fg in zip(prev.sounding, prev_assign))
    cur_pos = hand.hand_position_mm((n.midi, fg) for n, fg in zip(cur.sounding, cur_assign))
    shift = abs(cur_pos - prev_pos)
    phi[T_IDX["shift_mm"]] = max(0.0, shift - SHIFT_DEAD_ZONE_MM) / 100.0
    phi[T_IDX["shift_big"]] = max(0.0, shift - SHIFT_THRESHOLD_MM) / 100.0

    # Fitts's law time pressure for the distance the hand must actually
    # relocate: the key-to-key distance beyond what the finger pair can
    # reach from one position. Steps and thumb-passes within reach are
    # finger movements (the hand glides) and cost nothing here; a leap
    # beyond the pair's comfortable span is an aimed hand movement whose
    # minimum time grows with log2(1 + D / W).
    if math.isfinite(ioi) and ioi > 0:
        travel = 0.0
        width = kb.white_width_mm
        for pnote, pf, note, f in _paired_notes(prev, prev_assign, cur, cur_assign):
            t = hand_travel_mm(hand, pf, pnote.midi, f, note.midi)
            if t > travel:
                travel = t
                width = kb.target_width_mm(note.midi)
        if travel > 0:
            mt = fitts_time_sec(travel, width)
            phi[T_IDX["fitts_pressure"]] = max(0.0, mt - ioi) / 0.1
    return phi


# ---------------------------------------------------------------------------
# Second-order features
# ---------------------------------------------------------------------------

def second_order_features(
    pp: Event, pp_assign: Assignment,
    prev: Event, prev_assign: Assignment,
    cur: Event, cur_assign: Assignment,
    hand: HandProfile,
) -> np.ndarray:
    phi = np.zeros(N_SECOND, dtype=np.float64)
    kb = hand.keyboard
    pp_struck = new_note_fingers(pp, pp_assign)
    prev_struck = new_note_fingers(prev, prev_assign)
    cur_struck = new_note_fingers(cur, cur_assign)
    coupling = min(prev.coupling, cur.coupling)

    for j, (note, f) in enumerate(cur_struck):
        pidx = cur.pairing[j] if j < len(cur.pairing) else None
        if pidx is None or pidx >= len(prev_struck):
            continue
        pnote, pf = prev_struck[pidx]
        hidx = prev.pairing[pidx] if pidx < len(prev.pairing) else None
        if hidx is None or hidx >= len(pp_struck):
            continue
        hnote, hf = pp_struck[hidx]

        if {hf, pf, f} == {3, 4, 5}:
            phi[Q_IDX["rule_345"]] += 1.0

        cross_a = _is_crossing(hand, hf, hnote.midi, pf, pnote.midi) and 1 in (hf, pf)
        cross_b = _is_crossing(hand, pf, pnote.midi, f, note.midi) and 1 in (pf, f)
        if cross_a and cross_b:
            dir_a = pnote.midi - hnote.midi
            dir_b = note.midi - pnote.midi
            if dir_a * dir_b < 0:
                phi[Q_IDX["thumb_zigzag"]] += coupling

        if hnote.midi != note.midi:
            if hf == f:
                # The same finger two notes apart on different keys means
                # the whole hand has moved by that distance.
                moved = abs(kb.signed_distance_mm(hnote.midi, note.midi))
                phi[Q_IDX["pos_change_count"]] += coupling
                phi[Q_IDX["pos_change_size"]] += coupling * moved / kb.semitone_mm
            else:
                d = hand.natural_span_mm(hf, hnote.midi, f, note.midi)
                bounds = hand.bounds_mm(hf, f)
                if d > bounds.max_comf or d < bounds.min_comf:
                    phi[Q_IDX["pos_change_count"]] += coupling
                elif d > bounds.max_rel or d < bounds.min_rel:
                    phi[Q_IDX["pos_change_count"]] += 0.5 * coupling
                excess = max(0.0, d - bounds.max_rel, bounds.min_rel - d)
                phi[Q_IDX["pos_change_size"]] += coupling * excess / kb.semitone_mm
    return phi


# ---------------------------------------------------------------------------
# Per-piece feature tensors (shared by the decoder and the trainer)
# ---------------------------------------------------------------------------

class FeatureTensors:
    """Feature arrays for every event of one hand of one piece.

    ``state[k]``  has shape (n_k, N_STATE)
    ``trans[k]``  has shape (n_{k-1}, n_k, N_TRANS)      (k >= 1)
    ``second[k]`` has shape (n_{k-2}, n_{k-1}, n_k, N_SECOND) (k >= 2)
    ``feasible[k]`` is a boolean (n_{k-1}, n_k) mask of held-note consistency.
    """

    def __init__(self, events: Sequence[Event], hand: HandProfile, with_second_order: bool = True):
        from .events import transition_feasible

        self.events = list(events)
        self.hand = hand
        self.state: List[np.ndarray] = []
        self.trans: List[Optional[np.ndarray]] = []
        self.second: List[Optional[np.ndarray]] = []
        self.feasible: List[Optional[np.ndarray]] = []

        for k, ev in enumerate(self.events):
            n_k = len(ev.assignments)
            s = np.zeros((n_k, N_STATE), dtype=np.float32)
            for a, assign in enumerate(ev.assignments):
                s[a] = state_features(ev, assign, hand)
            self.state.append(s)

            if k == 0:
                self.trans.append(None)
                self.second.append(None)
                self.feasible.append(None)
                continue

            prev = self.events[k - 1]
            n_p = len(prev.assignments)
            t = np.zeros((n_p, n_k, N_TRANS), dtype=np.float32)
            feas = np.zeros((n_p, n_k), dtype=bool)
            for p, passign in enumerate(prev.assignments):
                for a, assign in enumerate(ev.assignments):
                    if not transition_feasible(prev, passign, ev, assign):
                        continue
                    feas[p, a] = True
                    t[p, a] = transition_features(prev, passign, ev, assign, hand)
            self.trans.append(t)
            self.feasible.append(feas)

            if k >= 2 and with_second_order:
                pp = self.events[k - 2]
                n_pp = len(pp.assignments)
                q = np.zeros((n_pp, n_p, n_k, N_SECOND), dtype=np.float32)
                for h, hassign in enumerate(pp.assignments):
                    for p, passign in enumerate(prev.assignments):
                        for a, assign in enumerate(ev.assignments):
                            if not feas[p, a]:
                                continue
                            q[h, p, a] = second_order_features(pp, hassign, prev, passign, ev, assign, hand)
                self.second.append(q)
            else:
                self.second.append(None)

    def path_features(self, path: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Summed feature vectors along a path of assignment indices."""
        fs = np.zeros(N_STATE)
        ft = np.zeros(N_TRANS)
        fq = np.zeros(N_SECOND)
        for k, a in enumerate(path):
            fs += self.state[k][a]
            if k >= 1 and self.trans[k] is not None:
                ft += self.trans[k][path[k - 1], a]
            if k >= 2 and self.second[k] is not None:
                fq += self.second[k][path[k - 2], path[k - 1], a]
        return fs, ft, fq
