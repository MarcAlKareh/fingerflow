"""Public entry point: assign fingers to one hand's notes.

    result = assign_fingering(hand_data, hand="right", hand_span_cm=23.0,
                              goal="expression")
    result.fingers            # {note_id: finger}
    result.finger_list        # fingers in the order of hand_data
    result.explanations       # {note_id: [human-readable reasons]}

``hand_data`` is the list produced by omr.parser: dicts with ``note_id``,
``pitch`` (MIDI), ``start_time_sec`` and ``duration_sec``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .events import Event, build_events, new_note_fingers, notes_from_hand_data
from .features import (
    FEATURE_LABELS, FeatureTensors, SECOND_ORDER_FEATURES, STATE_FEATURES, TRANSITION_FEATURES,
)
from .hand import HandProfile, hand_profile
from .keyboard import Keyboard
from .viterbi import decode
from .weights import Weights

EXPLANATION_MIN_COST = 0.15
EXPLANATION_MAX_ITEMS = 3


@dataclass
class FingeringResult:
    hand: str
    fingers: Dict[int, Optional[int]]
    finger_list: List[Optional[int]]
    total_cost: float
    events: List[Event] = field(default_factory=list)
    path: List[int] = field(default_factory=list)
    explanations: Dict[int, List[str]] = field(default_factory=dict)
    event_costs: Dict[int, float] = field(default_factory=dict)
    unfingered_note_ids: List[int] = field(default_factory=list)


def _explain_event(tensors: FeatureTensors, weights: Weights, path: Sequence[int], k: int) -> List[str]:
    contributions: List[tuple] = []
    a = path[k]
    contributions += [
        (name, float(v)) for name, v in zip(STATE_FEATURES, tensors.state[k][a] * weights.state)
    ]
    if k >= 1 and tensors.trans[k] is not None:
        contributions += [
            (name, float(v))
            for name, v in zip(TRANSITION_FEATURES, tensors.trans[k][path[k - 1], a] * weights.trans)
        ]
    if k >= 2 and tensors.second[k] is not None:
        contributions += [
            (name, float(v))
            for name, v in zip(
                SECOND_ORDER_FEATURES,
                tensors.second[k][path[k - 2], path[k - 1], a] * weights.second,
            )
        ]
    contributions = [(n, v) for n, v in contributions if v >= EXPLANATION_MIN_COST and n in FEATURE_LABELS]
    contributions.sort(key=lambda item: -item[1])
    return [FEATURE_LABELS[n] for n, _ in contributions[:EXPLANATION_MAX_ITEMS]]


def _event_cost(tensors: FeatureTensors, weights: Weights, path: Sequence[int], k: int) -> float:
    a = path[k]
    total = float(tensors.state[k][a] @ weights.state)
    if k >= 1 and tensors.trans[k] is not None:
        total += float(tensors.trans[k][path[k - 1], a] @ weights.trans)
    if k >= 2 and tensors.second[k] is not None:
        total += float(tensors.second[k][path[k - 2], path[k - 1], a] @ weights.second)
    return total


def assign_fingering(
    hand_data: Sequence[Dict],
    hand: str,
    hand_span_cm: Optional[float] = None,
    goal: Optional[str] = None,
    weights: Optional[Weights] = None,
    keyboard: Optional[Keyboard] = None,
    profile: Optional[HandProfile] = None,
    explain: bool = True,
) -> FingeringResult:
    """Compute the least-strain fingering for one hand."""
    if hand not in ("right", "left"):
        raise ValueError("hand must be 'right' or 'left'")
    if not hand_data:
        return FingeringResult(hand=hand, fingers={}, finger_list=[], total_cost=0.0)

    prof = profile or hand_profile(hand, hand_span_cm, keyboard)
    w = (weights or Weights.default()).with_preset(goal) if weights else Weights.default(goal)

    notes = notes_from_hand_data(hand_data)
    events = build_events(notes, prof)
    tensors = FeatureTensors(events, prof)
    result = decode(tensors, w)

    fingers: Dict[int, Optional[int]] = {n.note_id: None for n in notes}
    explanations: Dict[int, List[str]] = {}
    event_costs: Dict[int, float] = {}
    unfingered: List[int] = []
    for k, ev in enumerate(events):
        assign = ev.assignments[result.path[k]]
        reasons = _explain_event(tensors, w, result.path, k) if explain else []
        event_costs[k] = _event_cost(tensors, w, result.path, k)
        for note, finger in new_note_fingers(ev, assign):
            fingers[note.note_id] = int(finger)
            explanations[note.note_id] = reasons
        for note in ev.dropped:
            unfingered.append(note.note_id)

    finger_list = [fingers.get(int(item.get("note_id", idx))) for idx, item in enumerate(hand_data)]
    return FingeringResult(
        hand=hand,
        fingers=fingers,
        finger_list=finger_list,
        total_cost=result.cost,
        events=events,
        path=result.path,
        explanations=explanations,
        event_costs=event_costs,
        unfingered_note_ids=unfingered,
    )


def path_from_fingers(events: Sequence[Event], fingers: Dict[int, int]) -> List[Optional[int]]:
    """Map a fingering (note_id -> finger) to assignment indices per event.

    Returns None for an event whose fingering is not a legal assignment
    (non-monotone chord, finger changed on a held note, missing finger).
    """
    path: List[Optional[int]] = []
    for ev in events:
        wanted = []
        ok = True
        for note in ev.sounding:
            f = fingers.get(note.note_id)
            if f is None:
                ok = False
                break
            wanted.append(int(f))
        if not ok:
            path.append(None)
            continue
        try:
            path.append(ev.assignments.index(tuple(wanted)))
        except ValueError:
            path.append(None)
    return path


def cost_breakdown(
    hand_data: Sequence[Dict],
    hand: str,
    fingers: Sequence[Optional[int]],
    hand_span_cm: Optional[float] = None,
    goal: Optional[str] = None,
    weights: Optional[Weights] = None,
    keyboard: Optional[Keyboard] = None,
) -> Dict[str, float]:
    """Weighted cost of a given fingering, broken down by feature.

    Useful for comparing a human fingering with the engine's choice.
    Raises ValueError if the fingering is not a legal assignment.
    """
    prof = hand_profile(hand, hand_span_cm, keyboard)
    w = (weights or Weights.default()).with_preset(goal) if weights else Weights.default(goal)
    notes = notes_from_hand_data(hand_data)
    events = build_events(notes, prof)
    tensors = FeatureTensors(events, prof)
    by_id = {int(item.get("note_id", i)): f for i, (item, f) in enumerate(zip(hand_data, fingers))}
    path = path_from_fingers(events, by_id)
    if any(p is None for p in path):
        bad = [k for k, p in enumerate(path) if p is None]
        raise ValueError(f"Fingering is not a legal assignment at events {bad}")
    fs, ft, fq = tensors.path_features([int(p) for p in path])
    out: Dict[str, float] = {}
    for names, phi, wv in ((STATE_FEATURES, fs, w.state), (TRANSITION_FEATURES, ft, w.trans), (SECOND_ORDER_FEATURES, fq, w.second)):
        for name, value, weight in zip(names, phi, wv):
            if value * weight != 0.0:
                out[name] = float(value * weight)
    out["total"] = float(sum(v for k, v in out.items()))
    return out


def fingers_for_pitches(
    pitches: Sequence[int],
    hand: str,
    ioi_sec: float = 0.5,
    duration_sec: Optional[float] = None,
    **kwargs,
) -> List[Optional[int]]:
    """Convenience for tests and quick experiments: an isochronous line."""
    dur = ioi_sec if duration_sec is None else duration_sec
    hand_data = [
        {"note_id": i, "pitch": p, "start_time_sec": i * ioi_sec, "duration_sec": dur}
        for i, p in enumerate(pitches)
    ]
    return assign_fingering(hand_data, hand, **kwargs).finger_list
