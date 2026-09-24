"""Weight vectors for the feature families, with presets and JSON I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np

from .features import (
    N_SECOND, N_STATE, N_TRANS,
    SECOND_ORDER_FEATURES, STATE_FEATURES, TRANSITION_FEATURES,
)

# Hand-tuned defaults. Units: cost per semitone-equivalent for span
# features, per event/transition for indicators, per 100 mm for shifts,
# per 100 ms of excess movement time for Fitts pressure.
DEFAULT_WEIGHTS: Dict[str, float] = {
    # state
    "use_f1": 0.0, "use_f2": 0.0, "use_f3": 0.0, "use_f4": 0.35, "use_f5": 0.25,
    "weak_fast": 0.4,
    "thumb_black": 1.0, "two_black": 0.15, "four_black": 0.2, "five_black": 0.4,
    "chord_rel_out": 0.5, "chord_comf_out": 1.5, "chord_prac_out": 6.0,
    "chord_rel_in": 0.5, "chord_comf_in": 1.5, "chord_prac_in": 6.0,
    # transition
    "repeat_same_finger": 0.1, "repeat_fast": 1.5, "repeat_change": 0.3,
    "same_finger_diff_pitch": 0.8, "same_finger_diff_pitch_coupled": 8.0,
    "rel_out_thumb": 0.5, "rel_out_other": 0.7, "rel_in_thumb": 0.5, "rel_in_other": 0.7,
    "comf_out": 2.0, "comf_in": 2.0, "prac_out": 8.0, "prac_in": 8.0,
    "cross_1_2": 2.0, "cross_1_3": 1.6, "cross_1_4": 2.4, "cross_1_5": 6.0, "cross_amount": 0.3,
    "cross_other": 5.0, "cross_other_amount": 1.0,
    # A thumb pass is easier when the finger being passed is on a raised
    # black key (Parncutt's rule 12), hence the negative weight: it is a
    # discount on the fixed crossing cost, never a net reward.
    "cross_fast": 0.8, "cross_thumb_on_black": 1.0, "cross_other_on_black": -0.8,
    "three_four": 0.3, "four_black_three_white": 0.4,
    "depth_change": 0.3,
    "shift_mm": 0.4, "shift_big": 1.0, "fitts_pressure": 3.0,
    # second order
    "rule_345": 0.6, "thumb_zigzag": 1.5, "pos_change_count": 0.5, "pos_change_size": 0.3,
}

# Multipliers applied on top of the base weights for each goal.
PRESETS: Dict[str, Dict[str, float]] = {
    "expression": {
        # Favour legato connection and a settled hand: stretches and
        # crossings within a phrase count more, speed pressure less.
        "rel_out_thumb": 1.3, "rel_out_other": 1.3, "rel_in_thumb": 1.3, "rel_in_other": 1.3,
        "comf_out": 1.3, "comf_in": 1.3,
        "same_finger_diff_pitch_coupled": 1.5,
        "thumb_black": 1.3, "five_black": 1.3,
        "weak_fast": 0.7, "fitts_pressure": 0.7, "cross_fast": 0.7,
    },
    "speed": {
        # Favour fewer and smaller hand movements and strong fingers at
        # tempo; tolerate a little more stretch.
        "shift_mm": 1.5, "shift_big": 1.5, "fitts_pressure": 1.6,
        "weak_fast": 1.6, "cross_fast": 1.6, "repeat_fast": 1.5,
        "cross_1_3": 1.2, "cross_1_4": 1.2, "pos_change_count": 1.4,
        "rel_out_thumb": 0.8, "rel_out_other": 0.8, "comf_out": 0.85,
    },
    "beginner": {
        # Multipliers alone are not enough here; see PRESET_FLOORS.
    },
}

# Absolute lower bounds applied after the multipliers. A multiplier scales
# whatever training produced, and training on PIG produced the habits of
# professional pianists, who put the thumb on black keys far more freely
# than any beginner is taught to. A learner's teacher will say "thumb off
# the black keys" flatly, and a fingering that contradicts the teacher is
# worse than useless to the learner. So the beginner preset does not
# nudge that weight, it sets a floor on it.
#
# The value is empirical. On the Alla Turca, fingered by an advanced
# pianist who never puts the thumb on black, the built-in weights agreed
# on 41 of 61 notes; nothing changed until thumb_black reached 3.0, and at
# 6.0 the thumb left every black key and agreement rose to 50 of 61. That
# is one piece, so treat the number as a starting point and re-measure it
# when there are more.
PRESET_FLOORS: Dict[str, Dict[str, float]] = {
    "beginner": {"thumb_black": 6.0},
}


def _apply_preset(values: Dict[str, float], goal: str | None) -> Dict[str, float]:
    if not goal:
        return values
    for name, factor in PRESETS.get(goal, {}).items():
        values[name] = values.get(name, 0.0) * factor
    for name, floor in PRESET_FLOORS.get(goal, {}).items():
        values[name] = max(values.get(name, 0.0), floor)
    return values


@dataclass
class Weights:
    state: np.ndarray
    trans: np.ndarray
    second: np.ndarray

    @classmethod
    def from_dict(cls, values: Mapping[str, float]) -> "Weights":
        def vec(names: Iterable[str], size: int) -> np.ndarray:
            out = np.zeros(size, dtype=np.float64)
            for i, name in enumerate(names):
                out[i] = float(values.get(name, 0.0))
            return out

        return cls(
            state=vec(STATE_FEATURES, N_STATE),
            trans=vec(TRANSITION_FEATURES, N_TRANS),
            second=vec(SECOND_ORDER_FEATURES, N_SECOND),
        )

    @classmethod
    def default(cls, goal: str | None = None) -> "Weights":
        return cls.from_dict(_apply_preset(dict(DEFAULT_WEIGHTS), goal))

    def to_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(zip(STATE_FEATURES, self.state.tolist()))
        out.update(zip(TRANSITION_FEATURES, self.trans.tolist()))
        out.update(zip(SECOND_ORDER_FEATURES, self.second.tolist()))
        return out

    def with_preset(self, goal: str | None) -> "Weights":
        if not goal or (goal not in PRESETS and goal not in PRESET_FLOORS):
            return self
        return Weights.from_dict(_apply_preset(self.to_dict(), goal))

    def copy(self) -> "Weights":
        return Weights(self.state.copy(), self.trans.copy(), self.second.copy())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Weights":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
