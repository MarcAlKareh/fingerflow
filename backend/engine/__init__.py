"""FingerFlow fingering engine.

Biomechanical cost model over physical keyboard geometry, decoded with a
second-order Viterbi search over chord-aware finger assignments.
"""

from .fingering import FingeringResult, assign_fingering, fingers_for_pitches
from .hand import HandProfile, hand_profile
from .keyboard import Keyboard, STANDARD_KEYBOARD, DS6_KEYBOARD, DS5_5_KEYBOARD
from .weights import Weights, DEFAULT_WEIGHTS, PRESETS

__all__ = [
    "FingeringResult",
    "assign_fingering",
    "fingers_for_pitches",
    "HandProfile",
    "hand_profile",
    "Keyboard",
    "STANDARD_KEYBOARD",
    "DS6_KEYBOARD",
    "DS5_5_KEYBOARD",
    "Weights",
    "DEFAULT_WEIGHTS",
    "PRESETS",
]
