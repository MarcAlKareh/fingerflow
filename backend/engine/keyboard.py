"""Physical geometry of a piano keyboard.

Fingering strain depends on physical distances between keys, not on
semitone counts. A semitone is 23.5 mm between E and F but only about
14 mm between C and C#, and a black key sits roughly 10 cm further into
the keyboard than the front of a white key.

Dimensions follow the modern standard (Wikipedia, "Musical keyboard"):
octave span 164-165 mm, white keys about 23.5 mm wide at the front,
black keys about 13.7 mm wide. The twelve key heads at the back of the
keyboard are laid out with (almost) uniform spacing, so a black key is
modelled at the centre of its 1/12-octave slot at the back of the key
bed, while a white key is modelled at the centre of its front.

The narrow "DS" keyboards (DS6.0 = 152 mm octave, DS5.5 = 140 mm) are
supported through ``Keyboard.octave_mm`` so that the same fingering
engine can be used for reduced-size instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

WHITE_PITCH_CLASSES = (0, 2, 4, 5, 7, 9, 11)
BLACK_PITCH_CLASSES = (1, 3, 6, 8, 10)

# Index of the white key inside its octave, for each pitch class.
# Black keys map to the white key immediately below them.
_WHITE_INDEX = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6}


@dataclass(frozen=True)
class Keyboard:
    """Geometry constants for one keyboard."""

    octave_mm: float = 164.5
    black_width_mm: float = 13.7
    # Depth (distance into the keyboard) at which a finger typically
    # contacts the key surface. White keys are played near the front
    # unless neighbouring black keys are in use.
    white_depth_mm: float = 30.0
    black_depth_mm: float = 105.0

    @property
    def white_width_mm(self) -> float:
        return self.octave_mm / 7.0

    @property
    def semitone_mm(self) -> float:
        """Average lateral distance per semitone."""
        return self.octave_mm / 12.0

    def is_black(self, midi: int) -> bool:
        return midi % 12 in BLACK_PITCH_CLASSES

    def x_rear_mm(self, midi: int) -> float:
        """Lateral centre of a key at the back of the key bed.

        At the back, all twelve keys of an octave occupy (almost) equal
        slots, so this coordinate is uniform in semitones. It is the
        position a finger uses when the hand is "in" among the black keys.
        """
        return (int(midi) + 0.5) * self.semitone_mm

    def x_front_mm(self, midi: int) -> float:
        """Lateral centre of a key where a finger normally contacts it.

        White keys are played at the front, spaced one white-key width
        apart. Black keys only exist at the back.
        """
        octave, pc = divmod(int(midi), 12)
        if pc in BLACK_PITCH_CLASSES:
            return self.x_rear_mm(midi)
        return octave * self.octave_mm + (_WHITE_INDEX[pc] + 0.5) * self.white_width_mm

    def x_mm(self, midi: int) -> float:
        """Default lateral coordinate (uniform rear coordinate)."""
        return self.x_rear_mm(midi)

    def signed_distance_mm(self, midi_from: int, midi_to: int) -> float:
        """Lateral distance a hand must cover between two keys.

        If both keys are white the fingers sit at the front of the keys;
        if either is black the hand moves in and the white key is played
        at its rear head, where spacing is uniform.
        """
        if self.is_black(midi_from) or self.is_black(midi_to):
            return self.x_rear_mm(midi_to) - self.x_rear_mm(midi_from)
        return self.x_front_mm(midi_to) - self.x_front_mm(midi_from)

    def depth_mm(self, midi: int) -> float:
        return self.black_depth_mm if self.is_black(midi) else self.white_depth_mm

    def target_width_mm(self, midi: int) -> float:
        """Width of the key as a Fitts's-law target."""
        return self.black_width_mm if self.is_black(midi) else self.white_width_mm

    def semitones_equivalent(self, distance_mm: float) -> float:
        """Convert a lateral distance to average-semitone units."""
        return distance_mm / self.semitone_mm


STANDARD_KEYBOARD = Keyboard()
DS6_KEYBOARD = Keyboard(octave_mm=152.0)
DS5_5_KEYBOARD = Keyboard(octave_mm=140.0)


@lru_cache(maxsize=None)
def midi_to_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave, pc = divmod(int(midi), 12)
    return f"{names[pc]}{octave - 1}"


_NAME_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def name_to_midi(name: str) -> int:
    """Parse scientific pitch names such as C4, F#3, Bb5, Cb4 or E##2."""
    text = name.strip()
    if not text:
        raise ValueError("Empty pitch name")
    letter = text[0].upper()
    if letter not in _NAME_TO_PC:
        raise ValueError(f"Unknown pitch letter in {name!r}")
    idx = 1
    accidental = 0
    while idx < len(text) and text[idx] in "#b-+xn":
        char = text[idx]
        if char == "#" or char == "+":
            accidental += 1
        elif char == "b" or char == "-":
            accidental -= 1
        elif char == "x":
            accidental += 2
        idx += 1
    octave = int(text[idx:])
    return (octave + 1) * 12 + _NAME_TO_PC[letter] + accidental
