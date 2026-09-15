"""Hand model: which spans are relaxed, comfortable, practical or impossible.

The span tables follow Parncutt, Sloboda, Clarke, Raekallio and Desain,
"An ergonomic model of keyboard fingering for melodic fragments"
(Music Perception, 1997). For each finger pair they give six signed
bounds in semitones:

    MinPrac <= MinComf <= MinRel <= MaxRel <= MaxComf <= MaxPrac

A signed span is measured in the pair's natural direction: for the
right hand, from the lower-numbered finger towards the higher-numbered
one is positive when pitch rises; for the left hand the same is true
when pitch falls. Negative spans are crossings.

Spans inside [MinRel, MaxRel] cost nothing. Between the relaxed and
comfortable bounds the cost grows slowly, between comfortable and
practical it grows quickly, and beyond the practical bounds it is
nearly prohibitive.

Balliauw, Herremans, Palhazi Cuervo and Sörensen (2017) adapt the tables
to small, medium and large hands. Here the whole table is scaled
linearly by the ratio of the player's measured maximum thumb-to-little
finger spread to a reference spread, and all bounds are expressed in
millimetres on the physical keyboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from .keyboard import Keyboard, STANDARD_KEYBOARD

FINGERS = (1, 2, 3, 4, 5)

# (MinPrac, MinComf, MinRel, MaxRel, MaxComf, MaxPrac) in semitones.
#
# The crossing side (negative values) of the thumb pairs is wider than
# in Parncutt's original table. Parncutt's MinComf(1,3) = -2 would make
# the thumb-under from G to C in an ordinary C major arpeggio (a fourth)
# "uncomfortable", and MinComf(1,4) = -1 would do the same to the
# thumb-under in a first-inversion arpeggio (a third). Jacobs (2001) and
# later work relax these bounds; the values below allow a fourth under
# 3, a third under 4 and a third under 2 at comfortable cost.
PARNCUTT_SPANS: Dict[Tuple[int, int], Tuple[int, int, int, int, int, int]] = {
    (1, 2): (-6, -4, 1, 5, 8, 10),
    (1, 3): (-8, -5, 3, 7, 10, 12),
    (1, 4): (-7, -4, 5, 9, 12, 14),
    (1, 5): (-4, -1, 7, 10, 13, 15),
    # Non-thumb pairs: Parncutt's comfortable and practical maxima are
    # widened by one semitone. Spans here are physical (a white-key third
    # is 47 mm, 3.4 average semitones, whether major or minor), and the
    # original semitone bounds were calibrated on melodic fragments that
    # sit mostly on white keys.
    (2, 3): (1, 1, 1, 2, 4, 6),
    (2, 4): (1, 1, 3, 4, 6, 8),
    (2, 5): (2, 2, 5, 6, 9, 11),
    (3, 4): (1, 1, 1, 2, 3, 5),
    (3, 5): (1, 1, 3, 4, 6, 8),
    (4, 5): (1, 1, 1, 2, 4, 6),
}

# Relaxed lateral position of each finger relative to finger 3, in
# semitones, for a right hand (mirrored for the left). Midpoints of the
# relaxed ranges above.
RELAXED_FINGER_OFFSET_ST = {1: -5.0, 2: -1.5, 3: 0.0, 4: 1.5, 5: 3.5}

# When the thumb is on a white key and a longer finger is on a black
# key, the finger reaches into the keyboard rather than sideways, so the
# lateral span can be smaller than the relaxed minimum without any
# compression. This credit (in semitones) is added to the lateral span
# for non-crossed thumb pairs, by the finger that is on the black key.
BLACK_KEY_REACH_CREDIT_ST = {2: 1.0, 3: 2.0, 4: 1.0, 5: 0.5}

# Maximum thumb-to-little finger spread that the Parncutt tables are
# assumed to describe. MaxPrac(1,5) = 15 semitones is a minor tenth,
# about 205 mm between key centres, which a hand spreading about 21 cm
# can reach.
REFERENCE_SPAN_CM = 21.0
MIN_SCALE = 0.6
MAX_SCALE = 1.45


@dataclass(frozen=True)
class SpanBounds:
    """Signed span bounds for one finger pair, in millimetres."""

    min_prac: float
    min_comf: float
    min_rel: float
    max_rel: float
    max_comf: float
    max_prac: float


@dataclass(frozen=True)
class HandProfile:
    """A player's hand on a particular keyboard."""

    side: str = "right"
    span_cm: float = 20.0
    keyboard: Keyboard = field(default=STANDARD_KEYBOARD)
    reference_span_cm: float = REFERENCE_SPAN_CM

    def __post_init__(self) -> None:
        if self.side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        if not (10.0 <= self.span_cm <= 32.0):
            raise ValueError("span_cm must be between 10 and 32 cm")

    @property
    def is_right(self) -> bool:
        return self.side == "right"

    @property
    def direction(self) -> int:
        """+1 if higher-numbered fingers naturally sit on higher keys."""
        return 1 if self.is_right else -1

    @property
    def scale(self) -> float:
        raw = self.span_cm / self.reference_span_cm
        return max(MIN_SCALE, min(MAX_SCALE, raw))

    @property
    def semitone_mm(self) -> float:
        return self.keyboard.semitone_mm

    def bounds_mm(self, finger_a: int, finger_b: int) -> SpanBounds:
        """Span bounds for a finger pair, in millimetres, scaled to this hand.

        Reach and squeeze do not scale alike. A larger hand spreads further,
        so the outward bounds grow with span, and a longer thumb passes
        further under, so the crossing bounds (negative values) grow too.

        But a larger hand does not lose the ability to put two fingers on
        neighbouring keys. How close a pair can sit is set by finger breadth
        and the width of the keys, not by how far the hand spreads. Scaling
        the positive minimum bounds up with span produced the absurd verdict
        that a 23 cm hand cannot play a chromatic semitone with fingers 2 and
        3, something every pianist does constantly. So those bounds shrink
        for a small hand but never grow past their reference value.
        """
        pair = (min(finger_a, finger_b), max(finger_a, finger_b))
        if pair[0] == pair[1]:
            raise ValueError("Span bounds are defined for two different fingers")
        st = PARNCUTT_SPANS[pair]
        reach = self.semitone_mm * self.scale
        squeeze = self.semitone_mm * min(1.0, self.scale)
        values = [
            value * (squeeze if index < 3 and value > 0 else reach)
            for index, value in enumerate(st)
        ]
        return SpanBounds(*values)

    def natural_span_mm(
        self, finger_a: int, midi_a: int, finger_b: int, midi_b: int
    ) -> float:
        """Signed span between two fingers in the pair's natural direction.

        Positive means the higher-numbered finger lies on the side of
        the keyboard where it naturally belongs (to the right for the
        right hand, to the left for the left hand). Negative means the
        fingers are crossed.
        """
        if finger_a == finger_b:
            raise ValueError("natural_span_mm needs two different fingers")
        if finger_a < finger_b:
            lo_midi, hi_midi = midi_a, midi_b
        else:
            lo_midi, hi_midi = midi_b, midi_a
        return self.direction * self.keyboard.signed_distance_mm(lo_midi, hi_midi)

    def finger_offset_mm(self, finger: int) -> float:
        """Relaxed lateral offset of a finger from the hand centre."""
        return (
            self.direction
            * RELAXED_FINGER_OFFSET_ST[finger]
            * self.semitone_mm
            * self.scale
        )

    def hand_position_mm(self, notes: Iterable[Tuple[int, int]]) -> float:
        """Estimated centre of the hand for a set of (midi, finger) pairs.

        Each note implies a hand centre at its key minus the relaxed
        offset of the finger playing it; the estimate is their mean.
        """
        total = 0.0
        count = 0
        for midi, finger in notes:
            total += self.keyboard.x_rear_mm(midi) - self.finger_offset_mm(finger)
            count += 1
        if count == 0:
            raise ValueError("hand_position_mm needs at least one note")
        return total / count

    def is_natural_order(self, fingers: Iterable[int], midis: Iterable[int]) -> bool:
        """True if fingers are strictly monotone in pitch in the natural sense."""
        pairs = sorted(zip(midis, fingers))
        for (m0, f0), (m1, f1) in zip(pairs, pairs[1:]):
            if m0 == m1:
                return False
            if self.direction * (f1 - f0) <= 0:
                return False
        return True


def hand_profile(side: str, span_cm: float | None, keyboard: Keyboard | None = None) -> HandProfile:
    """Convenience constructor with defaults."""
    return HandProfile(
        side=side,
        span_cm=float(span_cm) if span_cm else 20.0,
        keyboard=keyboard or STANDARD_KEYBOARD,
    )
