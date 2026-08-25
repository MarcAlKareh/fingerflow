import math
from typing import Dict, Tuple

# Tuning Constants
ALPHA = 1.5
BETA = 0.3
GAMMA = 0.1
C_13 = 10.0  # Cost for thumb crossing middle finger
C_14 = 20.0  # Cost for thumb crossing ring finger
INFINITY = float('inf')

# Relaxed span mapping (in semitones). 
S_RELAXED: Dict[Tuple[int, int], int] = {
    (1, 2): 3, (1, 3): 5, (1, 4): 7, (1, 5): 9,
    (2, 3): 2, (2, 4): 4, (2, 5): 5,
    (3, 4): 2, (3, 5): 4,
    (4, 5): 2,
}

def get_relaxed_span(f1: int, f2: int) -> int:
    """Gets the ideal natural span between two fingers."""
    if f1 == f2:
        return 0
    pair = (min(f1, f2), max(f1, f2))
    return S_RELAXED.get(pair, 0)

def calculate_stretch_penalty(pitch_i: int, pitch_next: int, f_i: int, f_next: int) -> float:
    """Calculates exponential strain based on physical key distance vs relaxed hand span."""
    if f_i == f_next and pitch_i != pitch_next:
        return INFINITY  # Impossible to play two different notes simultaneously with one finger
        
    delta_p = abs(pitch_next - pitch_i)
    s_relaxed = get_relaxed_span(f_i, f_next)
    
    return ALPHA * math.exp(BETA * abs(delta_p - s_relaxed))

def calculate_cross_penalty(pitch_i: int, pitch_next: int, f_i: int, f_next: int, hand: str) -> float:
    """Applies specific penalties for thumb-under / finger-over rules depending on the hand."""
    if hand == "right":
        # RH Cross: Pitch goes UP, but finger goes DOWN (e.g., E(3) -> F(1))
        is_crossing_under = pitch_next > pitch_i and f_next < f_i
        # RH Cross Over: Pitch goes DOWN, but finger goes UP (e.g., F(1) -> E(3))
        is_crossing_over = pitch_next < pitch_i and f_next > f_i
    else:
        # LH Cross: Pitch goes UP, but finger goes UP (e.g., G(1) -> A(3))
        is_crossing_under = pitch_next > pitch_i and f_next > f_i
        # LH Cross Over: Pitch goes DOWN, but finger goes DOWN (e.g., A(3) -> G(1))
        is_crossing_over = pitch_next < pitch_i and f_next < f_i

    if is_crossing_under or is_crossing_over:
        if 1 in (f_i, f_next):
            other_finger = f_next if f_i == 1 else f_i
            if other_finger == 3: return C_13
            if other_finger == 4: return C_14
            if other_finger == 5: return INFINITY # Highly unadvisable to cross pinky and thumb
            
    return 0.0

def calculate_transition_cost(
    pitch_i: int, pitch_next: int, 
    f_i: int, f_next: int, 
    time_i: float, time_next: float,
    hand: str
) -> float:
    """Computes the total transition cost W_final between two notes."""
    delta_t = time_next - time_i
    if delta_t <= 0:
        delta_t = 0.01  # Prevent division by zero for simultaneous chord notes
        
    w_stretch = calculate_stretch_penalty(pitch_i, pitch_next, f_i, f_next)
    w_cross = calculate_cross_penalty(pitch_i, pitch_next, f_i, f_next, hand)
    
    return w_stretch * (1 + (GAMMA / delta_t)) + w_cross