import math
from typing import Dict, Tuple

# Tuning Constants
ALPHA = 1.5
BETA = 0.3
GAMMA = 0.1
C_13 = 10.0  # Cost for thumb under middle finger
C_14 = 20.0  # Cost for thumb under ring finger
INFINITY = float('inf')

# S_relaxed mapping (in semitones) for the Right Hand. 
# Example: Relaxed distance between Thumb (1) and Index (2) is roughly 2-4 semitones.
S_RELAXED_RH: Dict[Tuple[int, int], int] = {
    (1, 2): 3, (1, 3): 5, (1, 4): 7, (1, 5): 9,
    (2, 3): 2, (2, 4): 4, (2, 5): 5,
    (3, 4): 2, (3, 5): 4,
    (4, 5): 2,
}

def get_relaxed_span(f1: int, f2: int) -> int:
    """Gets the ideal natural span between two fingers."""
    if f1 == f2:
        return 0
    # Ensure smaller finger is first for dict lookup
    pair = (min(f1, f2), max(f1, f2))
    return S_RELAXED_RH.get(pair, 0)

def calculate_stretch_penalty(pitch_i: int, pitch_next: int, f_i: int, f_next: int) -> float:
    """Calculates exponential strain based on physical key distance vs relaxed hand span."""
    if f_i == f_next and pitch_i != pitch_next:
        # Huge penalty for trying to play two different notes simultaneously with the same finger
        return INFINITY 
        
    delta_p = abs(pitch_next - pitch_i)
    s_relaxed = get_relaxed_span(f_i, f_next)
    
    # Formula 1: W_stretch = alpha * e^(beta * |delta_p - S_relaxed|)
    w_stretch = ALPHA * math.exp(BETA * abs(delta_p - s_relaxed))
    return w_stretch

def calculate_cross_penalty(pitch_i: int, pitch_next: int, f_i: int, f_next: int) -> float:
    """Applies specific penalties for thumb-under / finger-over rules (Right Hand)."""
    # If pitch goes up but finger number goes down, it's a cross
    if pitch_next > pitch_i and f_next < f_i:
        if f_next == 1:
            if f_i == 3: return C_13
            if f_i == 4: return C_14
            if f_i == 5: return INFINITY # Highly unadvisable
            
    # If pitch goes down but finger number goes up (crossing over thumb)
    if pitch_next < pitch_i and f_next > f_i:
        if f_i == 1:
            if f_next == 3: return C_13
            if f_next == 4: return C_14
            if f_next == 5: return INFINITY
            
    return 0.0

def calculate_transition_cost(
    pitch_i: int, pitch_next: int, 
    f_i: int, f_next: int, 
    time_i: float, time_next: float
) -> float:
    """Computes the total transition cost W_final between two notes."""
    delta_t = time_next - time_i
    if delta_t <= 0:
        delta_t = 0.01  # Prevent division by zero for simultaneous chord notes
        
    w_stretch = calculate_stretch_penalty(pitch_i, pitch_next, f_i, f_next)
    w_cross = calculate_cross_penalty(pitch_i, pitch_next, f_i, f_next)
    
    # Formula 3: W_final = W_stretch * (1 + gamma / delta_t) + W_cross
    w_final = w_stretch * (1 + (GAMMA / delta_t)) + w_cross
    return w_final