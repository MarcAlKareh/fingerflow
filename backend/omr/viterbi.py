from typing import List, Dict, Tuple
from .cost_calculator import calculate_transition_cost, INFINITY

# Constants for 2nd Order Markov Chain (Temporal Finger Memory)
LAMBDA_PENALTY = 2.0
TAU_REFRACTORY = 0.120 # Minimum reset time for a single digit (120ms)

def run_2nd_order_viterbi(hand_data: List[Dict]) -> List[int]:
    """
    Finds the optimal path of fingers through the entire note array.
    Uses a 2nd-Order State Space: 25 state pairs per note (5^2).
    """
    if not hand_data:
        return []
    
    n_notes = len(hand_data)
    fingers = [1, 2, 3, 4, 5]
    
    # Viterbi table: dp[note_index][(f_prev, f_curr)] = min_cost
    dp = [{} for _ in range(n_notes)]
    
    # Backpointers to reconstruct the optimal sequence
    backpointers = [{} for _ in range(n_notes)]
    
    # Initialize the first note (note 0). Since there is no f_prev, we assume (f, f) with 0 cost.
    for f in fingers:
        dp[0][(f, f)] = 0.0
        
    # Iterate through all subsequent notes
    for i in range(1, n_notes):
        p_curr = hand_data[i]["pitch"]
        p_prev = hand_data[i-1]["pitch"]
        t_curr = hand_data[i]["start_time_sec"]
        t_prev = hand_data[i-1]["start_time_sec"]
        
        # Get start time of note i-2 for the rapid digit recovery penalty
        t_prev_prev = hand_data[i-2]["start_time_sec"] if i >= 2 else 0.0
        
        for f_curr in fingers:
            for f_prev in fingers:
                current_state = (f_prev, f_curr)
                min_cost = INFINITY
                best_prev_prev = None
                
                for f_prev_prev in fingers:
                    previous_state = (f_prev_prev, f_prev)
                    
                    if previous_state not in dp[i-1]:
                        continue
                        
                    # Standard transition cost
                    cost = calculate_transition_cost(
                        p_prev, p_curr, f_prev, f_curr, t_prev, t_curr
                    )
                    
                    # 2nd-Order Rapid Digit Repetition Penalty (Temporal Memory)
                    if f_curr == f_prev_prev:
                        dt_digit = t_curr - t_prev_prev
                        if dt_digit > TAU_REFRACTORY:
                            cost += LAMBDA_PENALTY / (dt_digit - TAU_REFRACTORY)
                        else:
                            cost += INFINITY # Physically impossible to reset that fast
                    
                    total_cost = dp[i-1][previous_state] + cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev_prev = f_prev_prev
                
                dp[i][current_state] = min_cost
                backpointers[i][current_state] = best_prev_prev

    # Traceback to find the best path
    best_final_state = min(dp[n_notes-1], key=dp[n_notes-1].get)
    
    optimal_path = [best_final_state[1], best_final_state[0]]
    current_state = best_final_state
    
    for i in range(n_notes - 1, 1, -1):
        f_prev_prev = backpointers[i][current_state]
        optimal_path.append(f_prev_prev)
        current_state = (f_prev_prev, current_state[0])
        
    return optimal_path[::-1] # Reverse the path to go from start to finish