from typing import List, Dict
from .cost_calculator import calculate_transition_cost, INFINITY

LAMBDA_PENALTY = 2.0
TAU_REFRACTORY = 0.120 # Minimum reset time for a single digit (120ms)

def run_2nd_order_viterbi(hand_data: List[Dict], hand: str) -> List[int]:
    """Finds the optimal path of fingers using a 2nd-Order Markov Chain."""
    if not hand_data:
        return []
    
    n_notes = len(hand_data)
    fingers = [1, 2, 3, 4, 5]
    
    dp = [{} for _ in range(n_notes)]
    backpointers = [{} for _ in range(n_notes)]
    
    for f in fingers:
        dp[0][(f, f)] = 0.0
        
    for i in range(1, n_notes):
        p_curr, t_curr = hand_data[i]["pitch"], hand_data[i]["start_time_sec"]
        p_prev, t_prev = hand_data[i-1]["pitch"], hand_data[i-1]["start_time_sec"]
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
                        
                    cost = calculate_transition_cost(
                        p_prev, p_curr, f_prev, f_curr, t_prev, t_curr, hand
                    )
                    
                    # Rapid Digit Repetition Penalty
                    if f_curr == f_prev_prev:
                        dt_digit = t_curr - t_prev_prev
                        if dt_digit > TAU_REFRACTORY:
                            cost += LAMBDA_PENALTY / (dt_digit - TAU_REFRACTORY)
                        else:
                            cost += INFINITY
                    
                    total_cost = dp[i-1][previous_state] + cost
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev_prev = f_prev_prev
                
                dp[i][current_state] = min_cost
                backpointers[i][current_state] = best_prev_prev

    best_final_state = min(dp[n_notes-1], key=dp[n_notes-1].get)
    optimal_path = [best_final_state[1], best_final_state[0]]
    current_state = best_final_state
    
    for i in range(n_notes - 1, 1, -1):
        f_prev_prev = backpointers[i][current_state]
        optimal_path.append(f_prev_prev)
        current_state = (f_prev_prev, current_state[0])
        
    return optimal_path[::-1]