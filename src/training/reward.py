"""
Reward shaping module for RL training (v3).

Key changes from v2:
- Forward progress reduced from 0.05 to 0.01 (guidance, not dominant signal)
- Goal entry remains strong at 5.0
- Added hop bonus to reward multi-hop moves that bypass blockers
- Increased step penalty to discourage shuffling
- Added escalating goal proximity bonus (pins near goal get extra reward)
"""

# Default reward constants (v3)
REWARD_FORWARD_PROGRESS = 0.01    # Reduced: guidance only, not dominant
REWARD_GOAL_ENTRY = 5.0           # Strong: main learning signal
REWARD_GOAL_EXIT = -6.0           # Harsh penalty for leaving goal
REWARD_WIN = 50.0
REWARD_LOSE = -10.0
REWARD_DRAW = -2.0
REWARD_STEP_PENALTY = -0.02       # Doubled: stronger time pressure
REWARD_HOP_BONUS = 0.15           # New: reward for multi-hop moves
REWARD_GOAL_PROXIMITY = 0.03      # New: per-pin bonus for being within 3 cells of goal


def compute_step_reward(
    dist_before,
    dist_after,
    pins_in_goal_before,
    pins_in_goal_after,
    won,
    lost,
    drawn,
    is_hop=False,
    pins_near_goal=0,
    reward_forward=REWARD_FORWARD_PROGRESS,
    reward_goal_entry=REWARD_GOAL_ENTRY,
    reward_goal_exit=REWARD_GOAL_EXIT,
    reward_win=REWARD_WIN,
    reward_lose=REWARD_LOSE,
    reward_draw=REWARD_DRAW,
    reward_step_penalty=REWARD_STEP_PENALTY,
    reward_hop_bonus=REWARD_HOP_BONUS,
    reward_goal_proximity=REWARD_GOAL_PROXIMITY,
):
    """
    Compute the reward for a single step in the RL training loop.

    Args:
        dist_before: Total distance of pins from goal before step
        dist_after: Total distance of pins from goal after step
        pins_in_goal_before: Number of pins in goal before step
        pins_in_goal_after: Number of pins in goal after step
        won: Boolean indicating if the agent won
        lost: Boolean indicating if the agent lost
        drawn: Boolean indicating if the game was drawn
        is_hop: Boolean indicating if the move was a multi-hop jump
        pins_near_goal: Number of pins within 3 cells of any goal cell (not in goal)
        reward_forward: Reward per unit distance reduced
        reward_goal_entry: Reward per pin entering goal
        reward_goal_exit: Reward per pin exiting goal
        reward_win: Reward for winning
        reward_lose: Reward for losing
        reward_draw: Reward for drawing
        reward_step_penalty: Per-step time penalty
        reward_hop_bonus: Bonus for multi-hop moves
        reward_goal_proximity: Per-pin bonus for being near goal

    Returns:
        float: The computed reward for this step
    """
    reward = 0.0

    # 0. Per-step time penalty — encourages faster play
    reward += reward_step_penalty

    # 1. Forward progress reward (reduced — guidance only)
    distance_reduction = dist_before - dist_after
    reward += reward_forward * distance_reduction

    # 2. Goal entry and exit rewards (main learning signal)
    pin_change = pins_in_goal_after - pins_in_goal_before
    if pin_change > 0:
        reward += reward_goal_entry * pin_change
    elif pin_change < 0:
        reward += reward_goal_exit * abs(pin_change)

    # 3. Hop bonus — encourages learning to chain jumps over blockers
    if is_hop:
        reward += reward_hop_bonus

    # 4. Goal proximity bonus — escalating reward as pins approach goal
    reward += reward_goal_proximity * pins_near_goal

    # 5. Terminal state rewards
    if won:
        reward += reward_win
    elif lost:
        reward += reward_lose
    elif drawn:
        reward += reward_draw

    return float(reward)
