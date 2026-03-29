"""
greedy_agent.py — A greedy policy for Chinese Checkers.

Scoring heuristic per candidate move (pin_id, dest):
    score = dist(current_pos -> nearest_goal) - dist(dest -> nearest_goal)
            + 5.0  if dest is inside the goal triangle

The move with the highest score is chosen.  If no candidate improves the
score (all scores <= 0) the first legal move is used as a fallback so the
agent never stalls.
"""


def _min_dist_to_goal(board_wrapper, pos_idx: int, goal_indices: list) -> int:
    """Return the minimum axial distance from *pos_idx* to any goal cell."""
    return min(board_wrapper.axial_distance(pos_idx, g) for g in goal_indices)


def greedy_policy(board_wrapper, colour: str):
    """
    Choose the legal move for *colour* that maximises the greedy score.

    Parameters
    ----------
    board_wrapper : BoardWrapper
        The current game state.
    colour : str
        The colour whose turn it is.

    Returns
    -------
    (pin_id, dest) : tuple[int, int]
        The chosen pin identifier and destination cell index.

    Raises
    ------
    ValueError
        If there are no legal moves available for *colour*.
    """
    legal = board_wrapper.get_legal_moves(colour)
    if not legal:
        raise ValueError(f"No legal moves available for colour '{colour}'")

    goal_indices = board_wrapper.get_goal_indices(colour)
    goal_set = set(goal_indices)

    best_score = None
    best_move = None
    fallback_move = None

    # Build a quick lookup: pin_id -> current axial index
    pin_positions = {pin.id: pin.axialindex for pin in board_wrapper.pins[colour]}

    for pin_id, dests in legal.items():
        current_pos = pin_positions[pin_id]
        dist_current = _min_dist_to_goal(board_wrapper, current_pos, goal_indices)

        for dest in dests:
            # Record the first legal move as fallback
            if fallback_move is None:
                fallback_move = (pin_id, dest)

            dist_dest = _min_dist_to_goal(board_wrapper, dest, goal_indices)
            score = dist_current - dist_dest
            if dest in goal_set:
                score += 5.0

            if best_score is None or score > best_score:
                best_score = score
                best_move = (pin_id, dest)

    # Return the best move found; fall back to the first legal move if needed.
    return best_move if best_move is not None else fallback_move
