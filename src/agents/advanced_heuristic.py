"""
advanced_heuristic.py — Composite scoring heuristic agent for Chinese Checkers.

Key improvements over greedy:
    1. look_ahead_bonus   : simulate the best move available AFTER this move (1-ply lookahead)
    2. chain_hop_bonus    : strongly reward multi-hop moves (actual distance covered)
    3. goal_fill_order    : prefer filling goal cells that unblock others (back-to-front)
    4. dont_leave_goal    : heavy penalty for moving a pin already in the goal
    5. opponent_blocking  : penalise moves that open long hop lanes for the opponent

The look-ahead is the critical differentiator — greedy picks the best single step,
advanced picks the move that sets up the best *next* step.
"""

from src.env.board_wrapper import BoardWrapper


def _min_dist_to_goal(board_wrapper: BoardWrapper, pos_idx: int, goal_indices: list) -> int:
    return min(board_wrapper.axial_distance(pos_idx, g) for g in goal_indices)


def _total_dist(board_wrapper: BoardWrapper, colour: str, goal_indices: list) -> int:
    """Sum of min distances for all pins not yet in goal."""
    goal_set = set(goal_indices)
    total = 0
    for pin in board_wrapper.pins[colour]:
        if pin.axialindex not in goal_set:
            total += _min_dist_to_goal(board_wrapper, pin.axialindex, goal_indices)
    return total


def _pins_in_goal(board_wrapper: BoardWrapper, colour: str, goal_set: set) -> int:
    return sum(1 for pin in board_wrapper.pins[colour] if pin.axialindex in goal_set)


def _score_position(board_wrapper: BoardWrapper, colour: str, goal_indices: list, goal_set: set) -> float:
    """Evaluate the board position for *colour* — used for look-ahead."""
    pins = _pins_in_goal(board_wrapper, colour, goal_set)
    dist = _total_dist(board_wrapper, colour, goal_indices)
    return pins * 100.0 + max(0.0, 200.0 - dist)


def _best_lookahead_score(board_wrapper: BoardWrapper, colour: str, goal_indices: list, goal_set: set) -> float:
    """Return the best position score achievable in one more move."""
    legal = board_wrapper.get_legal_moves(colour)
    if not legal:
        return _score_position(board_wrapper, colour, goal_indices, goal_set)

    best = -float('inf')
    for pin_id, dests in legal.items():
        pin = board_wrapper._pin_by_id(colour, pin_id)
        old_idx = pin.axialindex
        for dest in dests:
            # Simulate move
            pin.axialindex = dest
            score = _score_position(board_wrapper, colour, goal_indices, goal_set)
            # Undo move
            pin.axialindex = old_idx
            if score > best:
                best = score
    return best


def _opponent_hop_potential(board_wrapper: BoardWrapper, colour: str, opponent: str) -> int:
    """Count how many opponent pins have a long hop available (dist > 2 move)."""
    if opponent not in board_wrapper.pins:
        return 0
    opp_legal = board_wrapper.get_legal_moves(opponent)
    count = 0
    for pin_id, dests in opp_legal.items():
        pin = board_wrapper._pin_by_id(opponent, pin_id)
        for dest in dests:
            if board_wrapper.axial_distance(pin.axialindex, dest) > 2:
                count += 1
                break
    return count


def advanced_heuristic_policy(board_wrapper: BoardWrapper, colour: str) -> tuple:
    """
    Choose the legal move for *colour* using look-ahead + composite heuristic.

    Parameters
    ----------
    board_wrapper : BoardWrapper
    colour : str

    Returns
    -------
    (pin_id, dest) : tuple[int, int]
    """
    legal = board_wrapper.get_legal_moves(colour)
    if not legal:
        raise ValueError(f"No legal moves for '{colour}'")

    goal_indices = board_wrapper.get_goal_indices(colour)
    goal_set = set(goal_indices)
    colours = board_wrapper.colours
    opponent = next((c for c in colours if c != colour), None)

    pin_positions = {pin.id: pin.axialindex for pin in board_wrapper.pins[colour]}
    current_score = _score_position(board_wrapper, colour, goal_indices, goal_set)

    # Pre-compute opponent hop potential before any move
    opp_hop_before = _opponent_hop_potential(board_wrapper, colour, opponent) if opponent else 0

    best_score = None
    best_move = None
    fallback_move = None

    for pin_id, dests in legal.items():
        current_pos = pin_positions[pin_id]
        pin = board_wrapper._pin_by_id(colour, pin_id)
        in_goal_now = current_pos in goal_set

        for dest in dests:
            if fallback_move is None:
                fallback_move = (pin_id, dest)

            # 1. Don't leave goal — very heavy penalty
            if in_goal_now and dest not in goal_set:
                score = -1000.0
                if best_score is None or score > best_score:
                    best_score = score
                    best_move = (pin_id, dest)
                continue

            # Simulate move in place (fast — no board copy)
            pin.axialindex = dest

            # 2. Immediate position improvement
            new_score = _score_position(board_wrapper, colour, goal_indices, goal_set)
            immediate = new_score - current_score

            # 3. Look-ahead: best score achievable next move
            lookahead = _best_lookahead_score(board_wrapper, colour, goal_indices, goal_set)
            lookahead_gain = lookahead - new_score

            # 4. Chain hop bonus: actual axial distance covered
            move_dist = board_wrapper.axial_distance(current_pos, dest)
            hop_bonus = move_dist * 0.3 if move_dist > 1 else 0.0

            # 5. Opponent blocking: did this move reduce opponent hop potential?
            opp_hop_after = _opponent_hop_potential(board_wrapper, colour, opponent) if opponent else 0
            blocking_bonus = (opp_hop_before - opp_hop_after) * 0.5

            # Undo move
            pin.axialindex = current_pos

            total = immediate * 1.0 + lookahead_gain * 0.6 + hop_bonus + blocking_bonus

            if best_score is None or total > best_score:
                best_score = total
                best_move = (pin_id, dest)

    return best_move if best_move is not None else fallback_move
