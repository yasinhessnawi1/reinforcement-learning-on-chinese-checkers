"""
advanced_heuristic.py — Composite scoring heuristic agent for Chinese Checkers.

Scoring components per candidate move (pin_id, dest):
    1. forward_distance   : weighted distance reduction toward goal
    2. chain_hop_bonus    : reward for multi-hop moves (hop chains cover more ground)
    3. center_control     : bonus for occupying centre-board positions
    4. compactness        : prefer moves that keep own pins clustered (enables chaining)
    5. opponent_blocking  : penalise leaving easy hop paths open for opponent

Slots into curriculum between greedy and self-play as a harder training opponent.
"""

import math
from src.env.board_wrapper import BoardWrapper

# Centre cells are those with small axial distance from the board centre (0, 0).
_CENTER_RADIUS = 3


def _min_dist_to_goal(board_wrapper: BoardWrapper, pos_idx: int, goal_indices: list) -> int:
    return min(board_wrapper.axial_distance(pos_idx, g) for g in goal_indices)


def _centroid(board_wrapper: BoardWrapper, colour: str) -> tuple[float, float]:
    """Return the (mean_q, mean_r) centroid of all pins for *colour*."""
    q_sum = r_sum = 0.0
    pins = board_wrapper.pins[colour]
    for pin in pins:
        cell = board_wrapper.board.cells[pin.axialindex]
        q_sum += cell.q
        r_sum += cell.r
    n = len(pins)
    return (q_sum / n, r_sum / n)


def _axial_dist_qr(q1: float, r1: float, q2: float, r2: float) -> float:
    """Axial distance between two (q, r) coordinates (float version)."""
    dq = q1 - q2
    dr = r1 - r2
    ds = (-q1 - r1) - (-q2 - r2)
    return (abs(dq) + abs(dr) + abs(ds)) / 2.0


def _board_center_dist(board_wrapper: BoardWrapper, cell_idx: int) -> int:
    """Return the axial distance from a cell to the board centre (index of (0,0))."""
    center_key = (0, 0)
    if center_key in board_wrapper.board.index_of:
        center_idx = board_wrapper.board.index_of[center_key]
        return board_wrapper.axial_distance(cell_idx, center_idx)
    # Fallback: compute from coordinates
    cell = board_wrapper.board.cells[cell_idx]
    return (abs(cell.q) + abs(cell.r) + abs(-cell.q - cell.r)) // 2


def _compactness_score(
    board_wrapper: BoardWrapper, colour: str, pin_id: int, dest: int
) -> float:
    """Score how much moving *pin_id* to *dest* keeps pins together.

    Higher = more compact = more hop chains available.
    """
    pins = board_wrapper.pins[colour]
    # Current centroid (excluding the moving pin)
    others = [p for p in pins if p.id != pin_id]
    if not others:
        return 0.0
    q_sum = sum(board_wrapper.board.cells[p.axialindex].q for p in others)
    r_sum = sum(board_wrapper.board.cells[p.axialindex].r for p in others)
    n = len(others)
    cq, cr = q_sum / n, r_sum / n

    dest_cell = board_wrapper.board.cells[dest]
    dist = _axial_dist_qr(dest_cell.q, dest_cell.r, cq, cr)
    # Closer to centroid = more compact; normalise by board radius ~8
    return max(0.0, (8.0 - dist) / 8.0)


def _hop_length(board_wrapper: BoardWrapper, start_idx: int, dest_idx: int) -> int:
    """Estimate number of hops: axial_distance / 2 (each hop covers ~2 cells)."""
    dist = board_wrapper.axial_distance(start_idx, dest_idx)
    return max(1, dist // 2)


def _is_adjacent(board_wrapper: BoardWrapper, idx_a: int, idx_b: int) -> bool:
    return board_wrapper.axial_distance(idx_a, idx_b) == 1


def _opponent_easy_hops(board_wrapper: BoardWrapper, colour: str, opponent_colour: str) -> int:
    """Count opponent pins that can reach the opponent's goal in one long hop chain.

    Used to penalise moves that leave the opponent's path clear.
    This is a lightweight approximation: count how many opponent pins are
    adjacent to a same-colour pin (a ready hop-off platform).
    """
    own_positions = {p.axialindex for p in board_wrapper.pins[colour]}
    opp_pins = board_wrapper.pins[opponent_colour]
    count = 0
    for opp_pin in opp_pins:
        for own_pos in own_positions:
            if _is_adjacent(board_wrapper, opp_pin.axialindex, own_pos):
                count += 1
                break
    return count


def advanced_heuristic_policy(board_wrapper: BoardWrapper, colour: str) -> tuple:
    """
    Choose the legal move for *colour* using a composite heuristic.

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

    # Identify opponent
    colours = board_wrapper.colours
    opponent = next((c for c in colours if c != colour), None)

    pin_positions = {pin.id: pin.axialindex for pin in board_wrapper.pins[colour]}

    best_score = None
    best_move = None
    fallback_move = None

    for pin_id, dests in legal.items():
        current_pos = pin_positions[pin_id]
        dist_current = _min_dist_to_goal(board_wrapper, current_pos, goal_indices)
        center_dist_current = _board_center_dist(board_wrapper, current_pos)

        for dest in dests:
            if fallback_move is None:
                fallback_move = (pin_id, dest)

            # 1. Forward distance: weighted distance improvement
            dist_dest = _min_dist_to_goal(board_wrapper, dest, goal_indices)
            forward = (dist_current - dist_dest) * 2.0
            if dest in goal_set:
                forward += 10.0  # strong goal-entry bonus

            # 2. Chain hop bonus: reward multi-hop moves
            is_single_step = board_wrapper.axial_distance(current_pos, dest) == 1
            if not is_single_step:
                hops = _hop_length(board_wrapper, current_pos, dest)
                hop_bonus = 0.5 * hops
            else:
                hop_bonus = 0.0

            # 3. Centre control: slight bonus for near-centre during opening
            center_dist_dest = _board_center_dist(board_wrapper, dest)
            # Reward moving closer to centre only when many pins still far from goal
            pins_in_goal = board_wrapper.pins_in_goal(colour)
            center_weight = max(0.0, (5 - pins_in_goal) / 5.0) * 0.3
            center_bonus = center_weight * (center_dist_current - center_dist_dest)

            # 4. Compactness: keep pins grouped when not many are in goal yet
            compact_weight = max(0.0, (7 - pins_in_goal) / 7.0) * 0.4
            compact_bonus = compact_weight * _compactness_score(
                board_wrapper, colour, pin_id, dest
            )

            # 5. Opponent blocking: penalise giving opponent easy hops
            # (lightweight: only check if this move opens a lane)
            blocking_penalty = 0.0
            if opponent is not None:
                # Moving a pin away from a position adjacent to opponent may help them
                opp_pos_set = {p.axialindex for p in board_wrapper.pins[opponent]}
                if _is_adjacent(board_wrapper, current_pos, next(iter(opp_pos_set), -1)):
                    blocking_penalty = -0.2  # small penalty, not main driver

            score = forward + hop_bonus + center_bonus + compact_bonus + blocking_penalty

            if best_score is None or score > best_score:
                best_score = score
                best_move = (pin_id, dest)

    return best_move if best_move is not None else fallback_move
