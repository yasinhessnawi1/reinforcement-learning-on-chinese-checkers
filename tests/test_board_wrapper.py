"""
Tests for src/env/board_wrapper.py
"""
import sys
import os

# Make sure src is importable regardless of where pytest is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from src.env.board_wrapper import BoardWrapper, COLOUR_OPPOSITES


@pytest.fixture(scope='module')
def wrapper():
    """BoardWrapper with red and blue active."""
    return BoardWrapper(['red', 'blue'])


# ---------------------------------------------------------------------------
# Board structure
# ---------------------------------------------------------------------------

def test_board_has_121_cells(wrapper):
    assert len(wrapper.board.cells) == 121


# ---------------------------------------------------------------------------
# Initial placement
# ---------------------------------------------------------------------------

def test_initial_pieces_count_red(wrapper):
    pieces = wrapper.get_pieces('red')
    assert len(pieces) == 10


def test_initial_pieces_count_blue(wrapper):
    pieces = wrapper.get_pieces('blue')
    assert len(pieces) == 10


def test_no_overlap_between_colours(wrapper):
    red_positions = {p['pos'] for p in wrapper.get_pieces('red')}
    blue_positions = {p['pos'] for p in wrapper.get_pieces('blue')}
    assert red_positions.isdisjoint(blue_positions)


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------

def test_legal_moves_returns_dict(wrapper):
    moves = wrapper.get_legal_moves('red')
    assert isinstance(moves, dict)


def test_legal_moves_non_empty(wrapper):
    moves = wrapper.get_legal_moves('red')
    assert len(moves) > 0, "Red should have at least one legal move at game start"


def test_legal_moves_destinations_are_lists(wrapper):
    moves = wrapper.get_legal_moves('red')
    for pin_id, dests in moves.items():
        assert isinstance(dests, list)
        assert len(dests) > 0


# ---------------------------------------------------------------------------
# apply_move
# ---------------------------------------------------------------------------

def test_apply_move_changes_position():
    """Create a fresh wrapper so we don't pollute the module-level fixture."""
    bw = BoardWrapper(['red', 'blue'])
    moves = bw.get_legal_moves('red')
    assert moves, "Need at least one legal move"

    # Pick the first movable pin and its first destination
    pin_id = next(iter(moves))
    dest = moves[pin_id][0]

    # Record original position
    orig_pos = next(p['pos'] for p in bw.get_pieces('red') if p['id'] == pin_id)
    assert orig_pos != dest  # sanity check

    success = bw.apply_move('red', pin_id, dest)
    assert success is True

    new_pos = next(p['pos'] for p in bw.get_pieces('red') if p['id'] == pin_id)
    assert new_pos == dest
    assert new_pos != orig_pos


def test_apply_move_returns_false_for_occupied():
    """Moving to an occupied cell should fail."""
    bw = BoardWrapper(['red', 'blue'])
    red_pieces = bw.get_pieces('red')
    blue_pieces = bw.get_pieces('blue')

    # Try to move a red pin to a cell that blue already occupies
    red_pin_id = red_pieces[0]['id']
    occupied_dest = blue_pieces[0]['pos']

    result = bw.apply_move('red', red_pin_id, occupied_dest)
    assert result is False


# ---------------------------------------------------------------------------
# Goal / home indices
# ---------------------------------------------------------------------------

def test_goal_indices_are_opposite_home(wrapper):
    """Red's goal should equal blue's home and vice-versa."""
    red_goal = set(wrapper.get_goal_indices('red'))
    blue_home = set(wrapper.get_home_indices('blue'))
    assert red_goal == blue_home


def test_home_indices_are_opposite_goal(wrapper):
    blue_goal = set(wrapper.get_goal_indices('blue'))
    red_home = set(wrapper.get_home_indices('red'))
    assert blue_goal == red_home


def test_home_indices_count(wrapper):
    assert len(wrapper.get_home_indices('red')) == 10
    assert len(wrapper.get_home_indices('blue')) == 10


def test_goal_indices_count(wrapper):
    assert len(wrapper.get_goal_indices('red')) == 10
    assert len(wrapper.get_goal_indices('blue')) == 10


# ---------------------------------------------------------------------------
# check_win
# ---------------------------------------------------------------------------

def test_check_win_initially_false(wrapper):
    assert wrapper.check_win('red') is False
    assert wrapper.check_win('blue') is False


def test_check_win_true_when_all_in_goal():
    """Manually teleport all red pins to blue's home cells and check win."""
    bw = BoardWrapper(['red', 'blue'])
    goal_indices = bw.get_goal_indices('red')

    # Clear occupied flag on old positions, set on goal positions
    for pin, goal_idx in zip(bw.pins['red'], goal_indices):
        bw.board.cells[pin.axialindex].occupied = False
        pin.axialindex = goal_idx
        bw.board.cells[goal_idx].occupied = True

    assert bw.check_win('red') is True


# ---------------------------------------------------------------------------
# axial_distance
# ---------------------------------------------------------------------------

def test_axial_distance_same_cell(wrapper):
    assert wrapper.axial_distance(0, 0) == 0


def test_axial_distance_adjacent_cells(wrapper):
    """Find two cells that are known neighbours and verify distance == 1."""
    board = wrapper.board
    # Take the first cell and look for a neighbour via index_of
    from itertools import islice
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    found = False
    for idx_a, cell in enumerate(board.cells):
        for dq, dr in directions:
            nq, nr = cell.q + dq, cell.r + dr
            idx_b = board.index_of.get((nq, nr))
            if idx_b is not None:
                assert wrapper.axial_distance(idx_a, idx_b) == 1
                found = True
                break
        if found:
            break
    assert found, "Should have found at least one pair of adjacent cells"


def test_axial_distance_symmetry(wrapper):
    idx_a, idx_b = 0, 5
    assert wrapper.axial_distance(idx_a, idx_b) == wrapper.axial_distance(idx_b, idx_a)


# ---------------------------------------------------------------------------
# total_distance_to_goal / pins_in_goal
# ---------------------------------------------------------------------------

def test_total_distance_to_goal_positive(wrapper):
    dist = wrapper.total_distance_to_goal('red')
    assert dist > 0


def test_pins_in_goal_initially_zero(wrapper):
    assert wrapper.pins_in_goal('red') == 0
    assert wrapper.pins_in_goal('blue') == 0


def test_pins_in_goal_after_win():
    bw = BoardWrapper(['red', 'blue'])
    goal_indices = bw.get_goal_indices('red')
    for pin, goal_idx in zip(bw.pins['red'], goal_indices):
        bw.board.cells[pin.axialindex].occupied = False
        pin.axialindex = goal_idx
        bw.board.cells[goal_idx].occupied = True
    assert bw.pins_in_goal('red') == 10


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------

def test_clone_is_independent():
    """Changes to the clone must not affect the original."""
    bw = BoardWrapper(['red', 'blue'])
    cloned = bw.clone()

    # Apply a move on the clone
    moves = cloned.get_legal_moves('red')
    assert moves
    pin_id = next(iter(moves))
    dest = moves[pin_id][0]
    cloned.apply_move('red', pin_id, dest)

    # The original should be unchanged
    orig_pos = next(p['pos'] for p in bw.get_pieces('red') if p['id'] == pin_id)
    clone_pos = next(p['pos'] for p in cloned.get_pieces('red') if p['id'] == pin_id)
    assert orig_pos != clone_pos


def test_clone_has_same_state():
    bw = BoardWrapper(['red', 'blue'])
    cloned = bw.clone()
    assert bw.get_pieces('red') == cloned.get_pieces('red')
    assert bw.get_pieces('blue') == cloned.get_pieces('blue')


# ---------------------------------------------------------------------------
# check_draw
# ---------------------------------------------------------------------------

def test_check_draw_initially_false(wrapper):
    """At the start of a game there should be legal moves for all colours."""
    assert wrapper.check_draw('red') is False
    assert wrapper.check_draw('blue') is False


# ---------------------------------------------------------------------------
# COLOUR_OPPOSITES constant
# ---------------------------------------------------------------------------

def test_colour_opposites_symmetric():
    for c, opp in COLOUR_OPPOSITES.items():
        assert COLOUR_OPPOSITES[opp] == c
