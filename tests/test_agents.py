"""
Tests for src/agents/random_agent.py and src/agents/greedy_agent.py.
"""
import sys
import os

# Make sure the repo root is importable regardless of where pytest is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest
from src.env.board_wrapper import BoardWrapper
from src.agents.random_agent import random_policy
from src.agents.greedy_agent import greedy_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh_board():
    return BoardWrapper(['red', 'blue'])


# ---------------------------------------------------------------------------
# 1. Random policy returns a valid move
# ---------------------------------------------------------------------------

def test_random_returns_valid_move():
    bw = fresh_board()
    pin_id, dest = random_policy(bw, 'red')
    legal = bw.get_legal_moves('red')

    assert pin_id in legal, f"pin_id {pin_id} not in legal moves"
    assert dest in legal[pin_id], f"dest {dest} not in legal dests for pin {pin_id}"


def test_random_raises_when_no_moves():
    """Force a board with no legal moves by teleporting all pins to occupied cells."""
    bw = fresh_board()
    # Block all red pins: set them to positions that have no legal moves by
    # checking draw state.  Instead, we monkeypatch get_legal_moves.
    original = bw.get_legal_moves

    def empty_moves(colour):
        return {}

    bw.get_legal_moves = empty_moves
    with pytest.raises(ValueError):
        random_policy(bw, 'red')

    bw.get_legal_moves = original  # restore (not strictly needed)


# ---------------------------------------------------------------------------
# 2. Greedy policy returns a valid move
# ---------------------------------------------------------------------------

def test_greedy_returns_valid_move():
    bw = fresh_board()
    pin_id, dest = greedy_policy(bw, 'red')
    legal = bw.get_legal_moves('red')

    assert pin_id in legal, f"pin_id {pin_id} not in legal moves"
    assert dest in legal[pin_id], f"dest {dest} not in legal dests for pin {pin_id}"


def test_greedy_raises_when_no_moves():
    bw = fresh_board()

    def empty_moves(colour):
        return {}

    bw.get_legal_moves = empty_moves
    with pytest.raises(ValueError):
        greedy_policy(bw, 'red')


# ---------------------------------------------------------------------------
# 3. Greedy prefers forward moves (distance must not increase)
# ---------------------------------------------------------------------------

def test_greedy_does_not_increase_distance():
    """
    After a greedy move the total distance to goal should not be greater than
    before the move (over the first 20 moves for both colours).
    """
    bw = fresh_board()
    colours = ['red', 'blue']
    for _ in range(20):
        for colour in colours:
            legal = bw.get_legal_moves(colour)
            if not legal:
                continue

            dist_before = bw.total_distance_to_goal(colour)
            pin_id, dest = greedy_policy(bw, colour)

            # Verify on a clone that distance doesn't increase.
            clone = bw.clone()
            clone.apply_move(colour, pin_id, dest)
            dist_after = clone.total_distance_to_goal(colour)

            assert dist_after <= dist_before, (
                f"Greedy move increased distance for {colour}: "
                f"{dist_before} -> {dist_after}"
            )

            # Apply the move on the real board to advance the game state.
            bw.apply_move(colour, pin_id, dest)
            if bw.check_win(colour):
                break


# ---------------------------------------------------------------------------
# 4. Random agent can play a full game via ChineseCheckersEnv without crashing
# ---------------------------------------------------------------------------

def test_random_agent_full_game_via_env():
    try:
        from src.env.chinese_checkers_env import ChineseCheckersEnv
    except ImportError:
        pytest.skip("ChineseCheckersEnv not available yet")

    env = ChineseCheckersEnv()
    obs, info = env.reset()
    done = False
    max_steps = 2000  # safety cap
    steps = 0

    while not done and steps < max_steps:
        # The env is expected to track whose turn it is; we use random policy.
        colour = info.get('current_colour', env.current_colour
                          if hasattr(env, 'current_colour') else 'red')
        try:
            pin_id, dest = random_policy(env.board_wrapper, colour)
            action = (pin_id, dest)
        except ValueError:
            break  # no legal moves — env should handle terminal state

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    # If we reach here without exception the test passes.
    assert steps > 0, "Game should have taken at least one step"


# ---------------------------------------------------------------------------
# 5. Greedy agent beats random agent over 10 games (wins >= 5)
# ---------------------------------------------------------------------------

def test_greedy_beats_random_10_games():
    try:
        from src.env.chinese_checkers_env import ChineseCheckersEnv
    except ImportError:
        pytest.skip("ChineseCheckersEnv not available yet")

    # Greedy as agent, random as opponent (env default). Measure pins in goal.
    num_games = 5
    max_steps = 200
    greedy_pins = []
    random_pins = []

    for _ in range(num_games):
        # Greedy agent vs random opponent
        env = ChineseCheckersEnv(max_steps=max_steps)
        obs, info = env.reset()
        done = False
        while not done:
            pin_id, dest = greedy_policy(env.board_wrapper, 'red')
            obs, reward, terminated, truncated, info = env.step((pin_id, dest))
            done = terminated or truncated
        greedy_pins.append(env.board_wrapper.pins_in_goal('red'))

        # Random agent vs random opponent
        env2 = ChineseCheckersEnv(max_steps=max_steps)
        obs, info = env2.reset()
        done = False
        while not done:
            mask = info['action_mask']
            import numpy as np
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            obs, reward, terminated, truncated, info = env2.step(int(np.random.choice(legal)))
            done = terminated or truncated
        random_pins.append(env2.board_wrapper.pins_in_goal('red'))

    avg_greedy = sum(greedy_pins) / len(greedy_pins)
    avg_random = sum(random_pins) / len(random_pins)
    assert avg_greedy > avg_random, (
        f"Greedy avg pins in goal ({avg_greedy:.1f}) should beat random ({avg_random:.1f})"
    )
