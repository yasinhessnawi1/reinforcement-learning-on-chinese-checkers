"""Tests for evaluation framework: arena and Elo tracker."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multi system single machine minimal'))

from src.evaluation.arena import play_game, run_arena, arena_summary
from src.evaluation.evaluator import compute_elo_update, EloTracker
from src.agents.random_agent import random_policy
from src.agents.greedy_agent import greedy_policy


# --- Arena tests ---

def test_play_game_returns_result():
    result = play_game(random_policy, random_policy, max_steps=50)
    assert 'winner' in result
    assert 'steps' in result
    assert 'agent_score' in result
    assert result['winner'] in ('agent', 'opponent', 'draw', 'truncated')


def test_play_game_result_types():
    result = play_game(random_policy, random_policy, max_steps=50)
    assert isinstance(result['steps'], int)
    assert isinstance(result['total_reward'], float)
    assert isinstance(result['agent_score'], float)
    assert isinstance(result['pins_in_goal'], int)


def test_run_arena_count():
    results = run_arena(random_policy, random_policy, num_games=3, max_steps=50)
    assert len(results) == 3


def test_arena_summary_keys():
    results = run_arena(random_policy, random_policy, num_games=5, max_steps=50)
    summary = arena_summary(results)
    assert 'num_games' in summary
    assert 'agent_wins' in summary
    assert 'opponent_wins' in summary
    assert 'draws' in summary
    assert 'truncated' in summary
    assert 'win_rate' in summary
    assert 'avg_steps' in summary
    assert 'avg_pins_in_goal' in summary
    assert 'avg_tournament_score' in summary


def test_arena_summary_values():
    results = run_arena(random_policy, random_policy, num_games=5, max_steps=50)
    summary = arena_summary(results)
    assert summary['num_games'] == 5
    total = summary['agent_wins'] + summary['opponent_wins'] + summary['draws'] + summary['truncated']
    assert total == 5
    assert 0.0 <= summary['win_rate'] <= 1.0


def test_greedy_vs_random_arena():
    results = run_arena(greedy_policy, random_policy, num_games=5, max_steps=100)
    summary = arena_summary(results)
    # Greedy should at least get some pins in goal
    assert summary['avg_pins_in_goal'] >= 0


# --- Elo tests ---

def test_elo_winner_gains():
    new_a, new_b = compute_elo_update(1200, 1200, winner='a')
    assert new_a > 1200
    assert new_b < 1200


def test_elo_loser_loses():
    new_a, new_b = compute_elo_update(1200, 1200, winner='b')
    assert new_a < 1200
    assert new_b > 1200


def test_elo_draw_equal_ratings():
    new_a, new_b = compute_elo_update(1200, 1200, winner='draw')
    # Draw between equal players: no change
    assert abs(new_a - 1200) < 0.01
    assert abs(new_b - 1200) < 0.01


def test_elo_draw_unequal_ratings():
    new_a, new_b = compute_elo_update(1400, 1200, winner='draw')
    # Higher rated player loses rating on draw, lower gains
    assert new_a < 1400
    assert new_b > 1200


def test_elo_tracker_initial():
    tracker = EloTracker()
    assert tracker.get_rating('agent1') == 1200


def test_elo_tracker_record_game():
    tracker = EloTracker()
    new_a, new_b = tracker.record_game('agent1', 'agent2', 'a')
    assert tracker.get_rating('agent1') > 1200
    assert tracker.get_rating('agent2') < 1200
    assert len(tracker.history) == 1


def test_elo_tracker_multiple_games():
    tracker = EloTracker()
    tracker.record_game('a', 'b', 'a')
    tracker.record_game('a', 'b', 'a')
    tracker.record_game('a', 'b', 'a')
    assert tracker.get_rating('a') > tracker.get_rating('b')
    assert len(tracker.history) == 3
