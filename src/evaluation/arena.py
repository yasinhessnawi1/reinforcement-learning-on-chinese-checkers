"""
arena.py — Game-play orchestration for evaluating agents in Chinese Checkers.

Provides play_game(), run_arena(), and arena_summary() for head-to-head
evaluation of two policies.
"""

import math
from src.env.chinese_checkers_env import ChineseCheckersEnv


def play_game(agent_policy, opponent_policy, max_steps=1000) -> dict:
    """Play one full game between agent_policy and opponent_policy.

    Parameters
    ----------
    agent_policy : callable(board_wrapper, colour) -> (pin_id, dest)
        The agent being evaluated; always plays as 'red'.
    opponent_policy : callable(board_wrapper, colour) -> (pin_id, dest)
        The opponent; always plays as 'blue'.
    max_steps : int
        Episode truncation limit.

    Returns
    -------
    dict with keys:
        winner          : 'agent' | 'opponent' | 'draw' | 'truncated'
        steps           : int
        total_reward    : float
        agent_score     : float   (approximate tournament score)
        pins_in_goal    : int
        distance_to_goal: float
    """
    env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=max_steps)
    obs, info = env.reset()

    total_reward = 0.0
    done = False

    while not done:
        board_wrapper = env.board_wrapper
        pin_id, dest = agent_policy(board_wrapper, env._AGENT_COLOUR)
        obs, reward, terminated, truncated, info = env.step((pin_id, dest))
        total_reward += reward
        done = terminated or truncated

    # Determine winner
    board_wrapper = env.board_wrapper
    steps = env._step_count

    if terminated:
        if board_wrapper.check_win(env._AGENT_COLOUR):
            winner = 'agent'
        elif board_wrapper.check_win(env._OPPONENT_COLOUR):
            winner = 'opponent'
        else:
            winner = 'draw'
    else:
        winner = 'truncated'

    pins_in_goal = board_wrapper.pins_in_goal(env._AGENT_COLOUR)
    distance_to_goal = board_wrapper.total_distance_to_goal(env._AGENT_COLOUR)

    # Approximate tournament score
    sigma = 4 if steps < 45 else 18
    agent_score = (
        pins_in_goal * 100
        + max(0, 200 - distance_to_goal)
        + math.exp(-((steps - 45) ** 2) / (2 * (sigma ** 2)))
    )

    return {
        'winner': winner,
        'steps': steps,
        'total_reward': total_reward,
        'agent_score': agent_score,
        'pins_in_goal': pins_in_goal,
        'distance_to_goal': distance_to_goal,
    }


def run_arena(agent_policy, opponent_policy, num_games=100, max_steps=1000) -> list:
    """Run multiple games and return a list of result dicts.

    Parameters
    ----------
    agent_policy : callable
    opponent_policy : callable
    num_games : int
    max_steps : int

    Returns
    -------
    list of dicts, each from play_game()
    """
    results = []
    for _ in range(num_games):
        result = play_game(agent_policy, opponent_policy, max_steps=max_steps)
        results.append(result)
    return results


def arena_summary(results: list) -> dict:
    """Compute aggregate statistics from a list of play_game() result dicts.

    Parameters
    ----------
    results : list of dicts

    Returns
    -------
    dict with keys:
        num_games, agent_wins, opponent_wins, draws, truncated,
        win_rate, avg_steps, avg_pins_in_goal, avg_tournament_score
    """
    num_games = len(results)
    if num_games == 0:
        return {
            'num_games': 0,
            'agent_wins': 0,
            'opponent_wins': 0,
            'draws': 0,
            'truncated': 0,
            'win_rate': 0.0,
            'avg_steps': 0.0,
            'avg_pins_in_goal': 0.0,
            'avg_tournament_score': 0.0,
        }

    agent_wins = sum(1 for r in results if r['winner'] == 'agent')
    opponent_wins = sum(1 for r in results if r['winner'] == 'opponent')
    draws = sum(1 for r in results if r['winner'] == 'draw')
    truncated = sum(1 for r in results if r['winner'] == 'truncated')

    win_rate = agent_wins / num_games
    avg_steps = sum(r['steps'] for r in results) / num_games
    avg_pins_in_goal = sum(r['pins_in_goal'] for r in results) / num_games
    avg_tournament_score = sum(r['agent_score'] for r in results) / num_games

    return {
        'num_games': num_games,
        'agent_wins': agent_wins,
        'opponent_wins': opponent_wins,
        'draws': draws,
        'truncated': truncated,
        'win_rate': win_rate,
        'avg_steps': avg_steps,
        'avg_pins_in_goal': avg_pins_in_goal,
        'avg_tournament_score': avg_tournament_score,
    }
