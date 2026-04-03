"""
Arena evaluation script for PPO models vs Greedy and Random baselines.

Runs 20 games per matchup with max_steps=1000.
"""

import sys
import os
import numpy as np

# Add project root and multi system folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTI_SYSTEM_DIR = os.path.join(PROJECT_ROOT, 'multi system single machine minimal')
for p in [PROJECT_ROOT, MULTI_SYSTEM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sb3_contrib import MaskablePPO

from src.evaluation.arena import run_arena, arena_summary
from src.agents.greedy_agent import greedy_policy
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.env.state_encoder import StateEncoder
from src.env.board_wrapper import BoardWrapper


def make_ppo_policy(model_path: str):
    """Create a policy function from a MaskablePPO model.

    The returned callable has signature: policy(board_wrapper, colour) -> (pin_id, dest).
    """
    model = MaskablePPO.load(model_path)
    encoder = StateEncoder(grid_size=17, num_channels=10)
    mapper = ActionMapper(num_pins=10, num_cells=121)
    turn_order = ["red", "blue"]

    def policy(board_wrapper, colour):
        # Build observation
        obs = encoder.encode(board_wrapper, current_colour=colour, turn_order=turn_order)

        # Build action mask
        legal_moves = board_wrapper.get_legal_moves(colour)
        action_mask = mapper.build_action_mask(legal_moves)

        # Predict with masking (deterministic)
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
        action = int(action)

        pin_id, dest = mapper.decode(action)
        return (pin_id, dest)

    return policy


def random_policy(board_wrapper, colour):
    """Uniform random legal move policy."""
    legal_moves = board_wrapper.get_legal_moves(colour)
    if not legal_moves:
        raise ValueError(f"No legal moves for {colour}")
    pin_ids = list(legal_moves.keys())
    pin_id = pin_ids[np.random.randint(len(pin_ids))]
    dests = legal_moves[pin_id]
    dest = dests[np.random.randint(len(dests))]
    return (pin_id, dest)


def print_results_table(all_results: list):
    """Print results as a formatted table."""
    header = (
        f"{'Matchup':<35} {'Wins':>5} {'Losses':>7} {'Draws':>6} "
        f"{'Trunc':>6} {'WinRate':>8} {'AvgSteps':>9} {'AvgPins':>8} {'AvgScore':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for label, summary in all_results:
        print(
            f"{label:<35} {summary['agent_wins']:>5} {summary['opponent_wins']:>7} "
            f"{summary['draws']:>6} {summary['truncated']:>6} "
            f"{summary['win_rate']:>8.1%} {summary['avg_steps']:>9.1f} "
            f"{summary['avg_pins_in_goal']:>8.1f} {summary['avg_tournament_score']:>9.1f}"
        )

    print("=" * len(header))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Arena evaluation for PPO models.")
    parser.add_argument('--model', type=str, required=True, help='Path to MaskablePPO .zip model')
    parser.add_argument('--num-games', type=int, default=20, help='Games per matchup (default: 20)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per game (default: 1000)')
    args = parser.parse_args()

    from src.agents.advanced_heuristic import advanced_heuristic_policy

    num_games = args.num_games
    max_steps = args.max_steps

    print(f"Loading model from: {args.model}")
    model_policy = make_ppo_policy(args.model)

    matchups = [
        ("Model vs Greedy", model_policy, greedy_policy),
        ("Model vs Random", model_policy, random_policy),
        ("Model vs Advanced", model_policy, advanced_heuristic_policy),
        ("Greedy vs Random", greedy_policy, random_policy),
    ]

    all_results = []
    for label, agent, opponent in matchups:
        print(f"\nRunning: {label} ({num_games} games, max_steps={max_steps})...")
        results = run_arena(agent, opponent, num_games=num_games, max_steps=max_steps)
        summary = arena_summary(results)
        all_results.append((label, summary))
        print(f"  -> Win rate: {summary['win_rate']:.1%}, "
              f"Avg steps: {summary['avg_steps']:.1f}, "
              f"Avg pins in goal: {summary['avg_pins_in_goal']:.1f}")

    print_results_table(all_results)


if __name__ == "__main__":
    main()
