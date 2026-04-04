"""Arena evaluation with MCTS inference agent.

The MCTS agent uses:
  - PPO policy network as priors (guides tree search)
  - Heuristic position evaluation (leaf values)
  - Min-max Q-value normalization
"""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTI_SYSTEM_DIR = os.path.join(PROJECT_ROOT, 'multi system single machine minimal')
for p in [PROJECT_ROOT, MULTI_SYSTEM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--sims', type=int, default=50, help='MCTS simulations per move')
parser.add_argument('--num-games', type=int, default=5, help='Games per matchup')
parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per game')
args = parser.parse_args()

from sb3_contrib import MaskablePPO
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.env.state_encoder import StateEncoder
from src.search.mcts import MCTS
from src.evaluation.arena import run_arena, arena_summary
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy

model = MaskablePPO.load(args.model)
mapper = ActionMapper(num_pins=10, num_cells=121)
encoder = StateEncoder(grid_size=17, num_channels=10)


def make_mcts_policy(num_sims):
    """Create an MCTS inference policy that works with the arena."""
    mcts = MCTS(
        model,
        num_simulations=num_sims,
        c_puct=1.5,
        dirichlet_epsilon=0.0,  # No noise for tournament
        use_network_value=False,  # Use heuristic value
    )

    def policy(board_wrapper, colour):
        # Build a temporary env matching the current board state
        # The arena calls policy(board_wrapper, colour) but MCTS needs a full env.
        # We create a minimal env and inject the board state.
        env = ChineseCheckersEnv(opponent_policy=None, max_steps=1000)
        env.reset()
        env._board = board_wrapper
        env._no_opponent = True

        action = mcts.select_action(env, temperature=0.0)
        pin_id, dest = mapper.decode(action)
        return (pin_id, dest)

    return policy


def make_ppo_policy():
    """Pure PPO policy (no MCTS)."""
    turn_order = ["red", "blue"]

    def policy(board_wrapper, colour):
        obs = encoder.encode(board_wrapper, current_colour=colour, turn_order=turn_order)
        legal_moves = board_wrapper.get_legal_moves(colour)
        action_mask = mapper.build_action_mask(legal_moves)
        action, _ = model.predict(
            np.expand_dims(obs, 0),
            action_masks=np.expand_dims(action_mask, 0),
            deterministic=True,
        )
        pin_id, dest = mapper.decode(int(action[0]))
        return (pin_id, dest)

    return policy


def random_policy(board_wrapper, colour):
    legal_moves = board_wrapper.get_legal_moves(colour)
    pin_ids = list(legal_moves.keys())
    pin_id = pin_ids[np.random.randint(len(pin_ids))]
    dests = legal_moves[pin_id]
    dest = dests[np.random.randint(len(dests))]
    return (pin_id, dest)


print(f"Model: {args.model}")
print(f"MCTS: {args.sims} sims, heuristic value, min-max normalization")
print(f"Games: {args.num_games} per matchup, max_steps={args.max_steps}\n")

mcts_policy = make_mcts_policy(args.sims)
ppo_policy = make_ppo_policy()

matchups = [
    ("MCTS vs Random", mcts_policy, random_policy),
    ("MCTS vs Greedy", mcts_policy, greedy_policy),
    ("MCTS vs Advanced", mcts_policy, advanced_heuristic_policy),
    ("Pure PPO vs Random", ppo_policy, random_policy),
    ("Pure PPO vs Greedy", ppo_policy, greedy_policy),
    ("Advanced vs Random", advanced_heuristic_policy, random_policy),
    ("Advanced vs Greedy", advanced_heuristic_policy, greedy_policy),
]

all_results = []
for label, agent, opponent in matchups:
    print(f"Running: {label} ({args.num_games} games)...")
    results = run_arena(agent, opponent, num_games=args.num_games, max_steps=args.max_steps)
    s = arena_summary(results)
    all_results.append((label, s))
    print(f"  Wins: {s['agent_wins']}, Pins: {s['avg_pins_in_goal']:.1f}, "
          f"Score: {s['avg_tournament_score']:.0f}\n")

print("\n" + "=" * 85)
print(f"{'Matchup':<30} {'Wins':>5} {'Pins':>6} {'Score':>7} {'AvgSteps':>9}")
print("=" * 85)
for label, s in all_results:
    print(f"{label:<30} {s['agent_wins']:>5} {s['avg_pins_in_goal']:>6.1f} "
          f"{s['avg_tournament_score']:>7.0f} {s['avg_steps']:>9.0f}")
print("=" * 85)
