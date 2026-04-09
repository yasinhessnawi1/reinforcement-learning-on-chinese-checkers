"""Arena evaluation: hybrid agent that combines advanced heuristic with MCTS.

Strategy: Use advanced heuristic as the primary policy (proven 8 pins vs greedy).
Use MCTS to search for better moves when time permits. If MCTS finds a move
with significantly higher visit count than the heuristic's choice, use it.
"""
import sys
import os
import time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTI_SYSTEM_DIR = os.path.join(PROJECT_ROOT, 'multi system single machine minimal')
for p in [PROJECT_ROOT, MULTI_SYSTEM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--sims', type=int, default=50)
parser.add_argument('--num-games', type=int, default=5)
parser.add_argument('--max-steps', type=int, default=1000)
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


def make_hybrid_policy(num_sims, opponent_model=None):
    """Advanced heuristic + MCTS override when search finds better move."""
    mcts = MCTS(
        model,
        num_simulations=num_sims,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        use_network_value=False,
        opponent_policy=opponent_model,
    )

    def policy(board_wrapper, colour):
        # 1. Get advanced heuristic's choice (fast, proven)
        heuristic_move = advanced_heuristic_policy(board_wrapper, colour)
        heuristic_action = mapper.encode(heuristic_move[0], heuristic_move[1])

        # 2. Run MCTS search
        env = ChineseCheckersEnv(opponent_policy=None, max_steps=1000)
        env.reset()
        env._board = board_wrapper
        env._no_opponent = True

        root = mcts.run(env)

        # 3. Compare: does MCTS strongly prefer a different move?
        best_mcts_action = max(root.children, key=lambda a: root.children[a].N)
        best_mcts_visits = root.children[best_mcts_action].N
        heuristic_visits = root.children.get(heuristic_action, None)
        heuristic_v = heuristic_visits.N if heuristic_visits else 0

        # If MCTS strongly agrees with heuristic or heuristic has >30% of best visits, keep it
        # Otherwise use MCTS's choice
        if heuristic_v >= best_mcts_visits * 0.3:
            return heuristic_move
        else:
            pin_id, dest = mapper.decode(best_mcts_action)
            return (pin_id, dest)

    return policy


def make_pure_mcts_policy(num_sims, opponent_model=None):
    """Pure MCTS policy."""
    mcts = MCTS(
        model,
        num_simulations=num_sims,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        use_network_value=False,
        opponent_policy=opponent_model,
    )

    def policy(board_wrapper, colour):
        env = ChineseCheckersEnv(opponent_policy=None, max_steps=1000)
        env.reset()
        env._board = board_wrapper
        env._no_opponent = True
        action = mcts.select_action(env, temperature=0.0)
        pin_id, dest = mapper.decode(action)
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
print(f"MCTS: {args.sims} sims")
print(f"Games: {args.num_games} per matchup, max_steps={args.max_steps}\n")

hybrid_2p = make_hybrid_policy(args.sims, opponent_model=greedy_policy)
mcts_2p = make_pure_mcts_policy(args.sims, opponent_model=greedy_policy)
mcts_solo = make_pure_mcts_policy(args.sims, opponent_model=None)

matchups = [
    ("Hybrid-2p vs Greedy", hybrid_2p, greedy_policy),
    ("Hybrid-2p vs Random", hybrid_2p, random_policy),
    ("MCTS-2p vs Greedy", mcts_2p, greedy_policy),
    ("MCTS-solo vs Greedy", mcts_solo, greedy_policy),
    ("Advanced vs Greedy", advanced_heuristic_policy, greedy_policy),
    ("Advanced vs Random", advanced_heuristic_policy, random_policy),
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
