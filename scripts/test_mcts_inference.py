"""Test MCTS at inference time: PPO policy as priors, search at game time.

Compares:
  1. Pure PPO (no search)
  2. PPO + MCTS (heuristic value, N simulations)
  3. PPO + MCTS (network value, N simulations)
  4. Advanced heuristic (baseline)
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
parser.add_argument('--sims', type=int, default=50, help='MCTS simulations per move')
parser.add_argument('--steps', type=int, default=100, help='Max steps per game')
parser.add_argument('--games', type=int, default=3, help='Number of games')
args = parser.parse_args()

from sb3_contrib import MaskablePPO
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.search.mcts import MCTS

model = MaskablePPO.load(args.model)
mapper = ActionMapper(num_pins=10, num_cells=121)


def run_pure_ppo(num_games, max_steps):
    """Run games with pure PPO policy (no MCTS)."""
    results = []
    for g in range(num_games):
        env = ChineseCheckersEnv(opponent_policy=None, max_steps=max_steps)
        obs, _ = env.reset()
        t0 = time.time()
        for step in range(max_steps):
            mask = env.action_masks()
            action, _ = model.predict(
                np.expand_dims(obs, 0),
                action_masks=np.expand_dims(mask, 0),
                deterministic=True,
            )
            obs, _, term, trunc, _ = env.step(int(action[0]))
            if term or trunc:
                break
        elapsed = time.time() - t0
        pins = env._board.pins_in_goal(env._AGENT_COLOUR)
        dist = env._board.total_distance_to_goal(env._AGENT_COLOUR)
        results.append((pins, dist, elapsed))
        print(f"    Game {g}: pins={pins}, dist={dist:.0f}, {elapsed:.1f}s")
    return results


def run_mcts_inference(num_games, max_steps, num_sims, use_network_value=False):
    """Run games with PPO + MCTS at inference time."""
    mcts = MCTS(
        model,
        num_simulations=num_sims,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,  # No noise for tournament play
        use_network_value=use_network_value,
    )
    results = []
    for g in range(num_games):
        env = ChineseCheckersEnv(opponent_policy=None, max_steps=max_steps)
        obs, _ = env.reset()
        t0 = time.time()
        for step in range(max_steps):
            action = mcts.select_action(env, temperature=0.0)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
            if (step + 1) % 25 == 0:
                pins = env._board.pins_in_goal(env._AGENT_COLOUR)
                dist = env._board.total_distance_to_goal(env._AGENT_COLOUR)
                elapsed = time.time() - t0
                print(f"      step {step+1}: pins={pins}, dist={dist:.0f}, {elapsed:.1f}s")
        elapsed = time.time() - t0
        pins = env._board.pins_in_goal(env._AGENT_COLOUR)
        dist = env._board.total_distance_to_goal(env._AGENT_COLOUR)
        results.append((pins, dist, elapsed))
        print(f"    Game {g}: pins={pins}, dist={dist:.0f}, {elapsed:.1f}s")
    return results


print(f"Model: {args.model}")
print(f"Settings: {args.sims} sims, {args.steps} max steps, {args.games} games\n")

print(f"=== Pure PPO (no search) ===")
ppo_results = run_pure_ppo(args.games, args.steps)
avg_pins = np.mean([r[0] for r in ppo_results])
avg_dist = np.mean([r[1] for r in ppo_results])
avg_time = np.mean([r[2] for r in ppo_results])
print(f"  Avg: pins={avg_pins:.1f}, dist={avg_dist:.0f}, time={avg_time:.1f}s\n")

print(f"=== PPO + MCTS ({args.sims} sims, heuristic value) ===")
mcts_h_results = run_mcts_inference(args.games, args.steps, args.sims, use_network_value=False)
avg_pins = np.mean([r[0] for r in mcts_h_results])
avg_dist = np.mean([r[1] for r in mcts_h_results])
avg_time = np.mean([r[2] for r in mcts_h_results])
print(f"  Avg: pins={avg_pins:.1f}, dist={avg_dist:.0f}, time={avg_time:.1f}s\n")

print(f"=== PPO + MCTS ({args.sims} sims, network value) ===")
mcts_n_results = run_mcts_inference(args.games, args.steps, args.sims, use_network_value=True)
avg_pins = np.mean([r[0] for r in mcts_n_results])
avg_dist = np.mean([r[1] for r in mcts_n_results])
avg_time = np.mean([r[2] for r in mcts_n_results])
print(f"  Avg: pins={avg_pins:.1f}, dist={avg_dist:.0f}, time={avg_time:.1f}s\n")

print("=== Summary ===")
print(f"{'Method':<35} {'Avg Pins':>10} {'Avg Dist':>10} {'Avg Time':>10}")
print("-" * 65)
for name, results in [
    ("Pure PPO", ppo_results),
    (f"PPO + MCTS {args.sims}sim (heuristic)", mcts_h_results),
    (f"PPO + MCTS {args.sims}sim (network)", mcts_n_results),
]:
    avg_p = np.mean([r[0] for r in results])
    avg_d = np.mean([r[1] for r in results])
    avg_t = np.mean([r[2] for r in results])
    print(f"{name:<35} {avg_p:>10.1f} {avg_d:>10.0f} {avg_t:>10.1f}s")
