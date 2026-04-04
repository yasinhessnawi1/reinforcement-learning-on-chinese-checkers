"""Debug arena: test heuristic agents to verify arena is working."""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTI_SYSTEM_DIR = os.path.join(PROJECT_ROOT, 'multi system single machine minimal')
for p in [PROJECT_ROOT, MULTI_SYSTEM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evaluation.arena import run_arena, arena_summary
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy


def random_policy(board_wrapper, colour):
    legal_moves = board_wrapper.get_legal_moves(colour)
    pin_ids = list(legal_moves.keys())
    pin_id = pin_ids[np.random.randint(len(pin_ids))]
    dests = legal_moves[pin_id]
    dest = dests[np.random.randint(len(dests))]
    return (pin_id, dest)


print("=== Test 1: Advanced vs Random (5 games, max_steps=1000) ===")
results = run_arena(advanced_heuristic_policy, random_policy, num_games=5, max_steps=1000)
s = arena_summary(results)
print(f"  Wins: {s['agent_wins']}, Pins: {s['avg_pins_in_goal']:.1f}, Score: {s['avg_tournament_score']:.1f}")
for i, r in enumerate(results):
    print(f"  Game {i}: winner={r['winner']}, steps={r['steps']}, pins={r['pins_in_goal']}, dist={r['distance_to_goal']:.0f}")

print("\n=== Test 2: Advanced vs Greedy (5 games, max_steps=1000) ===")
results2 = run_arena(advanced_heuristic_policy, greedy_policy, num_games=5, max_steps=1000)
s2 = arena_summary(results2)
print(f"  Wins: {s2['agent_wins']}, Pins: {s2['avg_pins_in_goal']:.1f}, Score: {s2['avg_tournament_score']:.1f}")
for i, r in enumerate(results2):
    print(f"  Game {i}: winner={r['winner']}, steps={r['steps']}, pins={r['pins_in_goal']}, dist={r['distance_to_goal']:.0f}")

print("\n=== Test 3: Greedy vs Random (5 games, max_steps=1000) ===")
results3 = run_arena(greedy_policy, random_policy, num_games=5, max_steps=1000)
s3 = arena_summary(results3)
print(f"  Wins: {s3['agent_wins']}, Pins: {s3['avg_pins_in_goal']:.1f}, Score: {s3['avg_tournament_score']:.1f}")
for i, r in enumerate(results3):
    print(f"  Game {i}: winner={r['winner']}, steps={r['steps']}, pins={r['pins_in_goal']}, dist={r['distance_to_goal']:.0f}")
