"""Run baseline evaluations: random vs random, greedy vs random, etc."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multi system single machine minimal'))

from src.agents.random_agent import random_policy
from src.agents.greedy_agent import greedy_policy
from src.evaluation.arena import run_arena, arena_summary


def print_summary(name, s):
    print(f"\n--- {name} ---")
    print(f"  Games: {s['num_games']}")
    print(f"  Agent wins: {s['agent_wins']} ({s['win_rate']:.1%})")
    print(f"  Opponent wins: {s['opponent_wins']}")
    print(f"  Draws: {s['draws']}, Truncated: {s['truncated']}")
    print(f"  Avg steps: {s['avg_steps']:.1f}")
    print(f"  Avg pins in goal: {s['avg_pins_in_goal']:.1f}")
    print(f"  Avg tournament score: {s['avg_tournament_score']:.1f}")


def main():
    num_games = 20
    max_steps = 200

    print("=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    matchups = [
        ("Random vs Random", random_policy, random_policy),
        ("Greedy (agent) vs Random (opponent)", greedy_policy, random_policy),
        ("Random (agent) vs Greedy (opponent)", random_policy, greedy_policy),
        ("Greedy vs Greedy", greedy_policy, greedy_policy),
    ]

    for name, agent, opponent in matchups:
        results = run_arena(agent, opponent, num_games=num_games, max_steps=max_steps)
        summary = arena_summary(results)
        print_summary(name, summary)

    print("\n" + "=" * 60)
    print("DONE")


if __name__ == "__main__":
    main()
