"""
mcts_self_play.py — Parallel MCTS self-play data generator.

Generates (state, policy, value) training triples by running MCTS-guided
self-play games.  Each game produces one triple per step.

Usage
-----
    python src/training/mcts_self_play.py \\
        --model models/exp005/best/best_model.zip \\
        --num-games 100 \\
        --simulations 200 \\
        --out data/mcts_games_001.npz

Output format (saved as .npz):
    states  : float32 (N, 10, 17, 17)
    policies: float32 (N, 1210)
    values  : float32 (N,)
    game_ids: int32   (N,)  — which game each sample came from

Design decisions:
  - Sequential game generation (parallelism via multiprocessing in orchestrator)
  - Temperature 1.0 for first 20 moves, then 0.1 (anneal toward deterministic)
  - Outcome-based value: +1 if game won, -1 if lost/truncated
  - Symmetry augmentation: each sample also generates its mirror image (doubles data)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal'))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import MCTS
from src.training.symmetry import ReflectionSymmetry

# After this many moves in a game, reduce temperature for sharper play
_TEMP_THRESHOLD = 20
_HIGH_TEMP = 1.0
_LOW_TEMP = 0.1


def mask_fn(env):
    return env.action_masks()


def _play_game(
    model,
    mcts: MCTS,
    sym: ReflectionSymmetry,
    max_steps: int = 1000,
    use_symmetry: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play a single self-play game and return training triples.

    Returns
    -------
    List of (obs, action_probs, outcome) tuples.
    outcome is filled in retrospectively after the game ends.
    """
    env = ChineseCheckersEnv(opponent_policy=None, max_steps=max_steps)
    obs, info = env.reset()

    step_data: list[tuple[np.ndarray, np.ndarray]] = []  # (obs, probs) before outcome known
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        temp = _HIGH_TEMP if step < _TEMP_THRESHOLD else _LOW_TEMP

        # MCTS needs env.clone() to work — we pass the live env
        action_probs = mcts.get_action_probs(env, temperature=temp)

        step_data.append((obs.copy(), action_probs.copy()))

        # Sample action from MCTS distribution
        action = int(np.random.choice(len(action_probs), p=action_probs))
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

    # Determine outcome
    agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board is not None else False
    outcome = 1.0 if agent_won else -1.0

    # Build triples
    triples = []
    for s_obs, s_probs in step_data:
        triples.append((s_obs, s_probs, outcome))
        # Symmetry augmentation: mirror image
        if use_symmetry:
            mirror_obs = sym.reflect_obs(s_obs)
            mirror_probs = sym.reflect_action_mask(s_probs.astype(bool)).astype(np.float32)
            # Renormalise mirror probs
            total = mirror_probs.sum()
            if total > 0:
                mirror_probs = mirror_probs / total
            else:
                mirror_probs = s_probs.copy()  # fallback
            triples.append((mirror_obs, mirror_probs, outcome))

    return triples


def generate_games(
    model,
    num_games: int = 100,
    num_simulations: int = 200,
    max_steps: int = 1000,
    use_symmetry: bool = True,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> dict[str, np.ndarray]:
    """Generate self-play games and return a dataset dict.

    Parameters
    ----------
    model : MaskablePPO
    num_games : int
    num_simulations : int — MCTS simulations per move
    max_steps : int
    use_symmetry : bool — double data with reflection augmentation
    c_puct : float
    dirichlet_alpha : float
    dirichlet_epsilon : float

    Returns
    -------
    dict with keys 'states', 'policies', 'values', 'game_ids'
    """
    mcts = MCTS(
        model,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    sym = ReflectionSymmetry()

    all_states = []
    all_policies = []
    all_values = []
    all_game_ids = []

    for game_idx in range(num_games):
        triples = _play_game(model, mcts, sym, max_steps=max_steps, use_symmetry=use_symmetry)
        for obs, probs, value in triples:
            all_states.append(obs)
            all_policies.append(probs)
            all_values.append(value)
            all_game_ids.append(game_idx)

        if (game_idx + 1) % 10 == 0:
            print(f"  Game {game_idx + 1}/{num_games} — {len(all_states)} samples so far")

    return {
        'states':   np.array(all_states,   dtype=np.float32),
        'policies': np.array(all_policies, dtype=np.float32),
        'values':   np.array(all_values,   dtype=np.float32),
        'game_ids': np.array(all_game_ids, dtype=np.int32),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate MCTS self-play training data.")
    parser.add_argument('--model', type=str, required=True, help='Path to MaskablePPO .zip checkpoint')
    parser.add_argument('--num-games', type=int, default=100, help='Number of self-play games (default: 100)')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move (default: 200)')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max game steps (default: 1000)')
    parser.add_argument('--out', type=str, default='data/mcts_games.npz', help='Output .npz file path')
    parser.add_argument('--no-symmetry', action='store_true', help='Disable reflection augmentation')
    parser.add_argument('--c-puct', type=float, default=1.5)
    parser.add_argument('--dirichlet-alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.25)
    args = parser.parse_args(argv)

    print(f"Loading model from {args.model}")
    model = MaskablePPO.load(args.model)

    print(f"Generating {args.num_games} games with {args.simulations} simulations/move ...")
    dataset = generate_games(
        model,
        num_games=args.num_games,
        num_simulations=args.simulations,
        max_steps=args.max_steps,
        use_symmetry=not args.no_symmetry,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
    )

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, **dataset)
    n = len(dataset['states'])
    print(f"Saved {n} samples to {args.out}")
    print(f"  states:   {dataset['states'].shape}")
    print(f"  policies: {dataset['policies'].shape}")
    print(f"  values:   {dataset['values'].shape}")


if __name__ == '__main__':
    main()
