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
        --workers 16 \\
        --out data/mcts_games_001.npz

Output format (saved as .npz):
    states  : float32 (N, 10, 17, 17)
    policies: float32 (N, 1210)
    values  : float32 (N,)
    game_ids: int32   (N,)  — which game each sample came from
"""

import argparse
import os
import sys
import multiprocessing as mp
from functools import partial

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal'))

from sb3_contrib import MaskablePPO
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import MCTS
from src.training.symmetry import ReflectionSymmetry

# Temperature schedule
_TEMP_THRESHOLD = 20
_HIGH_TEMP = 1.0
_LOW_TEMP = 0.1


def _play_single_game(
    game_idx: int,
    model_path: str,
    num_simulations: int,
    max_steps: int,
    use_symmetry: bool,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
) -> list[tuple[np.ndarray, np.ndarray, float, int]]:
    """Play one MCTS self-play game in a worker process.

    Returns list of (obs, probs, outcome, game_idx) tuples.
    """
    # Each worker loads its own model copy (avoids pickling issues)
    model = MaskablePPO.load(model_path, device='cpu')
    mcts = MCTS(
        model,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    sym = ReflectionSymmetry() if use_symmetry else None

    env = ChineseCheckersEnv(opponent_policy=None, max_steps=max_steps)
    obs, info = env.reset()

    step_data: list[tuple[np.ndarray, np.ndarray]] = []
    step = 0
    terminated = truncated = False

    while not (terminated or truncated):
        temp = _HIGH_TEMP if step < _TEMP_THRESHOLD else _LOW_TEMP
        action_probs = mcts.get_action_probs(env, temperature=temp)
        step_data.append((obs.copy(), action_probs.copy()))

        action = int(np.random.choice(len(action_probs), p=action_probs))
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

    agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board is not None else False
    outcome = 1.0 if agent_won else -1.0

    results = []
    for s_obs, s_probs in step_data:
        results.append((s_obs, s_probs, outcome, game_idx))
        if sym is not None:
            mirror_obs = sym.reflect_obs(s_obs)
            mirror_probs = sym.reflect_action_mask(s_probs.astype(bool)).astype(np.float32)
            total = mirror_probs.sum()
            if total > 0:
                mirror_probs = mirror_probs / total
            else:
                mirror_probs = s_probs.copy()
            results.append((mirror_obs, mirror_probs, outcome, game_idx))

    return results


def generate_games_parallel(
    model_path: str,
    num_games: int = 100,
    num_simulations: int = 200,
    max_steps: int = 1000,
    use_symmetry: bool = True,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    workers: int = 8,
) -> dict[str, np.ndarray]:
    """Generate self-play games in parallel and return a dataset dict."""

    worker_fn = partial(
        _play_single_game,
        model_path=model_path,
        num_simulations=num_simulations,
        max_steps=max_steps,
        use_symmetry=use_symmetry,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    all_states = []
    all_policies = []
    all_values = []
    all_game_ids = []

    print(f"  Launching {workers} workers for {num_games} games ...")

    with mp.Pool(processes=workers) as pool:
        results_iter = pool.imap_unordered(worker_fn, range(num_games))
        completed = 0
        for game_results in results_iter:
            for obs, probs, value, gid in game_results:
                all_states.append(obs)
                all_policies.append(probs)
                all_values.append(value)
                all_game_ids.append(gid)
            completed += 1
            if completed % 10 == 0 or completed == num_games:
                print(f"  {completed}/{num_games} games done — {len(all_states)} samples")

    return {
        'states':   np.array(all_states,   dtype=np.float32),
        'policies': np.array(all_policies, dtype=np.float32),
        'values':   np.array(all_values,   dtype=np.float32),
        'game_ids': np.array(all_game_ids, dtype=np.int32),
    }


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
    """Generate self-play games sequentially (single-process fallback)."""
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
        env = ChineseCheckersEnv(opponent_policy=None, max_steps=max_steps)
        obs, info = env.reset()
        step_data = []
        step = 0
        terminated = truncated = False

        while not (terminated or truncated):
            temp = _HIGH_TEMP if step < _TEMP_THRESHOLD else _LOW_TEMP
            action_probs = mcts.get_action_probs(env, temperature=temp)
            step_data.append((obs.copy(), action_probs.copy()))
            action = int(np.random.choice(len(action_probs), p=action_probs))
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

        agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board is not None else False
        outcome = 1.0 if agent_won else -1.0

        for s_obs, s_probs in step_data:
            all_states.append(s_obs)
            all_policies.append(s_probs)
            all_values.append(outcome)
            all_game_ids.append(game_idx)
            if use_symmetry:
                mirror_obs = sym.reflect_obs(s_obs)
                mirror_probs = sym.reflect_action_mask(s_probs.astype(bool)).astype(np.float32)
                total = mirror_probs.sum()
                if total > 0:
                    mirror_probs = mirror_probs / total
                else:
                    mirror_probs = s_probs.copy()
                all_states.append(mirror_obs)
                all_policies.append(mirror_probs)
                all_values.append(outcome)
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
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel workers. 0=auto (cpu_count), 1=sequential (default: 0)')
    args = parser.parse_args(argv)

    print(f"Loading model from {args.model}")

    workers = args.workers if args.workers > 0 else mp.cpu_count()

    if workers <= 1:
        model = MaskablePPO.load(args.model)
        print(f"Generating {args.num_games} games sequentially with {args.simulations} sims/move ...")
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
    else:
        print(f"Generating {args.num_games} games with {workers} workers, {args.simulations} sims/move ...")
        dataset = generate_games_parallel(
            model_path=args.model,
            num_games=args.num_games,
            num_simulations=args.simulations,
            max_steps=args.max_steps,
            use_symmetry=not args.no_symmetry,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            workers=workers,
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
