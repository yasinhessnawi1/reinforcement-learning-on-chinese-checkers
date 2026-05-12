"""Multi-player heuristic warmstart data generator.

Plays N-player heuristic-vs-heuristic games (N drawn from {2,3,4,5,6}) and
records training samples from EVERY player's perspective using the multicolour
StateEncoder. Sharpens policy targets at temp=0.3 (the trick that gave us
the d20 jump from greedy=5 to greedy=7).

Each game contributes ~max_moves × N samples (one per ply). With ~5000 games
and an average of N≈4 + max_moves=80, we get ~1.6M samples — comparable to
the 669k of the original red-only warmstart but with multi-colour coverage.
"""
import os
# Pin BLAS thread counts BEFORE numpy is imported. With 32 worker procs on a
# 96-core box, each worker's BLAS otherwise grabs N threads → ~6× oversub
# and a 7× slowdown (measured: 14 s/game vs 1.92 s/game per worker).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
import math
import random
import numpy as np
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
except Exception:
    pass
from dataclasses import dataclass, field
from typing import List

from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder, _K_TO_RED_FRAME
from src.env.action_mapper import ActionMapper
from src.training.warmstart_generator import _noisy_heuristic_policy
from src.search.mcts import _score_colour


PRIMARY_COLOURS = ['red', 'lawn green', 'yellow']
COMPLEMENT = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}


def _pick_colours(n: int) -> List[str]:
    """Pick N colours that the game.py server would assign for N players.
    Game logic: alternate primary/complement, starting from a random primary.
    """
    primaries = list(PRIMARY_COLOURS)
    random.shuffle(primaries)
    out = []
    pi = 0
    for i in range(n):
        if i % 2 == 0:
            out.append(primaries[pi])
        else:
            out.append(COMPLEMENT[primaries[pi]])
            pi += 1
    return out


@dataclass
class MultiColourWarmStartConfig:
    num_games: int = 5000
    max_moves: int = 80           # max moves per player
    min_pins_to_keep: int = 2     # discard games where everyone < this
    n_choices: tuple = (2, 3, 4, 5, 6)  # uniform sampling over these
    n_weights: tuple = (3, 2, 2, 2, 1)   # bias toward 2-3 player slightly
    dirichlet_alpha: float = 0.5
    noise_fraction: float = 0.25
    fast_heuristic: bool = True
    sharpen_temp: float = 0.3     # final target sharpening temperature
    output_dir: str = "experiments/exp_d22_multicolour"


def _sharpen(p: np.ndarray, mask: np.ndarray, temp: float) -> np.ndarray:
    """Re-softmax probability mass on legal actions at lower temperature.
    Identical to /tmp/sharpen_warmstart.py logic.
    """
    out = np.zeros_like(p)
    legal = p[mask]
    if legal.sum() == 0:
        # uniform fallback
        out[mask] = 1.0 / max(mask.sum(), 1)
        return out
    logits = np.log(legal + 1e-12)
    sharp = np.exp(logits / temp)
    sharp /= sharp.sum()
    out[mask] = sharp
    return out


def _compute_game_value(board: BoardWrapper, colour: str, my_score: float,
                        max_score: float, win_count: int, n_players: int) -> float:
    """Per-perspective value target.

    Use normalised tournament-score differential to my best opponent.
    Range [-1, 1]. If we have most pins-in-goal among players, value > 0.
    """
    # my_score - best_other_score, normalised by ~1100 like the existing code.
    best_other = -1e9
    for c, plist in board.pins.items():
        if c == colour:
            continue
        s = _score_colour(board, c)
        if s > best_other:
            best_other = s
    diff = my_score - best_other
    return max(-1.0, min(1.0, diff / 1100.0))


def generate_multicolour_warmstart_data(config: MultiColourWarmStartConfig) -> dict:
    """Run heuristic-vs-heuristic N-player games, encode from every perspective,
    return a dict matching the existing warmstart .npz schema.
    """
    encoder = StateEncoder(grid_size=17, num_channels=10, mode="multicolour")
    mapper = ActionMapper(num_pins=10, num_cells=121)

    all_obs = []
    all_masks = []
    all_policies = []
    all_values = []

    games_played = 0
    games_discarded = 0
    samples_per_n = {n: 0 for n in config.n_choices}
    start = time.time()

    n_weights_total = sum(config.n_weights)

    for game_idx in range(config.num_games):
        # Pick N for this game by weighted choice
        r = random.random() * n_weights_total
        cum = 0
        n = config.n_choices[0]
        for nc, w in zip(config.n_choices, config.n_weights):
            cum += w
            if r < cum:
                n = nc; break

        colours = _pick_colours(n)
        board = BoardWrapper(colours)
        max_total_steps = config.max_moves * n
        step_count = 0
        trajectories = []  # per-step records: (colour, obs, mask, policy)

        while step_count < max_total_steps:
            colour = colours[step_count % n]
            legal = board.get_legal_moves(colour)
            if not legal:
                break

            # Heuristic move
            try:
                pin_id, dest, full_dist = _noisy_heuristic_policy(
                    board, colour,
                    alpha=config.dirichlet_alpha,
                    noise_frac=config.noise_fraction,
                    fast=config.fast_heuristic,
                )
            except ValueError:
                break

            # Encode obs from this colour's perspective (multicolour)
            obs = encoder.encode_multicolour(board, colour, colours)
            mask_raw = mapper.build_action_mask(legal)

            # Rotate the action target into the same frame as the obs.
            # If colour requires k×60° rotation to land in red's frame, the
            # action distribution must also be rotated by k.
            k = encoder.k_to_red_frame(colour)
            mask_rot = encoder.rotate_action_distribution_k(
                mask_raw.astype(np.bool_), k
            ).astype(np.bool_)
            policy_rot = encoder.rotate_action_distribution_k(full_dist, k)

            # Sharpen the policy target
            sharp = _sharpen(policy_rot, mask_rot, config.sharpen_temp)

            trajectories.append({
                "colour": colour,
                "obs": obs.astype(np.float32),
                "mask": mask_rot.astype(np.bool_),
                "policy": sharp.astype(np.float32),
            })

            board.apply_move(colour, pin_id, dest)
            step_count += 1
            if board.check_win(colour):
                break

        # Compute final per-colour values
        any_kept = False
        # Discard if every colour has < min_pins
        max_pins = max(board.pins_in_goal(c) for c in colours)
        if max_pins < config.min_pins_to_keep:
            games_discarded += 1
            continue

        # Per-colour value target
        per_colour_value = {}
        for c in colours:
            my_s = _score_colour(board, c)
            per_colour_value[c] = _compute_game_value(
                board, c, my_s, 0, 0, n
            )

        for step in trajectories:
            v = per_colour_value[step["colour"]]
            all_obs.append(step["obs"])
            all_masks.append(step["mask"])
            all_policies.append(step["policy"])
            all_values.append(v)
            samples_per_n[n] += 1
            any_kept = True

        if any_kept:
            games_played += 1

        if (game_idx + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (game_idx + 1) / elapsed
            print(f"  Game {game_idx + 1}/{config.num_games}: "
                  f"{len(all_obs)} samples, {games_discarded} discarded, "
                  f"{rate:.1f} g/s, "
                  f"per-N: {dict(samples_per_n)}", flush=True)

    obs_arr = np.stack(all_obs).astype(np.float32)
    masks_arr = np.stack(all_masks).astype(np.bool_)
    policies_arr = np.stack(all_policies).astype(np.float32)
    values_arr = np.array(all_values, dtype=np.float32)
    print(f"\n  Done: {games_played} games, {games_discarded} discarded, "
          f"{obs_arr.shape[0]} samples", flush=True)
    print(f"  Per-N samples: {samples_per_n}", flush=True)
    return {
        "obs": obs_arr,
        "action_masks": masks_arr,
        "policies": policies_arr,
        "values": values_arr,
    }


def save_data(data: dict, path: str) -> None:
    """Atomic write: save to a temp file then rename, so a crash mid-write
    leaves either the old version or nothing — never a corrupt npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def _worker_generate(args_tuple):
    """Worker entry: receives (num_games, max_moves, seed, worker_idx,
    chunks_dir) and SAVES its chunk to disk before returning. Saving from
    the worker (not the master) means the data survives even if the master
    is killed mid-run, which is what we actually want from "fault tolerance".
    Returns the path of the saved chunk for the master to log."""
    import os
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass
    num_games, max_moves, seed, worker_idx, chunks_dir = args_tuple
    random.seed(seed)
    np.random.seed(seed)
    cfg = MultiColourWarmStartConfig(num_games=num_games, max_moves=max_moves)
    data = generate_multicolour_warmstart_data(cfg)
    if chunks_dir:
        os.makedirs(chunks_dir, exist_ok=True)
        chunk_path = os.path.join(chunks_dir, f"chunk_{worker_idx:03d}.npz")
        np.savez_compressed(chunk_path, **data)
        return {"chunk_path": chunk_path, "n_samples": data["obs"].shape[0]}
    return {"chunk_path": None, "n_samples": data["obs"].shape[0], "data": data}


def generate_parallel(num_games: int, max_moves: int, num_workers: int,
                       output_dir: str, base_seed: int = 42) -> dict:
    """Split num_games across num_workers processes. Aggregate outputs.

    SAVES EACH WORKER'S CHUNK to disk as it completes, so a crash mid-run
    doesn't lose everything. The chunks are stored as
    `{output_dir}/chunks/chunk_{worker_idx:03d}.npz` and merged at the end.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    per_worker = max(1, num_games // num_workers)
    jobs = []
    remaining = num_games
    seed = base_seed
    for w in range(num_workers):
        n = per_worker if w < num_workers - 1 else remaining
        # Worker writes its own chunk to disk before returning, so killing
        # the master can't lose completed work.
        jobs.append((n, max_moves, seed + w, w, chunks_dir))
        remaining -= n
    print(f"[parallel] {num_workers} workers, {per_worker} games each "
          f"(total {num_games}); chunks → {chunks_dir}", flush=True)

    # Use 'spawn' so each worker re-runs module imports — that lets the
    # OMP/MKL/OPENBLAS env vars set at the top of this module take effect
    # BEFORE numpy/BLAS init in the worker (fork would inherit the parent's
    # already-initialised BLAS thread pool, defeating the point).
    import multiprocessing as mp
    mp_ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_worker_generate, j): i for i, j in enumerate(jobs)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            try:
                r = f.result()
                done += 1
                print(f"[parallel] worker {i} done ({done}/{num_workers}); "
                      f"+{r['n_samples']} samples → {r['chunk_path']}",
                      flush=True)
            except Exception as e:
                print(f"[parallel] worker {i} FAILED: {e}", flush=True)

    return _merge_chunks(chunks_dir)


def _merge_chunks(chunks_dir: str) -> dict:
    """Load every chunk_*.npz in chunks_dir and concatenate into one dict."""
    import glob
    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.npz")))
    if not chunk_files:
        raise RuntimeError(f"No chunks found in {chunks_dir}")
    obs = []; masks = []; pols = []; vals = []
    for cf in chunk_files:
        d = np.load(cf)
        obs.append(d["obs"]); masks.append(d["action_masks"])
        pols.append(d["policies"]); vals.append(d["values"])
    out = {
        "obs": np.concatenate(obs, axis=0),
        "action_masks": np.concatenate(masks, axis=0),
        "policies": np.concatenate(pols, axis=0),
        "values": np.concatenate(vals, axis=0),
    }
    print(f"[merge] {len(chunk_files)} chunks → {out['obs'].shape[0]} samples",
          flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=5000)
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--output-dir", default="experiments/exp_d22_multicolour")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=1,
                    help=">1 enables parallel generation across processes.")
    args = ap.parse_args()

    if args.num_workers > 1:
        data = generate_parallel(args.num_games, args.max_moves,
                                  args.num_workers, args.output_dir, args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        cfg = MultiColourWarmStartConfig(
            num_games=args.num_games,
            max_moves=args.max_moves,
            output_dir=args.output_dir,
        )
        data = generate_multicolour_warmstart_data(cfg)

    out_path = os.path.join(args.output_dir, "warmstart_data.npz")
    save_data(data, out_path)
    print(f"Saved to {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")
