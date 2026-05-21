#!/usr/bin/env python3
"""
collect_scm_d37_gpu.py - Collect SCM training trajectories on GPU using batched MCTS.

Used to train a native-d37 SCM for the tournament agent. The CPU collection
pipeline takes ~63 hours for 1100 games at 200 sims/move; with batched MCTS
on GPU it should take roughly 6-12 hours.

Single process, single GPU. Incremental chunk-based .npz saving. Resume by
counting existing chunks.
"""

import os
import sys
import argparse
import time
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.search.batched_mcts import BatchedAlphaZeroMCTS
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.scm_self_play import (
    play_game_collect_scm_data,
    _save_traj_chunk,
)
from src.agents.greedy_agent import greedy_policy


def _existing_chunk_count(save_dir: str) -> int:
    """Return number of existing chunk_*.npz files (for resume)."""
    if not os.path.isdir(save_dir):
        return 0
    return len([f for f in os.listdir(save_dir) if f.startswith("chunk_") and f.endswith(".npz")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SCM data on GPU + batched MCTS")
    parser.add_argument("--checkpoint", type=str, default="experiments/d37/best_so_far.pt")
    parser.add_argument("--output", type=str, default="experiments/scm_d37/scm_trajectories")
    parser.add_argument("--num-games", type=int, default=1100)
    parser.add_argument("--chunk-size", type=int, default=25,
                        help="Games per .npz chunk (smaller = more frequent saves)")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--mcts-batch-size", type=int, default=8,
                        help="Sims to evaluate per network forward pass")
    parser.add_argument("--no-heuristic-value", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.output, exist_ok=True)

    # --- Load model ---
    net = AlphaZeroNet(NetworkConfig(num_blocks=9, num_filters=96), device=device)
    net.load_checkpoint(args.checkpoint)
    print(f"Loaded policy: {args.checkpoint} ({net.parameter_count():,} params)")

    sp_config = SelfPlayConfig(
        num_simulations=args.sims,
        use_heuristic_value=not args.no_heuristic_value,
        max_moves=args.max_moves,
    )

    # --- Resume: count existing chunks ---
    n_existing_chunks = _existing_chunk_count(args.output)
    games_already_done = n_existing_chunks * args.chunk_size
    print(f"Resume check: {n_existing_chunks} existing chunks "
          f"(~{games_already_done} games already collected)")
    print(f"Target: {args.num_games} games total")
    if games_already_done >= args.num_games:
        print("  Target already met. Nothing to do.")
        return

    remaining = args.num_games - games_already_done
    print(f"  Will collect {remaining} more games")

    # --- Build batched MCTS ---
    # Use heuristic value AND batching for max throughput. Single shared
    # engine across all games is fine since each game resets to a fresh env.
    mcts = BatchedAlphaZeroMCTS(
        network=net,
        num_simulations=args.sims,
        c_puct=sp_config.c_puct,
        dirichlet_alpha=sp_config.dirichlet_alpha,
        dirichlet_epsilon=sp_config.dirichlet_epsilon,
        use_heuristic_value=sp_config.use_heuristic_value,
        opponent_policy=greedy_policy,
        batch_size=args.mcts_batch_size,
    )
    print(f"  MCTS: batched (batch_size={args.mcts_batch_size}, sims={args.sims})")

    # --- Collect ---
    pending: list = []
    chunk_idx = n_existing_chunks
    t_start = time.time()
    games_done_this_run = 0

    for i in range(remaining):
        t_game = time.time()
        _, scm_traj = play_game_collect_scm_data(
            network=net,
            config=sp_config,
            opponent_policy=greedy_policy,
            mcts_engine=mcts,
        )
        elapsed_game = time.time() - t_game

        if len(scm_traj.steps) >= 5:
            pending.append(scm_traj)

        games_done_this_run += 1
        global_done = games_already_done + games_done_this_run

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = games_done_this_run / elapsed
            eta_sec = (remaining - games_done_this_run) / rate if rate > 0 else 0
            print(f"  [{global_done}/{args.num_games}] "
                  f"last_game={elapsed_game:.0f}s, "
                  f"rate={rate:.2f} games/s, "
                  f"ETA={eta_sec/3600:.1f}h",
                  flush=True)

        # Save chunk
        if len(pending) >= args.chunk_size:
            path = os.path.join(args.output, f"chunk_{chunk_idx:04d}.npz")
            _save_traj_chunk(pending, path)
            print(f"  Saved {path} ({len(pending)} trajs)", flush=True)
            pending = []
            chunk_idx += 1

    # Flush any remainder
    if pending:
        path = os.path.join(args.output, f"chunk_{chunk_idx:04d}.npz")
        _save_traj_chunk(pending, path)
        print(f"  Final chunk saved: {path} ({len(pending)} trajs)", flush=True)

    total = time.time() - t_start
    print(f"\nDone. {games_done_this_run} games in {total/3600:.1f}h "
          f"({total/games_done_this_run:.0f}s/game)")


if __name__ == "__main__":
    main()
