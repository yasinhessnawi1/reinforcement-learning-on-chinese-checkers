#!/usr/bin/env python3
"""
collect_scm_d41_multi_gpu.py - Multiplayer SCM collection on GPU using batched MCTS.

Setup matches the tournament target:
  - Base policy: experiments/exp_d41_multiN/best_so_far.pt
  - 1500-2000 trajectories with mixed player counts (2-6)
  - Weights bias toward 6p (matches tournament: 1,4,6,8,12 for N=2..6)
  - 200 MCTS sims/move (same as SCM v2 for comparable search stats)
  - Single GPU, batched MCTS for throughput

Each N-player game produces N trajectories (one per perspective), so the
trajectory count is higher than the game count.

Incremental .npz chunk saves with resume support (counts existing chunks).
"""

import os
import sys
import argparse
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.search.batched_mcts import BatchedAlphaZeroMCTS
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.scm_self_play import (
    SCMTrajectory, SCMTrajectoryStep, _save_traj_chunk,
)
from src.network.search_conditioned_modulator import extract_search_features
from src.env.board_wrapper import BoardWrapper


def _existing_chunk_count(save_dir: str) -> int:
    if not os.path.isdir(save_dir):
        return 0
    return len([f for f in os.listdir(save_dir) if f.startswith("chunk_") and f.endswith(".npz")])


def _sample_n_players(n_choices: tuple, n_weights: tuple) -> int:
    """Weighted random sample from n_choices."""
    total = sum(n_weights)
    r = random.random() * total
    for nc, w in zip(n_choices, n_weights):
        r -= w
        if r <= 0:
            return nc
    return n_choices[-1]


def play_one_multi_game_gpu(
    network: AlphaZeroNet,
    mcts_engine: BatchedAlphaZeroMCTS,
    n_players: int,
    config: SelfPlayConfig,
) -> list[SCMTrajectory]:
    """Play one N-player game with shared batched MCTS engine.

    Mirrors play_game_collect_scm_data_multi but uses the batched engine for
    network inference. The engine is re-used across moves; each call to
    engine.run(proxy) starts a fresh search.
    """
    from src.training.multi_n_self_play import (
        _pick_colours, _make_proxy_env_mc,
        _compute_value_per_colour, _ENCODER_MC, _MAPPER,
    )

    colours = _pick_colours(n_players)
    board = BoardWrapper(colours)
    max_total_steps = config.max_moves * n_players

    scm_steps_per_colour: dict[str, list[SCMTrajectoryStep]] = {c: [] for c in colours}
    move_counts: dict[str, int] = {c: 0 for c in colours}
    step_count = 0
    winner = None

    while step_count < max_total_steps:
        colour = colours[step_count % n_players]
        legal = board.get_legal_moves(colour)
        if not legal:
            break

        proxy = _make_proxy_env_mc(board, colour, step_count, max_total_steps,
                                   turn_order=colours)
        action_mask_raw = _MAPPER.build_action_mask(legal)
        obs = _ENCODER_MC.encode_multicolour(board, colour, colours)
        k = _ENCODER_MC.k_to_red_frame(colour)

        if k != 0:
            mask_canon = _ENCODER_MC.rotate_action_distribution_k(
                action_mask_raw.astype(np.bool_), k
            ).astype(np.bool_)
        else:
            mask_canon = action_mask_raw.astype(np.bool_)

        temp = 1.0 if move_counts[colour] < config.temperature_moves else config.temperature_low

        # Batched MCTS run
        root = mcts_engine.run(proxy)
        num_actions = 1210
        action_probs_raw = mcts_engine._visits_to_probs(
            root, num_actions, action_mask_raw, temp,
        )

        if k != 0:
            action_probs_canon = _ENCODER_MC.rotate_action_distribution_k(action_probs_raw, k)
        else:
            action_probs_canon = action_probs_raw

        raw_logits, _, _ = network.predict_raw_logits(obs, mask_canon)
        raw_policy_canon = network.predict(obs, mask_canon)[0]

        stats = mcts_engine._extract_stats_from_root(
            root, num_actions, mask_canon,
            raw_policy=raw_policy_canon,
            turn_number=move_counts[colour],
        )
        search_features = extract_search_features(stats)

        scm_steps_per_colour[colour].append(SCMTrajectoryStep(
            search_features=search_features,
            raw_logits=raw_logits,
            action_mask=mask_canon.copy(),
            mcts_policy=action_probs_canon.copy(),
        ))

        if temp < 1e-6:
            action = int(np.argmax(action_probs_raw))
        else:
            action = int(np.random.choice(len(action_probs_raw), p=action_probs_raw))
        pin_id, dest = _MAPPER.decode(action)
        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1

        if board.check_win(colour):
            winner = colour
            break

    values = _compute_value_per_colour(board, colours, winner, move_counts)
    trajectories = []
    for colour in colours:
        steps = scm_steps_per_colour[colour]
        if len(steps) >= 5:
            trajectories.append(SCMTrajectory(steps=steps, game_outcome=values[colour]))
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Multiplayer SCM collection on GPU + batched MCTS")
    parser.add_argument("--checkpoint", type=str, default="experiments/exp_d41_multiN/best_so_far.pt")
    parser.add_argument("--output", type=str, default="experiments/scm_d41_multi/scm_trajectories")
    parser.add_argument("--num-games", type=int, default=400,
                        help="Games to play; expected trajectories = N_avg * games (~4x)")
    parser.add_argument("--target-trajectories", type=int, default=1800,
                        help="Stop once this many trajectories collected (overrides --num-games)")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--max-moves", type=int, default=100)
    parser.add_argument("--mcts-batch-size", type=int, default=8)
    parser.add_argument("--n-choices", type=str, default="2,3,4,5,6")
    parser.add_argument("--n-weights", type=str, default="1,4,6,8,12",
                        help="Tournament-matched: heavy 6p bias")
    parser.add_argument("--no-heuristic-value", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_choices = tuple(int(x) for x in args.n_choices.split(","))
    n_weights = tuple(int(x) for x in args.n_weights.split(","))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(args.output, exist_ok=True)

    net = AlphaZeroNet(NetworkConfig(num_blocks=9, num_filters=96), device=device)
    net.load_checkpoint(args.checkpoint)
    print(f"Loaded policy: {args.checkpoint} ({net.parameter_count():,} params)")

    sp_config = SelfPlayConfig(
        num_simulations=args.sims,
        use_heuristic_value=not args.no_heuristic_value,
        max_moves=args.max_moves,
    )

    n_existing_chunks = _existing_chunk_count(args.output)
    print(f"Resume: {n_existing_chunks} existing chunks "
          f"(~{n_existing_chunks * args.chunk_size} trajectories)")
    trajs_already = n_existing_chunks * args.chunk_size
    if trajs_already >= args.target_trajectories:
        print("  Target already met.")
        return

    print(f"Player count weights: {dict(zip(n_choices, n_weights))}")
    print(f"Target trajectories: {args.target_trajectories} "
          f"(remaining {args.target_trajectories - trajs_already})")

    mcts = BatchedAlphaZeroMCTS(
        network=net,
        num_simulations=args.sims,
        c_puct=sp_config.c_puct,
        dirichlet_alpha=sp_config.dirichlet_alpha,
        dirichlet_epsilon=sp_config.dirichlet_epsilon,
        use_heuristic_value=sp_config.use_heuristic_value,
        opponent_policy=None,
        batch_size=args.mcts_batch_size,
    )
    print(f"MCTS: batched (batch_size={args.mcts_batch_size}, sims={args.sims})")

    pending: list[SCMTrajectory] = []
    chunk_idx = n_existing_chunks
    total_trajs_this_run = 0
    games_played = 0
    t_start = time.time()

    while trajs_already + total_trajs_this_run < args.target_trajectories:
        n_players = _sample_n_players(n_choices, n_weights)
        t_game = time.time()
        trajs = play_one_multi_game_gpu(net, mcts, n_players, sp_config)
        game_dt = time.time() - t_game
        games_played += 1

        for t in trajs:
            pending.append(t)
            total_trajs_this_run += 1

        if games_played % 2 == 0 or games_played == 1:
            done_now = trajs_already + total_trajs_this_run
            elapsed = time.time() - t_start
            traj_rate = total_trajs_this_run / elapsed if elapsed > 0 else 0
            game_rate = games_played / elapsed if elapsed > 0 else 0
            remaining = args.target_trajectories - done_now
            eta_h = (remaining / traj_rate / 3600) if traj_rate > 0 else 0
            print(f"  game {games_played} (N={n_players}, {game_dt:.0f}s, "
                  f"+{len(trajs)} trajs) | total={done_now}/{args.target_trajectories} "
                  f"| {traj_rate:.2f} trajs/s, {game_rate*60:.1f} games/min | "
                  f"ETA {eta_h:.1f}h", flush=True)

        while len(pending) >= args.chunk_size:
            to_save = pending[:args.chunk_size]
            pending = pending[args.chunk_size:]
            path = os.path.join(args.output, f"chunk_{chunk_idx:04d}.npz")
            _save_traj_chunk(to_save, path)
            print(f"  Saved {path} ({len(to_save)} trajs)", flush=True)
            chunk_idx += 1

    # Flush leftover
    if pending:
        path = os.path.join(args.output, f"chunk_{chunk_idx:04d}.npz")
        _save_traj_chunk(pending, path)
        print(f"  Final chunk: {path} ({len(pending)} trajs)", flush=True)

    total_time = time.time() - t_start
    print(f"\nDone. {games_played} games, {total_trajs_this_run} trajs in {total_time/3600:.1f}h")


if __name__ == "__main__":
    main()
