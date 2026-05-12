"""d29: Parallel-game self-play training.

Differences from d27:
  - Generation is multi-process (4 workers × 8 games = 32 games/iter, vs 30
    games serially in d27). GPU was at 24% util in d27 — easy 3-4× headroom.
  - Each worker is GPU-attached; V100 32GB fits 4-8 procs at <1 GB each.
  - Master only does training and eval (still single-process for the SGD).

Inherits all the d27 safety nets:
  - thread caps at module top
  - league_delay
  - snapshot gate by eval pins
  - revert-to-best on degradation
  - --resume from latest.pt

Usage:
    bash scripts/launch_d29.sh
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"

import io
import sys
import time
import json
import math
import random
import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
except Exception:
    pass

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.parallel_self_play import run_iteration_parallel
from src.training.d27_league_train import (
    _save_snapshot, _list_snapshots, train_one_iteration,
    quick_eval_vs_greedy,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None,
                    help="Starting checkpoint .pt. If omitted, network "
                          "starts from random init (Kaiming) — useful for "
                          "fresh-init self-play with a strong opponent.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--games-per-worker", type=int, default=8)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--league-fraction", type=float, default=0.3)
    ap.add_argument("--league-delay", type=int, default=5)
    ap.add_argument("--snapshot-gate-pins", type=float, default=5.5)
    ap.add_argument("--revert-pins", type=float, default=4.5)
    ap.add_argument("--num-blocks", type=int, default=9)
    ap.add_argument("--num-filters", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--value-loss-weight", type=float, default=0.3)
    ap.add_argument("--mcts-batch-size", type=int, default=16)
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--use-gumbel", action="store_true")
    ap.add_argument("--num-considered-actions", type=int, default=16)
    ap.add_argument("--use-heuristic-value", action="store_true")
    ap.add_argument("--win-filter-min-pins", type=int, default=0,
                    help="Only keep games where someone won OR max_pins >= "
                          "this. Biases data toward decisive endgames so the "
                          "value head learns from real outcomes. 0=disabled.")
    ap.add_argument("--max-attempts-per-game", type=int, default=4,
                    help="With win-filter, max retries per game slot. Worker "
                          "tries up to this × games-per-worker before giving up.")
    ap.add_argument("--seed", type=int, default=29292)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-games", type=int, default=20,
                    help="Number of games per eval. 6 was too noisy "
                          "(d29 iter 3: 6-game eval=6.33, 20-game val=3.65). "
                          "20 games costs ~90s but the gate decision "
                          "is way more reliable.")
    ap.add_argument("--heuristic-opponent", choices=["advanced", "greedy"],
                    default=None,
                    help="If set, opponent in every league game is the "
                          "named heuristic (no MCTS, no network). Trains "
                          "directly against the kind of strong play we'll "
                          "actually face. Pair with league_fraction>0 and "
                          "league_delay=0.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    snapshots_dir = os.path.join(args.output_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    net_cfg = NetworkConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = AlphaZeroNet(net_cfg, device=device)

    start_iter = 1
    latest_path = os.path.join(args.output_dir, "latest.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = network.load_checkpoint(latest_path)
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"RESUMING from {latest_path} (continuing at iter {start_iter})",
              flush=True)
    elif args.start is not None:
        network.load_checkpoint(args.start)
        print(f"Loaded {args.start} (params={network.parameter_count()}, "
              f"device={device})", flush=True)
    else:
        print(f"FRESH random init (params={network.parameter_count()}, "
              f"device={device}) — no --start checkpoint provided",
              flush=True)

    sp_cfg = SelfPlayConfig(
        num_simulations=args.sims,
        mcts_batch_size=args.mcts_batch_size,
        use_heuristic_value=bool(args.use_heuristic_value),
        use_batched_mcts=True,
        max_moves=args.max_moves,
        min_pins_to_keep=2,
    )
    sp_cfg_dict = asdict(sp_cfg)
    net_cfg_dict = asdict(net_cfg)

    best_path = os.path.join(args.output_dir, "best_so_far.pt")
    best_pins = -1.0
    if os.path.exists(best_path):
        try:
            ck = torch.load(best_path, map_location="cpu", weights_only=False)
            best_pins = float(ck.get("avg_pins", -1.0))
            print(f"Existing best_so_far: {best_pins:.2f} pins", flush=True)
        except Exception:
            pass

    if start_iter == 1:
        init_snap_path = os.path.join(snapshots_dir, "snap_iter000.pt")
        # Don't overwrite an existing snap_iter000.pt — that lets callers
        # pre-seed a strong opponent into the league pool before launching.
        if os.path.exists(init_snap_path):
            print(f"Keeping existing initial snapshot at {init_snap_path}",
                  flush=True)
        else:
            init_snap = _save_snapshot(network, snapshots_dir, 0)
            print(f"Initial snapshot → {init_snap}", flush=True)
        # CRITICAL: establish a baseline best_pins from the seed model so
        # the first eval doesn't blindly become "best" even if it's worse
        # than what we started with. Without this, an iter-3 collapse to
        # 1.33 pins would overwrite a 6.33 starting model as "best".
        if best_pins < 0 and args.start is not None:
            print("Establishing baseline eval on seed model...", flush=True)
            t0 = time.time()
            base = quick_eval_vs_greedy(network, sp_cfg,
                                        num_games=args.eval_games,
                                        n_players=2)
            best_pins = float(base["avg_pins"])
            network.save_checkpoint(best_path, iteration=0,
                                    extra={"encoder_mode": "multicolour",
                                            "avg_pins": best_pins})
            print(f"Seed baseline: {best_pins:.2f} pins (saved as best_so_far) "
                  f"in {time.time()-t0:.0f}s", flush=True)
        elif best_pins < 0:
            # Fresh-init: skip baseline eval (network is random, ~0 pins)
            # AND save the random init as best_so_far so revert has something
            # to fall back to. The first eval that's actually >0 will replace it.
            best_pins = -0.01  # anything >= 0 will become "new best"
            network.save_checkpoint(best_path, iteration=0,
                                    extra={"encoder_mode": "multicolour",
                                            "avg_pins": best_pins})
            print(f"Skipping baseline eval (random init); seeded best_so_far "
                  f"at {best_pins} pins", flush=True)

    log_f = open(log_path, "a")

    for it in range(start_iter, args.iterations + 1):
        t_iter = time.time()

        # Always save current state to disk so workers can load it
        cur_path = os.path.join(args.output_dir, "current_for_workers.pt")
        network.save_checkpoint(cur_path, iteration=it,
                                extra={"encoder_mode": "multicolour"})

        snap_paths = _list_snapshots(snapshots_dir)
        eff_league_frac = (
            0.0 if it <= args.league_delay else args.league_fraction
        )
        opp_ckpts = snap_paths if eff_league_frac > 0 else []

        print(f"\n=== Iteration {it}/{args.iterations} ===", flush=True)
        print(f"  workers={args.num_workers}, games/worker={args.games_per_worker} "
              f"(total={args.num_workers * args.games_per_worker}), "
              f"sims={args.sims}, league_frac={eff_league_frac:.2f}, "
              f"snapshots_in_pool={len(snap_paths)}", flush=True)

        chunks_dir = os.path.join(args.output_dir, f"iter_{it:03d}_chunks")
        # Clean stale chunks if any
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)

        t0 = time.time()
        data, stats = run_iteration_parallel(
            current_ckpt=cur_path,
            opponent_ckpts=opp_ckpts,
            net_cfg_dict=net_cfg_dict,
            sp_cfg_dict=sp_cfg_dict,
            output_dir=chunks_dir,
            num_workers=args.num_workers,
            games_per_worker=args.games_per_worker,
            league_fraction=eff_league_frac,
            max_moves=args.max_moves,
            base_seed=args.seed + it * 10000,
            use_gumbel=bool(args.use_gumbel),
            num_considered_actions=args.num_considered_actions,
            win_filter_min_pins=args.win_filter_min_pins,
            max_attempts_per_game=args.max_attempts_per_game,
            heuristic_opponent=args.heuristic_opponent,
        )
        gen_time = time.time() - t0
        n_samples = data["obs"].shape[0]
        print(f"  gen done: {n_samples} samples in {gen_time:.0f}s "
              f"({stats['completed']} games, {stats['self_play']} SP, "
              f"{stats['league_play']} league, "
              f"current_wins_in_league={stats['current_wins_in_league']}/"
              f"{stats['league_play']})", flush=True)
        print(f"  per-N: {stats['per_n']}", flush=True)
        if stats.get("attempts", 0) > 0:
            print(f"  win-filter: {stats.get('kept', 0)} kept / "
                  f"{stats.get('attempts', 0)} attempts; "
                  f"{stats.get('filtered_low_pins', 0)} filtered_low_pins, "
                  f"{stats.get('discarded', 0)} hard-discarded",
                  flush=True)

        # We can delete chunks after merging (saved memory)
        try:
            shutil.rmtree(chunks_dir)
        except Exception:
            pass

        if n_samples == 0:
            print("  ⚠ no samples this iteration", flush=True)
            continue

        # Convert arrays back to the list-of-dicts format that train_one_iteration
        # expects. Build it lazily — only one dict per sample, no memmap views.
        samples = [
            {"obs": data["obs"][i],
             "action_mask": data["action_masks"][i],
             "policy_target": data["policies"][i],
             "value_target": float(data["values"][i])}
            for i in range(n_samples)
        ]
        # Drop the array refs so GC can collect the big arrays after the
        # samples list is consumed by train_one_iteration (which re-stacks).
        del data

        t0 = time.time()
        train_stats = train_one_iteration(
            network, samples,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            value_loss_weight=args.value_loss_weight,
        )
        del samples  # free the list of dicts immediately after training
        # Use n_samples (computed earlier from data["obs"].shape[0]) for logging
        # since `samples` is now gone.
        train_time = time.time() - t0
        last_ep = train_stats["epochs"][-1]
        print(f"  train: {train_stats['n_samples']} samples × {args.epochs} ep "
              f"in {train_time:.0f}s; final pi={last_ep['pi_loss']:.4f} "
              f"v={last_ep['v_loss']:.4f}", flush=True)

        eval_result = None
        if it % args.eval_every == 0:
            t0 = time.time()
            eval_result = quick_eval_vs_greedy(network, sp_cfg,
                                                num_games=args.eval_games,
                                                n_players=2)
            print(f"  eval vs greedy (n=2, {args.eval_games} games): "
                  f"{eval_result['avg_pins']:.2f} pins, "
                  f"{eval_result['wins']}/{args.eval_games} wins, "
                  f"{eval_result['truncated']}/{args.eval_games} trunc "
                  f"(in {time.time()-t0:.0f}s)", flush=True)

            if eval_result["avg_pins"] > best_pins:
                best_pins = float(eval_result["avg_pins"])
                network.save_checkpoint(
                    best_path, iteration=it,
                    extra={"encoder_mode": "multicolour", "avg_pins": best_pins},
                )
                print(f"  new best: {best_pins:.2f} pins → {best_path}",
                      flush=True)

            if args.revert_pins >= 0 and eval_result["avg_pins"] < args.revert_pins:
                print(f"  ⚠ avg_pins {eval_result['avg_pins']:.2f} < "
                      f"{args.revert_pins:.2f}; REVERTING to best_so_far.pt",
                      flush=True)
                if os.path.exists(best_path):
                    network.load_checkpoint(best_path)

        if it % args.snapshot_every == 0:
            gate = args.snapshot_gate_pins
            avg_pins = eval_result["avg_pins"] if eval_result else None
            should_snap = (gate < 0) or (avg_pins is not None and avg_pins >= gate)
            if should_snap:
                sp = _save_snapshot(network, snapshots_dir, it)
                print(f"  snapshot → {sp}", flush=True)
            else:
                print(f"  skip snapshot (avg_pins={avg_pins} < gate={gate:.2f})",
                      flush=True)

        network.save_checkpoint(latest_path, iteration=it,
                                extra={"encoder_mode": "multicolour"})

        log_f.write(json.dumps({
            "iteration": it, "samples": n_samples, "gen_time": gen_time,
            "train_time": train_time, "stats": stats,
            "train_loss_final": last_ep, "eval": eval_result,
            "iter_time": time.time() - t_iter,
        }) + "\n")
        log_f.flush()
        print(f"  iter total: {time.time() - t_iter:.0f}s", flush=True)

    log_f.close()
    final = os.path.join(args.output_dir, "final.pt")
    network.save_checkpoint(final, iteration=args.iterations,
                            extra={"encoder_mode": "multicolour"})
    print(f"\nFinal model → {final}", flush=True)


if __name__ == "__main__":
    main()
