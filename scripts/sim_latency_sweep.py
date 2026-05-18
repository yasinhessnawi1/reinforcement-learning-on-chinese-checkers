"""Measure MCTS search latency across sim counts to plan tournament budget.

Outputs (per device):
  - forward_ms: cost of one neural-net forward pass
  - per-sim ms: average ms per simulation inside batched MCTS
  - total ms: full search time at each sim count
  - max sims to fit a 2-second move budget (PER_MOVE_HARD_CAP)
  - max sims to fit 1.6 seconds (conservative tournament cap)

Usage:
  ./venv/bin/python scripts/sim_latency_sweep.py --device cuda
  ./venv/bin/python scripts/sim_latency_sweep.py --device cpu --threads 4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.env.state_encoder import StateEncoder
from src.env.action_mapper import ActionMapper
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.board_wrapper import BoardWrapper
from src.search.batched_mcts import BatchedAlphaZeroMCTS
from src.search.mcts import AlphaZeroMCTS


def build_proxy_env(encoder, mapper, colours):
    """Build a fresh ChineseCheckersEnv with the multicolour encoder."""
    proxy = ChineseCheckersEnv.__new__(ChineseCheckersEnv)
    proxy.render_mode = None
    proxy.max_steps = 200
    proxy.observation_space = None
    proxy.action_space = type("Space", (), {"n": 1210})()
    proxy._encoder = encoder
    proxy._mapper = mapper
    proxy._AGENT_COLOUR = colours[0]
    proxy._OPPONENT_COLOUR = colours[1]
    proxy._TURN_ORDER = list(colours)
    proxy._no_opponent = True
    proxy._opponent_policy = None
    proxy._board = BoardWrapper(list(colours))
    proxy._step_count = 0
    proxy._terminated = False
    proxy._truncated = False
    return proxy


def measure_forward(net, encoder, mapper, colours, n=20):
    """Time one network forward pass averaged over n calls."""
    proxy = build_proxy_env(encoder, mapper, colours)
    obs = encoder.encode_multicolour(proxy._board, colours[0], colours)
    mask = np.ones(1210, dtype=np.bool_)
    # Warmup
    for _ in range(3):
        net.predict(obs, mask)
    t0 = time.perf_counter()
    for _ in range(n):
        net.predict(obs, mask)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / n


def measure_mcts(net, encoder, mapper, colours, sims, batch_size, use_heuristic_value=True, n=3):
    """Time full MCTS search averaged over n searches."""
    # Build fresh env each call so the search starts from the same root
    times = []
    for _ in range(n):
        proxy = build_proxy_env(encoder, mapper, colours)
        mcts = BatchedAlphaZeroMCTS(
            network=net,
            num_simulations=sims,
            batch_size=batch_size,
            dirichlet_epsilon=0.0,
            use_heuristic_value=use_heuristic_value,
        )
        t0 = time.perf_counter()
        _ = mcts.select_action(proxy, temperature=0.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/exp_d35_winonly/best_so_far.pt",
                    help="Policy checkpoint to use for the sweep")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=0,
                    help="Override torch thread count (CPU only)")
    ap.add_argument("--sims", type=str, default="50,80,120,160,200,300,400,600,800",
                    help="Comma-separated sim counts")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-players", type=int, default=4,
                    help="Number of players in the proxy game (affects encoder)")
    ap.add_argument("--out", default=None, help="Write JSON to this path")
    args = ap.parse_args()

    if args.device == "cpu":
        nthreads = args.threads if args.threads > 0 else min(4, os.cpu_count() or 4)
        torch.set_num_threads(nthreads)
        print(f"[setup] torch threads = {nthreads}")
    device = args.device
    print(f"[setup] device = {device}")
    print(f"[setup] checkpoint = {args.checkpoint}")

    net = AlphaZeroNet(NetworkConfig(num_blocks=9, num_filters=96), device=device)
    net.load_checkpoint(args.checkpoint)
    encoder = StateEncoder(grid_size=17, num_channels=10, mode="multicolour")
    mapper = ActionMapper(num_pins=10, num_cells=121)
    # Need an initial encode to populate rotation tables for multicolour
    proxy_init = build_proxy_env(encoder, mapper,
                                  BoardWrapper(["red", "lawn green", "yellow", "blue"]).colour_order
                                  if args.n_players == 4 else ["red", "blue"])
    encoder.encode_multicolour(proxy_init._board, proxy_init._AGENT_COLOUR, proxy_init._TURN_ORDER)

    colours_4p = ["red", "lawn green", "yellow", "blue"]
    colours_2p = ["red", "blue"]
    colours = colours_4p if args.n_players >= 4 else colours_2p

    sim_counts = [int(x) for x in args.sims.split(",") if x.strip()]
    print(f"[setup] sim counts: {sim_counts}")

    # Forward pass baseline
    fwd_ms = measure_forward(net, encoder, mapper, colours, n=20)
    print(f"\n[forward] {fwd_ms:.2f} ms / call ({device})")

    results = {
        "device": device,
        "checkpoint": args.checkpoint,
        "n_players": args.n_players,
        "forward_ms": fwd_ms,
        "sweep": [],
    }

    print(f"\n{'sims':>6} {'ms/move':>10} {'std':>7} {'per-sim ms':>11} {'speedup':>9} "
          f"{'fit-2.0s':>9} {'fit-1.6s':>9}")
    print("-" * 70)
    for sims in sim_counts:
        ms, std = measure_mcts(net, encoder, mapper, colours, sims, args.batch_size, n=3)
        per_sim = ms / sims
        speedup = (sims * fwd_ms) / ms if ms > 0 else 0.0  # vs naive non-batched
        fit_2000 = "OK" if ms <= 2000 else "TOO SLOW"
        fit_1600 = "OK" if ms <= 1600 else "TOO SLOW"
        results["sweep"].append({
            "sims": sims, "ms_per_move": ms, "std_ms": std,
            "per_sim_ms": per_sim, "speedup_vs_naive": speedup,
            "fits_2000ms": ms <= 2000, "fits_1600ms": ms <= 1600,
        })
        print(f"{sims:>6} {ms:>10.1f} {std:>7.1f} {per_sim:>11.3f} {speedup:>9.2f}x "
              f"{fit_2000:>9} {fit_1600:>9}")

    # Recommended ceilings
    sims_fit_2000 = max((r["sims"] for r in results["sweep"] if r["fits_2000ms"]), default=0)
    sims_fit_1600 = max((r["sims"] for r in results["sweep"] if r["fits_1600ms"]), default=0)
    print(f"\n[recommend] max sims that fit 2.0s budget: {sims_fit_2000}")
    print(f"[recommend] max sims that fit 1.6s budget: {sims_fit_1600}")
    results["max_sims_2000ms"] = sims_fit_2000
    results["max_sims_1600ms"] = sims_fit_1600

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
