"""Validate exp_d27_league_v2/best_so_far.pt with a 20-game eval at the SAME
config the training-time eval uses (50 sims, network-value disabled,
heuristic-anchored leaves). Runs in main process; pin it to cores 4-5 via
taskset to avoid stepping on the live d27 trainer (cores 0-3)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "2"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

import time
import json
import argparse
from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.d27_league_train import quick_eval_vs_greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--use-heuristic-value", action="store_true")
    args = ap.parse_args()

    cfg = NetworkConfig(num_blocks=9, num_filters=96)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AlphaZeroNet(cfg, device=device)
    ckpt = net.load_checkpoint(args.ckpt)
    print(f"Loaded {args.ckpt} (iter={ckpt.get('iteration')}, "
          f"avg_pins_train={ckpt.get('avg_pins')})", flush=True)

    sp_cfg = SelfPlayConfig(
        num_simulations=args.sims,
        mcts_batch_size=16,
        use_heuristic_value=bool(args.use_heuristic_value),
        use_batched_mcts=True,
        max_moves=80,
        min_pins_to_keep=2,
    )

    print(f"Eval: {args.games} games, {args.sims} sims, "
          f"n_players={args.n_players}, "
          f"use_heuristic_value={args.use_heuristic_value}", flush=True)
    t0 = time.time()
    r = quick_eval_vs_greedy(net, sp_cfg, num_games=args.games,
                                n_players=args.n_players)
    dt = time.time() - t0
    print(json.dumps(r, indent=2), flush=True)
    print(f"\n{args.games} games in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
