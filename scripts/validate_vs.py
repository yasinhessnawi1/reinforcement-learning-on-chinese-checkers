"""General validation: agent (network + MCTS) vs configurable opponent.

Opponents:
  - greedy   : agents/greedy_agent.greedy_policy
  - advanced : agents/advanced_heuristic.advanced_heuristic_policy

Usage:
    python scripts/validate_vs.py \
        --ckpt experiments/exp_d27_league_v2/snapshots/snap_iter015.pt \
        --opponent advanced --games 20 --sims 200 --n-players 2 \
        --use-heuristic-value
"""
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
from dataclasses import asdict
import numpy as np

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.multi_n_self_play import (
    _pick_colours, _make_proxy_env_mc, _create_mcts_engine_mc,
    _ENCODER_MC, _MAPPER,
)
from src.env.board_wrapper import BoardWrapper
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy


OPP_FNS = {
    "greedy": greedy_policy,
    "advanced": advanced_heuristic_policy,
}


def eval_vs(network, sp_cfg, opponent_fn, num_games, n_players):
    """Network plays colours[0]; opponent plays the rest."""
    eval_cfg = SelfPlayConfig(**{**asdict(sp_cfg),
                                  "temperature_moves": 0,
                                  "temperature_low": 0.0,
                                  "dirichlet_epsilon": 0.0})
    engine = _create_mcts_engine_mc(network, eval_cfg)
    pins_total = []; wins = 0; trunc = 0; opp_wins = 0
    for gi in range(num_games):
        colours = _pick_colours(n_players)
        our_colour = colours[0]
        board = BoardWrapper(colours)
        max_steps = eval_cfg.max_moves * n_players
        sc = 0; winner = None
        while sc < max_steps:
            colour = colours[sc % n_players]
            legal = board.get_legal_moves(colour)
            if not legal:
                break
            if colour == our_colour:
                proxy = _make_proxy_env_mc(board, colour, sc, max_steps,
                                            turn_order=colours)
                ap, _ = engine.get_action_probs_and_value(proxy, temperature=0.0)
                action = int(np.argmax(ap))
                pin_id, dest = _MAPPER.decode(action)
            else:
                try:
                    pin_id, dest = opponent_fn(board, colour)
                except Exception:
                    pin_id = next(iter(legal.keys())); dest = legal[pin_id][0]
            board.apply_move(colour, pin_id, dest)
            sc += 1
            if board.check_win(colour):
                winner = colour; break
        pins_total.append(board.pins_in_goal(our_colour))
        if winner == our_colour: wins += 1
        elif winner is None: trunc += 1
        else: opp_wins += 1
    return {"games": num_games, "avg_pins": float(np.mean(pins_total)),
            "wins": wins, "opp_wins": opp_wins, "truncated": trunc,
            "min_pins": int(min(pins_total)), "max_pins": int(max(pins_total))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--opponent", choices=["greedy", "advanced"], default="greedy")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--n-players", type=int, default=2)
    ap.add_argument("--use-heuristic-value", action="store_true")
    ap.add_argument("--num-blocks", type=int, default=9)
    ap.add_argument("--num-filters", type=int, default=96)
    args = ap.parse_args()

    cfg = NetworkConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = AlphaZeroNet(cfg, device=device)
    ckpt = net.load_checkpoint(args.ckpt)
    print(f"Loaded {args.ckpt} "
          f"(iter={ckpt.get('iteration')}, "
          f"avg_pins_train={ckpt.get('avg_pins')})", flush=True)

    sp_cfg = SelfPlayConfig(
        num_simulations=args.sims, mcts_batch_size=16,
        use_heuristic_value=bool(args.use_heuristic_value),
        use_batched_mcts=True,
        max_moves=80, min_pins_to_keep=2,
    )

    opp_fn = OPP_FNS[args.opponent]
    print(f"Eval: {args.games} games, {args.sims} sims, "
          f"n_players={args.n_players}, opp={args.opponent}, "
          f"use_heuristic_value={args.use_heuristic_value}", flush=True)
    t0 = time.time()
    r = eval_vs(net, sp_cfg, opp_fn, args.games, args.n_players)
    dt = time.time() - t0
    print(json.dumps(r, indent=2), flush=True)
    print(f"\n{args.games} games in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
