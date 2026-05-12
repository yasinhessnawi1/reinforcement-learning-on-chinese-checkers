"""Multi-player free-for-all arena: pit N different agents against each other.

Each game has n_players seats, filled by user-specified agents. Seat order
is rotated across games to remove first-mover bias. Tracks wins, avg pins,
and avg tournament score per agent.

Agents:
  ckpt:<path>           — network + MCTS, sims and heuristic-value configurable
  advanced              — advanced heuristic policy
  greedy                — greedy policy

Usage:
  python scripts/arena_multiplayer.py \\
    --agent ckpt:experiments/exp_d35_winonly/best_so_far.pt:d35 \\
    --agent ckpt:experiments/exp_d39_sims600/best_so_far.pt:d39 \\
    --agent advanced:adv \\
    --agent greedy:gr \\
    --games 30 --sims 400 --use-heuristic-value
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
from typing import Optional, List, Dict, Callable
import numpy as np

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.multi_n_self_play import (
    _pick_colours, _make_proxy_env_mc, _create_mcts_engine_mc,
    _ENCODER_MC, _MAPPER, _tournament_score,
)
from src.env.board_wrapper import BoardWrapper
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy


def _parse_agent_spec(spec: str, sp_cfg: SelfPlayConfig, device: str):
    """Parse 'ckpt:path:label' / 'advanced:label' / 'greedy:label' → (label, fn).

    fn(board, colour, sc, max_steps, turn_order) → (pin_id, dest)
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind == "advanced":
        label = parts[1] if len(parts) > 1 else "advanced"
        def fn(board, colour, sc, max_steps, turn_order):
            return advanced_heuristic_policy(board, colour)
        return label, fn, None
    if kind == "greedy":
        label = parts[1] if len(parts) > 1 else "greedy"
        def fn(board, colour, sc, max_steps, turn_order):
            return greedy_policy(board, colour)
        return label, fn, None
    if kind == "ckpt":
        path = parts[1]
        label = parts[2] if len(parts) > 2 else os.path.basename(path)
        net_cfg = NetworkConfig(num_blocks=9, num_filters=96)
        net = AlphaZeroNet(net_cfg, device=device)
        ckpt = net.load_checkpoint(path)
        print(f"  loaded {label}: {path} (iter={ckpt.get('iteration')})",
              flush=True)
        eval_cfg = SelfPlayConfig(**{**asdict(sp_cfg),
                                      "temperature_moves": 0,
                                      "temperature_low": 0.0,
                                      "dirichlet_epsilon": 0.0})
        engine = _create_mcts_engine_mc(net, eval_cfg)
        def fn(board, colour, sc, max_steps, turn_order):
            proxy = _make_proxy_env_mc(board, colour, sc, max_steps,
                                        turn_order=turn_order)
            ap, _ = engine.get_action_probs_and_value(proxy, temperature=0.0)
            action = int(np.argmax(ap))
            return _MAPPER.decode(action)
        return label, fn, net
    raise ValueError(f"unknown agent spec: {spec}")


def play_one_game(agent_fns: List[Callable], labels: List[str],
                  seat_assignment: List[int], n_players: int,
                  max_moves: int, game_time_sec: float = 100.0):
    """Play one game. seat_assignment[i] = agent index for seat (colour) i.

    Returns dict with per-seat stats: pins_in_goal, tournament_score, winner.
    """
    colours = _pick_colours(n_players)
    board = BoardWrapper(colours)
    max_steps = max_moves * n_players
    sc = 0
    winner = None
    move_counts = {c: 0 for c in colours}
    t_per_agent = {i: 0.0 for i in range(len(agent_fns))}

    while sc < max_steps:
        colour = colours[sc % n_players]
        legal = board.get_legal_moves(colour)
        if not legal:
            break
        agent_idx = seat_assignment[sc % n_players]
        fn = agent_fns[agent_idx]
        t0 = time.time()
        try:
            pin_id, dest = fn(board, colour, sc, max_steps, colours)
        except Exception:
            pin_id = next(iter(legal.keys())); dest = legal[pin_id][0]
        t_per_agent[agent_idx] += time.time() - t0
        board.apply_move(colour, pin_id, dest)
        sc += 1
        move_counts[colour] += 1
        if board.check_win(colour):
            winner = colour
            break

    # Per-seat outcomes
    seats = []
    winner_seat = None
    for i, colour in enumerate(colours):
        agent_idx = seat_assignment[i]
        pins = board.pins_in_goal(colour)
        # Use 30s as a reasonable approximation; we don't track real per-seat
        # time here. The tournament_score formula uses time_score = max(0, 100 - t).
        score = _tournament_score(board, colour, move_counts[colour],
                                   time_sec=30.0)
        seats.append({
            "seat": i,
            "colour": colour,
            "agent_idx": agent_idx,
            "agent_label": labels[agent_idx],
            "pins": pins,
            "score": score,
            "is_winner": (winner == colour),
        })
        if winner == colour:
            winner_seat = i

    return {
        "winner": winner,
        "winner_seat": winner_seat,
        "seats": seats,
        "moves": sc,
        "agent_time_sec": t_per_agent,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", action="append", required=True,
                    help="Repeat for each agent. Specs: 'ckpt:path:label', "
                          "'advanced:label', 'greedy:label'.")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--n-players", type=int, default=4,
                    help="Seats per game (2/3/4/5/6).")
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--mcts-batch-size", type=int, default=16)
    ap.add_argument("--use-heuristic-value", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None,
                    help="Optional path to write JSON results")
    args = ap.parse_args()

    if len(args.agent) < args.n_players:
        print(f"ERROR: need at least {args.n_players} agents for "
              f"{args.n_players}-player games (got {len(args.agent)})",
              file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sp_cfg = SelfPlayConfig(
        num_simulations=args.sims, mcts_batch_size=args.mcts_batch_size,
        use_heuristic_value=bool(args.use_heuristic_value),
        use_batched_mcts=True,
        max_moves=args.max_moves, min_pins_to_keep=2,
    )

    print(f"Loading {len(args.agent)} agents...", flush=True)
    agent_fns = []; labels = []; nets = []
    for spec in args.agent:
        label, fn, net = _parse_agent_spec(spec, sp_cfg, device)
        labels.append(label)
        agent_fns.append(fn)
        nets.append(net)

    print(f"\nArena: {args.games} games, n_players={args.n_players}, "
          f"sims={args.sims}, agents={labels}", flush=True)

    # Per-agent aggregate stats
    stats = {i: {"label": labels[i], "games_played": 0, "wins": 0,
                  "pins_total": 0, "score_total": 0.0,
                  "time_total": 0.0, "moves_total": 0}
             for i in range(len(labels))}
    games_log = []

    t0_all = time.time()
    for gi in range(args.games):
        # Pick which agents play this game (need exactly n_players)
        # If we have more agents than seats, randomly select n_players of them
        # If we have exactly n_players, use all
        if len(labels) > args.n_players:
            chosen = list(np.random.choice(len(labels), size=args.n_players,
                                            replace=False))
        else:
            chosen = list(range(len(labels)))
        # Random rotation of seat assignment to remove first-mover bias
        np.random.shuffle(chosen)
        # seat_assignment[i] = agent_idx that plays seat i
        seat_assignment = chosen

        t_game = time.time()
        result = play_one_game(agent_fns, labels, seat_assignment,
                                args.n_players, args.max_moves,
                                game_time_sec=100.0)
        game_dt = time.time() - t_game

        # Update per-agent stats
        for seat_info in result["seats"]:
            idx = seat_info["agent_idx"]
            stats[idx]["games_played"] += 1
            stats[idx]["wins"] += 1 if seat_info["is_winner"] else 0
            stats[idx]["pins_total"] += seat_info["pins"]
            stats[idx]["score_total"] += seat_info["score"]
            stats[idx]["time_total"] += result["agent_time_sec"][idx]
        # Total moves played by each agent (each seat contributes its colour's moves)
        # We only logged per-agent time; per-agent moves we can recompute from seats.

        winner_label = (labels[seat_assignment[result["winner_seat"]]]
                        if result["winner_seat"] is not None else "none")
        seat_str = "/".join(labels[i] for i in seat_assignment)
        print(f"  game {gi+1:>3}/{args.games}: seats=[{seat_str}], "
              f"moves={result['moves']}, "
              f"winner={winner_label} (in {game_dt:.0f}s)", flush=True)
        games_log.append({
            "game": gi+1, "seat_assignment": seat_assignment,
            "winner": winner_label, "seats": result["seats"],
            "duration_sec": game_dt,
        })

    total_dt = time.time() - t0_all

    print(f"\n=== Final results ({args.games} games, {total_dt:.0f}s total) ===",
          flush=True)
    print(f"{'Agent':<20} {'Played':>6} {'Wins':>5} {'WinRate':>8} "
          f"{'AvgPins':>8} {'AvgScore':>9} {'AvgTime':>8}", flush=True)
    summary = {}
    for i in range(len(labels)):
        s = stats[i]
        played = max(s["games_played"], 1)
        win_rate = s["wins"] / played
        avg_pins = s["pins_total"] / played
        avg_score = s["score_total"] / played
        avg_time = s["time_total"] / played
        summary[labels[i]] = {
            "games_played": s["games_played"],
            "wins": s["wins"],
            "win_rate": win_rate,
            "avg_pins": avg_pins,
            "avg_score": avg_score,
            "avg_time_per_game": avg_time,
        }
        print(f"{labels[i]:<20} {s['games_played']:>6} {s['wins']:>5} "
              f"{win_rate:>7.1%} {avg_pins:>8.2f} {avg_score:>9.1f} "
              f"{avg_time:>7.1f}s", flush=True)

    if args.out:
        # numpy types aren't JSON-serializable; coerce
        def _np(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            raise TypeError(repr(o))
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "games": games_log,
                        "config": {
                            "games": args.games,
                            "n_players": args.n_players,
                            "sims": args.sims,
                            "max_moves": args.max_moves,
                            "use_heuristic_value": args.use_heuristic_value,
                            "agents": labels,
                        }}, f, indent=2, default=_np)
        print(f"\nResults JSON → {args.out}", flush=True)


if __name__ == "__main__":
    main()
