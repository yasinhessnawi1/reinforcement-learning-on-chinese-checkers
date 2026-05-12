"""d27: Self-play with league opponents and network value head.

Single-process design (the bottleneck is per-forward GPU latency for tiny
batches, not CPU parallelism — empirically ResNet 9x96 forward = 1.2s on
1-thread CPU vs 5ms on GPU; spawning workers gains nothing if each worker
has to call torch on CPU).

Differences from prior failed self-play attempts (d23/d24):
  1. Trains the network VALUE head (use_heuristic_value=False), so the
     network learns its own value rather than being capped by the heuristic.
     value_loss_weight=1.0 (not 0.25).
  2. League play: snapshot every K iterations, sample uniformly per game
     (with prob `league_fraction`); rest is pure self-play. Stops degenerate
     cycles vs the current network's specific weaknesses.
  3. Multi-N: every game picks N from {2..6} weighted toward 2-3 players.

Run:
    python -m src.training.d27_league_train \\
        --start experiments/exp_d22_multicolour/warmstart_model.pt \\
        --output-dir experiments/exp_d27_league \\
        --iterations 50 --games-per-iter 30 --sims 50
"""
import os
# Pin BLAS/OMP thread counts BEFORE numpy/torch import. The container has a
# 6-CPU cgroup limit but PyTorch reads `nproc` which sees the host's 96
# cores → spawns 48 intra-op + 96 inter-op threads → 144-way contention on
# 6 CPUs → load avg explodes → orchestrator restarts the pod.
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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
# Hard-cap torch thread pools regardless of host nproc. 4 + 2 = 6, matching
# our cgroup quota. Without this torch spawns 48+96 = 144 threads.
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
except Exception:
    pass

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig, TrainingSample
from src.training.multi_n_self_play import (
    _pick_colours, _make_proxy_env_mc, _create_mcts_engine_mc,
    _ENCODER_MC, _MAPPER, _sharpen, _compute_value_per_colour,
)
from src.env.board_wrapper import BoardWrapper
from src.agents.greedy_agent import greedy_policy


# --------------------------------------------------------------------------- #
# Snapshot helpers
# --------------------------------------------------------------------------- #
def _save_snapshot(network: AlphaZeroNet, snapshots_dir: str, iteration: int) -> str:
    os.makedirs(snapshots_dir, exist_ok=True)
    path = os.path.join(snapshots_dir, f"snap_iter{iteration:03d}.pt")
    network.save_checkpoint(path, iteration=iteration,
                            extra={"encoder_mode": "multicolour"})
    return path


def _list_snapshots(snapshots_dir: str) -> list[str]:
    if not os.path.isdir(snapshots_dir):
        return []
    return sorted(
        os.path.join(snapshots_dir, f) for f in os.listdir(snapshots_dir)
        if f.startswith("snap_iter") and f.endswith(".pt")
    )


# --------------------------------------------------------------------------- #
# League game: cur vs cur OR cur vs snapshot opponent.
# Records samples only from the *current* network's positions.
# --------------------------------------------------------------------------- #
# Engine kwargs (use_gumbel, num_considered_actions). Module-level so we
# don't have to thread it through every function signature. Set by main()
# from CLI args before running anything.
_ENGINE_KWARGS = {}


def play_league_game(
    cur_net: AlphaZeroNet,
    opp_net: AlphaZeroNet | None,   # None => pure self-play
    n_players: int,
    sp_cfg: SelfPlayConfig,
    sharpen_temp: float = 0.3,
):
    colours = _pick_colours(n_players)
    if opp_net is None:
        role_map = {c: "current" for c in colours}
    else:
        cur_colour = random.choice(colours)
        role_map = {c: ("current" if c == cur_colour else "opponent")
                    for c in colours}

    cur_engine = _create_mcts_engine_mc(cur_net, sp_cfg, **_ENGINE_KWARGS)
    if opp_net is not None:
        opp_engine = _create_mcts_engine_mc(opp_net, sp_cfg, **_ENGINE_KWARGS)
        engines = {c: (cur_engine if role_map[c] == "current" else opp_engine)
                   for c in colours}
    else:
        engines = {c: cur_engine for c in colours}

    board = BoardWrapper(colours)
    max_total_steps = sp_cfg.max_moves * n_players
    trajectories = []
    step_count = 0
    move_counts = {c: 0 for c in colours}
    winner = None

    while step_count < max_total_steps:
        colour = colours[step_count % n_players]
        legal = board.get_legal_moves(colour)
        if not legal:
            break

        proxy = _make_proxy_env_mc(board, colour, step_count, max_total_steps,
                                    turn_order=colours)
        action_mask = _MAPPER.build_action_mask(legal)
        obs = _ENCODER_MC.encode_multicolour(board, colour, colours)

        temp = 1.0 if move_counts[colour] < sp_cfg.temperature_moves \
                else sp_cfg.temperature_low
        engine = engines[colour]
        action_probs, mcts_value = engine.get_action_probs_and_value(
            proxy, temperature=temp
        )
        k = _ENCODER_MC.k_to_red_frame(colour)
        mask_canon = _ENCODER_MC.rotate_action_distribution_k(
            action_mask.astype(np.bool_), k
        ).astype(np.bool_)
        policy_canon = _ENCODER_MC.rotate_action_distribution_k(action_probs, k)
        sharp = _sharpen(policy_canon, mask_canon, sharpen_temp)

        if role_map[colour] == "current":
            trajectories.append({
                "colour": colour,
                "obs": obs.copy(),
                "action_mask": mask_canon.copy(),
                "policy_target": sharp.copy(),
                "mcts_value": mcts_value,
            })

        if temp < 1e-6:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))
        pin_id, dest = _MAPPER.decode(action)
        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1
        if board.check_win(colour):
            winner = colour
            break

    max_pins = max(board.pins_in_goal(c) for c in colours)
    if max_pins < sp_cfg.min_pins_to_keep:
        return {"samples": [], "winner": winner, "current_won": False}

    values = _compute_value_per_colour(board, colours, winner, move_counts)
    lam = sp_cfg.value_target_lambda
    samples = []
    for step in trajectories:
        v_outcome = values[step["colour"]]
        v_mcts = step["mcts_value"]
        v = max(-1.0, min(1.0, lam * v_outcome + (1.0 - lam) * v_mcts))
        samples.append({
            "obs": step["obs"],
            "action_mask": step["action_mask"],
            "policy_target": step["policy_target"],
            "value_target": float(v),
        })

    current_won = (winner is not None and role_map.get(winner) == "current")
    return {"samples": samples, "winner": winner, "current_won": current_won,
            "n_players": n_players}


# --------------------------------------------------------------------------- #
# Iteration: generate N games sequentially on GPU.
# --------------------------------------------------------------------------- #
def generate_iteration_serial(
    cur_net: AlphaZeroNet,
    snapshot_paths: list[str],
    net_cfg: NetworkConfig,
    sp_cfg: SelfPlayConfig,
    games_per_iter: int,
    n_choices=(2, 3, 4, 5, 6),
    n_weights=(3, 2, 2, 2, 1),
    sharpen_temp: float = 0.3,
    league_fraction: float = 0.5,
    device: str = "cuda",
):
    """Generate self-play data sequentially on the same device as cur_net.
    Snapshots are loaded lazily into a single shared opp_net instance so we
    don't blow up GPU memory."""
    n_w_total = sum(n_weights)
    samples = []
    stats = {"completed": 0, "discarded": 0, "self_play": 0, "league_play": 0,
             "current_wins_in_league": 0, "winner_breakdown": {},
             "per_n": {n: 0 for n in n_choices}}

    # Reusable opponent network (we swap state_dicts for each league game)
    opp_net = AlphaZeroNet(net_cfg, device=device)

    t_start = time.time()
    for gi in range(games_per_iter):
        # Pick N
        r = random.random() * n_w_total
        cum = 0
        n = n_choices[0]
        for nc, w in zip(n_choices, n_weights):
            cum += w
            if r < cum:
                n = nc; break

        # Pick opponent
        if snapshot_paths and random.random() < league_fraction:
            snap_path = random.choice(snapshot_paths)
            opp_net.load_checkpoint(snap_path)
            opp_for_game = opp_net
            mode = "league"
        else:
            opp_for_game = None
            mode = "self"

        try:
            r = play_league_game(cur_net, opp_for_game, n, sp_cfg, sharpen_temp)
        except Exception as e:
            print(f"  game {gi}: error {e}", flush=True)
            stats["discarded"] += 1
            continue
        if not r["samples"]:
            stats["discarded"] += 1
            continue

        samples.extend(r["samples"])
        stats["completed"] += 1
        stats["per_n"][n] += 1
        if mode == "self":
            stats["self_play"] += 1
        else:
            stats["league_play"] += 1
            if r["current_won"]:
                stats["current_wins_in_league"] += 1
        w = r["winner"] or "none"
        stats["winner_breakdown"][w] = stats["winner_breakdown"].get(w, 0) + 1

        if (gi + 1) % 5 == 0 or (gi + 1) == games_per_iter:
            elapsed = time.time() - t_start
            rate = (gi + 1) / elapsed if elapsed > 0 else 0
            eta = (games_per_iter - (gi + 1)) / max(rate, 1e-6)
            print(f"  game {gi+1}/{games_per_iter}: {len(samples)} samples, "
                  f"{stats['discarded']} discarded, {rate:.2f} g/s, "
                  f"ETA {eta:.0f}s", flush=True)

    return samples, stats


# --------------------------------------------------------------------------- #
# Trainer: one pass through the iteration's data.
# --------------------------------------------------------------------------- #
def train_one_iteration(
    network: AlphaZeroNet,
    samples: list[dict],
    batch_size: int = 256,
    epochs: int = 4,
    lr: float = 1e-4,
    value_loss_weight: float = 1.0,  # 1.0 because we WANT network value to learn
    weight_decay: float = 1e-4,
):
    if not samples:
        return {"epochs": [], "n_samples": 0}

    device = network.device
    model = network.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    obs_arr = np.stack([s["obs"] for s in samples]).astype(np.float32)
    mask_arr = np.stack([s["action_mask"] for s in samples]).astype(np.bool_)
    pol_arr = np.stack([s["policy_target"] for s in samples]).astype(np.float32)
    val_arr = np.array([s["value_target"] for s in samples], dtype=np.float32)
    n = obs_arr.shape[0]

    obs_t = torch.from_numpy(obs_arr).to(device)
    mask_t = torch.from_numpy(mask_arr).to(device)
    pol_t = torch.from_numpy(pol_arr).to(device)
    val_t = torch.from_numpy(val_arr).to(device)

    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        ep_total = 0.0; ep_pi = 0.0; ep_v = 0.0; nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            o = obs_t[idx]; m = mask_t[idx]; p = pol_t[idx]; v = val_t[idx]
            opt.zero_grad()
            policy_logits, value = model(o)
            policy_logits = policy_logits.masked_fill(~m, -1e9)
            log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
            policy_loss = -(p * log_probs).sum(dim=-1).mean()
            value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), v)
            loss = policy_loss + value_loss_weight * value_loss
            loss.backward()
            opt.step()
            ep_total += loss.item(); ep_pi += policy_loss.item()
            ep_v += value_loss.item(); nb += 1
        losses.append({
            "epoch": ep + 1,
            "train_loss": ep_total / max(nb, 1),
            "pi_loss": ep_pi / max(nb, 1),
            "v_loss": ep_v / max(nb, 1),
        })
    model.eval()
    return {"epochs": losses, "n_samples": n}


# --------------------------------------------------------------------------- #
# Eval vs greedy heuristic at fixed temp=0 (deterministic).
# --------------------------------------------------------------------------- #
def quick_eval_vs_greedy(
    network: AlphaZeroNet,
    sp_cfg: SelfPlayConfig,
    num_games: int = 6,
    n_players: int = 2,
):
    """Network plays colours[0]; greedy plays the rest. Returns avg pins."""
    eval_cfg = SelfPlayConfig(**{**asdict(sp_cfg),
                                  "temperature_moves": 0,
                                  "temperature_low": 0.0,
                                  "dirichlet_epsilon": 0.0})
    engine = _create_mcts_engine_mc(network, eval_cfg, **_ENGINE_KWARGS)
    pins_total = []; wins = 0; trunc = 0
    for gi in range(num_games):
        colours = _pick_colours(n_players)
        our_colour = colours[0]
        board = BoardWrapper(colours)
        max_steps = eval_cfg.max_moves * n_players
        sc = 0; winner = None
        while sc < max_steps:
            colour = colours[sc % n_players]
            legal = board.get_legal_moves(colour)
            if not legal: break
            if colour == our_colour:
                proxy = _make_proxy_env_mc(board, colour, sc, max_steps,
                                            turn_order=colours)
                ap, _ = engine.get_action_probs_and_value(proxy, temperature=0.0)
                action = int(np.argmax(ap))
                pin_id, dest = _MAPPER.decode(action)
            else:
                try:
                    pin_id, dest = greedy_policy(board, colour)
                except Exception:
                    pin_id = next(iter(legal.keys())); dest = legal[pin_id][0]
            board.apply_move(colour, pin_id, dest)
            sc += 1
            if board.check_win(colour):
                winner = colour; break
        pins_total.append(board.pins_in_goal(our_colour))
        if winner == our_colour: wins += 1
        if winner is None: trunc += 1
    return {"games": num_games, "avg_pins": float(np.mean(pins_total)),
            "wins": wins, "truncated": trunc}


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--games-per-iter", type=int, default=30)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--league-fraction", type=float, default=0.5)
    ap.add_argument("--num-blocks", type=int, default=9)
    ap.add_argument("--num-filters", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--value-loss-weight", type=float, default=1.0)
    ap.add_argument("--mcts-batch-size", type=int, default=16)
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="Resume from {output_dir}/latest.pt if it exists "
                          "(otherwise loads --start). Picks up at the next "
                          "iteration after the last logged one.")
    ap.add_argument("--use-heuristic-value", action="store_true",
                    help="Anchor MCTS leaf value to heuristic (still trains "
                          "network value head from real game outcomes).")
    ap.add_argument("--league-delay", type=int, default=0,
                    help="Number of iters at the start with league_fraction=0 "
                          "(pure self-play) before league play kicks in.")
    ap.add_argument("--snapshot-gate-pins", type=float, default=-1.0,
                    help="If >=0, only add a snapshot to the league pool when "
                          "the iter's eval avg_pins is at or above this value. "
                          "Prevents poisoning the pool with degraded models.")
    ap.add_argument("--revert-pins", type=float, default=-1.0,
                    help="If >=0, after an eval, if avg_pins is below this "
                          "value, revert the model to best_so_far.pt and skip "
                          "the next training step.")
    ap.add_argument("--use-gumbel", action="store_true",
                    help="Use Gumbel MCTS (Sequential Halving + completed-Q) "
                          "instead of vanilla AlphaZero MCTS. Sample-efficient "
                          "at low sim counts; recommended sims=32-64.")
    ap.add_argument("--num-considered-actions", type=int, default=16,
                    help="Top-k actions to consider at root in Gumbel MCTS "
                          "(only used with --use-gumbel).")
    args = ap.parse_args()

    # Wire engine kwargs to the module-level dict that play_league_game
    # / quick_eval read.
    global _ENGINE_KWARGS
    _ENGINE_KWARGS = {
        "use_gumbel": bool(args.use_gumbel),
        "num_considered_actions": int(args.num_considered_actions),
    }

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    snapshots_dir = os.path.join(args.output_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    net_cfg = NetworkConfig(num_blocks=args.num_blocks, num_filters=args.num_filters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = AlphaZeroNet(net_cfg, device=device)

    # Resume logic: if --resume and latest.pt exists, load it AND figure out
    # which iteration to resume from by reading the last entry of train_log.
    start_iter = 1
    latest_path = os.path.join(args.output_dir, "latest.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = network.load_checkpoint(latest_path)
        last_it = ckpt.get("iteration", 0)
        start_iter = last_it + 1
        print(f"RESUMING from {latest_path} (last iter={last_it}, "
              f"continuing at iter {start_iter})", flush=True)
    else:
        network.load_checkpoint(args.start)
        print(f"Loaded {args.start} (params={network.parameter_count()}, "
              f"device={device})", flush=True)

    sp_cfg = SelfPlayConfig(
        num_simulations=args.sims,
        mcts_batch_size=args.mcts_batch_size,
        # Heuristic value at MCTS leaves anchors search to a known-decent
        # signal while we still train the *network's* value head from real
        # game outcomes (in train_one_iteration).
        use_heuristic_value=bool(args.use_heuristic_value),
        use_batched_mcts=True,
        max_moves=args.max_moves,
        min_pins_to_keep=2,
    )

    # Track the best model seen so far (for revert-to-best).
    best_path = os.path.join(args.output_dir, "best_so_far.pt")
    best_pins = -1.0
    if os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_pins = float(ckpt.get("avg_pins", -1.0))
            print(f"Existing best_so_far.pt: avg_pins={best_pins:.2f}", flush=True)
        except Exception:
            best_pins = -1.0

    # Initial snapshot at iter 0 (so league pool isn't empty for iter 1).
    # Skip if resuming — snapshots are already on disk.
    if start_iter == 1:
        init_snap = _save_snapshot(network, snapshots_dir, 0)
        print(f"Initial snapshot → {init_snap}", flush=True)
    else:
        existing_snaps = _list_snapshots(snapshots_dir)
        print(f"Resuming with {len(existing_snaps)} existing snapshots in pool",
              flush=True)

    log_f = open(log_path, "a")

    for it in range(start_iter, args.iterations + 1):
        t_iter = time.time()
        snap_paths = _list_snapshots(snapshots_dir)
        # League delay: for the first `league_delay` iters, do pure self-play
        # only. This lets the model adapt to its own data before being mixed
        # against snapshots that may include early-iter (potentially worse)
        # versions of itself.
        effective_league_frac = (
            0.0 if it <= args.league_delay else args.league_fraction
        )
        print(f"\n=== Iteration {it}/{args.iterations} ===", flush=True)
        print(f"  snapshots in pool: {len(snap_paths)}; "
              f"sims={args.sims}; league_frac={effective_league_frac:.2f}",
              flush=True)

        t0 = time.time()
        samples, stats = generate_iteration_serial(
            cur_net=network,
            snapshot_paths=snap_paths,
            net_cfg=net_cfg,
            sp_cfg=sp_cfg,
            games_per_iter=args.games_per_iter,
            sharpen_temp=0.3,
            league_fraction=effective_league_frac,
            device=device,
        )
        gen_time = time.time() - t0
        print(f"  gen done: {len(samples)} samples in {gen_time:.0f}s "
              f"({stats['completed']} games, {stats['self_play']} SP, "
              f"{stats['league_play']} league, "
              f"current_wins_in_league={stats['current_wins_in_league']}/"
              f"{stats['league_play']})", flush=True)
        print(f"  per-N: {stats['per_n']}", flush=True)

        if not samples:
            print("  ⚠ no samples this iteration", flush=True)
            continue

        t0 = time.time()
        train_stats = train_one_iteration(
            network, samples,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            value_loss_weight=args.value_loss_weight,
        )
        train_time = time.time() - t0
        last_ep = train_stats["epochs"][-1]
        print(f"  train: {train_stats['n_samples']} samples × {args.epochs} ep "
              f"in {train_time:.0f}s; final pi={last_ep['pi_loss']:.4f} "
              f"v={last_ep['v_loss']:.4f}", flush=True)

        # Eval FIRST so we can gate the snapshot on it.
        eval_result = None
        if it % args.eval_every == 0:
            t0 = time.time()
            eval_result = quick_eval_vs_greedy(network, sp_cfg, num_games=6, n_players=2)
            print(f"  eval vs greedy (n=2, 6 games): {eval_result['avg_pins']:.1f} "
                  f"avg pins, {eval_result['wins']}/6 wins, "
                  f"{eval_result['truncated']}/6 trunc "
                  f"(in {time.time()-t0:.0f}s)", flush=True)

            # Update best-so-far if this iter is better.
            if eval_result["avg_pins"] > best_pins:
                best_pins = float(eval_result["avg_pins"])
                network.save_checkpoint(
                    best_path, iteration=it,
                    extra={"encoder_mode": "multicolour",
                            "avg_pins": best_pins},
                )
                print(f"  new best: {best_pins:.2f} pins → {best_path}",
                      flush=True)

            # Revert if degraded too far below threshold.
            if args.revert_pins >= 0 and eval_result["avg_pins"] < args.revert_pins:
                print(f"  ⚠ avg_pins {eval_result['avg_pins']:.2f} < "
                      f"{args.revert_pins:.2f} threshold; "
                      f"REVERTING to best_so_far.pt", flush=True)
                if os.path.exists(best_path):
                    network.load_checkpoint(best_path)

        # Snapshot — but only if it passes the gate (or gate disabled).
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

        latest = os.path.join(args.output_dir, "latest.pt")
        network.save_checkpoint(latest, iteration=it,
                                extra={"encoder_mode": "multicolour"})

        log_f.write(json.dumps({
            "iteration": it, "samples": len(samples), "gen_time": gen_time,
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
