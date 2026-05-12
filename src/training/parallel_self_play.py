"""Parallel self-play game generation.

Spawns N worker processes, each running independent self-play games using a
shared model checkpoint loaded from disk. Each worker writes its samples to
its own .npz chunk; master merges them at the end of an iteration.

Why this isn't in d27_league_train.py:
  - That trainer is single-process by design (the GPU is shared, can't easily
    have multiple procs each loading the network).
  - We saw container crashes from too many concurrent CPU procs.
  - Here, with careful 4-worker setup pinned to cores 0-3 and explicit thread
    caps, we get ~3-4× the game throughput on the same GPU (24% util previously).

Each worker:
  - Loads `current_ckpt_path` and (optionally) `opponent_ckpt_path` ONCE on start
  - Plays `games_per_worker` games (mix of self-play and league)
  - Saves all samples to `output_dir/chunk_<worker_id>.npz`
  - Returns sample counts + winner stats

Usage (from a trainer driver):
    from src.training.parallel_self_play import run_iteration_parallel
    samples, stats = run_iteration_parallel(
        current_ckpt="latest.pt",
        opponent_ckpts=["snap_iter015.pt"],
        net_cfg_dict=asdict(NetworkConfig(num_blocks=9, num_filters=96)),
        sp_cfg_dict=asdict(SelfPlayConfig(num_simulations=50, ...)),
        num_workers=4,
        games_per_worker=8,  # total 32 games
        league_fraction=0.3,
        output_dir="experiments/exp_d29/iter_001_chunks",
    )
"""
import os
# Pin BLAS threads BEFORE numpy is imported in this module.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import io
import math
import time
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

import numpy as np


def _worker_play_chunk(args):
    """Worker entry: play `games_per_worker` games and save samples to disk.

    args = (worker_id, current_ckpt, opponent_ckpts, net_cfg_dict, sp_cfg_dict,
            n_choices, n_weights, sharpen_temp, league_fraction,
            games_per_worker, max_moves, seed, use_gumbel,
            num_considered_actions, output_dir)
    """
    import os
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

    import sys
    sys.path.insert(0, "/home/coder/reinforcement-learning-on-chinese-checkers")

    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    (worker_id, current_ckpt, opponent_ckpts, net_cfg_dict, sp_cfg_dict,
     n_choices, n_weights, sharpen_temp, league_fraction,
     games_per_worker, max_moves, seed, use_gumbel,
     num_considered_actions, output_dir,
     win_filter_min_pins, max_attempts_per_game,
     heuristic_opponent, per_colour_min_pins) = args

    random.seed(seed); np.random.seed(seed)

    from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
    from src.training.alphazero_self_play import SelfPlayConfig
    from src.training.multi_n_self_play import (
        _pick_colours, _make_proxy_env_mc, _create_mcts_engine_mc,
        _ENCODER_MC, _MAPPER, _sharpen, _compute_value_per_colour,
    )
    from src.env.board_wrapper import BoardWrapper
    # Heuristic opponent (advanced/greedy). Loaded only if needed.
    if heuristic_opponent == "advanced":
        from src.agents.advanced_heuristic import advanced_heuristic_policy as _opp_fn
    elif heuristic_opponent == "greedy":
        from src.agents.greedy_agent import greedy_policy as _opp_fn
    else:
        _opp_fn = None

    # Workers go on GPU — V100 32GB has ample headroom for 4-8 procs at <1GB each.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net_cfg = NetworkConfig(**net_cfg_dict)
    cur_net = AlphaZeroNet(net_cfg, device=device)
    cur_net.load_checkpoint(current_ckpt)
    cur_net.model.eval()

    # Opponent networks (one per snapshot path; share if same path)
    opp_nets = {}
    for p in (opponent_ckpts or []):
        if p in opp_nets:
            continue
        n = AlphaZeroNet(net_cfg, device=device)
        n.load_checkpoint(p)
        n.model.eval()
        opp_nets[p] = n

    sp_cfg = SelfPlayConfig(**sp_cfg_dict)
    eng_kwargs = {"use_gumbel": use_gumbel,
                  "num_considered_actions": num_considered_actions}
    cur_engine = _create_mcts_engine_mc(cur_net, sp_cfg, **eng_kwargs)
    opp_engines = {p: _create_mcts_engine_mc(n, sp_cfg, **eng_kwargs)
                   for p, n in opp_nets.items()}

    n_w_total = sum(n_weights)
    samples = []
    stats = {"completed": 0, "discarded": 0, "self_play": 0, "league_play": 0,
             "current_wins_in_league": 0, "winner_breakdown": {},
             "per_n": {n: 0 for n in n_choices},
             "kept": 0, "filtered_low_pins": 0, "filtered_no_winner": 0,
             "attempts": 0}

    games_kept = 0
    while games_kept < games_per_worker and stats["attempts"] < max_attempts_per_game * games_per_worker:
        stats["attempts"] += 1
        gi = games_kept
        r = random.random() * n_w_total
        cum = 0
        n = n_choices[0]
        for nc, w in zip(n_choices, n_weights):
            cum += w
            if r < cum:
                n = nc; break

        # Choose opponent. With heuristic_opponent set, the opponent is
        # always the heuristic — no league_fraction needed (ignored).
        if _opp_fn is not None and league_fraction > 0:
            mode = "vs_heuristic"
            opp_engine = None  # heuristic doesn't need an engine
        elif opponent_ckpts and random.random() < league_fraction:
            opp_path = random.choice(opponent_ckpts)
            opp_engine = opp_engines[opp_path]
            mode = "league"
        else:
            opp_engine = None
            mode = "self"

        colours = _pick_colours(n)
        if mode == "self":
            role_map = {c: "current" for c in colours}
            engines = {c: cur_engine for c in colours}
        elif mode == "vs_heuristic":
            cur_colour = random.choice(colours)
            role_map = {c: ("current" if c == cur_colour else "opponent")
                        for c in colours}
            engines = {c: cur_engine for c in colours}  # opp uses _opp_fn fast-path
        else:  # league
            cur_colour = random.choice(colours)
            role_map = {c: ("current" if c == cur_colour else "opponent")
                        for c in colours}
            engines = {c: (cur_engine if role_map[c] == "current"
                          else opp_engine)
                       for c in colours}

        board = BoardWrapper(colours)
        max_total_steps = max_moves * n
        traj = []
        sc = 0
        move_counts = {c: 0 for c in colours}
        winner = None

        while sc < max_total_steps:
            colour = colours[sc % n]
            legal = board.get_legal_moves(colour)
            if not legal:
                break

            # Heuristic-opponent fast path: opponent makes a deterministic
            # heuristic move with no MCTS, no network. We don't record
            # training samples for opponent moves anyway.
            if _opp_fn is not None and role_map[colour] == "opponent":
                try:
                    pin_id, dest = _opp_fn(board, colour)
                except Exception:
                    pin_id = next(iter(legal.keys())); dest = legal[pin_id][0]
                board.apply_move(colour, pin_id, dest)
                sc += 1
                move_counts[colour] += 1
                if board.check_win(colour):
                    winner = colour; break
                continue

            proxy = _make_proxy_env_mc(board, colour, sc, max_total_steps,
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
            policy_canon = _ENCODER_MC.rotate_action_distribution_k(
                action_probs, k
            )
            sharp = _sharpen(policy_canon, mask_canon, sharpen_temp)
            if role_map[colour] == "current":
                traj.append({
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
            sc += 1
            move_counts[colour] += 1
            if board.check_win(colour):
                winner = colour; break

        max_pins = max(board.pins_in_goal(c) for c in colours)
        if max_pins < sp_cfg.min_pins_to_keep:
            stats["discarded"] += 1
            continue

        # Win-filter: only keep games where someone won OR a player got
        # near the goal. This biases training data toward decisive
        # endgames, which is what the value head + endgame policy needs.
        # win_filter_min_pins=0 disables the filter.
        if win_filter_min_pins > 0:
            if winner is None and max_pins < win_filter_min_pins:
                stats["filtered_low_pins"] += 1
                continue

        values = _compute_value_per_colour(board, colours, winner, move_counts)
        lam = sp_cfg.value_target_lambda
        # Per-colour decisive filter: only keep steps from colour c if c won
        # OR c reached at least per_colour_min_pins. Drops "losing trajectory"
        # data so the model learns from winners and near-winners only.
        # Note: this is on top of the game-level win_filter_min_pins.
        per_colour_pins = {c: board.pins_in_goal(c) for c in colours}
        kept_per_colour = 0
        dropped_per_colour = 0
        for step in traj:
            c = step["colour"]
            if per_colour_min_pins > 0:
                c_won = (winner == c)
                c_close = per_colour_pins.get(c, 0) >= per_colour_min_pins
                if not (c_won or c_close):
                    dropped_per_colour += 1
                    continue
            v_outcome = values[c]
            v_mcts = step["mcts_value"]
            v = max(-1.0, min(1.0, lam * v_outcome + (1.0 - lam) * v_mcts))
            samples.append({
                "obs": step["obs"],
                "action_mask": step["action_mask"],
                "policy_target": step["policy_target"],
                "value_target": float(v),
            })
            kept_per_colour += 1
        stats["per_colour_kept"] = stats.get("per_colour_kept", 0) + kept_per_colour
        stats["per_colour_dropped"] = stats.get("per_colour_dropped", 0) + dropped_per_colour
        stats["completed"] += 1
        stats["kept"] += 1
        games_kept += 1
        stats["per_n"][n] += 1
        if mode == "self":
            stats["self_play"] += 1
        else:
            # Both "league" and "vs_heuristic" use league_play counter so we
            # see win rate against the chosen opponent.
            stats["league_play"] += 1
            if winner is not None and role_map.get(winner) == "current":
                stats["current_wins_in_league"] += 1
        w = winner or "none"
        stats["winner_breakdown"][w] = stats["winner_breakdown"].get(w, 0) + 1

    # Save chunk to disk
    os.makedirs(output_dir, exist_ok=True)
    chunk_path = os.path.join(output_dir, f"chunk_{worker_id:03d}.npz")
    if samples:
        obs_arr = np.stack([s["obs"] for s in samples]).astype(np.float32)
        mask_arr = np.stack([s["action_mask"] for s in samples]).astype(np.bool_)
        pol_arr = np.stack([s["policy_target"] for s in samples]).astype(np.float32)
        val_arr = np.array([s["value_target"] for s in samples], dtype=np.float32)
        np.savez_compressed(chunk_path, obs=obs_arr, action_masks=mask_arr,
                            policies=pol_arr, values=val_arr)
    else:
        # Empty chunk file so master knows worker finished
        np.savez_compressed(chunk_path, obs=np.zeros((0,10,17,17), dtype=np.float32),
                            action_masks=np.zeros((0,1210), dtype=np.bool_),
                            policies=np.zeros((0,1210), dtype=np.float32),
                            values=np.zeros((0,), dtype=np.float32))

    return {"chunk_path": chunk_path, "stats": stats,
            "n_samples": len(samples)}


def run_iteration_parallel(
    current_ckpt: str,
    opponent_ckpts: list[str],
    net_cfg_dict: dict,
    sp_cfg_dict: dict,
    output_dir: str,
    num_workers: int = 4,
    games_per_worker: int = 8,
    n_choices: tuple = (2, 3, 4, 5, 6),
    n_weights: tuple = (3, 2, 2, 2, 1),
    sharpen_temp: float = 0.3,
    league_fraction: float = 0.3,
    max_moves: int = 80,
    base_seed: int = 0,
    use_gumbel: bool = False,
    num_considered_actions: int = 16,
    win_filter_min_pins: int = 0,
    max_attempts_per_game: int = 5,
    heuristic_opponent: Optional[str] = None,
    per_colour_min_pins: int = 0,
):
    """Run one iteration of parallel self-play. Returns (samples_list, stats).

    win_filter_min_pins: if > 0, only keep games where someone won OR
        max_pins >= this threshold. Biases training data toward decisive
        endgames so the value head learns from real outcomes.
    max_attempts_per_game: cap on retries when win-filter rejects a game.
        With 4 attempts per slot worker tries up to 4× games_per_worker.
    heuristic_opponent: "advanced" or "greedy" to play vs that fixed
        heuristic in every league game. None disables (uses snapshot
        league as before).
    per_colour_min_pins: if > 0, only emit training samples from colour c
        when c won OR c.pins_in_goal >= this. Drops losing-trajectory data
        per colour. 7 means "we only train on the winner's trajectory or
        a colour that got close enough to be worth imitating."
    """
    args_list = []
    for w in range(num_workers):
        args_list.append((
            w, current_ckpt, list(opponent_ckpts or []), net_cfg_dict,
            sp_cfg_dict, n_choices, n_weights, sharpen_temp, league_fraction,
            games_per_worker, max_moves, base_seed + w * 1000, use_gumbel,
            num_considered_actions, output_dir,
            int(win_filter_min_pins), int(max_attempts_per_game),
            heuristic_opponent, int(per_colour_min_pins),
        ))

    chunks_paths = []
    agg = {"completed": 0, "discarded": 0, "self_play": 0, "league_play": 0,
           "current_wins_in_league": 0, "winner_breakdown": {},
           "per_n": {n: 0 for n in n_choices},
           "kept": 0, "filtered_low_pins": 0, "filtered_no_winner": 0,
           "attempts": 0, "per_colour_kept": 0, "per_colour_dropped": 0}

    def _accumulate(wid, r):
        nonlocal agg
        chunks_paths.append(r["chunk_path"])
        s = r["stats"]
        agg["completed"] += s["completed"]
        agg["discarded"] += s["discarded"]
        agg["self_play"] += s["self_play"]
        agg["league_play"] += s["league_play"]
        agg["current_wins_in_league"] += s["current_wins_in_league"]
        agg["kept"] += s.get("kept", 0)
        agg["filtered_low_pins"] += s.get("filtered_low_pins", 0)
        agg["filtered_no_winner"] += s.get("filtered_no_winner", 0)
        agg["attempts"] += s.get("attempts", 0)
        agg["per_colour_kept"] += s.get("per_colour_kept", 0)
        agg["per_colour_dropped"] += s.get("per_colour_dropped", 0)
        for n, c in s["per_n"].items():
            agg["per_n"][n] = agg["per_n"].get(n, 0) + c
        for w_, c in s["winner_breakdown"].items():
            agg["winner_breakdown"][w_] = \
                agg["winner_breakdown"].get(w_, 0) + c

    if num_workers <= 1 and os.environ.get("INLINE_WORKER"):
        # Inline mode: run "workers" sequentially in the main process.
        # No subprocess spawning means no extra CUDA contexts — useful when
        # the GPU is shared and worker spawn fails with OOM.
        done = 0
        for a in args_list:
            wid = a[0]
            try:
                r = _worker_play_chunk(a)
                done += 1
                _accumulate(wid, r)
                print(f"  [worker {wid}] done ({done}/{max(num_workers,1)}), "
                      f"+{r['n_samples']} samples → {r['chunk_path']}",
                      flush=True)
            except Exception as e:
                print(f"  [worker {wid}] FAILED: {e}", flush=True)
    else:
        mp_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
            futs = {ex.submit(_worker_play_chunk, a): a[0] for a in args_list}
            done = 0
            for f in as_completed(futs):
                wid = futs[f]
                try:
                    r = f.result()
                    done += 1
                    _accumulate(wid, r)
                    print(f"  [worker {wid}] done ({done}/{num_workers}), "
                          f"+{r['n_samples']} samples → {r['chunk_path']}",
                          flush=True)
                except Exception as e:
                    print(f"  [worker {wid}] FAILED: {e}", flush=True)

    # Merge chunks into 4 big numpy arrays (NOT a list of dicts — that
    # was the memory bomb in d29 attempt 1: indexing each npz row created
    # views that pinned the underlying memmap, ballooning RSS to ~94GB).
    # Loading whole arrays then concatenating uses ~67MB total for 5k samples.
    obs_list = []; mask_list = []; pol_list = []; val_list = []
    for cp in sorted(chunks_paths):
        try:
            d = np.load(cp)
            # .copy() forces eager load and detaches from the memmap.
            obs_list.append(d["obs"][:].copy())
            mask_list.append(d["action_masks"][:].copy())
            pol_list.append(d["policies"][:].copy())
            val_list.append(d["values"][:].copy())
            d.close()
        except Exception as e:
            print(f"  ⚠ couldn't load {cp}: {e}", flush=True)

    if not obs_list:
        return {"obs": np.zeros((0,10,17,17), dtype=np.float32),
                "action_masks": np.zeros((0,1210), dtype=np.bool_),
                "policies": np.zeros((0,1210), dtype=np.float32),
                "values": np.zeros((0,), dtype=np.float32)}, agg

    return {"obs": np.concatenate(obs_list, axis=0),
            "action_masks": np.concatenate(mask_list, axis=0),
            "policies": np.concatenate(pol_list, axis=0),
            "values": np.concatenate(val_list, axis=0)}, agg
