"""MCTS-enhanced warmstart data generator.

Plays N-player heuristic-vs-heuristic games (so the trajectories are diverse,
covering the kinds of positions players actually reach), but labels each
position's policy target with **MCTS visit distributions** from a strong
network (d22) instead of the heuristic softmax.

Why: the d22 sharpening trick gave +2 pins by giving the policy head sharper
targets. MCTS at 200 sims should produce even sharper, strategically deeper
targets than the noisy_heuristic_policy (which is just softmax-of-heuristic-
scores + Dirichlet noise). The d22 model itself does the searching; we just
record what it would do at every position.

Multi-N coverage: weighted N=2..6 like the existing multi-colour generator.
Uses encoder_v2 (multicolour) so this data is compatible with d22-class models.

Output: same .npz schema (obs, action_masks, policies, values) so existing
pretrain_on_warmstart works as-is.
"""
import io
import os
import sys
import time
import math
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional


PRIMARY_COLOURS = ['red', 'lawn green', 'yellow']
COMPLEMENT = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}


def _pick_colours(n: int) -> List[str]:
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
class MCTSWarmConfig:
    num_games: int = 1500
    max_moves: int = 80
    min_pins_to_keep: int = 2
    n_choices: tuple = (2, 3, 4, 5, 6)
    n_weights: tuple = (3, 2, 2, 2, 1)
    mcts_sims: int = 200
    mcts_batch_size: int = 8
    sharpen_temp: float = 0.3      # final sharpening on top of MCTS visits
    use_heuristic_value: bool = True
    output_dir: str = "experiments/exp_d25_mcts_warmstart"


def _sharpen(p: np.ndarray, mask: np.ndarray, temp: float) -> np.ndarray:
    out = np.zeros_like(p)
    legal = p[mask]
    if legal.sum() == 0:
        out[mask] = 1.0 / max(int(mask.sum()), 1)
        return out
    logits = np.log(legal + 1e-12)
    sharp = np.exp(logits / temp)
    sharp /= sharp.sum()
    out[mask] = sharp
    return out


def _compute_value_per_colour(board, colours, winner, move_counts):
    """Same logic as multi_n_self_play._compute_value_per_colour."""
    from src.search.mcts import _score_colour as _sc
    n = len(colours)
    if winner is not None:
        out = {}
        for c in colours:
            out[c] = 1.0 if c == winner else -1.0 / max(n - 1, 1)
        return out
    scores = {c: _sc(board, c) for c in colours}
    out = {}
    for c in colours:
        my = scores[c]
        best_other = max((s for cc, s in scores.items() if cc != c), default=my)
        out[c] = max(-1.0, min(1.0, (my - best_other) / 1100.0))
    return out


def _make_proxy_env_mc(board, colour, step_count, max_steps, turn_order, encoder, mapper):
    from src.env.chinese_checkers_env import ChineseCheckersEnv
    proxy = ChineseCheckersEnv.__new__(ChineseCheckersEnv)
    proxy.render_mode = None
    proxy.max_steps = max_steps
    proxy.observation_space = None
    proxy.action_space = type('Space', (), {'n': 1210})()
    proxy._encoder = encoder
    proxy._mapper = mapper
    proxy._AGENT_COLOUR = colour
    next_opp = colour
    if turn_order and colour in turn_order:
        i = turn_order.index(colour)
        for off in range(1, len(turn_order)):
            cand = turn_order[(i + off) % len(turn_order)]
            if cand != colour:
                next_opp = cand; break
    proxy._OPPONENT_COLOUR = next_opp
    proxy._TURN_ORDER = list(turn_order)
    proxy._no_opponent = True
    proxy._opponent_policy = None
    proxy._board = board.clone()
    proxy._step_count = step_count
    proxy._terminated = False
    proxy._truncated = False
    return proxy


def _play_one_game_mcts_labelled(network, n_players, cfg, encoder, mapper):
    """Heuristic-vs-heuristic game; record MCTS visit distributions per move."""
    from src.env.board_wrapper import BoardWrapper
    from src.search.batched_mcts import BatchedAlphaZeroMCTS
    from src.training.warmstart_generator import _noisy_heuristic_policy

    colours = _pick_colours(n_players)
    board = BoardWrapper(colours)
    max_total_steps = cfg.max_moves * n_players

    mcts = BatchedAlphaZeroMCTS(
        network=network, num_simulations=cfg.mcts_sims,
        batch_size=cfg.mcts_batch_size, dirichlet_epsilon=0.0,
        use_heuristic_value=cfg.use_heuristic_value,
    )

    trajectories = []
    step_count = 0
    move_counts = {c: 0 for c in colours}
    winner = None

    while step_count < max_total_steps:
        colour = colours[step_count % n_players]
        legal = board.get_legal_moves(colour)
        if not legal:
            break

        # Run MCTS to label this position
        proxy = _make_proxy_env_mc(board, colour, step_count, max_total_steps,
                                    colours, encoder, mapper)
        action_mask_raw = mapper.build_action_mask(legal)
        try:
            visit_probs_raw = mcts.get_action_probs(proxy, temperature=1.0)
        except Exception as e:
            break

        # Encode obs in canonical (rotated) frame, rotate mask + visits to match
        obs = encoder.encode_multicolour(board, colour, colours)
        k = encoder.k_to_red_frame(colour)
        mask_canon = encoder.rotate_action_distribution_k(
            action_mask_raw.astype(np.bool_), k
        ).astype(np.bool_)
        visits_canon = encoder.rotate_action_distribution_k(visit_probs_raw, k)

        # Sharpen
        sharp = _sharpen(visits_canon, mask_canon, cfg.sharpen_temp)

        trajectories.append({
            "colour": colour,
            "obs": obs.copy(),
            "mask": mask_canon.copy(),
            "policy": sharp.copy(),
        })

        # Move chosen by HEURISTIC (not MCTS) — keeps trajectories diverse
        try:
            pin_id, dest, _ = _noisy_heuristic_policy(
                board, colour, alpha=0.5, noise_frac=0.25, fast=True,
            )
        except ValueError:
            break

        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1
        if board.check_win(colour):
            winner = colour
            break

    # Per-colour value targets
    max_pins = max(board.pins_in_goal(c) for c in colours)
    if max_pins < cfg.min_pins_to_keep:
        return []

    values = _compute_value_per_colour(board, colours, winner, move_counts)
    samples = []
    for step in trajectories:
        samples.append({
            "obs": step["obs"],
            "mask": step["mask"],
            "policy": step["policy"],
            "value": values[step["colour"]],
        })
    return samples


def _worker_generate_chunk(args):
    """Run a chunk of games on CPU. Returns dict of arrays."""
    model_bytes, net_cfg_dict, cfg_dict, n_games, base_seed = args
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    random.seed(base_seed)
    np.random.seed(base_seed)
    from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
    from src.env.state_encoder import StateEncoder
    from src.env.action_mapper import ActionMapper

    nc = NetworkConfig(**net_cfg_dict)
    net = AlphaZeroNet(nc, device="cpu")
    sd = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    net.model.load_state_dict(sd)
    net.model.eval()

    enc = StateEncoder(grid_size=17, num_channels=10, mode="multicolour")
    mapper = ActionMapper(num_pins=10, num_cells=121)
    cfg = MCTSWarmConfig(**cfg_dict)

    out_obs = []; out_mask = []; out_pol = []; out_val = []
    n_weights_total = sum(cfg.n_weights)
    for gi in range(n_games):
        r = random.random() * n_weights_total
        cum = 0
        n = cfg.n_choices[0]
        for nc_, w in zip(cfg.n_choices, cfg.n_weights):
            cum += w
            if r < cum:
                n = nc_; break
        try:
            samples = _play_one_game_mcts_labelled(net, n, cfg, enc, mapper)
        except Exception:
            continue
        for s in samples:
            out_obs.append(s["obs"])
            out_mask.append(s["mask"])
            out_pol.append(s["policy"])
            out_val.append(s["value"])
    return {
        "obs": np.stack(out_obs).astype(np.float32) if out_obs else np.zeros((0,10,17,17), dtype=np.float32),
        "action_masks": np.stack(out_mask).astype(np.bool_) if out_mask else np.zeros((0,1210), dtype=np.bool_),
        "policies": np.stack(out_pol).astype(np.float32) if out_pol else np.zeros((0,1210), dtype=np.float32),
        "values": np.array(out_val, dtype=np.float32),
    }


def generate_parallel(network, num_games: int, cfg: MCTSWarmConfig,
                       num_workers: int = 16, base_seed: int = 42) -> dict:
    """Split num_games across workers, aggregate."""
    from dataclasses import asdict
    import torch as _torch

    buf = io.BytesIO()
    _torch.save(network.model.state_dict(), buf)
    model_bytes = buf.getvalue()
    net_cfg_dict = asdict(network.config)
    cfg_dict = asdict(cfg)

    per_worker = max(1, num_games // num_workers)
    jobs = []
    remaining = num_games
    for w in range(num_workers):
        n = per_worker if w < num_workers - 1 else remaining
        jobs.append((model_bytes, net_cfg_dict, cfg_dict, n, base_seed + w))
        remaining -= n
    print(f"[parallel] {num_workers} workers, {per_worker} games each "
          f"(total {num_games}, sims/move={cfg.mcts_sims})", flush=True)

    chunks = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_worker_generate_chunk, j): i for i, j in enumerate(jobs)}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            try:
                d = f.result()
                chunks.append(d)
                done += 1
                print(f"[parallel] worker {i} done ({done}/{num_workers}); "
                      f"+{d['obs'].shape[0]} samples", flush=True)
            except Exception as e:
                print(f"[parallel] worker {i} FAILED: {e}", flush=True)

    out = {
        "obs": np.concatenate([c["obs"] for c in chunks], axis=0),
        "action_masks": np.concatenate([c["action_masks"] for c in chunks], axis=0),
        "policies": np.concatenate([c["policies"] for c in chunks], axis=0),
        "values": np.concatenate([c["values"] for c in chunks], axis=0),
    }
    return out


def save_data(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)


def main():
    import argparse
    sys.path.insert(0, '/home/coder/reinforcement-learning-on-chinese-checkers')
    sys.path.insert(0, '/home/coder/reinforcement-learning-on-chinese-checkers/multi system single machine minimal')
    from src.network.alphazero_net import AlphaZeroNet, NetworkConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", required=True)
    ap.add_argument("--num-games", type=int, default=1500)
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--mcts-sims", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = MCTSWarmConfig(
        num_games=args.num_games,
        max_moves=args.max_moves,
        mcts_sims=args.mcts_sims,
        output_dir=args.output_dir,
    )

    # Load source model on CPU; we'll serialize and pass to workers
    nc = NetworkConfig(num_blocks=9, num_filters=96)
    net = AlphaZeroNet(nc, device="cpu")
    net.load_checkpoint(args.source_model)
    print(f"Loaded source: {args.source_model}", flush=True)

    data = generate_parallel(net, args.num_games, cfg,
                              num_workers=args.num_workers, base_seed=args.seed)
    out_path = os.path.join(args.output_dir, "warmstart_data.npz")
    save_data(data, out_path)
    print(f"Saved {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB), "
          f"{data['obs'].shape[0]} samples", flush=True)


if __name__ == "__main__":
    main()
