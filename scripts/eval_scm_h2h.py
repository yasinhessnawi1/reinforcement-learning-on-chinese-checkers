#!/usr/bin/env python3
"""
eval_scm_h2h.py - Head-to-head evaluation of two SCM models in multiplayer games.

Compares native d41 SCM vs the existing SCM v2 (trained on d10) by having
each take the agent slot in N-player games against the same MCTS opponents.

Reports per-N: avg pins, wins, tournament score, with statistical tests.
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import torch.nn.functional as F

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.network.search_conditioned_modulator import (
    SCMConfig, SearchConditionedModulator, extract_search_features,
)
from src.training.scm_trainer import SCMTrainer
from src.training.alphazero_self_play import SelfPlayConfig
from src.search.batched_mcts import BatchedAlphaZeroMCTS
from src.env.board_wrapper import BoardWrapper


def load_scm(path: str, device: str) -> SearchConditionedModulator:
    scm = SearchConditionedModulator(SCMConfig(hidden_dim=128, max_shift=2.0))
    trainer = SCMTrainer(scm, device=device)
    trainer.load_checkpoint(path)
    scm.to(torch.device(device))
    scm.eval()
    return scm


def play_game(
    network: AlphaZeroNet,
    scm: SearchConditionedModulator | None,
    n_players: int,
    config: SelfPlayConfig,
    blend_alpha: float,
    device: str,
) -> dict:
    """Play one N-player game. SCM (if given) used by agent slot; opponents = plain MCTS.

    Returns dict with agent_pins, agent_wins (bool), tournament_score.
    """
    from src.training.multi_n_self_play import (
        _pick_colours, _make_proxy_env_mc, _tournament_score,
        _ENCODER_MC, _MAPPER,
    )

    colours = _pick_colours(n_players)
    agent_colour = colours[0]
    board = BoardWrapper(colours)
    max_total_steps = config.max_moves * n_players

    # Both agent and opponents use batched MCTS on GPU for speed
    agent_engine = BatchedAlphaZeroMCTS(
        network=network,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=0.0,
        use_heuristic_value=config.use_heuristic_value,
        opponent_policy=None,
        batch_size=8,
    )
    # Same engine for opponents (no SCM)
    opp_engine = agent_engine

    if scm is not None:
        hidden = scm.init_hidden(batch_size=1, device=device)
    else:
        hidden = None

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

        if colour == agent_colour and scm is not None:
            k = _ENCODER_MC.k_to_red_frame(colour)
            obs = _ENCODER_MC.encode_multicolour(board, colour, colours)
            action_mask_raw = _MAPPER.build_action_mask(legal)
            mask_canon = (
                _ENCODER_MC.rotate_action_distribution_k(action_mask_raw.astype(np.bool_), k).astype(np.bool_)
                if k != 0 else action_mask_raw.astype(np.bool_)
            )

            root = agent_engine.run(proxy)
            mcts_probs_raw = agent_engine._visits_to_probs(root, 1210, action_mask_raw, 0.1)
            mcts_probs_canon = (
                _ENCODER_MC.rotate_action_distribution_k(mcts_probs_raw, k)
                if k != 0 else mcts_probs_raw
            )

            raw_logits, _, _ = network.predict_raw_logits(obs, mask_canon)
            raw_policy_canon = network.predict(obs, mask_canon)[0]
            stats = agent_engine._extract_stats_from_root(
                root, 1210, mask_canon, raw_policy=raw_policy_canon,
                turn_number=move_counts[colour],
            )
            features = extract_search_features(stats)

            features_t = torch.tensor(features[np.newaxis], dtype=torch.float32, device=device)
            logits_t = torch.tensor(raw_logits[np.newaxis], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_canon[np.newaxis], dtype=torch.bool, device=device)

            with torch.no_grad():
                modulated, gate, shift, hidden = scm.modulate_logits(logits_t, features_t, hidden)
                modulated = modulated.masked_fill(~mask_t, -1e9)
                scm_probs_canon = F.softmax(modulated, dim=-1).squeeze(0).cpu().numpy()

            blended_canon = (1.0 - blend_alpha) * mcts_probs_canon + blend_alpha * scm_probs_canon
            tot = blended_canon.sum()
            if tot > 0:
                blended_canon = blended_canon / tot
            else:
                blended_canon = mcts_probs_canon

            blended_raw = (
                _ENCODER_MC.rotate_action_distribution_k(blended_canon, 6 - k)
                if k != 0 else blended_canon
            )
            action = int(np.argmax(blended_raw))
        elif colour == agent_colour:
            # Baseline path: no SCM
            probs = opp_engine.get_action_probs(proxy, temperature=0.1)
            action = int(np.argmax(probs))
        else:
            probs = opp_engine.get_action_probs(proxy, temperature=0.1)
            action = int(np.argmax(probs))

        pin_id, dest = _MAPPER.decode(action)
        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1
        if board.check_win(colour):
            winner = colour
            break

    agent_pins = board.pins_in_goal(agent_colour)
    return {
        "agent_pins": agent_pins,
        "agent_won": (winner == agent_colour),
        "tournament_score": _tournament_score(board, agent_colour, move_counts[agent_colour]),
        "n_players": n_players,
    }


def run_condition(
    network, scm, n_players, num_games, config, alpha, device, label,
):
    results = []
    t0 = time.time()
    for i in range(num_games):
        out = play_game(network, scm, n_players, config, alpha, device)
        results.append(out)
        if (i + 1) % 5 == 0 or i == 0:
            pins = [r["agent_pins"] for r in results]
            wins = sum(r["agent_won"] for r in results)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (num_games - i - 1) / rate / 60
            print(f"    [{label} N={n_players}] {i+1}/{num_games}  "
                  f"avg_pins={np.mean(pins):.2f}  wins={wins}/{i+1}  "
                  f"ETA={eta:.1f}min", flush=True)
    return results


def summarize(results: list) -> dict:
    pins = np.array([r["agent_pins"] for r in results], dtype=float)
    wins = sum(r["agent_won"] for r in results)
    scores = np.array([r["tournament_score"] for r in results], dtype=float)
    return {
        "n": len(results),
        "avg_pins": float(pins.mean()),
        "std_pins": float(pins.std()),
        "max_pins": int(pins.max()),
        "wins": int(wins),
        "win_rate": wins / len(results) if results else 0.0,
        "avg_score": float(scores.mean()),
        "pins_list": pins.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="experiments/exp_d41_multiN/best_so_far.pt")
    p.add_argument("--scm-new", type=str, default="experiments/scm_d41_multi/scm_model.pt")
    p.add_argument("--scm-old", type=str, default="experiments/scm_v2/scm_model.pt")
    p.add_argument("--output", type=str, default="experiments/scm_d41_multi/h2h_results.json")
    p.add_argument("--num-games", type=int, default=20, help="games per N per SCM")
    p.add_argument("--n-list", type=str, default="2,3,4,5,6")
    p.add_argument("--blend-alpha", type=float, default=0.2)
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--max-moves", type=int, default=100)
    p.add_argument("--include-baseline", action="store_true",
                   help="Also run no-SCM baseline (slower)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    net = AlphaZeroNet(NetworkConfig(num_blocks=9, num_filters=96), device=device)
    net.load_checkpoint(args.checkpoint)
    print(f"Network: {args.checkpoint}")

    scm_new = load_scm(args.scm_new, device)
    print(f"SCM (new, d41): {args.scm_new}")
    scm_old = load_scm(args.scm_old, device)
    print(f"SCM (old, v2):  {args.scm_old}")

    config = SelfPlayConfig(
        num_simulations=args.sims,
        use_heuristic_value=True,
        max_moves=args.max_moves,
    )

    n_list = [int(x) for x in args.n_list.split(",")]
    all_results: dict = {"config": vars(args), "per_n": {}}

    for n in n_list:
        all_results["per_n"][str(n)] = {}

        if args.include_baseline:
            print(f"\n=== N={n}: Baseline (no SCM) ===")
            base = run_condition(net, None, n, args.num_games, config, 0.0, device, "baseline")
            all_results["per_n"][str(n)]["baseline"] = summarize(base)

        print(f"\n=== N={n}: SCM v2 (d10-trained), alpha={args.blend_alpha} ===")
        old_r = run_condition(net, scm_old, n, args.num_games, config, args.blend_alpha, device, "scm_v2")
        all_results["per_n"][str(n)]["scm_v2"] = summarize(old_r)

        print(f"\n=== N={n}: SCM new (d41-native), alpha={args.blend_alpha} ===")
        new_r = run_condition(net, scm_new, n, args.num_games, config, args.blend_alpha, device, "scm_new")
        all_results["per_n"][str(n)]["scm_new"] = summarize(new_r)

        # Quick H2H comparison
        try:
            from scipy.stats import ttest_ind, fisher_exact
            old_pins = [r["agent_pins"] for r in old_r]
            new_pins = [r["agent_pins"] for r in new_r]
            t_p = ttest_ind(new_pins, old_pins, equal_var=False).pvalue
            old_w = sum(r["agent_won"] for r in old_r)
            new_w = sum(r["agent_won"] for r in new_r)
            _, f_p = fisher_exact(
                [[new_w, args.num_games - new_w], [old_w, args.num_games - old_w]],
                alternative='two-sided',
            )
            all_results["per_n"][str(n)]["comparison"] = {
                "welch_p_pins": float(t_p),
                "fisher_p_wins": float(f_p),
                "delta_mean_pins": float(np.mean(new_pins) - np.mean(old_pins)),
            }
            print(f"  delta_pins(new-old)={np.mean(new_pins)-np.mean(old_pins):+.3f}  "
                  f"Welch p={t_p:.4f}  Fisher p={f_p:.4f}")
        except Exception as e:
            print(f"  Stats error: {e}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: new SCM (d41-native) vs old SCM (v2, d10-trained)")
    print("=" * 70)
    print(f"{'N':<4}{'Old avg':>10}{'New avg':>10}{'d_pins':>8}{'Old W':>7}{'New W':>7}{'p_pins':>10}")
    for n in n_list:
        d = all_results["per_n"][str(n)]
        oa = d["scm_v2"]["avg_pins"]
        na = d["scm_new"]["avg_pins"]
        ow = d["scm_v2"]["wins"]
        nw = d["scm_new"]["wins"]
        p = d.get("comparison", {}).get("welch_p_pins", float("nan"))
        print(f"{n:<4}{oa:>10.2f}{na:>10.2f}{na-oa:>+8.2f}{ow:>7}{nw:>7}{p:>10.4f}")


if __name__ == "__main__":
    main()
