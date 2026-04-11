#!/usr/bin/env python3
"""
run_alphazero.py — Main entry point for AlphaZero training pipeline.

Supports three modes:
  1. warmstart   — Generate heuristic self-play data and pre-train
  2. train       — Full AlphaZero self-play training loop
  3. evaluate    — Evaluate a checkpoint in arena

Examples:
  # Generate warm-start data and pre-train
  python scripts/run_alphazero.py warmstart --num-games 5000

  # Train from warm-started checkpoint
  python scripts/run_alphazero.py train --resume checkpoints/warmstart/best_model.pt

  # Train from scratch with Gumbel MCTS
  python scripts/run_alphazero.py train --mcts gumbel --sims 32

  # Evaluate a checkpoint
  python scripts/run_alphazero.py evaluate --checkpoint checkpoints/alphazero/best_model.pt
"""

import os
import sys
import argparse
import json
import time

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_train import TrainingConfig, train_alphazero, evaluate_model
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.warmstart_generator import (
    WarmStartConfig,
    generate_warmstart_data,
    save_warmstart_data,
    load_warmstart_data,
    pretrain_on_warmstart,
)


def cmd_warmstart(args):
    """Generate warm-start data and pre-train the network."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    # Step 1: Generate data (or load existing)
    data_path = os.path.join(args.output, "warmstart_data.npz")
    if os.path.exists(data_path) and not args.regenerate:
        print(f"Loading existing warm-start data from {data_path}")
        data = load_warmstart_data(data_path)
        print(f"  Loaded {data['obs'].shape[0]} samples")
    else:
        print(f"Generating {args.num_games} warm-start games...")
        config = WarmStartConfig(
            num_games=args.num_games,
            max_moves=args.max_moves,
            output_dir=args.output,
        )
        data = generate_warmstart_data(config)
        save_warmstart_data(data, args.output)

    # Step 2: Pre-train
    print(f"\nPre-training network on {data['obs'].shape[0]} samples...")
    net_config = NetworkConfig(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.n_heads,
        use_auxiliary_head=args.use_auxiliary_head,
    )
    network = AlphaZeroNet(net_config, device=device)
    print(f"  Network: {network.parameter_count():,} parameters")

    log = pretrain_on_warmstart(
        network,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stop_patience=args.patience,
    )

    # Save
    os.makedirs(args.output, exist_ok=True)
    checkpoint_path = os.path.join(args.output, "warmstart_model.pt")
    network.save_checkpoint(checkpoint_path, iteration=0, extra={"pretrain_log": log})
    print(f"\nSaved warm-started model to {checkpoint_path}")

    # Quick eval
    print("\nEvaluating warm-started model...")
    results = evaluate_model(network, num_games=10, max_steps=300, use_mcts=False)
    n = 10
    print(f"  vs Random:   pins={results['vs_random']['avg_pins_in_goal']:.1f}, "
          f"score={results['vs_random']['avg_tournament_score']:.1f}, "
          f"wins={results['vs_random']['agent_wins']}/{n}")
    print(f"  vs Greedy:   pins={results['vs_greedy']['avg_pins_in_goal']:.1f}, "
          f"score={results['vs_greedy']['avg_tournament_score']:.1f}, "
          f"wins={results['vs_greedy']['agent_wins']}/{n}")
    print(f"  vs Advanced: pins={results['vs_advanced']['avg_pins_in_goal']:.1f}, "
          f"score={results['vs_advanced']['avg_tournament_score']:.1f}, "
          f"wins={results['vs_advanced']['agent_wins']}/{n}")

    # Save log
    with open(os.path.join(args.output, "warmstart_log.json"), "w") as f:
        json.dump({"pretrain_log": log, "eval_results": results}, f, indent=2, default=str)


def cmd_train(args):
    """Run full AlphaZero training loop."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    net_config = NetworkConfig(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        lr=args.lr,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.n_heads,
        use_auxiliary_head=args.use_auxiliary_head,
    )

    sp_config = SelfPlayConfig(
        num_simulations=args.sims,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        temperature_moves=args.temp_moves,
        max_moves=args.max_moves,
        use_heuristic_value=args.heuristic_value,
        augment_symmetry=not args.no_augment,
    )

    train_config = TrainingConfig(
        network=net_config,
        self_play=sp_config,
        games_per_iteration=args.games_per_iter,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs_per_iter,
        replay_buffer_size=args.buffer_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        num_iterations=args.iterations,
        eval_games=args.eval_games,
        win_threshold=args.win_threshold,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    print("AlphaZero Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Architecture: ResNet {args.num_blocks}x{args.num_filters}")
    print(f"  MCTS: {args.mcts} with {args.sims} simulations")
    print(f"  Games/iteration: {args.games_per_iter}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Resume from: {args.resume or 'scratch'}")

    train_alphazero(train_config, resume_from=args.resume)


def cmd_evaluate(args):
    """Evaluate a checkpoint."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    net_config = NetworkConfig(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.n_heads,
    )
    network = AlphaZeroNet(net_config, device=device)
    network.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Network: {network.parameter_count():,} parameters")

    # Use batched MCTS if requested
    use_batched = args.batched_mcts

    value_src = "heuristic" if args.heuristic_value else "network"
    print(f"\nEvaluating ({args.num_games} games, max_steps={args.max_steps}, "
          f"mcts={'batched' if use_batched else 'standard' if args.use_mcts else 'none'}, "
          f"sims={args.sims}, value={value_src})...")
    eval_start = time.time()
    results = evaluate_model(
        network,
        num_games=args.num_games,
        max_steps=args.max_steps,
        use_mcts=args.use_mcts,
        mcts_sims=args.sims,
        use_batched_mcts=use_batched,
        mcts_batch_size=args.mcts_batch_size,
        use_heuristic_value=args.heuristic_value,
    )
    eval_elapsed = time.time() - eval_start

    def _fmt_matchup(key: str, label: str) -> str:
        m = results[key]
        base = (
            f"  {label}: pins={m['avg_pins_in_goal']:.1f}, "
            f"score={m['avg_tournament_score']:.1f}, "
            f"wins={m['agent_wins']}/{args.num_games}"
        )
        # Move-count breakdown (new arena_summary fields, backward compatible)
        if m.get('agent_wins', 0) > 0 and 'avg_steps_win' in m:
            steps_line = (
                f" [win moves avg={m['avg_steps_win']:.0f}, "
                f"range {m['min_steps_win']}-{m['max_steps_win']}]"
            )
        elif 'avg_steps_truncated' in m and m.get('truncated', 0) > 0:
            steps_line = f" [truncated avg={m['avg_steps_truncated']:.0f}]"
        else:
            steps_line = f" [avg_steps={m.get('avg_steps', 0):.0f}]"
        return base + steps_line

    print()
    print(_fmt_matchup('vs_random', 'vs Random  '))
    print(_fmt_matchup('vs_greedy', 'vs Greedy  '))
    print(_fmt_matchup('vs_advanced', 'vs Advanced'))

    # Timing: total / per game / real per-move budget using observed move counts
    total_games = args.num_games * 3  # 3 matchups
    per_game = eval_elapsed / max(total_games, 1)

    # Prefer observed avg_steps across all matchups for realistic per-move timing.
    observed_steps = [
        results[k].get('avg_steps', 0) for k in ('vs_random', 'vs_greedy', 'vs_advanced')
    ]
    observed_steps = [s for s in observed_steps if s > 0]
    if observed_steps:
        real_avg_moves = sum(observed_steps) / len(observed_steps)
    else:
        real_avg_moves = args.max_steps
    per_move_ms = per_game / max(real_avg_moves, 1) * 1000

    print(
        f"\n  Timing: total={eval_elapsed:.1f}s, per_game={per_game:.2f}s, "
        f"~per_move={per_move_ms:.1f}ms (based on avg {real_avg_moves:.0f} moves/game)"
    )
    if args.use_mcts or args.batched_mcts:
        sims_per_sec = (args.sims / (per_move_ms / 1000)) if per_move_ms > 0 else 0.0
        sims_in_10s = sims_per_sec * 10.0

        # Tournament budget check: 60s / real_avg_moves per move
        tournament_budget_ms = 60_000 / max(real_avg_moves, 1)
        full_game_ms = per_move_ms * real_avg_moves
        budget_status = "OK" if full_game_ms <= 60_000 else "OVER"
        print(
            f"  MCTS: {args.sims} sims/move @ ~{sims_per_sec:.0f} sims/sec "
            f"(~{sims_in_10s:.0f} sims in 10s turn budget)"
        )
        print(
            f"  Tournament budget: {full_game_ms/1000:.1f}s/game "
            f"(budget=60s, per-move cap={tournament_budget_ms:.0f}ms) [{budget_status}]"
        )

    # Save results
    results["timing"] = {
        "total_s": eval_elapsed,
        "per_game_s": per_game,
        "per_move_ms_estimate": per_move_ms,
        "sims_per_move": args.sims if (args.use_mcts or args.batched_mcts) else 0,
        "value_source": value_src,
    }
    output = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output}")

    # Export if requested
    if args.export_onnx:
        onnx_path = args.checkpoint.replace('.pt', '.onnx')
        network.export_onnx(onnx_path)
    if args.export_torchscript:
        ts_path = args.checkpoint.replace('.pt', '.ts')
        network.export_torchscript(ts_path)


def cmd_enhanced_warmstart(args):
    """Generate MCTS-enhanced or endgame-focused warm-start data."""
    from src.training.enhanced_warmstart import (
        EnhancedWarmStartConfig,
        generate_mcts_warmstart_data,
        generate_endgame_data,
        save_enhanced_data,
    )

    config = EnhancedWarmStartConfig(
        num_games=args.num_games,
        max_moves=args.max_moves,
        mcts_simulations=args.mcts_sims,
        temperature=args.temperature,
        endgame_min_pins=args.endgame_min_pins,
        endgame_max_pins=args.endgame_max_pins,
        augment_symmetry=not args.no_augment,
        output_dir=args.output,
    )

    if args.mode in ("mcts", "both"):
        print(f"Generating {config.num_games} MCTS-enhanced warm-start games "
              f"({config.mcts_simulations} sims/move)...")
        data = generate_mcts_warmstart_data(config)
        save_enhanced_data(data, config.output_dir, prefix="mcts")

    if args.mode in ("endgame", "both"):
        print(f"\nGenerating {config.num_games} endgame-focused games "
              f"(start at {config.endgame_min_pins}-{config.endgame_max_pins} pins)...")
        data = generate_endgame_data(config)
        save_enhanced_data(data, config.output_dir, prefix="endgame")


def _add_arch_args(parser):
    """Add architecture arguments shared across subcommands."""
    parser.add_argument("--architecture", type=str, default="resnet",
                        choices=["resnet", "pin_transformer", "gateau"],
                        help="Network architecture")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Transformer/GATEAU hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Attention heads for transformer/GATEAU")
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument("--use-auxiliary-head", action="store_true",
                        help="Add pins_in_goal auxiliary prediction head")


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chinese Checkers Training")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- warmstart ---
    ws = subparsers.add_parser("warmstart", help="Generate warm-start data and pre-train")
    ws.add_argument("--num-games", type=int, default=5000)
    ws.add_argument("--max-moves", type=int, default=200)
    ws.add_argument("--output", type=str, default="experiments/exp_d3_warmstart")
    ws.add_argument("--regenerate", action="store_true", help="Regenerate data even if exists")
    ws.add_argument("--epochs", type=int, default=50)
    ws.add_argument("--batch-size", type=int, default=256)
    ws.add_argument("--lr", type=float, default=1e-3)
    ws.add_argument("--patience", type=int, default=5)
    _add_arch_args(ws)
    ws.add_argument("--cpu", action="store_true")

    # --- enhanced-warmstart ---
    ew = subparsers.add_parser("enhanced-warmstart",
                                help="MCTS-enhanced or endgame warm-start data")
    ew.add_argument("--mode", choices=["mcts", "endgame", "both"], default="mcts")
    ew.add_argument("--num-games", type=int, default=1000)
    ew.add_argument("--max-moves", type=int, default=200)
    ew.add_argument("--mcts-sims", type=int, default=50)
    ew.add_argument("--temperature", type=float, default=1.0)
    ew.add_argument("--endgame-min-pins", type=int, default=5)
    ew.add_argument("--endgame-max-pins", type=int, default=8)
    ew.add_argument("--output", type=str, default="experiments/enhanced_warmstart")
    ew.add_argument("--no-augment", action="store_true")

    # --- train ---
    tr = subparsers.add_parser("train", help="Run AlphaZero training loop")
    tr.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    tr.add_argument("--mcts", type=str, default="puct", choices=["puct", "gumbel", "batched"])
    tr.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    tr.add_argument("--mcts-batch-size", type=int, default=8,
                    help="Leaf batch size for batched MCTS")
    tr.add_argument("--c-puct", type=float, default=1.5)
    tr.add_argument("--dirichlet-alpha", type=float, default=0.3)
    tr.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    tr.add_argument("--temp-moves", type=int, default=30)
    tr.add_argument("--max-moves", type=int, default=200)
    tr.add_argument("--heuristic-value", action="store_true")
    tr.add_argument("--no-augment", action="store_true")
    tr.add_argument("--games-per-iter", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--epochs-per-iter", type=int, default=10)
    tr.add_argument("--buffer-size", type=int, default=50000)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--lr-decay", type=float, default=0.99)
    tr.add_argument("--iterations", type=int, default=30)
    tr.add_argument("--eval-games", type=int, default=20)
    tr.add_argument("--win-threshold", type=float, default=0.55)
    tr.add_argument("--checkpoint-dir", type=str, default="experiments/exp_d1_alphazero")
    _add_arch_args(tr)
    tr.add_argument("--cpu", action="store_true")

    # --- evaluate ---
    ev = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    ev.add_argument("--checkpoint", type=str, required=True)
    ev.add_argument("--num-games", type=int, default=20)
    ev.add_argument("--max-steps", type=int, default=300)
    ev.add_argument("--use-mcts", action="store_true")
    ev.add_argument("--batched-mcts", action="store_true",
                    help="Use batched MCTS (faster on GPU)")
    ev.add_argument("--sims", type=int, default=50)
    ev.add_argument("--heuristic-value", action="store_true",
                    help="Use heuristic leaf evaluation instead of network value head")
    ev.add_argument("--mcts-batch-size", type=int, default=8)
    _add_arch_args(ev)
    ev.add_argument("--cpu", action="store_true")
    ev.add_argument("--export-onnx", action="store_true", help="Export to ONNX format")
    ev.add_argument("--export-torchscript", action="store_true", help="Export to TorchScript")

    args = parser.parse_args()

    if args.command == "warmstart":
        cmd_warmstart(args)
    elif args.command == "enhanced-warmstart":
        cmd_enhanced_warmstart(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
