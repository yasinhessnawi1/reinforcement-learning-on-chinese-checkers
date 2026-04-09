#!/usr/bin/env python3
"""
run_arch_comparison.py — Fair 3-way architecture comparison.

Generates warm-start data ONCE, then trains all 3 architectures on the
exact same data with the same hyperparameters. Results are saved to a
shared experiment directory for direct comparison.

Usage:
    # Full comparison (generates data + trains all 3)
    python scripts/run_arch_comparison.py --num-games 2000

    # Skip data generation (use existing data)
    python scripts/run_arch_comparison.py --data-path experiments/exp_d5_arch_compare/warmstart_data.npz

    # Train only specific architectures
    python scripts/run_arch_comparison.py --architectures resnet pin_transformer
"""

import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_train import evaluate_model
from src.training.warmstart_generator import (
    WarmStartConfig,
    generate_warmstart_data,
    save_warmstart_data,
    load_warmstart_data,
    pretrain_on_warmstart,
)


ARCHITECTURES = {
    "resnet": {
        "architecture": "resnet",
        "num_blocks": 6,
        "num_filters": 64,
        "d_model": 64,
        "n_heads": 4,
    },
    "pin_transformer": {
        "architecture": "pin_transformer",
        "num_blocks": 4,
        "num_filters": 64,
        "d_model": 128,
        "n_heads": 4,
    },
    "gateau": {
        "architecture": "gateau",
        "num_blocks": 4,
        "num_filters": 64,
        "d_model": 128,
        "n_heads": 4,
    },
}


def run_comparison(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Architecture Comparison (EXP-D5)")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # --- Step 1: Generate or load data ---
    data_path = args.data_path
    if data_path and os.path.exists(data_path):
        print(f"\n[1/3] Loading existing data from {data_path}")
        data = load_warmstart_data(data_path)
    else:
        data_path = os.path.join(output_dir, "warmstart_data.npz")
        if os.path.exists(data_path) and not args.regenerate:
            print(f"\n[1/3] Loading existing data from {data_path}")
            data = load_warmstart_data(data_path)
        else:
            print(f"\n[1/3] Generating {args.num_games} warm-start games...")
            config = WarmStartConfig(
                num_games=args.num_games,
                max_moves=args.max_moves,
                output_dir=output_dir,
            )
            data = generate_warmstart_data(config)
            save_warmstart_data(data, output_dir)

    print(f"  Data: {data['obs'].shape[0]} samples")

    # --- Step 2: Train each architecture ---
    results = {}
    architectures_to_run = args.architectures or list(ARCHITECTURES.keys())

    for arch_name in architectures_to_run:
        if arch_name not in ARCHITECTURES:
            print(f"  WARNING: Unknown architecture '{arch_name}', skipping")
            continue

        arch_config = ARCHITECTURES[arch_name]
        print(f"\n{'='*60}")
        print(f"[2/3] Training: {arch_name}")
        print(f"{'='*60}")

        net_config = NetworkConfig(
            num_blocks=arch_config["num_blocks"],
            num_filters=arch_config["num_filters"],
            architecture=arch_config["architecture"],
            d_model=arch_config["d_model"],
            n_heads=arch_config["n_heads"],
            use_auxiliary_head=args.use_auxiliary_head,
        )

        network = AlphaZeroNet(net_config, device=device)
        param_count = network.parameter_count()
        print(f"  Parameters: {param_count:,}")

        train_start = time.time()
        log = pretrain_on_warmstart(
            network,
            data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            early_stop_patience=args.patience,
        )
        train_time = time.time() - train_start

        # Save model
        arch_dir = os.path.join(output_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)
        checkpoint_path = os.path.join(arch_dir, "warmstart_model.pt")
        network.save_checkpoint(checkpoint_path, iteration=0, extra={"pretrain_log": log})
        print(f"  Saved to {checkpoint_path}")

        # --- Step 3: Evaluate ---
        print(f"\n  Evaluating {arch_name}...")
        eval_start = time.time()
        eval_results = evaluate_model(
            network,
            num_games=args.eval_games,
            max_steps=300,
            use_mcts=False,
        )
        eval_time = time.time() - eval_start

        vs_random = eval_results["vs_random"]
        vs_greedy = eval_results["vs_greedy"]
        vs_advanced = eval_results["vs_advanced"]

        print(f"  vs Random:   pins={vs_random['avg_pins_in_goal']:.1f}, "
              f"score={vs_random['avg_tournament_score']:.1f}, "
              f"wins={vs_random['agent_wins']}/{args.eval_games}")
        print(f"  vs Greedy:   pins={vs_greedy['avg_pins_in_goal']:.1f}, "
              f"score={vs_greedy['avg_tournament_score']:.1f}, "
              f"wins={vs_greedy['agent_wins']}/{args.eval_games}")
        print(f"  vs Advanced: pins={vs_advanced['avg_pins_in_goal']:.1f}, "
              f"score={vs_advanced['avg_tournament_score']:.1f}, "
              f"wins={vs_advanced['agent_wins']}/{args.eval_games}")

        # Measure inference speed
        speed_start = time.time()
        dummy_obs = np.random.randn(10, 17, 17).astype(np.float32)
        dummy_mask = np.ones(1210, dtype=bool)
        for _ in range(100):
            network.predict(dummy_obs, dummy_mask)
        inference_ms = (time.time() - speed_start) / 100 * 1000

        results[arch_name] = {
            "parameters": param_count,
            "train_time_s": train_time,
            "eval_time_s": eval_time,
            "inference_ms": inference_ms,
            "final_train_loss": log[-1]["train_loss"]["total_loss"] if log else None,
            "final_val_loss": log[-1]["val_loss"]["total_loss"] if log else None,
            "epochs_trained": len(log),
            "vs_random": vs_random,
            "vs_greedy": vs_greedy,
            "vs_advanced": vs_advanced,
        }

        print(f"  Train time: {train_time:.1f}s, Inference: {inference_ms:.1f}ms/move")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Architecture':<20} {'Params':>10} {'Train(s)':>10} {'Inf(ms)':>10} "
          f"{'R.pins':>8} {'G.pins':>8} {'A.pins':>8}")
    print("-" * 86)

    for arch_name, r in results.items():
        print(f"{arch_name:<20} {r['parameters']:>10,} {r['train_time_s']:>10.1f} "
              f"{r['inference_ms']:>10.1f} "
              f"{r['vs_random']['avg_pins_in_goal']:>8.1f} "
              f"{r['vs_greedy']['avg_pins_in_goal']:>8.1f} "
              f"{r['vs_advanced']['avg_pins_in_goal']:>8.1f}")

    # Save results
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="3-way architecture comparison")
    parser.add_argument("--num-games", type=int, default=2000,
                        help="Warm-start games to generate (shared across all architectures)")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to existing .npz data (skip generation)")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--architectures", nargs="+", default=None,
                        choices=list(ARCHITECTURES.keys()),
                        help="Which architectures to train (default: all 3)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--use-auxiliary-head", action="store_true")
    parser.add_argument("--output", type=str, default="experiments/exp_d5_arch_compare")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
