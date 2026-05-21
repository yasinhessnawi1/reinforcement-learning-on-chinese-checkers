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


def cmd_finetune(args):
    """Fine-tune an existing checkpoint on a .npz dataset (e.g. endgame data)."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    net_config = NetworkConfig(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.n_heads,
        use_auxiliary_head=args.use_auxiliary_head,
    )
    network = AlphaZeroNet(net_config, device=device)
    network.load_checkpoint(args.resume)
    print(f"Loaded checkpoint: {args.resume}")
    print(f"Network: {network.parameter_count():,} parameters")

    data = load_warmstart_data(args.data)
    n = data["obs"].shape[0]
    print(f"Loaded {n} samples from {args.data}")

    # Optionally mix in original warmstart data to prevent forgetting
    if args.mix_warmstart:
        ws_data = load_warmstart_data(args.mix_warmstart)
        mix_n = min(ws_data["obs"].shape[0], n // 2)  # at most 50% original
        idx = np.random.choice(ws_data["obs"].shape[0], mix_n, replace=False)
        for key in data:
            data[key] = np.concatenate([data[key], ws_data[key][idx]], axis=0)
        print(f"Mixed in {mix_n} samples from {args.mix_warmstart} "
              f"(total: {data['obs'].shape[0]})")

    log = pretrain_on_warmstart(
        network,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stop_patience=args.patience,
    )

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "finetuned_model.pt")
    network.save_checkpoint(out_path, iteration=0, extra={"finetune_log": log})
    print(f"\nSaved fine-tuned model to {out_path}")

    print("\nEvaluating fine-tuned model...")
    results = evaluate_model(network, num_games=10, max_steps=300, use_mcts=False)
    n_games = 10
    print(f"  vs Random:   pins={results['vs_random']['avg_pins_in_goal']:.1f}, "
          f"wins={results['vs_random']['agent_wins']}/{n_games}")
    print(f"  vs Greedy:   pins={results['vs_greedy']['avg_pins_in_goal']:.1f}, "
          f"wins={results['vs_greedy']['agent_wins']}/{n_games}")
    print(f"  vs Advanced: pins={results['vs_advanced']['avg_pins_in_goal']:.1f}, "
          f"wins={results['vs_advanced']['agent_wins']}/{n_games}")

    with open(os.path.join(args.output, "finetune_log.json"), "w") as f:
        json.dump({"finetune_log": log, "eval_results": results}, f, indent=2, default=str)


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
        use_batched_mcts=args.batched_mcts_selfplay,
        mcts_batch_size=args.mcts_batch_size,
        value_target_lambda=args.value_target_lambda,
        entropy_routing=args.entropy_routing,
        entropy_low=args.entropy_low,
        entropy_high=args.entropy_high,
        deep_sims_multiplier=args.deep_sims_multiplier,
    )

    # Curriculum config
    curriculum_mix = {
        "greedy": args.curriculum_greedy,
        "advanced": args.curriculum_advanced,
        "self_play": args.curriculum_selfplay,
    }

    train_config = TrainingConfig(
        network=net_config,
        self_play=sp_config,
        games_per_iteration=args.games_per_iter,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs_per_iter,
        replay_buffer_size=args.buffer_size,
        lr=args.lr,
        lr_decay=args.lr_decay,
        value_loss_weight=args.value_loss_weight,
        num_workers=args.num_workers,
        num_iterations=args.iterations,
        eval_games=args.eval_games,
        eval_interval=args.eval_interval,
        eval_sims=args.eval_sims,
        win_threshold=args.win_threshold,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        use_per=args.per,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        use_curriculum=args.curriculum,
        curriculum_mix=curriculum_mix,
    )

    print("AlphaZero Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Architecture: ResNet {args.num_blocks}x{args.num_filters}")
    print(f"  MCTS: {args.mcts} with {args.sims} simulations"
          f"{' (batched, batch_size=' + str(args.mcts_batch_size) + ')' if args.batched_mcts_selfplay else ''}")
    print(f"  Games/iteration: {args.games_per_iter}")
    workers_str = f"{args.num_workers}" if args.num_workers > 0 else "auto"
    print(f"  Parallel workers: {workers_str}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Eval: every {args.eval_interval} iters, {args.eval_games} games, {args.eval_sims} sims")
    if args.curriculum:
        print(f"  Curriculum: greedy={args.curriculum_greedy:.0%}, "
              f"advanced={args.curriculum_advanced:.0%}, "
              f"self_play={args.curriculum_selfplay:.0%}")
    print(f"  Value target: {args.value_target_lambda:.0%} game outcome + "
          f"{1-args.value_target_lambda:.0%} MCTS root value")
    if args.entropy_routing:
        print(f"  Search MoE: entropy<{args.entropy_low}=raw, "
              f">{args.entropy_high}={args.sims*args.deep_sims_multiplier}sims, "
              f"else={args.sims}sims")
    if args.warmstart_data:
        print(f"  Reservoir: {args.warmstart_data} (20% of batches)")
    if args.endgame_data:
        print(f"  Endgame pool: {args.endgame_data} (10% of batches)")
    print(f"  Resume from: {args.resume or 'scratch'}")

    train_alphazero(
        train_config,
        resume_from=args.resume,
        warmstart_data_path=args.warmstart_data,
        endgame_data_path=args.endgame_data,
        use_true_self_play=not args.legacy_self_play,
    )


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

        # Tournament budget check: use advanced win move count if available
        # (that's the realistic tournament game length), otherwise fall back to avg.
        adv = results.get('vs_advanced', {})
        if adv.get('agent_wins', 0) > 0 and 'avg_steps_win' in adv:
            tournament_moves = adv['avg_steps_win']
            budget_basis = f"advanced win avg={tournament_moves:.0f} moves"
        else:
            tournament_moves = real_avg_moves
            budget_basis = f"all-game avg={tournament_moves:.0f} moves"

        full_game_ms = per_move_ms * tournament_moves
        budget_status = "OK" if full_game_ms <= 60_000 else "OVER"
        print(
            f"  MCTS: {args.sims} sims/move @ ~{sims_per_sec:.0f} sims/sec "
            f"(~{sims_in_10s:.0f} sims in 10s turn budget)"
        )
        print(
            f"  Tournament budget: {full_game_ms/1000:.1f}s/game "
            f"(basis: {budget_basis}, budget=60s) [{budget_status}]"
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


def cmd_scm(args):
    """Train and evaluate Search-Conditioned Modulation (SCM)."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")
    print(f"SCM Pipeline — phase: {args.phase}")

    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, "scm_logs")

    # Load frozen policy network
    net_config = NetworkConfig(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.n_heads,
        use_auxiliary_head=args.use_auxiliary_head,
    )
    network = AlphaZeroNet(net_config, device=device)
    network.load_checkpoint(args.checkpoint)
    print(f"  Loaded policy: {network.parameter_count():,} params")
    print(f"  Checkpoint: {args.checkpoint}")

    from src.network.search_conditioned_modulator import SCMConfig, SearchConditionedModulator
    from src.training.scm_trainer import SCMTrainConfig, SCMTrainer
    from src.training.scm_self_play import (
        collect_scm_training_data,
        evaluate_with_scm,
        load_traj_chunks,
    )
    from src.training.alphazero_self_play import SelfPlayConfig as SPConfig

    # Build opponent for data collection / eval
    from src.agents.greedy_agent import greedy_policy
    opponent_policy = greedy_policy

    sp_config = SPConfig(
        num_simulations=args.sims,
        use_heuristic_value=args.heuristic_value,
        max_moves=args.max_moves,
    )

    # Create SCM
    scm_config = SCMConfig(
        hidden_dim=args.scm_hidden,
        max_shift=args.max_shift,
        identity_reg_weight=args.identity_reg,
    )
    scm = SearchConditionedModulator(scm_config)
    print(f"  SCM GRU: {scm.param_count():,} params (hidden={args.scm_hidden})")

    scm_ckpt_path = os.path.join(args.output, "scm_model.pt")

    # Load existing SCM if provided
    if args.scm_checkpoint:
        trainer = SCMTrainer(scm, device=device)
        trainer.load_checkpoint(args.scm_checkpoint)
        print(f"  Resumed SCM from: {args.scm_checkpoint}")

    # ---- Phase 1: Collect ----
    traj_dir = os.path.join(args.output, "scm_trajectories")

    if args.phase in ("collect", "all"):
        num_w = getattr(args, "num_workers", 0)
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting SCM training data ({args.collect_games} games, workers={num_w or 'auto'})")
        print(f"{'='*60}")

        trajectories = collect_scm_training_data(
            network=network,
            num_games=args.collect_games,
            config=sp_config,
            opponent_policy=opponent_policy,
            save_dir=traj_dir,
            num_workers=num_w,
            chunk_size=50,
        )

        total_steps = sum(len(t.steps) for t in trajectories)
        print(f"  Collected {len(trajectories)} trajectories ({total_steps} steps)")
        print(f"  Chunks saved to {traj_dir}/")

    # ---- Phase 2: Train ----
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Training SCM GRU ({args.train_epochs} epochs)")
        print(f"{'='*60}")

        if not os.path.isdir(traj_dir):
            print(f"  ERROR: No trajectory data at {traj_dir}/. Run 'collect' phase first.")
            return

        trajectories = load_traj_chunks(traj_dir)
        print(f"  Loaded {len(trajectories)} trajectories from chunks")

        train_config = SCMTrainConfig(
            lr=args.scm_lr,
            identity_reg_weight=args.identity_reg,
            num_epochs=args.train_epochs,
            log_interval=5,
        )
        trainer = SCMTrainer(scm, config=train_config, device=device)

        losses = trainer.train_on_trajectories(trajectories, verbose=True)
        print(f"\n  Final losses: {losses}")

        trainer.save_checkpoint(scm_ckpt_path)
        print(f"  Saved SCM checkpoint to {scm_ckpt_path}")

    # ---- Phase 3: Evaluate ----
    if args.phase in ("eval", "all"):
        # Parse blend alphas
        blend_alphas = [float(x) for x in args.blend_alphas.split(",")]

        print(f"\n{'='*60}")
        print(f"Phase 3: Evaluating SCM ({args.eval_games} games per condition)")
        print(f"  Blend levels: {blend_alphas}")
        print(f"{'='*60}")

        if os.path.exists(scm_ckpt_path) and not args.scm_checkpoint:
            trainer = SCMTrainer(scm, device=device)
            trainer.load_checkpoint(scm_ckpt_path)
            print(f"  Loaded SCM from {scm_ckpt_path}")

        scm.to(torch.device(device))

        results = evaluate_with_scm(
            network=network,
            scm=scm,
            num_games=args.eval_games,
            config=sp_config,
            opponent_policy=opponent_policy,
            log_dir=log_dir,
            blend_alphas=blend_alphas,
        )

        # Save results
        results_path = os.path.join(args.output, "scm_eval_results.json")
        serializable = {}
        for key, val in results.items():
            if key == "conditions":
                serializable["conditions"] = {}
                for cond_name, cond_data in val.items():
                    serializable["conditions"][cond_name] = {
                        k: v for k, v in cond_data.items()
                        if k != "pins_list"
                    }
                    serializable["conditions"][cond_name]["pins_list"] = cond_data.get("pins_list", [])
            else:
                serializable[key] = val
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Results saved to {results_path}")

    print(f"\n{'='*60}")
    print("SCM pipeline complete!")
    print(f"{'='*60}")


def cmd_scm_multi(args):
    """Multiplayer SCM: collect, train, evaluate with 2-6 players."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")
    print(f"Multiplayer SCM Pipeline — phase: {args.phase}")

    os.makedirs(args.output, exist_ok=True)
    log_dir = os.path.join(args.output, "scm_logs")

    # Load frozen policy network
    net_config = NetworkConfig(
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        architecture=args.architecture,
        d_model=args.d_model,
        n_heads=args.n_heads,
        use_auxiliary_head=args.use_auxiliary_head,
    )
    network = AlphaZeroNet(net_config, device=device)
    network.load_checkpoint(args.checkpoint)
    print(f"  Loaded policy: {network.parameter_count():,} params")
    print(f"  Checkpoint: {args.checkpoint}")

    from src.network.search_conditioned_modulator import SCMConfig, SearchConditionedModulator
    from src.training.scm_trainer import SCMTrainConfig, SCMTrainer
    from src.training.scm_self_play import (
        collect_scm_training_data_multi,
        evaluate_with_scm_multi,
        load_traj_chunks,
    )
    from src.training.alphazero_self_play import SelfPlayConfig as SPConfig

    n_choices = tuple(int(x) for x in args.n_choices.split(","))
    n_weights = tuple(int(x) for x in args.n_weights.split(","))

    sp_config = SPConfig(
        num_simulations=args.sims,
        use_heuristic_value=args.heuristic_value,
        max_moves=args.max_moves,
    )

    # Create SCM
    scm_config = SCMConfig(
        hidden_dim=args.scm_hidden,
        max_shift=args.max_shift,
        identity_reg_weight=args.identity_reg,
    )
    scm = SearchConditionedModulator(scm_config)
    print(f"  SCM GRU: {scm.param_count():,} params (hidden={args.scm_hidden})")
    print(f"  Player counts: {n_choices} (weights: {n_weights})")

    scm_ckpt_path = os.path.join(args.output, "scm_model.pt")

    if args.scm_checkpoint:
        trainer = SCMTrainer(scm, device=device)
        trainer.load_checkpoint(args.scm_checkpoint)
        print(f"  Resumed SCM from: {args.scm_checkpoint}")

    # ---- Phase 1: Collect ----
    traj_dir = os.path.join(args.output, "scm_trajectories")

    if args.phase in ("collect", "all"):
        num_w = getattr(args, "num_workers", 0)
        print(f"\n{'='*60}")
        print(f"Phase 1: Collecting multiplayer SCM data ({args.collect_games} games)")
        print(f"  Player counts: {n_choices}, weights: {n_weights}")
        print(f"  Workers: {num_w or 'auto'}")
        print(f"{'='*60}")

        trajectories = collect_scm_training_data_multi(
            network=network,
            num_games=args.collect_games,
            config=sp_config,
            n_choices=n_choices,
            n_weights=n_weights,
            save_dir=traj_dir,
            num_workers=num_w,
            chunk_size=50,
        )

        total_steps = sum(len(t.steps) for t in trajectories)
        print(f"  Collected {len(trajectories)} trajectories ({total_steps} steps)")
        print(f"  Chunks saved to {traj_dir}/")

    # ---- Phase 2: Train ----
    if args.phase in ("train", "all"):
        print(f"\n{'='*60}")
        print(f"Phase 2: Training SCM GRU ({args.train_epochs} epochs)")
        print(f"{'='*60}")

        if not os.path.isdir(traj_dir):
            print(f"  ERROR: No trajectory data at {traj_dir}/. Run 'collect' phase first.")
            return

        trajectories = load_traj_chunks(traj_dir)
        print(f"  Loaded {len(trajectories)} trajectories from chunks")

        train_config = SCMTrainConfig(
            lr=args.scm_lr,
            identity_reg_weight=args.identity_reg,
            num_epochs=args.train_epochs,
            log_interval=5,
        )
        trainer = SCMTrainer(scm, config=train_config, device=device)

        losses = trainer.train_on_trajectories(trajectories, verbose=True)
        print(f"\n  Final losses: {losses}")

        trainer.save_checkpoint(scm_ckpt_path)
        print(f"  Saved SCM checkpoint to {scm_ckpt_path}")

    # ---- Phase 3: Evaluate ----
    if args.phase in ("eval", "all"):
        blend_alphas = [float(x) for x in args.blend_alphas.split(",")]

        print(f"\n{'='*60}")
        print(f"Phase 3: Evaluating SCM ({args.eval_games} games, {args.eval_players} players)")
        print(f"  Blend levels: {blend_alphas}")
        print(f"{'='*60}")

        if os.path.exists(scm_ckpt_path) and not args.scm_checkpoint:
            trainer = SCMTrainer(scm, device=device)
            trainer.load_checkpoint(scm_ckpt_path)
            print(f"  Loaded SCM from {scm_ckpt_path}")

        scm.to(torch.device(device))

        results = evaluate_with_scm_multi(
            network=network,
            scm=scm,
            num_games=args.eval_games,
            config=sp_config,
            n_players=args.eval_players,
            blend_alphas=blend_alphas,
            log_dir=log_dir,
        )

        # Save results
        results_path = os.path.join(args.output, "scm_multi_eval_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {results_path}")

    print(f"\n{'='*60}")
    print("Multiplayer SCM pipeline complete!")
    print(f"{'='*60}")


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

    # --- finetune ---
    ft = subparsers.add_parser("finetune", help="Fine-tune a checkpoint on a .npz dataset")
    ft.add_argument("--resume", type=str, required=True, help="Checkpoint to fine-tune from")
    ft.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    ft.add_argument("--output", type=str, default="experiments/exp_finetune")
    ft.add_argument("--epochs", type=int, default=30)
    ft.add_argument("--batch-size", type=int, default=256)
    ft.add_argument("--lr", type=float, default=1e-4, help="Lower LR for fine-tuning (default 1e-4)")
    ft.add_argument("--patience", type=int, default=5)
    ft.add_argument("--mix-warmstart", type=str, default=None,
                    help="Optional original warmstart .npz to mix in (prevents forgetting)")
    _add_arch_args(ft)
    ft.add_argument("--cpu", action="store_true")

    # --- train ---
    tr = subparsers.add_parser("train", help="Run AlphaZero training loop")
    tr.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    tr.add_argument("--mcts", type=str, default="puct", choices=["puct", "gumbel", "batched"])
    tr.add_argument("--sims", type=int, default=200, help="MCTS simulations per move")
    tr.add_argument("--mcts-batch-size", type=int, default=16,
                    help="Leaf batch size for batched MCTS (default 16, increase for big GPUs)")
    tr.add_argument("--c-puct", type=float, default=1.5)
    tr.add_argument("--dirichlet-alpha", type=float, default=0.3)
    tr.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    tr.add_argument("--temp-moves", type=int, default=15)
    tr.add_argument("--max-moves", type=int, default=100)
    tr.add_argument("--heuristic-value", action="store_true")
    tr.add_argument("--batched-mcts-selfplay", action="store_true", default=True,
                    help="Use batched MCTS for self-play (default: ON, 4-8x faster on GPU)")
    tr.add_argument("--no-batched-mcts", dest="batched_mcts_selfplay", action="store_false",
                    help="Disable batched MCTS, use single-inference MCTS")
    tr.add_argument("--value-target-lambda", type=float, default=0.8,
                    help="Blend: lambda*game_outcome + (1-lambda)*mcts_value (default 0.8)")
    tr.add_argument("--entropy-routing", action="store_true",
                    help="Search MoE: route MCTS depth by policy entropy")
    tr.add_argument("--entropy-low", type=float, default=0.5,
                    help="Below this entropy: skip MCTS (default 0.5)")
    tr.add_argument("--entropy-high", type=float, default=2.0,
                    help="Above this entropy: deep search 3x sims (default 2.0)")
    tr.add_argument("--deep-sims-multiplier", type=int, default=3,
                    help="Sim multiplier for high-entropy positions (default 3)")
    tr.add_argument("--no-augment", action="store_true")
    tr.add_argument("--games-per-iter", type=int, default=100)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--epochs-per-iter", type=int, default=8)
    tr.add_argument("--buffer-size", type=int, default=50000)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lr-decay", type=float, default=0.995)
    tr.add_argument("--value-loss-weight", type=float, default=0.25,
                    help="Scale value loss (default 0.25, lower = focus gradient on policy)")
    tr.add_argument("--iterations", type=int, default=30)
    tr.add_argument("--num-workers", type=int, default=0,
                    help="Parallel game workers (0=auto, 1=serial). Uses ProcessPoolExecutor.")
    tr.add_argument("--eval-games", type=int, default=20)
    tr.add_argument("--eval-interval", type=int, default=3,
                    help="Run eval every N iterations (default 3)")
    tr.add_argument("--eval-sims", type=int, default=100,
                    help="MCTS sims for eval games (default 100, must be reliable)")
    tr.add_argument("--win-threshold", type=float, default=0.55)
    tr.add_argument("--checkpoint-dir", type=str, default="experiments/exp_d1_alphazero")
    tr.add_argument("--warmstart-data", type=str, default=None,
                    help="Path to warm-start .npz for replay buffer reservoir (20%% of batches)")
    tr.add_argument("--endgame-data", type=str, default=None,
                    help="Path to endgame .npz data for dedicated pool (10%% of batches)")
    tr.add_argument("--legacy-self-play", action="store_true",
                    help="Use legacy single-agent self-play (agent vs random) instead of true 2-player")
    tr.add_argument("--per", action="store_true",
                    help="Enable Prioritized Experience Replay")
    tr.add_argument("--per-alpha", type=float, default=0.6,
                    help="PER priority exponent (default 0.6)")
    tr.add_argument("--per-beta-start", type=float, default=0.4,
                    help="PER importance-sampling start beta (default 0.4)")
    tr.add_argument("--curriculum", action="store_true", default=True,
                    help="Use opponent curriculum (default: ON)")
    tr.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                    help="Disable curriculum, use pure self-play")
    tr.add_argument("--curriculum-greedy", type=float, default=0.50,
                    help="Fraction of curriculum games vs greedy (default 0.50)")
    tr.add_argument("--curriculum-advanced", type=float, default=0.50,
                    help="Fraction of curriculum games vs advanced (default 0.50)")
    tr.add_argument("--curriculum-selfplay", type=float, default=0.0,
                    help="Fraction of curriculum games as self-play (default 0.0)")
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

    # --- scm (Search-Conditioned Modulation) ---
    scm = subparsers.add_parser("scm", help="Train and evaluate SCM (Search-Conditioned Modulation)")
    scm.add_argument("--checkpoint", type=str, required=True,
                     help="Frozen policy checkpoint to modulate")
    scm.add_argument("--phase", type=str, default="all",
                     choices=["collect", "train", "eval", "all"],
                     help="SCM pipeline phase: collect data, train GRU, evaluate, or all")
    scm.add_argument("--collect-games", type=int, default=200,
                     help="Number of games for SCM data collection (default 200)")
    scm.add_argument("--train-epochs", type=int, default=20,
                     help="SCM training epochs (default 20)")
    scm.add_argument("--eval-games", type=int, default=50,
                     help="Games per condition for evaluation (default 50)")
    scm.add_argument("--sims", type=int, default=200,
                     help="MCTS simulations per move (default 200)")
    scm.add_argument("--scm-hidden", type=int, default=128,
                     help="GRU hidden dimension (default 128)")
    scm.add_argument("--scm-lr", type=float, default=1e-3,
                     help="SCM learning rate (default 1e-3)")
    scm.add_argument("--identity-reg", type=float, default=0.01,
                     help="Identity regularization weight (default 0.01)")
    scm.add_argument("--max-shift", type=float, default=2.0,
                     help="Max shift magnitude (default 2.0)")
    scm.add_argument("--blend-alphas", type=str, default="0.3,0.5,0.7,1.0",
                     help="Comma-separated blend levels to test (default: 0.3,0.5,0.7,1.0)")
    scm.add_argument("--output", type=str, default="experiments/scm",
                     help="Output directory")
    scm.add_argument("--scm-checkpoint", type=str, default=None,
                     help="Resume SCM from existing checkpoint")
    scm.add_argument("--heuristic-value", action="store_true")
    scm.add_argument("--max-moves", type=int, default=100)
    scm.add_argument("--num-workers", type=int, default=0,
                     help="Parallel workers for collection (0=auto)")
    _add_arch_args(scm)
    scm.add_argument("--cpu", action="store_true")

    # --- scm-multi (multiplayer SCM) ---
    scm_m = subparsers.add_parser("scm-multi",
                                  help="Multiplayer SCM: collect, train, eval with 2-6 players")
    scm_m.add_argument("--checkpoint", type=str, required=True,
                       help="Frozen policy checkpoint to modulate")
    scm_m.add_argument("--phase", type=str, default="all",
                       choices=["collect", "train", "eval", "all"],
                       help="Pipeline phase")
    scm_m.add_argument("--collect-games", type=int, default=200,
                       help="Number of multiplayer games for data collection")
    scm_m.add_argument("--train-epochs", type=int, default=50,
                       help="SCM training epochs")
    scm_m.add_argument("--eval-games", type=int, default=30,
                       help="Games per condition for evaluation")
    scm_m.add_argument("--eval-players", type=int, default=4,
                       help="Number of players for evaluation games (default 4)")
    scm_m.add_argument("--sims", type=int, default=200,
                       help="MCTS simulations per move")
    scm_m.add_argument("--scm-hidden", type=int, default=128,
                       help="GRU hidden dimension")
    scm_m.add_argument("--scm-lr", type=float, default=1e-3,
                       help="SCM learning rate")
    scm_m.add_argument("--identity-reg", type=float, default=0.01,
                       help="Identity regularization weight")
    scm_m.add_argument("--max-shift", type=float, default=2.0,
                       help="Max shift magnitude")
    scm_m.add_argument("--blend-alphas", type=str, default="0.1,0.2,0.3",
                       help="Comma-separated blend levels to test")
    scm_m.add_argument("--n-choices", type=str, default="2,3,4,5,6",
                       help="Player counts to sample from (comma-separated)")
    scm_m.add_argument("--n-weights", type=str, default="1,2,2,2,3",
                       help="Relative weights for each player count")
    scm_m.add_argument("--output", type=str, default="experiments/scm_multi",
                       help="Output directory")
    scm_m.add_argument("--scm-checkpoint", type=str, default=None,
                       help="Resume SCM from existing checkpoint")
    scm_m.add_argument("--heuristic-value", action="store_true")
    scm_m.add_argument("--max-moves", type=int, default=100)
    scm_m.add_argument("--num-workers", type=int, default=0,
                       help="Parallel workers for collection (0=auto)")
    _add_arch_args(scm_m)
    scm_m.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    if args.command == "warmstart":
        cmd_warmstart(args)
    elif args.command == "enhanced-warmstart":
        cmd_enhanced_warmstart(args)
    elif args.command == "finetune":
        cmd_finetune(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "scm":
        cmd_scm(args)
    elif args.command == "scm-multi":
        cmd_scm_multi(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
