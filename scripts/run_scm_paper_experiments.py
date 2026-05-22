#!/usr/bin/env python3
"""
run_scm_paper_experiments.py — Run all experiments for the SCM paper revision.

Covers:
  Part 1A: Core results (strong model vs greedy + heuristic, >=200 games)
  Part 1C: Ablations (identity SCM, untrained SCM, temperature, Dirichlet)
  Part 1D: Interpretability rerun on strong model
  Part 1B: Statistical tests (Fisher, Welch t, bootstrap CI, Wilson CI)

Usage:
  # Full pipeline: collect data, train SCM on strong model, then run all evals
  python scripts/run_scm_paper_experiments.py --phase all

  # Just evaluation (if SCM already trained)
  python scripts/run_scm_paper_experiments.py --phase eval

  # Just ablations
  python scripts/run_scm_paper_experiments.py --phase ablations

  # Just stats (from saved results)
  python scripts/run_scm_paper_experiments.py --phase stats
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.network.search_conditioned_modulator import (
    SCMConfig, SearchConditionedModulator, extract_search_features,
)
from src.training.scm_trainer import SCMTrainConfig, SCMTrainer
from src.training.scm_self_play import (
    collect_scm_training_data, evaluate_with_scm, load_traj_chunks,
    play_game_with_scm,
)
from src.training.alphazero_self_play import SelfPlayConfig
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import AlphaZeroMCTS
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "experiments/scm_v2_paper"
STRONG_CKPT = "experiments/d37/best_so_far.pt"
WEAK_CKPT = "experiments/exp_d10_local_continued/best_model.pt"
EXISTING_SCM = "experiments/scm_v2/scm_model.pt"

NET_CONFIG = NetworkConfig(num_blocks=9, num_filters=96)
SCM_CFG = SCMConfig(hidden_dim=128, max_shift=2.0, identity_reg_weight=0.001)
SP_CONFIG = SelfPlayConfig(
    num_simulations=200,
    use_heuristic_value=True,
    max_moves=100,
)

N_CORE = 200       # games per core condition
N_ABLATION = 100   # games per ablation condition
N_INTERP = 200     # games for interpretability
BEST_ALPHA = 0.25  # best blend from prior sweep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_network(ckpt_path: str, device: str = "cpu") -> AlphaZeroNet:
    net = AlphaZeroNet(NET_CONFIG, device=device)
    net.load_checkpoint(ckpt_path)
    return net


def load_scm(ckpt_path: str, device: str = "cpu") -> SearchConditionedModulator:
    scm = SearchConditionedModulator(SCM_CFG)
    trainer = SCMTrainer(scm, device=device)
    trainer.load_checkpoint(ckpt_path)
    scm.to(torch.device(device))
    return scm


def make_fresh_scm() -> SearchConditionedModulator:
    """Create a new untrained SCM with random weights."""
    return SearchConditionedModulator(SCM_CFG)


def make_identity_scm() -> SearchConditionedModulator:
    """Create SCM that acts as identity (gate=1, shift=0)."""
    scm = SearchConditionedModulator(SCM_CFG)
    # Override gate bias to very large value so sigmoid -> 1.0
    with torch.no_grad():
        scm.gate_head.bias.fill_(20.0)
        scm.gate_head.weight.fill_(0.0)
        scm.shift_head.bias.fill_(0.0)
        scm.shift_head.weight.fill_(0.0)
    return scm


def _save_progress(progress_path: str | None, pins_list: list) -> None:
    """Write current pins_list to disk for resume support."""
    if not progress_path:
        return
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump({"pins": pins_list, "n": len(pins_list)}, f)


def _load_progress(progress_path: str | None) -> list:
    """Load partial pins_list if it exists."""
    if not progress_path or not os.path.exists(progress_path):
        return []
    with open(progress_path) as f:
        data = json.load(f)
    return data.get("pins", [])


def run_baseline_games(
    network: AlphaZeroNet,
    opponent_policy,
    num_games: int,
    config: SelfPlayConfig,
    label: str = "",
    verbose: bool = True,
    progress_path: str | None = None,
) -> list[int]:
    """Run baseline (no SCM) games, return list of agent pins."""
    pins_list = _load_progress(progress_path)
    if pins_list:
        print(f"    [resume] {label} baseline: {len(pins_list)} games loaded from disk")
    for i in range(len(pins_list), num_games):
        env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
        obs, info = env.reset()
        mcts = AlphaZeroMCTS(
            network=network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=0.0,
            use_heuristic_value=config.use_heuristic_value,
            opponent_policy=opponent_policy,
        )
        done = False
        while not done:
            probs = mcts.get_action_probs(env, temperature=0.1)
            action = int(np.argmax(probs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        pins_list.append(agent_pins)
        _save_progress(progress_path, pins_list)

        if verbose and (i + 1) % 5 == 0:
            print(f"    {label} baseline: {i+1}/{num_games}  avg={np.mean(pins_list):.2f}", flush=True)
    return pins_list


def run_scm_games(
    network: AlphaZeroNet,
    scm: SearchConditionedModulator,
    opponent_policy,
    num_games: int,
    config: SelfPlayConfig,
    blend_alpha: float,
    label: str = "",
    log_dir: str | None = None,
    verbose: bool = True,
    progress_path: str | None = None,
) -> list[int]:
    """Run SCM-modulated games, return list of agent pins."""
    from src.network.search_conditioned_modulator import SCMLogger

    logger = SCMLogger(log_dir) if log_dir else None
    pins_list = _load_progress(progress_path)
    if pins_list:
        print(f"    [resume] {label} SCM: {len(pins_list)} games loaded from disk")

    for i in range(len(pins_list), num_games):
        gid = logger.new_game() if logger else None
        _, _, game_info = play_game_with_scm(
            network=network,
            scm=scm,
            config=SelfPlayConfig(
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=0.0,
                temperature_moves=0,
                temperature_low=0.1,
                max_moves=config.max_moves,
                use_heuristic_value=config.use_heuristic_value,
            ),
            opponent_policy=opponent_policy,
            logger=logger,
            game_id=gid,
            blend_alpha=blend_alpha,
        )
        pins_list.append(game_info["agent_pins_in_goal"])
        _save_progress(progress_path, pins_list)

        if verbose and (i + 1) % 5 == 0:
            print(f"    {label} SCM {blend_alpha:.0%}: {i+1}/{num_games}  avg={np.mean(pins_list):.2f}", flush=True)

    if logger and logger.num_buffered > 0:
        logger.flush()

    return pins_list


def run_temperature_games(
    network: AlphaZeroNet,
    opponent_policy,
    num_games: int,
    config: SelfPlayConfig,
    temperature: float,
    label: str = "",
    verbose: bool = True,
    progress_path: str | None = None,
) -> list[int]:
    """Run games with temperature-scaled logits (no SCM)."""
    pins_list = _load_progress(progress_path)
    if pins_list:
        print(f"    [resume] {label} temp={temperature}: {len(pins_list)} games loaded from disk")
    for i in range(len(pins_list), num_games):
        env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
        obs, info = env.reset()
        mcts = AlphaZeroMCTS(
            network=network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=0.0,
            use_heuristic_value=config.use_heuristic_value,
            opponent_policy=opponent_policy,
        )
        done = False
        while not done:
            probs = mcts.get_action_probs(env, temperature=temperature)
            action = int(np.argmax(probs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        pins_list.append(agent_pins)
        _save_progress(progress_path, pins_list)

        if verbose and (i + 1) % 5 == 0:
            print(f"    {label} temp={temperature}: {i+1}/{num_games}  avg={np.mean(pins_list):.2f}", flush=True)
    return pins_list


def run_dirichlet_games(
    network: AlphaZeroNet,
    opponent_policy,
    num_games: int,
    config: SelfPlayConfig,
    dir_alpha: float,
    dir_epsilon: float = 0.25,
    label: str = "",
    verbose: bool = True,
    progress_path: str | None = None,
) -> list[int]:
    """Run games with Dirichlet root noise in MCTS (no SCM)."""
    pins_list = _load_progress(progress_path)
    if pins_list:
        print(f"    [resume] {label} dirichlet={dir_alpha}: {len(pins_list)} games loaded from disk")
    for i in range(len(pins_list), num_games):
        env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
        obs, info = env.reset()
        mcts = AlphaZeroMCTS(
            network=network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=dir_alpha,
            dirichlet_epsilon=dir_epsilon,
            use_heuristic_value=config.use_heuristic_value,
            opponent_policy=opponent_policy,
        )
        done = False
        while not done:
            probs = mcts.get_action_probs(env, temperature=0.1)
            action = int(np.argmax(probs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        pins_list.append(agent_pins)
        _save_progress(progress_path, pins_list)

        if verbose and (i + 1) % 5 == 0:
            print(f"    {label} dirichlet={dir_alpha}: {i+1}/{num_games}  avg={np.mean(pins_list):.2f}", flush=True)
    return pins_list


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = wins / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def fisher_exact(wins_a: int, n_a: int, wins_b: int, n_b: int) -> float:
    """Two-sided Fisher's exact test on win counts."""
    from scipy.stats import fisher_exact as _fisher
    table = [[wins_a, n_a - wins_a], [wins_b, n_b - wins_b]]
    _, p = _fisher(table, alternative='two-sided')
    return p


def welch_t_test(a: list, b: list) -> tuple[float, float]:
    """Welch's t-test. Returns (p_value, cohens_d)."""
    from scipy.stats import ttest_ind
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    stat, p = ttest_ind(a_arr, b_arr, equal_var=False)
    pooled_std = np.sqrt((a_arr.std()**2 + b_arr.std()**2) / 2)
    d = (a_arr.mean() - b_arr.mean()) / pooled_std if pooled_std > 0 else 0.0
    return (p, d)


def bootstrap_ci(a: list, b: list, n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap 95% CI on difference in means (a - b). Returns (mean_diff, lo, hi)."""
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sa = rng.choice(a_arr, size=len(a_arr), replace=True)
        sb = rng.choice(b_arr, size=len(b_arr), replace=True)
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(diffs, [alpha * 100, (1 - alpha) * 100])
    return (float(diffs.mean()), float(lo), float(hi))


def compute_condition_stats(pins: list) -> dict:
    """Compute summary stats for a condition."""
    arr = np.array(pins, dtype=float)
    wins = sum(1 for p in pins if p >= 10)
    n = len(pins)
    wlo, whi = wilson_ci(wins, n)
    return {
        "n": n,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "max": int(arr.max()),
        "wins": wins,
        "win_rate": wins / n if n > 0 else 0.0,
        "wilson_ci_lo": wlo,
        "wilson_ci_hi": whi,
        "pins_list": pins,
    }


def compute_comparison_stats(baseline_pins: list, scm_pins: list) -> dict:
    """Compute comparison statistics between baseline and SCM."""
    b_wins = sum(1 for p in baseline_pins if p >= 10)
    s_wins = sum(1 for p in scm_pins if p >= 10)
    n_b, n_s = len(baseline_pins), len(scm_pins)

    fisher_p = fisher_exact(s_wins, n_s, b_wins, n_b)
    welch_p, cohens_d = welch_t_test(scm_pins, baseline_pins)
    boot_mean, boot_lo, boot_hi = bootstrap_ci(scm_pins, baseline_pins)

    return {
        "fisher_p": fisher_p,
        "welch_p": welch_p,
        "cohens_d": cohens_d,
        "bootstrap_mean_diff": boot_mean,
        "bootstrap_ci_lo": boot_lo,
        "bootstrap_ci_hi": boot_hi,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase_collect_and_train(args):
    """Collect SCM training data on strong model and train."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    traj_dir = os.path.join(OUTPUT_DIR, "scm_trajectories")

    network = load_network(STRONG_CKPT, device)
    print(f"Loaded strong model: {STRONG_CKPT} ({network.parameter_count():,} params)")

    # Collect
    print(f"\n{'='*60}")
    print(f"Collecting {args.collect_games} games for SCM training on strong model")
    print(f"{'='*60}")

    trajectories = collect_scm_training_data(
        network=network,
        num_games=args.collect_games,
        config=SP_CONFIG,
        opponent_policy=greedy_policy,
        save_dir=traj_dir,
        num_workers=args.num_workers,
        chunk_size=50,
    )
    total_steps = sum(len(t.steps) for t in trajectories)
    print(f"  Collected {len(trajectories)} trajectories ({total_steps} steps)")

    # Train
    print(f"\n{'='*60}")
    print(f"Training SCM GRU ({args.train_epochs} epochs)")
    print(f"{'='*60}")

    scm = SearchConditionedModulator(SCM_CFG)
    train_config = SCMTrainConfig(
        lr=1e-3,
        identity_reg_weight=0.001,
        num_epochs=args.train_epochs,
        log_interval=5,
    )
    trainer = SCMTrainer(scm, config=train_config, device=device)
    losses = trainer.train_on_trajectories(trajectories, verbose=True)
    print(f"  Final losses: {losses}")

    ckpt_path = os.path.join(OUTPUT_DIR, "scm_strong_model.pt")
    trainer.save_checkpoint(ckpt_path)
    print(f"  Saved SCM to {ckpt_path}")


def _condition_path(name: str) -> str:
    """Path for per-condition checkpoint file."""
    return os.path.join(OUTPUT_DIR, "conditions", f"{name}.json")


def _save_condition(name: str, pins: list) -> None:
    """Save raw pins list for a single condition (resume support)."""
    os.makedirs(os.path.join(OUTPUT_DIR, "conditions"), exist_ok=True)
    with open(_condition_path(name), "w") as f:
        json.dump({"name": name, "pins": pins, "n": len(pins)}, f)


def _load_condition(name: str) -> list | None:
    """Load saved pins list if present, else None."""
    p = _condition_path(name)
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
        return data["pins"]
    return None


def _run_or_resume(name: str, n_target: int, runner) -> list:
    """Run a condition, or resume from saved if already done.

    `runner` is a callable taking `progress_path` and returning the pins list.
    Partial progress is stored at the condition-specific path so a crash
    mid-condition doesn't lose all games.
    """
    final = _load_condition(name)
    if final is not None and len(final) >= n_target:
        print(f"  [resume] {name}: loaded {len(final)} games from disk")
        return final
    progress_path = _condition_path(name)
    pins = runner(progress_path)
    _save_condition(name, pins)
    return pins


def phase_eval(args):
    """Run all core evaluations: strong model vs greedy and heuristic."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    network = load_network(STRONG_CKPT, device)
    print(f"Loaded strong model: {STRONG_CKPT}")

    scm_path = os.path.join(OUTPUT_DIR, "scm_strong_model.pt")
    if not os.path.exists(scm_path):
        print(f"  WARNING: {scm_path} not found, using existing SCM from scm_v2")
        scm_path = EXISTING_SCM
    scm = load_scm(scm_path, device)
    print(f"Loaded SCM from {scm_path}")

    alpha = args.best_alpha

    def heuristic_opponent(board, colour):
        return advanced_heuristic_policy(board, colour)

    # --- Condition 1: Strong vs Greedy, Baseline ---
    print(f"\n{'='*60}")
    print(f"Condition 1: Strong vs Greedy, Baseline ({N_CORE} games)")
    print(f"{'='*60}")
    pins_1 = _run_or_resume(
        "strong_greedy_baseline", N_CORE,
        lambda pp: run_baseline_games(network, greedy_policy, N_CORE, SP_CONFIG, "vs_greedy", progress_path=pp),
    )
    results["strong_greedy_baseline"] = compute_condition_stats(pins_1)

    # --- Condition 2: Strong vs Greedy, SCM ---
    print(f"\n{'='*60}")
    print(f"Condition 2: Strong vs Greedy, SCM alpha={alpha} ({N_CORE} games)")
    print(f"{'='*60}")
    pins_2 = _run_or_resume(
        "strong_greedy_scm", N_CORE,
        lambda pp: run_scm_games(network, scm, greedy_policy, N_CORE, SP_CONFIG, alpha, "vs_greedy", progress_path=pp),
    )
    results["strong_greedy_scm"] = compute_condition_stats(pins_2)
    results["strong_greedy_comparison"] = compute_comparison_stats(pins_1, pins_2)

    # --- Condition 3: Strong vs Heuristic, Baseline ---
    # NOTE: SCM vs heuristic skipped — baseline already wins ~100% (10/10 pins),
    # so SCM has no headroom to demonstrate improvement. The heuristic baseline
    # number still serves to characterize the strong model's behavior.
    N_HEURISTIC = 50  # reduced — baseline is deterministic at 10 pins
    print(f"\n{'='*60}")
    print(f"Condition 3: Strong vs Heuristic, Baseline ({N_HEURISTIC} games)")
    print(f"{'='*60}")
    pins_3 = _run_or_resume(
        "strong_heuristic_baseline", N_HEURISTIC,
        lambda pp: run_baseline_games(network, heuristic_opponent, N_HEURISTIC, SP_CONFIG, "vs_heuristic", progress_path=pp),
    )
    results["strong_heuristic_baseline"] = compute_condition_stats(pins_3)

    # --- Cross-model transfer: existing SCM (trained on d10) applied to d37 ---
    print(f"\n{'='*60}")
    print(f"Cross-model transfer: existing SCM (d10-trained) on d37 ({N_CORE} games)")
    print(f"{'='*60}")
    transfer_scm = load_scm(EXISTING_SCM, device)
    pins_transfer = _run_or_resume(
        "transfer_greedy_scm", N_CORE,
        lambda pp: run_scm_games(network, transfer_scm, greedy_policy, N_CORE, SP_CONFIG, 0.2, "transfer", progress_path=pp),
    )
    results["transfer_greedy_scm"] = compute_condition_stats(pins_transfer)
    results["transfer_greedy_comparison"] = compute_comparison_stats(pins_1, pins_transfer)

    # Save
    _save_results(results, "core_results.json")
    print(f"\n  Core results saved to {OUTPUT_DIR}/core_results.json")


def phase_ablations(args):
    """Run focused ablation set (4 variants, N=50 each, vs greedy).

    Rationale: vs greedy the baseline is locked at 7.0 and trained SCM reaches
    7.45 — that's the contrast we need ablations to characterize. vs heuristic
    the baseline already wins 100% so noise injection has no signal to detect.

    Reduced from spec's 8 variants (4 temps + 3 dirichlet + identity + untrained)
    to 4 variants (identity + untrained + temp=1.0 + dirichlet=0.3) for tractable
    local compute (~50 hours instead of ~250).
    """
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    network = load_network(STRONG_CKPT, device)
    alpha = args.best_alpha

    # Reuse condition-1 baseline if already done (saves 50 games)
    saved_baseline = _load_condition("strong_greedy_baseline")
    if saved_baseline is not None and len(saved_baseline) >= 50:
        print(f"  [reuse] Using existing strong_greedy_baseline ({len(saved_baseline)} games) as ablation baseline")
        pins_base = saved_baseline[:50] if len(saved_baseline) > 50 else saved_baseline
    else:
        print(f"\n{'='*60}")
        print(f"Ablation baseline: Strong vs Greedy (50 games)")
        print(f"{'='*60}")
        pins_base = _run_or_resume(
            "ablation_baseline", 50,
            lambda pp: run_baseline_games(network, greedy_policy, 50, SP_CONFIG, "abl_base", progress_path=pp),
        )
    results["baseline"] = compute_condition_stats(pins_base)

    # Reuse condition-2 trained SCM if already done
    saved_scm = _load_condition("strong_greedy_scm")
    if saved_scm is not None and len(saved_scm) >= 50:
        print(f"  [reuse] Using existing strong_greedy_scm ({len(saved_scm)} games) as trained_scm reference")
        pins_trained = saved_scm[:50] if len(saved_scm) > 50 else saved_scm
    else:
        scm_path = os.path.join(OUTPUT_DIR, "scm_strong_model.pt")
        if not os.path.exists(scm_path):
            scm_path = EXISTING_SCM
        trained_scm = load_scm(scm_path, device)
        print(f"\n{'='*60}")
        print(f"Ablation trained SCM: alpha={alpha} (50 games)")
        print(f"{'='*60}")
        pins_trained = _run_or_resume(
            "ablation_trained_scm", 50,
            lambda pp: run_scm_games(network, trained_scm, greedy_policy, 50, SP_CONFIG, alpha, "abl_trained", progress_path=pp),
        )
    results["trained_scm"] = compute_condition_stats(pins_trained)

    # Ablation 1: Identity SCM (gate=1, shift=0) — instrumentation check
    print(f"\n{'='*60}")
    print(f"Ablation 1: Identity SCM (50 games)")
    print(f"{'='*60}")
    id_scm = make_identity_scm().to(torch.device(device))
    pins_id = _run_or_resume(
        "ablation_identity_scm", 50,
        lambda pp: run_scm_games(network, id_scm, greedy_policy, 50, SP_CONFIG, alpha, "abl_identity", progress_path=pp),
    )
    results["identity_scm"] = compute_condition_stats(pins_id)

    # Ablation 2: Untrained (random init) SCM
    print(f"\n{'='*60}")
    print(f"Ablation 2: Untrained SCM (50 games)")
    print(f"{'='*60}")
    rand_scm = make_fresh_scm().to(torch.device(device))
    pins_rand = _run_or_resume(
        "ablation_untrained_scm", 50,
        lambda pp: run_scm_games(network, rand_scm, greedy_policy, 50, SP_CONFIG, alpha, "abl_untrained", progress_path=pp),
    )
    results["untrained_scm"] = compute_condition_stats(pins_rand)

    # Ablation 3: Temperature tau=1.0 (full sample from policy)
    print(f"\n{'='*60}")
    print(f"Ablation 3: Temperature tau=1.0 (50 games)")
    print(f"{'='*60}")
    pins_temp = _run_or_resume(
        "ablation_temp_1.0", 50,
        lambda pp: run_temperature_games(network, greedy_policy, 50, SP_CONFIG, 1.0, "abl_temp10", progress_path=pp),
    )
    results["temp_1.0"] = compute_condition_stats(pins_temp)

    # Ablation 4: Dirichlet alpha=0.3 (AlphaZero-style root noise)
    print(f"\n{'='*60}")
    print(f"Ablation 4: Dirichlet alpha=0.3 (50 games)")
    print(f"{'='*60}")
    pins_dir = _run_or_resume(
        "ablation_dirichlet_0.3", 50,
        lambda pp: run_dirichlet_games(network, greedy_policy, 50, SP_CONFIG, 0.3, 0.25, "abl_dir03", progress_path=pp),
    )
    results["dirichlet_0.3"] = compute_condition_stats(pins_dir)

    # Comparisons vs baseline and vs trained SCM
    for key in list(results.keys()):
        if key not in ("baseline", "trained_scm"):
            results[f"{key}_vs_baseline"] = compute_comparison_stats(
                results["baseline"]["pins_list"], results[key]["pins_list"]
            )
            results[f"{key}_vs_trained"] = compute_comparison_stats(
                results["trained_scm"]["pins_list"], results[key]["pins_list"]
            )

    _save_results(results, "ablation_results.json")
    print(f"\n  Ablation results saved to {OUTPUT_DIR}/ablation_results.json")


def phase_interpretability(args):
    """Run interpretability analysis on strong model."""
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_dir = os.path.join(OUTPUT_DIR, "interp_logs")
    os.makedirs(log_dir, exist_ok=True)

    network = load_network(STRONG_CKPT, device)
    scm_path = os.path.join(OUTPUT_DIR, "scm_strong_model.pt")
    if not os.path.exists(scm_path):
        scm_path = EXISTING_SCM
    scm = load_scm(scm_path, device)

    alpha = args.best_alpha
    print(f"\n{'='*60}")
    print(f"Interpretability: {N_INTERP} games with logging, alpha={alpha}")
    print(f"{'='*60}")

    pins = run_scm_games(
        network, scm, greedy_policy, N_INTERP, SP_CONFIG, alpha,
        "interp", log_dir=log_dir,
    )
    stats = compute_condition_stats(pins)
    _save_results({"interpretability": stats}, "interp_results.json")
    print(f"  Interpretability logs saved to {log_dir}/")
    print(f"  Run scripts/analyze_scm_logs.py on {log_dir} to generate plots")


def phase_stats(args):
    """Compute statistics from saved results and generate CSV + notes."""
    print(f"\n{'='*60}")
    print("Computing statistics and generating deliverables")
    print(f"{'='*60}")

    core_path = os.path.join(OUTPUT_DIR, "core_results.json")
    ablation_path = os.path.join(OUTPUT_DIR, "ablation_results.json")

    rows = []

    if os.path.exists(core_path):
        with open(core_path) as f:
            core = json.load(f)
        rows.extend(_results_to_rows(core, "core"))

    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)
        rows.extend(_results_to_rows(ablation, "ablation"))

    # Write CSV
    csv_path = os.path.join(OUTPUT_DIR, "results_table.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {csv_path} ({len(rows)} rows)")

    # Write results note
    note_path = os.path.join(OUTPUT_DIR, "results_note.md")
    _write_results_note(note_path, core_path, ablation_path)
    print(f"  Wrote {note_path}")


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def _save_results(results: dict, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    serializable = {}
    for key, val in results.items():
        if isinstance(val, dict):
            serializable[key] = {
                k: v for k, v in val.items() if k != "pins_list"
            }
            if "pins_list" in val:
                serializable[key]["pins_list"] = val["pins_list"]
        else:
            serializable[key] = val
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _results_to_rows(data: dict, group: str) -> list[dict]:
    """Convert results dict to flat CSV rows."""
    rows = []
    for key, val in data.items():
        if not isinstance(val, dict) or "n" not in val:
            continue
        row = {
            "group": group,
            "condition": key,
            "n": val.get("n", ""),
            "mean_pins": f"{val.get('mean', 0):.2f}",
            "std_pins": f"{val.get('std', 0):.2f}",
            "median_pins": f"{val.get('median', 0):.1f}",
            "max_pins": val.get("max", ""),
            "wins": val.get("wins", ""),
            "win_rate": f"{val.get('win_rate', 0):.4f}",
            "wilson_ci": f"[{val.get('wilson_ci_lo', 0):.4f}, {val.get('wilson_ci_hi', 0):.4f}]",
        }
        # Check for comparison stats
        comp_key = f"{key}_vs_baseline" if f"{key}_vs_baseline" in data else None
        if not comp_key:
            # Try matching comparison keys
            for ck in data:
                if ck.endswith("_comparison") and key.replace("_baseline", "").replace("_scm", "") in ck:
                    comp_key = ck
                    break
        if comp_key and comp_key in data:
            comp = data[comp_key]
            row["fisher_p"] = f"{comp.get('fisher_p', ''):.4f}" if comp.get("fisher_p") is not None else ""
            row["welch_p"] = f"{comp.get('welch_p', ''):.4f}" if comp.get("welch_p") is not None else ""
            row["cohens_d"] = f"{comp.get('cohens_d', ''):.3f}" if comp.get("cohens_d") is not None else ""
            row["bootstrap_diff"] = f"{comp.get('bootstrap_mean_diff', ''):.3f}" if comp.get("bootstrap_mean_diff") is not None else ""
            row["bootstrap_ci"] = (
                f"[{comp.get('bootstrap_ci_lo', ''):.3f}, {comp.get('bootstrap_ci_hi', ''):.3f}]"
                if comp.get("bootstrap_ci_lo") is not None else ""
            )
        else:
            row.update({"fisher_p": "", "welch_p": "", "cohens_d": "", "bootstrap_diff": "", "bootstrap_ci": ""})

        rows.append(row)
    return rows


def _write_results_note(path: str, core_path: str, ablation_path: str) -> None:
    """Write results_note.md summarizing significance."""
    lines = ["# SCM Paper Results Summary\n"]

    if os.path.exists(core_path):
        with open(core_path) as f:
            core = json.load(f)
        lines.append("## Core Results\n")
        for key in ["strong_greedy", "strong_heuristic", "transfer_greedy"]:
            comp_key = f"{key}_comparison"
            if comp_key in core:
                comp = core[comp_key]
                base_key = f"{key}_baseline"
                scm_key = f"{key}_scm"
                base = core.get(base_key, {})
                scm_data = core.get(scm_key, {})

                sig = "YES" if comp.get("welch_p", 1.0) < 0.05 else "NO"
                lines.append(f"### {key}")
                lines.append(f"- Baseline: {base.get('mean', '?'):.2f} +/- {base.get('std', '?'):.2f} pins (n={base.get('n', '?')})")
                lines.append(f"- SCM: {scm_data.get('mean', '?'):.2f} +/- {scm_data.get('std', '?'):.2f} pins (n={scm_data.get('n', '?')})")
                lines.append(f"- Welch t-test p={comp.get('welch_p', '?'):.4f}, Cohen's d={comp.get('cohens_d', '?'):.3f}")
                lines.append(f"- Fisher exact p={comp.get('fisher_p', '?'):.4f}")
                lines.append(f"- Bootstrap 95% CI on diff: [{comp.get('bootstrap_ci_lo', '?'):.3f}, {comp.get('bootstrap_ci_hi', '?'):.3f}]")
                lines.append(f"- **Significant at p<0.05: {sig}**\n")

    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)
        lines.append("## Ablation Results\n")
        base = ablation.get("baseline", {})
        trained = ablation.get("trained_scm", {})
        lines.append(f"- Baseline: {base.get('mean', '?'):.2f} pins")
        lines.append(f"- Trained SCM: {trained.get('mean', '?'):.2f} pins\n")

        for key in ablation:
            if key.endswith("_vs_baseline") and not key.startswith("trained"):
                cond = key.replace("_vs_baseline", "")
                comp = ablation[key]
                cond_data = ablation.get(cond, {})
                sig = "YES" if comp.get("welch_p", 1.0) < 0.05 else "NO"
                lines.append(f"### {cond}")
                lines.append(f"- Mean: {cond_data.get('mean', '?'):.2f} pins")
                lines.append(f"- vs baseline: Welch p={comp.get('welch_p', '?'):.4f}, d={comp.get('cohens_d', '?'):.3f}")
                lines.append(f"- Significant vs baseline: {sig}\n")

    # Bonferroni note
    lines.append("## Bonferroni Correction Note\n")
    lines.append("With k comparisons, the corrected threshold is p < 0.05/k.")
    lines.append("Check each p-value against the corrected threshold above.\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCM Paper Experiments")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["collect", "eval", "ablations", "interp", "stats", "all"],
                        help="Which phase to run")
    parser.add_argument("--collect-games", type=int, default=1100,
                        help="Games for SCM data collection (default 1100)")
    parser.add_argument("--train-epochs", type=int, default=100,
                        help="SCM training epochs (default 100)")
    parser.add_argument("--best-alpha", type=float, default=BEST_ALPHA,
                        help=f"Best blend alpha (default {BEST_ALPHA})")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel workers (0=auto)")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    start = time.time()

    if args.phase in ("collect", "all"):
        phase_collect_and_train(args)

    if args.phase in ("eval", "all"):
        phase_eval(args)

    if args.phase in ("ablations", "all"):
        phase_ablations(args)

    if args.phase in ("interp", "all"):
        phase_interpretability(args)

    if args.phase in ("stats", "all"):
        phase_stats(args)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
