#!/usr/bin/env python3
"""
analyze_scm_logs.py — Interpretability analysis for SCM modulation logs.

Reads JSONL log files produced by SCMLogger and generates:
1. Modulation by game phase (opening/midgame/endgame)
2. Modulation by advantage (ahead/behind/tied)
3. Modulation by search confidence (certain/uncertain)
4. Hidden state trajectory analysis
5. Action override rate by phase
6. Summary statistics table

Usage:
    python scripts/analyze_scm_logs.py --log-dir experiments/scm/scm_logs
    python scripts/analyze_scm_logs.py --log-file experiments/scm/scm_logs/scm_logs_1234.jsonl
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_logs(log_path: str) -> list[dict]:
    """Load SCM log entries from a JSONL file or directory of JSONL files."""
    path = Path(log_path)
    entries = []

    if path.is_dir():
        for f in sorted(path.glob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        entries.append(json.loads(line))
    elif path.is_file():
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    entries.append(json.loads(line))
    else:
        print(f"ERROR: {log_path} not found")
        sys.exit(1)

    return entries


def classify_advantage(entry: dict) -> str:
    """Classify whether the agent is ahead, behind, or tied."""
    diff = entry["pins_in_goal"] - entry["opp_pins_in_goal"]
    if diff > 0:
        return "ahead"
    elif diff < 0:
        return "behind"
    return "tied"


def classify_confidence(entry: dict) -> str:
    """Classify search confidence into high/medium/low."""
    entropy = entry["search_confidence"]
    if entropy < 1.0:
        return "high"
    elif entropy < 2.5:
        return "medium"
    return "low"


def analyze_by_group(entries: list[dict], group_fn, group_name: str) -> dict:
    """Aggregate modulation stats by a grouping function."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        groups[group_fn(e)].append(e)

    results = {}
    for label, group in sorted(groups.items()):
        n = len(group)
        gate_mag = np.mean([e["gate_magnitude"] for e in group])
        shift_mag = np.mean([e["shift_magnitude"] for e in group])
        gate_mean = np.mean([e["gate_mean"] for e in group])
        gate_std = np.mean([e["gate_std"] for e in group])
        shift_mean = np.mean([e["shift_mean"] for e in group])
        shift_std = np.mean([e["shift_std"] for e in group])
        override_rate = np.mean([1.0 if e["changed_top1"] else 0.0 for e in group])
        hidden_norm = np.mean([e["gru_hidden_norm"] for e in group])
        kl = np.mean([e["policy_search_kl"] for e in group])

        results[label] = {
            "count": n,
            "gate_magnitude": round(gate_mag, 4),
            "shift_magnitude": round(shift_mag, 4),
            "gate_mean": round(gate_mean, 4),
            "gate_std": round(gate_std, 4),
            "shift_mean": round(shift_mean, 4),
            "shift_std": round(shift_std, 4),
            "override_rate": round(override_rate, 4),
            "hidden_norm": round(hidden_norm, 4),
            "policy_search_kl": round(kl, 4),
        }

    return results


def print_table(title: str, results: dict) -> None:
    """Print a formatted table of analysis results."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    if not results:
        print("  No data")
        return

    header = f"  {'Group':<12} {'N':>6} {'GateMag':>9} {'ShiftMag':>9} {'GateMu':>8} {'ShiftMu':>8} {'Override':>9} {'HidNorm':>9} {'KL':>8}"
    print(header)
    print(f"  {'-'*76}")

    for label, stats in results.items():
        print(
            f"  {label:<12} {stats['count']:>6} "
            f"{stats['gate_magnitude']:>9.4f} {stats['shift_magnitude']:>9.4f} "
            f"{stats['gate_mean']:>8.4f} {stats['shift_mean']:>8.4f} "
            f"{stats['override_rate']:>9.1%} {stats['hidden_norm']:>9.4f} "
            f"{stats['policy_search_kl']:>8.4f}"
        )


def analyze_hidden_state_trajectory(entries: list[dict]) -> None:
    """Analyze how the GRU hidden state norm evolves across game turns."""
    games: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        games[e["game_id"]].append(e)

    print(f"\n{'='*80}")
    print("  Hidden State Trajectory (GRU hidden norm over game turns)")
    print(f"{'='*80}")

    # Average hidden norm at each turn position
    turn_norms: dict[int, list[float]] = defaultdict(list)
    for game_id, game_entries in games.items():
        for e in sorted(game_entries, key=lambda x: x["turn"]):
            turn_norms[e["turn"]].append(e["gru_hidden_norm"])

    print(f"\n  {'Turn':>6} {'AvgHidNorm':>12} {'StdHidNorm':>12} {'N':>6}")
    print(f"  {'-'*40}")
    for turn in sorted(turn_norms.keys()):
        norms = turn_norms[turn]
        print(f"  {turn:>6} {np.mean(norms):>12.4f} {np.std(norms):>12.4f} {len(norms):>6}")


def analyze_gate_selectivity(entries: list[dict]) -> None:
    """Check if certain action slots are consistently gated differently."""
    print(f"\n{'='*80}")
    print("  Gate Selectivity (top-5 actions: are certain slots consistently gated?)")
    print(f"{'='*80}")

    # Aggregate top-5 gate values across all entries
    all_top5_gates = [e["top5_gate"] for e in entries if "top5_gate" in e]
    all_top5_shifts = [e["top5_shift"] for e in entries if "top5_shift" in e]

    if all_top5_gates:
        gates_arr = np.array(all_top5_gates)
        shifts_arr = np.array(all_top5_shifts)
        print(f"\n  Position in top-5 visited actions:")
        print(f"  {'Rank':>6} {'AvgGate':>10} {'StdGate':>10} {'AvgShift':>10} {'StdShift':>10}")
        print(f"  {'-'*48}")
        for i in range(min(5, gates_arr.shape[1])):
            print(
                f"  {i+1:>6} {gates_arr[:, i].mean():>10.4f} {gates_arr[:, i].std():>10.4f} "
                f"{shifts_arr[:, i].mean():>10.4f} {shifts_arr[:, i].std():>10.4f}"
            )


def generate_plots(entries: list[dict], plot_dir: str) -> list[str]:
    """Generate all interpretability plots. Returns list of saved file paths."""
    os.makedirs(plot_dir, exist_ok=True)
    saved: list[str] = []

    games: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        games[e["game_id"]].append(e)

    # Consistent colours
    PHASE_COLORS = {"opening": "#2196F3", "midgame": "#FF9800", "endgame": "#4CAF50"}
    ADV_COLORS = {"ahead": "#4CAF50", "behind": "#F44336", "tied": "#9E9E9E"}
    CONF_COLORS = {"high": "#2196F3", "medium": "#FF9800", "low": "#F44336"}

    # ---- Figure 1: Hidden State Trajectory ----
    fig, ax = plt.subplots(figsize=(12, 5))
    turn_norms: dict[int, list[float]] = defaultdict(list)
    for game_entries in games.values():
        for e in game_entries:
            turn_norms[e["turn"]].append(e["gru_hidden_norm"])

    turns = sorted(turn_norms.keys())
    means = [np.mean(turn_norms[t]) for t in turns]
    stds = [np.std(turn_norms[t]) for t in turns]
    means_arr, stds_arr = np.array(means), np.array(stds)

    ax.plot(turns, means, color="#1565C0", linewidth=2, label="Mean hidden norm")
    ax.fill_between(turns, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.2, color="#1565C0", label="\u00b11 std")
    # Phase boundaries
    ax.axvline(x=10, color="#999", linestyle="--", alpha=0.6, label="Phase boundary")
    ax.axvline(x=30, color="#999", linestyle="--", alpha=0.6)
    ax.text(5, max(means) * 0.95, "Opening", ha="center", fontsize=10, color="#2196F3", weight="bold")
    ax.text(20, max(means) * 0.95, "Midgame", ha="center", fontsize=10, color="#FF9800", weight="bold")
    ax.text(max(turns) * 0.65, max(means) * 0.95, "Endgame", ha="center", fontsize=10, color="#4CAF50", weight="bold")
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("GRU Hidden State Norm", fontsize=12)
    ax.set_title("GRU Hidden State Trajectory Across Game", fontsize=14, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    p = os.path.join(plot_dir, "01_hidden_state_trajectory.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ---- Figure 2: Modulation by Game Phase (grouped bar chart) ----
    phase_groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        phase_groups[e["phase"]].append(e)

    phases = ["opening", "midgame", "endgame"]
    metrics = {
        "Gate Magnitude": lambda g: np.mean([e["gate_magnitude"] for e in g]),
        "Shift Magnitude": lambda g: np.mean([e["shift_magnitude"] for e in g]),
        "Override Rate (%)": lambda g: np.mean([100.0 if e["changed_top1"] else 0.0 for e in g]),
        "Hidden Norm": lambda g: np.mean([e["gru_hidden_norm"] for e in g]),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (metric_name, metric_fn) in zip(axes, metrics.items()):
        vals = [metric_fn(phase_groups[ph]) if ph in phase_groups else 0 for ph in phases]
        colors = [PHASE_COLORS[ph] for ph in phases]
        bars = ax.bar(phases, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(metric_name, fontsize=11, weight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Modulation by Game Phase", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(plot_dir, "02_modulation_by_phase.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ---- Figure 3: Modulation by Advantage ----
    adv_groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        adv_groups[classify_advantage(e)].append(e)

    adv_order = ["behind", "tied", "ahead"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    adv_metrics = {
        "Shift Magnitude": lambda g: np.mean([e["shift_magnitude"] for e in g]),
        "Override Rate (%)": lambda g: np.mean([100.0 if e["changed_top1"] else 0.0 for e in g]),
        "Gate Mean": lambda g: np.mean([e["gate_mean"] for e in g]),
    }
    for ax, (mname, mfn) in zip(axes, adv_metrics.items()):
        vals = [mfn(adv_groups[a]) if a in adv_groups else 0 for a in adv_order]
        colors = [ADV_COLORS[a] for a in adv_order]
        bars = ax.bar(adv_order, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(mname, fontsize=11, weight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Modulation by Advantage State", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(plot_dir, "03_modulation_by_advantage.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ---- Figure 4: Modulation by Search Confidence ----
    conf_groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        conf_groups[classify_confidence(e)].append(e)

    conf_order = ["high", "medium", "low"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    conf_metrics = {
        "Override Rate (%)": lambda g: np.mean([100.0 if e["changed_top1"] else 0.0 for e in g]),
        "Gate Mean": lambda g: np.mean([e["gate_mean"] for e in g]),
        "Shift Magnitude": lambda g: np.mean([e["shift_magnitude"] for e in g]),
    }
    for ax, (mname, mfn) in zip(axes, conf_metrics.items()):
        vals = [mfn(conf_groups[c]) if c in conf_groups else 0 for c in conf_order]
        colors = [CONF_COLORS[c] for c in conf_order]
        bars = ax.bar(conf_order, vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(mname, fontsize=11, weight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Modulation by Search Confidence", fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(plot_dir, "04_modulation_by_confidence.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # ---- Figure 5: Gate & Shift over turns (per-turn means) ----
    turn_gate_mag: dict[int, list[float]] = defaultdict(list)
    turn_shift_mag: dict[int, list[float]] = defaultdict(list)
    turn_override: dict[int, list[float]] = defaultdict(list)
    for e in entries:
        turn_gate_mag[e["turn"]].append(e["gate_magnitude"])
        turn_shift_mag[e["turn"]].append(e["shift_magnitude"])
        turn_override[e["turn"]].append(1.0 if e["changed_top1"] else 0.0)

    turns_sorted = sorted(turn_gate_mag.keys())

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Gate magnitude
    gm = [np.mean(turn_gate_mag[t]) for t in turns_sorted]
    ax1.plot(turns_sorted, gm, color="#E91E63", linewidth=1.5)
    ax1.axvline(x=10, color="#999", linestyle="--", alpha=0.5)
    ax1.axvline(x=30, color="#999", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Gate Magnitude", fontsize=11)
    ax1.set_title("Modulation Strength Over Game Turns", fontsize=14, weight="bold")
    ax1.grid(True, alpha=0.3)

    # Shift magnitude
    sm = [np.mean(turn_shift_mag[t]) for t in turns_sorted]
    ax2.plot(turns_sorted, sm, color="#9C27B0", linewidth=1.5)
    ax2.axvline(x=10, color="#999", linestyle="--", alpha=0.5)
    ax2.axvline(x=30, color="#999", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Shift Magnitude", fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Override rate
    ovr = [np.mean(turn_override[t]) * 100 for t in turns_sorted]
    ax3.plot(turns_sorted, ovr, color="#FF5722", linewidth=1.5)
    ax3.axvline(x=10, color="#999", linestyle="--", alpha=0.5)
    ax3.axvline(x=30, color="#999", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Override Rate (%)", fontsize=11)
    ax3.set_xlabel("Turn Number", fontsize=11)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(plot_dir, "05_modulation_over_turns.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ---- Figure 6: Per-game hidden state trajectories (spaghetti + mean) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for game_id, game_entries in games.items():
        sorted_e = sorted(game_entries, key=lambda x: x["turn"])
        turns_g = [e["turn"] for e in sorted_e]
        norms_g = [e["gru_hidden_norm"] for e in sorted_e]
        ax.plot(turns_g, norms_g, alpha=0.15, color="#1565C0", linewidth=0.8)

    # Mean on top
    ax.plot(turns, means, color="#D32F2F", linewidth=2.5, label="Mean across games", zorder=10)
    ax.axvline(x=10, color="#999", linestyle="--", alpha=0.6)
    ax.axvline(x=30, color="#999", linestyle="--", alpha=0.6)
    ax.set_xlabel("Turn Number", fontsize=12)
    ax.set_ylabel("GRU Hidden State Norm", fontsize=12)
    ax.set_title("Per-Game Hidden State Trajectories", fontsize=14, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(plot_dir, "06_per_game_trajectories.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ---- Figure 7: Gate selectivity for top-5 actions ----
    all_top5_gates = [e["top5_gate"] for e in entries if "top5_gate" in e and len(e["top5_gate"]) == 5]
    all_top5_shifts = [e["top5_shift"] for e in entries if "top5_shift" in e and len(e["top5_shift"]) == 5]

    if all_top5_gates:
        gates_arr = np.array(all_top5_gates)
        shifts_arr = np.array(all_top5_shifts)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ranks = [f"Rank {i+1}" for i in range(5)]

        gate_means = [gates_arr[:, i].mean() for i in range(5)]
        gate_stds = [gates_arr[:, i].std() for i in range(5)]
        ax1.bar(ranks, gate_means, yerr=gate_stds, color="#2196F3",
                edgecolor="white", linewidth=1.5, capsize=4)
        ax1.set_title("Gate Values by Visit Rank", fontsize=11, weight="bold")
        ax1.set_ylabel("Gate Value")
        ax1.axhline(y=1.0, color="#999", linestyle="--", alpha=0.5, label="Identity (1.0)")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        shift_means = [shifts_arr[:, i].mean() for i in range(5)]
        shift_stds = [shifts_arr[:, i].std() for i in range(5)]
        colors = ["#4CAF50" if s > 0 else "#F44336" for s in shift_means]
        ax2.bar(ranks, shift_means, yerr=shift_stds, color=colors,
                edgecolor="white", linewidth=1.5, capsize=4)
        ax2.set_title("Shift Values by Visit Rank", fontsize=11, weight="bold")
        ax2.set_ylabel("Shift Value")
        ax2.axhline(y=0.0, color="#999", linestyle="--", alpha=0.5, label="Identity (0.0)")
        ax2.legend()
        ax2.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Gate Selectivity: Top-1 Boosted, Lower Ranks Suppressed",
                     fontsize=13, weight="bold", y=1.02)
        fig.tight_layout()
        p = os.path.join(plot_dir, "07_gate_selectivity.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Analyze SCM modulation logs")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory of JSONL log files")
    parser.add_argument("--log-file", type=str, default=None, help="Single JSONL log file")
    parser.add_argument("--output", type=str, default=None, help="Save analysis to JSON file")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory for plots (default: <log_location>/plots)")
    args = parser.parse_args()

    log_path = args.log_dir or args.log_file
    if not log_path:
        print("ERROR: Provide --log-dir or --log-file")
        sys.exit(1)

    entries = load_logs(log_path)
    print(f"Loaded {len(entries)} log entries")

    if not entries:
        print("No entries to analyze")
        return

    # Unique games
    game_ids = set(e["game_id"] for e in entries)
    print(f"From {len(game_ids)} games")

    # 1. By game phase
    phase_results = analyze_by_group(entries, lambda e: e["phase"], "Game Phase")
    print_table("Modulation by Game Phase", phase_results)

    # 2. By advantage
    advantage_results = analyze_by_group(entries, classify_advantage, "Advantage")
    print_table("Modulation by Advantage (ahead/behind/tied)", advantage_results)

    # 3. By search confidence
    confidence_results = analyze_by_group(entries, classify_confidence, "Confidence")
    print_table("Modulation by Search Confidence", confidence_results)

    # 4. Hidden state trajectory
    analyze_hidden_state_trajectory(entries)

    # 5. Gate selectivity
    analyze_gate_selectivity(entries)

    # 6. Overall summary
    print(f"\n{'='*80}")
    print("  Overall Summary")
    print(f"{'='*80}")
    overall_override = np.mean([1.0 if e["changed_top1"] else 0.0 for e in entries])
    overall_gate_mag = np.mean([e["gate_magnitude"] for e in entries])
    overall_shift_mag = np.mean([e["shift_magnitude"] for e in entries])
    print(f"  Total turns analyzed:  {len(entries)}")
    print(f"  Action override rate:  {overall_override:.1%}")
    print(f"  Avg gate magnitude:    {overall_gate_mag:.4f}")
    print(f"  Avg shift magnitude:   {overall_shift_mag:.4f}")

    # Key research question: does modulation vary systematically?
    if phase_results:
        phases = list(phase_results.values())
        gate_range = max(p["gate_magnitude"] for p in phases) - min(p["gate_magnitude"] for p in phases)
        shift_range = max(p["shift_magnitude"] for p in phases) - min(p["shift_magnitude"] for p in phases)
        override_range = max(p["override_rate"] for p in phases) - min(p["override_rate"] for p in phases)
        print(f"\n  Phase-dependent variation (range across phases):")
        print(f"    Gate magnitude range:   {gate_range:.4f}")
        print(f"    Shift magnitude range:  {shift_range:.4f}")
        print(f"    Override rate range:     {override_range:.1%}")
        if gate_range > 0.5 or shift_range > 0.5 or override_range > 0.1:
            print(f"  >>> SIGNIFICANT phase-dependent modulation detected! <<<")
        else:
            print(f"  (Modulation appears roughly uniform across phases)")

    # Generate plots
    plot_dir = args.plot_dir
    if plot_dir is None:
        base = args.log_dir or str(Path(args.log_file).parent)
        plot_dir = os.path.join(base, "plots")
    print(f"\nGenerating plots in {plot_dir}...")
    saved_plots = generate_plots(entries, plot_dir)
    for p in saved_plots:
        print(f"  Saved: {p}")
    print(f"  {len(saved_plots)} plots generated")

    # Save if requested
    if args.output:
        analysis = {
            "num_entries": len(entries),
            "num_games": len(game_ids),
            "by_phase": phase_results,
            "by_advantage": advantage_results,
            "by_confidence": confidence_results,
            "overall_override_rate": round(overall_override, 4),
            "overall_gate_magnitude": round(overall_gate_mag, 4),
            "overall_shift_magnitude": round(overall_shift_mag, 4),
        }
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n  Analysis saved to {args.output}")


if __name__ == "__main__":
    main()
