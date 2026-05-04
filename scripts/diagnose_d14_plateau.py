"""
diagnose_d14_plateau.py — Self-contained diagnostics for the d14 plateau.

Runs four diagnostics:
  1. MCTS-vs-policy quality (50 random states; sims 50/100/200)
  2. Value-head sanity (correlation of net/heuristic value vs ground truth)
  3. Architecture capacity (policy entropy vs MCTS entropy)
  4. Gradient-signal check (value vs policy grad norms at vw=0.25 and vw=1.0)

Reports numbers; no theorising past data.
"""
import os
import sys
import numpy as np
import random

# Ensure project root on path
PROJECT_ROOT = "/home/coder/reinforcement-learning-on-chinese-checkers"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.env.state_encoder import StateEncoder
from src.env.board_wrapper import BoardWrapper
from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.search.batched_mcts import BatchedAlphaZeroMCTS
from src.search.mcts import _heuristic_value, _score_colour
from src.training.warmstart_generator import _noisy_heuristic_policy
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy

CHECKPOINT = os.path.join(PROJECT_ROOT, "experiments/exp_d14_perspective_fix/best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 12345

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_network():
    cfg = NetworkConfig(num_blocks=9, num_filters=96, architecture="resnet")
    net = AlphaZeroNet(config=cfg, device=DEVICE)
    net.load_checkpoint(CHECKPOINT)
    net.model.eval()
    return net


def make_env_for_board(board: BoardWrapper, agent_colour: str = "red", opp_colour: str = "blue") -> ChineseCheckersEnv:
    """Build a stripped MCTS-friendly env around an existing board."""
    env = ChineseCheckersEnv(opponent_policy="none", max_steps=200)
    env._AGENT_COLOUR = agent_colour
    env._OPPONENT_COLOUR = opp_colour
    env._TURN_ORDER = [agent_colour, opp_colour]
    env._board = board
    env._step_count = 0
    env._terminated = False
    env._truncated = False
    env._no_opponent = True
    return env


def collect_random_states(num_states: int, alpha: float = 0.5, max_attempts: int = 5000) -> list[BoardWrapper]:
    """Play heuristic-vs-heuristic noisy games and snapshot mid-game boards."""
    states = []
    while len(states) < num_states:
        board = BoardWrapper(["red", "blue"])
        # Random number of moves played before snapshot (10..40)
        target_moves = np.random.randint(10, 40)
        moves_done = 0
        colours = ["red", "blue"]
        legal_each = True
        for _ in range(target_moves * 2):
            colour = colours[moves_done % 2]
            if board.check_win("red") or board.check_win("blue"):
                legal_each = False
                break
            try:
                pid, dest, _ = _noisy_heuristic_policy(board, colour, alpha=alpha, noise_frac=0.25, fast=True)
            except ValueError:
                legal_each = False
                break
            board.apply_move(colour, pid, dest)
            moves_done += 1
            if moves_done >= target_moves * 2:
                break
        if not legal_each:
            continue
        # Make sure red has legal moves
        if not board.get_legal_moves("red"):
            continue
        states.append(board)
        if len(states) >= num_states:
            break
    return states


def play_greedy_vs_greedy(board: BoardWrapper, mover_colour: str, num_moves: int = 30) -> dict:
    """Play `num_moves` half-moves of greedy-vs-greedy on a CLONE.

    Returns final pins_in_goal for each colour and final total_distance.
    """
    b = board.clone()
    other = "blue" if mover_colour == "red" else "red"
    colours = [mover_colour, other]
    for i in range(num_moves):
        c = colours[i % 2]
        legal = b.get_legal_moves(c)
        if not legal:
            break
        if b.check_win("red") or b.check_win("blue"):
            break
        try:
            pid, dest = greedy_policy(b, c)
        except (ValueError, Exception):
            break
        b.apply_move(c, pid, dest)
    return {
        "red_pins": b.pins_in_goal("red"),
        "blue_pins": b.pins_in_goal("blue"),
        "red_dist": b.total_distance_to_goal("red"),
        "blue_dist": b.total_distance_to_goal("blue"),
    }


def evaluate_move_quality(board: BoardWrapper, mover: str, pin_id: int, dest: int, num_followup_moves: int = 30) -> float:
    """Apply a move on a clone, then play 30 greedy half-moves, then return mover-perspective score.

    Score = (mover_pins_gain) - (opponent_pins_gain) + small distance bonus.
    Higher = better outcome for mover.
    """
    b = board.clone()
    other = "blue" if mover == "red" else "red"
    initial_mover_pins = b.pins_in_goal(mover)
    initial_other_pins = b.pins_in_goal(other)
    initial_mover_dist = b.total_distance_to_goal(mover)
    initial_other_dist = b.total_distance_to_goal(other)
    b.apply_move(mover, pin_id, dest)
    # Now opponent moves first in the rollout (mover already moved)
    out = play_greedy_vs_greedy(b, mover_colour=other, num_moves=num_followup_moves)
    final_mover_pins = out["red_pins"] if mover == "red" else out["blue_pins"]
    final_other_pins = out["red_pins"] if other == "red" else out["blue_pins"]
    final_mover_dist = out["red_dist"] if mover == "red" else out["blue_dist"]
    final_other_dist = out["red_dist"] if other == "red" else out["blue_dist"]
    # Pin advance
    mover_pins_gain = final_mover_pins - initial_mover_pins
    other_pins_gain = final_other_pins - initial_other_pins
    # Distance reduction (positive = mover got closer)
    mover_dist_drop = initial_mover_dist - final_mover_dist
    other_dist_drop = initial_other_dist - final_other_dist
    # Combined score, mover-perspective
    score = (mover_pins_gain - other_pins_gain) * 100.0 + (mover_dist_drop - other_dist_drop) * 1.0
    return float(score)


# ----------------------------------------------------------------------
# Diagnostic 1: MCTS-vs-policy quality
# ----------------------------------------------------------------------

def diagnostic_1_mcts_vs_policy(net, states: list[BoardWrapper]):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: MCTS-vs-policy quality")
    print("=" * 70)

    encoder = StateEncoder(grid_size=17, num_channels=10)
    mapper = ActionMapper(num_pins=10, num_cells=121)

    sim_counts = [50, 100, 200]
    results = {}

    for sims in sim_counts:
        print(f"\n  --- sims={sims} ---")
        n_disagreements = 0
        n_mcts_wins = 0
        n_policy_wins = 0
        n_ties = 0
        score_diffs = []  # mcts_score - policy_score

        for i, board in enumerate(states):
            mover = "red"
            other = "blue"
            # Network argmax (raw policy)
            obs = encoder.encode(board, current_colour=mover, turn_order=[mover, other])
            legal = board.get_legal_moves(mover)
            mask = mapper.build_action_mask(legal)
            if encoder.needs_rotation(mover):
                mask_canon = encoder.rotate_action_distribution(mask.astype(np.bool_)).astype(np.bool_)
                priors_canon, _ = net.predict(obs, mask_canon)
                priors_raw = encoder.rotate_action_distribution(priors_canon)
            else:
                priors_raw, _ = net.predict(obs, mask)
            policy_action = int(np.argmax(priors_raw))
            policy_pid, policy_dest = mapper.decode(policy_action)

            # MCTS choice
            env = make_env_for_board(board.clone(), agent_colour=mover, opp_colour=other)
            mcts = BatchedAlphaZeroMCTS(
                network=net,
                num_simulations=sims,
                batch_size=8,
                dirichlet_epsilon=0.0,  # eval mode
                use_heuristic_value=True,
            )
            try:
                mcts_action = mcts.select_action(env, temperature=0.0)
            except Exception as e:
                print(f"    state {i}: MCTS error {e}")
                continue
            mcts_pid, mcts_dest = mapper.decode(mcts_action)

            if mcts_action == policy_action:
                continue  # Agreement

            n_disagreements += 1
            # Evaluate both moves with greedy-vs-greedy rollout
            policy_score = evaluate_move_quality(board, mover, policy_pid, policy_dest, num_followup_moves=30)
            mcts_score = evaluate_move_quality(board, mover, mcts_pid, mcts_dest, num_followup_moves=30)
            score_diffs.append(mcts_score - policy_score)
            if mcts_score > policy_score + 1e-6:
                n_mcts_wins += 1
            elif policy_score > mcts_score + 1e-6:
                n_policy_wins += 1
            else:
                n_ties += 1

        total_decisions = len(states)
        agreement_rate = (total_decisions - n_disagreements) / total_decisions
        if n_disagreements > 0:
            mcts_win_frac = n_mcts_wins / n_disagreements
            mean_diff = float(np.mean(score_diffs))
            median_diff = float(np.median(score_diffs))
        else:
            mcts_win_frac = float("nan")
            mean_diff = 0.0
            median_diff = 0.0

        print(f"    decisions={total_decisions}, agreements={total_decisions - n_disagreements} "
              f"({agreement_rate:.1%}), disagreements={n_disagreements}")
        if n_disagreements > 0:
            print(f"    MCTS wins: {n_mcts_wins}/{n_disagreements} ({mcts_win_frac:.1%})")
            print(f"    Policy wins: {n_policy_wins}/{n_disagreements} ({n_policy_wins / n_disagreements:.1%})")
            print(f"    Ties: {n_ties}/{n_disagreements}")
            print(f"    mean(mcts_score - policy_score) = {mean_diff:.2f}")
            print(f"    median(mcts_score - policy_score) = {median_diff:.2f}")

        results[sims] = {
            "agreements": total_decisions - n_disagreements,
            "disagreements": n_disagreements,
            "mcts_wins": n_mcts_wins,
            "policy_wins": n_policy_wins,
            "ties": n_ties,
            "mcts_win_frac": mcts_win_frac,
            "mean_score_diff": mean_diff,
        }
    return results


# ----------------------------------------------------------------------
# Diagnostic 2: Value-head sanity
# ----------------------------------------------------------------------

def diagnostic_2_value_head(net, states: list[BoardWrapper]):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Value-head sanity")
    print("=" * 70)

    encoder = StateEncoder(grid_size=17, num_channels=10)
    mapper = ActionMapper(num_pins=10, num_cells=121)

    # Filter for "interesting" states (pin diff >=2)
    interesting = []
    for board in states:
        diff = board.pins_in_goal("red") - board.pins_in_goal("blue")
        if abs(diff) >= 2:
            interesting.append(board)
    # If we don't have 100, augment by playing more games (but use what we have)
    print(f"  Filtered to {len(interesting)} states with |pin_diff| >= 2")
    target = min(100, len(interesting))
    if target < 30:
        # Generate more — these are rare in early game
        extra_needed = 100 - len(interesting)
        attempts = 0
        while len(interesting) < 100 and attempts < 500:
            attempts += 1
            board = BoardWrapper(["red", "blue"])
            target_moves = np.random.randint(20, 60)
            colours = ["red", "blue"]
            ok = True
            for k in range(target_moves * 2):
                c = colours[k % 2]
                if board.check_win("red") or board.check_win("blue"):
                    ok = False
                    break
                try:
                    pid, dest, _ = _noisy_heuristic_policy(board, c, alpha=0.5, noise_frac=0.25, fast=True)
                except ValueError:
                    ok = False
                    break
                board.apply_move(c, pid, dest)
            if not ok:
                continue
            if abs(board.pins_in_goal("red") - board.pins_in_goal("blue")) >= 2 and board.get_legal_moves("red"):
                interesting.append(board)
        print(f"  After augmentation: {len(interesting)} states")

    interesting = interesting[:100]
    net_values = []
    heur_values = []
    truth_values = []  # +1 = red ahead at end; -1 = blue ahead

    for board in interesting:
        mover = "red"
        other = "blue"
        obs = encoder.encode(board, current_colour=mover, turn_order=[mover, other])
        legal = board.get_legal_moves(mover)
        mask = mapper.build_action_mask(legal)
        if encoder.needs_rotation(mover):
            mask_canon = encoder.rotate_action_distribution(mask.astype(np.bool_)).astype(np.bool_)
            _, vnet = net.predict(obs, mask_canon)
        else:
            _, vnet = net.predict(obs, mask)

        env = make_env_for_board(board.clone(), agent_colour=mover, opp_colour=other)
        vheur = _heuristic_value(env)

        # Ground truth: 50 greedy half-moves
        out = play_greedy_vs_greedy(board, mover_colour=mover, num_moves=50)
        final_red = out["red_pins"]
        final_blue = out["blue_pins"]
        final_red_dist = out["red_dist"]
        final_blue_dist = out["blue_dist"]
        # Red-perspective ground truth
        pins_term = final_red - final_blue
        dist_term = (final_blue_dist - final_red_dist) / 100.0  # rough scale
        truth = pins_term + dist_term
        # Normalize to ~[-1, 1] by clipping
        truth = max(-10.0, min(10.0, truth))

        net_values.append(vnet)
        heur_values.append(vheur)
        truth_values.append(truth)

    net_values = np.array(net_values)
    heur_values = np.array(heur_values)
    truth_values = np.array(truth_values)

    def safe_corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    corr_net = safe_corr(net_values, truth_values)
    corr_heur = safe_corr(heur_values, truth_values)
    corr_net_heur = safe_corr(net_values, heur_values)

    print(f"\n  N = {len(net_values)} positions")
    print(f"  net values: mean={net_values.mean():.3f}, std={net_values.std():.3f}, min={net_values.min():.3f}, max={net_values.max():.3f}")
    print(f"  heuristic:  mean={heur_values.mean():.3f}, std={heur_values.std():.3f}, min={heur_values.min():.3f}, max={heur_values.max():.3f}")
    print(f"  truth:      mean={truth_values.mean():.3f}, std={truth_values.std():.3f}, min={truth_values.min():.3f}, max={truth_values.max():.3f}")
    print(f"\n  corr(network_value,    ground_truth) = {corr_net:+.3f}")
    print(f"  corr(heuristic_value,  ground_truth) = {corr_heur:+.3f}")
    print(f"  corr(network_value,    heuristic)    = {corr_net_heur:+.3f}")

    return {
        "net_corr_truth": corr_net,
        "heur_corr_truth": corr_heur,
        "net_std": float(net_values.std()),
        "heur_std": float(heur_values.std()),
    }


# ----------------------------------------------------------------------
# Diagnostic 3: Architecture capacity (policy sharpness)
# ----------------------------------------------------------------------

def diagnostic_3_capacity(net, states: list[BoardWrapper]):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Architecture capacity (policy sharpness)")
    print("=" * 70)

    encoder = StateEncoder(grid_size=17, num_channels=10)
    mapper = ActionMapper(num_pins=10, num_cells=121)

    n = min(10, len(states))
    print(f"  Using first {n} states\n")

    policy_entropies = []
    policy_max = []
    mcts_entropies = []
    mcts_max = []
    legal_counts = []

    print(f"  {'idx':>3}  {'legal':>5}  {'pol_max':>7}  {'pol_H':>6}  {'mcts_max':>8}  {'mcts_H':>6}  {'log(legal)':>10}")
    for i, board in enumerate(states[:n]):
        mover = "red"
        other = "blue"
        obs = encoder.encode(board, current_colour=mover, turn_order=[mover, other])
        legal = board.get_legal_moves(mover)
        mask = mapper.build_action_mask(legal)
        n_legal = int(mask.sum())
        if encoder.needs_rotation(mover):
            mask_canon = encoder.rotate_action_distribution(mask.astype(np.bool_)).astype(np.bool_)
            priors_canon, _ = net.predict(obs, mask_canon)
            priors_raw = encoder.rotate_action_distribution(priors_canon)
        else:
            priors_raw, _ = net.predict(obs, mask)
        legal_priors = priors_raw[mask.astype(np.bool_)]
        legal_priors = legal_priors / legal_priors.sum()
        pol_H = -float((legal_priors * np.log(np.clip(legal_priors, 1e-12, 1.0))).sum())
        pol_max = float(legal_priors.max())

        # MCTS visits at 50 sims
        env = make_env_for_board(board.clone(), agent_colour=mover, opp_colour=other)
        mcts = BatchedAlphaZeroMCTS(
            network=net,
            num_simulations=50,
            batch_size=8,
            dirichlet_epsilon=0.0,
            use_heuristic_value=True,
        )
        try:
            probs = mcts.get_action_probs(env, temperature=1.0)
        except Exception as e:
            print(f"    state {i}: MCTS error {e}")
            continue
        legal_probs = probs[mask.astype(np.bool_)]
        s = legal_probs.sum()
        if s > 0:
            legal_probs = legal_probs / s
        mcts_H = -float((legal_probs * np.log(np.clip(legal_probs, 1e-12, 1.0))).sum())
        mcts_max_v = float(legal_probs.max())

        policy_entropies.append(pol_H)
        policy_max.append(pol_max)
        mcts_entropies.append(mcts_H)
        mcts_max.append(mcts_max_v)
        legal_counts.append(n_legal)

        print(f"  {i:>3d}  {n_legal:>5d}  {pol_max:>7.3f}  {pol_H:>6.3f}  "
              f"{mcts_max_v:>8.3f}  {mcts_H:>6.3f}  {np.log(n_legal):>10.3f}")

    print(f"\n  Mean policy entropy: {np.mean(policy_entropies):.3f}  (max possible ~{np.log(np.mean(legal_counts)):.3f} if uniform)")
    print(f"  Mean MCTS entropy:   {np.mean(mcts_entropies):.3f}")
    print(f"  Mean policy max-prob: {np.mean(policy_max):.3f}")
    print(f"  Mean MCTS max-prob:   {np.mean(mcts_max):.3f}")
    return {
        "policy_H": float(np.mean(policy_entropies)),
        "mcts_H": float(np.mean(mcts_entropies)),
        "policy_max": float(np.mean(policy_max)),
        "mcts_max": float(np.mean(mcts_max)),
    }


# ----------------------------------------------------------------------
# Diagnostic 4: Gradient signal
# ----------------------------------------------------------------------

def diagnostic_4_gradients(states: list[BoardWrapper]):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: Gradient-signal check")
    print("=" * 70)

    encoder = StateEncoder(grid_size=17, num_channels=10)
    mapper = ActionMapper(num_pins=10, num_cells=121)

    # Build a synthetic batch from real obs
    B = 32
    obs_list = []
    mask_list = []
    pi_list = []
    val_list = []

    boards = states[:B]
    if len(boards) < B:
        boards = boards + states[: B - len(boards)]
    for board in boards:
        mover = "red"
        other = "blue"
        obs = encoder.encode(board, current_colour=mover, turn_order=[mover, other])
        legal = board.get_legal_moves(mover)
        mask = mapper.build_action_mask(legal)
        # Synthetic policy target: uniform over legal
        pi = mask.astype(np.float32)
        pi = pi / pi.sum()
        # Synthetic value target: random in [-1, 1] but informative
        v = float(np.tanh((board.pins_in_goal("red") - board.pins_in_goal("blue")) / 3.0))
        obs_list.append(obs)
        mask_list.append(mask)
        pi_list.append(pi)
        val_list.append(v)

    obs_b = np.stack(obs_list)
    mask_b = np.stack(mask_list)
    pi_b = np.stack(pi_list)
    val_b = np.array(val_list, dtype=np.float32)

    def grad_norms(value_loss_weight: float):
        # Re-load fresh net each time so prior gradients don't leak
        cfg = NetworkConfig(num_blocks=9, num_filters=96, architecture="resnet")
        net = AlphaZeroNet(config=cfg, device=DEVICE)
        net.load_checkpoint(CHECKPOINT)
        net.model.train()
        optimizer = net.create_optimizer()

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=net.device)
        mask_t = torch.tensor(mask_b, dtype=torch.bool, device=net.device)
        pi_t = torch.tensor(pi_b, dtype=torch.float32, device=net.device)
        v_t = torch.tensor(val_b, dtype=torch.float32, device=net.device)

        # Forward
        logits, values = net.model(obs_t)
        logits_m = logits.masked_fill(~mask_t, -1e9)
        log_probs = torch.nn.functional.log_softmax(logits_m, dim=-1)
        policy_loss = -(pi_t * log_probs).sum(-1).mean()
        value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), v_t)

        # Compute policy-only gradient
        optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        policy_grads = []
        for p in net.model.parameters():
            if p.grad is not None:
                policy_grads.append(p.grad.detach().clone().flatten())
        policy_grad_norm = float(torch.cat(policy_grads).norm().item()) if policy_grads else 0.0

        # Compute value-only gradient (scaled by weight)
        optimizer.zero_grad()
        scaled_value = value_loss_weight * value_loss
        scaled_value.backward()
        value_grads = []
        for p in net.model.parameters():
            if p.grad is not None:
                value_grads.append(p.grad.detach().clone().flatten())
        value_grad_norm = float(torch.cat(value_grads).norm().item()) if value_grads else 0.0

        return policy_grad_norm, value_grad_norm, float(policy_loss.item()), float(value_loss.item())

    print(f"  Batch B={B}\n")
    for vw in [0.25, 1.0]:
        pgn, vgn, pl, vl = grad_norms(vw)
        ratio = vgn / pgn if pgn > 0 else float("nan")
        print(f"  value_loss_weight = {vw}")
        print(f"    policy_loss = {pl:.4f}, value_loss = {vl:.4f}")
        print(f"    grad_norm(policy_only) = {pgn:.4f}")
        print(f"    grad_norm(value_only*vw) = {vgn:.4f}")
        print(f"    value/policy gradient ratio = {ratio:.3f}\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Loading network...")
    net = load_network()
    print(f"Loaded. Param count: {net.parameter_count():,}")

    print("\nGenerating 50 random mid-game states (heuristic vs heuristic)...")
    states = collect_random_states(num_states=50, alpha=0.5)
    print(f"Collected {len(states)} states.")

    r1 = diagnostic_1_mcts_vs_policy(net, states)
    r2 = diagnostic_2_value_head(net, states)
    r3 = diagnostic_3_capacity(net, states)
    diagnostic_4_gradients(states)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY (numbers only)")
    print("=" * 70)
    print("D1 MCTS-vs-policy:")
    for sims, info in r1.items():
        print(f"  sims={sims}: disagreements={info['disagreements']}, "
              f"mcts_win_frac={info['mcts_win_frac']:.3f}, "
              f"mean_score_diff={info['mean_score_diff']:.2f}")
    print(f"D2 value: corr(net,truth)={r2['net_corr_truth']:+.3f}, "
          f"corr(heur,truth)={r2['heur_corr_truth']:+.3f}, "
          f"net_std={r2['net_std']:.3f}")
    print(f"D3 entropy: policy_H={r3['policy_H']:.3f}, mcts_H={r3['mcts_H']:.3f}, "
          f"policy_max={r3['policy_max']:.3f}, mcts_max={r3['mcts_max']:.3f}")
    print("D4: see grad-norm output above")


if __name__ == "__main__":
    main()
