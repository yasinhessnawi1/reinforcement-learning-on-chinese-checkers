"""
warmstart_generator.py — Generate supervised training data from heuristic self-play.

Plays advanced_heuristic vs advanced_heuristic with Dirichlet noise on move
selection to create diverse training data for the warm-start phase.

Records (state, action_mask, move_distribution, game_outcome) triples.

Usage:
    python -m src.training.warmstart_generator --num-games 5000 --output data/warmstart
"""

import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.env.state_encoder import StateEncoder
from src.agents.advanced_heuristic import advanced_heuristic_policy
from src.search.mcts import _score_colour
from src.training.symmetry import ReflectionSymmetry


@dataclass(frozen=True)
class WarmStartConfig:
    """Configuration for warm-start data generation."""
    num_games: int = 5000
    max_moves: int = 200          # per player
    dirichlet_alpha: float = 0.5  # noise on move selection
    noise_fraction: float = 0.25  # fraction of noise mixed in
    min_pins_to_keep: int = 2     # heuristic always scores 2+; filters truly degenerate games
    augment_symmetry: bool = True
    fast_heuristic: bool = True   # skip lookahead (~10x faster gen, still good data)
    output_dir: str = "data/warmstart"


def _noisy_heuristic_policy(board_wrapper, colour, alpha: float = 0.5, noise_frac: float = 0.25, fast: bool = False):
    """Heuristic policy with Dirichlet noise for diversity.

    Scores all legal moves using the advanced heuristic, converts to
    probabilities via softmax, mixes in Dirichlet noise, then samples.

    Parameters
    ----------
    fast : bool
        If True, skip 1-ply lookahead (O(moves) vs O(moves²)). ~10x faster.
        Slightly weaker play but still produces meaningful training data.

    Returns (pin_id, dest, move_distribution) where move_distribution
    is an array of shape (1210,) with probabilities for each action.
    """
    from src.agents.advanced_heuristic import (
        _score_position, _best_lookahead_score, _opponent_hop_potential,
    )

    legal = board_wrapper.get_legal_moves(colour)
    if not legal:
        raise ValueError(f"No legal moves for '{colour}'")

    goal_indices = board_wrapper.get_goal_indices(colour)
    goal_set = set(goal_indices)
    colours = board_wrapper.colours
    opponent = next((c for c in colours if c != colour), None)

    current_score = _score_position(board_wrapper, colour, goal_indices, goal_set)
    opp_hop_before = _opponent_hop_potential(board_wrapper, colour, opponent) if opponent else 0

    mapper = ActionMapper(num_pins=10, num_cells=121)

    # Score all legal moves
    action_scores = {}
    for pin_id, dests in legal.items():
        pin = board_wrapper._pin_by_id(colour, pin_id)
        current_pos = pin.axialindex
        in_goal_now = current_pos in goal_set

        for dest in dests:
            action = mapper.encode(pin_id, dest)

            if in_goal_now and dest not in goal_set:
                action_scores[action] = -1000.0
                continue

            pin.axialindex = dest
            new_score = _score_position(board_wrapper, colour, goal_indices, goal_set)
            immediate = new_score - current_score

            if fast:
                lookahead_gain = 0.0
            else:
                lookahead = _best_lookahead_score(board_wrapper, colour, goal_indices, goal_set)
                lookahead_gain = lookahead - new_score

            move_dist = board_wrapper.axial_distance(current_pos, dest)
            hop_bonus = move_dist * 0.3 if move_dist > 1 else 0.0
            opp_hop_after = _opponent_hop_potential(board_wrapper, colour, opponent) if opponent else 0
            blocking_bonus = (opp_hop_before - opp_hop_after) * 0.5
            pin.axialindex = current_pos

            action_scores[action] = immediate * 1.0 + lookahead_gain * 0.6 + hop_bonus + blocking_bonus

    # Convert to probability distribution
    actions = list(action_scores.keys())
    scores = np.array([action_scores[a] for a in actions], dtype=np.float32)

    # Softmax
    scores -= scores.max()
    exp_scores = np.exp(scores)
    probs = exp_scores / exp_scores.sum()

    # Mix in Dirichlet noise
    noise = np.random.dirichlet([alpha] * len(actions))
    mixed_probs = (1 - noise_frac) * probs + noise_frac * noise
    mixed_probs /= mixed_probs.sum()

    # Build full distribution (1210-dim)
    full_dist = np.zeros(1210, dtype=np.float32)
    for i, a in enumerate(actions):
        full_dist[a] = mixed_probs[i]

    # Sample action
    chosen_idx = np.random.choice(len(actions), p=mixed_probs)
    chosen_action = actions[chosen_idx]
    pin_id, dest = mapper.decode(chosen_action)

    return pin_id, dest, full_dist


def generate_warmstart_data(config: WarmStartConfig = WarmStartConfig()) -> dict:
    """Generate warm-start training data from heuristic self-play.

    Returns dict with:
        obs: np.ndarray (N, C, H, W)
        action_masks: np.ndarray (N, 1210), bool
        policies: np.ndarray (N, 1210)
        values: np.ndarray (N,)
    """
    encoder = StateEncoder(grid_size=17, num_channels=10)
    sym = ReflectionSymmetry() if config.augment_symmetry else None

    all_obs = []
    all_masks = []
    all_policies = []
    all_values = []

    games_played = 0
    games_discarded = 0
    start_time = time.time()

    for game_idx in range(config.num_games):
        # Create env with noisy heuristic as opponent
        fast = config.fast_heuristic

        def opp_policy(bw, colour):
            pid, dest, _ = _noisy_heuristic_policy(
                bw, colour,
                alpha=config.dirichlet_alpha,
                noise_frac=config.noise_fraction,
                fast=fast,
            )
            return pid, dest

        env = ChineseCheckersEnv(opponent_policy=opp_policy, max_steps=config.max_moves)
        obs, info = env.reset()

        game_trajectory = []
        done = False

        while not done:
            action_mask = env.action_masks()

            # Agent uses noisy heuristic
            pin_id, dest, move_dist = _noisy_heuristic_policy(
                env._board, env._AGENT_COLOUR,
                alpha=config.dirichlet_alpha,
                noise_frac=config.noise_fraction,
                fast=fast,
            )

            game_trajectory.append({
                "obs": obs.copy(),
                "action_mask": action_mask.copy(),
                "policy": move_dist.copy(),
            })

            action = env._mapper.encode(pin_id, dest)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Compute game value
        agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board else False
        opp_won = env._board.check_win(env._OPPONENT_COLOUR) if env._board else False

        if agent_won:
            game_value = 1.0
        elif opp_won:
            game_value = -1.0
        else:
            # Truncated: use score differential
            agent_score = _score_colour(env._board, env._AGENT_COLOUR)
            opp_score = _score_colour(env._board, env._OPPONENT_COLOUR)
            game_value = max(-1.0, min(1.0, (agent_score - opp_score) / 1100.0))

        # Check for degenerate games
        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        opp_pins = env._board.pins_in_goal(env._OPPONENT_COLOUR) if env._board else 0

        if agent_pins < config.min_pins_to_keep and opp_pins < config.min_pins_to_keep:
            games_discarded += 1
            continue

        games_played += 1

        # Add trajectory to dataset
        for step_data in game_trajectory:
            all_obs.append(step_data["obs"])
            all_masks.append(step_data["action_mask"])
            all_policies.append(step_data["policy"])
            all_values.append(game_value)

            # Symmetry augmentation
            if sym is not None:
                r_obs = sym.reflect_obs(step_data["obs"])
                r_mask = sym.reflect_action_mask(step_data["action_mask"])
                r_policy = sym.reflect_action_mask(step_data["policy"])
                total = r_policy.sum()
                if total > 0:
                    r_policy = r_policy / total
                    all_obs.append(r_obs)
                    all_masks.append(r_mask)
                    all_policies.append(r_policy)
                    all_values.append(game_value)

        if (game_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (game_idx + 1) / elapsed
            print(
                f"  Game {game_idx + 1}/{config.num_games}: "
                f"{len(all_obs)} samples, "
                f"{games_discarded} discarded, "
                f"{rate:.1f} games/s"
            )

    elapsed = time.time() - start_time
    print(
        f"  Warmstart complete: {games_played} games, "
        f"{len(all_obs)} samples, "
        f"{games_discarded} discarded, "
        f"{elapsed:.1f}s total"
    )

    return {
        "obs": np.stack(all_obs),
        "action_masks": np.stack(all_masks),
        "policies": np.stack(all_policies),
        "values": np.array(all_values, dtype=np.float32),
    }


def save_warmstart_data(data: dict, output_dir: str):
    """Save warm-start data as .npz file."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / "warmstart_data.npz"
    np.savez_compressed(
        str(filepath),
        obs=data["obs"],
        action_masks=data["action_masks"],
        policies=data["policies"],
        values=data["values"],
    )
    print(f"  Saved to {filepath} ({data['obs'].shape[0]} samples)")


def load_warmstart_data(path: str) -> dict:
    """Load warm-start data from .npz file."""
    data = np.load(path)
    return {
        "obs": data["obs"],
        "action_masks": data["action_masks"],
        "policies": data["policies"],
        "values": data["values"],
    }


def pretrain_on_warmstart(
    network,
    data: dict,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    early_stop_patience: int = 5,
    validation_fraction: float = 0.1,
) -> list[dict]:
    """Supervised pre-training on warm-start data.

    Trains with CE(policy) + MSE(value), with early stopping to maintain
    malleability for the RL stage.

    Parameters
    ----------
    network : AlphaZeroNet
    data : dict from generate_warmstart_data
    epochs, batch_size, lr : training hyperparameters
    early_stop_patience : stop if val loss doesn't improve for this many epochs
    validation_fraction : fraction of data held out for validation

    Returns
    -------
    list of dicts with per-epoch training stats.
    """
    import torch

    obs = data["obs"]
    masks = data["action_masks"]
    policies = data["policies"]
    values = data["values"]

    n = len(obs)
    n_val = int(n * validation_fraction)
    n_train = n - n_val

    # Shuffle and split
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    optimizer = torch.optim.Adam(
        network.model.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    training_log = []

    for epoch in range(epochs):
        # Training
        np.random.shuffle(train_idx)
        epoch_loss = {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = train_idx[start:end]

            losses = network.train_step(
                obs[batch_idx],
                masks[batch_idx],
                policies[batch_idx],
                values[batch_idx],
                optimizer,
            )
            for k in epoch_loss:
                epoch_loss[k] += losses[k]
            n_batches += 1

        for k in epoch_loss:
            epoch_loss[k] /= max(1, n_batches)

        scheduler.step()

        # Validation
        val_loss = {"policy_loss": 0, "value_loss": 0, "total_loss": 0}
        n_val_batches = 0
        network.model.eval()

        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                end = min(start + batch_size, n_val)
                batch_idx = val_idx[start:end]

                obs_t = torch.tensor(obs[batch_idx], dtype=torch.float32, device=network.device)
                mask_t = torch.tensor(masks[batch_idx], dtype=torch.bool, device=network.device)
                pi_t = torch.tensor(policies[batch_idx], dtype=torch.float32, device=network.device)
                v_t = torch.tensor(values[batch_idx], dtype=torch.float32, device=network.device)

                logits, vals = network.model(obs_t)
                logits = logits.masked_fill(~mask_t, -1e9)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pl = -torch.sum(pi_t * log_probs, dim=-1).mean()
                vl = torch.nn.functional.mse_loss(vals.squeeze(-1), v_t)

                val_loss["policy_loss"] += pl.item()
                val_loss["value_loss"] += vl.item()
                val_loss["total_loss"] += (pl + vl).item()
                n_val_batches += 1

        for k in val_loss:
            val_loss[k] /= max(1, n_val_batches)

        log_entry = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
        }
        training_log.append(log_entry)

        print(
            f"  Epoch {epoch + 1}/{epochs}: "
            f"train_loss={epoch_loss['total_loss']:.4f} "
            f"(pi={epoch_loss['policy_loss']:.4f}, v={epoch_loss['value_loss']:.4f}), "
            f"val_loss={val_loss['total_loss']:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Early stopping
        if val_loss["total_loss"] < best_val_loss:
            best_val_loss = val_loss["total_loss"]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch + 1} (patience={early_stop_patience})")
                break

    return training_log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate warm-start training data")
    parser.add_argument("--num-games", type=int, default=5000)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/warmstart")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    config = WarmStartConfig(
        num_games=args.num_games,
        max_moves=args.max_moves,
        output_dir=args.output,
        augment_symmetry=not args.no_augment,
    )

    print(f"Generating {config.num_games} warm-start games...")
    data = generate_warmstart_data(config)
    save_warmstart_data(data, config.output_dir)
