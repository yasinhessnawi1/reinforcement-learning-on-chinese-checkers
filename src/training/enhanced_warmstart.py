"""
enhanced_warmstart.py — MCTS-enhanced warm-start data generation.

Instead of just using the raw heuristic for move selection, this wraps
the heuristic in MCTS search to produce near-optimal move distributions.

Think of it as: the raw heuristic is a "textbook" — it gives reasonable moves.
The MCTS-enhanced version is "grandmaster annotated games" — it gives moves
that are validated by search.

Also includes endgame-focused data generation: starts from positions where
5-7 pins are already in goal, training the model to finish games (the
hardest and most tournament-critical phase).

Usage:
    python -m src.training.enhanced_warmstart --mode mcts --num-games 1000
    python -m src.training.enhanced_warmstart --mode endgame --num-games 2000
"""

import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.env.state_encoder import StateEncoder
from src.agents.advanced_heuristic import advanced_heuristic_policy
from src.search.mcts import _score_colour, _heuristic_value, MCTS, MCTSNode, MinMaxStats, _get_heuristic_priors
from src.training.symmetry import ReflectionSymmetry


@dataclass(frozen=True)
class EnhancedWarmStartConfig:
    """Configuration for MCTS-enhanced warm-start data generation."""
    num_games: int = 1000
    max_moves: int = 200
    # MCTS settings for move evaluation
    mcts_simulations: int = 50       # sims per move for policy targets
    c_puct: float = 1.5
    temperature: float = 1.0         # >0 for diverse training data
    # Endgame settings
    endgame_mode: bool = False       # start from mid/endgame positions
    endgame_min_pins: int = 5        # min pins in goal for endgame start
    endgame_max_pins: int = 8        # max pins in goal for endgame start
    # Data quality
    min_pins_to_keep: int = 2
    augment_symmetry: bool = True
    output_dir: str = "data/enhanced_warmstart"


def _heuristic_mcts_policy(env, num_simulations: int = 50, c_puct: float = 1.5,
                            temperature: float = 1.0) -> tuple[int, np.ndarray]:
    """Use MCTS with heuristic priors to generate move + policy distribution.

    Unlike raw heuristic (which picks one move), this runs MCTS to explore
    multiple lines and produces a search-validated visit distribution.

    Returns (action, policy_distribution) where policy_distribution is
    shape (1210,) with probabilities for each action.
    """
    # Create a heuristic-priors MCTS engine
    mcts = MCTS(
        model=None,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        use_network_value=False,  # use heuristic value
        use_heuristic_priors=True,  # use heuristic for priors too
    )

    root = mcts.run(env)
    num_actions = env.action_space.n

    visits = np.zeros(num_actions, dtype=np.float32)
    for action, child in root.children.items():
        visits[action] = child.N

    if visits.sum() == 0:
        mask = env.action_masks()
        visits = mask.astype(np.float32)

    # Temperature-based action distribution
    if temperature < 1e-6:
        probs = np.zeros(num_actions, dtype=np.float32)
        probs[np.argmax(visits)] = 1.0
    else:
        counts_temp = visits ** (1.0 / temperature)
        total = counts_temp.sum()
        probs = counts_temp / total if total > 0 else counts_temp

    # Sample action
    if temperature < 1e-6:
        action = int(np.argmax(probs))
    else:
        action = int(np.random.choice(len(probs), p=probs))

    return action, probs


def _advance_to_endgame(env, target_pins: int) -> bool:
    """Play heuristic moves until agent has `target_pins` pins in goal.

    Returns True if successfully reached target, False if game ended first.
    """
    for _ in range(400):  # max moves to reach endgame position
        if env._terminated or env._truncated:
            return False

        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        if agent_pins >= target_pins:
            return True

        # Use heuristic to advance
        action_mask = env.action_masks()
        if action_mask.sum() == 0:
            return False

        try:
            pin_id, dest = advanced_heuristic_policy(env._board, env._AGENT_COLOUR)
            action = env._mapper.encode(pin_id, dest)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return False
        except (ValueError, KeyError):
            return False

    return False


def generate_mcts_warmstart_data(
    config: EnhancedWarmStartConfig = EnhancedWarmStartConfig(),
) -> dict:
    """Generate warm-start data using MCTS with heuristic priors.

    Each move is evaluated by MCTS search (not just the raw heuristic),
    producing higher-quality policy targets.

    Returns dict with obs, action_masks, policies, values arrays.
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
        # Opponent uses raw heuristic (fast)
        env = ChineseCheckersEnv(
            opponent_policy=advanced_heuristic_policy,
            max_steps=config.max_moves,
        )
        obs, info = env.reset()

        # If endgame mode, advance to endgame position first
        if config.endgame_mode:
            target_pins = np.random.randint(
                config.endgame_min_pins, config.endgame_max_pins + 1
            )
            reached = _advance_to_endgame(env, target_pins)
            if not reached:
                games_discarded += 1
                continue
            obs = env._get_obs()

        game_trajectory = []
        done = False

        while not done:
            action_mask = env.action_masks()
            if action_mask.sum() == 0:
                break

            # MCTS-enhanced move selection
            action, move_dist = _heuristic_mcts_policy(
                env,
                num_simulations=config.mcts_simulations,
                c_puct=config.c_puct,
                temperature=config.temperature,
            )

            game_trajectory.append({
                "obs": obs.copy(),
                "action_mask": action_mask.copy(),
                "policy": move_dist.copy(),
            })

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
            agent_score = _score_colour(env._board, env._AGENT_COLOUR)
            opp_score = _score_colour(env._board, env._OPPONENT_COLOUR)
            game_value = max(-1.0, min(1.0, (agent_score - opp_score) / 1100.0))

        # Filter degenerate games
        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        opp_pins = env._board.pins_in_goal(env._OPPONENT_COLOUR) if env._board else 0

        if agent_pins < config.min_pins_to_keep and opp_pins < config.min_pins_to_keep:
            games_discarded += 1
            continue

        games_played += 1

        for step_data in game_trajectory:
            all_obs.append(step_data["obs"])
            all_masks.append(step_data["action_mask"])
            all_policies.append(step_data["policy"])
            all_values.append(game_value)

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

        if (game_idx + 1) % 10 == 0:
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
        f"  MCTS warmstart complete: {games_played} games, "
        f"{len(all_obs)} samples, "
        f"{games_discarded} discarded, "
        f"{elapsed:.1f}s total"
    )

    if not all_obs:
        return {
            "obs": np.zeros((0, 10, 17, 17), dtype=np.float32),
            "action_masks": np.zeros((0, 1210), dtype=np.bool_),
            "policies": np.zeros((0, 1210), dtype=np.float32),
            "values": np.zeros((0,), dtype=np.float32),
        }

    return {
        "obs": np.stack(all_obs),
        "action_masks": np.stack(all_masks),
        "policies": np.stack(all_policies),
        "values": np.array(all_values, dtype=np.float32),
    }


def generate_endgame_data(
    config: EnhancedWarmStartConfig = EnhancedWarmStartConfig(),
) -> dict:
    """Generate training data focused on endgame positions (5-8 pins in goal).

    This is where most games stall and where tournament points are won.
    The model needs to learn precise endgame play — which pins to move
    and in what order to achieve 10/10.

    Uses MCTS with heuristic priors for high-quality move distributions.
    """
    endgame_config = EnhancedWarmStartConfig(
        num_games=config.num_games,
        max_moves=100,  # shorter since we start mid-game
        mcts_simulations=config.mcts_simulations,
        c_puct=config.c_puct,
        temperature=config.temperature,
        endgame_mode=True,
        endgame_min_pins=config.endgame_min_pins,
        endgame_max_pins=config.endgame_max_pins,
        min_pins_to_keep=0,  # keep all endgame data
        augment_symmetry=config.augment_symmetry,
        output_dir=config.output_dir,
    )
    return generate_mcts_warmstart_data(endgame_config)


def save_enhanced_data(data: dict, output_dir: str, prefix: str = "enhanced") -> None:
    """Save enhanced warm-start data as .npz file."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{prefix}_warmstart_data.npz"
    np.savez_compressed(
        str(filepath),
        obs=data["obs"],
        action_masks=data["action_masks"],
        policies=data["policies"],
        values=data["values"],
    )
    print(f"  Saved to {filepath} ({data['obs'].shape[0]} samples)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced warm-start data generation")
    parser.add_argument("--mode", choices=["mcts", "endgame", "both"], default="mcts")
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="data/enhanced_warmstart")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    config = EnhancedWarmStartConfig(
        num_games=args.num_games,
        mcts_simulations=args.mcts_sims,
        temperature=args.temperature,
        augment_symmetry=not args.no_augment,
        output_dir=args.output,
    )

    if args.mode in ("mcts", "both"):
        print(f"Generating {config.num_games} MCTS-enhanced warm-start games...")
        data = generate_mcts_warmstart_data(config)
        save_enhanced_data(data, config.output_dir, prefix="mcts")

    if args.mode in ("endgame", "both"):
        print(f"\nGenerating {config.num_games} endgame-focused games...")
        endgame_data = generate_endgame_data(config)
        save_enhanced_data(endgame_data, config.output_dir, prefix="endgame")
