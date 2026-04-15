"""
alphazero_self_play.py — Self-play game generation for AlphaZero training.

Generates training data by playing games using MCTS + neural network,
recording (state, action_mask, mcts_visit_distribution, game_outcome).

Supports both standard PUCT MCTS and Gumbel MCTS via the mcts_engine parameter.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import AlphaZeroMCTS, _heuristic_value, _score_colour
from src.training.symmetry import ReflectionSymmetry


@dataclass
class TrainingSample:
    """A single training sample from self-play."""
    obs: np.ndarray           # shape (C, H, W)
    action_mask: np.ndarray   # shape (num_actions,), bool
    policy_target: np.ndarray # shape (num_actions,), MCTS visit distribution
    value_target: float       # game outcome in [-1, 1]


@dataclass(frozen=True)
class SelfPlayConfig:
    """Configuration for self-play game generation."""
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 30      # play with temp=1 for first N moves
    temperature_low: float = 0.1     # then switch to low temperature
    max_moves: int = 200             # max moves per player (400 total steps)
    min_pins_to_keep: int = 0        # discard games where both score < this (0 = keep all)
    use_heuristic_value: bool = True  # use heuristic for leaf eval (essential early training)
    augment_symmetry: bool = True     # double data via reflection
    use_batched_mcts: bool = False    # use BatchedAlphaZeroMCTS for faster GPU inference
    mcts_batch_size: int = 8          # batch size for batched MCTS leaf evaluation
    value_target_lambda: float = 0.6  # blend: lambda*game_outcome + (1-lambda)*mcts_value
    entropy_routing: bool = False     # route search depth by policy entropy (Search MoE)
    entropy_low: float = 0.5         # below this: skip MCTS, use raw policy (0 sims)
    entropy_high: float = 2.0        # above this: use 3x sims (deep search)
    deep_sims_multiplier: int = 3    # multiplier for high-entropy positions


def _compute_game_value(env: ChineseCheckersEnv, agent_won: bool, opponent_won: bool) -> float:
    """Compute the value target for a completed game.

    For terminated games: +1 (win), -1 (loss).
    For truncated games: normalized tournament score differential.
    """
    if agent_won:
        return 1.0
    if opponent_won:
        return -1.0

    # Truncated: use tournament score differential
    board = env._board
    agent_score = _score_colour(board, env._AGENT_COLOUR)
    opp_score = _score_colour(board, env._OPPONENT_COLOUR)
    relative = agent_score - opp_score
    # Normalize to [-1, 1], with small penalty for not finishing
    normalized = relative / 1100.0
    # Penalize truncation slightly: even a winning position is worth less if not finished
    truncation_penalty = 0.1 * (env._step_count / env.max_steps)
    value = max(-1.0, min(1.0, normalized)) - truncation_penalty
    return max(-1.0, min(1.0, value))


def play_one_game(
    network,
    config: SelfPlayConfig = SelfPlayConfig(),
    opponent_policy=None,
    mcts_engine=None,
) -> list[TrainingSample]:
    """Play one self-play game and collect training samples.

    Parameters
    ----------
    network : AlphaZeroNet
        Neural network for MCTS priors and value.
    config : SelfPlayConfig
        Game generation configuration.
    opponent_policy : callable or None
        If None, self-play (agent plays both sides via MCTS).
        If provided, agent plays red, opponent plays blue.
    mcts_engine : AlphaZeroMCTS or GumbelMCTS or None
        Pre-configured MCTS engine. If None, creates AlphaZeroMCTS from config.

    Returns
    -------
    list[TrainingSample] — training samples from this game (may be empty if degenerate).
    """
    # Create environment
    env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
    obs, info = env.reset()

    # Create MCTS engine if not provided
    if mcts_engine is None:
        mcts_engine = AlphaZeroMCTS(
            network=network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            use_heuristic_value=config.use_heuristic_value,
            opponent_policy=opponent_policy,
        )

    # Collect game trajectory
    trajectory: list[dict] = []
    recent_states: list[int] = []  # hashes of recent board states for repetition detection
    move_count = 0

    done = False
    while not done:
        action_mask = env.action_masks()

        # Choose temperature based on move count
        temperature = 1.0 if move_count < config.temperature_moves else config.temperature_low

        # Run MCTS to get visit distribution
        action_probs = mcts_engine.get_action_probs(env, temperature=temperature)

        # Record sample (value filled in later with game outcome)
        trajectory.append({
            "obs": obs.copy(),
            "action_mask": action_mask.copy(),
            "policy_target": action_probs.copy(),
        })

        # Sample action from MCTS distribution
        if temperature < 1e-6:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))

        # Track board state for repetition detection
        state_hash = _board_hash(env)
        recent_states.append(state_hash)
        if len(recent_states) > 8:
            recent_states.pop(0)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        move_count += 1

    # Compute game outcome value
    agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board is not None else False
    opponent_won = (
        env._board.check_win(env._OPPONENT_COLOUR)
        if env._board is not None and not env._no_opponent
        else False
    )
    game_value = _compute_game_value(env, agent_won, opponent_won)

    # Check for degenerate games
    agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
    opp_pins = 0
    if env._board and not env._no_opponent:
        opp_pins = env._board.pins_in_goal(env._OPPONENT_COLOUR)

    if agent_pins < config.min_pins_to_keep and opp_pins < config.min_pins_to_keep:
        return []  # Degenerate game — discard

    # Build training samples with game outcome as value target
    samples = []
    for step_data in trajectory:
        samples.append(TrainingSample(
            obs=step_data["obs"],
            action_mask=step_data["action_mask"],
            policy_target=step_data["policy_target"],
            value_target=game_value,
        ))

    # Symmetry augmentation: double the data
    if config.augment_symmetry:
        sym = ReflectionSymmetry()
        augmented = []
        for sample in samples:
            reflected_obs = sym.reflect_obs(sample.obs)
            reflected_mask = sym.reflect_action_mask(sample.action_mask)
            reflected_policy = sym.reflect_action_mask(sample.policy_target)

            # Renormalize reflected policy (some actions may not have valid reflections)
            total = reflected_policy.sum()
            if total > 0:
                reflected_policy = reflected_policy / total
                augmented.append(TrainingSample(
                    obs=reflected_obs,
                    action_mask=reflected_mask,
                    policy_target=reflected_policy,
                    value_target=sample.value_target,
                ))
        samples.extend(augmented)

    return samples


def _board_hash(env: ChineseCheckersEnv) -> int:
    """Quick hash of current board state for repetition detection."""
    board = env._board
    positions = []
    for colour in board.colours:
        for pin in board.pins[colour]:
            positions.append(pin.axialindex)
    return hash(tuple(positions))


def generate_self_play_data(
    network,
    num_games: int = 100,
    config: SelfPlayConfig = SelfPlayConfig(),
    opponent_policy=None,
    mcts_engine=None,
    verbose: bool = True,
) -> list[TrainingSample]:
    """Generate training data from multiple self-play games.

    Parameters
    ----------
    network : AlphaZeroNet
    num_games : int
    config : SelfPlayConfig
    opponent_policy : callable or None
    mcts_engine : AlphaZeroMCTS or GumbelMCTS or None
    verbose : bool

    Returns
    -------
    list[TrainingSample] — all samples from all games.
    """
    all_samples: list[TrainingSample] = []
    games_played = 0
    games_discarded = 0

    for i in range(num_games):
        samples = play_one_game(
            network=network,
            config=config,
            opponent_policy=opponent_policy,
            mcts_engine=mcts_engine,
        )

        if samples:
            all_samples.extend(samples)
            games_played += 1
        else:
            games_discarded += 1

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  Self-play: {i + 1}/{num_games} games, "
                f"{len(all_samples)} samples, "
                f"{games_discarded} discarded"
            )

    if verbose:
        print(
            f"  Self-play complete: {games_played} games, "
            f"{len(all_samples)} samples, "
            f"{games_discarded} discarded"
        )

    return all_samples
