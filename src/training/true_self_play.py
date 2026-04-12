"""
true_self_play.py — Proper 2-player self-play for AlphaZero training.

Unlike the original alphazero_self_play.py which used env.step() (agent vs
random/fixed opponent), this module plays BOTH sides with MCTS + the current
network, recording training samples from both perspectives.

Key differences from the original:
  1. Both red and blue use MCTS + network to select moves.
  2. Both sides' observations and MCTS policies are recorded.
  3. Value targets are perspective-correct: +1 for winner, -1 for loser.
  4. No env.step() — we apply moves directly on the board to avoid
     the single-agent assumption baked into ChineseCheckersEnv.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder
from src.env.action_mapper import ActionMapper
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import AlphaZeroMCTS, _heuristic_value, _score_colour
from src.training.symmetry import ReflectionSymmetry
from src.training.alphazero_self_play import TrainingSample, SelfPlayConfig


# Shared stateless helpers — created once, reused across games.
_ENCODER = StateEncoder(grid_size=17, num_channels=10)
_MAPPER = ActionMapper(num_pins=10, num_cells=121)
_COLOURS = ["red", "blue"]
_TURN_ORDER = ["red", "blue"]


def _make_proxy_env(board: BoardWrapper, colour: str, step_count: int, max_steps: int) -> ChineseCheckersEnv:
    """Create a lightweight proxy env for MCTS, positioned for `colour` as the agent.

    MCTS reads env._AGENT_COLOUR, env._board, env._mapper, etc.  We set
    _AGENT_COLOUR to `colour` so the search tree evaluates from the correct
    perspective.  The proxy has no opponent (solo mode) so env.step() would
    only move the agent colour — which is exactly what MCTS needs.
    """
    proxy = ChineseCheckersEnv.__new__(ChineseCheckersEnv)
    proxy.render_mode = None
    proxy.max_steps = max_steps
    proxy.observation_space = None  # unused by MCTS
    proxy.action_space = type('Space', (), {'n': 1210})()  # MCTS reads .n
    proxy._encoder = _ENCODER
    proxy._mapper = _MAPPER
    proxy._AGENT_COLOUR = colour
    proxy._OPPONENT_COLOUR = "blue" if colour == "red" else "red"
    proxy._TURN_ORDER = _TURN_ORDER
    proxy._no_opponent = True
    proxy._opponent_policy = None
    proxy._board = board.clone()
    proxy._step_count = step_count
    proxy._terminated = False
    proxy._truncated = False
    return proxy


def _encode_obs(board: BoardWrapper, colour: str) -> np.ndarray:
    """Encode the board from `colour`'s perspective."""
    return _ENCODER.encode(board, current_colour=colour, turn_order=_TURN_ORDER)


def _get_action_mask(board: BoardWrapper, colour: str) -> np.ndarray:
    """Build action mask for `colour`."""
    legal_moves = board.get_legal_moves(colour)
    return _MAPPER.build_action_mask(legal_moves)


def _compute_game_values(board: BoardWrapper, red_won: bool, blue_won: bool,
                          step_count: int, max_steps: int) -> dict[str, float]:
    """Compute value targets for both players.

    Returns dict: {'red': float, 'blue': float} in [-1, 1].
    """
    if red_won:
        return {"red": 1.0, "blue": -1.0}
    if blue_won:
        return {"red": -1.0, "blue": 1.0}

    # Truncated: use normalized score differential
    red_score = _score_colour(board, "red")
    blue_score = _score_colour(board, "blue")
    diff = (red_score - blue_score) / max(red_score + blue_score, 1.0)
    # Small truncation penalty
    trunc_pen = 0.1 * (step_count / max(max_steps, 1))
    red_val = max(-1.0, min(1.0, diff - trunc_pen))
    blue_val = max(-1.0, min(1.0, -diff - trunc_pen))
    return {"red": red_val, "blue": blue_val}


def play_one_game_true_selfplay(
    network,
    config: SelfPlayConfig = SelfPlayConfig(),
    mcts_engine_factory=None,
) -> list[TrainingSample]:
    """Play one game where BOTH sides use MCTS + network, recording both perspectives.

    Parameters
    ----------
    network : AlphaZeroNet
        Neural network for MCTS priors and value.
    config : SelfPlayConfig
        Game generation configuration.
    mcts_engine_factory : callable(colour) -> MCTS engine, or None
        Factory to create per-colour MCTS engines. If None, creates
        AlphaZeroMCTS instances from config.

    Returns
    -------
    list[TrainingSample] — training samples from both sides.
    """
    # Create a fresh board with both colours
    board = BoardWrapper(["red", "blue"])
    max_total_steps = config.max_moves * 2  # total half-moves (both sides)

    # Create MCTS engines for each colour
    if mcts_engine_factory is not None:
        engines = {c: mcts_engine_factory(c) for c in _COLOURS}
    else:
        engines = {}
        for colour in _COLOURS:
            engines[colour] = AlphaZeroMCTS(
                network=network,
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=config.dirichlet_epsilon,
                use_heuristic_value=config.use_heuristic_value,
            )

    # Trajectories: list of (colour, obs, action_mask, mcts_policy) per half-move
    trajectories: list[dict] = []
    step_count = 0  # total half-moves
    move_counts = {"red": 0, "blue": 0}  # per-colour moves for temperature schedule
    winner = None

    while step_count < max_total_steps:
        # Alternate: red on even steps, blue on odd
        colour = _COLOURS[step_count % 2]

        # Check if this colour has legal moves
        legal_moves = board.get_legal_moves(colour)
        if not legal_moves:
            break  # draw / stalemate

        # Encode observation from this colour's perspective
        obs = _encode_obs(board, colour)
        action_mask = _get_action_mask(board, colour)

        # Temperature schedule based on this colour's move count
        temp = 1.0 if move_counts[colour] < config.temperature_moves else config.temperature_low

        # Create proxy env for MCTS (MCTS reads env._board, env._AGENT_COLOUR, etc.)
        proxy = _make_proxy_env(board, colour, step_count, max_total_steps)

        # Run MCTS
        action_probs = engines[colour].get_action_probs(proxy, temperature=temp)

        # Record sample (value filled in after game ends)
        trajectories.append({
            "colour": colour,
            "obs": obs.copy(),
            "action_mask": action_mask.copy(),
            "policy_target": action_probs.copy(),
        })

        # Sample action
        if temp < 1e-6:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))

        # Apply move on the shared board
        pin_id, dest = _MAPPER.decode(action)
        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1

        # Check for win
        if board.check_win(colour):
            winner = colour
            break

    # Compute value targets for both sides
    red_won = (winner == "red")
    blue_won = (winner == "blue")
    values = _compute_game_values(board, red_won, blue_won, step_count, max_total_steps)

    # Check for degenerate games
    red_pins = board.pins_in_goal("red")
    blue_pins = board.pins_in_goal("blue")
    if red_pins < config.min_pins_to_keep and blue_pins < config.min_pins_to_keep:
        return []

    # Build training samples with perspective-correct value targets
    samples = []
    for step_data in trajectories:
        colour = step_data["colour"]
        samples.append(TrainingSample(
            obs=step_data["obs"],
            action_mask=step_data["action_mask"],
            policy_target=step_data["policy_target"],
            value_target=values[colour],
        ))

    # Symmetry augmentation
    if config.augment_symmetry:
        sym = ReflectionSymmetry()
        augmented = []
        for sample in samples:
            reflected_obs = sym.reflect_obs(sample.obs)
            reflected_mask = sym.reflect_action_mask(sample.action_mask)
            reflected_policy = sym.reflect_action_mask(sample.policy_target)

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


def generate_true_self_play_data(
    network,
    num_games: int = 100,
    config: SelfPlayConfig = SelfPlayConfig(),
    verbose: bool = True,
) -> list[TrainingSample]:
    """Generate training data from multiple true self-play games.

    Parameters
    ----------
    network : AlphaZeroNet
    num_games : int
    config : SelfPlayConfig
    verbose : bool

    Returns
    -------
    list[TrainingSample] — all samples from all games.
    """
    all_samples: list[TrainingSample] = []
    games_played = 0
    games_discarded = 0
    total_wins = {"red": 0, "blue": 0, "truncated": 0}

    for i in range(num_games):
        samples = play_one_game_true_selfplay(
            network=network,
            config=config,
        )

        if samples:
            # Track outcomes from value targets
            if samples[0].value_target > 0.5:
                # First sample is red's — if positive, red won
                total_wins["red"] += 1
            elif samples[0].value_target < -0.5:
                total_wins["blue"] += 1
            else:
                total_wins["truncated"] += 1

            all_samples.extend(samples)
            games_played += 1
        else:
            games_discarded += 1

        if verbose and (i + 1) % 10 == 0:
            print(
                f"  Self-play: {i + 1}/{num_games} games, "
                f"{len(all_samples)} samples, "
                f"{games_discarded} discarded, "
                f"wins R={total_wins['red']}/B={total_wins['blue']}/T={total_wins['truncated']}"
            )

    if verbose:
        print(
            f"  Self-play complete: {games_played} games, "
            f"{len(all_samples)} samples, "
            f"{games_discarded} discarded, "
            f"wins R={total_wins['red']}/B={total_wins['blue']}/T={total_wins['truncated']}"
        )

    return all_samples
