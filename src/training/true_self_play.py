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

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder
from src.env.action_mapper import ActionMapper
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import AlphaZeroMCTS, _heuristic_value, _score_colour
from src.search.batched_mcts import BatchedAlphaZeroMCTS
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


def _create_mcts_engine(network, config: SelfPlayConfig):
    """Create MCTS engine based on config (standard or batched)."""
    if config.use_batched_mcts:
        return BatchedAlphaZeroMCTS(
            network=network,
            num_simulations=config.num_simulations,
            batch_size=config.mcts_batch_size,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            use_heuristic_value=config.use_heuristic_value,
        )
    return AlphaZeroMCTS(
        network=network,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        use_heuristic_value=config.use_heuristic_value,
    )


def _compute_policy_entropy(network, obs: np.ndarray, action_mask: np.ndarray) -> float:
    """Compute entropy of the raw policy distribution (no MCTS).

    Used by Search MoE to route search depth by position difficulty.
    Low entropy = confident position (skip MCTS), high = confused (deep search).
    """
    probs, _ = network.predict(obs, action_mask)
    # Entropy: -sum(p * log(p)) for p > 0
    valid = probs > 1e-8
    entropy = -np.sum(probs[valid] * np.log(probs[valid]))
    return float(entropy)


def _raw_policy_action(network, obs: np.ndarray, action_mask: np.ndarray,
                       temperature: float) -> tuple[np.ndarray, float]:
    """Get action probs from raw network policy (no MCTS). Returns (probs, value)."""
    probs, value = network.predict(obs, action_mask)
    return probs, float(value)


def _encode_obs(board: BoardWrapper, colour: str) -> np.ndarray:
    """Encode the board from `colour`'s perspective."""
    return _ENCODER.encode(board, current_colour=colour, turn_order=_TURN_ORDER)


def _get_action_mask(board: BoardWrapper, colour: str) -> np.ndarray:
    """Build action mask for `colour`."""
    legal_moves = board.get_legal_moves(colour)
    return _MAPPER.build_action_mask(legal_moves)


def _tournament_score(board: BoardWrapper, colour: str, move_count: int,
                      time_sec: float = 30.0) -> float:
    """Compute tournament score for a colour using the official formula.

    Score = pin_score + distance_score + time_score + move_score

    Components (from game.py):
      pin_score:      pins_in_goal * 100.0  (max 1000)
      distance_score: max(0, 200 - total_dist)  (max 200)
      time_score:     max(0, 100 - time_sec)  (max 100)
      move_score:     Gaussian(move_count, center=45, sigma=4/18)  (max 1.0)
    """
    pins_in_goal = board.pins_in_goal(colour)
    pin_score = pins_in_goal * 100.0

    goal_indices = board.get_goal_indices(colour)
    goal_set = set(goal_indices)
    total_dist = 0
    for pin in board.pins[colour]:
        if pin.axialindex not in goal_set:
            min_d = min(board.axial_distance(pin.axialindex, g) for g in goal_indices)
            total_dist += min_d
    distance_score = max(0.0, 200.0 - total_dist)

    time_score = max(0.0, 100.0 - time_sec)

    sigma = 4 if move_count < 45 else 18
    move_score = math.exp(-((move_count - 45) ** 2) / (2 * sigma ** 2)) if move_count > 0 else 0.0

    return pin_score + distance_score + time_score + move_score


def _compute_game_values(board: BoardWrapper, red_won: bool, blue_won: bool,
                          step_count: int, max_steps: int,
                          move_counts: dict[str, int] | None = None) -> dict[str, float]:
    """Compute value targets for both players using tournament-aligned scoring.

    Uses the official tournament formula to compute scores for both players,
    then normalizes the differential to [-1, 1].

    Terminal wins still get +1/-1 to provide clear signal.
    Truncated games use normalized tournament score differential.

    Returns dict: {'red': float, 'blue': float} in [-1, 1].
    """
    if red_won:
        return {"red": 1.0, "blue": -1.0}
    if blue_won:
        return {"red": -1.0, "blue": 1.0}

    # Use tournament scoring for truncated games
    red_moves = move_counts["red"] if move_counts else step_count // 2
    blue_moves = move_counts["blue"] if move_counts else step_count // 2

    red_score = _tournament_score(board, "red", red_moves)
    blue_score = _tournament_score(board, "blue", blue_moves)

    # Normalize differential to [-1, 1]
    # Max possible score is ~1301 (1000 + 200 + 100 + 1), min is 0
    max_score = 1301.0
    diff = (red_score - blue_score) / max_score

    # Stronger truncation penalty: failing to win is bad, not neutral.
    # Base penalty scales with how far through the game we are.
    # Additional "draw penalty" when scores are close (both failed to win).
    trunc_pen = 0.15 * (step_count / max(max_steps, 1))

    # If neither side has >7 pins in goal, both are doing poorly — penalize harder
    red_pins = board.pins_in_goal("red")
    blue_pins = board.pins_in_goal("blue")
    if red_pins < 8 and blue_pins < 8:
        # Scale penalty by how far from winning (10 pins) both sides are
        avg_missing = (10 - red_pins + 10 - blue_pins) / 2.0
        draw_pen = 0.05 * (avg_missing / 10.0)  # up to 0.05 extra penalty
        trunc_pen += draw_pen

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
        engines = {c: _create_mcts_engine(network, config) for c in _COLOURS}

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

        # Search MoE: route search depth by policy entropy
        if config.entropy_routing:
            entropy = _compute_policy_entropy(network, obs, action_mask)
            if entropy < config.entropy_low:
                # Confident position — use raw policy, skip MCTS
                action_probs, mcts_value = _raw_policy_action(network, obs, action_mask, temp)
            elif entropy > config.entropy_high:
                # Confused position — deep search (3x sims)
                proxy = _make_proxy_env(board, colour, step_count, max_total_steps)
                deep_engine = _create_mcts_engine(network, config)
                deep_engine.num_simulations = config.num_simulations * config.deep_sims_multiplier
                action_probs, mcts_value = deep_engine.get_action_probs_and_value(proxy, temperature=temp)
            else:
                # Normal position — standard search
                proxy = _make_proxy_env(board, colour, step_count, max_total_steps)
                action_probs, mcts_value = engines[colour].get_action_probs_and_value(proxy, temperature=temp)
        else:
            # No routing — always use standard engine
            proxy = _make_proxy_env(board, colour, step_count, max_total_steps)
            action_probs, mcts_value = engines[colour].get_action_probs_and_value(proxy, temperature=temp)

        # Record sample (game outcome value filled in after game ends)
        trajectories.append({
            "colour": colour,
            "obs": obs.copy(),
            "action_mask": action_mask.copy(),
            "policy_target": action_probs.copy(),
            "mcts_value": mcts_value,
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
    values = _compute_game_values(board, red_won, blue_won, step_count, max_total_steps,
                                   move_counts=move_counts)

    # Check for degenerate games
    red_pins = board.pins_in_goal("red")
    blue_pins = board.pins_in_goal("blue")
    if red_pins < config.min_pins_to_keep and blue_pins < config.min_pins_to_keep:
        return []

    # Build training samples with blended value targets.
    # Blend MCTS per-step value with game outcome to give the value head
    # useful per-position signal (especially in truncated games where
    # game outcome ≈ 0 for all positions).
    lam = config.value_target_lambda
    samples = []
    for step_data in trajectories:
        colour = step_data["colour"]
        game_val = values[colour]
        mcts_val = step_data["mcts_value"]
        blended = lam * game_val + (1.0 - lam) * mcts_val
        blended = max(-1.0, min(1.0, blended))
        samples.append(TrainingSample(
            obs=step_data["obs"],
            action_mask=step_data["action_mask"],
            policy_target=step_data["policy_target"],
            value_target=blended,
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


def play_one_game_vs_heuristic(
    network,
    opponent_policy_fn,
    config: SelfPlayConfig = SelfPlayConfig(),
) -> list[TrainingSample]:
    """Play one game: network+MCTS (red) vs a heuristic opponent (blue).

    Only the agent's (red's) trajectory is recorded for training.  The
    heuristic opponent provides a strong learning signal without needing
    MCTS on the blue side.

    Parameters
    ----------
    network : AlphaZeroNet
    opponent_policy_fn : callable(board_wrapper, colour) -> (pin_id, dest)
        Heuristic policy for the opponent (blue).
    config : SelfPlayConfig

    Returns
    -------
    list[TrainingSample] — training samples from the agent's perspective.
    """
    board = BoardWrapper(["red", "blue"])
    max_total_steps = config.max_moves * 2

    engine = _create_mcts_engine(network, config)

    trajectories: list[dict] = []
    step_count = 0
    agent_moves = 0
    opp_moves = 0
    winner = None

    while step_count < max_total_steps:
        colour = _COLOURS[step_count % 2]

        legal_moves = board.get_legal_moves(colour)
        if not legal_moves:
            break

        if colour == "red":
            # Agent's turn — use MCTS + network
            obs = _encode_obs(board, "red")
            action_mask = _get_action_mask(board, "red")
            temp = 1.0 if agent_moves < config.temperature_moves else config.temperature_low

            # Search MoE: route by entropy
            if config.entropy_routing:
                entropy = _compute_policy_entropy(network, obs, action_mask)
                if entropy < config.entropy_low:
                    action_probs, mcts_value = _raw_policy_action(network, obs, action_mask, temp)
                elif entropy > config.entropy_high:
                    proxy = _make_proxy_env(board, "red", step_count, max_total_steps)
                    deep_engine = _create_mcts_engine(network, config)
                    deep_engine.num_simulations = config.num_simulations * config.deep_sims_multiplier
                    action_probs, mcts_value = deep_engine.get_action_probs_and_value(proxy, temperature=temp)
                else:
                    proxy = _make_proxy_env(board, "red", step_count, max_total_steps)
                    action_probs, mcts_value = engine.get_action_probs_and_value(proxy, temperature=temp)
            else:
                proxy = _make_proxy_env(board, "red", step_count, max_total_steps)
                action_probs, mcts_value = engine.get_action_probs_and_value(proxy, temperature=temp)

            trajectories.append({
                "obs": obs.copy(),
                "action_mask": action_mask.copy(),
                "policy_target": action_probs.copy(),
                "mcts_value": mcts_value,
            })

            if temp < 1e-6:
                action = int(np.argmax(action_probs))
            else:
                action = int(np.random.choice(len(action_probs), p=action_probs))

            pin_id, dest = _MAPPER.decode(action)
            board.apply_move("red", pin_id, dest)
            agent_moves += 1
        else:
            # Opponent's turn — use heuristic policy directly
            opp_pin_id, opp_dest = opponent_policy_fn(board, "blue")
            board.apply_move("blue", opp_pin_id, opp_dest)
            opp_moves += 1

        step_count += 1

        if board.check_win(colour):
            winner = colour
            break

    # Compute value target for the agent (red)
    red_won = (winner == "red")
    blue_won = (winner == "blue")
    mc = {"red": agent_moves, "blue": opp_moves}
    values = _compute_game_values(board, red_won, blue_won, step_count, max_total_steps,
                                   move_counts=mc)
    agent_value = values["red"]

    # Check for degenerate games
    if board.pins_in_goal("red") < config.min_pins_to_keep:
        return []

    # Blend MCTS per-step value with game outcome
    lam = config.value_target_lambda
    samples = []
    for step_data in trajectories:
        mcts_val = step_data["mcts_value"]
        blended = lam * agent_value + (1.0 - lam) * mcts_val
        blended = max(-1.0, min(1.0, blended))
        samples.append(TrainingSample(
            obs=step_data["obs"],
            action_mask=step_data["action_mask"],
            policy_target=step_data["policy_target"],
            value_target=blended,
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


def generate_curriculum_data(
    network,
    num_games: int,
    opponent_mix: dict[str, float],
    config: SelfPlayConfig = SelfPlayConfig(),
    verbose: bool = True,
) -> list[TrainingSample]:
    """Generate training data using a curriculum of mixed opponents.

    Parameters
    ----------
    network : AlphaZeroNet
    num_games : int
        Total games to generate across all opponent types.
    opponent_mix : dict[str, float]
        Mapping of opponent type to fraction of games. Keys:
        - 'self_play': true 2-player self-play (both sides MCTS+network)
        - 'greedy': greedy heuristic opponent
        - 'advanced': advanced heuristic opponent
        Fractions should sum to ~1.0.
    config : SelfPlayConfig
    verbose : bool

    Returns
    -------
    list[TrainingSample]
    """
    from src.agents.greedy_agent import greedy_policy
    from src.agents.advanced_heuristic import advanced_heuristic_policy

    opponent_policies = {
        "greedy": greedy_policy,
        "advanced": advanced_heuristic_policy,
    }

    # Compute per-type game counts
    game_counts = {}
    remaining = num_games
    for opp_type, fraction in sorted(opponent_mix.items(), key=lambda x: x[1]):
        count = max(1, int(num_games * fraction)) if fraction > 0 else 0
        game_counts[opp_type] = min(count, remaining)
        remaining -= game_counts[opp_type]
    # Give any rounding remainder to the largest bucket
    if remaining > 0:
        largest = max(opponent_mix, key=opponent_mix.get)
        game_counts[largest] += remaining

    all_samples: list[TrainingSample] = []
    stats = {t: {"played": 0, "discarded": 0, "samples": 0, "wins": 0, "truncated": 0}
             for t in game_counts}

    for opp_type, count in game_counts.items():
        if count <= 0:
            continue

        if verbose:
            print(f"  Curriculum: {count} games vs {opp_type}...")

        for i in range(count):
            if opp_type == "self_play":
                samples = play_one_game_true_selfplay(
                    network=network,
                    config=config,
                )
            else:
                samples = play_one_game_vs_heuristic(
                    network=network,
                    opponent_policy_fn=opponent_policies[opp_type],
                    config=config,
                )

            if samples:
                all_samples.extend(samples)
                stats[opp_type]["played"] += 1
                stats[opp_type]["samples"] += len(samples)
                # Track win vs truncation for diagnostics
                val = samples[0].value_target
                if abs(val) > 0.5:
                    stats[opp_type]["wins"] += 1
                else:
                    stats[opp_type]["truncated"] += 1
            else:
                stats[opp_type]["discarded"] += 1

    if verbose:
        parts = []
        total_played = 0
        total_discarded = 0
        for opp_type, s in stats.items():
            parts.append(f"{opp_type}={s['played']}g/{s['samples']}s"
                         f"(W{s['wins']}/T{s['truncated']})")
            total_played += s["played"]
            total_discarded += s["discarded"]
        print(f"  Curriculum complete: {total_played} games, "
              f"{len(all_samples)} samples, "
              f"{total_discarded} discarded [{', '.join(parts)}]")

    return all_samples
