"""Multi-N self-play game generation for d22-class multicolour models.

Plays N-player games (N=2..6) where ALL players use MCTS+network on the
multicolour encoder. Records training samples from EVERY player's perspective.

Differs from `true_self_play.py` (which is hardcoded for 2-player red-vs-blue)
in:
  - N variable per game (drawn from a configurable distribution)
  - Colours assigned via the same primary/complement rotation game.py uses
  - MCTS engines instantiated per playing colour (so each MCTS uses the
    multicolour encoder with the colour's k-rotation)
  - Per-perspective value targets via tournament-score differential
  - Sharpened policy targets (temp=0.3, the d20/d22 trick)
"""
import io
import math
import os
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List

from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder
from src.env.action_mapper import ActionMapper
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.search.mcts import AlphaZeroMCTS, _heuristic_value, _score_colour
from src.search.batched_mcts import BatchedAlphaZeroMCTS
from src.training.alphazero_self_play import TrainingSample, SelfPlayConfig


PRIMARY_COLOURS = ['red', 'lawn green', 'yellow']
COMPLEMENT = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}

# Stateless helpers — created once
_ENCODER_MC = StateEncoder(grid_size=17, num_channels=10, mode="multicolour")
_MAPPER = ActionMapper(num_pins=10, num_cells=121)


def _pick_colours(n: int) -> List[str]:
    """Same logic as game.py: alternate primary/complement starting from a
    random primary."""
    primaries = list(PRIMARY_COLOURS)
    random.shuffle(primaries)
    out = []
    pi = 0
    for i in range(n):
        if i % 2 == 0:
            out.append(primaries[pi])
        else:
            out.append(COMPLEMENT[primaries[pi]])
            pi += 1
    return out


def _make_proxy_env_mc(board, colour: str, step_count: int, max_steps: int,
                       turn_order: List[str]):
    """Proxy env using the multicolour encoder for the given playing colour."""
    proxy = ChineseCheckersEnv.__new__(ChineseCheckersEnv)
    proxy.render_mode = None
    proxy.max_steps = max_steps
    proxy.observation_space = None
    proxy.action_space = type('Space', (), {'n': 1210})()
    proxy._encoder = _ENCODER_MC
    proxy._mapper = _MAPPER
    proxy._AGENT_COLOUR = colour
    # _OPPONENT_COLOUR is needed by some code paths (single-player MCTS).
    # In multi-N we set it to the next opponent in turn order.
    next_opp = colour
    if turn_order and colour in turn_order:
        i = turn_order.index(colour)
        for off in range(1, len(turn_order)):
            cand = turn_order[(i + off) % len(turn_order)]
            if cand != colour:
                next_opp = cand; break
    proxy._OPPONENT_COLOUR = next_opp
    proxy._TURN_ORDER = list(turn_order)
    proxy._no_opponent = True
    proxy._opponent_policy = None
    proxy._board = board.clone()
    proxy._step_count = step_count
    proxy._terminated = False
    proxy._truncated = False
    return proxy


def _create_mcts_engine_mc(network, sp_config: SelfPlayConfig):
    """MCTS for the multicolour encoder. NOTE: uses NO opponent_policy in the
    tree — opponents are just other agents whose moves happen outside the
    MCTS tree (single-player tree per move). This is the same simplification
    `play_one_game_true_selfplay` uses: each MCTS run searches the agent's
    next move as if no one else moves, and the outer game loop alternates
    colours."""
    if sp_config.use_batched_mcts:
        return BatchedAlphaZeroMCTS(
            network=network,
            num_simulations=sp_config.num_simulations,
            batch_size=sp_config.mcts_batch_size,
            c_puct=sp_config.c_puct,
            dirichlet_alpha=sp_config.dirichlet_alpha,
            dirichlet_epsilon=sp_config.dirichlet_epsilon,
            use_heuristic_value=sp_config.use_heuristic_value,
            opponent_policy=None,
        )
    return AlphaZeroMCTS(
        network=network,
        num_simulations=sp_config.num_simulations,
        c_puct=sp_config.c_puct,
        dirichlet_alpha=sp_config.dirichlet_alpha,
        dirichlet_epsilon=sp_config.dirichlet_epsilon,
        use_heuristic_value=sp_config.use_heuristic_value,
        opponent_policy=None,
    )


def _tournament_score(board: BoardWrapper, colour: str, move_count: int,
                      time_sec: float = 30.0) -> float:
    """Same formula as game.py's compute_scores."""
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


def _compute_value_per_colour(board: BoardWrapper, colours: List[str],
                              winner: Optional[str], move_counts: dict
                              ) -> dict:
    """Per-colour value target in [-1, 1].

    Terminal: winner +1, non-winner -1/(N-1) so they sum near zero.
    Non-terminal: tournament-score differential vs the best other colour,
    normalised by 1100. This matches the structure used in
    `_compute_game_values` for 2-player.
    """
    n = len(colours)
    if winner is not None:
        out = {}
        for c in colours:
            if c == winner:
                out[c] = 1.0
            else:
                out[c] = -1.0 / max(n - 1, 1)
        return out
    # Truncated: per-colour score-differential
    scores = {c: _tournament_score(board, c, move_counts.get(c, 0)) for c in colours}
    out = {}
    for c in colours:
        my = scores[c]
        best_other = max((s for cc, s in scores.items() if cc != c), default=my)
        diff = my - best_other
        out[c] = max(-1.0, min(1.0, diff / 1100.0))
    return out


def _sharpen(p: np.ndarray, mask: np.ndarray, temp: float = 0.3) -> np.ndarray:
    """Re-softmax probability mass on legal actions at lower temperature."""
    out = np.zeros_like(p)
    legal = p[mask]
    if legal.sum() == 0:
        out[mask] = 1.0 / max(int(mask.sum()), 1)
        return out
    logits = np.log(legal + 1e-12)
    sharp = np.exp(logits / temp)
    sharp /= sharp.sum()
    out[mask] = sharp
    return out


def play_one_game_multi_n(
    network,
    n_players: int,
    config: SelfPlayConfig = SelfPlayConfig(),
    sharpen_temp: float = 0.3,
) -> List[TrainingSample]:
    """Play one N-player self-play game; return samples from all colours."""
    colours = _pick_colours(n_players)
    board = BoardWrapper(colours)
    max_total_steps = config.max_moves * n_players

    engine = _create_mcts_engine_mc(network, config)

    trajectories = []
    step_count = 0
    move_counts = {c: 0 for c in colours}
    winner = None

    while step_count < max_total_steps:
        colour = colours[step_count % n_players]
        legal = board.get_legal_moves(colour)
        if not legal:
            break

        proxy = _make_proxy_env_mc(board, colour, step_count, max_total_steps,
                                    turn_order=colours)
        action_mask = _MAPPER.build_action_mask(legal)
        # Encode through the multicolour path
        obs = _ENCODER_MC.encode_multicolour(board, colour, colours)

        temp = 1.0 if move_counts[colour] < config.temperature_moves else config.temperature_low
        action_probs, mcts_value = engine.get_action_probs_and_value(proxy, temperature=temp)
        # action_probs is in raw (unrotated) frame; rotate it to the canonical
        # (red) frame before storage so it matches the rotated obs.
        k = _ENCODER_MC.k_to_red_frame(colour)
        mask_canon = _ENCODER_MC.rotate_action_distribution_k(
            action_mask.astype(np.bool_), k
        ).astype(np.bool_)
        policy_canon = _ENCODER_MC.rotate_action_distribution_k(action_probs, k)
        # Sharpen
        sharp = _sharpen(policy_canon, mask_canon, sharpen_temp)

        trajectories.append({
            "colour": colour,
            "obs": obs.copy(),
            "action_mask": mask_canon.copy(),
            "policy_target": sharp.copy(),
            "mcts_value": mcts_value,
        })

        # Sample action in raw frame
        if temp < 1e-6:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))
        pin_id, dest = _MAPPER.decode(action)
        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1
        if board.check_win(colour):
            winner = colour
            break

    # Compute per-colour value targets
    values = _compute_value_per_colour(board, colours, winner, move_counts)

    # Filter degenerate
    max_pins = max(board.pins_in_goal(c) for c in colours)
    if max_pins < config.min_pins_to_keep:
        return []

    samples = []
    lam = config.value_target_lambda
    for step in trajectories:
        v_outcome = values[step["colour"]]
        v_mcts = step["mcts_value"]
        v = max(-1.0, min(1.0, lam * v_outcome + (1.0 - lam) * v_mcts))
        samples.append(TrainingSample(
            obs=step["obs"],
            action_mask=step["action_mask"],
            policy_target=step["policy_target"],
            value_target=v,
        ))
    return samples


def generate_multi_n_self_play(
    network,
    num_games: int,
    config: SelfPlayConfig = SelfPlayConfig(),
    n_choices: tuple = (2, 3, 4, 5, 6),
    n_weights: tuple = (3, 2, 2, 2, 1),
    sharpen_temp: float = 0.3,
    verbose: bool = True,
) -> List[TrainingSample]:
    """Generate samples from N-player self-play games (single-process)."""
    n_weights_total = sum(n_weights)
    samples = []
    discarded = 0
    for i in range(num_games):
        r = random.random() * n_weights_total
        cum = 0
        n = n_choices[0]
        for nc, w in zip(n_choices, n_weights):
            cum += w
            if r < cum:
                n = nc; break
        try:
            game_samples = play_one_game_multi_n(network, n, config, sharpen_temp)
        except Exception as e:
            if verbose:
                print(f"  game {i}: error {e}", flush=True)
            discarded += 1
            continue
        if not game_samples:
            discarded += 1
            continue
        samples.extend(game_samples)
        if verbose and (i + 1) % 10 == 0:
            print(f"  multi-N self-play: {i+1}/{num_games}, "
                  f"{len(samples)} samples, {discarded} discarded",
                  flush=True)
    return samples


# ----------------- Parallel worker helpers ------------------------------
def _worker_play_multi_n(args):
    model_bytes, net_cfg_dict, sp_cfg_dict, n_players, sharpen_temp = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
    net = AlphaZeroNet(NetworkConfig(**net_cfg_dict), device="cpu")
    sd = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    net.model.load_state_dict(sd)
    net.model.eval()
    cfg = SelfPlayConfig(**sp_cfg_dict)
    samples = play_one_game_multi_n(net, n_players, cfg, sharpen_temp)
    return [
        {"obs": s.obs, "action_mask": s.action_mask,
         "policy_target": s.policy_target, "value_target": s.value_target}
        for s in samples
    ]


def generate_multi_n_self_play_parallel(
    network,
    num_games: int,
    config: SelfPlayConfig = SelfPlayConfig(),
    n_choices: tuple = (2, 3, 4, 5, 6),
    n_weights: tuple = (3, 2, 2, 2, 1),
    sharpen_temp: float = 0.3,
    num_workers: int = 16,
    verbose: bool = True,
) -> List[TrainingSample]:
    """Parallel multi-N self-play across CPU workers."""
    from dataclasses import asdict
    from src.network.alphazero_net import NetworkConfig
    import torch as _torch

    # Serialize network weights once
    buf = io.BytesIO()
    _torch.save(network.model.state_dict(), buf)
    model_bytes = buf.getvalue()
    net_cfg_dict = asdict(network.config)
    sp_cfg_dict = asdict(config)
    sp_cfg_dict["use_batched_mcts"] = False  # single-thread workers, no batching

    # Pick N for each game by weighted choice
    n_weights_total = sum(n_weights)
    args_list = []
    for _ in range(num_games):
        r = random.random() * n_weights_total
        cum = 0
        n = n_choices[0]
        for nc, w in zip(n_choices, n_weights):
            cum += w
            if r < cum:
                n = nc; break
        args_list.append((model_bytes, net_cfg_dict, sp_cfg_dict, n, sharpen_temp))

    samples = []
    discarded = 0
    completed = 0
    n_breakdown = {n: 0 for n in n_choices}
    if verbose:
        print(f"  multi-N self-play: {num_games} games on {num_workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_worker_play_multi_n, a): a[3] for a in args_list}
        for f in as_completed(futures):
            n = futures[f]
            completed += 1
            try:
                result = f.result(timeout=900)
            except Exception as e:
                if verbose:
                    print(f"  worker error: {e}", flush=True)
                discarded += 1
                continue
            if not result:
                discarded += 1
                continue
            samples.extend(TrainingSample(**d) for d in result)
            n_breakdown[n] += 1
            if verbose and completed % 5 == 0:
                print(f"  progress: {completed}/{num_games}, "
                      f"{len(samples)} samples, {discarded} discarded",
                      flush=True)
    if verbose:
        print(f"  done: {len(samples)} samples, {discarded} discarded, "
              f"per-N: {n_breakdown}", flush=True)
    return samples
