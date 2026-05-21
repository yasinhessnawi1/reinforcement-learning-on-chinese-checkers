"""
scm_self_play.py — Self-play with Search-Conditioned Modulation.

Two modes:
1. Data collection: play games with frozen policy + MCTS, collect
   (search_features, raw_logits, action_mask, mcts_dist) per turn
   for training the SCM GRU.

2. SCM inference: play games with SCM modulating the final action
   selection, logging interpretability data.
"""

import numpy as np
from typing import Optional

from src.env.board_wrapper import BoardWrapper
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.network.alphazero_net import AlphaZeroNet
from src.network.search_conditioned_modulator import (
    SCMConfig,
    SCMLogger,
    SearchConditionedModulator,
    SearchStats,
    extract_search_features,
)
from src.search.mcts import AlphaZeroMCTS, _score_colour
from src.training.alphazero_self_play import SelfPlayConfig, TrainingSample, _compute_game_value
from src.training.scm_trainer import SCMTrajectory, SCMTrajectoryStep

import torch
import torch.nn.functional as F


def play_game_collect_scm_data(
    network: AlphaZeroNet,
    config: SelfPlayConfig = SelfPlayConfig(),
    opponent_policy=None,
    mcts_engine: Optional[AlphaZeroMCTS] = None,
) -> tuple[list[TrainingSample], SCMTrajectory]:
    """Play one game and collect both training samples and SCM trajectory data.

    Returns
    -------
    (training_samples, scm_trajectory)
    """
    env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
    obs, info = env.reset()

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

    trajectory: list[dict] = []
    scm_steps: list[SCMTrajectoryStep] = []
    move_count = 0
    done = False

    while not done:
        action_mask = env.action_masks()
        temperature = 1.0 if move_count < config.temperature_moves else config.temperature_low

        # Run MCTS with stats extraction
        action_probs, search_stats = mcts_engine.get_action_probs_with_stats(
            env, temperature=temperature, turn_number=move_count,
        )

        # Get raw logits from the frozen policy network
        raw_logits, _, _ = network.predict_raw_logits(obs, action_mask)

        # Extract search features for SCM
        search_features = extract_search_features(search_stats)

        # Record for standard training
        trajectory.append({
            "obs": obs.copy(),
            "action_mask": action_mask.copy(),
            "policy_target": action_probs.copy(),
        })

        # Record for SCM training
        scm_steps.append(SCMTrajectoryStep(
            search_features=search_features,
            raw_logits=raw_logits,
            action_mask=action_mask.copy(),
            mcts_policy=action_probs.copy(),
        ))

        # Sample action
        if temperature < 1e-6:
            action = int(np.argmax(action_probs))
        else:
            action = int(np.random.choice(len(action_probs), p=action_probs))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        move_count += 1

    # Compute game outcome
    agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board is not None else False
    opponent_won = (
        env._board.check_win(env._OPPONENT_COLOUR)
        if env._board is not None and not env._no_opponent
        else False
    )
    game_value = _compute_game_value(env, agent_won, opponent_won)

    # Build training samples
    samples = [
        TrainingSample(
            obs=step["obs"],
            action_mask=step["action_mask"],
            policy_target=step["policy_target"],
            value_target=game_value,
        )
        for step in trajectory
    ]

    scm_traj = SCMTrajectory(steps=scm_steps, game_outcome=game_value)

    return samples, scm_traj


def play_game_with_scm(
    network: AlphaZeroNet,
    scm: SearchConditionedModulator,
    config: SelfPlayConfig = SelfPlayConfig(),
    opponent_policy=None,
    mcts_engine: Optional[AlphaZeroMCTS] = None,
    logger: Optional[SCMLogger] = None,
    game_id: Optional[str] = None,
    blend_alpha: float = 0.3,
) -> tuple[list[TrainingSample], float, dict]:
    """Play one game using SCM to modulate action selection.

    The flow per turn:
    1. Run MCTS normally (frozen policy)
    2. Extract search stats → feed to GRU → get (gate, shift)
    3. Modulate raw policy logits: final = gate * logits + shift
    4. Blend modulated distribution with MCTS visit probs
    5. Sample action from blended distribution

    Parameters
    ----------
    blend_alpha : float
        SCM influence: 0.0 = pure MCTS, 1.0 = pure SCM-modulated policy.

    Returns
    -------
    (training_samples, game_value, game_info)
    """
    env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
    obs, info = env.reset()

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

    device = next(scm.parameters()).device
    hidden = scm.init_hidden(batch_size=1, device=device)
    scm.eval()

    trajectory: list[dict] = []
    move_count = 0
    done = False

    while not done:
        action_mask = env.action_masks()
        temperature = 1.0 if move_count < config.temperature_moves else config.temperature_low

        # 1. Run MCTS with stats
        mcts_probs, search_stats = mcts_engine.get_action_probs_with_stats(
            env, temperature=temperature, turn_number=move_count,
        )

        # 2. Get raw logits and extract features
        raw_logits, _, _ = network.predict_raw_logits(obs, action_mask)
        search_features = extract_search_features(search_stats)

        # 3. SCM modulation
        features_t = torch.tensor(
            search_features[np.newaxis], dtype=torch.float32, device=device,
        )
        logits_t = torch.tensor(
            raw_logits[np.newaxis], dtype=torch.float32, device=device,
        )
        mask_t = torch.tensor(
            action_mask[np.newaxis], dtype=torch.bool, device=device,
        )

        with torch.no_grad():
            modulated_logits, gate, shift, hidden = scm.modulate_logits(
                logits_t, features_t, hidden,
            )
            modulated_logits = modulated_logits.masked_fill(~mask_t, -1e9)
            scm_probs = F.softmax(modulated_logits, dim=-1).squeeze(0).cpu().numpy()

        # 4. Blend SCM probs with MCTS probs
        # blend_alpha: 0.0 = pure MCTS, 1.0 = pure SCM-modulated policy
        blended_probs = (1.0 - blend_alpha) * mcts_probs + blend_alpha * scm_probs
        # Renormalize
        total = blended_probs.sum()
        if total > 0:
            blended_probs = blended_probs / total
        else:
            blended_probs = mcts_probs

        # 5. Log interpretability data
        if logger is not None and game_id is not None:
            pins_in_goal = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
            opp_pins = 0
            if env._board and not env._no_opponent:
                opp_pins = env._board.pins_in_goal(env._OPPONENT_COLOUR)

            gate_np = gate.squeeze(0).cpu().numpy()
            shift_np = shift.squeeze(0).cpu().numpy()
            hidden_np = hidden.squeeze(0).cpu().numpy()

            logger.log_turn(
                game_id=game_id,
                turn_number=move_count,
                pins_in_goal=pins_in_goal,
                opponent_pins_in_goal=opp_pins,
                search_stats=search_stats,
                gate=gate_np,
                shift=shift_np,
                hidden_state=hidden_np,
                raw_logits=raw_logits,
                modulated_logits=modulated_logits.squeeze(0).cpu().numpy(),
                action_mask=action_mask,
            )

        # Record sample
        trajectory.append({
            "obs": obs.copy(),
            "action_mask": action_mask.copy(),
            "policy_target": blended_probs.copy(),
        })

        # Sample action
        if temperature < 1e-6:
            action = int(np.argmax(blended_probs))
        else:
            action = int(np.random.choice(len(blended_probs), p=blended_probs))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        move_count += 1

    # Compute outcome
    agent_won = env._board.check_win(env._AGENT_COLOUR) if env._board is not None else False
    opponent_won = (
        env._board.check_win(env._OPPONENT_COLOUR)
        if env._board is not None and not env._no_opponent
        else False
    )
    game_value = _compute_game_value(env, agent_won, opponent_won)

    agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
    opp_pins = 0
    if env._board and not env._no_opponent:
        opp_pins = env._board.pins_in_goal(env._OPPONENT_COLOUR)

    game_info = {
        "agent_pins_in_goal": agent_pins,
        "opponent_pins_in_goal": opp_pins,
        "agent_won": agent_won,
        "opponent_won": opponent_won,
        "moves": move_count,
        "game_value": game_value,
        "truncated": not agent_won and not opponent_won,
    }

    samples = [
        TrainingSample(
            obs=step["obs"],
            action_mask=step["action_mask"],
            policy_target=step["policy_target"],
            value_target=game_value,
        )
        for step in trajectory
    ]

    return samples, game_value, game_info


def _scm_traj_to_dict(traj: SCMTrajectory) -> dict:
    """Convert SCMTrajectory to a dict of numpy arrays for efficient storage."""
    n = len(traj.steps)
    dim_feat = traj.steps[0].search_features.shape[0]
    dim_act = traj.steps[0].raw_logits.shape[0]

    features = np.zeros((n, dim_feat), dtype=np.float32)
    logits = np.zeros((n, dim_act), dtype=np.float32)
    masks = np.zeros((n, dim_act), dtype=np.bool_)
    policies = np.zeros((n, dim_act), dtype=np.float32)

    for j, step in enumerate(traj.steps):
        features[j] = step.search_features
        logits[j] = step.raw_logits
        masks[j] = step.action_mask
        policies[j] = step.mcts_policy

    return {
        "features": features,
        "logits": logits,
        "masks": masks,
        "policies": policies,
        "outcome": np.float32(traj.game_outcome),
    }


def _dict_to_scm_traj(d: dict) -> SCMTrajectory:
    """Reconstruct SCMTrajectory from a stored dict."""
    n = d["features"].shape[0]
    steps = []
    for j in range(n):
        steps.append(SCMTrajectoryStep(
            search_features=d["features"][j],
            raw_logits=d["logits"][j],
            action_mask=d["masks"][j],
            mcts_policy=d["policies"][j],
        ))
    return SCMTrajectory(steps=steps, game_outcome=float(d["outcome"]))


def _save_traj_chunk(trajectories: list[SCMTrajectory], path: str) -> None:
    """Save a chunk of trajectories as compressed .npz (memory-efficient)."""
    arrays = {}
    for i, traj in enumerate(trajectories):
        d = _scm_traj_to_dict(traj)
        for key, val in d.items():
            arrays[f"{i}_{key}"] = val
    arrays["num_trajectories"] = np.array([len(trajectories)], dtype=np.int32)
    np.savez_compressed(path, **arrays)


def load_traj_chunks(traj_dir: str) -> list[SCMTrajectory]:
    """Load all trajectory chunks from a directory."""
    from pathlib import Path
    trajectories: list[SCMTrajectory] = []
    chunk_files = sorted(Path(traj_dir).glob("chunk_*.npz"))
    for cf in chunk_files:
        data = np.load(str(cf), allow_pickle=False)
        n = int(data["num_trajectories"][0])
        for i in range(n):
            d = {
                "features": data[f"{i}_features"],
                "logits": data[f"{i}_logits"],
                "masks": data[f"{i}_masks"],
                "policies": data[f"{i}_policies"],
                "outcome": data[f"{i}_outcome"],
            }
            trajectories.append(_dict_to_scm_traj(d))
    return trajectories


def _worker_collect_one_game(
    model_state_bytes: bytes,
    net_config_dict: dict,
    sp_config_dict: dict,
) -> dict | None:
    """Worker function: play one game, return SCM trajectory as picklable dict.

    Runs in a subprocess on CPU.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    n_threads = os.environ.get("AZ_WORKER_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", n_threads)
    os.environ.setdefault("MKL_NUM_THREADS", n_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n_threads)

    import io
    import torch
    try:
        torch.set_num_threads(int(n_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
    from src.training.alphazero_self_play import SelfPlayConfig
    from src.agents.greedy_agent import greedy_policy

    nc = NetworkConfig(**net_config_dict)
    net = AlphaZeroNet(nc, device="cpu")
    state_dict = torch.load(io.BytesIO(model_state_bytes), map_location="cpu", weights_only=True)
    net.model.load_state_dict(state_dict)
    net.model.eval()

    config = SelfPlayConfig(**sp_config_dict)
    _, scm_traj = play_game_collect_scm_data(
        network=net, config=config, opponent_policy=greedy_policy,
    )

    if len(scm_traj.steps) < 5:
        return None

    return _scm_traj_to_dict(scm_traj)


def collect_scm_training_data(
    network: AlphaZeroNet,
    num_games: int = 100,
    config: SelfPlayConfig = SelfPlayConfig(),
    opponent_policy=None,
    verbose: bool = True,
    save_dir: Optional[str] = None,
    num_workers: int = 0,
    chunk_size: int = 50,
) -> list[SCMTrajectory]:
    """Generate SCM training trajectories with parallel workers and incremental saves.

    Parameters
    ----------
    save_dir : str or None
        If set, saves trajectory chunks to disk every chunk_size games.
        This prevents RAM OOM on large runs.
    num_workers : int
        0 = auto (cpu_count - 1), 1 = serial.
    chunk_size : int
        Save a .npz chunk every this many games.
    """
    import io
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers <= 0:
        # Cap auto-detect: each PyTorch subprocess uses ~1.5GB RAM for DLLs alone.
        # On typical machines (16GB RAM), 4 workers is safe; more risks page file OOM.
        max_auto = 4
        num_workers = min(num_games, max(1, min(os.cpu_count() - 1, max_auto)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Serialize model weights for workers
    buf = io.BytesIO()
    torch.save(network.model.state_dict(), buf)
    model_bytes = buf.getvalue()

    net_config_dict = {
        "in_channels": network.config.in_channels,
        "num_actions": network.config.num_actions,
        "num_blocks": network.config.num_blocks,
        "num_filters": network.config.num_filters,
        "architecture": network.config.architecture,
        "d_model": network.config.d_model,
        "n_heads": network.config.n_heads,
        "use_auxiliary_head": network.config.use_auxiliary_head,
    }

    sp_dict = {
        "num_simulations": config.num_simulations,
        "c_puct": config.c_puct,
        "dirichlet_alpha": config.dirichlet_alpha,
        "dirichlet_epsilon": config.dirichlet_epsilon,
        "temperature_moves": config.temperature_moves,
        "temperature_low": config.temperature_low,
        "max_moves": config.max_moves,
        "use_heuristic_value": config.use_heuristic_value,
        "augment_symmetry": False,  # no symmetry for SCM data
    }

    trajectories: list[SCMTrajectory] = []
    total_steps = 0
    chunk_idx = 0
    pending_trajs: list[SCMTrajectory] = []

    if num_workers <= 1:
        # Serial fallback
        if verbose:
            print(f"  SCM data: serial mode ({num_games} games)")
        for i in range(num_games):
            _, scm_traj = play_game_collect_scm_data(
                network=network, config=config, opponent_policy=opponent_policy,
            )
            if len(scm_traj.steps) >= 5:
                pending_trajs.append(scm_traj)
                total_steps += len(scm_traj.steps)

            # Incremental save
            if save_dir and len(pending_trajs) >= chunk_size:
                chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx:04d}.npz")
                _save_traj_chunk(pending_trajs, chunk_path)
                trajectories.extend(pending_trajs)
                if verbose:
                    print(f"  Saved chunk {chunk_idx} ({len(pending_trajs)} games) to {chunk_path}")
                pending_trajs = []
                chunk_idx += 1

            if verbose and (i + 1) % 10 == 0:
                n_done = len(trajectories) + len(pending_trajs)
                print(f"  SCM data: {i + 1}/{num_games} games, {n_done} valid, {total_steps} steps")
    else:
        # Parallel collection — submit in small batches to limit memory
        if verbose:
            print(f"  SCM data: {num_workers} parallel workers ({num_games} games)")

        completed = 0
        submitted = 0
        batch_size = num_workers * 2  # keep pool fed without queuing everything

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while completed < num_games:
                # Submit next batch
                to_submit = min(batch_size, num_games - submitted)
                futures = []
                for _ in range(to_submit):
                    fut = executor.submit(
                        _worker_collect_one_game, model_bytes, net_config_dict, sp_dict,
                    )
                    futures.append(fut)
                    submitted += 1

                # Collect results from this batch
                for future in futures:
                    result = future.result()
                    completed += 1
                    if result is not None:
                        traj = _dict_to_scm_traj(result)
                        pending_trajs.append(traj)
                        total_steps += len(traj.steps)

                    # Incremental save
                    if save_dir and len(pending_trajs) >= chunk_size:
                        chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx:04d}.npz")
                        _save_traj_chunk(pending_trajs, chunk_path)
                        trajectories.extend(pending_trajs)
                        if verbose:
                            print(f"  Saved chunk {chunk_idx} ({len(pending_trajs)} games) to {chunk_path}")
                        pending_trajs = []
                        chunk_idx += 1

                    if verbose and completed % 10 == 0:
                        n_done = len(trajectories) + len(pending_trajs)
                        print(f"  SCM data: {completed}/{num_games} done, {n_done} valid, {total_steps} steps")

    # Save remaining
    if save_dir and pending_trajs:
        chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx:04d}.npz")
        _save_traj_chunk(pending_trajs, chunk_path)
        trajectories.extend(pending_trajs)
        if verbose:
            print(f"  Saved final chunk {chunk_idx} ({len(pending_trajs)} games) to {chunk_path}")
    elif not save_dir:
        trajectories.extend(pending_trajs)

    if verbose:
        avg_len = total_steps / len(trajectories) if trajectories else 0
        print(
            f"  SCM data collection complete: {len(trajectories)} games, "
            f"{total_steps} steps, avg {avg_len:.1f} steps/game"
        )

    return trajectories


def evaluate_with_scm(
    network: AlphaZeroNet,
    scm: SearchConditionedModulator,
    num_games: int = 50,
    config: SelfPlayConfig = SelfPlayConfig(),
    opponent_policy=None,
    log_dir: Optional[str] = None,
    blend_alphas: Optional[list[float]] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate agent with SCM modulation at multiple blend levels vs baseline.

    Reports actual pins_in_goal (the real metric) not just value scores.
    Tests each blend_alpha separately so we can find the optimal blend.
    """
    if blend_alphas is None:
        blend_alphas = [0.3]

    logger = SCMLogger(log_dir) if log_dir else None
    results: dict = {"conditions": {}}

    # --- Baseline (no SCM) ---
    if verbose:
        print(f"\n  Baseline (no SCM) — {num_games} games...")

    from src.training.alphazero_self_play import play_one_game
    base_pins: list[int] = []
    base_opp_pins: list[int] = []
    for i in range(num_games):
        env = ChineseCheckersEnv(opponent_policy=opponent_policy, max_steps=config.max_moves)
        obs, info = env.reset()

        mcts = AlphaZeroMCTS(
            network=network,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=0.0,  # no noise for eval
            use_heuristic_value=config.use_heuristic_value,
            opponent_policy=opponent_policy,
        )

        done = False
        move_count = 0
        while not done:
            temp = 0.1  # near-greedy for eval
            probs = mcts.get_action_probs(env, temperature=temp)
            action = int(np.argmax(probs))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            move_count += 1

        agent_pins = env._board.pins_in_goal(env._AGENT_COLOUR) if env._board else 0
        opp_pins = 0
        if env._board and not env._no_opponent:
            opp_pins = env._board.pins_in_goal(env._OPPONENT_COLOUR)
        base_pins.append(agent_pins)
        base_opp_pins.append(opp_pins)

        if verbose and (i + 1) % 10 == 0:
            print(f"    Baseline: {i+1}/{num_games}  avg_pins={np.mean(base_pins):.1f}  "
                  f"avg_opp_pins={np.mean(base_opp_pins):.1f}")

    results["conditions"]["baseline"] = {
        "avg_pins": float(np.mean(base_pins)),
        "avg_opp_pins": float(np.mean(base_opp_pins)),
        "max_pins": int(np.max(base_pins)),
        "pins_list": base_pins,
        "actual_wins": sum(1 for p in base_pins if p == 10),
    }

    # --- SCM at each blend level ---
    for alpha in blend_alphas:
        label = f"scm_blend_{alpha:.0%}"
        if verbose:
            print(f"\n  SCM blend={alpha:.0%} — {num_games} games...")

        scm_pins: list[int] = []
        scm_opp_pins: list[int] = []
        for i in range(num_games):
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
                blend_alpha=alpha,
            )
            scm_pins.append(game_info["agent_pins_in_goal"])
            scm_opp_pins.append(game_info["opponent_pins_in_goal"])

            if verbose and (i + 1) % 10 == 0:
                print(f"    SCM {alpha:.0%}: {i+1}/{num_games}  avg_pins={np.mean(scm_pins):.1f}  "
                      f"avg_opp_pins={np.mean(scm_opp_pins):.1f}")

        results["conditions"][label] = {
            "avg_pins": float(np.mean(scm_pins)),
            "avg_opp_pins": float(np.mean(scm_opp_pins)),
            "max_pins": int(np.max(scm_pins)),
            "pins_list": scm_pins,
            "actual_wins": sum(1 for p in scm_pins if p == 10),
            "blend_alpha": alpha,
        }

    # Flush logs
    if logger and logger.num_buffered > 0:
        log_path = logger.flush()
        results["log_path"] = str(log_path)

    # Summary
    if verbose:
        print(f"\n  {'='*65}")
        print(f"  {'Condition':<20} {'AvgPins':>8} {'MaxPins':>8} {'OppPins':>8} {'Wins':>6}")
        print(f"  {'-'*65}")
        for label, cond in results["conditions"].items():
            print(f"  {label:<20} {cond['avg_pins']:>8.1f} {cond['max_pins']:>8d} "
                  f"{cond['avg_opp_pins']:>8.1f} {cond['actual_wins']:>5d}/{num_games}")
        print(f"  {'='*65}")

    return results


# ======================================================================
# Multiplayer SCM (2-6 players)
# ======================================================================

def play_game_collect_scm_data_multi(
    network: AlphaZeroNet,
    n_players: int,
    config: SelfPlayConfig = SelfPlayConfig(),
) -> list[SCMTrajectory]:
    """Play one N-player game and collect SCM trajectory data for EVERY player.

    Uses the multicolour encoder (60° hex rotation) and multi_n_self_play
    infrastructure. Returns one SCMTrajectory per colour — the GRU will
    learn from all perspectives.

    Parameters
    ----------
    network : AlphaZeroNet — frozen policy network
    n_players : int — number of players (2-6)
    config : SelfPlayConfig — MCTS settings

    Returns
    -------
    list[SCMTrajectory] — one per colour, each containing per-turn SCM steps
    """
    from src.training.multi_n_self_play import (
        _pick_colours,
        _make_proxy_env_mc,
        _create_mcts_engine_mc,
        _compute_value_per_colour,
        _ENCODER_MC,
        _MAPPER,
    )

    colours = _pick_colours(n_players)
    board = BoardWrapper(colours)
    max_total_steps = config.max_moves * n_players

    # Force non-batched MCTS so we have access to _extract_stats_from_root
    engine = AlphaZeroMCTS(
        network=network,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        use_heuristic_value=config.use_heuristic_value,
        opponent_policy=None,
    )

    # Per-colour SCM step collectors
    scm_steps_per_colour: dict[str, list[SCMTrajectoryStep]] = {c: [] for c in colours}
    move_counts: dict[str, int] = {c: 0 for c in colours}
    step_count = 0
    winner = None

    while step_count < max_total_steps:
        colour = colours[step_count % n_players]
        legal = board.get_legal_moves(colour)
        if not legal:
            break

        proxy = _make_proxy_env_mc(board, colour, step_count, max_total_steps,
                                   turn_order=colours)
        action_mask_raw = _MAPPER.build_action_mask(legal)

        # Encode obs in multicolour mode (rotated to canonical red frame)
        obs = _ENCODER_MC.encode_multicolour(board, colour, colours)

        # Get k-rotation for this colour
        k = _ENCODER_MC.k_to_red_frame(colour)

        # Rotate mask to canonical frame for network
        if k != 0:
            mask_canon = _ENCODER_MC.rotate_action_distribution_k(
                action_mask_raw.astype(np.bool_), k
            ).astype(np.bool_)
        else:
            mask_canon = action_mask_raw.astype(np.bool_)

        # Run MCTS once (operates in raw frame via proxy env)
        temp = 1.0 if move_counts[colour] < config.temperature_moves else config.temperature_low
        root = engine.run(proxy)
        num_actions = 1210
        action_probs_raw = engine._visits_to_probs(
            root, num_actions, action_mask_raw, temp,
        )

        # Rotate action probs to canonical frame
        if k != 0:
            action_probs_canon = _ENCODER_MC.rotate_action_distribution_k(action_probs_raw, k)
        else:
            action_probs_canon = action_probs_raw

        # Get raw logits from network in canonical frame
        raw_logits, _, _ = network.predict_raw_logits(obs, mask_canon)

        # Get raw policy for SearchStats (network output in canonical frame)
        raw_policy_canon = network.predict(obs, mask_canon)[0]

        # Build SearchStats from the MCTS root
        stats = engine._extract_stats_from_root(
            root, num_actions, mask_canon,
            raw_policy=raw_policy_canon,
            turn_number=move_counts[colour],
        )

        search_features = extract_search_features(stats)

        scm_steps_per_colour[colour].append(SCMTrajectoryStep(
            search_features=search_features,
            raw_logits=raw_logits,
            action_mask=mask_canon.copy(),
            mcts_policy=action_probs_canon.copy(),
        ))

        # Sample action in raw frame
        if temp < 1e-6:
            action = int(np.argmax(action_probs_raw))
        else:
            action = int(np.random.choice(len(action_probs_raw), p=action_probs_raw))
        pin_id, dest = _MAPPER.decode(action)
        board.apply_move(colour, pin_id, dest)
        step_count += 1
        move_counts[colour] += 1

        if board.check_win(colour):
            winner = colour
            break

    # Compute per-colour game outcomes
    values = _compute_value_per_colour(board, colours, winner, move_counts)

    trajectories = []
    for colour in colours:
        steps = scm_steps_per_colour[colour]
        if len(steps) >= 5:
            trajectories.append(SCMTrajectory(
                steps=steps,
                game_outcome=values[colour],
            ))

    return trajectories


def _rotate_if_needed(encoder, dist: np.ndarray, k: int) -> np.ndarray:
    """Rotate distribution by k×60° if k != 0."""
    if k != 0:
        return encoder.rotate_action_distribution_k(dist, k)
    return dist


def _worker_collect_one_game_multi(
    model_state_bytes: bytes,
    net_config_dict: dict,
    sp_config_dict: dict,
    n_players: int,
) -> list[dict] | None:
    """Worker: play one N-player game, return SCM trajectories as dicts."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    n_threads = os.environ.get("AZ_WORKER_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", n_threads)
    os.environ.setdefault("MKL_NUM_THREADS", n_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", n_threads)

    import io as _io
    import torch as _torch
    try:
        _torch.set_num_threads(int(n_threads))
        _torch.set_num_interop_threads(1)
    except Exception:
        pass

    from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
    from src.training.alphazero_self_play import SelfPlayConfig

    nc = NetworkConfig(**net_config_dict)
    net = AlphaZeroNet(nc, device="cpu")
    state_dict = _torch.load(_io.BytesIO(model_state_bytes), map_location="cpu", weights_only=True)
    net.model.load_state_dict(state_dict)
    net.model.eval()

    config = SelfPlayConfig(**sp_config_dict)
    trajectories = play_game_collect_scm_data_multi(
        network=net, n_players=n_players, config=config,
    )

    if not trajectories:
        return None

    return [_scm_traj_to_dict(t) for t in trajectories]


def collect_scm_training_data_multi(
    network: AlphaZeroNet,
    num_games: int = 100,
    config: SelfPlayConfig = SelfPlayConfig(),
    n_choices: tuple = (2, 3, 4, 5, 6),
    n_weights: tuple = (3, 2, 2, 2, 1),
    verbose: bool = True,
    save_dir: Optional[str] = None,
    num_workers: int = 0,
    chunk_size: int = 50,
) -> list[SCMTrajectory]:
    """Generate multiplayer SCM training trajectories.

    Each game produces one SCMTrajectory per colour (N trajectories from an
    N-player game). Player count is sampled from n_choices with n_weights.

    Parameters
    ----------
    n_choices : tuple — possible player counts
    n_weights : tuple — relative weights for each player count
    save_dir : str or None — save .npz chunks incrementally
    num_workers : int — 0=auto, 1=serial
    chunk_size : int — trajectories per .npz chunk
    """
    import io as _io
    import os
    import random
    from concurrent.futures import ProcessPoolExecutor

    if num_workers <= 0:
        max_auto = 4
        num_workers = min(num_games, max(1, min(os.cpu_count() - 1, max_auto)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Serialize model weights
    buf = _io.BytesIO()
    torch.save(network.model.state_dict(), buf)
    model_bytes = buf.getvalue()

    net_config_dict = {
        "in_channels": network.config.in_channels,
        "num_actions": network.config.num_actions,
        "num_blocks": network.config.num_blocks,
        "num_filters": network.config.num_filters,
        "architecture": network.config.architecture,
        "d_model": network.config.d_model,
        "n_heads": network.config.n_heads,
        "use_auxiliary_head": network.config.use_auxiliary_head,
    }
    sp_dict = {
        "num_simulations": config.num_simulations,
        "c_puct": config.c_puct,
        "dirichlet_alpha": config.dirichlet_alpha,
        "dirichlet_epsilon": config.dirichlet_epsilon,
        "temperature_moves": config.temperature_moves,
        "temperature_low": config.temperature_low,
        "max_moves": config.max_moves,
        "use_heuristic_value": config.use_heuristic_value,
        "augment_symmetry": False,
    }

    # Pre-pick player counts for each game
    n_weights_total = sum(n_weights)

    def _sample_n() -> int:
        r = random.random() * n_weights_total
        cum = 0
        for nc, w in zip(n_choices, n_weights):
            cum += w
            if r < cum:
                return nc
        return n_choices[-1]

    trajectories: list[SCMTrajectory] = []
    total_steps = 0
    chunk_idx = 0
    pending_trajs: list[SCMTrajectory] = []

    def _flush_chunk() -> None:
        nonlocal chunk_idx, pending_trajs
        if save_dir and len(pending_trajs) >= chunk_size:
            chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx:04d}.npz")
            _save_traj_chunk(pending_trajs[:chunk_size], chunk_path)
            trajectories.extend(pending_trajs[:chunk_size])
            if verbose:
                print(f"  Saved chunk {chunk_idx} ({chunk_size} trajs) to {chunk_path}")
            pending_trajs = pending_trajs[chunk_size:]
            chunk_idx += 1

    if num_workers <= 1:
        if verbose:
            print(f"  Multi-player SCM data: serial mode ({num_games} games)")
        for i in range(num_games):
            n = _sample_n()
            try:
                game_trajs = play_game_collect_scm_data_multi(
                    network=network, n_players=n, config=config,
                )
            except Exception as e:
                if verbose:
                    print(f"  game {i}: error {e}")
                continue
            for t in game_trajs:
                pending_trajs.append(t)
                total_steps += len(t.steps)
            _flush_chunk()
            if verbose and (i + 1) % 10 == 0:
                n_done = len(trajectories) + len(pending_trajs)
                print(f"  SCM multi data: {i+1}/{num_games} games, {n_done} trajs, {total_steps} steps")
    else:
        if verbose:
            print(f"  Multi-player SCM data: {num_workers} workers ({num_games} games)")

        completed = 0
        submitted = 0
        batch_size = num_workers * 2

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            while completed < num_games:
                to_submit = min(batch_size, num_games - submitted)
                futures = []
                for _ in range(to_submit):
                    n = _sample_n()
                    fut = executor.submit(
                        _worker_collect_one_game_multi,
                        model_bytes, net_config_dict, sp_dict, n,
                    )
                    futures.append(fut)
                    submitted += 1

                for future in futures:
                    try:
                        result = future.result(timeout=900)
                    except Exception as e:
                        if verbose:
                            print(f"  worker error: {e}")
                        completed += 1
                        continue
                    completed += 1
                    if result is not None:
                        for d in result:
                            traj = _dict_to_scm_traj(d)
                            pending_trajs.append(traj)
                            total_steps += len(traj.steps)
                    _flush_chunk()

                    if verbose and completed % 10 == 0:
                        n_done = len(trajectories) + len(pending_trajs)
                        print(f"  SCM multi data: {completed}/{num_games} done, {n_done} trajs, {total_steps} steps")

    # Save remaining
    if save_dir and pending_trajs:
        chunk_path = os.path.join(save_dir, f"chunk_{chunk_idx:04d}.npz")
        _save_traj_chunk(pending_trajs, chunk_path)
        trajectories.extend(pending_trajs)
        if verbose:
            print(f"  Saved final chunk {chunk_idx} ({len(pending_trajs)} trajs) to {chunk_path}")
    elif not save_dir:
        trajectories.extend(pending_trajs)

    if verbose:
        avg_len = total_steps / len(trajectories) if trajectories else 0
        print(
            f"  Multi-player SCM data complete: {len(trajectories)} trajs from "
            f"{num_games} games, {total_steps} steps, avg {avg_len:.1f} steps/traj"
        )

    return trajectories


def evaluate_with_scm_multi(
    network: AlphaZeroNet,
    scm: SearchConditionedModulator,
    num_games: int = 50,
    config: SelfPlayConfig = SelfPlayConfig(),
    n_players: int = 4,
    blend_alphas: Optional[list[float]] = None,
    log_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Evaluate SCM modulation in N-player games.

    The agent (always the first colour) uses SCM-modulated policy while
    all opponents use MCTS with the same frozen network (no SCM).

    Reports per-condition: avg pins, max pins, wins, avg tournament score.
    """
    from src.training.multi_n_self_play import (
        _pick_colours,
        _make_proxy_env_mc,
        _create_mcts_engine_mc,
        _tournament_score,
        _ENCODER_MC,
        _MAPPER,
    )

    if blend_alphas is None:
        blend_alphas = [0.2]

    logger = SCMLogger(log_dir) if log_dir else None
    results: dict = {"conditions": {}, "n_players": n_players}

    # --- Baseline (no SCM) ---
    if verbose:
        print(f"\n  Baseline (no SCM, {n_players} players) — {num_games} games...")

    base_pins: list[int] = []
    base_scores: list[float] = []
    base_wins = 0

    for i in range(num_games):
        colours = _pick_colours(n_players)
        agent_colour = colours[0]  # agent is always first
        board = BoardWrapper(colours)
        max_total_steps = config.max_moves * n_players
        engine = _create_mcts_engine_mc(network, config)

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
            temp = 0.1  # near-greedy for eval
            probs = engine.get_action_probs(proxy, temperature=temp)
            action = int(np.argmax(probs))
            pin_id, dest = _MAPPER.decode(action)
            board.apply_move(colour, pin_id, dest)
            step_count += 1
            move_counts[colour] += 1
            if board.check_win(colour):
                winner = colour
                break

        agent_pins = board.pins_in_goal(agent_colour)
        agent_won = (winner == agent_colour)
        t_score = _tournament_score(board, agent_colour, move_counts[agent_colour])
        base_pins.append(agent_pins)
        base_scores.append(t_score)
        if agent_won:
            base_wins += 1

        if verbose and (i + 1) % 10 == 0:
            print(f"    Baseline: {i+1}/{num_games}  avg_pins={np.mean(base_pins):.1f}  "
                  f"wins={base_wins}")

    results["conditions"]["baseline"] = {
        "avg_pins": float(np.mean(base_pins)),
        "max_pins": int(np.max(base_pins)),
        "avg_score": float(np.mean(base_scores)),
        "actual_wins": base_wins,
        "pins_list": base_pins,
    }

    # --- SCM at each blend level ---
    for alpha in blend_alphas:
        label = f"scm_blend_{alpha:.0%}"
        if verbose:
            print(f"\n  SCM blend={alpha:.0%} ({n_players} players) — {num_games} games...")

        scm_pins: list[int] = []
        scm_scores: list[float] = []
        scm_wins = 0
        device = next(scm.parameters()).device

        for i in range(num_games):
            colours = _pick_colours(n_players)
            agent_colour = colours[0]
            board = BoardWrapper(colours)
            max_total_steps = config.max_moves * n_players
            # Use non-batched MCTS for agent (need root stats for SCM)
            agent_engine = AlphaZeroMCTS(
                network=network,
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                dirichlet_alpha=config.dirichlet_alpha,
                dirichlet_epsilon=0.0,  # no noise for eval
                use_heuristic_value=config.use_heuristic_value,
                opponent_policy=None,
            )
            opp_engine = _create_mcts_engine_mc(network, config)

            hidden = scm.init_hidden(batch_size=1, device=device)
            scm.eval()

            step_count = 0
            move_counts = {c: 0 for c in colours}
            winner = None
            gid = logger.new_game() if logger else None

            while step_count < max_total_steps:
                colour = colours[step_count % n_players]
                legal = board.get_legal_moves(colour)
                if not legal:
                    break

                proxy = _make_proxy_env_mc(board, colour, step_count, max_total_steps,
                                           turn_order=colours)

                if colour == agent_colour:
                    # Agent turn: use SCM modulation
                    k = _ENCODER_MC.k_to_red_frame(colour)
                    obs = _ENCODER_MC.encode_multicolour(board, colour, colours)
                    action_mask_raw = _MAPPER.build_action_mask(legal)

                    if k != 0:
                        mask_canon = _ENCODER_MC.rotate_action_distribution_k(
                            action_mask_raw.astype(np.bool_), k
                        ).astype(np.bool_)
                    else:
                        mask_canon = action_mask_raw.astype(np.bool_)

                    # MCTS with root access for search stats
                    root = agent_engine.run(proxy)
                    mcts_probs_raw = agent_engine._visits_to_probs(
                        root, 1210, action_mask_raw, 0.1,
                    )
                    if k != 0:
                        mcts_probs_canon = _ENCODER_MC.rotate_action_distribution_k(mcts_probs_raw, k)
                    else:
                        mcts_probs_canon = mcts_probs_raw

                    # Raw logits + search stats in canonical frame
                    raw_logits, _, _ = network.predict_raw_logits(obs, mask_canon)
                    raw_policy_canon = network.predict(obs, mask_canon)[0]
                    stats = agent_engine._extract_stats_from_root(
                        root, 1210, mask_canon,
                        raw_policy=raw_policy_canon,
                        turn_number=move_counts[colour],
                    )
                    search_features = extract_search_features(stats)

                    # SCM modulation
                    features_t = torch.tensor(
                        search_features[np.newaxis], dtype=torch.float32, device=device,
                    )
                    logits_t = torch.tensor(
                        raw_logits[np.newaxis], dtype=torch.float32, device=device,
                    )
                    mask_t = torch.tensor(
                        mask_canon[np.newaxis], dtype=torch.bool, device=device,
                    )

                    with torch.no_grad():
                        modulated_logits, gate, shift, hidden = scm.modulate_logits(
                            logits_t, features_t, hidden,
                        )
                        modulated_logits = modulated_logits.masked_fill(~mask_t, -1e9)
                        scm_probs_canon = F.softmax(modulated_logits, dim=-1).squeeze(0).cpu().numpy()

                    # Blend in canonical frame
                    blended_canon = (1.0 - alpha) * mcts_probs_canon + alpha * scm_probs_canon
                    total = blended_canon.sum()
                    if total > 0:
                        blended_canon = blended_canon / total
                    else:
                        blended_canon = mcts_probs_canon

                    # Rotate back to raw frame for action selection
                    if k != 0:
                        # Inverse rotation: rotate by (6 - k) to undo k
                        blended_raw = _ENCODER_MC.rotate_action_distribution_k(blended_canon, 6 - k)
                    else:
                        blended_raw = blended_canon

                    action = int(np.argmax(blended_raw))

                    # Log
                    if logger and gid:
                        gate_np = gate.squeeze(0).cpu().numpy()
                        shift_np = shift.squeeze(0).cpu().numpy()
                        hidden_np = hidden.squeeze(0).cpu().numpy()
                        logger.log_turn(
                            game_id=gid,
                            turn_number=move_counts[colour],
                            pins_in_goal=board.pins_in_goal(agent_colour),
                            opponent_pins_in_goal=max(
                                board.pins_in_goal(c) for c in colours if c != agent_colour
                            ),
                            search_stats=stats,
                            gate=gate_np,
                            shift=shift_np,
                            hidden_state=hidden_np,
                            raw_logits=raw_logits,
                            modulated_logits=modulated_logits.squeeze(0).cpu().numpy(),
                            action_mask=mask_canon,
                        )
                else:
                    # Opponent turn: plain MCTS
                    temp = 0.1
                    probs = opp_engine.get_action_probs(proxy, temperature=temp)
                    action = int(np.argmax(probs))

                pin_id, dest = _MAPPER.decode(action)
                board.apply_move(colour, pin_id, dest)
                step_count += 1
                move_counts[colour] += 1
                if board.check_win(colour):
                    winner = colour
                    break

            agent_pins = board.pins_in_goal(agent_colour)
            agent_won = (winner == agent_colour)
            t_score = _tournament_score(board, agent_colour, move_counts[agent_colour])
            scm_pins.append(agent_pins)
            scm_scores.append(t_score)
            if agent_won:
                scm_wins += 1

            if verbose and (i + 1) % 10 == 0:
                print(f"    SCM {alpha:.0%}: {i+1}/{num_games}  avg_pins={np.mean(scm_pins):.1f}  "
                      f"wins={scm_wins}")

        results["conditions"][label] = {
            "avg_pins": float(np.mean(scm_pins)),
            "max_pins": int(np.max(scm_pins)),
            "avg_score": float(np.mean(scm_scores)),
            "actual_wins": scm_wins,
            "blend_alpha": alpha,
            "pins_list": scm_pins,
        }

    # Flush logs
    if logger and logger.num_buffered > 0:
        log_path = logger.flush()
        results["log_path"] = str(log_path)

    # Summary
    if verbose:
        print(f"\n  {'='*70}")
        print(f"  {'Condition':<20} {'AvgPins':>8} {'MaxPins':>8} {'AvgScore':>9} {'Wins':>6}")
        print(f"  {'-'*70}")
        for label, cond in results["conditions"].items():
            print(f"  {label:<20} {cond['avg_pins']:>8.1f} {cond['max_pins']:>8d} "
                  f"{cond['avg_score']:>9.1f} {cond['actual_wins']:>5d}/{num_games}")
        print(f"  {'='*70}")

    return results
