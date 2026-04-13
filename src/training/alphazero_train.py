"""
alphazero_train.py — AlphaZero training loop for Chinese Checkers.

Main loop:
  1. Self-play: generate games using current network + MCTS
  2. Train: sample mini-batches from replay buffer, optimize CE(policy) + MSE(value)
  3. Evaluate: play arena games vs greedy + advanced heuristic
  4. Checkpoint: save if new model wins >55% vs previous best

Supports warm-start from supervised pre-training on heuristic data.
"""

import time
import json
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import (
    TrainingSample,
    SelfPlayConfig,
    generate_self_play_data,
    play_one_game,
)
from src.training.true_self_play import generate_true_self_play_data, generate_curriculum_data
from src.search.mcts import AlphaZeroMCTS
from src.evaluation.arena import play_game, arena_summary
from src.agents.greedy_agent import greedy_policy
from src.agents.advanced_heuristic import advanced_heuristic_policy


@dataclass
class TrainingConfig:
    """Full AlphaZero training configuration."""
    # Network
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # Self-play
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    games_per_iteration: int = 100

    # Training
    batch_size: int = 256
    epochs_per_iteration: int = 10
    replay_buffer_size: int = 50_000
    lr: float = 1e-3
    lr_decay: float = 0.99           # per iteration
    weight_decay: float = 1e-4

    # Evaluation
    eval_games: int = 20
    win_threshold: float = 0.55      # new model must win >55% to replace

    # Prioritized Experience Replay
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4

    # Opponent curriculum (fractions must sum to ~1.0; ignored if use_curriculum=False)
    use_curriculum: bool = False
    curriculum_mix: dict = field(default_factory=lambda: {
        "greedy": 0.30,
        "advanced": 0.30,
        "self_play": 0.40,
    })

    # Loop
    num_iterations: int = 30
    checkpoint_dir: str = "experiments/exp_d1_alphazero"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PrioritizedPool:
    """Priority-weighted sampling pool for Prioritized Experience Replay.

    Each sample has a priority proportional to its last training loss.
    Sampling probability = priority^alpha / sum(priority^alpha).
    Importance-sampling weights correct for the bias: w_i = (N * P(i))^(-beta).

    Used as a drop-in replacement for a plain list when PER is enabled.
    """

    def __init__(self, max_size: int, alpha: float = 0.6, beta_start: float = 0.4):
        self.max_size = max_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.samples: list[TrainingSample] = []
        self.priorities: np.ndarray = np.zeros(max_size, dtype=np.float64)
        self._idx = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return len(self.samples)

    def __bool__(self) -> bool:
        return len(self.samples) > 0

    def add(self, sample: TrainingSample):
        """Add a sample with max priority (will be corrected after first training)."""
        if len(self.samples) < self.max_size:
            self.samples.append(sample)
        else:
            self.samples[self._idx] = sample
        self.priorities[self._idx] = self._max_priority ** self.alpha
        self._idx = (self._idx + 1) % self.max_size

    def sample(self, count: int) -> tuple[list[int], list[TrainingSample], np.ndarray]:
        """Sample indices, samples, and importance-sampling weights.

        Returns (indices, samples, is_weights).
        """
        n = len(self.samples)
        probs = self.priorities[:n] / self.priorities[:n].sum()

        indices = np.random.choice(n, size=count, p=probs, replace=True)
        samples = [self.samples[i] for i in indices]

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalize

        return indices.tolist(), samples, weights.astype(np.float32)

    def update_priorities(self, indices: list[int], losses: np.ndarray):
        """Update priorities for sampled indices based on per-sample loss."""
        for idx, loss in zip(indices, losses):
            priority = (abs(loss) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def anneal_beta(self, progress: float):
        """Anneal beta from beta_start toward 1.0 as training progresses."""
        self.beta = self.beta_start + progress * (1.0 - self.beta_start)


class ReplayBuffer:
    """Multi-pool replay buffer with configurable sampling ratios (Data MoE).

    Maintains separate pools for different data sources:
    - 'main': circular buffer for self-play data (default pool)
    - 'reservoir': fixed warm-start samples (never evicted)
    - Additional named pools for curriculum data, endgame data, etc.

    When sampling a batch, each pool contributes according to its ratio.
    Pools that are empty have their share redistributed to non-empty pools.
    """

    def __init__(self, max_size: int, reservoir_ratio: float = 0.2,
                 use_per: bool = False, per_alpha: float = 0.6, per_beta_start: float = 0.4):
        self.max_size = max_size
        self.use_per = use_per
        # Pool ratios: name -> fraction of batch.  'main' gets the remainder.
        self._pool_ratios: dict[str, float] = {}
        if reservoir_ratio > 0:
            self._pool_ratios["reservoir"] = reservoir_ratio
        # Pools: name -> list of TrainingSample (or PrioritizedPool for main)
        self._per_pool: PrioritizedPool | None = None
        if use_per:
            self._per_pool = PrioritizedPool(max_size, alpha=per_alpha, beta_start=per_beta_start)
        self._pools: dict[str, list[TrainingSample]] = {
            "main": [],
            "reservoir": [],
        }
        self._main_idx = 0  # circular index for main pool (unused if PER)

    def seed_reservoir(self, samples: list[TrainingSample]):
        """Load warm-start samples into the reservoir (called once at startup)."""
        self._pools["reservoir"] = list(samples)
        print(f"  Reservoir seeded with {len(self._pools['reservoir'])} warm-start samples")

    def add_pool(self, name: str, ratio: float):
        """Register a named pool with a sampling ratio.

        The 'main' pool gets 1.0 minus the sum of all other ratios.
        """
        if name not in self._pools:
            self._pools[name] = []
        self._pool_ratios[name] = ratio

    def add(self, samples: list[TrainingSample], pool: str = "main"):
        """Add samples to a pool.  Main pool is circular; others append."""
        if pool == "main" and self._per_pool is not None:
            for sample in samples:
                self._per_pool.add(sample)
            return

        target = self._pools.setdefault(pool, [])
        if pool == "main":
            for sample in samples:
                if len(target) < self.max_size:
                    target.append(sample)
                else:
                    target[self._main_idx] = sample
                self._main_idx = (self._main_idx + 1) % self.max_size
        else:
            target.extend(samples)

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch, mixing pools according to their ratios.

        Returns (obs_batch, mask_batch, policy_batch, value_batch).
        When PER is enabled, also stores last-sampled indices for priority update.
        """
        # Compute effective counts per pool (skip empty pools, redistribute)
        pool_counts: dict[str, int] = {}
        active_ratios: dict[str, float] = {}

        for name, ratio in self._pool_ratios.items():
            if self._pools.get(name):
                active_ratios[name] = ratio

        # Main pool gets the remainder — check PER pool or plain list
        main_has_data = (self._per_pool and len(self._per_pool) > 0) or bool(self._pools.get("main"))
        main_ratio = max(0.0, 1.0 - sum(active_ratios.values()))
        if main_has_data:
            active_ratios["main"] = main_ratio

        total_ratio = sum(active_ratios.values())
        if total_ratio < 1e-6:
            raise ValueError("ReplayBuffer is empty — no pools have data")

        remaining = batch_size
        for name in sorted(active_ratios.keys()):
            count = max(1, int(batch_size * active_ratios[name] / total_ratio))
            pool_counts[name] = min(count, remaining)
            remaining -= pool_counts[name]
        if remaining > 0 and pool_counts:
            largest = max(pool_counts, key=pool_counts.get)
            pool_counts[largest] += remaining

        all_samples = []
        self._last_per_indices: list[int] | None = None
        self._last_per_weights: np.ndarray | None = None

        for name, count in pool_counts.items():
            if count <= 0:
                continue
            if name == "main" and self._per_pool is not None and len(self._per_pool) > 0:
                indices, samples, weights = self._per_pool.sample(count)
                self._last_per_indices = indices
                self._last_per_weights = weights
                all_samples.extend(samples)
            else:
                pool = self._pools.get(name, [])
                if not pool:
                    continue
                indices = np.random.randint(0, len(pool), size=count)
                all_samples.extend(pool[i] for i in indices)

        np.random.shuffle(all_samples)

        obs = np.stack([s.obs for s in all_samples])
        masks = np.stack([s.action_mask for s in all_samples])
        policies = np.stack([s.policy_target for s in all_samples])
        values = np.array([s.value_target for s in all_samples], dtype=np.float32)

        return obs, masks, policies, values

    def update_priorities(self, losses: np.ndarray):
        """Update PER priorities for the last sampled main-pool batch."""
        if self._per_pool is not None and self._last_per_indices is not None:
            self._per_pool.update_priorities(self._last_per_indices, losses)

    def anneal_per_beta(self, progress: float):
        """Anneal PER beta toward 1.0.  progress in [0, 1]."""
        if self._per_pool is not None:
            self._per_pool.anneal_beta(progress)

    def __len__(self) -> int:
        total = sum(len(p) for p in self._pools.values())
        if self._per_pool is not None:
            total += len(self._per_pool)
        return total


def _create_alphazero_arena_policy(
    network: AlphaZeroNet,
    num_sims: int = 50,
    use_heuristic_value: bool = False,
):
    """Create an arena-compatible policy function using MCTS + network.

    Returns a callable(board_wrapper, colour) -> (pin_id, dest).
    """
    from src.env.chinese_checkers_env import ChineseCheckersEnv

    def policy(board_wrapper, colour):
        # Create a temporary env from the board state for MCTS
        env = ChineseCheckersEnv(opponent_policy="none", max_steps=200)
        env._board = board_wrapper
        env._step_count = 0
        env._terminated = False
        env._truncated = False
        env._no_opponent = True

        mcts = AlphaZeroMCTS(
            network=network,
            num_simulations=num_sims,
            dirichlet_epsilon=0.0,  # no noise during eval
            use_heuristic_value=use_heuristic_value,
        )
        action = mcts.select_action(env, temperature=0.0)
        pin_id, dest = env._mapper.decode(action)
        return pin_id, dest

    return policy


def _create_raw_policy(network: AlphaZeroNet):
    """Create an arena-compatible policy using just the network (no MCTS).

    Returns a callable(board_wrapper, colour) -> (pin_id, dest).
    """
    from src.env.chinese_checkers_env import ChineseCheckersEnv
    from src.env.action_mapper import ActionMapper

    mapper = ActionMapper(num_pins=10, num_cells=121)

    def policy(board_wrapper, colour):
        from src.env.state_encoder import StateEncoder
        encoder = StateEncoder(grid_size=17, num_channels=10)
        obs = encoder.encode(board_wrapper, current_colour="red", turn_order=["red", "blue"])
        legal_moves = board_wrapper.get_legal_moves(colour)
        action_mask = mapper.build_action_mask(legal_moves)

        probs, _ = network.predict(obs, action_mask)
        action = int(np.argmax(probs))
        pin_id, dest = mapper.decode(action)
        return pin_id, dest

    return policy


def _create_batched_mcts_arena_policy(
    network: AlphaZeroNet,
    num_sims: int = 50,
    batch_size: int = 8,
    use_heuristic_value: bool = False,
):
    """Create arena policy using batched MCTS for faster GPU inference."""
    from src.env.chinese_checkers_env import ChineseCheckersEnv
    from src.search.batched_mcts import BatchedAlphaZeroMCTS

    def policy(board_wrapper, colour):
        env = ChineseCheckersEnv(opponent_policy="none", max_steps=200)
        env._board = board_wrapper
        env._step_count = 0
        env._terminated = False
        env._truncated = False
        env._no_opponent = True

        mcts = BatchedAlphaZeroMCTS(
            network=network,
            num_simulations=num_sims,
            batch_size=batch_size,
            dirichlet_epsilon=0.0,
            use_heuristic_value=use_heuristic_value,
        )
        action = mcts.select_action(env, temperature=0.0)
        pin_id, dest = env._mapper.decode(action)
        return pin_id, dest

    return policy


def evaluate_model(
    network: AlphaZeroNet,
    num_games: int = 20,
    max_steps: int = 300,
    use_mcts: bool = False,
    mcts_sims: int = 25,
    use_batched_mcts: bool = False,
    mcts_batch_size: int = 8,
    use_heuristic_value: bool = False,
) -> dict:
    """Evaluate network against greedy and advanced heuristic.

    Returns dict with arena summary stats for each opponent.
    """
    if use_batched_mcts:
        agent_policy = _create_batched_mcts_arena_policy(
            network,
            num_sims=mcts_sims,
            batch_size=mcts_batch_size,
            use_heuristic_value=use_heuristic_value,
        )
    elif use_mcts:
        agent_policy = _create_alphazero_arena_policy(
            network, num_sims=mcts_sims, use_heuristic_value=use_heuristic_value
        )
    else:
        agent_policy = _create_raw_policy(network)

    results = {}

    # vs random
    from src.agents.random_agent import random_policy
    random_results = []
    for _ in range(num_games):
        result = play_game(agent_policy, random_policy, max_steps=max_steps)
        random_results.append(result)
    results["vs_random"] = arena_summary(random_results)

    # vs greedy
    greedy_results = []
    for _ in range(num_games):
        result = play_game(agent_policy, greedy_policy, max_steps=max_steps)
        greedy_results.append(result)
    results["vs_greedy"] = arena_summary(greedy_results)

    # vs advanced heuristic
    advanced_results = []
    for _ in range(num_games):
        result = play_game(agent_policy, advanced_heuristic_policy, max_steps=max_steps)
        advanced_results.append(result)
    results["vs_advanced"] = arena_summary(advanced_results)

    return results


def _compare_models(
    new_network: AlphaZeroNet,
    best_network: AlphaZeroNet,
    num_games: int = 10,
    max_steps: int = 300,
    mcts_sims: int = 16,
    use_heuristic_value: bool = True,
) -> float:
    """Compare new model vs best model using MCTS-based anchor evaluation.

    Both models play vs greedy using MCTS (not raw policy) as the anchor.
    Raw policy evaluation is misleading after self-play training softens
    the policy distribution — MCTS evaluation reflects actual play strength.

    Returns float in [0, 1] representing new model's relative quality score.
    """
    new_policy = _create_alphazero_arena_policy(
        new_network, num_sims=mcts_sims, use_heuristic_value=use_heuristic_value
    )
    best_policy = _create_alphazero_arena_policy(
        best_network, num_sims=mcts_sims, use_heuristic_value=use_heuristic_value
    )

    # Evaluate both vs greedy (stable external anchor)
    new_greedy_scores = []
    best_greedy_scores = []

    for _ in range(num_games):
        new_r = play_game(new_policy, greedy_policy, max_steps=max_steps)
        best_r = play_game(best_policy, greedy_policy, max_steps=max_steps)
        new_greedy_scores.append(new_r["pins_in_goal"])
        best_greedy_scores.append(best_r["pins_in_goal"])

    new_avg = float(np.mean(new_greedy_scores))
    best_avg = float(np.mean(best_greedy_scores))

    # Also do head-to-head for tiebreaking (but weighted less)
    h2h_points = 0.0
    h2h_games = max(num_games // 2, 2)
    for _ in range(h2h_games):
        new_r = play_game(new_policy, best_policy, max_steps=max_steps)
        best_r = play_game(best_policy, new_policy, max_steps=max_steps)
        if new_r["agent_score"] > best_r["agent_score"]:
            h2h_points += 1.0
        elif new_r["agent_score"] == best_r["agent_score"]:
            h2h_points += 0.5
    h2h_rate = h2h_points / h2h_games

    # New model score: 70% anchor improvement + 30% h2h
    if new_avg > best_avg + 0.5:
        anchor_score = 1.0
    elif new_avg >= best_avg - 0.5:
        anchor_score = 0.5
    else:
        anchor_score = 0.0

    combined = 0.7 * anchor_score + 0.3 * h2h_rate

    print(f"    Gatekeeper (MCTS {mcts_sims}sims): new_avg_pins={new_avg:.2f}, "
          f"best_avg_pins={best_avg:.2f}, h2h={h2h_rate:.2f}, combined={combined:.2f}")

    return combined


def train_alphazero(
    config: TrainingConfig = TrainingConfig(),
    resume_from: Optional[str] = None,
    warmstart_data_path: Optional[str] = None,
    use_true_self_play: bool = True,
):
    """Main AlphaZero training loop.

    Parameters
    ----------
    config : TrainingConfig
    resume_from : str or None
        Path to checkpoint to resume from.
    warmstart_data_path : str or None
        Path to warm-start .npz data to seed the replay buffer reservoir.
        If provided, 20% of each training batch is drawn from this data.
    use_true_self_play : bool
        If True (default), use true 2-player self-play where both sides
        use MCTS + network.  If False, use legacy single-agent mode
        (agent vs random opponent).
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    # Initialize networks
    current_net = AlphaZeroNet(config.network, device=config.device)
    best_net = AlphaZeroNet(config.network, device=config.device)

    start_iteration = 0

    if resume_from:
        print(f"Resuming from {resume_from}")
        checkpoint = current_net.load_checkpoint(resume_from)
        start_iteration = checkpoint.get("iteration", 0) + 1
        best_net.copy_weights_from(current_net)

    print(f"Network: {current_net.parameter_count():,} parameters")
    print(f"Device: {config.device}")
    print(f"Self-play: {'true 2-player' if use_true_self_play else 'legacy (agent vs random)'}")
    print(f"Config: {config.games_per_iteration} games/iter, "
          f"{config.self_play.num_simulations} sims, "
          f"{config.num_iterations} iterations")

    # Optimizer with LR schedule
    optimizer = torch.optim.Adam(
        current_net.model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    replay_buffer = ReplayBuffer(
        config.replay_buffer_size,
        reservoir_ratio=0.2 if warmstart_data_path else 0.0,
        use_per=config.use_per,
        per_alpha=config.per_alpha,
        per_beta_start=config.per_beta_start,
    )

    # Seed reservoir with warm-start data if provided
    if warmstart_data_path:
        from src.training.warmstart_generator import load_warmstart_data
        from src.training.alphazero_self_play import TrainingSample as TS
        ws_data = load_warmstart_data(warmstart_data_path)
        reservoir_samples = []
        n = ws_data["obs"].shape[0]
        # Sample up to 10k from warm-start to keep reservoir manageable
        max_reservoir = min(n, 10_000)
        indices = np.random.choice(n, size=max_reservoir, replace=False) if n > max_reservoir else np.arange(n)
        for idx in indices:
            reservoir_samples.append(TS(
                obs=ws_data["obs"][idx],
                action_mask=ws_data["action_masks"][idx],
                policy_target=ws_data["policies"][idx],
                value_target=float(ws_data["values"][idx]),
            ))
        replay_buffer.seed_reservoir(reservoir_samples)

    training_log: list[dict] = []

    for iteration in range(start_iteration, config.num_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{config.num_iterations}")
        print(f"{'='*60}")

        # ----- 1. Self-Play -----
        print(f"\n[1/3] Self-play: generating {config.games_per_iteration} games...")
        sp_start = time.time()

        if config.use_curriculum:
            samples = generate_curriculum_data(
                network=current_net,
                num_games=config.games_per_iteration,
                opponent_mix=config.curriculum_mix,
                config=config.self_play,
                verbose=True,
            )
        elif use_true_self_play:
            samples = generate_true_self_play_data(
                network=current_net,
                num_games=config.games_per_iteration,
                config=config.self_play,
                verbose=True,
            )
        else:
            samples = generate_self_play_data(
                network=current_net,
                num_games=config.games_per_iteration,
                config=config.self_play,
                verbose=True,
            )

        replay_buffer.add(samples)
        sp_time = time.time() - sp_start
        print(f"  Self-play: {len(samples)} new samples, "
              f"buffer: {len(replay_buffer)}, "
              f"time: {sp_time:.1f}s")

        # ----- 2. Training -----
        if len(replay_buffer) < config.batch_size:
            print("[2/3] Not enough samples for training, skipping...")
            continue

        print(f"\n[2/3] Training: {config.epochs_per_iteration} epochs, "
              f"batch_size={config.batch_size}...")
        train_start = time.time()

        epoch_losses = []
        for epoch in range(config.epochs_per_iteration):
            num_batches = max(1, len(replay_buffer) // config.batch_size)
            epoch_loss = {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

            for _ in range(num_batches):
                obs_b, mask_b, policy_b, value_b = replay_buffer.sample_batch(
                    config.batch_size
                )
                losses = current_net.train_step(
                    obs_b, mask_b, policy_b, value_b, optimizer
                )
                for k in epoch_loss:
                    epoch_loss[k] += losses[k]

            for k in epoch_loss:
                epoch_loss[k] /= num_batches
            epoch_losses.append(epoch_loss)

        scheduler.step()

        # Anneal PER beta toward 1.0 over training
        if config.use_per:
            progress = (iteration - start_iteration + 1) / max(1, config.num_iterations - start_iteration)
            replay_buffer.anneal_per_beta(progress)

        avg_loss = {
            k: np.mean([e[k] for e in epoch_losses])
            for k in epoch_losses[0]
        }
        train_time = time.time() - train_start
        print(f"  Training: policy_loss={avg_loss['policy_loss']:.4f}, "
              f"value_loss={avg_loss['value_loss']:.4f}, "
              f"total_loss={avg_loss['total_loss']:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.6f}, "
              f"time: {train_time:.1f}s")

        # ----- 3. Evaluation -----
        print(f"\n[3/3] Evaluating...")
        eval_start = time.time()

        # MCTS-based eval (raw policy misleading after self-play training)
        eval_results = evaluate_model(
            current_net,
            num_games=max(config.eval_games // 4, 5),  # fewer games since MCTS is slower
            max_steps=300,
            use_mcts=True,
            mcts_sims=16,
            use_heuristic_value=config.self_play.use_heuristic_value,
        )

        vs_random = eval_results["vs_random"]
        vs_greedy = eval_results["vs_greedy"]
        vs_advanced = eval_results["vs_advanced"]
        eval_time = time.time() - eval_start

        mcts_eval_n = max(config.eval_games // 4, 5)
        print(f"  [MCTS 16-sim eval, {mcts_eval_n} games each]")
        print(f"  vs Random:   pins={vs_random['avg_pins_in_goal']:.1f}, "
              f"score={vs_random['avg_tournament_score']:.1f}, "
              f"wins={vs_random['agent_wins']}/{mcts_eval_n}")
        print(f"  vs Greedy:   pins={vs_greedy['avg_pins_in_goal']:.1f}, "
              f"score={vs_greedy['avg_tournament_score']:.1f}, "
              f"wins={vs_greedy['agent_wins']}/{mcts_eval_n}")
        print(f"  vs Advanced: pins={vs_advanced['avg_pins_in_goal']:.1f}, "
              f"score={vs_advanced['avg_tournament_score']:.1f}, "
              f"wins={vs_advanced['agent_wins']}/{mcts_eval_n}")
        print(f"  Eval time: {eval_time:.1f}s")

        # Compare with best model (using MCTS, not raw policy)
        win_rate = _compare_models(
            current_net, best_net, num_games=6, max_steps=300,
            mcts_sims=16, use_heuristic_value=config.self_play.use_heuristic_value,
        )
        print(f"  vs Best model: win_rate={win_rate:.2f}")

        if win_rate > config.win_threshold:
            print(f"  ** New best model! (win_rate {win_rate:.2f} > {config.win_threshold})")
            best_net.copy_weights_from(current_net)
            best_net.save_checkpoint(
                checkpoint_dir / "best_model.pt",
                iteration=iteration,
                extra={"eval_results": eval_results, "win_rate": win_rate},
            )
        else:
            print(f"  Model not improved (win_rate {win_rate:.2f} <= {config.win_threshold})")

        # Save iteration checkpoint
        current_net.save_checkpoint(
            checkpoint_dir / f"iteration_{iteration:04d}.pt",
            iteration=iteration,
            extra={"eval_results": eval_results, "avg_loss": avg_loss},
        )

        iter_time = time.time() - iter_start
        log_entry = {
            "iteration": iteration,
            "samples": len(samples),
            "buffer_size": len(replay_buffer),
            "avg_loss": avg_loss,
            "vs_random_pins": vs_random["avg_pins_in_goal"],
            "vs_random_score": vs_random["avg_tournament_score"],
            "vs_random_wins": vs_random["agent_wins"],
            "vs_greedy_pins": vs_greedy["avg_pins_in_goal"],
            "vs_greedy_score": vs_greedy["avg_tournament_score"],
            "vs_greedy_wins": vs_greedy["agent_wins"],
            "vs_advanced_pins": vs_advanced["avg_pins_in_goal"],
            "vs_advanced_score": vs_advanced["avg_tournament_score"],
            "vs_advanced_wins": vs_advanced["agent_wins"],
            "win_rate_vs_best": win_rate,
            "lr": scheduler.get_last_lr()[0],
            "time_self_play": sp_time,
            "time_train": train_time,
            "time_eval": eval_time,
            "time_total": iter_time,
        }
        training_log.append(log_entry)

        # Save training log
        with open(checkpoint_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"\n  Iteration {iteration + 1} complete in {iter_time:.1f}s")

    # Final save
    best_net.save_checkpoint(checkpoint_dir / "final_best.pt", iteration=config.num_iterations)
    print(f"\nTraining complete! Best model saved to {checkpoint_dir / 'final_best.pt'}")

    return training_log
