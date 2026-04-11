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

    # Loop
    num_iterations: int = 30
    checkpoint_dir: str = "experiments/exp_d1_alphazero"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """Circular replay buffer for training samples."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: list[TrainingSample] = []
        self._idx = 0

    def add(self, samples: list[TrainingSample]):
        """Add samples to the buffer."""
        for sample in samples:
            if len(self.buffer) < self.max_size:
                self.buffer.append(sample)
            else:
                self.buffer[self._idx] = sample
            self._idx = (self._idx + 1) % self.max_size

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample a random batch.

        Returns (obs_batch, mask_batch, policy_batch, value_batch).
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        obs = np.stack([self.buffer[i].obs for i in indices])
        masks = np.stack([self.buffer[i].action_mask for i in indices])
        policies = np.stack([self.buffer[i].policy_target for i in indices])
        values = np.array([self.buffer[i].value_target for i in indices], dtype=np.float32)

        return obs, masks, policies, values

    def __len__(self) -> int:
        return len(self.buffer)


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
    num_games: int = 20,
    max_steps: int = 300,
) -> float:
    """Play new model vs best model and return new model's win fraction.

    Both sides play from the agent seat (half the games each), and the side
    with the higher tournament score in that game wins. Draws (equal score,
    including mirrored deterministic-argmax games) count as 0.5 to avoid
    the old "always 0.50" artefact where a score threshold was symmetric.
    """
    new_policy = _create_raw_policy(new_network)
    best_policy = _create_raw_policy(best_network)

    new_points = 0.0
    total = 0

    # Half the games: new as agent, best as opponent
    for _ in range(num_games):
        new_result = play_game(new_policy, best_policy, max_steps=max_steps)
        best_result = play_game(best_policy, new_policy, max_steps=max_steps)

        # Head-to-head on the same seat comparison: higher tournament score wins.
        # new-as-agent vs best-as-agent — whichever got more score wins that pairing.
        if new_result["agent_score"] > best_result["agent_score"]:
            new_points += 1.0
        elif new_result["agent_score"] < best_result["agent_score"]:
            pass
        else:
            new_points += 0.5  # tie

        # Also credit terminal wins directly (winner='agent' strictly > truncated)
        if new_result["winner"] == "agent" and best_result["winner"] != "agent":
            new_points += 0.25
        elif best_result["winner"] == "agent" and new_result["winner"] != "agent":
            new_points -= 0.25

        total += 1

    return max(0.0, min(1.0, new_points / max(total, 1)))


def train_alphazero(config: TrainingConfig = TrainingConfig(), resume_from: Optional[str] = None):
    """Main AlphaZero training loop.

    Parameters
    ----------
    config : TrainingConfig
    resume_from : str or None
        Path to checkpoint to resume from.
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

    replay_buffer = ReplayBuffer(config.replay_buffer_size)
    training_log: list[dict] = []

    for iteration in range(start_iteration, config.num_iterations):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{config.num_iterations}")
        print(f"{'='*60}")

        # ----- 1. Self-Play -----
        print(f"\n[1/3] Self-play: generating {config.games_per_iteration} games...")
        sp_start = time.time()

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

        # Quick eval with raw policy (no MCTS — fast)
        eval_results = evaluate_model(
            current_net,
            num_games=config.eval_games,
            max_steps=300,
            use_mcts=False,
        )

        vs_random = eval_results["vs_random"]
        vs_greedy = eval_results["vs_greedy"]
        vs_advanced = eval_results["vs_advanced"]
        eval_time = time.time() - eval_start

        print(f"  vs Random:   pins={vs_random['avg_pins_in_goal']:.1f}, "
              f"score={vs_random['avg_tournament_score']:.1f}, "
              f"wins={vs_random['agent_wins']}/{config.eval_games}")
        print(f"  vs Greedy:   pins={vs_greedy['avg_pins_in_goal']:.1f}, "
              f"score={vs_greedy['avg_tournament_score']:.1f}, "
              f"wins={vs_greedy['agent_wins']}/{config.eval_games}")
        print(f"  vs Advanced: pins={vs_advanced['avg_pins_in_goal']:.1f}, "
              f"score={vs_advanced['avg_tournament_score']:.1f}, "
              f"wins={vs_advanced['agent_wins']}/{config.eval_games}")
        print(f"  Eval time: {eval_time:.1f}s")

        # Compare with best model
        win_rate = _compare_models(current_net, best_net, num_games=10, max_steps=200)
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
