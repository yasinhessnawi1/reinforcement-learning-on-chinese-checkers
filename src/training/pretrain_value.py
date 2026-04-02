"""
pretrain_value.py — Pre-train the value head on heuristic-labelled positions.

Enhancement 2 from the pipeline plan: generate 100k board positions, label them
with a heuristic value, then train only the features_extractor + value_net while
freezing action_net.  This gives the value function a meaningful starting point
before PPO training begins.

Value label formula (mirrors tournament scoring):
    v = pins_in_goal * 100 + max(0, 200 - total_distance_to_goal)
    Normalised to [-1, 1] by dividing by 1200 (theoretical max).

Usage
-----
    # Generate 100k labelled positions, train value head, save updated model
    python src/training/pretrain_value.py \\
        --model models/exp005/best/best_model.zip \\
        --out models/pretrained_value \\
        --num-positions 100000 \\
        --epochs 20

    # Then resume PPO training from the updated model:
    python src/training/train_ppo.py --resume models/pretrained_value/model.zip ...
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal'))

from sb3_contrib import MaskablePPO
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.agents.greedy_agent import greedy_policy

# Theoretical max score: 10 pins * 100 + 200 distance bonus = 1200
_VALUE_SCALE = 1200.0


def _heuristic_value(env: ChineseCheckersEnv) -> float:
    """Compute a normalised heuristic value for the current agent's board position.

    Returns float in [0, 1] — higher = closer to winning.
    """
    board = env._board
    colour = env._AGENT_COLOUR
    pins = board.pins_in_goal(colour) * 100.0
    dist = board.total_distance_to_goal(colour)
    dist_score = max(0.0, 200.0 - dist)
    raw = pins + dist_score
    return raw / _VALUE_SCALE


def generate_positions(
    num_positions: int = 100_000,
    max_steps_per_game: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate board positions by playing greedy vs greedy games and recording states.

    Parameters
    ----------
    num_positions : int
    max_steps_per_game : int

    Returns
    -------
    (states, values) : (N, 10, 17, 17) float32, (N,) float32
    """
    env = ChineseCheckersEnv(opponent_policy=greedy_policy, max_steps=max_steps_per_game)

    states = []
    values = []
    collected = 0

    print(f"Collecting {num_positions:,} positions via greedy vs greedy ...")

    while collected < num_positions:
        obs, info = env.reset()
        terminated = truncated = False

        while not (terminated or truncated) and collected < num_positions:
            # Record current position
            states.append(obs.copy())
            values.append(_heuristic_value(env))
            collected += 1

            # Greedy agent step
            action_mask = info['action_mask']
            legal_moves = env._board.get_legal_moves(env._AGENT_COLOUR)
            if not legal_moves:
                break
            pin_id, dest = greedy_policy(env._board, env._AGENT_COLOUR)
            from src.env.action_mapper import ActionMapper
            mapper = ActionMapper()
            action = mapper.encode(pin_id, dest)

            obs, reward, terminated, truncated, info = env.step(action)

        if collected % 10_000 == 0:
            print(f"  {collected:,} / {num_positions:,} positions collected")

    states_arr = np.array(states[:num_positions], dtype=np.float32)
    values_arr = np.array(values[:num_positions], dtype=np.float32)
    return states_arr, values_arr


def pretrain_value_head(
    model: "MaskablePPO",
    states: np.ndarray,
    values: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 3e-4,
    verbose: bool = True,
) -> list[dict]:
    """Pre-train value head while freezing the action (policy) head.

    Parameters
    ----------
    model : MaskablePPO
    states : (N, C, H, W) float32
    values : (N,) float32 — normalised value labels in [0, 1]
    epochs : int
    batch_size : int
    lr : float

    Returns
    -------
    List of per-epoch loss records.
    """
    policy = model.policy
    device = next(policy.parameters()).device

    # Freeze the policy (action) head — only train features_extractor + value_net + mlp vf
    for param in policy.action_net.parameters():
        param.requires_grad = False

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    states_t = torch.tensor(states, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)

    dataset = TensorDataset(states_t, values_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = []
    policy.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_states, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()

            features = policy.extract_features(batch_states)
            if policy.share_features_extractor:
                _, latent_vf = policy.mlp_extractor(features)
            else:
                _, vf_features = features
                latent_vf = policy.mlp_extractor.forward_critic(vf_features)

            pred = policy.value_net(latent_vf).squeeze(-1)  # (B,)
            loss = nn.functional.mse_loss(pred, batch_values)

            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        record = {'epoch': epoch + 1, 'value_loss': avg_loss}
        history.append(record)

        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs} — value_loss={avg_loss:.5f}")

    # Unfreeze action head for subsequent PPO training
    for param in policy.action_net.parameters():
        param.requires_grad = True

    policy.set_training_mode(False)
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pre-train value head on heuristic positions.")
    parser.add_argument('--model', type=str, required=True, help='Input MaskablePPO .zip')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--num-positions', type=int, default=100_000, help='Positions to generate (default: 100000)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--data-out', type=str, default=None,
                        help='If set, save generated positions to this .npz path for reuse')
    parser.add_argument('--data-in', type=str, default=None,
                        help='If set, load positions from this .npz instead of generating')
    args = parser.parse_args(argv)

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model)

    if args.data_in:
        print(f"Loading positions from {args.data_in}")
        d = np.load(args.data_in)
        states, values = d['states'], d['values']
        print(f"  {len(states):,} positions loaded")
    else:
        states, values = generate_positions(
            num_positions=args.num_positions,
        )
        if args.data_out:
            os.makedirs(os.path.dirname(args.data_out) or '.', exist_ok=True)
            np.savez_compressed(args.data_out, states=states, values=values)
            print(f"Positions saved to {args.data_out}")

    print(f"\nPre-training value head for {args.epochs} epochs ...")
    history = pretrain_value_head(
        model, states, values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True,
    )

    os.makedirs(args.out, exist_ok=True)
    save_path = os.path.join(args.out, 'model')
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    hist_path = os.path.join(args.out, 'pretrain_history.npz')
    np.savez(
        hist_path,
        epochs=np.array([h['epoch'] for h in history]),
        value_loss=np.array([h['value_loss'] for h in history]),
    )
    print(f"Training history saved to {hist_path}")


if __name__ == '__main__':
    main()
