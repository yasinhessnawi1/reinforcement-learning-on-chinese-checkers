"""
mcts_train.py — Train the policy/value network on MCTS-generated data.

AlphaZero training:
  - Policy loss: cross-entropy between MCTS visit distribution and network policy
  - Value loss:  MSE between network value and game outcome
  - L2 weight decay on all parameters

The network being trained is the MaskablePPO policy.  We access the raw
torch modules (features_extractor, mlp_extractor, action_net, value_net)
and optimise them directly on the MCTS dataset.

Usage
-----
    python src/training/mcts_train.py \\
        --model models/exp005/best/best_model.zip \\
        --data data/mcts_games_001.npz \\
        --out models/mcts_iter1 \\
        --epochs 10

Multiple iterations:
    for i in 1 2 3 4 5; do
        python src/training/mcts_self_play.py --model models/mcts_iter$((i-1))/model.zip --out data/iter${i}.npz
        python src/training/mcts_train.py --model models/mcts_iter$((i-1))/model.zip --data data/iter${i}.npz --out models/mcts_iter${i}
    done
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


def _policy_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss between network logits and MCTS target distribution.

    Parameters
    ----------
    logits : (B, num_actions) — raw logits from action_net
    target_probs : (B, num_actions) — MCTS visit-count probabilities

    Returns
    -------
    scalar loss
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    # KL divergence / cross-entropy: -sum(target * log_pred)
    loss = -(target_probs * log_probs).sum(dim=-1).mean()
    return loss


def _value_loss(pred_values: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
    """MSE loss between predicted value and actual outcome."""
    return nn.functional.mse_loss(pred_values.squeeze(-1), target_values)


def train_on_dataset(
    model: "MaskablePPO",
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    policy_coef: float = 1.0,
    value_coef: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """Fine-tune model on MCTS-generated data.

    Parameters
    ----------
    model : MaskablePPO — the model to update (in place)
    states : (N, C, H, W) float32
    policies : (N, num_actions) float32
    values : (N,) float32
    epochs : int
    batch_size : int
    lr : float
    weight_decay : float
    policy_coef : float — weight for policy loss
    value_coef : float — weight for value loss

    Returns
    -------
    List of per-epoch loss dicts: {'epoch', 'policy_loss', 'value_loss', 'total_loss'}
    """
    policy = model.policy
    device = next(policy.parameters()).device

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Build dataset
    states_t = torch.tensor(states, dtype=torch.float32)
    policies_t = torch.tensor(policies, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)

    dataset = TensorDataset(states_t, policies_t, values_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = []
    policy.train()

    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        n_batches = 0

        for batch_states, batch_policies, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()

            # Forward pass through policy network
            features = policy.extract_features(batch_states)
            if policy.share_features_extractor:
                latent_pi, latent_vf = policy.mlp_extractor(features)
            else:
                pi_features, vf_features = features
                latent_pi = policy.mlp_extractor.forward_actor(pi_features)
                latent_vf = policy.mlp_extractor.forward_critic(vf_features)

            logits = policy.action_net(latent_pi)          # (B, num_actions)
            pred_values = policy.value_net(latent_vf)      # (B, 1)

            p_loss = _policy_loss(logits, batch_policies)
            v_loss = _value_loss(pred_values, batch_values)
            total = policy_coef * p_loss + value_coef * v_loss

            total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_policy_loss += p_loss.item()
            epoch_value_loss += v_loss.item()
            n_batches += 1

        avg_p = epoch_policy_loss / max(n_batches, 1)
        avg_v = epoch_value_loss / max(n_batches, 1)
        avg_total = policy_coef * avg_p + value_coef * avg_v
        record = {
            'epoch': epoch + 1,
            'policy_loss': avg_p,
            'value_loss': avg_v,
            'total_loss': avg_total,
        }
        history.append(record)

        if verbose:
            print(
                f"  Epoch {epoch + 1}/{epochs} — "
                f"policy={avg_p:.4f}  value={avg_v:.4f}  total={avg_total:.4f}"
            )

    policy.set_training_mode(False)
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train policy/value network on MCTS data.")
    parser.add_argument('--model', type=str, required=True, help='Input MaskablePPO .zip checkpoint')
    parser.add_argument('--data', type=str, required=True, help='MCTS dataset .npz file')
    parser.add_argument('--out', type=str, required=True, help='Output directory for updated model')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 weight decay (default: 1e-4)')
    parser.add_argument('--policy-coef', type=float, default=1.0, help='Policy loss coefficient (default: 1.0)')
    parser.add_argument('--value-coef', type=float, default=1.0, help='Value loss coefficient (default: 1.0)')
    args = parser.parse_args(argv)

    print(f"Loading model: {args.model}")
    model = MaskablePPO.load(args.model)

    print(f"Loading dataset: {args.data}")
    data = np.load(args.data)
    states   = data['states']
    policies = data['policies']
    values   = data['values']
    print(f"  {len(states)} samples loaded")

    print(f"\nTraining for {args.epochs} epochs ...")
    history = train_on_dataset(
        model,
        states, policies, values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        policy_coef=args.policy_coef,
        value_coef=args.value_coef,
        verbose=True,
    )

    os.makedirs(args.out, exist_ok=True)
    save_path = os.path.join(args.out, 'model')
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    # Save training history
    hist_path = os.path.join(args.out, 'train_history.npz')
    np.savez(
        hist_path,
        epochs=np.array([h['epoch'] for h in history]),
        policy_loss=np.array([h['policy_loss'] for h in history]),
        value_loss=np.array([h['value_loss'] for h in history]),
        total_loss=np.array([h['total_loss'] for h in history]),
    )
    print(f"Training history saved to {hist_path}")


if __name__ == '__main__':
    main()
