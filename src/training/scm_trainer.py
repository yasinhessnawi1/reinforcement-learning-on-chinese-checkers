"""
scm_trainer.py — Training loop for the Search-Conditioned Modulator (SCM).

Trains the GRU on full game trajectories: each trajectory is a sequence of
(search_features, raw_logits, action_mask, mcts_visit_distribution) tuples.

The GRU processes the sequence step-by-step (maintaining hidden state) and
learns to produce (gate, shift) that bring softmax(gate * logits + shift)
closer to the MCTS visit distribution.

Loss = KL(mcts_dist || softmax(modulated_logits)) + λ * identity_reg
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.network.search_conditioned_modulator import (
    SCMConfig,
    SearchConditionedModulator,
    SearchStats,
    extract_search_features,
)


@dataclass
class SCMTrajectoryStep:
    """One turn's data for SCM training."""
    search_features: np.ndarray    # (search_feature_dim,)
    raw_logits: np.ndarray         # (num_actions,) — from frozen policy
    action_mask: np.ndarray        # (num_actions,) bool
    mcts_policy: np.ndarray        # (num_actions,) — MCTS visit distribution


@dataclass
class SCMTrajectory:
    """A full game trajectory for SCM training."""
    steps: list[SCMTrajectoryStep] = field(default_factory=list)
    game_outcome: float = 0.0      # +1 win, -1 loss


@dataclass(frozen=True)
class SCMTrainConfig:
    """Configuration for SCM training."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    identity_reg_weight: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 10
    min_trajectory_length: int = 5    # skip very short games
    log_interval: int = 5


class SCMTrainer:
    """Trains the SearchConditionedModulator GRU on game trajectories."""

    def __init__(
        self,
        scm: SearchConditionedModulator,
        config: SCMTrainConfig = SCMTrainConfig(),
        device: str = "cpu",
    ) -> None:
        self.scm = scm
        self.config = config
        self.device = torch.device(device)
        self.scm.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.scm.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def train_on_trajectories(
        self,
        trajectories: list[SCMTrajectory],
        verbose: bool = True,
    ) -> dict[str, float]:
        """Train the SCM on a batch of game trajectories.

        Each trajectory is processed sequentially (BPTT through the game)
        so the GRU learns temporal patterns across turns.

        Returns
        -------
        dict with averaged loss components.
        """
        valid = [
            t for t in trajectories
            if len(t.steps) >= self.config.min_trajectory_length
        ]
        if not valid:
            return {"kl_loss": 0.0, "identity_loss": 0.0, "total_loss": 0.0}

        self.scm.train()

        epoch_stats: dict[str, list[float]] = {
            "kl_loss": [], "identity_loss": [], "total_loss": [],
        }

        for epoch in range(self.config.num_epochs):
            np.random.shuffle(valid)
            epoch_kl = 0.0
            epoch_id = 0.0
            epoch_total = 0.0
            n_steps = 0

            for traj in valid:
                loss = self._train_one_trajectory(traj)
                epoch_kl += loss["kl_loss"]
                epoch_id += loss["identity_loss"]
                epoch_total += loss["total_loss"]
                n_steps += len(traj.steps)

            n_traj = len(valid)
            epoch_stats["kl_loss"].append(epoch_kl / n_traj)
            epoch_stats["identity_loss"].append(epoch_id / n_traj)
            epoch_stats["total_loss"].append(epoch_total / n_traj)

            if verbose and (epoch + 1) % self.config.log_interval == 0:
                print(
                    f"  SCM epoch {epoch + 1}/{self.config.num_epochs}: "
                    f"KL={epoch_kl / n_traj:.4f}  "
                    f"ID={epoch_id / n_traj:.4f}  "
                    f"total={epoch_total / n_traj:.4f}  "
                    f"({n_steps} steps across {n_traj} games)"
                )

        self.scm.eval()

        return {
            k: float(np.mean(v)) for k, v in epoch_stats.items()
        }

    def _train_one_trajectory(self, traj: SCMTrajectory) -> dict[str, float]:
        """BPTT through one game trajectory."""
        hidden = self.scm.init_hidden(batch_size=1, device=self.device)

        total_kl = torch.tensor(0.0, device=self.device)
        total_id = torch.tensor(0.0, device=self.device)

        for step in traj.steps:
            features_t = torch.tensor(
                step.search_features[np.newaxis], dtype=torch.float32, device=self.device,
            )
            logits_t = torch.tensor(
                step.raw_logits[np.newaxis], dtype=torch.float32, device=self.device,
            )
            mask_t = torch.tensor(
                step.action_mask[np.newaxis], dtype=torch.bool, device=self.device,
            )
            target_t = torch.tensor(
                step.mcts_policy[np.newaxis], dtype=torch.float32, device=self.device,
            )

            modulated, gate, shift, hidden = self.scm.modulate_logits(
                logits_t, features_t, hidden,
            )

            # Mask illegal actions
            modulated = modulated.masked_fill(~mask_t, -1e9)

            # KL(target || softmax(modulated)) — only over legal actions
            log_probs = F.log_softmax(modulated, dim=-1)
            # Compute per-element KL, then sum only over legal actions
            target_clamped = target_t.clamp(min=1e-8)
            per_element_kl = target_clamped * (target_clamped.log() - log_probs)
            # Zero out illegal positions so they don't contribute
            per_element_kl = per_element_kl * mask_t.float()
            kl = per_element_kl.sum()
            total_kl = total_kl + kl

            # Identity regularization
            id_loss = self.scm.identity_regularization_loss(gate, shift)
            total_id = total_id + id_loss

        n = len(traj.steps)
        avg_kl = total_kl / n
        avg_id = total_id / n
        loss = avg_kl + self.config.identity_reg_weight * avg_id

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.scm.parameters(), self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "kl_loss": avg_kl.item(),
            "identity_loss": avg_id.item(),
            "total_loss": loss.item(),
        }

    def save_checkpoint(self, path: str | Path, extra: Optional[dict] = None) -> None:
        """Save SCM weights."""
        checkpoint = {
            "scm_state_dict": self.scm.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scm_config": {
                "search_feature_dim": self.scm.config.search_feature_dim,
                "hidden_dim": self.scm.config.hidden_dim,
                "num_actions": self.scm.config.num_actions,
                "max_shift": self.scm.config.max_shift,
                "identity_reg_weight": self.scm.config.identity_reg_weight,
            },
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load SCM weights."""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.scm.load_state_dict(checkpoint["scm_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scm.eval()
        return checkpoint
