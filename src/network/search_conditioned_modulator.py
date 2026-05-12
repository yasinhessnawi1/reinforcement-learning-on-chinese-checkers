"""
search_conditioned_modulator.py — Search-Conditioned Modulation (SCM).

A small GRU that observes MCTS search statistics after each turn, maintains
hidden state across turns within a game, and outputs FiLM parameters
(gate + shift) that modulate the policy network's logits.

The policy network's weights are never touched. The GRU learns to read
search dynamics and adjust the policy output based on accumulated
game-level understanding.

Research question: does the GRU learn interpretable modulation patterns
across game phases (opening/midgame/endgame), advantage states, and
search confidence levels?
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SCMConfig:
    """Configuration for the Search-Conditioned Modulator."""
    search_feature_dim: int = 14       # input features per turn
    hidden_dim: int = 128              # GRU hidden state size
    num_actions: int = 1210            # action space (10 pins x 121 cells)
    max_shift: float = 2.0            # clamp shift magnitude
    identity_reg_weight: float = 0.01  # L2 penalty toward identity (gate=1, shift=0)
    modulation_blend: float = 1.0      # 0=no modulation, 1=full modulation
    top_k: int = 5                     # top-K actions for feature extraction


# ---------------------------------------------------------------------------
# Search Feature Extraction
# ---------------------------------------------------------------------------

@dataclass
class SearchStats:
    """Statistics extracted from a completed MCTS search."""
    visit_counts: np.ndarray      # (num_actions,) raw visit counts
    q_values: np.ndarray          # (num_actions,) Q-values per action (0 if unvisited)
    raw_policy: np.ndarray        # (num_actions,) network's raw policy priors
    action_mask: np.ndarray       # (num_actions,) bool legal actions
    root_value: float             # weighted average Q at root
    max_depth: int                # deepest node reached
    total_visits: int             # sum of all visit counts
    turn_number: int              # which turn in the game


def extract_search_features(
    stats: SearchStats, config: SCMConfig = SCMConfig()
) -> np.ndarray:
    """Extract a fixed-size feature vector from MCTS search statistics.

    Returns
    -------
    np.ndarray, shape (search_feature_dim,) = (14,)
        [top5_visit_fracs(5), top5_q_values(5), kl_div(1),
         root_value(1), max_depth_norm(1), turn_norm(1)]
    """
    top_k = config.top_k

    # Normalized visit fractions
    total = stats.visit_counts.sum()
    visit_fracs = stats.visit_counts / total if total > 0 else stats.visit_counts

    # Top-K by visit count
    top_indices = np.argsort(visit_fracs)[-top_k:][::-1]

    top_visit_fracs = np.zeros(top_k, dtype=np.float32)
    top_q_vals = np.zeros(top_k, dtype=np.float32)
    for i, idx in enumerate(top_indices):
        top_visit_fracs[i] = visit_fracs[idx]
        top_q_vals[i] = stats.q_values[idx]

    # Policy-search KL divergence
    # KL(search || policy) — how much the search disagreed with the raw policy
    search_dist = visit_fracs[stats.action_mask]
    policy_dist = stats.raw_policy[stats.action_mask]
    # Avoid log(0)
    eps = 1e-8
    search_dist = np.clip(search_dist, eps, 1.0)
    policy_dist = np.clip(policy_dist, eps, 1.0)
    # Renormalize after clipping
    search_dist = search_dist / search_dist.sum()
    policy_dist = policy_dist / policy_dist.sum()
    kl_div = float(np.sum(search_dist * np.log(search_dist / policy_dist)))
    kl_div = min(kl_div, 10.0)  # cap extreme values

    # Normalized depth and turn
    max_depth_norm = min(stats.max_depth / 50.0, 1.0)
    turn_norm = min(stats.turn_number / 60.0, 1.0)

    features = np.concatenate([
        top_visit_fracs,                         # 5
        top_q_vals,                              # 5
        np.array([kl_div], dtype=np.float32),    # 1
        np.array([stats.root_value], dtype=np.float32),  # 1
        np.array([max_depth_norm], dtype=np.float32),    # 1
        np.array([turn_norm], dtype=np.float32),         # 1
    ])

    return features  # shape (14,)


# ---------------------------------------------------------------------------
# GRU Modulator Network
# ---------------------------------------------------------------------------

class SearchConditionedModulator(nn.Module):
    """GRU that reads search stats and outputs FiLM modulation parameters.

    Architecture:
        search_features (14,) → GRU (128 hidden) → gate (1210,) + shift (1210,)

    The gate and shift are applied to raw policy logits:
        modulated = gate * logits + shift

    Gate is initialized near 1.0, shift near 0.0 (identity modulation).
    """

    def __init__(self, config: SCMConfig = SCMConfig()) -> None:
        super().__init__()
        self.config = config

        # GRU cell (single-step, we call it per turn)
        self.gru = nn.GRUCell(
            input_size=config.search_feature_dim,
            hidden_size=config.hidden_dim,
        )

        # Output heads: project hidden state to gate and shift
        self.gate_head = nn.Linear(config.hidden_dim, config.num_actions)
        self.shift_head = nn.Linear(config.hidden_dim, config.num_actions)

        # Initialize gate bias positive so sigmoid starts near 1.0 (identity)
        nn.init.constant_(self.gate_head.bias, 2.0)  # sigmoid(2.0) ≈ 0.88
        # Initialize shift bias at 0
        nn.init.constant_(self.shift_head.bias, 0.0)
        # Small weights so initial modulation is minimal
        nn.init.normal_(self.gate_head.weight, std=0.01)
        nn.init.normal_(self.shift_head.weight, std=0.01)

    def forward(
        self,
        search_features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward pass.

        Parameters
        ----------
        search_features : (batch, search_feature_dim) or (search_feature_dim,)
        hidden : (batch, hidden_dim) or None

        Returns
        -------
        gate : (batch, num_actions) in (0, 1) via sigmoid
        shift : (batch, num_actions) in (-max_shift, max_shift) via tanh
        new_hidden : (batch, hidden_dim)
        """
        if search_features.dim() == 1:
            search_features = search_features.unsqueeze(0)

        batch_size = search_features.size(0)
        if hidden is None:
            hidden = torch.zeros(
                batch_size, self.config.hidden_dim,
                device=search_features.device, dtype=search_features.dtype,
            )

        new_hidden = self.gru(search_features, hidden)

        gate = torch.sigmoid(self.gate_head(new_hidden))
        shift = torch.tanh(self.shift_head(new_hidden)) * self.config.max_shift

        return gate, shift, new_hidden

    def modulate_logits(
        self,
        raw_logits: torch.Tensor,
        search_features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        blend: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply FiLM modulation to raw policy logits.

        Parameters
        ----------
        raw_logits : (batch, num_actions) — from the frozen policy network
        search_features : (batch, search_feature_dim)
        hidden : (batch, hidden_dim) or None
        blend : float or None — override config.modulation_blend

        Returns
        -------
        modulated_logits : (batch, num_actions)
        gate : (batch, num_actions)
        shift : (batch, num_actions)
        new_hidden : (batch, hidden_dim)
        """
        gate, shift, new_hidden = self.forward(search_features, hidden)

        b = blend if blend is not None else self.config.modulation_blend
        if b < 1.0:
            # Interpolate toward identity
            gate = (1.0 - b) * torch.ones_like(gate) + b * gate
            shift = b * shift

        modulated = gate * raw_logits + shift
        return modulated, gate, shift, new_hidden

    def identity_regularization_loss(
        self, gate: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """L2 penalty pushing gate toward 1 and shift toward 0."""
        gate_loss = ((gate - 1.0) ** 2).mean()
        shift_loss = (shift ** 2).mean()
        return gate_loss + shift_loss

    def init_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Create zero initial hidden state."""
        dev = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.config.hidden_dim, device=dev)

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Interpretability Logger
# ---------------------------------------------------------------------------

@dataclass
class SCMTurnLog:
    """Per-turn modulation data for interpretability analysis."""
    game_id: str
    turn_number: int
    game_phase: str                    # "opening", "midgame", "endgame"
    pins_in_goal: int
    opponent_pins_in_goal: int
    search_confidence: float           # entropy of MCTS visit distribution
    policy_search_kl: float
    gate_mean: float
    gate_std: float
    shift_mean: float
    shift_std: float
    gate_magnitude: float              # ||gate - 1||_2
    shift_magnitude: float             # ||shift||_2
    top5_gate_values: list[float] = field(default_factory=list)
    top5_shift_values: list[float] = field(default_factory=list)
    modulation_changed_top1: bool = False
    gru_hidden_norm: float = 0.0
    raw_top1_action: int = 0
    modulated_top1_action: int = 0


def classify_game_phase(turn: int) -> str:
    """Classify turn number into game phase."""
    if turn <= 10:
        return "opening"
    elif turn <= 30:
        return "midgame"
    return "endgame"


def compute_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    p = probs[probs > 1e-8]
    return float(-np.sum(p * np.log(p)))


class SCMLogger:
    """Collects and writes per-turn modulation logs for analysis."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._logs: list[SCMTurnLog] = []
        self._game_counter = 0

    def new_game(self) -> str:
        """Start a new game, return game ID."""
        self._game_counter += 1
        return f"game_{self._game_counter:05d}_{int(time.time())}"

    def log_turn(
        self,
        game_id: str,
        turn_number: int,
        pins_in_goal: int,
        opponent_pins_in_goal: int,
        search_stats: SearchStats,
        gate: np.ndarray,
        shift: np.ndarray,
        hidden_state: np.ndarray,
        raw_logits: np.ndarray,
        modulated_logits: np.ndarray,
        action_mask: np.ndarray,
    ) -> SCMTurnLog:
        """Record one turn's modulation data."""
        # MCTS visit distribution entropy
        total = search_stats.visit_counts.sum()
        visit_probs = search_stats.visit_counts / total if total > 0 else search_stats.visit_counts
        search_confidence = compute_entropy(visit_probs)

        # Policy-search KL
        features = extract_search_features(search_stats)
        policy_search_kl = float(features[10])  # index 10 is KL

        # Gate and shift statistics
        gate_mean = float(gate.mean())
        gate_std = float(gate.std())
        shift_mean = float(shift.mean())
        shift_std = float(shift.std())
        gate_magnitude = float(np.linalg.norm(gate - 1.0))
        shift_magnitude = float(np.linalg.norm(shift))

        # Top-5 by visit count
        top5_idx = np.argsort(search_stats.visit_counts)[-5:][::-1]
        top5_gate = [float(gate[i]) for i in top5_idx]
        top5_shift = [float(shift[i]) for i in top5_idx]

        # Did modulation change the top-1 action?
        masked_raw = raw_logits.copy()
        masked_raw[~action_mask] = -1e9
        masked_mod = modulated_logits.copy()
        masked_mod[~action_mask] = -1e9
        raw_top1 = int(np.argmax(masked_raw))
        mod_top1 = int(np.argmax(masked_mod))

        log_entry = SCMTurnLog(
            game_id=game_id,
            turn_number=turn_number,
            game_phase=classify_game_phase(turn_number),
            pins_in_goal=pins_in_goal,
            opponent_pins_in_goal=opponent_pins_in_goal,
            search_confidence=search_confidence,
            policy_search_kl=policy_search_kl,
            gate_mean=gate_mean,
            gate_std=gate_std,
            shift_mean=shift_mean,
            shift_std=shift_std,
            gate_magnitude=gate_magnitude,
            shift_magnitude=shift_magnitude,
            top5_gate_values=top5_gate,
            top5_shift_values=top5_shift,
            modulation_changed_top1=(raw_top1 != mod_top1),
            gru_hidden_norm=float(np.linalg.norm(hidden_state)),
            raw_top1_action=raw_top1,
            modulated_top1_action=mod_top1,
        )
        self._logs.append(log_entry)
        return log_entry

    def flush(self, filename: Optional[str] = None) -> Path:
        """Write all buffered logs to a JSONL file and clear buffer."""
        fname = filename or f"scm_logs_{int(time.time())}.jsonl"
        path = self.log_dir / fname
        with open(path, "w") as f:
            for log in self._logs:
                line = {
                    "game_id": log.game_id,
                    "turn": log.turn_number,
                    "phase": log.game_phase,
                    "pins_in_goal": log.pins_in_goal,
                    "opp_pins_in_goal": log.opponent_pins_in_goal,
                    "search_confidence": round(log.search_confidence, 4),
                    "policy_search_kl": round(log.policy_search_kl, 4),
                    "gate_mean": round(log.gate_mean, 4),
                    "gate_std": round(log.gate_std, 4),
                    "shift_mean": round(log.shift_mean, 4),
                    "shift_std": round(log.shift_std, 4),
                    "gate_magnitude": round(log.gate_magnitude, 4),
                    "shift_magnitude": round(log.shift_magnitude, 4),
                    "top5_gate": [round(v, 4) for v in log.top5_gate_values],
                    "top5_shift": [round(v, 4) for v in log.top5_shift_values],
                    "changed_top1": log.modulation_changed_top1,
                    "gru_hidden_norm": round(log.gru_hidden_norm, 4),
                    "raw_top1": log.raw_top1_action,
                    "mod_top1": log.modulated_top1_action,
                }
                f.write(json.dumps(line) + "\n")
        count = len(self._logs)
        self._logs.clear()
        return path

    @property
    def num_buffered(self) -> int:
        return len(self._logs)
