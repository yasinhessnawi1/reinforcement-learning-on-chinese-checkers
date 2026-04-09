"""
pin_transformer.py — Pin-Attention Transformer for Chinese Checkers.

Instead of treating the board as a 17x17 image (ResNet), this architecture
operates directly on the 20 pins (10 per player) as tokens.

Each pin token contains:
  - Pin position (axial q, r coordinates, normalized)
  - Pin features (distance to goal, in_goal flag, in_home flag)
  - Ownership (mine vs opponent)
  - Positional embedding of cell index

Cross-attention between ALL pins enables the model to learn:
  - Hop chain reasoning: "pin A can hop over pin B to reach C, then hop D to reach goal"
  - Blocking awareness: "opponent pin X blocks my shortest path"
  - Coordination: "moving pin A creates a hop bridge for pin B"

Architecture:
  1. Pin Encoder: embed each pin's features into d_model dimensions
  2. Transformer Encoder: N layers of self-attention between all pins
  3. Board Context: optional CNN branch for global board features
  4. Policy Head: per-pin attention + destination scoring
  5. Value Head: [CLS]-style aggregation of all pin representations
  6. Auxiliary Head: pins_in_goal regression (training signal)

The policy output maps to the 1210-action space (10 pins x 121 destinations)
by computing: for each pin, score all possible destinations using the pin's
learned representation + destination embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PinEncoder(nn.Module):
    """Encode raw pin features into d_model-dimensional token representations.

    Per-pin features (computed from board state):
      - position_q, position_r: normalized axial coordinates [-1, 1]
      - distance_to_nearest_goal: normalized [0, 1]
      - in_goal: binary (0 or 1)
      - in_home: binary (0 or 1)
      - is_mine: binary (1 for agent's pins, 0 for opponent's)
      - cell_index: integer [0, 120] → learned embedding

    Total raw features: 6 floats + 1 embedding
    """

    def __init__(self, d_model: int = 128, num_cells: int = 121):
        super().__init__()
        self.d_model = d_model
        self.num_cells = num_cells

        # Continuous features projection (6 features)
        self.feature_proj = nn.Linear(6, d_model // 2)

        # Cell position embedding (learned)
        self.cell_embedding = nn.Embedding(num_cells + 1, d_model // 2)  # +1 for padding

        # Combine
        self.combine = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, pin_features: torch.Tensor, cell_indices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pin_features : (B, num_pins, 6) — continuous features per pin
        cell_indices : (B, num_pins) — cell index per pin (long)

        Returns
        -------
        (B, num_pins, d_model) — pin token representations
        """
        feat_emb = self.feature_proj(pin_features)  # (B, P, d_model//2)
        cell_emb = self.cell_embedding(cell_indices)  # (B, P, d_model//2)
        combined = torch.cat([feat_emb, cell_emb], dim=-1)  # (B, P, d_model)
        return self.layer_norm(self.combine(combined))


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with pre-norm."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, seq_len, d_model)

        Returns
        -------
        (B, seq_len, d_model)
        """
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm feed-forward
        normed = self.norm2(x)
        x = x + self.ff(normed)

        return x


class DestinationScorer(nn.Module):
    """Score each (pin, destination) pair for the policy head.

    For each pin representation, computes a score for all 121 destinations
    using learned destination embeddings + bilinear attention.

    Output: (B, 10, 121) logits → reshaped to (B, 1210) for the action space.
    """

    def __init__(self, d_model: int = 128, num_cells: int = 121):
        super().__init__()
        self.num_cells = num_cells

        # Destination embeddings (shared across all pins)
        self.dest_embedding = nn.Embedding(num_cells, d_model)

        # Bilinear scoring: pin_repr @ W @ dest_repr
        self.pin_proj = nn.Linear(d_model, d_model, bias=False)

        # Temperature parameter for logit scaling
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model)))

    def forward(self, pin_reprs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pin_reprs : (B, 10, d_model) — agent's pin representations only

        Returns
        -------
        (B, 1210) — logits for each (pin, destination) action
        """
        B = pin_reprs.shape[0]

        # Project pins
        pin_proj = self.pin_proj(pin_reprs)  # (B, 10, d_model)

        # Get all destination embeddings
        dest_ids = torch.arange(self.num_cells, device=pin_reprs.device)
        dest_emb = self.dest_embedding(dest_ids)  # (121, d_model)

        # Bilinear score: each pin scores all destinations
        # (B, 10, d_model) @ (d_model, 121) = (B, 10, 121)
        logits = torch.matmul(pin_proj, dest_emb.T) * self.logit_scale

        # Reshape to flat action space: (B, 10*121) = (B, 1210)
        return logits.reshape(B, -1)


class PinTransformerNet(nn.Module):
    """Pin-Attention Transformer for Chinese Checkers.

    Architecture:
      - 20 pin tokens (10 agent + 10 opponent) through Transformer
      - [CLS] token for global board state
      - Policy head via per-pin destination scoring
      - Value head via [CLS] token
      - Auxiliary head for pins_in_goal prediction

    Parameters
    ----------
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    num_pins : int
        Pins per player (10).
    num_cells : int
        Board cells (121).
    num_actions : int
        Total actions (1210 = 10 pins x 121 cells).
    use_auxiliary_head : bool
        Whether to predict pins_in_goal as auxiliary loss.
    """

    BOARD_SIZE: int = 17  # for compatibility with AlphaZeroNet wrapper

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        num_pins: int = 10,
        num_cells: int = 121,
        num_actions: int = 1210,
        use_auxiliary_head: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_pins = num_pins
        self.num_cells = num_cells
        self.num_actions = num_actions
        self.use_auxiliary_head = use_auxiliary_head

        # Pin encoder
        self.pin_encoder = PinEncoder(d_model, num_cells)

        # [CLS] token for global state
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Policy head: per-pin destination scoring
        self.policy_head = DestinationScorer(d_model, num_cells)

        # Value head: from [CLS] token
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Tanh(),
        )

        # Auxiliary head: predict pins_in_goal (0-10)
        if use_auxiliary_head:
            self.aux_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # output in [0, 1], multiply by 10 for actual count
            )

    def _extract_pin_features(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pin positions and features from the grid observation.

        Fully vectorized — no Python loops over batch or pins. Runs entirely
        on GPU with tensor ops for fast training on large batches.

        Parameters
        ----------
        obs : (B, 10, 17, 17) — standard grid observation

        Returns
        -------
        agent_features : (B, 10, 6) — continuous features for agent pins
        agent_cells : (B, 10) — cell indices for agent pins
        opp_features : (B, 10, 6) — continuous features for opponent pins
        opp_cells : (B, 10) — cell indices for opponent pins
        """
        B = obs.shape[0]
        device = obs.device
        P = self.num_pins  # 10

        agent_map = obs[:, 0]   # (B, 17, 17)
        opp_map = obs[:, 1]     # (B, 17, 17)
        goal_map = obs[:, 2]    # (B, 17, 17)
        home_map = obs[:, 3]    # (B, 17, 17)

        agent_feats, agent_cells = self._extract_pins_vectorized(
            agent_map, goal_map, home_map, is_mine=True, device=device
        )
        opp_feats, opp_cells = self._extract_pins_vectorized(
            opp_map, goal_map, home_map, is_mine=False, device=device
        )

        return agent_feats, agent_cells, opp_feats, opp_cells

    def _extract_pins_vectorized(
        self, pin_map: torch.Tensor, goal_map: torch.Tensor,
        home_map: torch.Tensor, is_mine: bool, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized pin feature extraction for one side (agent or opponent).

        Parameters
        ----------
        pin_map : (B, 17, 17) — binary map of pin positions
        goal_map : (B, 17, 17) — goal triangle
        home_map : (B, 17, 17) — home triangle

        Returns
        -------
        features : (B, 10, 6)
        cell_indices : (B, 10)
        """
        B = pin_map.shape[0]
        P = self.num_pins

        # Flatten spatial dims: (B, 17, 17) -> (B, 289)
        pin_flat = pin_map.reshape(B, -1)  # (B, 289)
        goal_flat = goal_map.reshape(B, -1)
        home_flat = home_map.reshape(B, -1)

        # Find top-P pin positions per batch (by value, guaranteed binary)
        # topk on flat indices — gives us the grid positions
        # Use min(P, actual pins) — pad with zeros if fewer
        pin_counts = (pin_flat > 0.5).sum(dim=1)  # (B,)

        # Get indices of pins (topk works even if fewer than P, pads with 0s)
        # Multiply by pin_flat to only select actual pin positions
        _, flat_indices = torch.topk(pin_flat, k=P, dim=1)  # (B, P)

        # Convert flat index to (row, col)
        rows = flat_indices // 17  # (B, P)
        cols = flat_indices % 17   # (B, P)

        # Normalized coordinates
        q_norm = (cols.float() - 8.0) / 8.0  # (B, P)
        r_norm = (rows.float() - 8.0) / 8.0  # (B, P)

        # In goal: gather from goal_flat at pin positions
        in_goal = torch.gather(goal_flat, 1, flat_indices).float()  # (B, P)
        in_home = torch.gather(home_flat, 1, flat_indices).float()  # (B, P)

        # Distance to nearest goal: approximate via L1 distance to goal centroid
        # Fully vectorized goal center computation
        grid_r = torch.arange(17, device=device, dtype=torch.float32)
        grid_c = torch.arange(17, device=device, dtype=torch.float32)
        # (289,) arrays of row/col for each flat position
        flat_r = grid_r.repeat_interleave(17)  # [0,0,...0, 1,1,...1, ..., 16,16,...16]
        flat_c = grid_c.repeat(17)              # [0,1,...16, 0,1,...16, ...]

        goal_mask = (goal_flat > 0.5).float()  # (B, 289)
        goal_count = goal_mask.sum(dim=1).clamp(min=1)  # (B,)
        goal_center_r = (goal_mask * flat_r.unsqueeze(0)).sum(dim=1) / goal_count  # (B,)
        goal_center_c = (goal_mask * flat_c.unsqueeze(0)).sum(dim=1) / goal_count  # (B,)

        # Distance from each pin to goal center (L1, normalized)
        dist_to_goal = (
            (rows.float() - goal_center_r.unsqueeze(1)).abs() +
            (cols.float() - goal_center_c.unsqueeze(1)).abs()
        ) / 16.0  # (B, P), normalized
        dist_to_goal = dist_to_goal.clamp(0.0, 1.0)

        # Ownership
        ownership = torch.full((B, P), 1.0 if is_mine else 0.0, device=device)

        # Mask out padding (pins that don't exist)
        valid_mask = torch.arange(P, device=device).unsqueeze(0) < pin_counts.unsqueeze(1)  # (B, P)
        ownership = ownership * valid_mask.float()

        # Stack features: (B, P, 6)
        features = torch.stack([q_norm, r_norm, dist_to_goal, in_goal, in_home, ownership], dim=-1)

        # Zero out features for padding pins
        features = features * valid_mask.unsqueeze(-1).float()

        # Cell indices: row * 17 + col, clamped to [0, 120]
        cell_indices = (rows * 17 + cols).clamp(0, self.num_cells - 1).long()  # (B, P)
        # Padding pins get the padding index
        cell_indices = cell_indices * valid_mask.long() + (~valid_mask).long() * self.num_cells

        return features, cell_indices

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass compatible with AlphaZeroNet wrapper.

        Parameters
        ----------
        obs : (B, 10, 17, 17) — standard grid observation

        Returns
        -------
        policy_logits : (B, 1210)
        value : (B, 1)
        """
        B = obs.shape[0]

        # Extract pin features from grid observation
        agent_feats, agent_cells, opp_feats, opp_cells = self._extract_pin_features(obs)

        # Encode pins
        agent_tokens = self.pin_encoder(agent_feats, agent_cells)  # (B, 10, d_model)
        opp_tokens = self.pin_encoder(opp_feats, opp_cells)        # (B, 10, d_model)

        # Combine: [CLS] + agent_pins + opponent_pins = 21 tokens
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, agent_tokens, opp_tokens], dim=1)  # (B, 21, d_model)

        # Transformer layers (cross-attention between all pins)
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        tokens = self.final_norm(tokens)

        # Split outputs
        cls_out = tokens[:, 0]               # (B, d_model) — global state
        agent_out = tokens[:, 1:11]          # (B, 10, d_model) — agent pin reprs

        # Policy: score all (pin, destination) pairs
        policy_logits = self.policy_head(agent_out)  # (B, 1210)

        # Value: from [CLS] token
        value = self.value_head(cls_out)  # (B, 1)

        return policy_logits, value

    def forward_with_auxiliary(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with auxiliary pins_in_goal prediction.

        Returns (policy_logits, value, aux_pins_pred) where
        aux_pins_pred is (B, 1) in [0, 1] (multiply by 10 for count).
        """
        B = obs.shape[0]

        agent_feats, agent_cells, opp_feats, opp_cells = self._extract_pin_features(obs)
        agent_tokens = self.pin_encoder(agent_feats, agent_cells)
        opp_tokens = self.pin_encoder(opp_feats, opp_cells)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, agent_tokens, opp_tokens], dim=1)

        for layer in self.transformer_layers:
            tokens = layer(tokens)

        tokens = self.final_norm(tokens)

        cls_out = tokens[:, 0]
        agent_out = tokens[:, 1:11]

        policy_logits = self.policy_head(agent_out)
        value = self.value_head(cls_out)

        aux_pred = torch.zeros(B, 1, device=obs.device)
        if self.use_auxiliary_head:
            aux_pred = self.aux_head(cls_out)

        return policy_logits, value, aux_pred
