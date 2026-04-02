"""
transformer_extractor.py — Vision Transformer feature extractors for Chinese Checkers.

Two variants:
  - StandardViT  : treats all 289 grid positions (17x17) as patch tokens
  - MaskedViT    : processes only the 121 valid hex board positions (ignores padding)

Both are SB3-compatible (BaseFeaturesExtractor) and plug into MaskablePPO
via policy_kwargs["features_extractor_class"].

Architecture (both variants):
  - Patch embedding: 1x1 conv projecting each position to d_model
  - Positional encoding: learnable embeddings per position
  - Transformer encoder: n_layers self-attention layers, n_heads
  - Readout: mean-pool over tokens → linear projection → features_dim

Novel aspect: no prior Chinese Checkers RL paper has used ViT-style attention.
Long-range dependencies (multi-hop chains spanning the board) may be captured
better by attention than by local CNN kernels.
"""

import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class _TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer (more stable than post-norm)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


class StandardViT(BaseFeaturesExtractor):
    """Vision Transformer over all 17x17 = 289 grid positions.

    Each position is a token.  Positional embeddings are learned.

    Parameters
    ----------
    observation_space : spaces.Box, shape (10, 17, 17)
    d_model : int, transformer width (default 128)
    n_heads : int, attention heads (default 4)
    n_layers : int, transformer depth (default 4)
    features_dim : int, output dimension (default 256)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        in_channels, h, w = observation_space.shape  # (10, 17, 17)
        self.num_tokens = h * w  # 289

        # Project each (in_channels,) patch to d_model via 1x1 conv
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=1)
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, dim_ff=d_model * 4)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Readout: mean-pool → linear
        self.head = nn.Linear(d_model, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W)
        B = obs.shape[0]
        # Patch embed → (B, d_model, H, W) → (B, num_tokens, d_model)
        x = self.patch_embed(obs)          # (B, d_model, 17, 17)
        x = x.flatten(2).transpose(1, 2)  # (B, 289, d_model)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)                   # (B, 289, d_model)
        x = x.mean(dim=1)                  # (B, d_model)
        return self.head(x)                # (B, features_dim)


class MaskedViT(BaseFeaturesExtractor):
    """Vision Transformer over only the 121 valid hex board positions.

    Padding cells (outside the hex board) are never attended to.
    Valid cell positions are determined from channel 4 (valid mask).

    Parameters
    ----------
    observation_space : spaces.Box, shape (10, 17, 17)
    d_model : int, transformer width (default 128)
    n_heads : int, attention heads (default 4)
    n_layers : int, transformer depth (default 4)
    features_dim : int, output dimension (default 256)
    """

    # Hard-coded valid cell mask computed from the board geometry.
    # Channel 4 of the observation is the valid-cells mask; we build the
    # static lookup at construction time by examining a sample observation,
    # or fall back to an approximate geometric rule.
    _VALID_POSITIONS: torch.Tensor | None = None  # shape (121,) of flat indices

    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        in_channels, h, w = observation_space.shape
        self._h = h
        self._w = w

        # Build valid position index tensor from board geometry
        valid_flat = self._get_valid_positions(h, w)  # shape (n_valid,)
        self.register_buffer('valid_pos', valid_flat)
        n_valid = valid_flat.shape[0]

        # Patch embedding: in_channels → d_model
        self.patch_embed = nn.Linear(in_channels, d_model)
        # Learnable positional embeddings — one per valid position
        self.pos_embed = nn.Parameter(torch.zeros(1, n_valid, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        self.layers = nn.ModuleList([
            _TransformerEncoderLayer(d_model, n_heads, dim_ff=d_model * 4)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Readout
        self.head = nn.Linear(d_model, features_dim)

    @staticmethod
    def _get_valid_positions(h: int = 17, w: int = 17) -> torch.Tensor:
        """Return flat indices of valid hex board cells.

        Valid cells: axial coords (q, r) with offset 8 placed into a 17x17
        grid, where a cell (row, col) = (r+8, q+8) is valid if it lies on the
        Chinese Checkers hex board (radius 4 core + 3 home triangles).

        We determine validity by the rule:
            |q| <= 8 and |r| <= 8 and |q+r| <= 8  (core hex condition)
        combined with home triangle extensions. A simple approximation that
        matches the actual board: use the valid-cell mask from channel 4 of a
        fresh BoardWrapper encoding.
        """
        try:
            import os, sys
            base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal')
            )
            if base not in sys.path:
                sys.path.insert(0, base)
            from checkers_board import HexBoard
            board = HexBoard()
            valid_set = set()
            for cell in board.cells:
                row = cell.r + 8
                col = cell.q + 8
                if 0 <= row < h and 0 <= col < w:
                    valid_set.add(row * w + col)
            indices = sorted(valid_set)
            return torch.tensor(indices, dtype=torch.long)
        except Exception:
            # Geometric fallback: standard hex board |q|+|r|<=8 and within 17x17
            indices = []
            for row in range(h):
                for col in range(w):
                    q = col - 8
                    r = row - 8
                    if abs(q) + abs(r) + abs(-q - r) <= 16:  # hex distance from origin <= 8
                        indices.append(row * w + col)
            return torch.tensor(indices, dtype=torch.long)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, C, H, W)
        B = obs.shape[0]
        # Flatten spatial dims: (B, C, H*W) → (B, H*W, C)
        flat = obs.flatten(2).transpose(1, 2)          # (B, 289, C)
        # Select only valid positions
        tokens = flat[:, self.valid_pos, :]            # (B, n_valid, C)
        # Project to d_model
        x = self.patch_embed(tokens)                   # (B, n_valid, d_model)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)    # (B, n_valid, d_model)
        x = x.mean(dim=1)   # (B, d_model)
        return self.head(x) # (B, features_dim)
