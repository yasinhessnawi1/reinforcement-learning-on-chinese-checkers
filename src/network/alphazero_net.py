"""
alphazero_net.py — AlphaZero-style network wrapper for Chinese Checkers.

Wraps any backbone (ResNet, PinTransformer, GATEAU) with a unified interface:
  - predict(obs, action_mask) -> (policy_probs, value)
  - predict_batch(obs_batch, mask_batch) -> (policy_batch, value_batch)
  - save_checkpoint / load_checkpoint

Supports three architectures:
  - "resnet" (default): ChineseCheckersResNet — grid-based CNN
  - "pin_transformer": PinTransformerNet — cross-attention between pins
  - "gateau": GATEAUNet — graph attention on hex topology

Also supports auxiliary value head (pins_in_goal prediction) for richer
training signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from src.network.resnet import ChineseCheckersResNet


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for the AlphaZero network."""
    in_channels: int = 10
    num_actions: int = 1210
    num_blocks: int = 6
    num_filters: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Architecture selection
    architecture: str = "resnet"  # "resnet", "pin_transformer", "gateau"
    # Transformer / GATEAU settings
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    # Auxiliary head
    use_auxiliary_head: bool = False
    auxiliary_loss_weight: float = 0.1


def _build_model(config: NetworkConfig, device: torch.device) -> nn.Module:
    """Build the appropriate model architecture from config."""
    if config.architecture == "resnet":
        return ChineseCheckersResNet(
            in_channels=config.in_channels,
            num_actions=config.num_actions,
            num_blocks=config.num_blocks,
            num_filters=config.num_filters,
        ).to(device)
    elif config.architecture == "pin_transformer":
        from src.network.pin_transformer import PinTransformerNet
        return PinTransformerNet(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.num_blocks,
            d_ff=config.d_ff,
            dropout=config.dropout,
            num_actions=config.num_actions,
            use_auxiliary_head=config.use_auxiliary_head,
        ).to(device)
    elif config.architecture == "gateau":
        from src.network.gateau import GATEAUNet
        return GATEAUNet(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.num_blocks,
            dropout=config.dropout,
            num_actions=config.num_actions,
            use_auxiliary_head=config.use_auxiliary_head,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture}")


class AlphaZeroNet:
    """AlphaZero-style network wrapper.

    Provides a clean interface for MCTS and training without exposing
    PyTorch internals to the rest of the codebase.

    Supports ResNet, PinTransformer, and GATEAU architectures through
    the config.architecture parameter.

    Parameters
    ----------
    config : NetworkConfig
        Network architecture and optimizer configuration.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, config: NetworkConfig = NetworkConfig(), device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.model = _build_model(config, self.device)
        self.model.eval()

    def predict(
        self, obs: np.ndarray, action_mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Single-sample inference for MCTS.

        Parameters
        ----------
        obs : np.ndarray, shape (C, H, W)
        action_mask : np.ndarray, shape (num_actions,), dtype bool

        Returns
        -------
        (policy_probs, value) where:
            policy_probs: np.ndarray shape (num_actions,) — masked softmax
            value: float in [-1, 1]
        """
        obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(
            action_mask[np.newaxis], dtype=torch.bool, device=self.device
        )

        with torch.no_grad():
            logits, value = self.model(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            value_scalar = value.squeeze().cpu().item()

        return probs, value_scalar

    def predict_batch(
        self, obs_batch: np.ndarray, mask_batch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch inference for parallel self-play.

        Parameters
        ----------
        obs_batch : np.ndarray, shape (B, C, H, W)
        mask_batch : np.ndarray, shape (B, num_actions), dtype bool

        Returns
        -------
        (policy_batch, value_batch) where:
            policy_batch: np.ndarray shape (B, num_actions)
            value_batch: np.ndarray shape (B,) in [-1, 1]
        """
        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask_batch, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            logits, values = self.model(obs_t)
            logits = logits.masked_fill(~mask_t, -1e9)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

        return probs, values_np

    def train_step(
        self,
        obs_batch: np.ndarray,
        mask_batch: np.ndarray,
        target_policy: np.ndarray,
        target_value: np.ndarray,
        optimizer: torch.optim.Optimizer,
        target_pins_in_goal: np.ndarray | None = None,
    ) -> dict[str, float]:
        """One training step: CE(policy) + MSE(value) + optional auxiliary loss.

        Parameters
        ----------
        obs_batch : (B, C, H, W)
        mask_batch : (B, num_actions), bool
        target_policy : (B, num_actions) — MCTS visit distributions
        target_value : (B,) — game outcomes in [-1, 1]
        optimizer : torch.optim.Optimizer
        target_pins_in_goal : (B,) or None — normalized pins in goal [0, 1]

        Returns
        -------
        dict with 'policy_loss', 'value_loss', 'total_loss', optionally 'aux_loss'
        """
        self.model.train()

        obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask_batch, dtype=torch.bool, device=self.device)
        target_pi = torch.tensor(
            target_policy, dtype=torch.float32, device=self.device
        )
        target_v = torch.tensor(
            target_value, dtype=torch.float32, device=self.device
        )

        # Forward pass — use auxiliary head if available
        has_aux = (
            self.config.use_auxiliary_head
            and hasattr(self.model, 'forward_with_auxiliary')
        )

        if has_aux:
            logits, values, aux_pred = self.model.forward_with_auxiliary(obs_t)
        else:
            logits, values = self.model(obs_t)
            aux_pred = None

        # Policy loss: cross-entropy between MCTS visit distribution and network policy
        logits = logits.masked_fill(~mask_t, -1e9)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -torch.sum(target_pi * log_probs, dim=-1).mean()

        # Value loss: MSE between predicted value and game outcome
        value_loss = F.mse_loss(values.squeeze(-1), target_v)

        # Combined loss
        total_loss = policy_loss + value_loss

        result = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

        # Auxiliary loss: MSE for pins_in_goal prediction
        if has_aux and target_pins_in_goal is not None and aux_pred is not None:
            target_aux = torch.tensor(
                target_pins_in_goal, dtype=torch.float32, device=self.device
            )
            aux_loss = F.mse_loss(aux_pred.squeeze(-1), target_aux)
            total_loss = total_loss + self.config.auxiliary_loss_weight * aux_loss
            result["aux_loss"] = aux_loss.item()

        result["total_loss"] = total_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        self.model.eval()

        return result

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create Adam optimizer with config settings."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def save_checkpoint(self, path: str | Path, iteration: int = 0, extra: Optional[dict] = None):
        """Save model weights and metadata."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "in_channels": self.config.in_channels,
                "num_actions": self.config.num_actions,
                "num_blocks": self.config.num_blocks,
                "num_filters": self.config.num_filters,
                "architecture": self.config.architecture,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "use_auxiliary_head": self.config.use_auxiliary_head,
            },
            "iteration": iteration,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, str(path))

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load model weights. Returns the full checkpoint dict."""
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        return checkpoint

    def export_onnx(self, path: str | Path) -> None:
        """Export model to ONNX format for optimized inference.

        The exported model takes (obs, mask) and returns (policy_logits, value).
        Use with onnxruntime for ~1.5-2x speedup over PyTorch inference.
        """
        self.model.eval()
        dummy_obs = torch.randn(1, self.config.in_channels, 17, 17, device=self.device)

        torch.onnx.export(
            self.model,
            (dummy_obs,),
            str(path),
            input_names=["obs"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "policy_logits": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
            opset_version=17,
        )
        print(f"  Exported ONNX model to {path}")

    def export_torchscript(self, path: str | Path) -> None:
        """Export model to TorchScript for optimized inference."""
        self.model.eval()
        dummy_obs = torch.randn(1, self.config.in_channels, 17, 17, device=self.device)

        try:
            scripted = torch.jit.trace(self.model, (dummy_obs,))
            scripted.save(str(path))
            print(f"  Exported TorchScript model to {path}")
        except Exception as e:
            print(f"  TorchScript export failed (architecture may not support tracing): {e}")
            print("  Trying torch.jit.script instead...")
            scripted = torch.jit.script(self.model)
            scripted.save(str(path))
            print(f"  Exported TorchScript model to {path}")

    def copy_weights_from(self, other: "AlphaZeroNet"):
        """Copy weights from another network (for best-model tracking)."""
        self.model.load_state_dict(other.model.state_dict())
        self.model.eval()

    def parameter_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
