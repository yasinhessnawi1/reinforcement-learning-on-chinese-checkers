"""
ResNet-based neural network for the Chinese Checkers RL agent.

Architecture: ChineseCheckersResNet
  - Shared body: initial conv + N residual blocks
  - Policy head: outputs raw logits over the action space
  - Value head: outputs a scalar value estimate in [-1, 1]
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """A single residual block with two 3x3 convolutions and a skip connection."""

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu2(out + residual)
        return out


class ChineseCheckersResNet(nn.Module):
    """
    ResNet-based network for Chinese Checkers.

    Args:
        in_channels:  Number of input feature planes (default 10).
        num_actions:  Size of the policy output (default 1210).
        num_blocks:   Number of residual blocks in the shared body (default 6).
        num_filters:  Number of convolutional filters throughout the body (default 64).

    Forward input:
        x: (batch, in_channels, 17, 17) float32 tensor

    Forward output:
        policy_logits: (batch, num_actions) — raw logits, no softmax applied
        value:         (batch, 1)          — scalar estimate in [-1, 1]
    """

    BOARD_SIZE: int = 17  # spatial dimension of the board representation

    def __init__(
        self,
        in_channels: int = 10,
        num_actions: int = 1210,
        num_blocks: int = 6,
        num_filters: int = 64,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Shared body
        # ------------------------------------------------------------------
        self.input_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        self.input_relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_blocks)])

        # ------------------------------------------------------------------
        # Policy head
        # ------------------------------------------------------------------
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(2 * self.BOARD_SIZE * self.BOARD_SIZE, num_actions)

        # ------------------------------------------------------------------
        # Value head
        # ------------------------------------------------------------------
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu1 = nn.ReLU(inplace=True)
        self.value_fc1 = nn.Linear(1 * self.BOARD_SIZE * self.BOARD_SIZE, 256)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared body
        out = self.input_relu(self.input_bn(self.input_conv(x)))
        out = self.res_blocks(out)

        # Policy head
        p = self.policy_relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_relu1(self.value_bn(self.value_conv(out)))
        v = v.flatten(start_dim=1)
        v = self.value_relu2(self.value_fc1(v))
        value = self.value_tanh(self.value_fc2(v))

        return policy_logits, value
