"""Custom SB3 feature extractor wrapping ChineseCheckersResNet."""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.network.resnet import ResBlock


class ResNetFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor using a ResNet shared body with
    a 1x1 conv reduction (like AlphaZero) to keep parameter count low.

    Pipeline: input → Conv3x3 → N ResBlocks → Conv1x1(reduce) → BN → ReLU → flatten

    Args:
        observation_space: Gymnasium Box space, expected shape (10, 17, 17).
        num_blocks: Number of residual blocks (default 6).
        num_filters: Number of convolutional filters (default 64).
        reduce_channels: 1x1 conv output channels before flatten (default 4).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        num_blocks: int = 6,
        num_filters: int = 64,
        reduce_channels: int = 4,
    ) -> None:
        in_channels = observation_space.shape[0]
        board_size = observation_space.shape[1]  # 17
        features_dim = reduce_channels * board_size * board_size  # 4*17*17 = 1156

        super().__init__(observation_space, features_dim=features_dim)

        # Shared ResNet body
        self.input_conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        self.input_relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_blocks)])

        # Spatial reduction: 64 channels → reduce_channels via 1x1 conv
        self.reduce_conv = nn.Conv2d(num_filters, reduce_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(reduce_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.input_relu(self.input_bn(self.input_conv(observations)))
        x = self.res_blocks(x)
        x = self.reduce_relu(self.reduce_bn(self.reduce_conv(x)))
        return x.flatten(start_dim=1)
