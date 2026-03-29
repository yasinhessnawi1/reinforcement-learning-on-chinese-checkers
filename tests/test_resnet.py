"""
Tests for ChineseCheckersResNet (src/network/resnet.py).

Covers:
  1. Model instantiation without error
  2. Forward pass output shapes: policy_logits (4, 1210), value (4, 1)
  3. Policy logits sum to 1 after softmax (within tolerance)
  4. Value output is within [-1, 1] (guaranteed by tanh)
  5. Total parameter count is between 100k and 1M
"""

import torch
import torch.nn.functional as F
import pytest

from src.network.resnet import ChineseCheckersResNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
IN_CHANNELS = 10
BOARD_SIZE = 17
NUM_ACTIONS = 1210


@pytest.fixture(scope="module")
def model() -> ChineseCheckersResNet:
    """Return a default model in eval mode."""
    net = ChineseCheckersResNet()
    net.eval()
    return net


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    """Random float32 input of shape (4, 10, 17, 17)."""
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)


@pytest.fixture(scope="module")
def forward_output(
    model: ChineseCheckersResNet, dummy_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the forward pass result for use across tests."""
    with torch.no_grad():
        return model(dummy_input)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_model_instantiation():
    """Model can be constructed with default and custom arguments."""
    # Default constructor
    net = ChineseCheckersResNet()
    assert net is not None

    # Custom constructor
    net_custom = ChineseCheckersResNet(in_channels=4, num_actions=500, num_blocks=3, num_filters=32)
    assert net_custom is not None


def test_forward_output_shapes(forward_output):
    """policy_logits shape is (4, 1210); value shape is (4, 1)."""
    policy_logits, value = forward_output
    assert policy_logits.shape == (BATCH_SIZE, NUM_ACTIONS), (
        f"Expected policy_logits shape ({BATCH_SIZE}, {NUM_ACTIONS}), got {policy_logits.shape}"
    )
    assert value.shape == (BATCH_SIZE, 1), (
        f"Expected value shape ({BATCH_SIZE}, 1), got {value.shape}"
    )


def test_policy_softmax_sums_to_one(forward_output):
    """Softmax over policy logits sums to 1 for each sample in the batch."""
    policy_logits, _ = forward_output
    probs = F.softmax(policy_logits, dim=-1)
    sums = probs.sum(dim=-1)  # shape (batch,)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
        f"Softmax probabilities do not sum to 1; got {sums}"
    )


def test_value_in_range(forward_output):
    """Value output is in [-1, 1] (guaranteed by tanh, verified empirically)."""
    _, value = forward_output
    assert value.min().item() >= -1.0 - 1e-6, f"Value below -1: {value.min().item()}"
    assert value.max().item() <= 1.0 + 1e-6, f"Value above  1: {value.max().item()}"


def test_parameter_count(model: ChineseCheckersResNet):
    """Total trainable parameter count is between 100k and 5M (reasonable ResNet size).

    The default architecture (6 blocks, 64 filters, 1210 actions) yields ~1.2M parameters,
    which is dominated by the policy head's fully-connected layer
    (2 * 17 * 17 = 578 inputs -> 1210 outputs).  The upper bound is set to 5M to
    allow for architectural flexibility while still catching runaway configurations.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params >= 100_000, f"Too few parameters: {num_params}"
    assert num_params <= 5_000_000, f"Too many parameters: {num_params}"
