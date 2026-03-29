"""
Tests for src/env/state_encoder.py
"""
import sys
import os

# Ensure the repo root is importable regardless of where pytest is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest
from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def two_player_wrapper():
    """A fresh BoardWrapper with red and blue in starting positions."""
    return BoardWrapper(['red', 'blue'])


@pytest.fixture(scope='module')
def encoder():
    """Default StateEncoder (10 channels, 17x17 grid)."""
    return StateEncoder()


@pytest.fixture(scope='module')
def obs_red(two_player_wrapper, encoder):
    """Encoded observation from red's perspective."""
    return encoder.encode(two_player_wrapper, 'red', ['red', 'blue'])


@pytest.fixture(scope='module')
def obs_blue(two_player_wrapper, encoder):
    """Encoded observation from blue's perspective."""
    return encoder.encode(two_player_wrapper, 'blue', ['red', 'blue'])


# ---------------------------------------------------------------------------
# 1. Output shape and dtype
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_shape(self, obs_red):
        assert obs_red.shape == (10, 17, 17), (
            f"Expected (10, 17, 17), got {obs_red.shape}"
        )

    def test_dtype(self, obs_red):
        assert obs_red.dtype == np.float32, (
            f"Expected float32, got {obs_red.dtype}"
        )

    def test_shape_blue(self, obs_blue):
        assert obs_blue.shape == (10, 17, 17)

    def test_dtype_blue(self, obs_blue):
        assert obs_blue.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Valid mask channel (ch 4) has exactly 121 ones
# ---------------------------------------------------------------------------

class TestValidMask:
    def test_valid_mask_count_red(self, obs_red):
        count = int(obs_red[4].sum())
        assert count == 121, f"Expected 121 valid cells, got {count}"

    def test_valid_mask_count_blue(self, obs_blue):
        count = int(obs_blue[4].sum())
        assert count == 121, f"Expected 121 valid cells, got {count}"

    def test_valid_mask_values_binary(self, obs_red):
        """All entries in the valid mask should be 0.0 or 1.0."""
        unique = set(obs_red[4].flatten().tolist())
        assert unique.issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# 3. Own pieces channel (ch 0) has exactly 10 ones
# ---------------------------------------------------------------------------

class TestOwnPieces:
    def test_own_piece_count_red(self, obs_red):
        count = int(obs_red[0].sum())
        assert count == 10, f"Expected 10 own pieces for red, got {count}"

    def test_own_piece_count_blue(self, obs_blue):
        count = int(obs_blue[0].sum())
        assert count == 10, f"Expected 10 own pieces for blue, got {count}"


# ---------------------------------------------------------------------------
# 4. Opponent pieces channel (ch 1) has exactly 10 ones
# ---------------------------------------------------------------------------

class TestOpponentPieces:
    def test_opponent_piece_count_red(self, obs_red):
        count = int(obs_red[1].sum())
        assert count == 10, f"Expected 10 opponent pieces for red, got {count}"

    def test_opponent_piece_count_blue(self, obs_blue):
        count = int(obs_blue[1].sum())
        assert count == 10, f"Expected 10 opponent pieces for blue, got {count}"


# ---------------------------------------------------------------------------
# 5. Goal channel (ch 2) has exactly 10 ones
# ---------------------------------------------------------------------------

class TestGoalChannel:
    def test_goal_count_red(self, obs_red):
        count = int(obs_red[2].sum())
        assert count == 10, f"Expected 10 goal cells for red, got {count}"

    def test_goal_count_blue(self, obs_blue):
        count = int(obs_blue[2].sum())
        assert count == 10, f"Expected 10 goal cells for blue, got {count}"


# ---------------------------------------------------------------------------
# 6. Home channel (ch 3) has exactly 10 ones
# ---------------------------------------------------------------------------

class TestHomeChannel:
    def test_home_count_red(self, obs_red):
        count = int(obs_red[3].sum())
        assert count == 10, f"Expected 10 home cells for red, got {count}"

    def test_home_count_blue(self, obs_blue):
        count = int(obs_blue[3].sum())
        assert count == 10, f"Expected 10 home cells for blue, got {count}"


# ---------------------------------------------------------------------------
# 7. Rotation symmetry
# ---------------------------------------------------------------------------

class TestRotationSymmetry:
    """
    Red has no rotation applied; blue has 180-degree rotation applied.

    Red's own pieces (ch 0) sit at red's home (high-r positions).
    Blue's own pieces (ch 0) are encoded from blue's perspective, which means
    blue's home (low-r positions) has been rotated 180 degrees to appear
    at high-r positions in the output grid.

    Consequently:
      - Red's ch 0 positions, when rotated 180 degrees via (16-r, 16-c),
        should equal Blue's ch 1 positions (red pieces seen from blue's view).
      - Red's ch 1 positions, rotated 180 degrees, should equal Blue's ch 0.
    """

    def _channel_positions(self, obs: np.ndarray, channel: int) -> set:
        """Return the set of (row, col) grid positions that are 1.0 in channel."""
        rows, cols = np.where(obs[channel] == 1.0)
        return set(zip(rows.tolist(), cols.tolist()))

    def _rotate_positions(self, positions: set, size: int = 17) -> set:
        """Apply 180-degree rotation mapping: (r, c) -> (size-1-r, size-1-c)."""
        return {(size - 1 - r, size - 1 - c) for (r, c) in positions}

    def test_red_own_rotated_equals_blue_opponent(self, obs_red, obs_blue):
        """
        Red's own pieces, rotated 180°, should match Blue's opponent channel
        (because blue's encoder rotated blue's view by 180°, making red's
        pieces appear at the rotated positions).
        """
        red_own = self._channel_positions(obs_red, 0)
        blue_opp = self._channel_positions(obs_blue, 1)
        red_own_rotated = self._rotate_positions(red_own)
        assert red_own_rotated == blue_opp, (
            f"Red own rotated: {sorted(red_own_rotated)}\n"
            f"Blue opponent:   {sorted(blue_opp)}"
        )

    def test_blue_own_rotated_equals_red_opponent(self, obs_red, obs_blue):
        """
        Blue's own pieces (already in rotated view), when un-rotated (same
        operation), should match Red's opponent channel.
        """
        blue_own = self._channel_positions(obs_blue, 0)
        red_opp = self._channel_positions(obs_red, 1)
        blue_own_rotated = self._rotate_positions(blue_own)
        assert blue_own_rotated == red_opp, (
            f"Blue own rotated: {sorted(blue_own_rotated)}\n"
            f"Red opponent:     {sorted(red_opp)}"
        )

    def test_valid_mask_invariant_under_rotation(self, obs_red, obs_blue):
        """
        The valid mask must look the same from both perspectives because the
        board is point-symmetric under 180-degree rotation.
        """
        red_mask = self._channel_positions(obs_red, 4)
        blue_mask = self._channel_positions(obs_blue, 4)
        red_mask_rotated = self._rotate_positions(red_mask)
        assert red_mask_rotated == blue_mask, (
            "Valid mask is not symmetric under 180-degree rotation."
        )


# ---------------------------------------------------------------------------
# 8. No overlap between own and opponent piece channels
# ---------------------------------------------------------------------------

class TestNoOverlap:
    def test_no_overlap_red(self, obs_red):
        """Own and opponent piece channels must not share any active cell."""
        overlap = (obs_red[0] * obs_red[1]).sum()
        assert overlap == 0.0, (
            f"Own and opponent channels overlap at {int(overlap)} cell(s) for red"
        )

    def test_no_overlap_blue(self, obs_blue):
        overlap = (obs_blue[0] * obs_blue[1]).sum()
        assert overlap == 0.0, (
            f"Own and opponent channels overlap at {int(overlap)} cell(s) for blue"
        )

    def test_reserved_channels_zero(self, obs_red):
        """Channels 5-9 should be all zeros."""
        for ch in range(5, 10):
            total = obs_red[ch].sum()
            assert total == 0.0, f"Reserved channel {ch} is not zero (sum={total})"
