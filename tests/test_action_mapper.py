"""Tests for the ActionMapper class."""

import numpy as np
import pytest
from src.env.action_mapper import ActionMapper


class TestActionMapperInit:
    """Tests for ActionMapper initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        mapper = ActionMapper()
        assert mapper.num_pins == 10
        assert mapper.num_cells == 121
        assert mapper.num_actions == 1210

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        mapper = ActionMapper(num_pins=5, num_cells=25)
        assert mapper.num_pins == 5
        assert mapper.num_cells == 25
        assert mapper.num_actions == 125

    def test_num_actions_calculation(self):
        """Test that num_actions is correctly calculated."""
        mapper = ActionMapper()
        assert mapper.num_actions == mapper.num_pins * mapper.num_cells


class TestActionMapperEncode:
    """Tests for ActionMapper.encode method."""

    def test_encode_zero(self):
        """Test encoding (0, 0) produces 0."""
        mapper = ActionMapper()
        assert mapper.encode(0, 0) == 0

    def test_encode_last_action(self):
        """Test encoding (9, 120) produces 1209."""
        mapper = ActionMapper()
        assert mapper.encode(9, 120) == 1209

    def test_encode_various_pairs(self):
        """Test encoding various (pin_id, dest_index) pairs."""
        mapper = ActionMapper()
        # Test formula: action = pin_id * 121 + dest_index
        assert mapper.encode(0, 5) == 5
        assert mapper.encode(1, 0) == 121
        assert mapper.encode(1, 5) == 126
        assert mapper.encode(5, 60) == 5 * 121 + 60

    def test_encode_boundary_values(self):
        """Test encoding boundary values."""
        mapper = ActionMapper()
        assert mapper.encode(0, 120) == 120
        assert mapper.encode(9, 0) == 9 * 121


class TestActionMapperDecode:
    """Tests for ActionMapper.decode method."""

    def test_decode_zero(self):
        """Test decoding 0 produces (0, 0)."""
        mapper = ActionMapper()
        assert mapper.decode(0) == (0, 0)

    def test_decode_last_action(self):
        """Test decoding 1209 produces (9, 120)."""
        mapper = ActionMapper()
        assert mapper.decode(1209) == (9, 120)

    def test_decode_various_actions(self):
        """Test decoding various actions."""
        mapper = ActionMapper()
        assert mapper.decode(5) == (0, 5)
        assert mapper.decode(121) == (1, 0)
        assert mapper.decode(126) == (1, 5)
        assert mapper.decode(5 * 121 + 60) == (5, 60)

    def test_decode_boundary_values(self):
        """Test decoding boundary values."""
        mapper = ActionMapper()
        assert mapper.decode(120) == (0, 120)
        assert mapper.decode(9 * 121) == (9, 0)


class TestActionMapperRoundtrip:
    """Tests for encode/decode roundtrips."""

    def test_encode_decode_roundtrip_zero(self):
        """Test encode/decode roundtrip for (0, 0)."""
        mapper = ActionMapper()
        pin_id, dest_index = 0, 0
        action = mapper.encode(pin_id, dest_index)
        decoded_pin, decoded_dest = mapper.decode(action)
        assert decoded_pin == pin_id
        assert decoded_dest == dest_index

    def test_encode_decode_roundtrip_max(self):
        """Test encode/decode roundtrip for (9, 120)."""
        mapper = ActionMapper()
        pin_id, dest_index = 9, 120
        action = mapper.encode(pin_id, dest_index)
        decoded_pin, decoded_dest = mapper.decode(action)
        assert decoded_pin == pin_id
        assert decoded_dest == dest_index

    def test_encode_decode_roundtrip_all_actions(self):
        """Test encode/decode roundtrip for all possible actions."""
        mapper = ActionMapper()
        for action in range(mapper.num_actions):
            pin_id, dest_index = mapper.decode(action)
            encoded_action = mapper.encode(pin_id, dest_index)
            assert encoded_action == action


class TestActionMapperBuildActionMask:
    """Tests for ActionMapper.build_action_mask method."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        mapper = ActionMapper()
        legal_moves = {0: [5, 10], 1: [20]}
        mask = mapper.build_action_mask(legal_moves)
        assert mask.shape == (mapper.num_actions,)

    def test_mask_dtype(self):
        """Test that mask has correct dtype."""
        mapper = ActionMapper()
        legal_moves = {0: [5, 10]}
        mask = mapper.build_action_mask(legal_moves)
        assert mask.dtype == np.bool_

    def test_mask_all_false_empty_legal_moves(self):
        """Test that empty legal moves produces all-False mask."""
        mapper = ActionMapper()
        mask = mapper.build_action_mask({})
        assert np.all(~mask)
        assert mask.sum() == 0

    def test_mask_single_legal_move(self):
        """Test mask with a single legal move."""
        mapper = ActionMapper()
        legal_moves = {0: [5]}
        mask = mapper.build_action_mask(legal_moves)
        assert mask[mapper.encode(0, 5)] is True or mask[mapper.encode(0, 5)] == True
        assert mask.sum() == 1

    def test_mask_multiple_moves_single_pin(self):
        """Test mask with multiple moves from a single pin."""
        mapper = ActionMapper()
        legal_moves = {0: [5, 10, 15]}
        mask = mapper.build_action_mask(legal_moves)
        assert mask[mapper.encode(0, 5)] is True or mask[mapper.encode(0, 5)] == True
        assert mask[mapper.encode(0, 10)] is True or mask[mapper.encode(0, 10)] == True
        assert mask[mapper.encode(0, 15)] is True or mask[mapper.encode(0, 15)] == True
        assert mask.sum() == 3

    def test_mask_multiple_pins_multiple_moves(self):
        """Test mask with multiple moves from multiple pins."""
        mapper = ActionMapper()
        legal_moves = {0: [5, 10], 1: [20, 25, 30], 5: [100]}
        mask = mapper.build_action_mask(legal_moves)
        expected_true = 6
        assert mask.sum() == expected_true
        # Check specific True values
        assert mask[mapper.encode(0, 5)]
        assert mask[mapper.encode(0, 10)]
        assert mask[mapper.encode(1, 20)]
        assert mask[mapper.encode(1, 25)]
        assert mask[mapper.encode(1, 30)]
        assert mask[mapper.encode(5, 100)]

    def test_mask_no_false_positives(self):
        """Test that only legal moves are marked as True."""
        mapper = ActionMapper()
        legal_moves = {0: [5, 10], 2: [50]}
        mask = mapper.build_action_mask(legal_moves)
        # Check some positions that should be False
        assert not mask[mapper.encode(0, 0)]
        assert not mask[mapper.encode(0, 15)]
        assert not mask[mapper.encode(1, 5)]
        assert not mask[mapper.encode(2, 0)]
        assert not mask[mapper.encode(2, 51)]

    def test_mask_all_pins_all_cells(self):
        """Test mask with all pins and all cells as legal."""
        mapper = ActionMapper(num_pins=3, num_cells=10)
        legal_moves = {i: list(range(10)) for i in range(3)}
        mask = mapper.build_action_mask(legal_moves)
        assert mask.sum() == 30
        assert np.all(mask)

    def test_mask_specific_example(self):
        """Test mask with a specific realistic example."""
        mapper = ActionMapper()
        legal_moves = {
            0: [5, 10],
            3: [25, 30, 35],
            7: [100]
        }
        mask = mapper.build_action_mask(legal_moves)
        # Should have exactly 6 True values
        assert mask.sum() == 6
        # All True values should correspond to legal moves
        true_indices = np.where(mask)[0]
        for action_idx in true_indices:
            pin_id, dest_index = mapper.decode(action_idx)
            assert pin_id in legal_moves
            assert dest_index in legal_moves[pin_id]


class TestActionMapperCustomDimensions:
    """Tests for ActionMapper with custom dimensions."""

    def test_custom_dimensions_encode_decode(self):
        """Test encode/decode with custom dimensions."""
        mapper = ActionMapper(num_pins=5, num_cells=25)
        assert mapper.encode(0, 0) == 0
        assert mapper.encode(4, 24) == 124
        assert mapper.decode(0) == (0, 0)
        assert mapper.decode(124) == (4, 24)

    def test_custom_dimensions_mask(self):
        """Test mask building with custom dimensions."""
        mapper = ActionMapper(num_pins=3, num_cells=10)
        legal_moves = {0: [1, 2], 2: [5]}
        mask = mapper.build_action_mask(legal_moves)
        assert mask.shape == (30,)
        assert mask.sum() == 3
