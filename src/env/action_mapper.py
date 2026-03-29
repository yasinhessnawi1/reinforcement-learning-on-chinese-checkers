"""ActionMapper class for mapping between flat action indices and (pin_id, dest_index) pairs."""

import numpy as np


class ActionMapper:
    """Maps flat action indices to/from (pin_id, destination_index) pairs.

    Formula: action = pin_id * num_cells + dest_index
    For default 10 pins and 121 cells: action ranges from 0 to 1209.
    """

    def __init__(self, num_pins=10, num_cells=121):
        """Initialize the ActionMapper.

        Args:
            num_pins: Number of pins in the game (default: 10).
            num_cells: Number of cells on the board (default: 121).
        """
        self.num_pins = num_pins
        self.num_cells = num_cells
        self.num_actions = num_pins * num_cells

    def encode(self, pin_id, dest_index):
        """Encode a (pin_id, dest_index) pair to a flat action index.

        Args:
            pin_id: The ID of the pin (0 to num_pins - 1).
            dest_index: The destination cell index (0 to num_cells - 1).

        Returns:
            Flat action index (0 to num_actions - 1).
        """
        return pin_id * self.num_cells + dest_index

    def decode(self, action):
        """Decode a flat action index to a (pin_id, dest_index) pair.

        Args:
            action: Flat action index (0 to num_actions - 1).

        Returns:
            Tuple of (pin_id, dest_index).
        """
        pin_id = action // self.num_cells
        dest_index = action % self.num_cells
        return (pin_id, dest_index)

    def build_action_mask(self, legal_moves):
        """Build a boolean mask for legal actions.

        Args:
            legal_moves: Dictionary mapping pin_id to list of legal destination indices.
                        Example: {0: [5, 10, 15], 1: [20, 25]}

        Returns:
            Numpy array of shape (num_actions,) with dtype bool_.
            True for each legal (pin, dest) pair, False otherwise.
        """
        mask = np.zeros(self.num_actions, dtype=np.bool_)

        for pin_id, dest_indices in legal_moves.items():
            for dest_index in dest_indices:
                action = self.encode(pin_id, dest_index)
                mask[action] = True

        return mask
