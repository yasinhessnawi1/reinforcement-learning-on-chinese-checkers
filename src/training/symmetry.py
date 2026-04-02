"""
symmetry.py — Reflection symmetry augmentation for Chinese Checkers.

The board has left-right (q-axis) symmetry: reflecting q → -q gives an
equivalent board position.  We exploit this to double effective training
data by augmenting each (obs, action) pair with its mirror image.

Usage
-----
    from src.training.symmetry import ReflectionSymmetry
    sym = ReflectionSymmetry()
    obs_mirror, action_mirror = sym.reflect(obs, action)

How it works
------------
- Observation: flip channel data left-right (axis=-1, i.e. col axis).
  The 17x17 grid maps q in [-8, 8] to col in [0, 16].  Flipping cols
  is exactly the q → -q reflection.
- Action: action = pin_id * 121 + dest_index.  Both pin_id and dest_index
  must map to their reflected counterparts.  We precompute a lookup table
  cell_reflect[dest_index] and pin_reflect[pin_id] at construction time
  using the actual HexBoard geometry.
- Pin IDs: pins are placed in home-triangle order.  The reflection of
  pin_id i is the pin whose home cell is the q-mirror of pin i's home cell.
"""

import os
import sys
import numpy as np

_BASE_GAME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal')
)
if _BASE_GAME_DIR not in sys.path:
    sys.path.insert(0, _BASE_GAME_DIR)

from checkers_board import HexBoard  # noqa: E402


def _build_cell_reflection(board: HexBoard) -> np.ndarray:
    """Build a lookup table: cell_reflect[i] = index of the cell at (-q, r).

    For cells that have no mirror (shouldn't happen on a symmetric board)
    the entry is -1.
    """
    n = len(board.cells)
    reflect = np.full(n, -1, dtype=np.int32)
    index_of = board.index_of  # dict {(q, r): idx}
    for idx, cell in enumerate(board.cells):
        mirror_key = (-cell.q, cell.r)
        if mirror_key in index_of:
            reflect[idx] = index_of[mirror_key]
    return reflect


def _build_pin_reflection(board: HexBoard, colour: str, cell_reflect: np.ndarray) -> np.ndarray:
    """Build a lookup table: pin_reflect[i] = pin_id of the reflected pin.

    Pins are ordered by home-cell index (same order BoardWrapper uses when
    calling board.axial_of_colour).  The reflected pin is the one whose
    home cell is the mirror of pin i's home cell.
    """
    home_indices = board.axial_of_colour(colour)
    n = len(home_indices)
    # Map home cell index -> pin_id
    home_to_pin = {idx: pid for pid, idx in enumerate(home_indices)}
    reflect = np.full(n, -1, dtype=np.int32)
    for pin_id, home_idx in enumerate(home_indices):
        mirror_cell = cell_reflect[home_idx]
        if mirror_cell in home_to_pin:
            reflect[pin_id] = home_to_pin[mirror_cell]
    return reflect


class ReflectionSymmetry:
    """Precomputes the left-right reflection lookup tables and applies them.

    Parameters
    ----------
    agent_colour : str
        The colour the agent plays (default: 'red').
    num_cells : int
        Number of board cells (default: 121).
    num_pins : int
        Number of pins per player (default: 10).
    grid_size : int
        Side length of the observation grid (default: 17).
    num_channels : int
        Number of observation channels (default: 10).
    """

    def __init__(
        self,
        agent_colour: str = 'red',
        num_cells: int = 121,
        num_pins: int = 10,
        grid_size: int = 17,
        num_channels: int = 10,
    ):
        self.num_cells = num_cells
        self.num_pins = num_pins
        self.grid_size = grid_size
        self.num_channels = num_channels

        board = HexBoard()
        self._cell_reflect = _build_cell_reflection(board)
        self._pin_reflect = _build_pin_reflection(board, agent_colour, self._cell_reflect)

        # Precompute full action reflection table: shape (num_pins * num_cells,)
        self._action_reflect = self._build_action_reflect()

    def _build_action_reflect(self) -> np.ndarray:
        """Build action_reflect[a] = reflected action of a."""
        num_actions = self.num_pins * self.num_cells
        table = np.full(num_actions, -1, dtype=np.int32)
        for action in range(num_actions):
            pin_id = action // self.num_cells
            dest = action % self.num_cells
            rpin = self._pin_reflect[pin_id]
            rdest = self._cell_reflect[dest]
            if rpin >= 0 and rdest >= 0:
                table[action] = rpin * self.num_cells + rdest
        return table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return the left-right mirror of an observation tensor.

        Parameters
        ----------
        obs : np.ndarray, shape (num_channels, grid_size, grid_size)

        Returns
        -------
        np.ndarray, same shape, with columns flipped.
        """
        return np.flip(obs, axis=-1).copy()

    def reflect_action(self, action: int) -> int:
        """Return the reflected action index.

        Parameters
        ----------
        action : int in [0, num_pins * num_cells)

        Returns
        -------
        int — reflected action, or -1 if no valid reflection exists.
        """
        return int(self._action_reflect[action])

    def reflect_action_mask(self, mask: np.ndarray) -> np.ndarray:
        """Return the reflected action mask.

        Parameters
        ----------
        mask : np.ndarray, shape (num_pins * num_cells,), dtype bool

        Returns
        -------
        np.ndarray, same shape — reflected boolean mask.
        """
        reflected = np.zeros_like(mask)
        valid = self._action_reflect >= 0
        reflected[self._action_reflect[valid]] = mask[valid]
        return reflected

    def augment(
        self, obs: np.ndarray, action: int
    ) -> tuple[np.ndarray, int]:
        """Return (reflected_obs, reflected_action).

        Convenience wrapper combining reflect_obs + reflect_action.
        """
        return self.reflect_obs(obs), self.reflect_action(action)

    def augment_batch(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment a batch of (obs, action) pairs.

        Parameters
        ----------
        obs_batch : np.ndarray, shape (N, num_channels, grid_size, grid_size)
        action_batch : np.ndarray, shape (N,), dtype int

        Returns
        -------
        (reflected_obs_batch, reflected_action_batch) — same shapes.
        """
        reflected_obs = np.flip(obs_batch, axis=-1).copy()
        reflected_actions = self._action_reflect[action_batch]
        return reflected_obs, reflected_actions
