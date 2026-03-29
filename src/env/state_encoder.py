"""
StateEncoder: converts a BoardWrapper snapshot into a multi-channel float32
grid observation suitable for RL.
"""
import numpy as np
from src.env.board_wrapper import BoardWrapper

# Colours that do NOT need rotation (home triangles at high r values).
_NO_ROTATE = {'red', 'lawn green', 'yellow'}
# Colours that DO need 180-degree rotation (home triangles at low r values).
_ROTATE = {'blue', 'gray0', 'purple'}


class StateEncoder:
    """
    Encodes a board state as a (num_channels, grid_size, grid_size) float32
    array.

    Parameters
    ----------
    grid_size : int
        Side length of the 2-D grid.  Default 17 (covering axial coords
        q, r in [-8, 8] with offset 8).
    num_channels : int
        Total number of channels in the output tensor.  Default 10.
    """

    def __init__(self, grid_size: int = 17, num_channels: int = 10):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self._offset = 8  # maps axial coord in [-8,8] -> [0,16]

        # Built lazily on first encode() call so that no HexBoard is needed at
        # construction time.
        self._valid_mask: np.ndarray | None = None        # (grid_size, grid_size)
        self._cell_to_grid: dict | None = None            # cell_idx -> (row, col)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_maps(self, board_wrapper: BoardWrapper) -> None:
        """Populate _valid_mask and _cell_to_grid from the board."""
        offset = self._offset
        gs = self.grid_size

        valid_mask = np.zeros((gs, gs), dtype=np.float32)
        cell_to_grid: dict[int, tuple[int, int]] = {}

        for idx, cell in enumerate(board_wrapper.board.cells):
            row = cell.r + offset   # r-axis -> row
            col = cell.q + offset   # q-axis -> col
            if 0 <= row < gs and 0 <= col < gs:
                valid_mask[row, col] = 1.0
                cell_to_grid[idx] = (row, col)

        self._valid_mask = valid_mask
        self._cell_to_grid = cell_to_grid

    @staticmethod
    def _rotate180(grid: np.ndarray) -> np.ndarray:
        """Return a 180-degree-rotated copy of a 2-D array."""
        return np.flip(np.flip(grid, axis=0), axis=1).copy()

    def _needs_rotation(self, colour: str) -> bool:
        return colour in _ROTATE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        board_wrapper: BoardWrapper,
        current_colour: str,
        turn_order: list,
    ) -> np.ndarray:
        """
        Build and return the observation tensor.

        Parameters
        ----------
        board_wrapper : BoardWrapper
            Current game state.
        current_colour : str
            The colour whose turn it is (perspective for ch 0, 2, 3).
        turn_order : list[str]
            Ordered list of active colours.  The opponent is inferred as
            any colour in *turn_order* that is not *current_colour*.  For
            2-player games this is unambiguous; for more players only the
            immediate next opponent is placed in channel 1.

        Returns
        -------
        np.ndarray, shape (num_channels, grid_size, grid_size), dtype float32
        """
        # Lazily build board-structure helpers.
        if self._valid_mask is None or self._cell_to_grid is None:
            self._build_maps(board_wrapper)

        gs = self.grid_size
        obs = np.zeros((self.num_channels, gs, gs), dtype=np.float32)

        # --- Channel 4: valid board positions ---
        obs[4] = self._valid_mask.copy()

        # --- Identify opponent colour ---
        # For a 2-player game, the opponent is the other colour in turn_order.
        # We use the first non-current colour in turn_order.
        opponent_colour: str | None = None
        for c in turn_order:
            if c != current_colour:
                opponent_colour = c
                break

        cell_to_grid = self._cell_to_grid

        # --- Channel 0: current player's pieces ---
        for pin in board_wrapper.pins[current_colour]:
            if pin.axialindex in cell_to_grid:
                r, c = cell_to_grid[pin.axialindex]
                obs[0, r, c] = 1.0

        # --- Channel 1: opponent's pieces ---
        if opponent_colour is not None and opponent_colour in board_wrapper.pins:
            for pin in board_wrapper.pins[opponent_colour]:
                if pin.axialindex in cell_to_grid:
                    r, c = cell_to_grid[pin.axialindex]
                    obs[1, r, c] = 1.0

        # --- Channel 2: current player's goal triangle ---
        for idx in board_wrapper.get_goal_indices(current_colour):
            if idx in cell_to_grid:
                r, c = cell_to_grid[idx]
                obs[2, r, c] = 1.0

        # --- Channel 3: current player's home triangle ---
        for idx in board_wrapper.get_home_indices(current_colour):
            if idx in cell_to_grid:
                r, c = cell_to_grid[idx]
                obs[3, r, c] = 1.0

        # --- Channels 5-9: reserved / zeroed ---

        # --- Apply 180-degree rotation for perspective normalisation ---
        if self._needs_rotation(current_colour):
            for ch in range(self.num_channels):
                obs[ch] = self._rotate180(obs[ch])

        return obs
