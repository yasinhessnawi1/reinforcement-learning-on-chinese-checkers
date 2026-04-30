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
        # cell_rot180[i] = j where cell j is at axial (-q, -r) of cell i.
        # Used to rotate action targets (cell indices) for colours whose obs
        # is rotated 180° — without this, the policy head sees inconsistent
        # (rotated obs, raw action) pairs for blue/gray0/purple samples.
        self._cell_rot180: np.ndarray | None = None       # (num_cells,) int
        # Precomputed full 1210-action permutation derived from _cell_rot180,
        # used by rotate_action_distribution as a vectorised fancy-index.
        self._action_rot_perm: np.ndarray | None = None   # (num_pins*num_cells,)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_maps(self, board_wrapper: BoardWrapper) -> None:
        """Populate _valid_mask, _cell_to_grid, and _cell_rot180 from the board."""
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

        # Build the 180° cell-rotation table: cell i (q,r) maps to cell at (-q,-r).
        # _rotate180 flips both axes of the grid, so a cell originally at
        # grid position (row, col) ends up at (gs-1-row, gs-1-col). The cell
        # whose home is that grid position is the one at axial (-q, -r).
        index_of = board_wrapper.board.index_of  # (q, r) -> idx
        n = len(board_wrapper.board.cells)
        cell_rot180 = np.arange(n, dtype=np.int32)
        for idx, cell in enumerate(board_wrapper.board.cells):
            mirror_key = (-cell.q, -cell.r)
            if mirror_key in index_of:
                cell_rot180[idx] = index_of[mirror_key]
        self._cell_rot180 = cell_rot180

        # Precompute the full 1210-action permutation so rotate_action_distribution
        # is a single numpy fancy-index op instead of a Python double loop.
        # action_dst[a] = action index whose value goes INTO position a.
        # For action a = pin*121 + c, the rotated value lives at pin*121 + rot[c],
        # so we need action_src[pin*121 + rot[c]] = pin*121 + c, i.e. inverse.
        num_pins = 10
        num_cells = n
        action_perm = np.empty(num_pins * num_cells, dtype=np.int64)
        for pin in range(num_pins):
            base = pin * num_cells
            for c in range(num_cells):
                # value at action (pin, c) goes TO action (pin, rot[c])
                action_perm[base + cell_rot180[c]] = base + c
        # action_perm[i] = source index whose value lands at i
        self._action_rot_perm = action_perm

    @staticmethod
    def _rotate180(grid: np.ndarray) -> np.ndarray:
        """Return a 180-degree-rotated copy of a 2-D array."""
        return np.flip(np.flip(grid, axis=0), axis=1).copy()

    def _needs_rotation(self, colour: str) -> bool:
        return colour in _ROTATE

    def needs_rotation(self, colour: str) -> bool:
        """Public accessor: True iff this colour's obs gets rotated 180°."""
        return colour in _ROTATE

    def rotate_action_distribution(
        self, dist: np.ndarray, num_pins: int = 10, num_cells: int = 121
    ) -> np.ndarray:
        """Permute a 1210-dim action distribution to match a rotated obs.

        Action layout: action = pin_id * num_cells + cell_index. The pin_id
        is colour-relative (already perspective-correct — pin 0 is "my first
        pin" regardless of colour), so it doesn't change. The cell_index is
        absolute and must be mapped through `_cell_rot180` so that the
        action's spatial meaning matches the rotated observation.

        Vectorised: a single numpy fancy-index using a precomputed permutation
        table. Called many times per MCTS sim, so the python double-loop
        version is a measurable hot spot.

        Parameters
        ----------
        dist : np.ndarray, shape (num_pins * num_cells,)
            Action distribution or boolean mask in raw cell coordinates.
        num_pins, num_cells : int (kept for API compat; rotation is precomputed)

        Returns
        -------
        np.ndarray, same shape and dtype, with cell indices remapped.
        """
        if self._action_rot_perm is None:
            raise RuntimeError(
                "Encoder maps not initialised — call encode() at least once first"
            )
        return dist[self._action_rot_perm]

    def rotate_action(
        self, action: int, num_pins: int = 10, num_cells: int = 121
    ) -> int:
        """Map a single action index through the 180° cell rotation."""
        if self._cell_rot180 is None:
            raise RuntimeError(
                "Encoder maps not initialised — call encode() at least once first"
            )
        pin_id = action // num_cells
        cell = action % num_cells
        return int(pin_id * num_cells + self._cell_rot180[cell])

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
