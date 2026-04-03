"""
BoardWrapper: thin adapter around HexBoard/Pin for RL environment use (no TCP).
"""
import os
import sys
import io

# Add the base game directory to sys.path so we can import HexBoard and Pin.
_BASE_GAME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal')
)
if _BASE_GAME_DIR not in sys.path:
    sys.path.insert(0, _BASE_GAME_DIR)

from checkers_board import HexBoard        # noqa: E402
from checkers_pins import Pin              # noqa: E402

COLOUR_OPPOSITES = {
    'red': 'blue',
    'blue': 'red',
    'lawn green': 'gray0',
    'gray0': 'lawn green',
    'yellow': 'purple',
    'purple': 'yellow',
}


class BoardWrapper:
    """
    Thin adapter that wraps HexBoard and Pin objects for RL use.

    Parameters
    ----------
    colours : list[str]
        The colours that are active in this game, e.g. ['red', 'blue'].
    """

    def __init__(self, colours: list):
        self.colours = list(colours)
        self.board = HexBoard()

        # pins[colour] = list of Pin objects (length 10)
        self.pins: dict[str, list[Pin]] = {}
        for colour in self.colours:
            self._place_colour(colour)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_colour(self, colour: str):
        """Place 10 pins for *colour* on its home triangle."""
        home_indices = self.board.axial_of_colour(colour)
        pins = []
        for pin_id, idx in enumerate(home_indices):
            pin = Pin(self.board, idx, pin_id, color=colour)
            pins.append(pin)
        self.pins[colour] = pins

    def _pin_by_id(self, colour: str, pin_id: int) -> Pin:
        for pin in self.pins[colour]:
            if pin.id == pin_id:
                return pin
        raise ValueError(f"No pin with id={pin_id} for colour '{colour}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pieces(self, colour: str) -> list:
        """
        Return a list of dicts, one per pin: ``{'id': int, 'pos': int}``.
        """
        return [{'id': p.id, 'pos': p.axialindex} for p in self.pins[colour]]

    def get_legal_moves(self, colour: str) -> dict:
        """
        Return ``{pin_id: [dest_indices, ...]}``.
        Only includes pins that have at least one legal move.
        """
        moves = {}
        for pin in self.pins[colour]:
            dests = pin.getPossibleMoves()
            if dests:
                moves[pin.id] = dests
        return moves

    def apply_move(self, colour: str, pin_id: int, dest_index: int) -> bool:
        """
        Move the pin identified by *pin_id* (for *colour*) to *dest_index*.
        Returns True on success, False otherwise.
        """
        pin = self._pin_by_id(colour, pin_id)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = pin.placePin(dest_index)
        finally:
            sys.stdout = old_stdout
        return result

    def get_goal_indices(self, colour: str) -> list:
        """
        Return the list of cell indices that form the goal triangle for
        *colour* (i.e. the home triangle of the opposite colour).
        """
        opposite = COLOUR_OPPOSITES[colour]
        return self.board.axial_of_colour(opposite)

    def get_home_indices(self, colour: str) -> list:
        """Return the list of cell indices that form the home triangle for *colour*."""
        return self.board.axial_of_colour(colour)

    def check_win(self, colour: str) -> bool:
        """
        Return True if all 10 pins of *colour* are currently occupying the
        goal triangle (opposite colour's home cells).
        """
        goal = set(self.get_goal_indices(colour))
        for pin in self.pins[colour]:
            if pin.axialindex not in goal:
                return False
        return True

    def check_draw(self, colour: str) -> bool:
        """Return True if *colour* has no legal moves at all."""
        for pin in self.pins[colour]:
            if pin.getPossibleMoves():
                return False
        return True

    def axial_distance(self, idx_a: int, idx_b: int) -> int:
        """
        Return the hex (axial) distance between two cells given by their
        board indices.
        """
        cell_a = self.board.cells[idx_a]
        cell_b = self.board.cells[idx_b]
        dq = cell_a.q - cell_b.q
        dr = cell_a.r - cell_b.r
        ds = (-cell_a.q - cell_a.r) - (-cell_b.q - cell_b.r)
        return (abs(dq) + abs(dr) + abs(ds)) // 2

    def total_distance_to_goal(self, colour: str) -> int:
        """
        Return the sum of, for each pin *not* already in the goal triangle,
        the minimum axial distance from that pin to any goal cell.
        Pins already in the goal contribute 0.
        """
        goal_indices = self.get_goal_indices(colour)
        total = 0
        for pin in self.pins[colour]:
            if pin.axialindex not in set(goal_indices):
                min_dist = min(self.axial_distance(pin.axialindex, g) for g in goal_indices)
                total += min_dist
        return total

    def pins_in_goal(self, colour: str) -> int:
        """Return the number of *colour* pins currently in the goal triangle."""
        goal = set(self.get_goal_indices(colour))
        return sum(1 for pin in self.pins[colour] if pin.axialindex in goal)

    @staticmethod
    def _clone_board(board):
        """Create a fast copy of HexBoard: reuse geometry, copy only occupied flags.

        ~100x faster than copy.deepcopy by avoiding re-creation of all the
        coordinate/pixel/row data structures.
        """
        from checkers_board import BoardPosition, HexBoard
        new_board = object.__new__(HexBoard)
        new_board.R = board.R
        new_board.hole_radius = board.hole_radius
        new_board.spacing = board.spacing
        new_board.colour_opposites = board.colour_opposites  # immutable dict of strings

        # Shallow-copy cells list but create new BoardPosition objects
        # that share the immutable fields and copy only `occupied`.
        new_cells = []
        for cell in board.cells:
            new_cell = object.__new__(BoardPosition)
            new_cell.q = cell.q
            new_cell.r = cell.r
            new_cell.x = cell.x
            new_cell.y = cell.y
            new_cell.postype = cell.postype
            new_cell.occupied = cell.occupied  # the only mutable field
            new_cells.append(new_cell)
        new_board.cells = new_cells

        # index_of maps (q,r)->int — keys are immutable tuples, values are ints.
        # Safe to share since it's never mutated during gameplay.
        new_board.index_of = board.index_of

        # cartesian and _rows are display-only, never mutated during gameplay.
        new_board.cartesian = board.cartesian
        new_board._rows = board._rows

        return new_board

    def clone(self):
        """
        Return a fast clone of this BoardWrapper.

        Uses _clone_board() to copy only occupied flags instead of deepcopy.
        """
        new_wrapper = object.__new__(BoardWrapper)
        new_wrapper.colours = list(self.colours)

        new_wrapper.board = self._clone_board(self.board)

        # Rebuild Pin objects pointing at the new board.
        new_wrapper.pins = {}
        for colour in self.colours:
            new_pins = []
            for pin in self.pins[colour]:
                new_pin = Pin.__new__(Pin)
                new_pin.board = new_wrapper.board
                new_pin.axialindex = pin.axialindex
                new_pin.id = pin.id
                new_pin.color = pin.color
                new_pins.append(new_pin)
            new_wrapper.pins[colour] = new_pins

        return new_wrapper
