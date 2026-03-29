"""
ChineseCheckersEnv: single-agent Gymnasium environment for 2-player Chinese Checkers.

The agent always plays as 'red'.  After the agent's step() the opponent ('blue')
moves automatically via an internal policy (default: random legal move).

Compatible with sb3-contrib MaskablePPO through the action_masks() method.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder
from src.env.action_mapper import ActionMapper
from src.training.reward import compute_step_reward

# Axial directions for hop detection
_HEX_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]


class ChineseCheckersEnv(gym.Env):
    """Single-agent Gymnasium environment for 2-player Chinese Checkers.

    Parameters
    ----------
    opponent_policy : callable or None
        callable(board_wrapper, colour) -> (pin_id, dest_index).
        If None a random-legal-move policy is used.
    max_steps : int
        Episode truncation limit (default 200).
    render_mode : str or None
        Supported: 'ansi'.
    """

    metadata = {"render_modes": ["ansi"]}

    # Fixed identities
    _AGENT_COLOUR = "red"
    _OPPONENT_COLOUR = "blue"
    _TURN_ORDER = ["red", "blue"]

    def __init__(self, opponent_policy=None, max_steps=200, render_mode=None):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"], (
            f"Unsupported render_mode: {render_mode}"
        )
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10, 17, 17),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(1210)

        # Sub-components (created once; board state reset in reset())
        self._encoder = StateEncoder(grid_size=17, num_channels=10)
        self._mapper = ActionMapper(num_pins=10, num_cells=121)

        # Opponent policy
        self._opponent_policy = opponent_policy if opponent_policy is not None else self._random_opponent

        # Runtime state (initialised in reset())
        self._board: BoardWrapper | None = None
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Reset the environment to a new game.

        Returns
        -------
        obs : np.ndarray, shape (10, 17, 17)
        info : dict  — always contains 'action_mask'
        """
        super().reset(seed=seed)

        self._board = BoardWrapper([self._AGENT_COLOUR, self._OPPONENT_COLOUR])
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        obs = self._get_obs()
        info = {"action_mask": self.action_masks()}
        return obs, info

    def step(self, action: int):
        """Execute one agent step followed by one opponent step.

        Parameters
        ----------
        action : int
            Flat action index in [0, 1209].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self._board is not None, "Call reset() before step()."
        assert not (self._terminated or self._truncated), (
            "Episode has ended; call reset()."
        )

        # --- 1. Decode action ---
        # Accept either a flat int or a (pin_id, dest) tuple for compatibility.
        if isinstance(action, (tuple, list)):
            pin_id, dest = int(action[0]), int(action[1])
        else:
            pin_id, dest = self._mapper.decode(int(action))

        # --- 2. Snapshot metrics before agent move ---
        dist_before = self._board.total_distance_to_goal(self._AGENT_COLOUR)
        pins_before = self._board.pins_in_goal(self._AGENT_COLOUR)

        # --- 2b. Detect hop move before applying (pin position changes after apply) ---
        is_hop = self._is_hop_move(self._AGENT_COLOUR, pin_id, dest)

        # --- 3. Apply agent's move ---
        self._board.apply_move(self._AGENT_COLOUR, pin_id, dest)
        self._step_count += 1

        # --- 4. Check agent win ---
        agent_won = self._board.check_win(self._AGENT_COLOUR)
        if agent_won:
            dist_after = self._board.total_distance_to_goal(self._AGENT_COLOUR)
            pins_after = self._board.pins_in_goal(self._AGENT_COLOUR)
            pins_near = self._count_pins_near_goal(self._AGENT_COLOUR)
            reward = compute_step_reward(
                dist_before, dist_after,
                pins_before, pins_after,
                won=True, lost=False, drawn=False,
                is_hop=is_hop, pins_near_goal=pins_near,
            )
            self._terminated = True
            obs = self._get_obs()
            info = {"action_mask": self.action_masks()}
            return obs, reward, True, False, info

        # --- 5. Opponent moves ---
        opponent_legal = self._board.get_legal_moves(self._OPPONENT_COLOUR)
        if opponent_legal:
            opp_pin_id, opp_dest = self._opponent_policy(self._board, self._OPPONENT_COLOUR)
            self._board.apply_move(self._OPPONENT_COLOUR, opp_pin_id, opp_dest)

        # --- 6. Check opponent win ---
        opponent_won = self._board.check_win(self._OPPONENT_COLOUR)

        # --- 7. Check agent draw (no legal moves after opponent turn) ---
        agent_drawn = self._board.check_draw(self._AGENT_COLOUR)

        # --- 8. Compute reward ---
        dist_after = self._board.total_distance_to_goal(self._AGENT_COLOUR)
        pins_after = self._board.pins_in_goal(self._AGENT_COLOUR)
        pins_near = self._count_pins_near_goal(self._AGENT_COLOUR)

        reward = compute_step_reward(
            dist_before, dist_after,
            pins_before, pins_after,
            won=False,
            lost=opponent_won,
            drawn=agent_drawn,
            is_hop=is_hop,
            pins_near_goal=pins_near,
        )

        # --- 9. Truncation ---
        terminated = opponent_won or agent_drawn
        truncated = (self._step_count >= self.max_steps) and not terminated

        self._terminated = terminated
        self._truncated = truncated

        obs = self._get_obs()
        info = {"action_mask": self.action_masks()}
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of shape (1210,) for MaskablePPO.

        Called by sb3-contrib's MaskablePPO at each decision point.
        """
        if self._board is None or self._terminated or self._truncated:
            # Episode over — return all-False mask (won't be used, but must be valid shape)
            return np.zeros(1210, dtype=np.bool_)

        legal_moves = self._board.get_legal_moves(self._AGENT_COLOUR)
        return self._mapper.build_action_mask(legal_moves)

    def render(self):
        if self.render_mode == "ansi":
            if self._board is None:
                return "No active game."
            lines = []
            for colour in self._TURN_ORDER:
                pieces = self._board.get_pieces(colour)
                lines.append(f"{colour}: {[p['pos'] for p in pieces]}")
            return "\n".join(lines)

    @property
    def board_wrapper(self) -> "BoardWrapper | None":
        """Public accessor for the underlying BoardWrapper (used by agent tests)."""
        return self._board

    def set_opponent_policy(self, policy_fn):
        """Replace the opponent policy at runtime.

        Parameters
        ----------
        policy_fn : callable(board_wrapper, colour) -> (pin_id, dest_index)
        """
        self._opponent_policy = policy_fn

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Encode and return the current observation."""
        return self._encoder.encode(
            self._board,
            current_colour=self._AGENT_COLOUR,
            turn_order=self._TURN_ORDER,
        )

    def _is_hop_move(self, colour: str, pin_id: int, dest: int) -> bool:
        """Return True if the move from pin's current position to dest is a hop (not single-step)."""
        pin = self._board._pin_by_id(colour, pin_id)
        start_idx = pin.axialindex
        board = self._board.board

        start_cell = board.cells[start_idx]
        q0, r0 = start_cell.q, start_cell.r

        # Check if dest is a direct single-step neighbor
        for dq, dr in _HEX_DIRECTIONS:
            ni = board.index_of.get((q0 + dq, r0 + dr), None)
            if ni == dest:
                return False  # Single-step move, not a hop

        return True  # Must be a hop if it's a legal move but not adjacent

    def _count_pins_near_goal(self, colour: str, threshold: int = 3) -> int:
        """Count pins within `threshold` hex distance of any goal cell, excluding pins already in goal."""
        goal_indices = self._board.get_goal_indices(colour)
        goal_set = set(goal_indices)
        count = 0
        for pin in self._board.pins[colour]:
            if pin.axialindex in goal_set:
                continue  # Already in goal, don't count
            min_dist = min(self._board.axial_distance(pin.axialindex, g) for g in goal_indices)
            if min_dist <= threshold:
                count += 1
        return count

    @staticmethod
    def _random_opponent(board_wrapper: BoardWrapper, colour: str):
        """Select a uniformly random legal move for *colour*.

        Parameters
        ----------
        board_wrapper : BoardWrapper
        colour : str

        Returns
        -------
        (pin_id, dest_index) tuple
        """
        legal_moves = board_wrapper.get_legal_moves(colour)
        if not legal_moves:
            raise ValueError(f"_random_opponent called but {colour} has no legal moves.")

        pin_ids = list(legal_moves.keys())
        pin_id = pin_ids[np.random.randint(len(pin_ids))]
        dests = legal_moves[pin_id]
        dest = dests[np.random.randint(len(dests))]
        return (pin_id, dest)
