# =============================================================
# player.py — tournament client (AlphaZero ResNet 9x96 + Batched MCTS + heuristic value)
# =============================================================
#
# Move selection stack (each step falls back if the previous fails or runs out of time):
#   1. Batched MCTS (network priors + heuristic leaf value)  — primary
#   2. Standard MCTS                                          — backup
#   3. Raw network policy argmax                              — fast fallback
#   4. Advanced heuristic (1-ply lookahead + blocking)        — last resort
#   5. Greedy heuristic                                       — guaranteed move
#   6. Random legal move                                      — never-crash floor
#
# Time budgeting (per the README/game.py limits):
#   TURN_TIMEOUT_SEC = 10  per move
#   GAME_TIME_LIMIT_SEC = 60  cumulative over all our moves in a game
# We track our remaining game budget and shrink sim count adaptively. If the
# remaining budget is small, we drop to raw-policy or heuristic so we never
# bust the per-move or per-game cap.
#
# The agent never crashes: every step is wrapped in try/except. If the model
# fails to load we fall straight to advanced heuristic + greedy.

import os
import sys
import json
import time
import socket
import random
import traceback
from typing import Dict, Any, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HOST = os.getenv("CC_HOST", "127.0.0.1")
PORT = int(os.getenv("CC_PORT", "50555"))

# Per-move sim ceilings. Tuned at runtime based on detected hardware (see
# choose_sim_defaults()). Override either with CC_SIMS / CC_MIN_SIMS env vars.
# Defaults below are the "unknown hardware" floor — actual ceilings get raised
# automatically when a GPU is detected.
DEFAULT_SIMS = int(os.getenv("CC_SIMS", "0"))   # 0 = auto-detect
MIN_SIMS = int(os.getenv("CC_MIN_SIMS", "16"))
MCTS_BATCH_SIZE = int(os.getenv("CC_BATCH", "8"))

# Budget reserves (seconds). When time-left ≤ these thresholds, drop to faster
# strategies. Numbers are conservative — a missed move hurts far more than a
# slightly weaker one.
PER_MOVE_HARD_CAP = float(os.getenv("CC_PER_MOVE_HARD_CAP", "8.5"))   # under 10s
GAME_HARD_CAP = float(os.getenv("CC_GAME_HARD_CAP", "55.0"))          # under 60s
RAW_POLICY_BUDGET_SEC = float(os.getenv("CC_RAW_POLICY_BUDGET", "1.0"))
HEURISTIC_BUDGET_SEC = float(os.getenv("CC_HEURISTIC_BUDGET", "0.3"))

# Path to model checkpoint. Defaults to file next to this script.
# Looks in this order: CC_MODEL env, tournament_model.pt next to script,
# any best_model.pt in known experiment folders. The first found wins.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_model_path() -> str:
    env = os.getenv("CC_MODEL")
    if env and os.path.exists(env):
        return env
    candidates = [
        os.path.join(_HERE, "tournament_model.pt"),
        os.path.join(_HERE, "..", "experiments", "exp_d13_fast_cpu", "best_model.pt"),
        os.path.join(_HERE, "..", "experiments", "exp_d11_server", "best_model.pt"),
        os.path.join(_HERE, "..", "experiments", "exp_d5_resnet9x96", "warmstart_model.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return env or candidates[0]


MODEL_PATH = _resolve_model_path()

# Architecture defaults — must match the trained checkpoint.
NUM_BLOCKS = int(os.getenv("CC_NUM_BLOCKS", "9"))
NUM_FILTERS = int(os.getenv("CC_NUM_FILTERS", "96"))

DEBUG = os.getenv("CC_DEBUG", "0") not in ("0", "", "false", "False")
DEBUG_NET = os.getenv("DEBUG_NET", "0") not in ("0", "", "false", "False")


def debug(*args):
    if DEBUG or DEBUG_NET:
        print("[player]", *args, flush=True)


# ---------------------------------------------------------------------------
# Network protocol
# ---------------------------------------------------------------------------
def rpc(payload: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
    """Send JSON to server and receive JSON reply. Errors are returned, not raised."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((HOST, PORT))
        s.sendall(json.dumps(payload).encode("utf-8"))
        chunks = []
        while True:
            chunk = s.recv(1_000_000)
            if not chunk:
                break
            chunks.append(chunk)
            # Server sends a single response and closes; one recv is usually enough
            if len(chunks) >= 1 and chunk[-1:] in (b"}", b"]"):
                break
        data = b"".join(chunks)
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}
    finally:
        try:
            s.close()
        except Exception:
            pass

    if not data:
        return {"ok": False, "error": "no-response"}
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


# ---------------------------------------------------------------------------
# Game-side imports — done lazily so the script still runs without torch
# ---------------------------------------------------------------------------
def _add_project_to_path():
    """Add project root to sys.path so we can import src.* modules."""
    here = _HERE
    root = os.path.abspath(os.path.join(here, ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    # Also add this folder so checkers_board etc. resolve in src/env/board_wrapper
    if here not in sys.path:
        sys.path.insert(0, here)


_add_project_to_path()

# Always-available imports (pure python, no torch)
from checkers_board import HexBoard          # noqa: E402
from checkers_pins import Pin                # noqa: E402


# ---------------------------------------------------------------------------
# Heuristic policies (always available, no torch needed)
# ---------------------------------------------------------------------------
def _axial_dist(board: HexBoard, idx_a: int, idx_b: int) -> int:
    a = board.cells[idx_a]
    b = board.cells[idx_b]
    dq = abs(a.q - b.q)
    dr = abs(a.r - b.r)
    ds = abs((-a.q - a.r) - (-b.q - b.r))
    return max(dq, dr, ds)


def _min_dist_to_goal(board: HexBoard, pos_idx: int, goal_indices: List[int]) -> int:
    return min(_axial_dist(board, pos_idx, g) for g in goal_indices)


def greedy_choose(legal_moves: Dict[int, List[int]],
                  pin_positions: Dict[int, int],
                  board: HexBoard,
                  goal_indices: List[int]) -> Tuple[int, int]:
    """Greedy heuristic — choose the move that reduces distance to goal the most."""
    goal_set = set(goal_indices)
    best_score = None
    best_move = None
    fallback_move = None

    for pin_id, dests in legal_moves.items():
        if not dests:
            continue
        cur = pin_positions[pin_id]
        dist_cur = _min_dist_to_goal(board, cur, goal_indices)
        for dest in dests:
            if fallback_move is None:
                fallback_move = (pin_id, dest)
            score = dist_cur - _min_dist_to_goal(board, dest, goal_indices)
            if dest in goal_set:
                score += 5.0
            if best_score is None or score > best_score:
                best_score = score
                best_move = (pin_id, dest)

    return best_move if best_move is not None else fallback_move


def advanced_choose(legal_moves: Dict[int, List[int]],
                    pin_positions: Dict[int, int],
                    board: HexBoard,
                    goal_indices: List[int]) -> Tuple[int, int]:
    """Advanced heuristic with 1-ply lookahead and chain-hop bonus.

    Faster, simpler version of advanced_heuristic_policy that doesn't need
    BoardWrapper or opponent modeling. We only have JSON-derived state here.
    """
    goal_set = set(goal_indices)

    def position_score(positions: Dict[int, int]) -> float:
        # pins_in_goal * 100 + max(0, 200 - total_dist)
        pins_in = sum(1 for p in positions.values() if p in goal_set)
        total_dist = 0
        for p in positions.values():
            if p not in goal_set:
                total_dist += _min_dist_to_goal(board, p, goal_indices)
        return pins_in * 100.0 + max(0.0, 200.0 - total_dist)

    def lookahead_best(occupied: set, my_pos: Dict[int, int]) -> float:
        # Estimate the best position score reachable in one more move,
        # using only single-step + immediate-hop neighbors (cheap, no BFS).
        # Falls back to current if no candidates.
        best = position_score(my_pos)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        for pin_id, cur in my_pos.items():
            cell = board.cells[cur]
            cq, cr = cell.q, cell.r
            for dq, dr in directions:
                ni = board.index_of.get((cq + dq, cr + dr))
                if ni is not None and ni not in occupied:
                    new_pos = dict(my_pos)
                    new_pos[pin_id] = ni
                    s = position_score(new_pos)
                    if s > best:
                        best = s
                # one hop
                nj = board.index_of.get((cq + 2 * dq, cr + 2 * dr))
                if nj is not None and ni in occupied and nj not in occupied:
                    new_pos = dict(my_pos)
                    new_pos[pin_id] = nj
                    s = position_score(new_pos)
                    if s > best:
                        best = s
        return best

    occupied = set(pin_positions.values())
    cur_score = position_score(pin_positions)

    best_total = None
    best_move = None
    fallback = None

    for pin_id, dests in legal_moves.items():
        if not dests:
            continue
        cur = pin_positions[pin_id]
        in_goal_now = cur in goal_set
        for dest in dests:
            if fallback is None:
                fallback = (pin_id, dest)

            # Heavy penalty for leaving the goal triangle
            if in_goal_now and dest not in goal_set:
                tot = -1000.0
                if best_total is None or tot > best_total:
                    best_total = tot
                    best_move = (pin_id, dest)
                continue

            new_pos = dict(pin_positions)
            new_pos[pin_id] = dest
            new_occ = (occupied - {cur}) | {dest}
            immediate = position_score(new_pos) - cur_score
            la = lookahead_best(new_occ, new_pos)
            la_gain = la - position_score(new_pos)
            move_dist = _axial_dist(board, cur, dest)
            hop_bonus = move_dist * 0.3 if move_dist > 1 else 0.0

            tot = immediate * 1.0 + la_gain * 0.6 + hop_bonus
            if best_total is None or tot > best_total:
                best_total = tot
                best_move = (pin_id, dest)

    return best_move if best_move is not None else fallback


# ---------------------------------------------------------------------------
# Reconstruct a BoardWrapper-like state from the server's JSON
# ---------------------------------------------------------------------------
class JSONBoard:
    """Mirrors enough of BoardWrapper's API for StateEncoder + MCTS to work.

    Built from the JSON `pins` map provided by `get_state`. Any active
    colour (anything in `pins`) becomes a tracked colour. The game uses one
    shared HexBoard geometry so we just rebuild Pin objects against a fresh
    board with the right occupied flags.
    """

    COLOUR_OPPOSITES = {
        'red': 'blue', 'blue': 'red',
        'lawn green': 'gray0', 'gray0': 'lawn green',
        'yellow': 'purple', 'purple': 'yellow',
    }

    def __init__(self, pins_by_colour: Dict[str, List[int]]):
        self.board = HexBoard()
        # First, clear all 'occupied' flags (HexBoard sets them when Pin is
        # constructed via the home triangle, but here we set them explicitly).
        for cell in self.board.cells:
            cell.occupied = False

        self.colours = list(pins_by_colour.keys())
        self.pins: Dict[str, List[Pin]] = {}
        for colour, indices in pins_by_colour.items():
            pin_list = []
            for pid, idx in enumerate(indices):
                pin = Pin.__new__(Pin)
                pin.board = self.board
                pin.axialindex = int(idx)
                pin.id = pid
                pin.color = colour
                self.board.cells[int(idx)].occupied = True
                pin_list.append(pin)
            self.pins[colour] = pin_list

    # The BoardWrapper interface used by encoder / MCTS / heuristics:

    def get_legal_moves(self, colour: str) -> Dict[int, List[int]]:
        moves = {}
        for pin in self.pins[colour]:
            dests = pin.getPossibleMoves()
            if dests:
                moves[pin.id] = dests
        return moves

    def apply_move(self, colour: str, pin_id: int, dest_index: int) -> bool:
        pin = next((p for p in self.pins[colour] if p.id == pin_id), None)
        if pin is None:
            return False
        if self.board.cells[dest_index].occupied:
            return False
        self.board.cells[pin.axialindex].occupied = False
        pin.axialindex = int(dest_index)
        self.board.cells[int(dest_index)].occupied = True
        return True

    def get_goal_indices(self, colour: str) -> List[int]:
        opposite = self.COLOUR_OPPOSITES[colour]
        return self.board.axial_of_colour(opposite)

    def get_home_indices(self, colour: str) -> List[int]:
        return self.board.axial_of_colour(colour)

    def check_win(self, colour: str) -> bool:
        goal = set(self.get_goal_indices(colour))
        return all(p.axialindex in goal for p in self.pins[colour])

    def check_draw(self, colour: str) -> bool:
        return all(not p.getPossibleMoves() for p in self.pins[colour])

    def axial_distance(self, idx_a: int, idx_b: int) -> int:
        return _axial_dist(self.board, idx_a, idx_b)

    def total_distance_to_goal(self, colour: str) -> int:
        goal = self.get_goal_indices(colour)
        goal_set = set(goal)
        total = 0
        for pin in self.pins[colour]:
            if pin.axialindex not in goal_set:
                total += _min_dist_to_goal(self.board, pin.axialindex, goal)
        return total

    def pins_in_goal(self, colour: str) -> int:
        goal = set(self.get_goal_indices(colour))
        return sum(1 for p in self.pins[colour] if p.axialindex in goal)

    def get_pieces(self, colour: str) -> List[dict]:
        return [{'id': p.id, 'pos': p.axialindex} for p in self.pins[colour]]

    @staticmethod
    def _pin_by_id_static(pins, pin_id):
        for p in pins:
            if p.id == pin_id:
                return p
        return None

    def _pin_by_id(self, colour: str, pin_id: int):
        return self._pin_by_id_static(self.pins[colour], pin_id)

    def clone(self):
        # Fast clone: copy occupied flags and Pin positions.
        new = object.__new__(JSONBoard)
        new.colours = list(self.colours)
        # Reuse HexBoard but copy cells' occupied state
        from checkers_board import BoardPosition  # local import
        new_board = object.__new__(HexBoard)
        new_board.R = self.board.R
        new_board.hole_radius = self.board.hole_radius
        new_board.spacing = self.board.spacing
        new_board.colour_opposites = self.board.colour_opposites
        new_cells = []
        for cell in self.board.cells:
            nc = object.__new__(BoardPosition)
            nc.q = cell.q; nc.r = cell.r; nc.x = cell.x; nc.y = cell.y
            nc.postype = cell.postype; nc.occupied = cell.occupied
            new_cells.append(nc)
        new_board.cells = new_cells
        new_board.index_of = self.board.index_of
        new_board.cartesian = self.board.cartesian
        new_board._rows = self.board._rows
        new.board = new_board

        new.pins = {}
        for colour, plist in self.pins.items():
            new_pins = []
            for p in plist:
                np_ = Pin.__new__(Pin)
                np_.board = new_board
                np_.axialindex = p.axialindex
                np_.id = p.id
                np_.color = p.color
                new_pins.append(np_)
            new.pins[colour] = new_pins
        return new


# ---------------------------------------------------------------------------
# Network + MCTS — constructed once, lazily
# ---------------------------------------------------------------------------
class TournamentAgent:
    """Holds the model, encoder, mapper, and MCTS engine. None on init failure."""

    def __init__(self):
        self.network = None
        self.encoder = None
        self.mapper = None
        self.use_torch = False
        self._load()

    def _load(self):
        try:
            import torch  # noqa: F401
            from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
            from src.env.state_encoder import StateEncoder
            from src.env.action_mapper import ActionMapper

            self.has_cuda = bool(__import__("torch").cuda.is_available())
            device = "cuda" if self.has_cuda else "cpu"
            cfg = NetworkConfig(num_blocks=NUM_BLOCKS, num_filters=NUM_FILTERS)
            net = AlphaZeroNet(cfg, device=device)

            if not os.path.exists(MODEL_PATH):
                debug(f"Model file not found at {MODEL_PATH}; running heuristic-only")
                return
            net.load_checkpoint(MODEL_PATH)
            net.model.eval()

            self.network = net
            self.encoder = StateEncoder(grid_size=17, num_channels=10)
            self.mapper = ActionMapper(num_pins=10, num_cells=121)
            self.use_torch = True
            debug(f"Loaded model from {MODEL_PATH} (device={device}, params={net.parameter_count()})")

            # Calibrate forward-pass cost with one timed inference. Used by
            # choose_sims() to pick a sim count that fits the per-move budget.
            self.forward_ms = self._calibrate_forward()
            debug(f"Calibrated forward pass: {self.forward_ms:.1f}ms ({device})")
        except Exception as e:
            debug(f"Network unavailable, falling back to heuristic-only: {e}")
            if DEBUG:
                traceback.print_exc()
            self.network = None
            self.use_torch = False
            self.has_cuda = False
            self.forward_ms = 1000.0  # huge so MCTS path is avoided

    def _calibrate_forward(self) -> float:
        """Time a single forward pass to budget MCTS sims at runtime.

        Kept short (5 samples + 2 warmup) so initial load doesn't hit the
        connect timeout on slow CPU machines.
        """
        try:
            import numpy as np
            obs = np.zeros((10, 17, 17), dtype=np.float32)
            mask = np.ones(1210, dtype=np.bool_)
            # Warm up (compile / kernel launch / cuDNN init)
            for _ in range(2):
                self.network.predict(obs, mask)
            t0 = time.perf_counter()
            n = 5
            for _ in range(n):
                self.network.predict(obs, mask)
            ms = (time.perf_counter() - t0) / n * 1000.0
            # Guard against absurd values
            return max(1.0, min(ms, 500.0))
        except Exception:
            return 100.0  # safe pessimistic fallback

    # ------------------------------------------------------------------
    # MCTS engines (created per move so we can adjust sim count / batch)
    # ------------------------------------------------------------------
    def _make_proxy_env(self, board: JSONBoard, my_colour: str, opp_colour: str):
        """Build a minimal env-like object that MCTS expects.

        MCTS reads:
          env._board, env._mapper, env._AGENT_COLOUR, env._OPPONENT_COLOUR,
          env._TURN_ORDER, env._no_opponent, env._step_count, env.max_steps,
          env._terminated, env._truncated, env.action_space.n, env.clone(),
          env.action_masks() (used by some paths), env._get_obs() (some paths).
        """
        from src.env.chinese_checkers_env import ChineseCheckersEnv
        proxy = ChineseCheckersEnv.__new__(ChineseCheckersEnv)
        proxy.render_mode = None
        proxy.max_steps = 200
        proxy.observation_space = None
        proxy.action_space = type('Space', (), {'n': 1210})()
        proxy._encoder = self.encoder
        proxy._mapper = self.mapper
        proxy._AGENT_COLOUR = my_colour
        proxy._OPPONENT_COLOUR = opp_colour
        proxy._TURN_ORDER = [my_colour, opp_colour]
        proxy._no_opponent = True
        proxy._opponent_policy = None
        proxy._board = board
        proxy._step_count = 0
        proxy._terminated = False
        proxy._truncated = False
        return proxy

    # ------------------------------------------------------------------
    def select_with_mcts(self, board: JSONBoard, my_colour: str, opp_colour: str,
                         sims: int, batch_size: int, deadline: float
                        ) -> Optional[Tuple[int, int]]:
        """Run batched MCTS with heuristic value. Returns (pin_id, dest) or None."""
        if not self.use_torch or self.network is None:
            return None
        try:
            from src.search.batched_mcts import BatchedAlphaZeroMCTS
            env = self._make_proxy_env(board, my_colour, opp_colour)
            mcts = BatchedAlphaZeroMCTS(
                network=self.network,
                num_simulations=max(1, int(sims)),
                batch_size=max(1, int(batch_size)),
                dirichlet_epsilon=0.0,           # no exploration noise at play time
                use_heuristic_value=True,        # learned value head is unreliable
            )
            action = mcts.select_action(env, temperature=0.0)
            return self.mapper.decode(int(action))
        except Exception as e:
            debug(f"Batched MCTS failed: {e}")
            if DEBUG:
                traceback.print_exc()
            # Try standard MCTS as a backup
            try:
                from src.search.mcts import AlphaZeroMCTS
                env = self._make_proxy_env(board, my_colour, opp_colour)
                mcts = AlphaZeroMCTS(
                    network=self.network,
                    num_simulations=max(1, int(sims)),
                    dirichlet_epsilon=0.0,
                    use_heuristic_value=True,
                )
                action = mcts.select_action(env, temperature=0.0)
                return self.mapper.decode(int(action))
            except Exception as e2:
                debug(f"Standard MCTS also failed: {e2}")
                return None

    def select_with_raw_policy(self, board: JSONBoard, my_colour: str, opp_colour: str,
                               legal_moves: Dict[int, List[int]]
                              ) -> Optional[Tuple[int, int]]:
        """One forward pass; argmax over masked logits. Returns None on failure.

        Perspective handling: for blue/gray0/purple, the encoder returns a
        rotated obs and the policy head outputs in the canonical (rotated)
        frame. We rotate the action mask into that frame before predict, then
        rotate the priors back to raw frame for argmax + decode.
        """
        if not self.use_torch or self.network is None:
            return None
        try:
            import numpy as np
            obs = self.encoder.encode(board, current_colour=my_colour,
                                      turn_order=[my_colour, opp_colour])
            mask_raw = self.mapper.build_action_mask(legal_moves)
            if not mask_raw.any():
                return None
            if self.encoder.needs_rotation(my_colour):
                mask_canon = self.encoder.rotate_action_distribution(
                    mask_raw.astype(np.bool_)
                ).astype(np.bool_)
                probs_canon, _ = self.network.predict(obs, mask_canon)
                probs = self.encoder.rotate_action_distribution(probs_canon)
            else:
                probs, _ = self.network.predict(obs, mask_raw)
            action = int(np.argmax(probs))
            return self.mapper.decode(action)
        except Exception as e:
            debug(f"Raw policy failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Time-budgeted move selection
# ---------------------------------------------------------------------------
def pick_opponent(my_colour: str, turn_order: List[str],
                  pins: Dict[str, List[int]],
                  board: Optional["JSONBoard"] = None) -> str:
    """Pick the most relevant opponent for state encoding.

    For 2-player games we use our colour-pair opposite (the model was trained
    against this colour). For >2 players we pick the *most threatening*
    opponent — the one closest to their goal — and encode them into channel 1.
    The model still sees a 2-player layout but the channel-1 pins represent
    whichever player is winning. Falls back to turn-order next, then any
    other present colour.
    """
    other = [c for c in pins if c != my_colour]
    if not other:
        return my_colour  # degenerate; encoder will just see empty channel 1

    # Two-player: prefer the colour pair the model was trained on (if present)
    opposite = JSONBoard.COLOUR_OPPOSITES.get(my_colour)
    if len(other) == 1:
        return other[0]
    if len(pins) == 2 and opposite in other:
        return opposite

    # Three or more players: pick the leader (lowest distance-to-goal)
    if board is not None:
        try:
            best_colour = None
            best_dist = None
            for c in other:
                d = board.total_distance_to_goal(c)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_colour = c
            if best_colour is not None:
                return best_colour
        except Exception:
            pass

    # Fallback: next in turn order
    if turn_order and my_colour in turn_order:
        idx = turn_order.index(my_colour)
        for offset in range(1, len(turn_order)):
            cand = turn_order[(idx + offset) % len(turn_order)]
            if cand in pins and cand != my_colour:
                return cand

    if opposite in other:
        return opposite
    return other[0]


def estimate_remaining_budget(time_used_sec: float) -> float:
    return max(0.0, GAME_HARD_CAP - time_used_sec)


def choose_sims(remaining_game_budget: float, moves_made: int,
                forward_ms: float = 30.0,
                sims_ceiling: int = 0) -> int:
    """Pick sims/move from remaining game budget and measured forward-pass cost.

    Heuristic: assume max(8, 50-moves_made) more moves to play, divide the
    remaining game budget evenly. Then convert that per-move time to sim count
    using the calibrated forward-ms (each sim ≈ one forward pass with batched
    overhead amortised). Stay between MIN_SIMS and the ceiling.
    """
    expected_more_moves = max(8, 50 - moves_made)
    per_move_budget = remaining_game_budget / expected_more_moves
    # Reserve 100ms per move for overhead (encoding, JSON, etc.)
    usable_ms = max(0.0, per_move_budget * 1000 - 100)
    # Batched MCTS amortises ~1.3x faster than single forward; be conservative.
    sims_from_budget = int(usable_ms / max(forward_ms * 0.8, 1.0))

    if sims_ceiling <= 0:
        # Auto ceiling: GPU can sustain ~1000+ sims/move easily; CPU much less.
        # forward_ms < 8ms = GPU-class, allow up to 400.
        # forward_ms < 25ms = decent CPU, allow 200.
        # forward_ms >= 25ms = slow CPU, cap at 100.
        if forward_ms < 8:
            sims_ceiling = 400
        elif forward_ms < 25:
            sims_ceiling = 200
        else:
            sims_ceiling = 100

    target = min(sims_from_budget, sims_ceiling)
    target = max(target, MIN_SIMS)
    return target


# Precomputed colour list for fallback opponent pick (matches game.py)
_COLOUR_ORDER = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']


def select_move(agent: TournamentAgent,
                state: Dict[str, Any],
                my_colour: str,
                legal_moves: Dict[int, List[int]],
                time_used_sec: float,
                moves_made: int) -> Tuple[int, int]:
    """Top-level move selection with hierarchical fallbacks and time guard."""
    if not legal_moves:
        return (0, 0)  # caller will see "no movable" and skip

    # Build the JSON board (cheap, ~ms)
    pins = state.get("pins", {}) or {}
    if my_colour not in pins:
        # The server should always include our pins; safety net
        pins = {my_colour: list(legal_moves.keys()), **pins}
    try:
        board = JSONBoard(pins)
    except Exception as e:
        debug(f"JSONBoard construction failed ({e}); falling back to greedy on raw legal moves")
        # Bare-minimum random pick over legal moves
        pid = next(iter(legal_moves.keys()))
        return pid, legal_moves[pid][0]

    # Pin position lookup for heuristics
    pin_positions = {p.id: p.axialindex for p in board.pins[my_colour]}
    goal_indices = board.get_goal_indices(my_colour)

    # Pick the opponent colour for state encoding (most-threatening for 3+ players)
    turn_order = state.get("turn_order") or []
    opp_colour = pick_opponent(my_colour, turn_order, pins, board=board)

    # Time budget
    remaining = estimate_remaining_budget(time_used_sec)

    # If we're cooked on time, drop to greedy or heuristic
    if remaining <= HEURISTIC_BUDGET_SEC:
        try:
            return greedy_choose(legal_moves, pin_positions, board.board, goal_indices)
        except Exception:
            pid, dests = next(iter(legal_moves.items()))
            return pid, dests[0]

    # Decide sim count using measured forward-pass cost
    forward_ms = getattr(agent, "forward_ms", 30.0)
    ceiling = DEFAULT_SIMS  # 0 = auto in choose_sims()
    sims = choose_sims(remaining, moves_made, forward_ms=forward_ms,
                       sims_ceiling=ceiling)

    # Per-move deadline relative to wall clock
    move_start = time.perf_counter()
    per_move_budget = min(PER_MOVE_HARD_CAP, max(0.5, remaining - 1.0))
    deadline = move_start + per_move_budget

    # 1) Try batched MCTS
    if remaining > RAW_POLICY_BUDGET_SEC and agent.use_torch:
        # Defensive: shrink sims if the per-move wall-clock budget is tight.
        # Use measured forward-pass cost (already factored into choose_sims),
        # but apply a per-move cap so a single slow position can't blow up.
        ms_budget = per_move_budget * 1000 - 100  # 100ms overhead reserve
        per_sim_ms = forward_ms * 0.8  # batched amortisation
        capped = max(MIN_SIMS, int(ms_budget / max(per_sim_ms, 1.0)))
        sims_eff = max(MIN_SIMS, min(sims, capped))
        try:
            mv = agent.select_with_mcts(board, my_colour, opp_colour,
                                        sims=sims_eff, batch_size=MCTS_BATCH_SIZE,
                                        deadline=deadline)
            if mv is not None and _is_legal(mv, legal_moves):
                debug(f"MCTS chose pin {mv[0]} -> {mv[1]} (sims={sims_eff}, "
                      f"t={time.perf_counter()-move_start:.2f}s)")
                return mv
        except Exception as e:
            debug(f"MCTS layer raised: {e}")

    # 2) Raw policy fallback
    if agent.use_torch and (time.perf_counter() < deadline):
        try:
            mv = agent.select_with_raw_policy(board, my_colour, opp_colour, legal_moves)
            if mv is not None and _is_legal(mv, legal_moves):
                debug(f"Raw policy chose pin {mv[0]} -> {mv[1]}")
                return mv
        except Exception as e:
            debug(f"Raw policy raised: {e}")

    # 3) Advanced heuristic
    try:
        mv = advanced_choose(legal_moves, pin_positions, board.board, goal_indices)
        if mv is not None and _is_legal(mv, legal_moves):
            debug(f"Advanced heuristic chose pin {mv[0]} -> {mv[1]}")
            return mv
    except Exception as e:
        debug(f"Advanced heuristic raised: {e}")

    # 4) Greedy
    try:
        mv = greedy_choose(legal_moves, pin_positions, board.board, goal_indices)
        if mv is not None and _is_legal(mv, legal_moves):
            return mv
    except Exception as e:
        debug(f"Greedy raised: {e}")

    # 5) Random legal — never-crash floor
    pid = random.choice(list(legal_moves.keys()))
    return pid, random.choice(legal_moves[pid])


def _is_legal(mv: Tuple[int, int], legal_moves: Dict[int, List[int]]) -> bool:
    pid, dest = mv
    dests = legal_moves.get(pid) or legal_moves.get(int(pid))
    if not dests:
        return False
    return int(dest) in dests


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------
def render_json_board(state):
    """Lightweight ASCII summary used when DEBUG is set."""
    if not DEBUG:
        return
    pins = state.get("pins", {})
    print("=== BOARD STATE ===")
    for colour, indices in pins.items():
        print(f"  {colour}: {indices}")
    print("===================")


def main():
    print("==== Player ====")
    name = input("Enter name: ").strip()
    if not name:
        name = "az_agent"

    print("Loading model...", flush=True)
    agent = TournamentAgent()
    if agent.use_torch:
        print(f"  Model loaded (params={agent.network.parameter_count()}).", flush=True)
    else:
        print("  Running in heuristic-only mode (advanced + greedy fallbacks).", flush=True)

    # JOIN GAME
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]
    print(f"Joined game {game_id} as {colour}")

    # Wait for game to be ready
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        s = st.get("state", {}) if st.get("ok") else {}
        if s.get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    # In tournament mode you may want to auto-start: set CC_AUTOSTART=1
    if os.getenv("CC_AUTOSTART", "0") not in ("0", "", "false", "False"):
        print("Auto-starting...")
    else:
        try:
            input("Press ENTER to send START...")
        except EOFError:
            pass
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        s = st.get("state", {}) if st.get("ok") else {}
        if s.get("status") == "PLAYING":
            break
        time.sleep(0.3)
    print("=== GAME STARTED ===\n", flush=True)

    last_move_seen = 0
    timeoutnotice_move = -1
    my_time_used = 0.0
    my_moves_made = 0

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("State error:", st.get("error"))
            time.sleep(0.2)
            continue
        state = st["state"]

        # Update our cumulative time used (server tracks it)
        for pl in state.get("players", []):
            if pl.get("colour") == colour:
                sc = pl.get("score") or {}
                if "time_taken_sec" in sc:
                    my_time_used = float(sc["time_taken_sec"])
                if "moves" in sc:
                    my_moves_made = int(sc["moves"])
                break

        # Timeout messages
        if state.get("turn_timeout_notice") and timeoutnotice_move < state.get("move_count", 0):
            print("⚠ TIMEOUT:", state["turn_timeout_notice"])
            timeoutnotice_move = state.get("move_count", 0)

        # Finished?
        if state.get("status") == "FINISHED":
            print("\n=== GAME FINISHED ===")
            print("FINAL SCORES:")
            for pl in state.get("players", []):
                sc = pl.get("score")
                if sc:
                    print(
                        f"{pl['name']} ({pl['colour']}): "
                        f"{sc.get('final_score', 0):.1f} "
                        f"[time={sc.get('time_score', 0):.1f}, "
                        f"moves({sc.get('moves', 0)})={sc.get('move_score', 0):.1f}, "
                        f"pins={sc.get('pin_goal_score', 0):.1f}, "
                        f"dist={sc.get('distance_score', 0):.1f}]"
                    )
            print("======================")
            break

        # Show last move
        if state.get("move_count", 0) > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) {mv['from']}→{mv['to']}  "
                    f"[{mv.get('move_ms', 0):.1f}ms]"
                )
            last_move_seen = state.get("move_count", 0)

        # Our turn?
        if state.get("current_turn_colour") == colour and state.get("status") == "PLAYING":
            move_t0 = time.perf_counter()
            print(f"\nMy turn (move {my_moves_made + 1}, "
                  f"used {my_time_used:.1f}s of {GAME_HARD_CAP:.0f}s budget)", flush=True)

            # Get legal moves from server
            legal_req = rpc({"op": "get_legal_moves", "game_id": game_id, "player_id": player_id})
            if not legal_req.get("ok"):
                print("Error requesting legal moves:", legal_req.get("error"))
                time.sleep(0.3)
                continue

            raw_legal = legal_req.get("legal_moves", {}) or {}
            # Coerce keys to ints (JSON object keys are strings)
            legal_moves = {int(k): list(map(int, v)) for k, v in raw_legal.items() if v}
            if not legal_moves:
                print("No legal moves available; skipping turn.")
                time.sleep(0.3)
                continue

            try:
                pid, dest = select_move(
                    agent, state, colour, legal_moves,
                    time_used_sec=my_time_used,
                    moves_made=my_moves_made,
                )
            except Exception as e:
                print(f"select_move crashed: {e}; using random legal move", flush=True)
                if DEBUG:
                    traceback.print_exc()
                pid = next(iter(legal_moves.keys()))
                dest = legal_moves[pid][0]

            # Final sanity check before submitting
            if pid not in legal_moves or dest not in legal_moves[pid]:
                debug(f"chosen move (pin {pid} -> {dest}) not in legal_moves; correcting")
                pid = next(iter(legal_moves.keys()))
                dest = legal_moves[pid][0]

            decide_ms = (time.perf_counter() - move_t0) * 1000
            print(f"  -> pin {pid} to cell {dest}  ({decide_ms:.0f}ms decide)", flush=True)

            mv_resp = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": int(pid),
                "to_index": int(dest),
            })
            if not mv_resp.get("ok"):
                print("Move rejected:", mv_resp.get("error"))
                # Try a safe greedy fallback once
                try:
                    pin_positions = {int(p["id"]): int(p["pos"])
                                     for p in (mv_resp.get("state", {}).get("pins", {}).get(colour) or [])}
                except Exception:
                    pin_positions = {}
                # Just retry with the literal first legal move
                pid2 = next(iter(legal_moves.keys()))
                dest2 = legal_moves[pid2][0]
                rpc({"op": "move", "game_id": game_id, "player_id": player_id,
                     "pin_id": int(pid2), "to_index": int(dest2)})
            else:
                if mv_resp.get("status") == "WIN":
                    print("YOU WIN!", mv_resp.get("msg"))
                elif mv_resp.get("status") == "DRAW":
                    print("DRAW", mv_resp.get("msg"))

        time.sleep(0.2)


if __name__ == "__main__":
    main()
