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
HOST = os.getenv("CC_HOST", "10.245.30.229")
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
        # CURRENT (d41-multiN): paired with the native-d41 SCM in scm_d41_multi/.
        # H2H eval at N=6 (tournament's heaviest weight): native SCM wins 11/20
        # vs SCM v2's 5/20 (p=0.0007). Tournament-weighted (1,4,6,8,12 for N=2..6)
        # gives +0.24 expected pins and +39% expected wins over the v2 pairing.
        os.path.join(_HERE, "..", "experiments", "exp_d41_multiN", "best_so_far.pt"),
        # PREVIOUS CHAMPION (d35-best, win-only RL): 22/30 wins vs advanced @ sims=400.
        # ResNet 9x96, multicolour encoder, trained from d22 with replay buffer
        # + per-colour win filter (≥7 pins) + outcome-weighted loss + KL filter.
        # Handles N=2..6 player layouts. See DEC-NEW-023 in docs/DECISIONS.md.
        os.path.join(_HERE, "..", "experiments", "exp_d35_winonly", "best_so_far.pt"),
        # Fallback: d22 warmstart-only (supervised). 19/30 wins vs advanced.
        os.path.join(_HERE, "..", "experiments", "exp_d22_multicolour", "warmstart_model.pt"),
        # 1v1 specialist (legacy 180°-rotation, single-opponent encoder):
        # 7 pins vs greedy / 10-of-20 vs advanced at 50-sim eval. Used as a
        # fallback if the multicolour checkpoint is missing.
        os.path.join(_HERE, "..", "experiments", "exp_d20_sharpened", "warmstart_model.pt"),
        os.path.join(_HERE, "..", "experiments", "exp_d17_higher_sims", "best_model.pt"),
        os.path.join(_HERE, "..", "experiments", "exp_d14_perspective_fix", "best_model.pt"),
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

# Search-Conditioned Modulation (SCM): GRU that reads MCTS search stats
# per turn and FiLM-modulates the policy logits across game phases.
# Trained on 2-player vs greedy; pair with d35-best for 1v1.
# CC_SCM_CHECKPOINT=""  disables SCM and falls back to plain MCTS.
# CC_SCM_BLEND in [0,1]: how much to mix SCM-modulated probs into MCTS visit probs.
def _resolve_scm_path() -> str:
    env = os.getenv("CC_SCM_CHECKPOINT")
    if env is not None:
        return env  # explicit override (empty string = disabled)
    candidates = [
        os.path.join(_HERE, "scm_model.pt"),
        # CURRENT (native-d41): trained on 1800 multiplayer trajectories from
        # d41 itself, with tournament-matched player count weights (1,4,6,8,12
        # for N=2..6). Wins decisively at N=6 vs SCM v2 (avg 9.35 vs 8.10 pins,
        # 11 vs 5 wins, p=0.0007). Pair with experiments/exp_d41_multiN policy.
        os.path.join(_HERE, "..", "experiments", "scm_d41_multi", "scm_model.pt"),
        os.path.join(_HERE, "..", "experiments", "scm_v3_d37", "scm_model.pt"),
        os.path.join(_HERE, "..", "experiments", "scm_v2", "scm_model.pt"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

SCM_PATH = _resolve_scm_path()
# Blend default tuned by multi-N arena, NOT by 1v1.
# 1v1 vs greedy blend sweep (30 games each):
#   0%=6.0 → 10%=7.0 → 20%=7.1 → 30%=7.1 → 50%=7.5 → 70%=6.4 → 100%=4.9
# 4p arena (20 games, vs d35/d37/d38/adv/gr):
#   blend=0.2: scm-d35 = 63.6% wins, 9.27 avg pins, 1195 avg score
#   blend=0.5: scm-d35 = 36.4% wins, 8.73 avg pins, 1138 avg score
# Higher blend wins more pins in 1v1 vs greedy but loses ground in 4p
# against strong RL opponents. Tournament is multi-N → use 0.2.
SCM_BLEND = float(os.getenv("CC_SCM_BLEND", "0.2"))
SCM_SIMS_CEILING = int(os.getenv("CC_SCM_SIMS", "0"))  # 0 = follow normal ceiling

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
        # SCM state
        self.scm = None              # SearchConditionedModulator or None
        self.scm_hidden = None       # torch.Tensor or None (shape: 1, hidden_dim)
        self.scm_config = None       # SCMConfig
        self.scm_turn = 0            # turn counter inside current game
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
            ckpt = net.load_checkpoint(MODEL_PATH)
            net.model.eval()

            # Encoder mode is stamped on the checkpoint by the trainer. Defaults
            # to "legacy" (single-opp channel + 180° rotation only) for back-compat
            # with d20. Multicolour models (d22+) carry "multicolour" so player.py
            # routes inference through encode_multicolour.
            self.encoder_mode = ckpt.get("encoder_mode", "legacy")
            if not isinstance(self.encoder_mode, str):
                self.encoder_mode = "legacy"

            self.network = net
            self.encoder = StateEncoder(grid_size=17, num_channels=10,
                                        mode=self.encoder_mode)
            self.mapper = ActionMapper(num_pins=10, num_cells=121)
            self.use_torch = True
            debug(f"Loaded model from {MODEL_PATH} "
                  f"(device={device}, params={net.parameter_count()}, "
                  f"encoder_mode={self.encoder_mode})")

            # Calibrate forward-pass cost with one timed inference. Used by
            # choose_sims() to pick a sim count that fits the per-move budget.
            self.forward_ms = self._calibrate_forward()
            debug(f"Calibrated forward pass: {self.forward_ms:.1f}ms ({device})")

            # SCM (Search-Conditioned Modulation) — optional GRU that
            # FiLM-modulates policy logits using MCTS search stats. Trained
            # against 2-player vs greedy. Naturally pairs with the 1v1
            # specialist (d35-best) but works in any N as a logit-shaping head.
            self._load_scm(device)
        except Exception as e:
            debug(f"Network unavailable, falling back to heuristic-only: {e}")
            if DEBUG:
                traceback.print_exc()
            self.network = None
            self.use_torch = False
            self.has_cuda = False
            self.forward_ms = 1000.0  # huge so MCTS path is avoided
            self.encoder_mode = "legacy"

    def _load_scm(self, device: str) -> None:
        """Load the SCM module from SCM_PATH (best-effort).

        Failure is non-fatal — agent runs without SCM modulation.
        """
        self.scm = None
        self.scm_hidden = None
        self.scm_config = None
        if not SCM_PATH or not os.path.exists(SCM_PATH):
            debug(f"SCM disabled (path: {SCM_PATH!r})")
            return
        try:
            import torch
            from src.network.search_conditioned_modulator import (
                SearchConditionedModulator, SCMConfig,
            )
            ckpt = torch.load(SCM_PATH, map_location=device, weights_only=False)
            # Try to pull SCMConfig from ckpt; fall back to defaults.
            cfg_dict = ckpt.get("scm_config") if isinstance(ckpt, dict) else None
            if isinstance(cfg_dict, dict):
                cfg = SCMConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SCMConfig.__dataclass_fields__})
            else:
                cfg = SCMConfig()
            scm = SearchConditionedModulator(cfg).to(device)
            state_dict = None
            if isinstance(ckpt, dict):
                state_dict = (ckpt.get("scm_state_dict")
                              or ckpt.get("state_dict")
                              or ckpt.get("model_state_dict"))
            if state_dict is None and hasattr(ckpt, "state_dict"):
                state_dict = ckpt.state_dict()
            scm.load_state_dict(state_dict)
            scm.eval()
            self.scm = scm
            self.scm_config = cfg
            self.scm_hidden = scm.init_hidden(1, device=torch.device(device))
            self.scm_turn = 0
            debug(f"Loaded SCM from {SCM_PATH} (params={scm.param_count()}, blend={SCM_BLEND})")
        except Exception as e:
            debug(f"SCM load failed, continuing without it: {e}")
            if DEBUG:
                traceback.print_exc()
            self.scm = None
            self.scm_hidden = None
            self.scm_config = None

    def new_game(self) -> None:
        """Reset per-game state (called after joining a new game).

        Currently only resets the SCM GRU hidden state and turn counter so
        successive games don't carry over each other's modulation context.
        """
        self.scm_turn = 0
        if self.scm is not None:
            try:
                import torch
                device = next(self.scm.parameters()).device
                self.scm_hidden = self.scm.init_hidden(1, device=device)
            except Exception:
                self.scm_hidden = None

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
    def _make_proxy_env(self, board: JSONBoard, my_colour: str, opp_colour: str,
                        turn_order: Optional[List[str]] = None):
        """Build a minimal env-like object that MCTS expects.

        MCTS reads:
          env._board, env._mapper, env._AGENT_COLOUR, env._OPPONENT_COLOUR,
          env._TURN_ORDER, env._no_opponent, env._step_count, env.max_steps,
          env._terminated, env._truncated, env.action_space.n, env.clone(),
          env.action_masks() (used by some paths), env._get_obs() (some paths).

        For multicolour encoder mode, turn_order matters because
        encode_multicolour reads it to populate the "next opponent" channel.
        Pass the real server turn_order for correctness; default falls back to
        the 2-colour pair for legacy mode.
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
        proxy._TURN_ORDER = list(turn_order) if turn_order else [my_colour, opp_colour]
        proxy._no_opponent = True
        proxy._opponent_policy = None
        proxy._board = board
        proxy._step_count = 0
        proxy._terminated = False
        proxy._truncated = False
        return proxy

    # ------------------------------------------------------------------
    def select_with_mcts(self, board: JSONBoard, my_colour: str, opp_colour: str,
                         sims: int, batch_size: int, deadline: float,
                         turn_order: Optional[List[str]] = None
                        ) -> Optional[Tuple[int, int]]:
        """Run batched MCTS with heuristic value. Returns (pin_id, dest) or None."""
        if not self.use_torch or self.network is None:
            return None
        try:
            from src.search.batched_mcts import BatchedAlphaZeroMCTS
            env = self._make_proxy_env(board, my_colour, opp_colour, turn_order=turn_order)
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
                env = self._make_proxy_env(board, my_colour, opp_colour, turn_order=turn_order)
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

    def select_with_scm_mcts(self, board: JSONBoard, my_colour: str, opp_colour: str,
                              sims: int, deadline: float,
                              turn_order: Optional[List[str]] = None,
                              blend: float = SCM_BLEND
                              ) -> Optional[Tuple[int, int]]:
        """Standard MCTS + SCM-modulated logits, blended at `blend`.

        Pipeline:
          1) Run standard AlphaZeroMCTS to get (visit_probs, search_stats),
             both in the agent's raw frame (mapper-decodable).
          2) Compute raw policy logits via our own encode→predict_raw_logits
             call (this side handles multicolour rotation correctly).
          3) Modulate logits with SCM, advance hidden state.
          4) Softmax → scm_probs in agent frame.
          5) Combine final_probs = (1-blend)*visit_probs + blend*scm_probs,
             argmax, decode to (pin_id, dest).

        Falls back to raw MCTS visits if anything goes wrong.
        """
        if not self.use_torch or self.network is None or self.scm is None:
            return None
        try:
            import numpy as np
            import torch
            from src.search.mcts import AlphaZeroMCTS
            from src.network.search_conditioned_modulator import (
                SearchStats, extract_search_features,
            )

            env = self._make_proxy_env(board, my_colour, opp_colour, turn_order=turn_order)
            mcts = AlphaZeroMCTS(
                network=self.network,
                num_simulations=max(1, int(sims)),
                dirichlet_epsilon=0.0,
                use_heuristic_value=True,
            )
            visit_probs, stats = mcts.get_action_probs_with_stats(
                env, temperature=0.0, turn_number=self.scm_turn,
            )

            # If MCTS yielded nothing legal, bail.
            if visit_probs.sum() <= 0:
                return None

            # Re-extract raw logits in the agent (raw) frame, doing rotation
            # correctly for both legacy and multicolour encoders.
            mask_raw = env.action_masks().astype(np.bool_)
            obs = env._get_obs()
            if self.encoder_mode == "multicolour":
                k = self.encoder.k_to_red_frame(my_colour)
                mask_canon = self.encoder.rotate_action_distribution_k(
                    mask_raw, k
                ).astype(np.bool_)
                raw_logits_canon, _, _ = self.network.predict_raw_logits(obs, mask_canon)
                # Rotate logits back to raw frame so they align with mapper actions.
                raw_logits = self.encoder.rotate_action_distribution_k(
                    raw_logits_canon, (6 - k) % 6
                )
                # Override stats.raw_policy with a correctly-rotated policy
                # (MCTS's stats.raw_policy uses the buggy non-rotated path for
                # multicolour mode).
                masked_logits = np.where(mask_raw, raw_logits, -1e9)
                shifted = masked_logits - masked_logits.max()
                exp_l = np.exp(shifted)
                raw_policy_correct = exp_l / max(exp_l.sum(), 1e-12)
                stats = SearchStats(
                    visit_counts=stats.visit_counts,
                    q_values=stats.q_values,
                    raw_policy=raw_policy_correct.astype(np.float32),
                    action_mask=mask_raw,
                    root_value=stats.root_value,
                    max_depth=stats.max_depth,
                    total_visits=stats.total_visits,
                    turn_number=stats.turn_number,
                )
            elif self.encoder.needs_rotation(my_colour):
                mask_canon = self.encoder.rotate_action_distribution(mask_raw).astype(np.bool_)
                raw_logits_canon, _, _ = self.network.predict_raw_logits(obs, mask_canon)
                raw_logits = self.encoder.rotate_action_distribution(raw_logits_canon)
            else:
                raw_logits, _, _ = self.network.predict_raw_logits(obs, mask_raw)

            # Build SCM search features (in agent frame, like SCM was trained).
            feats_np = extract_search_features(stats, self.scm_config)
            device = next(self.scm.parameters()).device
            feats_t = torch.tensor(feats_np[np.newaxis], dtype=torch.float32, device=device)
            raw_logits_t = torch.tensor(raw_logits[np.newaxis], dtype=torch.float32, device=device)

            with torch.no_grad():
                modulated, _, _, new_hidden = self.scm.modulate_logits(
                    raw_logits_t, feats_t, self.scm_hidden, blend=1.0,
                )
                # Mask illegal actions before softmax.
                mask_t = torch.tensor(mask_raw[np.newaxis], dtype=torch.bool, device=device)
                modulated = modulated.masked_fill(~mask_t, -1e9)
                scm_probs = torch.softmax(modulated, dim=-1).squeeze(0).cpu().numpy()

            self.scm_hidden = new_hidden  # advance GRU state for next turn
            self.scm_turn += 1

            # Blend MCTS visit distribution with SCM-modulated policy.
            b = max(0.0, min(1.0, float(blend)))
            final_probs = (1.0 - b) * visit_probs + b * scm_probs
            # Re-mask just in case blending hits a tiny illegal entry.
            final_probs = np.where(mask_raw, final_probs, 0.0)
            if final_probs.sum() <= 0:
                final_probs = visit_probs  # safety: fall back to pure MCTS
            action = int(np.argmax(final_probs))
            mv = self.mapper.decode(action)
            return mv
        except Exception as e:
            debug(f"SCM+MCTS failed: {e}")
            if DEBUG:
                traceback.print_exc()
            return None

    def select_with_raw_policy(self, board: JSONBoard, my_colour: str, opp_colour: str,
                               legal_moves: Dict[int, List[int]],
                               turn_order: Optional[List[str]] = None
                              ) -> Optional[Tuple[int, int]]:
        """One forward pass; argmax over masked logits. Returns None on failure.

        Perspective handling: for legacy mode, the encoder rotates obs 180° for
        blue/gray0/purple and we rotate masks/priors with `rotate_action_distribution`.
        For multicolour mode, the encoder rotates by k×60° per colour and we use
        `rotate_action_distribution_k(., k)` instead.
        """
        if not self.use_torch or self.network is None:
            return None
        try:
            import numpy as np
            mask_raw = self.mapper.build_action_mask(legal_moves)
            if not mask_raw.any():
                return None

            if self.encoder_mode == "multicolour":
                # encode_multicolour rotates by k×60° for the playing colour;
                # mask + priors must use the same k.
                tor = turn_order or [my_colour, opp_colour]
                obs = self.encoder.encode_multicolour(board, my_colour, tor)
                k = self.encoder.k_to_red_frame(my_colour)
                mask_canon = self.encoder.rotate_action_distribution_k(
                    mask_raw.astype(np.bool_), k
                ).astype(np.bool_)
                probs_canon, _ = self.network.predict(obs, mask_canon)
                # Rotate priors back to raw frame: k×60° + (6-k)×60° = identity.
                probs = self.encoder.rotate_action_distribution_k(probs_canon, (6 - k) % 6)
            else:
                obs = self.encoder.encode(board, current_colour=my_colour,
                                          turn_order=[my_colour, opp_colour])
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
        # Auto ceiling, calibrated against actual V100 measurements:
        #   sims=400  → 347ms/move  → 80 moves * 347ms = 27.8s  (well under 60s)
        #   sims=800  → 661ms/move  → 80 moves * 661ms = 52.9s  (fits 60s budget)
        #   sims=1600 → 1415ms/move → 80 moves * 1.4s = 113s    (overruns)
        # GPU can comfortably do 800; CPU is much weaker and per-move cost
        # scales poorly with batched MCTS (since batches don't fill).
        if forward_ms < 8:
            sims_ceiling = 800   # GPU-class: was 400, but profile shows 800 fits
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
                moves_made: int,
                recent_moves: Optional[List[Tuple[int, int]]] = None
                ) -> Tuple[int, int]:
    """Top-level move selection with hierarchical fallbacks and time guard.

    Parameters
    ----------
    recent_moves : list of (pin_id, dest_cell) for our last few moves.
        Used to detect 2-move cycles (we pingpong the same pin between two
        cells). When the model would cycle, we pick a different move.
    """
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

    # Late-game heuristic-first override: past move 70 the model tends to
    # oscillate (heuristic value gradient flat near board fill-up). The
    # advanced heuristic with 1-ply lookahead picks decisive moves and
    # respects the "don't leave goal" rule.
    if moves_made >= 70:
        try:
            mv = advanced_choose(legal_moves, pin_positions, board.board, goal_indices)
            if mv is not None and _is_legal(mv, legal_moves) and not _would_cycle(mv, recent_moves):
                debug(f"Late-game heuristic chose pin {mv[0]} -> {mv[1]} (move {moves_made})")
                return mv
        except Exception as e:
            debug(f"Late-game heuristic raised: {e}")

    # Multiplayer (3+ active colours): only fall back to advanced heuristic if
    # the loaded model is the legacy 2-player one (encoder mode != multicolour).
    # The d22+ multicolour models handle N=2..6 natively.
    n_active_colours = len([c for c in pins if pins[c]])
    if n_active_colours >= 3 and getattr(agent, "encoder_mode", "legacy") != "multicolour":
        try:
            mv = advanced_choose(legal_moves, pin_positions, board.board, goal_indices)
            if mv is not None and _is_legal(mv, legal_moves) and not _would_cycle(mv, recent_moves):
                debug(f"Legacy-model multiplayer fallback: heuristic chose "
                      f"pin {mv[0]} -> {mv[1]} ({n_active_colours} colours)")
                return mv
        except Exception as e:
            debug(f"Multiplayer heuristic raised: {e}")

    # Per-move deadline relative to wall clock
    move_start = time.perf_counter()
    per_move_budget = min(PER_MOVE_HARD_CAP, max(0.5, remaining - 1.0))
    deadline = move_start + per_move_budget

    # Defensive: shrink sims if the per-move wall-clock budget is tight.
    # Use measured forward-pass cost (already factored into choose_sims),
    # but apply a per-move cap so a single slow position can't blow up.
    ms_budget = per_move_budget * 1000 - 100  # 100ms overhead reserve
    per_sim_ms = forward_ms * 0.8  # batched amortisation
    capped = max(MIN_SIMS, int(ms_budget / max(per_sim_ms, 1.0)))
    sims_eff = max(MIN_SIMS, min(sims, capped))

    # 1a) SCM-modulated MCTS — preferred when SCM is loaded. Uses standard
    # (non-batched) MCTS because only it exposes search-stat extraction.
    if (remaining > RAW_POLICY_BUDGET_SEC and agent.use_torch
            and getattr(agent, "scm", None) is not None):
        try:
            mv = agent.select_with_scm_mcts(board, my_colour, opp_colour,
                                            sims=sims_eff, deadline=deadline,
                                            turn_order=turn_order,
                                            blend=SCM_BLEND)
            if mv is not None and _is_legal(mv, legal_moves):
                if _would_cycle(mv, recent_moves):
                    alt = _alt_non_cycling_move(legal_moves, pin_positions,
                                                board.board, goal_indices,
                                                recent_moves, exclude=mv)
                    if alt is not None:
                        debug(f"SCM+MCTS chose {mv} but would cycle; using alt {alt}")
                        return alt
                debug(f"SCM+MCTS chose pin {mv[0]} -> {mv[1]} "
                      f"(sims={sims_eff}, blend={SCM_BLEND}, "
                      f"t={time.perf_counter()-move_start:.2f}s)")
                return mv
        except Exception as e:
            debug(f"SCM layer raised: {e}")

    # 1b) Batched MCTS (no SCM)
    if remaining > RAW_POLICY_BUDGET_SEC and agent.use_torch:
        try:
            mv = agent.select_with_mcts(board, my_colour, opp_colour,
                                        sims=sims_eff, batch_size=MCTS_BATCH_SIZE,
                                        deadline=deadline, turn_order=turn_order)
            if mv is not None and _is_legal(mv, legal_moves):
                # Cycle guard: if this move would put us back where we just
                # came from (2-move pingpong), try to pick a different move.
                if _would_cycle(mv, recent_moves):
                    alt = _alt_non_cycling_move(legal_moves, pin_positions,
                                                board.board, goal_indices,
                                                recent_moves, exclude=mv)
                    if alt is not None:
                        debug(f"MCTS chose {mv} but would cycle; using alt {alt}")
                        return alt
                debug(f"MCTS chose pin {mv[0]} -> {mv[1]} (sims={sims_eff}, "
                      f"t={time.perf_counter()-move_start:.2f}s)")
                return mv
        except Exception as e:
            debug(f"MCTS layer raised: {e}")

    # 2) Raw policy fallback
    if agent.use_torch and (time.perf_counter() < deadline):
        try:
            mv = agent.select_with_raw_policy(board, my_colour, opp_colour, legal_moves,
                                              turn_order=turn_order)
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


def _would_cycle(mv: Tuple[int, int],
                 recent_moves: Optional[List[Tuple[int, int]]]) -> bool:
    """Detect cycles in the last few moves.

    Flags `mv` as a cycle if:
      (a) the same (pin, dest) tuple appears in the last 4 moves, OR
      (b) the proposed move undoes the most recent move (i.e. pin would
          return to where it was 2 plies ago).
    """
    if not recent_moves:
        return False
    # (a) exact repetition within recent window
    if mv in recent_moves[-4:]:
        return True
    # (b) immediate undo: pin moved A→B last turn; now we'd move B→A
    if len(recent_moves) >= 1:
        last = recent_moves[-1]
        if mv[0] == last[0] and mv[1] != last[1]:
            # Same pin, different dest. If destination equals the cell the pin
            # was AT before its last move we'd undo it — but we only know the
            # "from" cell from the move pair. Approximation: if the same
            # (pin, target) shows up in moves 2-3 ago, we're pingponging.
            if len(recent_moves) >= 2 and mv == recent_moves[-2]:
                return True
    return False


def _alt_non_cycling_move(legal_moves: Dict[int, List[int]],
                          pin_positions: Dict[int, int],
                          board: HexBoard,
                          goal_indices: List[int],
                          recent_moves: Optional[List[Tuple[int, int]]],
                          exclude: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Find the next-best non-cycling move using greedy heuristic.

    Falls back to any non-excluded legal move if no good alternative.
    """
    goal_set = set(goal_indices)
    best = None
    best_score = None
    for pin_id, dests in legal_moves.items():
        for dest in dests:
            if (pin_id, dest) == exclude:
                continue
            if _would_cycle((pin_id, dest), recent_moves):
                continue
            cur = pin_positions.get(pin_id, 0)
            d_cur = _min_dist_to_goal(board, cur, goal_indices)
            d_dest = _min_dist_to_goal(board, dest, goal_indices)
            score = d_cur - d_dest
            if dest in goal_set:
                score += 5.0
            if best_score is None or score > best_score:
                best_score = score
                best = (pin_id, dest)
    if best is not None:
        return best
    # Last resort: any move that isn't the excluded one
    for pin_id, dests in legal_moves.items():
        for dest in dests:
            if (pin_id, dest) != exclude:
                return (pin_id, dest)
    return None


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

    # Reset per-game state (SCM GRU hidden, turn counter)
    try:
        agent.new_game()
    except Exception as e:
        debug(f"agent.new_game() failed: {e}")

    # Wait for game to be ready
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        s = st.get("state", {}) if st.get("ok") else {}
        if s.get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    # CC_NO_AUTOSTART skips sending start ourselves — useful when an external
    # harness sends start for all N players simultaneously (otherwise the
    # first 2 players to send start auto-launch the game before the rest
    # have joined).
    no_autostart = os.getenv("CC_NO_AUTOSTART", "0") not in ("0", "", "false", "False")
    if no_autostart:
        print(f"PLAYER_ID={player_id}", flush=True)  # for harness
    elif os.getenv("CC_AUTOSTART", "0") not in ("0", "", "false", "False"):
        print("Auto-starting...")
        rpc({"op": "start", "game_id": game_id, "player_id": player_id})
        print("Sent START")
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
    # Track our recent moves for the cycle guard. Bounded ring buffer; the
    # guard only reads the last 3-4 entries.
    recent_moves: List[Tuple[int, int]] = []

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
                    recent_moves=recent_moves,
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

            # Update recent-moves ring buffer (cap at 6)
            recent_moves.append((int(pid), int(dest)))
            if len(recent_moves) > 6:
                recent_moves.pop(0)

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
