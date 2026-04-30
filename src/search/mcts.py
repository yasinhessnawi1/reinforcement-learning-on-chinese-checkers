"""
mcts.py — PUCT Monte Carlo Tree Search for Chinese Checkers.

AlphaZero-style MCTS using the policy network to guide expansion and the
value network (or heuristic) to back up leaf evaluations.

Supports min-max Q-value normalization (MuZero-style) so the value scale
doesn't matter — works with both heuristic [-1,1] and PPO cumulative reward.

Two-player mode: after the agent's move, the opponent responds using a fast
deterministic policy (greedy or advanced heuristic). This models blocking
without exploding the tree — opponent moves are deterministic "chance" edges,
not full PUCT subtrees.

Usage
-----
    from src.search.mcts import MCTSAgent, MCTS
    agent = MCTSAgent(model, num_simulations=200)
    action = agent.select_action(env, temperature=1.0)

Key design decisions:
  - Tree nodes store visit counts N, total value W, and prior probability P
  - PUCT selection:  Q_norm(s,a) + c_puct * P(s,a) * sqrt(sum_N) / (1 + N(s,a))
  - Q_norm uses min-max normalization across the tree (MuZero approach)
  - Leaf evaluation: heuristic or PPO value network (configurable)
  - env.clone() enables tree branching without side effects
  - Temperature controls action sampling: temp=1 during training, temp→0 for tournament
  - Two-player: opponent responds after each agent move using a fast policy
"""

import math
import numpy as np
from typing import Optional


# Default PUCT exploration constant
_DEFAULT_C_PUCT = 1.5


class MinMaxStats:
    """Track min/max Q-values across the search tree for normalization.

    MuZero-style: normalizes Q-values to [0, 1] so PUCT works regardless
    of the value scale (heuristic [-1,1] or PPO cumulative [0,10+]).
    """

    def __init__(self):
        self.min_value: float = float('inf')
        self.max_value: float = float('-inf')

    def update(self, value: float):
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] based on observed range."""
        if self.max_value > self.min_value:
            return (value - self.min_value) / (self.max_value - self.min_value)
        return 0.5  # No range yet — neutral


class MCTSNode:
    """A node in the MCTS tree.

    Attributes
    ----------
    parent : MCTSNode or None
    action : int or None — action taken from parent to reach this node
    prior : float — policy network prior P(parent_state, action)
    N : int — visit count
    W : float — total backed-up value
    children : dict[int, MCTSNode]
    is_expanded : bool
    """

    __slots__ = ('parent', 'action', 'prior', 'N', 'W', 'children', 'is_expanded', 'colour')

    def __init__(self, parent: Optional["MCTSNode"], action: Optional[int], prior: float):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.N: int = 0
        self.W: float = 0.0
        self.children: dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False
        self.colour: str = ""  # Set by two-player MCTS to track whose turn

    @property
    def Q(self) -> float:
        """Mean action value (raw, unnormalized)."""
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, parent_N: int, c_puct: float, min_max: MinMaxStats) -> float:
        """PUCT score with min-max normalized Q value."""
        q_norm = min_max.normalize(self.Q) if self.N > 0 else 0.5
        u = c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return q_norm + u

    def is_root(self) -> bool:
        return self.parent is None


def _get_heuristic_priors(env) -> np.ndarray:
    """Generate action priors from heuristic move scoring (no neural network).

    Scores each legal move using the same components as the advanced heuristic
    (immediate improvement + lookahead), then applies softmax to get a
    probability distribution. This gives MCTS domain-informed priors without
    depending on the PPO policy network.
    """
    from src.agents.advanced_heuristic import (
        _score_position, _best_lookahead_score,
    )

    board = env._board
    colour = env._AGENT_COLOUR
    goal_indices = board.get_goal_indices(colour)
    goal_set = set(goal_indices)

    current_score = _score_position(board, colour, goal_indices, goal_set)

    num_actions = env.action_space.n
    scores = np.full(num_actions, -1e9, dtype=np.float32)

    legal_moves = board.get_legal_moves(colour)
    mapper = env._mapper

    for pin_id, dests in legal_moves.items():
        pin = board._pin_by_id(colour, pin_id)
        current_pos = pin.axialindex
        in_goal_now = current_pos in goal_set

        for dest in dests:
            action = mapper.encode(pin_id, dest)

            if in_goal_now and dest not in goal_set:
                scores[action] = -100.0
                continue

            pin.axialindex = dest
            new_score = _score_position(board, colour, goal_indices, goal_set)
            immediate = new_score - current_score

            lookahead = _best_lookahead_score(board, colour, goal_indices, goal_set)
            lookahead_gain = lookahead - new_score

            move_dist = board.axial_distance(current_pos, dest)
            hop_bonus = move_dist * 0.3 if move_dist > 1 else 0.0

            pin.axialindex = current_pos

            scores[action] = immediate * 1.0 + lookahead_gain * 0.6 + hop_bonus

    # Softmax over legal actions only (temperature=1.0)
    legal_mask = scores > -1e8
    if not legal_mask.any():
        # Fallback: uniform
        action_mask = env.action_masks()
        priors = action_mask.astype(np.float32)
        return priors / priors.sum() if priors.sum() > 0 else priors

    max_score = scores[legal_mask].max()
    exp_scores = np.zeros(num_actions, dtype=np.float32)
    exp_scores[legal_mask] = np.exp(scores[legal_mask] - max_score)
    priors = exp_scores / exp_scores.sum()
    return priors


def _get_policy_priors(model, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Run the policy network to get action priors.

    Parameters
    ----------
    model : MaskablePPO (sb3-contrib) — must have .policy attribute
    obs : np.ndarray, shape (C, H, W)
    action_mask : np.ndarray, shape (num_actions,), dtype bool

    Returns
    -------
    priors : np.ndarray, shape (num_actions,) — softmax policy probabilities
    """
    import torch
    policy = model.policy
    device = next(policy.parameters()).device

    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32, device=device)
    mask_t = torch.tensor(action_mask[np.newaxis], dtype=torch.bool, device=device)

    with torch.no_grad():
        features = policy.extract_features(obs_t)
        if policy.share_features_extractor:
            latent_pi, _ = policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = policy.mlp_extractor.forward_actor(pi_features)

        logits = policy.action_net(latent_pi)
        logits = logits.masked_fill(~mask_t, -1e9)
        priors = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return priors


def _get_network_value(model, obs: np.ndarray) -> float:
    """Run the value network to get a state value estimate.

    Parameters
    ----------
    model : MaskablePPO
    obs : np.ndarray, shape (C, H, W)

    Returns
    -------
    float — raw value estimate (scale depends on PPO training)
    """
    import torch
    policy = model.policy
    device = next(policy.parameters()).device

    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32, device=device)

    with torch.no_grad():
        features = policy.extract_features(obs_t)
        if policy.share_features_extractor:
            _, latent_vf = policy.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = policy.mlp_extractor.forward_critic(vf_features)

        value = policy.value_net(latent_vf).squeeze().cpu().item()

    return value


def _score_colour(board, colour: str) -> float:
    """Score a colour's position: pins_in_goal * 100 + max(0, 200 - total_dist) + bonuses.

    Returns raw score (not normalized). Used for both agent and opponent evaluation.
    """
    goal_indices = board.get_goal_indices(colour)
    goal_set = set(goal_indices)
    pins_in_goal = board.pins_in_goal(colour)

    # Per-pin distances (only for pins not in goal)
    pin_dists = []
    for pin in board.pins[colour]:
        if pin.axialindex in goal_set:
            continue
        min_d = min(board.axial_distance(pin.axialindex, g) for g in goal_indices)
        pin_dists.append(min_d)

    total_dist = sum(pin_dists)

    # Base score (same as advanced_heuristic._score_position)
    base = pins_in_goal * 100.0 + max(0.0, 200.0 - total_dist)

    # Hop potential: pins with multi-hop moves can advance faster
    hop_bonus = 0.0
    legal = board.get_legal_moves(colour)
    for pin_id, dests in legal.items():
        pin = board._pin_by_id(colour, pin_id)
        for dest in dests:
            if board.axial_distance(pin.axialindex, dest) > 2:
                hop_bonus += 3.0
                break

    # Near-goal bonus: pins within 2 steps are almost scored
    near_count = sum(1 for d in pin_dists if d <= 2)
    near_bonus = near_count * 8.0

    # Straggler penalty: furthest pin holds back the win
    straggler_penalty = 0.0
    if pin_dists:
        max_d = max(pin_dists)
        if max_d > 10:
            straggler_penalty = (max_d - 10) * 3.0

    return base + hop_bonus + near_bonus - straggler_penalty


def _heuristic_value(env) -> float:
    """Evaluate board position for MCTS value backup.

    Returns value in [-1, 1].

    Opponent-aware evaluation:
      - Scores both agent and opponent positions
      - Returns relative advantage: agent_score - opponent_score (normalized)
      - This makes the heuristic responsive to blocking and competitive dynamics

    Score components per colour:
      - pins_in_goal * 100 + max(0, 200 - total_distance)
      - hop_potential: pins with multi-hop moves available
      - near_goal: bonus for pins within 2 steps of goal
      - straggler: penalty for furthest-behind pin
    """
    board = env._board
    if board is None:
        return 0.0

    colour = env._AGENT_COLOUR
    agent_pins_in_goal = board.pins_in_goal(colour)

    if agent_pins_in_goal == 10:
        return 1.0

    if not env._no_opponent:
        opp_colour = env._OPPONENT_COLOUR
        if opp_colour in board.pins and board.check_win(opp_colour):
            return -1.0

    agent_score = _score_colour(board, colour)

    # Opponent-aware: compute relative advantage.
    # Check for opponent pins on the board regardless of _no_opponent flag,
    # because MCTS clones strip the opponent policy but keep opponent pins.
    opp_colour = env._OPPONENT_COLOUR
    if opp_colour in board.pins and len(board.pins[opp_colour]) > 0:
        opp_score = _score_colour(board, opp_colour)
        # Relative advantage: positive = agent ahead, negative = behind
        # Both scores are in range ~100-1200
        # Difference range: ~-1100 to +1100
        relative = agent_score - opp_score
        normalized = relative / 1100.0
        return max(-1.0, min(1.0, normalized))

    # True solo mode (no opponent pins on board at all)
    normalized = (agent_score - 650.0) / 550.0
    return max(-1.0, min(1.0, normalized))


class MCTS:
    """PUCT Monte Carlo Tree Search with min-max Q-value normalization.

    Parameters
    ----------
    model : MaskablePPO
        Trained policy with .policy for network inference.
    num_simulations : int
        Number of MCTS simulations per move decision.
    c_puct : float
        Exploration constant in PUCT formula.
    dirichlet_alpha : float
        Dirichlet noise alpha added to root priors (exploration during training).
    dirichlet_epsilon : float
        Fraction of Dirichlet noise mixed into root priors (0 = no noise).
    use_network_value : bool
        If True, use PPO value network for leaf evaluation (with min-max normalization).
        If False (default), use heuristic evaluation.
    opponent_policy : callable or None
        Policy for opponent moves inside the search tree.
        callable(board_wrapper, colour) -> (pin_id, dest_index).
        If provided, after each agent move in the tree the opponent responds
        deterministically, modeling blocking. If None, solo search (no opponent).
    """

    def __init__(
        self,
        model=None,
        num_simulations: int = 200,
        c_puct: float = _DEFAULT_C_PUCT,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_network_value: bool = False,
        opponent_policy=None,
        use_heuristic_priors: bool = False,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_network_value = use_network_value
        self.opponent_policy = opponent_policy
        self.use_heuristic_priors = use_heuristic_priors

    def _apply_opponent_move(self, env):
        """Apply a single deterministic opponent move on the board.

        Unlike env.step() which bundles agent+opponent, this directly
        manipulates the board so the tree sees the opponent's response
        without conflating it with the agent's action edges.

        Returns True if opponent won after their move.
        """
        if self.opponent_policy is None:
            return False

        board = env._board
        opp_colour = env._OPPONENT_COLOUR

        if opp_colour not in board.pins or len(board.pins[opp_colour]) == 0:
            return False

        opp_legal = board.get_legal_moves(opp_colour)
        if not opp_legal:
            return False

        opp_pin_id, opp_dest = self.opponent_policy(board, opp_colour)
        board.apply_move(opp_colour, opp_pin_id, opp_dest)
        return board.check_win(opp_colour)

    def _select(self, node: MCTSNode, sim_env, min_max: MinMaxStats) -> tuple[MCTSNode, bool]:
        """Traverse tree using PUCT until a leaf node is reached.

        Also steps sim_env through each action so it matches the leaf state.
        After each agent move, the opponent responds deterministically.

        Returns
        -------
        (node, terminal) — leaf node and whether the env reached terminal state.
        """
        terminal = False
        while node.is_expanded and node.children:
            parent_N = node.N
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                score = child.puct_score(parent_N, self.c_puct, min_max)
                if score > best_score:
                    best_score = score
                    best_child = child

            # Apply agent's move directly on the board (no opponent bundled)
            pin_id, dest = sim_env._mapper.decode(best_child.action)
            sim_env._board.apply_move(sim_env._AGENT_COLOUR, pin_id, dest)
            sim_env._step_count += 1
            node = best_child

            # Check if agent won
            if sim_env._board.check_win(sim_env._AGENT_COLOUR):
                terminal = True
                break

            # Check truncation
            if sim_env._step_count >= sim_env.max_steps:
                terminal = True
                break

            # Opponent responds
            if self._apply_opponent_move(sim_env):
                terminal = True
                break

        return node, terminal

    def _expand(self, node: MCTSNode, env) -> float:
        """Expand a leaf node: policy network for priors, heuristic/network for value.

        Returns the value estimate (scale handled by min-max normalization).
        """
        obs = env._get_obs()
        action_mask = env.action_masks()

        if action_mask.sum() == 0:
            node.is_expanded = True
            return 0.0

        if self.use_heuristic_priors:
            priors = _get_heuristic_priors(env)
        else:
            priors = _get_policy_priors(self.model, obs, action_mask)

        if self.use_network_value:
            value = _get_network_value(self.model, obs)
        else:
            value = _heuristic_value(env)

        # Add Dirichlet noise to root node for exploration diversity
        if node.is_root() and self.dirichlet_epsilon > 0:
            legal_actions = np.where(action_mask)[0]
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                priors[a] = (
                    (1 - self.dirichlet_epsilon) * priors[a]
                    + self.dirichlet_epsilon * noise[i]
                )

        # Create children for all legal actions
        legal_actions = np.where(action_mask)[0]
        for action in legal_actions:
            node.children[int(action)] = MCTSNode(
                parent=node,
                action=int(action),
                prior=float(priors[action]),
            )

        node.is_expanded = True
        return value

    def _backup(self, node: MCTSNode, value: float, min_max: MinMaxStats):
        """Propagate value back up the tree and update min-max stats."""
        current = node
        while current is not None:
            current.N += 1
            current.W += value
            min_max.update(current.Q)
            current = current.parent

    def run(self, env) -> MCTSNode:
        """Run MCTS simulations from the current env state.

        Parameters
        ----------
        env : ChineseCheckersEnv (after reset)
            Must support env.clone().

        Returns
        -------
        root : MCTSNode — the root node with visit counts populated.
        """
        root = MCTSNode(parent=None, action=None, prior=1.0)
        min_max = MinMaxStats()

        for _ in range(self.num_simulations):
            # Clone env for this simulation — always strip the env's opponent
            # since we handle opponent moves ourselves via _apply_opponent_move
            sim_env = env.clone(strip_opponent=True)

            # Selection: follow PUCT until leaf (also steps sim_env with opponent)
            node, terminal = self._select(root, sim_env, min_max)

            if terminal:
                # Game ended during tree traversal — use actual outcome
                agent_won = sim_env._board.check_win(sim_env._AGENT_COLOUR) if sim_env._board is not None else False
                value = 1.0 if agent_won else -1.0
            else:
                # Expansion + evaluation
                value = self._expand(node, sim_env)

            # Backup
            self._backup(node, value, min_max)

        return root

    def get_action_probs(self, env, temperature: float = 1.0) -> np.ndarray:
        """Run MCTS and return visit-count-based action distribution.

        Parameters
        ----------
        env : ChineseCheckersEnv
        temperature : float
            1.0 = proportional to visit counts (training)
            → 0 = argmax (tournament)

        Returns
        -------
        action_probs : np.ndarray, shape (num_actions,)
        """
        root = self.run(env)
        num_actions = env.action_space.n

        visits = np.zeros(num_actions, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.N

        if visits.sum() == 0:
            # Fallback: uniform over legal moves
            mask = env.action_masks()
            visits = mask.astype(np.float32)

        if temperature == 0 or temperature < 1e-6:
            # Deterministic: one-hot on argmax
            probs = np.zeros(num_actions, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:
            # Temperature scaling
            counts_temp = visits ** (1.0 / temperature)
            probs = counts_temp / counts_temp.sum()

        return probs

    def get_action_probs_and_value(self, env, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        """Run MCTS and return (action_probs, root_value).

        root_value is the visit-weighted average Q across children,
        representing the MCTS estimate of the current position's value.
        """
        root = self.run(env)
        num_actions = env.action_space.n

        visits = np.zeros(num_actions, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.N

        if visits.sum() == 0:
            mask = env.action_masks()
            visits = mask.astype(np.float32)

        if temperature == 0 or temperature < 1e-6:
            probs = np.zeros(num_actions, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:
            counts_temp = visits ** (1.0 / temperature)
            probs = counts_temp / counts_temp.sum()

        # Root value: visit-weighted average Q of children
        total_n = sum(c.N for c in root.children.values())
        if total_n > 0:
            root_value = sum(c.Q * c.N for c in root.children.values()) / total_n
        else:
            root_value = 0.0

        return probs, float(root_value)

    def select_action(self, env, temperature: float = 1.0) -> int:
        """Run MCTS and select an action.

        Parameters
        ----------
        env : ChineseCheckersEnv
        temperature : float

        Returns
        -------
        int — selected action index
        """
        probs = self.get_action_probs(env, temperature)
        if temperature == 0 or temperature < 1e-6:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))


class MCTSAgent:
    """Convenience wrapper adapting MCTS to the board_wrapper(colour) policy interface.

    Parameters
    ----------
    model : MaskablePPO
    num_simulations : int
    temperature : float
    """

    def __init__(self, model, num_simulations: int = 200, temperature: float = 0.0):
        self._mcts = MCTS(model, num_simulations=num_simulations)
        self._temperature = temperature

    def __call__(self, board_wrapper, colour: str):
        """Policy interface: board_wrapper, colour -> (pin_id, dest)."""
        raise NotImplementedError(
            "MCTSAgent cannot be called as a board_wrapper policy directly — "
            "it requires a full ChineseCheckersEnv.  Use MCTS.select_action(env) instead."
        )


# ---------------------------------------------------------------------------
# AlphaZero-compatible MCTS (uses AlphaZeroNet instead of SB3 model)
# ---------------------------------------------------------------------------

class AlphaZeroMCTS:
    """PUCT MCTS that uses AlphaZeroNet for policy priors and value estimation.

    Supports proper two-player search: each node stores whose turn it is,
    children represent moves by the current player, and values are negated
    during backup at player-change boundaries. This is essential for
    competitive play — the tree must model the opponent's responses.

    Parameters
    ----------
    network : AlphaZeroNet
        Neural network for policy priors and value estimation.
    num_simulations : int
        Number of MCTS simulations per move.
    c_puct : float
        Exploration constant in PUCT formula.
    dirichlet_alpha : float
        Dirichlet noise alpha for root exploration.
    dirichlet_epsilon : float
        Fraction of Dirichlet noise mixed into root priors (0 = no noise).
    use_heuristic_value : bool
        If True, use heuristic evaluation instead of network value head.
    opponent_policy : callable or None
        Deterministic opponent policy for 2-player search (legacy mode).
        When two_player=True, this is ignored — the network plays both sides.
    two_player : bool
        If True, use proper alternating two-player MCTS where both sides
        are searched using the network. Values are negated at each level.
    """

    def __init__(
        self,
        network=None,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_heuristic_value: bool = False,
        opponent_policy=None,
        two_player: bool = False,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_heuristic_value = use_heuristic_value
        self.opponent_policy = opponent_policy
        self.two_player = two_player

    def _apply_opponent_move(self, env) -> bool:
        """Apply opponent move. Returns True if opponent won."""
        if self.opponent_policy is None:
            return False

        board = env._board
        opp_colour = env._OPPONENT_COLOUR

        if opp_colour not in board.pins or len(board.pins[opp_colour]) == 0:
            return False

        opp_legal = board.get_legal_moves(opp_colour)
        if not opp_legal:
            return False

        opp_pin_id, opp_dest = self.opponent_policy(board, opp_colour)
        board.apply_move(opp_colour, opp_pin_id, opp_dest)
        return board.check_win(opp_colour)

    # ------------------------------------------------------------------
    # Two-player alternating MCTS (proper AlphaZero approach)
    # ------------------------------------------------------------------

    def _select_two_player(self, node: MCTSNode, board, mapper, min_max: MinMaxStats,
                           colours: list[str], max_steps: int, step_count: int) -> tuple:
        """Traverse tree using PUCT, alternating players at each level.

        Each node stores the colour of the player who moved to reach it via
        node.colour. Children represent moves by the OTHER player.

        Returns (node, terminal, leaf_colour, step_count).
        leaf_colour is the colour whose turn it is at the leaf.
        """
        terminal = False
        # Determine whose turn at root level: root.colour is the player
        # who will move from this position (set in run_two_player)
        current_colour_idx = colours.index(node.colour) if hasattr(node, 'colour') else 0

        while node.is_expanded and node.children:
            parent_N = node.N
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                score = child.puct_score(parent_N, self.c_puct, min_max)
                if score > best_score:
                    best_score = score
                    best_child = child

            # The node's colour is whose turn it is — apply their move
            moving_colour = colours[current_colour_idx]
            pin_id, dest = mapper.decode(best_child.action)
            board.apply_move(moving_colour, pin_id, dest)
            step_count += 1
            node = best_child

            if board.check_win(moving_colour):
                terminal = True
                break

            if step_count >= max_steps:
                terminal = True
                break

            # Alternate to other player
            current_colour_idx = 1 - current_colour_idx

        leaf_colour = colours[current_colour_idx]
        return node, terminal, leaf_colour, step_count

    def _expand_two_player(self, node: MCTSNode, board, mapper, encoder,
                           colour: str, turn_order: list[str]) -> float:
        """Expand leaf node for two-player MCTS.

        Evaluates position from `colour`'s perspective.
        Returns value from `colour`'s perspective.
        """
        obs = encoder.encode(board, current_colour=colour, turn_order=turn_order)
        legal_moves = board.get_legal_moves(colour)
        action_mask = mapper.build_action_mask(legal_moves)

        if action_mask.sum() == 0:
            node.is_expanded = True
            return 0.0

        priors, value = self.network.predict(obs, action_mask)

        if self.use_heuristic_value:
            # Heuristic from this colour's perspective
            agent_score = _score_colour(board, colour)
            opp_colour = turn_order[1] if colour == turn_order[0] else turn_order[0]
            if opp_colour in board.pins and len(board.pins[opp_colour]) > 0:
                opp_score = _score_colour(board, opp_colour)
                relative = agent_score - opp_score
                value = max(-1.0, min(1.0, relative / 1100.0))
            else:
                value = max(-1.0, min(1.0, (agent_score - 650.0) / 550.0))

        # Dirichlet noise at root
        if node.is_root() and self.dirichlet_epsilon > 0:
            legal_actions = np.where(action_mask)[0]
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                priors[a] = (
                    (1 - self.dirichlet_epsilon) * priors[a]
                    + self.dirichlet_epsilon * noise[i]
                )

        legal_actions = np.where(action_mask)[0]
        for action in legal_actions:
            node.children[int(action)] = MCTSNode(
                parent=node,
                action=int(action),
                prior=float(priors[action]),
            )

        node.is_expanded = True
        return value

    def _backup_two_player(self, node: MCTSNode, value: float, min_max: MinMaxStats):
        """Propagate value up the tree, negating at each level.

        In a proper two-player alternating tree, every level represents
        a different player. Value from the leaf's perspective is negated
        at each step up — what's good for one player is bad for the other.
        """
        current = node
        current_value = value
        while current is not None:
            current.N += 1
            current.W += current_value
            min_max.update(current.Q)
            if current.parent is not None:
                current_value = -current_value
            current = current.parent

    def run_two_player(self, board, mapper, encoder, root_colour: str,
                       turn_order: list[str], step_count: int = 0,
                       max_steps: int = 200) -> MCTSNode:
        """Run proper two-player alternating MCTS.

        Parameters
        ----------
        board : BoardWrapper — the current game state (will be cloned per sim).
        mapper : ActionMapper
        encoder : StateEncoder
        root_colour : str — whose turn it is at the root.
        turn_order : list[str] — e.g. ["red", "blue"]
        step_count : int — current total half-moves
        max_steps : int — truncation limit (total half-moves)

        Returns
        -------
        MCTSNode — root node with visit counts.
        """
        root = MCTSNode(parent=None, action=None, prior=1.0)
        root.colour = root_colour  # Track whose turn at root
        min_max = MinMaxStats()
        colours = list(turn_order)

        for _ in range(self.num_simulations):
            sim_board = board.clone()
            sim_step = step_count

            node, terminal, leaf_colour, sim_step = self._select_two_player(
                root, sim_board, mapper, min_max, colours, max_steps, sim_step
            )

            if terminal:
                # Determine who won from root_colour's perspective
                root_idx = colours.index(root_colour)
                opp_colour = colours[1 - root_idx]
                if sim_board.check_win(root_colour):
                    value = 1.0
                elif sim_board.check_win(opp_colour):
                    value = -1.0
                else:
                    # Truncated — use heuristic
                    agent_score = _score_colour(sim_board, root_colour)
                    opp_score = _score_colour(sim_board, opp_colour)
                    value = max(-1.0, min(1.0, (agent_score - opp_score) / 1100.0))
                # Convert to leaf_colour's perspective for backup
                if leaf_colour != root_colour:
                    value = -value
            else:
                value = self._expand_two_player(
                    node, sim_board, mapper, encoder, leaf_colour, turn_order
                )

            self._backup_two_player(node, value, min_max)

        return root

    # ------------------------------------------------------------------
    # Legacy single-player MCTS (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode, sim_env, min_max: MinMaxStats) -> tuple:
        """Traverse tree using PUCT until a leaf node (legacy single-player)."""
        terminal = False
        while node.is_expanded and node.children:
            parent_N = node.N
            best_score = -float('inf')
            best_child = None
            for child in node.children.values():
                score = child.puct_score(parent_N, self.c_puct, min_max)
                if score > best_score:
                    best_score = score
                    best_child = child

            pin_id, dest = sim_env._mapper.decode(best_child.action)
            sim_env._board.apply_move(sim_env._AGENT_COLOUR, pin_id, dest)
            sim_env._step_count += 1
            node = best_child

            if sim_env._board.check_win(sim_env._AGENT_COLOUR):
                terminal = True
                break

            if sim_env._step_count >= sim_env.max_steps:
                terminal = True
                break

            if self._apply_opponent_move(sim_env):
                terminal = True
                break

        return node, terminal

    def _expand(self, node: MCTSNode, env) -> float:
        """Expand leaf node using AlphaZeroNet for priors and value (legacy).

        Perspective handling: env._get_obs() returns the rotated obs for
        rotated colours (blue/gray0/purple), so the network outputs priors in
        the canonical (rotated) frame. We rotate the action_mask into the
        canonical frame for masking, then rotate priors back to the raw
        frame so the rest of MCTS (which uses raw cell indices) works
        unchanged.
        """
        obs = env._get_obs()
        action_mask = env.action_masks()  # raw frame

        if action_mask.sum() == 0:
            node.is_expanded = True
            return 0.0

        encoder = env._encoder
        rotated = encoder.needs_rotation(env._AGENT_COLOUR) if hasattr(encoder, 'needs_rotation') else False

        if rotated:
            mask_for_net = encoder.rotate_action_distribution(
                action_mask.astype(np.bool_)
            ).astype(np.bool_)
            priors_canon, value = self.network.predict(obs, mask_for_net)
            # Rotate priors back to raw frame so MCTS indexing matches
            # action_mask / legal_actions / mapper.decode().
            priors = encoder.rotate_action_distribution(priors_canon)
        else:
            priors, value = self.network.predict(obs, action_mask)

        if self.use_heuristic_value:
            value = _heuristic_value(env)

        if node.is_root() and self.dirichlet_epsilon > 0:
            legal_actions = np.where(action_mask)[0]
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                priors[a] = (
                    (1 - self.dirichlet_epsilon) * priors[a]
                    + self.dirichlet_epsilon * noise[i]
                )

        legal_actions = np.where(action_mask)[0]
        for action in legal_actions:
            node.children[int(action)] = MCTSNode(
                parent=node,
                action=int(action),
                prior=float(priors[action]),
            )

        node.is_expanded = True
        return value

    def _backup(self, node: MCTSNode, value: float, min_max: MinMaxStats):
        """Propagate value up the tree (legacy, no negation)."""
        current = node
        while current is not None:
            current.N += 1
            current.W += value
            min_max.update(current.Q)
            current = current.parent

    def run(self, env) -> MCTSNode:
        """Run MCTS simulations using the legacy single-player tree.

        If opponent_policy is set, opponent responses are applied after each
        agent move in the tree via _apply_opponent_move. This models blocking
        without requiring alternating-perspective evaluation.

        The two_player mode (run_two_player) is available for direct calls
        but not used by default — it requires the network to evaluate from
        both perspectives, which needs symmetric training data.
        """
        root = MCTSNode(parent=None, action=None, prior=1.0)
        min_max = MinMaxStats()

        for _ in range(self.num_simulations):
            sim_env = env.clone(strip_opponent=True)
            node, terminal = self._select(root, sim_env, min_max)

            if terminal:
                agent_won = (
                    sim_env._board.check_win(sim_env._AGENT_COLOUR)
                    if sim_env._board is not None
                    else False
                )
                value = 1.0 if agent_won else _heuristic_value(sim_env)
            else:
                value = self._expand(node, sim_env)

            self._backup(node, value, min_max)

        return root

    def _visits_to_probs(self, root: MCTSNode, num_actions: int,
                         action_mask: np.ndarray, temperature: float) -> np.ndarray:
        """Convert root visit counts to action probabilities."""
        visits = np.zeros(num_actions, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.N

        if visits.sum() == 0:
            visits = action_mask.astype(np.float32)

        if temperature == 0 or temperature < 1e-6:
            probs = np.zeros(num_actions, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:
            counts_temp = visits ** (1.0 / temperature)
            total = counts_temp.sum()
            probs = counts_temp / total if total > 0 else counts_temp

        return probs

    def get_action_probs(self, env, temperature: float = 1.0) -> np.ndarray:
        """Run MCTS and return visit-count-based action distribution."""
        root = self.run(env)
        num_actions = env.action_space.n
        mask = env.action_masks()
        return self._visits_to_probs(root, num_actions, mask, temperature)

    def get_action_probs_and_value(self, env, temperature: float = 1.0) -> tuple[np.ndarray, float]:
        """Run MCTS and return (action_probs, root_value)."""
        root = self.run(env)
        num_actions = env.action_space.n
        mask = env.action_masks()
        probs = self._visits_to_probs(root, num_actions, mask, temperature)

        total_n = sum(c.N for c in root.children.values())
        if total_n > 0:
            root_value = sum(c.Q * c.N for c in root.children.values()) / total_n
        else:
            root_value = 0.0

        return probs, float(root_value)

    def select_action(self, env, temperature: float = 1.0) -> int:
        """Run MCTS and select an action."""
        probs = self.get_action_probs(env, temperature)
        if temperature == 0 or temperature < 1e-6:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))
