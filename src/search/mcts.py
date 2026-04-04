"""
mcts.py — PUCT Monte Carlo Tree Search for Chinese Checkers.

AlphaZero-style MCTS using the policy network to guide expansion and the
value network to back up leaf evaluations.

Usage
-----
    from src.search.mcts import MCTSAgent
    agent = MCTSAgent(model, num_simulations=200)
    action = agent.select_action(env, temperature=1.0)

Key design decisions:
  - Tree nodes store visit counts N, total value W, and prior probability P
  - PUCT selection:  Q(s,a) + c_puct * P(s,a) * sqrt(sum_N) / (1 + N(s,a))
  - Leaf evaluation: value network prediction (no rollouts)
  - env.clone() enables tree branching without side effects
  - Temperature controls action sampling: temp=1 during training, temp→0 for tournament
"""

import math
import numpy as np
from typing import Optional


# Default PUCT exploration constant
_DEFAULT_C_PUCT = 1.5


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

    __slots__ = ('parent', 'action', 'prior', 'N', 'W', 'children', 'is_expanded')

    def __init__(self, parent: Optional["MCTSNode"], action: Optional[int], prior: float):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.N: int = 0
        self.W: float = 0.0
        self.children: dict[int, "MCTSNode"] = {}
        self.is_expanded: bool = False

    @property
    def Q(self) -> float:
        """Mean action value."""
        return self.W / self.N if self.N > 0 else 0.0

    def puct_score(self, parent_N: int, c_puct: float) -> float:
        """PUCT score for selecting this node from its parent."""
        u = c_puct * self.prior * math.sqrt(parent_N) / (1 + self.N)
        return self.Q + u

    def is_root(self) -> bool:
        return self.parent is None


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


def _heuristic_value(env) -> float:
    """Evaluate board position using game heuristics.

    Returns value in [-1, 1] where 1.0 = all pins in goal, -1.0 = worst.
    Uses pins_in_goal and distance_to_goal which directly correlate with winning.
    """
    board = env._board
    if board is None:
        return 0.0

    colour = env._AGENT_COLOUR
    pins_in_goal = board.pins_in_goal(colour)
    total_dist = board.total_distance_to_goal(colour)

    if pins_in_goal == 10:
        return 1.0

    # Max possible distance ~120 (10 pins × ~12 avg distance at start).
    # Normalize: more pins in goal and less distance = higher value.
    # pins_in_goal: 0-10 → 0.0-1.0
    # distance: 0-120 → 1.0-0.0
    pin_score = pins_in_goal / 10.0
    dist_score = max(0.0, 1.0 - total_dist / 120.0)

    # Combine: weighted average, map to [-1, 1]
    raw = 0.6 * pin_score + 0.4 * dist_score  # range [0, 1]
    return raw * 2.0 - 1.0  # map to [-1, 1]


class MCTS:
    """PUCT Monte Carlo Tree Search.

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
    """

    def __init__(
        self,
        model,
        num_simulations: int = 200,
        c_puct: float = _DEFAULT_C_PUCT,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def _select(self, node: MCTSNode, sim_env) -> tuple[MCTSNode, bool]:
        """Traverse tree using PUCT until a leaf node is reached.

        Also steps sim_env through each action so it matches the leaf state.

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
                score = child.puct_score(parent_N, self.c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
            # Step the sim env to match the tree traversal
            _, reward, terminated, truncated, _ = sim_env.step(best_child.action)
            node = best_child
            if terminated or truncated:
                terminal = True
                break
        return node, terminal

    def _expand(self, node: MCTSNode, env) -> float:
        """Expand a leaf node: policy network for priors, heuristic for value.

        Returns the heuristic value estimate in [-1, 1].
        """
        obs = env._get_obs()
        action_mask = env.action_masks()

        if action_mask.sum() == 0:
            node.is_expanded = True
            return 0.0

        priors = _get_policy_priors(self.model, obs, action_mask)
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

    def _backup(self, node: MCTSNode, value: float):
        """Propagate value back up the tree."""
        current = node
        while current is not None:
            current.N += 1
            current.W += value
            # Flip sign at each level (alternating players)
            # In self-play, the opponent's good state = our bad state.
            # Since this is single-agent env, we keep sign consistent.
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

        for _ in range(self.num_simulations):
            # 1. Clone env for this simulation
            sim_env = env.clone()

            # 2. Selection: follow PUCT until leaf (also steps sim_env)
            node, terminal = self._select(root, sim_env)

            if terminal:
                # Game ended during tree traversal — use actual outcome
                agent_won = sim_env._board.check_win(sim_env._AGENT_COLOUR) if sim_env._board is not None else False
                value = 1.0 if agent_won else -1.0
            else:
                # 3. Expansion + evaluation
                value = self._expand(node, sim_env)

            # 4. Backup
            self._backup(node, value)

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
