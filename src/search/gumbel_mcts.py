"""
gumbel_mcts.py — Gumbel AlphaZero MCTS for Chinese Checkers.

Implements the Gumbel MuZero / Gumbel AlphaZero action selection from
Danihelka et al. (ICLR 2022): "Policy improvement by planning with Gumbel."

Key insight: by adding Gumbel noise and using Sequential Halving at the root,
we can guarantee policy improvement with as few as 2-16 simulations — compared
to 800+ for standard AlphaZero MCTS.

Algorithm at the root:
  1. Sample Gumbel noise g(a) for each legal action a.
  2. Compute scores: g(a) + logit(a) where logit = log π(a|s) from the network.
  3. Select top-k actions by score (k = num_considered_actions).
  4. Sequential Halving: repeatedly halve the candidate set, allocating equal
     simulations to remaining candidates each round.
  5. Final selection: pick the action maximizing σ(g(a) + logit(a) + q̂(a)),
     where q̂(a) is the completed Q-value.

Interior nodes use standard PUCT selection (unchanged from AlphaZero).

References:
  - Danihelka et al. (2022) "Policy improvement by planning with Gumbel"
  - Hubert et al. (2021) "Learning and Planning in Complex Action Spaces"
"""

import math
import numpy as np
from typing import Optional

from src.search.mcts import MCTSNode, MinMaxStats, _heuristic_value


def _sample_gumbel(shape: tuple, eps: float = 1e-10) -> np.ndarray:
    """Sample from the standard Gumbel distribution: -log(-log(U))."""
    u = np.random.uniform(0, 1, shape)
    return -np.log(-np.log(u + eps) + eps)


def _sigma(logits: np.ndarray, max_value: float = 50.0) -> np.ndarray:
    """Numerically stable transformation: sigma(x) = log(sum(exp(x_i))) for normalization.

    Not needed for selection — we just use raw scores for argmax.
    """
    return logits


def _completed_q(node: MCTSNode, action: int, min_max: MinMaxStats) -> float:
    """Get the completed Q-value for an action at the root.

    If the action has been visited, return the normalized Q-value.
    If not visited, return the value estimate from the parent (prior mixed value).
    """
    if action in node.children and node.children[action].N > 0:
        child = node.children[action]
        return min_max.normalize(child.Q)
    # Unvisited action: use average value of visited children, or 0.5
    visited_qs = [
        min_max.normalize(c.Q)
        for c in node.children.values()
        if c.N > 0
    ]
    return np.mean(visited_qs) if visited_qs else 0.5


class GumbelMCTS:
    """Gumbel AlphaZero MCTS with Sequential Halving at root.

    Parameters
    ----------
    network : AlphaZeroNet
        Neural network for policy priors and value estimation.
    num_simulations : int
        Total simulation budget per move.
    num_considered_actions : int
        Top-k actions to consider at root (before Sequential Halving).
    c_puct : float
        Exploration constant for interior PUCT nodes.
    use_heuristic_value : bool
        Use heuristic instead of network value head.
    opponent_policy : callable or None
        Deterministic opponent for 2-player search.
    """

    def __init__(
        self,
        network=None,
        num_simulations: int = 32,
        num_considered_actions: int = 16,
        c_puct: float = 1.5,
        use_heuristic_value: bool = False,
        opponent_policy=None,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.num_considered_actions = num_considered_actions
        self.c_puct = c_puct
        self.use_heuristic_value = use_heuristic_value
        self.opponent_policy = opponent_policy

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

    def _select_interior(self, node: MCTSNode, sim_env, min_max: MinMaxStats) -> tuple:
        """Standard PUCT selection for interior nodes (below root)."""
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
        """Expand a leaf node using network for priors and value."""
        obs = env._get_obs()
        action_mask = env.action_masks()

        if action_mask.sum() == 0:
            node.is_expanded = True
            return 0.0

        priors, value = self.network.predict(obs, action_mask)

        if self.use_heuristic_value:
            value = _heuristic_value(env)

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
        """Propagate value up the tree."""
        current = node
        while current is not None:
            current.N += 1
            current.W += value
            min_max.update(current.Q)
            current = current.parent

    def _simulate_action(self, root: MCTSNode, action: int, env, min_max: MinMaxStats):
        """Run one simulation starting from a specific root action."""
        sim_env = env.clone(strip_opponent=True)

        # Apply root action
        child = root.children[action]
        pin_id, dest = sim_env._mapper.decode(action)
        sim_env._board.apply_move(sim_env._AGENT_COLOUR, pin_id, dest)
        sim_env._step_count += 1

        # Check terminal after root action
        terminal = False
        if sim_env._board.check_win(sim_env._AGENT_COLOUR):
            terminal = True
        elif sim_env._step_count >= sim_env.max_steps:
            terminal = True
        elif self._apply_opponent_move(sim_env):
            terminal = True

        if terminal:
            agent_won = sim_env._board.check_win(sim_env._AGENT_COLOUR) if sim_env._board else False
            value = 1.0 if agent_won else _heuristic_value(sim_env)
        else:
            # Continue with interior PUCT selection
            node, term = self._select_interior(child, sim_env, min_max)
            if term:
                agent_won = sim_env._board.check_win(sim_env._AGENT_COLOUR) if sim_env._board else False
                value = 1.0 if agent_won else _heuristic_value(sim_env)
            else:
                value = self._expand(node, sim_env)

            self._backup(node, value, min_max)
            return  # _backup already handled the full path

        # Backup from child to root
        self._backup(child, value, min_max)

    def run(self, env) -> tuple[MCTSNode, np.ndarray]:
        """Run Gumbel MCTS with Sequential Halving.

        Returns
        -------
        (root, gumbel_scores) where:
            root: MCTSNode with visit counts
            gumbel_scores: np.ndarray of final scores for action selection
        """
        root = MCTSNode(parent=None, action=None, prior=1.0)
        min_max = MinMaxStats()

        # Step 1: Get network policy and expand root
        obs = env._get_obs()
        action_mask = env.action_masks()

        if action_mask.sum() == 0:
            return root, np.zeros(env.action_space.n)

        priors, root_value = self.network.predict(obs, action_mask)

        if self.use_heuristic_value:
            root_value = _heuristic_value(env)

        legal_actions = np.where(action_mask)[0]
        for action in legal_actions:
            root.children[int(action)] = MCTSNode(
                parent=root,
                action=int(action),
                prior=float(priors[action]),
            )
        root.is_expanded = True
        root.N = 1
        root.W = root_value
        min_max.update(root_value)

        # Step 2: Sample Gumbel noise for legal actions
        num_legal = len(legal_actions)
        gumbel_noise = _sample_gumbel((num_legal,))

        # Log-priors for legal actions
        log_priors = np.log(np.clip(priors[legal_actions], 1e-10, 1.0))

        # Initial scores: g(a) + log π(a|s)
        scores = gumbel_noise + log_priors

        # Step 3: Select top-k actions
        k = min(self.num_considered_actions, num_legal)
        top_k_indices = np.argsort(scores)[-k:]  # indices into legal_actions
        candidates = legal_actions[top_k_indices].tolist()
        candidate_gumbel = {
            int(legal_actions[idx]): float(gumbel_noise[idx])
            for idx in top_k_indices
        }
        candidate_log_prior = {
            int(legal_actions[idx]): float(log_priors[idx])
            for idx in top_k_indices
        }

        # Step 4: Sequential Halving
        remaining_budget = self.num_simulations
        current_candidates = list(candidates)

        while len(current_candidates) > 1 and remaining_budget > 0:
            # Allocate simulations equally among remaining candidates
            n_candidates = len(current_candidates)
            # Number of halving rounds remaining
            rounds_left = max(1, int(math.ceil(math.log2(n_candidates))))
            sims_per_action = max(1, remaining_budget // (n_candidates * rounds_left))

            for action in current_candidates:
                for _ in range(sims_per_action):
                    if remaining_budget <= 0:
                        break
                    self._simulate_action(root, action, env, min_max)
                    remaining_budget -= 1

            # Compute completed Q-values and halve
            action_scores = []
            for action in current_candidates:
                q_hat = _completed_q(root, action, min_max)
                g = candidate_gumbel.get(action, 0.0)
                lp = candidate_log_prior.get(action, 0.0)
                action_scores.append((action, g + lp + q_hat))

            action_scores.sort(key=lambda x: x[1], reverse=True)
            half = max(1, len(action_scores) // 2)
            current_candidates = [a for a, _ in action_scores[:half]]

        # Use any remaining budget on the final candidate(s)
        for action in current_candidates:
            while remaining_budget > 0:
                self._simulate_action(root, action, env, min_max)
                remaining_budget -= 1

        # Step 5: Compute final scores for all legal actions
        num_actions = env.action_space.n
        final_scores = np.full(num_actions, -1e9, dtype=np.float32)
        for i, action in enumerate(legal_actions):
            a = int(action)
            q_hat = _completed_q(root, a, min_max)
            g = candidate_gumbel.get(a, float(gumbel_noise[i]))
            lp = float(log_priors[i])
            final_scores[a] = g + lp + q_hat

        return root, final_scores

    def get_action_probs(self, env, temperature: float = 1.0) -> np.ndarray:
        """Run Gumbel MCTS and return improved policy (visit-count distribution).

        For training data collection, we return the visit count distribution
        (which IS the improved policy under Gumbel AlphaZero).
        """
        root, _ = self.run(env)
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
            total = counts_temp.sum()
            probs = counts_temp / total if total > 0 else counts_temp

        return probs

    def select_action(self, env, temperature: float = 0.0) -> int:
        """Run Gumbel MCTS and select the best action.

        For tournament play (temperature=0), uses the Gumbel scores directly
        to select the action (this is the theoretically correct selection).
        For training (temperature>0), samples from visit distribution.
        """
        root, final_scores = self.run(env)

        if temperature == 0 or temperature < 1e-6:
            # Use Gumbel scores for deterministic selection
            return int(np.argmax(final_scores))
        else:
            # Sample from visit distribution
            num_actions = env.action_space.n
            visits = np.zeros(num_actions, dtype=np.float32)
            for action, child in root.children.items():
                visits[action] = child.N
            if visits.sum() == 0:
                mask = env.action_masks()
                visits = mask.astype(np.float32)
            counts_temp = visits ** (1.0 / temperature)
            total = counts_temp.sum()
            probs = counts_temp / total if total > 0 else counts_temp
            return int(np.random.choice(len(probs), p=probs))
