"""
MCTS (Monte Carlo Tree Search) for inference with a trained PPO value network.

Uses the trained MaskablePPO model's value function to evaluate board states
and its policy network to guide the search, similar to AlphaZero's approach.

Usage:
    from src.inference.mcts import MCTSAgent
    agent = MCTSAgent(model_path="models/best/best_model.zip", num_simulations=50)
    pin_id, dest = agent.select_action(board_wrapper, colour)
"""

import math
import numpy as np

from src.env.board_wrapper import BoardWrapper
from src.env.state_encoder import StateEncoder
from src.env.action_mapper import ActionMapper


class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = ('parent', 'action', 'prior', 'visit_count', 'value_sum', 'children', 'is_expanded')

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action           # (pin_id, dest) that led to this node
        self.prior = prior             # P(a|s) from policy network
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}             # action -> MCTSNode
        self.is_expanded = False

    @property
    def q_value(self):
        """Mean action value Q(s,a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits, c_puct=1.5):
        """Upper confidence bound score (PUCT formula from AlphaZero)."""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTSAgent:
    """
    MCTS agent that uses a trained MaskablePPO model for value estimation
    and policy priors.

    Parameters
    ----------
    model_path : str
        Path to saved MaskablePPO model (.zip).
    num_simulations : int
        Number of MCTS simulations per move (default 50).
    c_puct : float
        Exploration constant for PUCT formula (default 1.5).
    temperature : float
        Temperature for action selection. 0 = greedy, 1 = proportional to visits.
    """

    def __init__(self, model_path, num_simulations=50, c_puct=1.5, temperature=0.0):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        self._encoder = StateEncoder(grid_size=17, num_channels=10)
        self._mapper = ActionMapper(num_pins=10, num_cells=121)
        self._turn_order = ["red", "blue"]

    def select_action(self, board_wrapper, colour):
        """
        Run MCTS from the current board state and return the best action.

        Parameters
        ----------
        board_wrapper : BoardWrapper
        colour : str

        Returns
        -------
        (pin_id, dest_index) tuple
        """
        root = MCTSNode()

        # Expand root with policy priors
        self._expand_node(root, board_wrapper, colour)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_board = board_wrapper.clone()

            # 1. SELECT — traverse tree using UCB
            while node.is_expanded and node.children:
                node = self._select_child(node)

                # Apply the selected action to the simulated board
                pin_id, dest = node.action
                sim_board.apply_move(colour, pin_id, dest)

                # Opponent moves (simple greedy or random for simulation)
                self._simulate_opponent_move(sim_board, colour)

            # 2. EXPAND — if node hasn't been expanded yet
            if not node.is_expanded:
                self._expand_node(node, sim_board, colour)

            # 3. EVALUATE — use value network
            value = self._evaluate(sim_board, colour)

            # 4. BACKPROPAGATE
            self._backpropagate(node, value)

        # Select action based on visit counts
        return self._select_final_action(root)

    def _expand_node(self, node, board_wrapper, colour):
        """Expand a node by adding children for all legal actions with policy priors."""
        legal_moves = board_wrapper.get_legal_moves(colour)
        if not legal_moves:
            node.is_expanded = True
            return

        # Get policy priors from the model
        obs = self._encode_board(board_wrapper, colour)
        action_mask = self._build_action_mask(legal_moves)
        policy_probs = self._get_policy_probs(obs, action_mask)

        # Create child nodes for each legal action
        for pin_id, dests in legal_moves.items():
            for dest in dests:
                action = (pin_id, dest)
                flat_action = pin_id * 121 + dest
                prior = policy_probs[flat_action] if flat_action < len(policy_probs) else 0.0
                node.children[action] = MCTSNode(parent=node, action=action, prior=prior)

        node.is_expanded = True

    def _select_child(self, node):
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _evaluate(self, board_wrapper, colour):
        """Use the value network to evaluate a board state."""
        obs = self._encode_board(board_wrapper, colour)
        obs_tensor = self.model.policy.obs_to_tensor(obs[np.newaxis])[0]
        with __import__('torch').no_grad():
            value = self.model.policy.predict_values(obs_tensor)
        return float(value.item())

    def _get_policy_probs(self, obs, action_mask):
        """Get policy probabilities from the model, masked and normalized."""
        import torch

        obs_tensor = self.model.policy.obs_to_tensor(obs[np.newaxis])[0]
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            # Get log probs for all actions
            logits = dist.distribution.logits.cpu().numpy().flatten()

        # Apply mask: set illegal actions to -inf
        masked_logits = np.where(action_mask, logits, -1e9)

        # Softmax to get probabilities
        max_logit = np.max(masked_logits)
        exp_logits = np.exp(masked_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def _simulate_opponent_move(self, board_wrapper, agent_colour):
        """Simulate opponent's response (random legal move for speed)."""
        opponent_colour = "blue" if agent_colour == "red" else "red"
        legal_moves = board_wrapper.get_legal_moves(opponent_colour)
        if not legal_moves:
            return

        pin_ids = list(legal_moves.keys())
        pin_id = pin_ids[np.random.randint(len(pin_ids))]
        dests = legal_moves[pin_id]
        dest = dests[np.random.randint(len(dests))]
        board_wrapper.apply_move(opponent_colour, pin_id, dest)

    def _backpropagate(self, node, value):
        """Backpropagate the value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _select_final_action(self, root):
        """Select the final action from root based on visit counts."""
        if not root.children:
            raise ValueError("No children to select from — root has no legal moves")

        if self.temperature == 0:
            # Greedy: pick most visited child
            best_child = max(root.children.values(), key=lambda c: c.visit_count)
            return best_child.action

        # Temperature-based selection
        visits = np.array([c.visit_count for c in root.children.values()])
        actions = list(root.children.keys())

        if self.temperature == 1.0:
            probs = visits / visits.sum()
        else:
            log_visits = np.log(visits + 1e-8) / self.temperature
            max_log = np.max(log_visits)
            probs = np.exp(log_visits - max_log)
            probs = probs / probs.sum()

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def _encode_board(self, board_wrapper, colour):
        """Encode board state as observation tensor."""
        return self._encoder.encode(board_wrapper, current_colour=colour, turn_order=self._turn_order)

    def _build_action_mask(self, legal_moves):
        """Build boolean action mask from legal moves dict."""
        return self._mapper.build_action_mask(legal_moves)


def make_mcts_policy(model_path, num_simulations=50, c_puct=1.5, temperature=0.0):
    """
    Create an MCTS policy function compatible with arena.py.

    Returns a callable with signature: policy(board_wrapper, colour) -> (pin_id, dest)
    """
    agent = MCTSAgent(
        model_path=model_path,
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature=temperature,
    )

    def policy(board_wrapper, colour):
        return agent.select_action(board_wrapper, colour)

    return policy
