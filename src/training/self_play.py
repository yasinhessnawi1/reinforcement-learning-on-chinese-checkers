"""Self-play training loop with checkpoint pool."""
import os
import glob
import random

import numpy as np
from sb3_contrib import MaskablePPO

from src.agents.greedy_agent import greedy_policy


class CheckpointPool:
    """Maintains a pool of past model checkpoints for opponent diversity."""

    def __init__(self, pool_dir, max_size=20):
        self.pool_dir = pool_dir
        self.max_size = max_size
        os.makedirs(pool_dir, exist_ok=True)

    def save(self, model, step):
        """Save a model checkpoint to the pool."""
        path = os.path.join(self.pool_dir, f"checkpoint_{step}")
        model.save(path)
        # Prune oldest if too many
        checkpoints = sorted(glob.glob(os.path.join(self.pool_dir, "checkpoint_*.zip")))
        while len(checkpoints) > self.max_size:
            os.remove(checkpoints.pop(0))

    def sample(self):
        """Return path to a random checkpoint, or None if pool empty."""
        checkpoints = glob.glob(os.path.join(self.pool_dir, "checkpoint_*.zip"))
        if not checkpoints:
            return None
        return random.choice(checkpoints)

    def size(self):
        return len(glob.glob(os.path.join(self.pool_dir, "checkpoint_*.zip")))


def make_checkpoint_opponent(checkpoint_path):
    """Load a checkpoint and return a policy function compatible with ChineseCheckersEnv."""
    model = MaskablePPO.load(checkpoint_path)
    from src.env.state_encoder import StateEncoder
    from src.env.action_mapper import ActionMapper
    encoder = StateEncoder(grid_size=17, num_channels=10)
    mapper = ActionMapper(num_pins=10, num_cells=121)
    turn_order = ["red", "blue"]

    def policy(board_wrapper, colour):
        obs = encoder.encode(board_wrapper, current_colour=colour, turn_order=turn_order)
        obs_tensor = np.expand_dims(obs, 0)

        legal = board_wrapper.get_legal_moves(colour)
        mask = mapper.build_action_mask(legal)

        action, _ = model.predict(obs_tensor, action_masks=np.expand_dims(mask, 0), deterministic=False)
        pin_id, dest = mapper.decode(int(action[0]))
        return pin_id, dest

    return policy


def make_self_play_opponent(pool, greedy_fallback_ratio=0.3):
    """Return an opponent policy that samples from checkpoint pool or greedy.

    Each call to the returned function picks a fresh strategy:
    - With probability greedy_fallback_ratio: use greedy heuristic
    - Otherwise: load a random checkpoint from the pool
    - If pool is empty: always use greedy
    """
    def opponent(board_wrapper, colour):
        if random.random() < greedy_fallback_ratio or pool.size() == 0:
            return greedy_policy(board_wrapper, colour)

        checkpoint_path = pool.sample()
        if checkpoint_path is None:
            return greedy_policy(board_wrapper, colour)

        policy = make_checkpoint_opponent(checkpoint_path)
        return policy(board_wrapper, colour)

    return opponent
