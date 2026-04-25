#!/usr/bin/env python3
"""
Standalone worker: plays one self-play game and writes results to a temp file.

Usage:
    python scripts/worker_play_game.py <model_path> <opp_type> <output_path> [options]

Arguments:
    model_path  - Path to serialized model state_dict (.pt bytes file)
    opp_type    - "greedy", "advanced", or "self_play"
    output_path - Where to write the numpy result (.npz)

Options:
    --num-blocks N      (default 9)
    --num-filters N     (default 96)
    --sims N            (default 200)
    --batch-size N      (default 16)
    --heuristic-value   (flag)

Designed to be launched via subprocess.Popen for parallel game generation
when ProcessPoolExecutor is blocked by container environments.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU before any torch import

import sys
import argparse
import numpy as np
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.network.alphazero_net import AlphaZeroNet, NetworkConfig
from src.training.alphazero_self_play import SelfPlayConfig
from src.training.true_self_play import play_one_game_true_selfplay, play_one_game_vs_heuristic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("opp_type", type=str, choices=["greedy", "advanced", "self_play"])
    parser.add_argument("output_path", type=str)
    parser.add_argument("--num-blocks", type=int, default=9)
    parser.add_argument("--num-filters", type=int, default=96)
    parser.add_argument("--in-channels", type=int, default=10)
    parser.add_argument("--num-actions", type=int, default=1210)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heuristic-value", action="store_true")
    args = parser.parse_args()

    # Build network on CPU
    net_config = NetworkConfig(
        in_channels=args.in_channels,
        num_actions=args.num_actions,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        architecture="resnet",
    )
    net = AlphaZeroNet(net_config, device="cpu")

    # Load weights
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    net.model.load_state_dict(state_dict)
    net.model.eval()

    # Build self-play config
    config = SelfPlayConfig(
        num_simulations=args.sims,
        use_batched_mcts=False,  # CPU workers use non-batched
        mcts_batch_size=args.batch_size,
        use_heuristic_value=args.heuristic_value,
    )

    # Play game
    if args.opp_type == "self_play":
        samples = play_one_game_true_selfplay(network=net, config=config)
    else:
        from src.agents.greedy_agent import greedy_policy
        from src.agents.advanced_heuristic import advanced_heuristic_policy
        opp_policies = {"greedy": greedy_policy, "advanced": advanced_heuristic_policy}
        samples = play_one_game_vs_heuristic(
            network=net, opponent_policy_fn=opp_policies[args.opp_type], config=config,
        )

    if not samples:
        # Write empty marker
        np.savez_compressed(args.output_path, empty=np.array([1]))
        return

    # Pack samples into arrays
    obs_list = [s.obs for s in samples]
    mask_list = [s.action_mask for s in samples]
    policy_list = [s.policy_target for s in samples]
    value_list = [s.value_target for s in samples]

    np.savez_compressed(
        args.output_path,
        obs=np.stack(obs_list),
        action_mask=np.stack(mask_list),
        policy_target=np.stack(policy_list),
        value_target=np.array(value_list, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
