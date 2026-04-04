"""Quick diagnostic: check if the PPO model produces valid actions."""
import sys
import os
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MULTI_SYSTEM_DIR = os.path.join(PROJECT_ROOT, 'multi system single machine minimal')
for p in [PROJECT_ROOT, MULTI_SYSTEM_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sb3_contrib import MaskablePPO
from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.env.action_mapper import ActionMapper
from src.env.state_encoder import StateEncoder

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

model = MaskablePPO.load(args.model)
encoder = StateEncoder(grid_size=17, num_channels=10)
mapper = ActionMapper(num_pins=10, num_cells=121)

# Test 1: Use env directly (same as training)
print("=== Test 1: Model through env.step() (training path) ===")
env = ChineseCheckersEnv(opponent_policy=None, max_steps=20)
obs, info = env.reset()

for step in range(10):
    mask = env.action_masks()
    obs_batch = np.expand_dims(obs, 0)
    mask_batch = np.expand_dims(mask, 0)
    action, _ = model.predict(obs_batch, action_masks=mask_batch, deterministic=True)
    action = int(action[0])
    pin_id, dest = mapper.decode(action)
    is_legal = mask[action]
    print(f"  Step {step}: action={action} pin={pin_id} dest={dest} legal={is_legal} mask_sum={mask.sum()}")
    obs, reward, term, trunc, info = env.step(action)
    pins = env._board.pins_in_goal(env._AGENT_COLOUR)
    dist = env._board.total_distance_to_goal(env._AGENT_COLOUR)
    print(f"    reward={reward:.3f} pins_in_goal={pins} dist={dist:.1f}")
    if term or trunc:
        break

# Test 2: Use board_wrapper directly (arena path)
print("\n=== Test 2: Model through board_wrapper (arena path) ===")
env2 = ChineseCheckersEnv(opponent_policy=None, max_steps=20)
obs2, info2 = env2.reset()
turn_order = ["red", "blue"]

for step in range(10):
    bw = env2.board_wrapper
    obs_enc = encoder.encode(bw, current_colour="red", turn_order=turn_order)
    obs_env = env2._get_obs()

    # Check if encodings match
    match = np.allclose(obs_enc, obs_env)
    print(f"  Step {step}: obs match={match}", end="")
    if not match:
        diff = np.abs(obs_enc - obs_env)
        print(f" max_diff={diff.max():.4f} channels_differ={np.where(diff.max(axis=(1,2)) > 0)[0]}")
    else:
        print()

    legal_moves = bw.get_legal_moves("red")
    mask_bw = mapper.build_action_mask(legal_moves)
    mask_env = env2.action_masks()
    mask_match = np.array_equal(mask_bw, mask_env)
    print(f"    mask match={mask_match} bw_sum={mask_bw.sum()} env_sum={mask_env.sum()}")

    obs_batch = np.expand_dims(obs_enc, 0)
    mask_batch = np.expand_dims(mask_bw, 0)
    action, _ = model.predict(obs_batch, action_masks=mask_batch, deterministic=True)
    action = int(action[0])
    pin_id, dest = mapper.decode(action)
    is_legal = mask_bw[action]
    print(f"    action={action} pin={pin_id} dest={dest} legal={is_legal}")

    # Step env with the action
    obs2, reward, term, trunc, info2 = env2.step(action)
    pins = env2._board.pins_in_goal(env2._AGENT_COLOUR)
    dist = env2._board.total_distance_to_goal(env2._AGENT_COLOUR)
    print(f"    reward={reward:.3f} pins_in_goal={pins} dist={dist:.1f}")
    if term or trunc:
        break

# Test 3: Run 200 steps, track distance progress
print("\n=== Test 3: 200-step progress check ===")
env3 = ChineseCheckersEnv(opponent_policy=None, max_steps=200)
obs3, _ = env3.reset()
initial_dist = env3._board.total_distance_to_goal(env3._AGENT_COLOUR)
print(f"  Initial distance: {initial_dist:.1f}")

for step in range(200):
    mask = env3.action_masks()
    obs_batch = np.expand_dims(obs3, 0)
    mask_batch = np.expand_dims(mask, 0)
    action, _ = model.predict(obs_batch, action_masks=mask_batch, deterministic=True)
    action = int(action[0])
    obs3, reward, term, trunc, _ = env3.step(action)
    if term or trunc:
        print(f"  Game ended at step {step+1}")
        break
    if (step + 1) % 25 == 0:
        dist = env3._board.total_distance_to_goal(env3._AGENT_COLOUR)
        pins = env3._board.pins_in_goal(env3._AGENT_COLOUR)
        print(f"  Step {step+1}: dist={dist:.1f} pins={pins}")

final_dist = env3._board.total_distance_to_goal(env3._AGENT_COLOUR)
final_pins = env3._board.pins_in_goal(env3._AGENT_COLOUR)
print(f"  Final: dist={final_dist:.1f} pins={final_pins} (delta={initial_dist - final_dist:.1f})")
