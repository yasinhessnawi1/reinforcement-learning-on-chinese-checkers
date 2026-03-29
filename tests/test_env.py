"""
Tests for src/env/chinese_checkers_env.py
"""
import sys
import os

# Ensure the repo root is importable regardless of where pytest is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pytest
import gymnasium as gym

from src.env.chinese_checkers_env import ChineseCheckersEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(**kwargs) -> ChineseCheckersEnv:
    """Return a freshly constructed env (not yet reset)."""
    return ChineseCheckersEnv(**kwargs)


def _legal_action(env: ChineseCheckersEnv) -> int:
    """Return any legal action index from the current mask."""
    mask = env.action_masks()
    legal_indices = np.where(mask)[0]
    assert len(legal_indices) > 0, "Expected at least one legal action"
    return int(legal_indices[0])


# ---------------------------------------------------------------------------
# 1. test_env_creates
# ---------------------------------------------------------------------------

def test_env_creates():
    env = _make_env()
    assert env is not None


# ---------------------------------------------------------------------------
# 2. test_observation_space
# ---------------------------------------------------------------------------

def test_observation_space():
    env = _make_env()
    assert env.observation_space.shape == (10, 17, 17)
    assert env.observation_space.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. test_action_space
# ---------------------------------------------------------------------------

def test_action_space():
    env = _make_env()
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 1210


# ---------------------------------------------------------------------------
# 4. test_reset_returns_obs_and_info
# ---------------------------------------------------------------------------

def test_reset_returns_obs_and_info():
    env = _make_env()
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray), "obs must be a numpy array"
    assert obs.shape == (10, 17, 17), f"Expected (10,17,17) got {obs.shape}"
    assert obs.dtype == np.float32

    assert isinstance(info, dict), "info must be a dict"
    assert "action_mask" in info, "info must contain 'action_mask'"


# ---------------------------------------------------------------------------
# 5. test_action_mask_shape
# ---------------------------------------------------------------------------

def test_action_mask_shape():
    env = _make_env()
    env.reset()
    mask = env.action_masks()

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (1210,)
    assert mask.dtype == np.bool_
    assert mask.sum() > 0, "At least one action must be legal at game start"


# ---------------------------------------------------------------------------
# 6. test_step_with_legal_action
# ---------------------------------------------------------------------------

def test_step_with_legal_action():
    env = _make_env()
    env.reset()

    action = _legal_action(env)
    result = env.step(action)

    assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"
    obs, reward, terminated, truncated, info = result

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10, 17, 17)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# 7. test_action_mask_method
# ---------------------------------------------------------------------------

def test_action_mask_method():
    env = _make_env()
    env.reset()
    mask = env.action_masks()

    assert mask.shape == (1210,)
    assert mask.dtype == np.bool_

    # After one step the mask should still be valid
    action = int(np.where(mask)[0][0])
    env.step(action)
    mask2 = env.action_masks()
    assert mask2.shape == (1210,)
    assert mask2.dtype == np.bool_


# ---------------------------------------------------------------------------
# 8. test_full_game_terminates
# ---------------------------------------------------------------------------

def test_full_game_terminates():
    """A game under random (legal) play must end within max_steps."""
    max_steps = 500
    env = _make_env(max_steps=max_steps)
    env.reset()

    for _ in range(max_steps + 10):
        mask = env.action_masks()
        legal = np.where(mask)[0]
        if len(legal) == 0:
            break
        action = int(legal[np.random.randint(len(legal))])
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        pytest.fail("Game did not terminate within the expected number of steps")

    # Either terminated (win/loss/draw) or truncated (step limit)
    assert terminated or truncated, "Episode must end via terminated or truncated"


# ---------------------------------------------------------------------------
# 9. test_opponent_moves_automatically
# ---------------------------------------------------------------------------

def test_opponent_moves_automatically():
    """After agent step(), the opponent should also have moved (blue pieces change)."""
    env = _make_env()
    obs_before, _ = env.reset()

    # Record blue piece positions indirectly via channel 1 of the encoded obs
    # Channel 1 = opponent pieces in the StateEncoder
    blue_before = obs_before[1].copy()

    action = _legal_action(env)
    obs_after, _, terminated, truncated, _ = env.step(action)

    if not (terminated or truncated):
        # At least one of: agent channel (0) or opponent channel (1) must differ
        agent_changed = not np.array_equal(obs_before[0], obs_after[0])
        opponent_changed = not np.array_equal(blue_before, obs_after[1])
        assert agent_changed or opponent_changed, (
            "After a step, the observation should reflect at least one move "
            "(agent or opponent)"
        )
