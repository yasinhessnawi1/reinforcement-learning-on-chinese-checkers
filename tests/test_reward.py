"""
Tests for the reward shaping module (v3).
"""

import pytest
from src.training.reward import (
    compute_step_reward,
    REWARD_FORWARD_PROGRESS,
    REWARD_GOAL_ENTRY,
    REWARD_GOAL_EXIT,
    REWARD_WIN,
    REWARD_LOSE,
    REWARD_DRAW,
    REWARD_STEP_PENALTY,
    REWARD_HOP_BONUS,
    REWARD_GOAL_PROXIMITY,
)

SP = REWARD_STEP_PENALTY  # shorthand for step penalty in expected values


class TestForwardProgress:
    """Tests for forward progress reward."""

    def test_distance_decreased_positive_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=90,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
        )
        expected = SP + 10 * REWARD_FORWARD_PROGRESS
        assert reward == pytest.approx(expected)

    def test_distance_increased_negative_reward(self):
        reward = compute_step_reward(
            dist_before=90, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
        )
        expected = SP + (-10) * REWARD_FORWARD_PROGRESS
        assert reward == pytest.approx(expected)
        assert reward < 0

    def test_no_distance_change(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
        )
        assert reward == pytest.approx(SP)


class TestGoalEntry:
    """Tests for goal entry reward."""

    def test_single_pin_enters_goal(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=1,
            won=False, lost=False, drawn=False,
        )
        expected = SP + REWARD_GOAL_ENTRY
        assert reward == pytest.approx(expected)
        assert reward > 1.0

    def test_multiple_pins_enter_goal(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=1, pins_in_goal_after=3,
            won=False, lost=False, drawn=False,
        )
        expected = SP + 2 * REWARD_GOAL_ENTRY
        assert reward == pytest.approx(expected)


class TestGoalExit:
    """Tests for goal exit penalty."""

    def test_single_pin_exits_goal(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=1, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
        )
        expected = SP + REWARD_GOAL_EXIT
        assert reward == pytest.approx(expected)
        assert reward < -1.0

    def test_multiple_pins_exit_goal(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=3, pins_in_goal_after=1,
            won=False, lost=False, drawn=False,
        )
        expected = SP + 2 * REWARD_GOAL_EXIT
        assert reward == pytest.approx(expected)


class TestTerminalStates:
    """Tests for terminal state rewards."""

    def test_win_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=12, pins_in_goal_after=12,
            won=True, lost=False, drawn=False,
        )
        expected = SP + REWARD_WIN
        assert reward == pytest.approx(expected)
        assert reward >= 10.0

    def test_lose_penalty(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=True, drawn=False,
        )
        expected = SP + REWARD_LOSE
        assert reward == pytest.approx(expected)
        assert reward < 0

    def test_draw_penalty(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=True,
        )
        expected = SP + REWARD_DRAW
        assert reward == pytest.approx(expected)


class TestHopBonus:
    """Tests for hop move bonus (v3)."""

    def test_hop_move_gets_bonus(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            is_hop=True,
        )
        expected = SP + REWARD_HOP_BONUS
        assert reward == pytest.approx(expected)
        assert reward > SP  # Hop bonus should make it better than no-op

    def test_non_hop_move_no_bonus(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            is_hop=False,
        )
        assert reward == pytest.approx(SP)

    def test_hop_plus_forward_progress(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=95,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            is_hop=True,
        )
        expected = SP + 5 * REWARD_FORWARD_PROGRESS + REWARD_HOP_BONUS
        assert reward == pytest.approx(expected)


class TestGoalProximity:
    """Tests for goal proximity bonus (v3)."""

    def test_pins_near_goal_get_bonus(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            pins_near_goal=3,
        )
        expected = SP + 3 * REWARD_GOAL_PROXIMITY
        assert reward == pytest.approx(expected)

    def test_zero_pins_near_goal(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            pins_near_goal=0,
        )
        assert reward == pytest.approx(SP)

    def test_proximity_plus_goal_entry(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=1,
            won=False, lost=False, drawn=False,
            pins_near_goal=2,
        )
        expected = SP + REWARD_GOAL_ENTRY + 2 * REWARD_GOAL_PROXIMITY
        assert reward == pytest.approx(expected)


class TestCombinedRewards:
    """Tests for combined reward components."""

    def test_forward_progress_plus_goal_entry(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=90,
            pins_in_goal_before=0, pins_in_goal_after=1,
            won=False, lost=False, drawn=False,
        )
        expected = SP + 10 * REWARD_FORWARD_PROGRESS + REWARD_GOAL_ENTRY
        assert reward == pytest.approx(expected)

    def test_win_plus_forward_progress(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=90,
            pins_in_goal_before=12, pins_in_goal_after=12,
            won=True, lost=False, drawn=False,
        )
        expected = SP + 10 * REWARD_FORWARD_PROGRESS + REWARD_WIN
        assert reward == pytest.approx(expected)

    def test_goal_exit_plus_lose(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=2, pins_in_goal_after=0,
            won=False, lost=True, drawn=False,
        )
        expected = SP + 2 * REWARD_GOAL_EXIT + REWARD_LOSE
        assert reward == pytest.approx(expected)

    def test_all_bonuses_combined(self):
        """Hop + proximity + forward + goal entry."""
        reward = compute_step_reward(
            dist_before=100, dist_after=95,
            pins_in_goal_before=0, pins_in_goal_after=1,
            won=False, lost=False, drawn=False,
            is_hop=True, pins_near_goal=2,
        )
        expected = (SP
                    + 5 * REWARD_FORWARD_PROGRESS
                    + REWARD_GOAL_ENTRY
                    + REWARD_HOP_BONUS
                    + 2 * REWARD_GOAL_PROXIMITY)
        assert reward == pytest.approx(expected)


class TestCustomRewardConstants:
    """Tests for custom reward constant parameters."""

    def test_custom_forward_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=90,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            reward_forward=0.1, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(1.0)

    def test_custom_goal_entry_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=1,
            won=False, lost=False, drawn=False,
            reward_goal_entry=2.0, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(2.0)

    def test_custom_goal_exit_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=1, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            reward_goal_exit=-5.0, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(-5.0)

    def test_custom_win_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=12, pins_in_goal_after=12,
            won=True, lost=False, drawn=False,
            reward_win=100.0, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(100.0)

    def test_custom_lose_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=True, drawn=False,
            reward_lose=-10.0, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(-10.0)

    def test_custom_draw_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=True,
            reward_draw=-0.5, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(-0.5)

    def test_custom_hop_bonus(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            is_hop=True, reward_hop_bonus=0.5, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(0.5)

    def test_custom_proximity_reward(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            pins_near_goal=4, reward_goal_proximity=0.1, reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(0.4)


class TestStepPenalty:
    """Tests for per-step time penalty."""

    def test_step_penalty_always_applied(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
        )
        assert reward == pytest.approx(SP)
        assert reward < 0

    def test_custom_step_penalty(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            reward_step_penalty=-0.05,
        )
        assert reward == pytest.approx(-0.05)

    def test_zero_step_penalty(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=100,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
            reward_step_penalty=0.0,
        )
        assert reward == pytest.approx(0.0)


class TestReturnType:
    """Tests for return type."""

    def test_return_is_float(self):
        reward = compute_step_reward(
            dist_before=100, dist_after=90,
            pins_in_goal_before=0, pins_in_goal_after=0,
            won=False, lost=False, drawn=False,
        )
        assert isinstance(reward, float)
