# tests/test_viz_callback.py
import queue
import pytest
from unittest.mock import MagicMock, patch
from src.visualization import PinSnapshot


def make_callback(eval_freq=2):
    from src.visualization.viz_callback import VizCallback
    q = queue.Queue(maxsize=1)
    cb = VizCallback(viz_queue=q, eval_freq=eval_freq, max_steps=10)
    cb.model = MagicMock()
    cb.model.predict.return_value = (0, None)
    cb.n_calls = 0
    return cb, q


def test_skips_when_not_at_freq():
    cb, q = make_callback(eval_freq=5)
    cb.n_calls = 3
    with patch("src.visualization.viz_callback.ChineseCheckersEnv") as MockEnv:
        cb._on_step()
    MockEnv.assert_not_called()
    assert q.empty()


def test_puts_episode_on_queue_at_freq():
    cb, q = make_callback(eval_freq=5)
    cb.n_calls = 5

    mock_env = MagicMock()
    mock_env.reset.return_value = (MagicMock(), {})
    mock_env.step.return_value = (MagicMock(), 0.0, True, False, {})
    pin1 = MagicMock(); pin1.position = (1.0, 2.0); pin1.color = "red"
    pin2 = MagicMock(); pin2.position = (3.0, 4.0); pin2.color = "blue"
    mock_env._board.pins = {"red": [pin1], "blue": [pin2]}

    with patch("src.visualization.viz_callback.ChineseCheckersEnv", return_value=mock_env):
        cb._on_step()

    assert not q.empty()
    item = q.get_nowait()
    assert item["episode"] == 1
    assert len(item["frames"]) == 1
    frame = item["frames"][0]
    assert PinSnapshot((1.0, 2.0), "red") in frame
    assert PinSnapshot((3.0, 4.0), "blue") in frame


def test_drops_episode_when_queue_full():
    cb, q = make_callback(eval_freq=1)
    cb.n_calls = 1

    mock_env = MagicMock()
    mock_env.reset.return_value = (MagicMock(), {})
    mock_env.step.return_value = (MagicMock(), 0.0, True, False, {})
    mock_env._board.pins = {}

    with patch("src.visualization.viz_callback.ChineseCheckersEnv", return_value=mock_env):
        cb._on_step()   # fills queue (episode 1)
        cb.n_calls = 2
        cb._on_step()   # queue full — should not raise

    item = q.get_nowait()
    assert item["episode"] == 1


def test_returns_true_always():
    cb, q = make_callback(eval_freq=99)
    cb.n_calls = 1
    result = cb._on_step()
    assert result is True


def test_handles_empty_board_pins():
    cb, q = make_callback(eval_freq=1)
    cb.n_calls = 1

    mock_env = MagicMock()
    mock_env.reset.return_value = (MagicMock(), {})
    mock_env.step.return_value = (MagicMock(), 0.0, True, False, {})
    mock_env._board.pins = {}

    with patch("src.visualization.viz_callback.ChineseCheckersEnv", return_value=mock_env):
        cb._on_step()

    item = q.get_nowait()
    assert item["frames"] == [[]]
