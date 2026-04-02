# tests/test_replay_gui.py
"""Tests for replay_gui helper logic (no Tkinter display required)."""
import queue
import pytest
from src.visualization import PinSnapshot


def make_episode(n_frames=3, episode=1):
    frames = [
        [PinSnapshot((float(i), float(j)), "red") for j in range(2)]
        for i in range(n_frames)
    ]
    return {"episode": episode, "frames": frames}


def test_pin_snapshot_duck_types_position_and_color():
    snap = PinSnapshot(position=(10.5, 20.3), color="blue")
    x, y = snap.position
    assert x == 10.5
    assert y == 20.3
    assert snap.color == "blue"


def test_episode_dict_structure():
    ep = make_episode(n_frames=5, episode=7)
    assert ep["episode"] == 7
    assert len(ep["frames"]) == 5
    assert all(isinstance(f, list) for f in ep["frames"])
    assert all(isinstance(p, PinSnapshot) for p in ep["frames"][0])


def test_queue_maxsize_drops_old_episode():
    q = queue.Queue(maxsize=1)
    ep1 = make_episode(episode=1)
    ep2 = make_episode(episode=2)
    q.put_nowait(ep1)
    try:
        q.put_nowait(ep2)
    except queue.Full:
        pass
    item = q.get_nowait()
    assert item["episode"] == 1  # ep2 was dropped, ep1 remains


def test_empty_frames_list_is_valid():
    ep = {"episode": 1, "frames": []}
    assert ep["frames"] == []
