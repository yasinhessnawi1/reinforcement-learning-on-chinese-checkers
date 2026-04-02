"""replay_gui: Tkinter board visualizer running in a daemon thread.

Usage
-----
    from src.visualization.replay_gui import start_viz_thread
    viz_queue = start_viz_thread()
    viz_queue.put_nowait({"episode": 1, "frames": [...]})

Episodes are accumulated in memory. Use the listbox on the left to select
which episode to replay. The speed slider controls replay speed.
"""
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk

# Make baseline game importable
_BASE_GAME_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'multi system single machine minimal')
)
if _BASE_GAME_DIR not in sys.path:
    sys.path.insert(0, _BASE_GAME_DIR)

from checkers_board import HexBoard   # noqa: E402
from checkers_gui import BoardGUI     # noqa: E402


def start_viz_thread() -> queue.Queue:
    """Start the GUI daemon thread and return the episode queue.

    The caller pushes dicts: {"episode": int, "frames": list[list[PinSnapshot]]}.
    All episodes are stored; use the listbox to choose which to replay.
    """
    viz_queue: queue.Queue = queue.Queue()
    t = threading.Thread(target=_run_gui, args=(viz_queue,), daemon=True)
    t.start()
    return viz_queue


def _run_gui(viz_queue: queue.Queue) -> None:
    """Tkinter mainloop — runs entirely in the daemon thread."""
    board = HexBoard()
    gui = BoardGUI(board, [])
    root = gui.root
    root.title("Training Visualizer")

    # --- Episode store ---
    # List of {"episode": int, "frames": list[list[PinSnapshot]]}
    episodes: list[dict] = []

    # --- Layout: listbox on left, board canvas already packed by BoardGUI ---
    # Insert a frame on the left before the canvas
    left_frame = ttk.Frame(root)
    left_frame.pack(side="left", fill="y", padx=(4, 0), pady=4, before=gui.canvas)

    ttk.Label(left_frame, text="Episodes").pack()
    listbox = tk.Listbox(left_frame, width=18, selectmode="single")
    listbox.pack(fill="y", expand=True)

    # --- Speed slider (below canvas) ---
    speed_frame = ttk.Frame(root)
    speed_frame.pack(side="bottom", fill="x", padx=10, pady=4)
    ttk.Label(speed_frame, text="Slow").pack(side="left")
    speed_var = tk.DoubleVar(value=400.0)
    ttk.Scale(
        speed_frame, from_=1000, to=10, variable=speed_var,
        orient="horizontal", command=lambda _: None,
    ).pack(side="left", fill="x", expand=True)
    ttk.Label(speed_frame, text="Fast").pack(side="left")

    # --- Status label ---
    status_var = tk.StringVar(value="Select an episode to replay.")
    ttk.Label(root, textvariable=status_var).pack(side="bottom", pady=2)

    # --- Replay state ---
    state = {
        "after_id": None,
        "frames": [],
        "idx": 0,
        "episode": 0,
        "total": 0,
    }

    def _next_frame():
        if not state["frames"] or state["idx"] >= state["total"]:
            state["after_id"] = None
            status_var.set(f"Episode {state['episode']} — done ({state['total']} steps)")
            return
        gui.refresh(state["frames"][state["idx"]])
        state["idx"] += 1
        status_var.set(
            f"Episode {state['episode']} — Step {state['idx']}/{state['total']}"
        )
        delay = max(10, int(speed_var.get()))
        state["after_id"] = root.after(delay, _next_frame)

    def _play_episode(ep: dict):
        """Cancel any running replay and start the selected episode."""
        if state["after_id"] is not None:
            root.after_cancel(state["after_id"])
            state["after_id"] = None
        state["frames"] = ep["frames"]
        state["idx"] = 0
        state["episode"] = ep["episode"]
        state["total"] = len(ep["frames"])
        _next_frame()

    def _on_listbox_select(_):
        sel = listbox.curselection()
        if not sel:
            return
        _play_episode(episodes[sel[0]])

    listbox.bind("<<ListboxSelect>>", _on_listbox_select)

    # --- Poll queue for new episodes ---
    def _poll_queue():
        try:
            while True:
                item = viz_queue.get_nowait()
                episodes.append(item)
                label = f"Ep {item['episode']:>4}  ({len(item['frames'])} steps)"
                listbox.insert("end", label)
                # Auto-scroll to latest
                listbox.see("end")
        except queue.Empty:
            pass
        try:
            root.after(50, _poll_queue)
        except tk.TclError:
            return

    root.after(50, _poll_queue)
    root.mainloop()
