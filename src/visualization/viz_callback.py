"""VizCallback: SB3 callback that runs eval episodes and feeds the viz queue."""
import queue

from stable_baselines3.common.callbacks import BaseCallback

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.visualization import PinSnapshot


class VizCallback(BaseCallback):
    """Every eval_freq steps, run one deterministic episode and push frames to viz_queue.

    Parameters
    ----------
    viz_queue : queue.Queue(maxsize=1)
        Thread-safe queue shared with the GUI thread.
        Episodes are dropped silently when the queue is full.
    eval_freq : int
        How many training steps between visualizations.
    max_steps : int
        Episode length cap for the eval env.
    """

    def __init__(
        self,
        viz_queue: queue.Queue,
        eval_freq: int = 10_000,
        max_steps: int = 1000,
        opponent_policy=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.viz_queue = viz_queue
        self.eval_freq = eval_freq
        self.max_steps = max_steps
        self.opponent_policy = opponent_policy
        self._episode_count: int = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        env = ChineseCheckersEnv(
            opponent_policy=self.opponent_policy,
            max_steps=self.max_steps,
            render_mode=None,
        )
        obs, _ = env.reset()
        done = False
        frames: list[list[PinSnapshot]] = []

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = [
                PinSnapshot(position=pin.position, color=pin.color)
                for pin_list in env._board.pins.values()
                for pin in pin_list
            ]
            frames.append(frame)

        self._episode_count += 1
        try:
            self.viz_queue.put_nowait({"episode": self._episode_count, "frames": frames})
        except queue.Full:
            pass  # GUI hasn't consumed previous episode; silently drop

        return True
