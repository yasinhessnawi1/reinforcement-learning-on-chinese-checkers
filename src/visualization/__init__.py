from typing import NamedTuple


class PinSnapshot(NamedTuple):
    """Lightweight pin representation safe to cross thread boundaries."""
    position: tuple[float, float]
    color: str
