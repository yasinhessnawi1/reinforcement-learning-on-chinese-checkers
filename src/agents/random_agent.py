"""
random_agent.py — A random policy for Chinese Checkers.
"""
import random


def random_policy(board_wrapper, colour: str):
    """
    Choose a random legal move for *colour*.

    Parameters
    ----------
    board_wrapper : BoardWrapper
        The current game state.
    colour : str
        The colour whose turn it is.

    Returns
    -------
    (pin_id, dest) : tuple[int, int]
        The chosen pin identifier and destination cell index.

    Raises
    ------
    ValueError
        If there are no legal moves available for *colour*.
    """
    legal = board_wrapper.get_legal_moves(colour)
    if not legal:
        raise ValueError(f"No legal moves available for colour '{colour}'")

    pin_id = random.choice(list(legal.keys()))
    dest = random.choice(legal[pin_id])
    return pin_id, dest
