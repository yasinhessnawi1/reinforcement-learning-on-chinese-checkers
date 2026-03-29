"""Elo rating and evaluation utilities."""


def compute_elo_update(rating_a, rating_b, winner, k=32):
    """Update Elo ratings after a game.

    Args:
        rating_a: current Elo of player A
        rating_b: current Elo of player B
        winner: 'a', 'b', or 'draw'
        k: K-factor (default 32)

    Returns:
        (new_rating_a, new_rating_b)
    """
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1.0 - expected_a

    if winner == 'a':
        score_a, score_b = 1.0, 0.0
    elif winner == 'b':
        score_a, score_b = 0.0, 1.0
    else:
        score_a, score_b = 0.5, 0.5

    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * (score_b - expected_b)

    return new_a, new_b


class EloTracker:
    """Track Elo ratings for multiple agents over time."""

    def __init__(self, initial_rating=1200):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.history = []

    def get_rating(self, agent_name):
        return self.ratings.get(agent_name, self.initial_rating)

    def record_game(self, agent_a, agent_b, winner):
        """Record a game result and update ratings.

        winner: 'a', 'b', or 'draw'
        """
        ra = self.get_rating(agent_a)
        rb = self.get_rating(agent_b)
        new_a, new_b = compute_elo_update(ra, rb, winner)
        self.ratings[agent_a] = new_a
        self.ratings[agent_b] = new_b
        self.history.append((agent_a, agent_b, winner, {agent_a: new_a, agent_b: new_b}))
        return new_a, new_b
