"""Elo rating system for adapter tournament."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EloRating:
    """Track Elo ratings for multiple competitors."""
    K: float = 32.0
    initial: float = 1000.0
    ratings: dict = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.ratings.get(name, self.initial)

    def register(self, name: str):
        if name not in self.ratings:
            self.ratings[name] = self.initial

    def update(self, winner: str, loser: str):
        self.register(winner)
        self.register(loser)
        expected = self._expected(self.ratings[winner], self.ratings[loser])
        self.ratings[winner] += self.K * (1.0 - expected)
        self.ratings[loser] += self.K * (0.0 - (1.0 - expected))

    def draw(self, p1: str, p2: str):
        self.register(p1)
        self.register(p2)
        expected = self._expected(self.ratings[p1], self.ratings[p2])
        self.ratings[p1] += self.K * (0.5 - expected)
        self.ratings[p2] += self.K * (0.5 - (1.0 - expected))

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: -x[1])

    def spread(self) -> float:
        """Max rating minus min rating."""
        if not self.ratings:
            return 0.0
        return max(self.ratings.values()) - min(self.ratings.values())
