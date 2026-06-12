"""Tournament runner: round-robin matches between adapters/strategies."""

from __future__ import annotations
from typing import Callable, Optional
from dataclasses import dataclass, field
from .elo import EloRating
from .replay import Replay, ReplayStep


@dataclass
class MatchResult:
    p1: str
    p2: str
    winner: Optional[int]  # 0, 1, or None (draw)
    ticks: int = 0
    replay: Optional[Replay] = None


def _run_match(
    p1_name: str,
    p2_name: str,
    p1_fn: Callable,
    p2_fn: Callable,
    map_size: int = 16,
    max_ticks: int = 300,
    seed: Optional[int] = None,
    capture_replay: bool = False,
) -> MatchResult:
    """Run a single match between two strategy functions.

    Each strategy_fn has signature: fn(state_dict) -> action_dict | None
    """
    from ..engine import GameEngine
    from ..state_encoder import encode_state
    import random

    rng = random.Random(seed)
    actual_seed = rng.randint(0, 2**31)
    engine = GameEngine(map_size=map_size, max_ticks=max_ticks, seed=actual_seed)
    engine.setup_two_player()

    replay = Replay(seed=actual_seed, map_size=map_size, max_ticks=max_ticks)
    replay.metadata = {"p1": p1_name, "p2": p2_name}

    while not engine.game_over:
        for fid, strategy_fn in [(0, p1_fn), (1, p2_fn)]:
            state = engine.get_state(fid)
            try:
                action = strategy_fn(state)
            except Exception:
                action = None
            if action:
                engine.execute_action(fid, action)
                if capture_replay:
                    state_text = encode_state(state)
                    replay.steps.append(ReplayStep(
                        tick=engine.tick_count, faction_id=fid,
                        action=action, action_text=str(action),
                        state_snapshot=state_text))
        engine.tick()

    replay.winner = engine.winner
    replay.total_ticks = engine.tick_count

    return MatchResult(
        p1=p1_name, p2=p2_name,
        winner=0 if engine.winner == 0 else (1 if engine.winner == 1 else None),
        ticks=engine.tick_count,
        replay=replay if capture_replay else None,
    )


class TournamentRunner:
    """Run a round-robin tournament between named strategies."""

    def __init__(self, strategies: dict[str, Callable],
                 games_per_match: int = 20, map_size: int = 16,
                 max_ticks: int = 300, seed: Optional[int] = None):
        """
        Args:
            strategies: {name: fn(state) -> action} mapping
            games_per_match: number of games per pair (alternating positions)
            map_size: map dimension
            max_ticks: max ticks per game
            seed: base seed for reproducibility
        """
        self.strategies = strategies
        self.games_per_match = games_per_match
        self.map_size = map_size
        self.max_ticks = max_ticks
        self.seed = seed
        self.elo = EloRating()
        self.results: list[MatchResult] = []
        self._base_seed = seed

        for name in strategies:
            self.elo.register(name)

    def run(self) -> dict:
        """Run all matches, return results summary."""
        import random
        rng = random.Random(self._base_seed)
        names = list(self.strategies.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1, p2 = names[i], names[j]
                for g in range(self.games_per_match):
                    gseed = rng.randint(0, 2**31)
                    # Alternate starting positions
                    if g % 2 == 0:
                        result = _run_match(
                            p1, p2,
                            self.strategies[p1], self.strategies[p2],
                            self.map_size, self.max_ticks, seed=gseed)
                    else:
                        result = _run_match(
                            p2, p1,
                            self.strategies[p2], self.strategies[p1],
                            self.map_size, self.max_ticks, seed=gseed)
                        result.p1, result.p2 = p1, p2
                        if result.winner is not None:
                            result.winner = 1 - result.winner

                    self.results.append(result)

                    # Update Elo
                    if result.winner == 0:
                        self.elo.update(p1, p2)
                    elif result.winner == 1:
                        self.elo.update(p2, p1)
                    else:
                        self.elo.draw(p1, p2)

        return self.summary()

    def summary(self) -> dict:
        board = self.elo.leaderboard()
        wins = {name: 0 for name in self.strategies}
        losses = {name: 0 for name in self.strategies}
        draws = {name: 0 for name in self.strategies}
        for r in self.results:
            if r.winner == 0:
                wins[r.p1] += 1
                losses[r.p2] += 1
            elif r.winner == 1:
                wins[r.p2] += 1
                losses[r.p1] += 1
            else:
                draws[r.p1] += 1
                draws[r.p2] += 1

        return {
            "leaderboard": board,
            "elo_spread": self.elo.spread(),
            "total_games": len(self.results),
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

    def get_replays(self) -> list[Replay]:
        return [r.replay for r in self.results if r.replay is not None]

    def get_sft_buffer(self, faction_id: int = 0) -> list[dict]:
        """Collect all SFT examples from captured replays."""
        examples = []
        for replay in self.get_replays():
            examples.extend(replay.to_sft_examples(faction_id))
        return examples
