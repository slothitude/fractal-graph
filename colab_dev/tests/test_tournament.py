"""Stage 5 tests: tournament runner, Elo, replay."""

import pytest
from rts_engine.tournament.elo import EloRating
from rts_engine.tournament.replay import Replay, ReplayStep
from rts_engine.tournament.runner import TournamentRunner, _run_match
from rts_engine.opponents.simple_ai import SimpleAI


def _make_aggressive_strategy(state):
    """Aggressive: produce GI then rush center."""
    credits = state.get("credits", 0)
    my_units = [u["id"] for u in state.get("my_units", [])]
    enemy_units = [u for u in state.get("enemy_units", [])]

    if not my_units:
        for s in state.get("my_structures", []):
            if "B" in s["type"] and not s.get("queue"):
                if credits >= 100:
                    return {"action": "produce", "unit_type": "gi",
                            "structure_id": s["id"]}
        return None

    if enemy_units:
        return {"action": "attack", "unit_ids": my_units,
                "target_id": enemy_units[0]["id"]}
    return {"action": "move", "unit_ids": my_units, "x": 8, "y": 8}


def _make_passive_strategy(state):
    """Passive: just produce units, never attack."""
    credits = state.get("credits", 0)
    my_units = [u["id"] for u in state.get("my_units", [])]
    if len(my_units) >= 5:
        return None  # sit idle
    for s in state.get("my_structures", []):
        if "B" in s["type"] and not s.get("queue"):
            if credits >= 100:
                return {"action": "produce", "unit_type": "gi",
                        "structure_id": s["id"]}
    return None


class TestElo:
    def test_initial_rating(self):
        elo = EloRating()
        elo.register("alice")
        assert elo.get("alice") == 1000.0

    def test_update_winner(self):
        elo = EloRating()
        elo.register("alice")
        elo.register("bob")
        elo.update("alice", "bob")
        assert elo.get("alice") > 1000.0
        assert elo.get("bob") < 1000.0

    def test_draw(self):
        elo = EloRating()
        elo.register("alice")
        elo.register("bob")
        elo.draw("alice", "bob")
        assert elo.get("alice") == 1000.0
        assert elo.get("bob") == 1000.0

    def test_leaderboard(self):
        elo = EloRating()
        elo.register("alice")
        elo.register("bob")
        elo.update("alice", "bob")
        board = elo.leaderboard()
        assert board[0][0] == "alice"
        assert board[0][1] > board[1][1]

    def test_spread(self):
        elo = EloRating()
        elo.register("alice")
        elo.register("bob")
        elo.update("alice", "bob")
        assert elo.spread() > 0

    def test_spread_empty(self):
        elo = EloRating()
        assert elo.spread() == 0


class TestReplay:
    def test_create_replay(self):
        r = Replay(seed=42, map_size=16, max_ticks=300)
        assert r.seed == 42
        assert r.winner is None

    def test_add_step(self):
        r = Replay(seed=42, map_size=16, max_ticks=300)
        r.steps.append(ReplayStep(
            tick=0, faction_id=0,
            action={"action": "produce", "unit_type": "gi", "structure_id": 1},
            action_text="produce gi 1"))
        assert len(r.steps) == 1

    def test_sft_examples(self):
        r = Replay(seed=42, map_size=16, max_ticks=300)
        r.winner = 0
        r.steps.append(ReplayStep(
            tick=0, faction_id=0,
            action={"action": "move", "unit_ids": [1], "x": 8, "y": 8},
            action_text="move 1 8 8",
            state_snapshot={"tick": 0, "credits": 2000}))
        r.steps.append(ReplayStep(
            tick=5, faction_id=1,
            action={"action": "move", "unit_ids": [2], "x": 8, "y": 8},
            action_text="move 2 8 8"))
        examples = r.to_sft_examples(0)
        assert len(examples) == 1
        assert examples[0]["action"] == "move 1 8 8"

    def test_summary(self):
        r = Replay(seed=42, map_size=16, max_ticks=300)
        r.winner = 0
        r.total_ticks = 50
        s = r.summary()
        assert s["winner"] == 0
        assert s["total_ticks"] == 50


class TestRunner:
    def test_single_match(self):
        result = _run_match("aggro", "passive",
                            _make_aggressive_strategy, _make_passive_strategy,
                            max_ticks=50, seed=42)
        assert result.ticks <= 50

    def test_tournament_completes(self):
        strategies = {
            "aggro": _make_aggressive_strategy,
            "passive": _make_passive_strategy,
        }
        runner = TournamentRunner(
            strategies, games_per_match=4, max_ticks=50, seed=42)
        summary = runner.run()
        assert summary["total_games"] == 4

    def test_tournament_three_players(self):
        strategies = {
            "aggro": _make_aggressive_strategy,
            "passive": _make_passive_strategy,
            "aggro2": _make_aggressive_strategy,
        }
        runner = TournamentRunner(
            strategies, games_per_match=2, max_ticks=50, seed=42)
        summary = runner.run()
        # 3 choose 2 = 3 pairs, 2 games each = 6 total
        assert summary["total_games"] == 6
        assert len(summary["leaderboard"]) == 3
