"""Stage 2 tests: state encoder and action vocab."""

import pytest
from rts_engine.engine import GameEngine
from rts_engine.state_encoder import encode_state, count_tokens
from rts_engine.action_vocab import (
    parse_action, parse_action_text, format_tools_prompt,
    ACTION_TOOLS, VALID_STRUCTURE_TYPES, VALID_UNIT_TYPES,
)


class TestStateEncoder:
    def test_basic_encode(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        state = e.get_state(0)
        text = encode_state(state)
        assert "TICK:0" in text
        assert "CREDITS:2000" in text

    def test_includes_fog(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        state = e.get_state(0)
        text = encode_state(state)
        assert "FOG:" in text

    def test_token_count_under_400(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        state = e.get_state(0)
        text = encode_state(state)
        tokens = count_tokens(text)
        assert tokens < 400, f"Token count {tokens} exceeds 400"

    def test_round_trip_with_engine(self):
        """Encoded state from engine is parseable and contains expected keys."""
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        state = e.get_state(0)
        text = encode_state(state)
        lines = text.strip().split("\n")
        assert any("TICK:" in l for l in lines)
        assert any("CREDITS:" in l for l in lines)
        # Structures should appear
        assert any("S:" in l for l in lines)

    def test_units_after_production(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        barracks_id = next(s.id for s in e.factions[0].structures.values()
                             if s.structure_type.value == "barracks")
        e.execute_action(0, {"action": "produce", "unit_type": "gi",
                             "structure_id": barracks_id})
        for _ in range(10):
            e.tick()
        state = e.get_state(0)
        text = encode_state(state)
        assert "U:" in text  # should have a unit now

    def test_enemy_hidden_in_fog(self):
        """Enemy units not in visible range should not appear."""
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        state = e.get_state(0)
        text = encode_state(state)
        # Faction 0 starts top-left, faction 1 bottom-right
        # Enemy structures at ~(11,9) should not be visible initially
        assert "ES:" not in text

    def test_game_over_state(self):
        e = GameEngine(map_size=16, max_ticks=5, seed=42)
        e.setup_two_player()
        for _ in range(5):
            e.tick()
        state = e.get_state(0)
        text = encode_state(state)
        assert "GAME_OVER" in text


class TestActionVocab:
    def test_six_tools(self):
        assert len(ACTION_TOOLS) == 6
        names = {t["name"] for t in ACTION_TOOLS}
        assert names == {"build", "produce", "move", "attack", "stop", "guard"}

    def test_parse_action_dict(self):
        a = parse_action("build", {"structure_type": "barracks", "x": 3, "y": 3})
        assert a["action"] == "build"
        assert a["structure_type"] == "barracks"

    def test_parse_action_text_build(self):
        a = parse_action_text("build barracks 3 3")
        assert a == {"action": "build", "structure_type": "barracks", "x": 3, "y": 3}

    def test_parse_action_text_produce(self):
        a = parse_action_text("produce gi 1")
        assert a == {"action": "produce", "unit_type": "gi", "structure_id": 1}

    def test_parse_action_text_move(self):
        a = parse_action_text("move 1,2,3 8 8")
        assert a == {"action": "move", "unit_ids": [1, 2, 3], "x": 8, "y": 8}

    def test_parse_action_text_attack(self):
        a = parse_action_text("attack 1,2 5")
        assert a == {"action": "attack", "unit_ids": [1, 2], "target_id": 5}

    def test_parse_action_text_stop(self):
        a = parse_action_text("stop 1,3")
        assert a == {"action": "stop", "unit_ids": [1, 3]}

    def test_parse_action_text_guard(self):
        a = parse_action_text("guard 1,2 5 10")
        assert a == {"action": "guard", "unit_ids": [1, 2], "x": 5, "y": 10}

    def test_parse_action_text_case_insensitive(self):
        a = parse_action_text("BUILD Barracks 3 3")
        assert a == {"action": "build", "structure_type": "barracks", "x": 3, "y": 3}

    def test_parse_action_text_invalid_tool(self):
        assert parse_action_text("fly 1 2") is None

    def test_parse_action_text_invalid_structure(self):
        assert parse_action_text("build castle 3 3") is None

    def test_parse_action_text_empty(self):
        assert parse_action_text("") is None

    def test_format_tools_prompt(self):
        text = format_tools_prompt()
        assert "AVAILABLE ACTIONS:" in text
        assert "build(" in text
        assert "attack(" in text
