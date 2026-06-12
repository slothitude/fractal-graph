"""Stage 1 tests: game engine core."""

import time
import pytest
from rts_engine.game_map import GameMap, Terrain
from rts_engine.unit import Unit, UnitType, UnitState, UNIT_STATS, TYPE_MULTIPLIERS
from rts_engine.structure import Structure, StructureType, STRUCTURE_DEFS
from rts_engine.combat import resolve_combat, distance, TERRAIN_DEFENSE
from rts_engine.pathfinder import astar
from rts_engine.fog import FogManager, Visibility
from rts_engine.engine import GameEngine, ActionResult
from rts_engine.opponents.simple_ai import SimpleAI


class TestGameMap:
    def test_map_size(self):
        m = GameMap(size=16, seed=42)
        assert m.size == 16

    def test_mirror_symmetry(self):
        m = GameMap(size=16, seed=42)
        for y in range(16):
            for x in range(16):
                assert m.tiles[y][x].terrain == m.tiles[y][15 - x].terrain

    def test_base_corners_clear(self):
        m = GameMap(size=16, seed=42)
        for y in range(8):
            for x in range(5):
                assert m.is_passable(x, y), f"Blocked at ({x},{y})"
        for y in range(8, 16):
            for x in range(10, 16):
                assert m.is_passable(x, y), f"Blocked at ({x},{y})"

    def test_ore_placement(self):
        m = GameMap(size=16, seed=42)
        ore_tiles = [(x, y) for y in range(16) for x in range(16)
                     if m.tiles[y][x].ore > 0]
        assert len(ore_tiles) >= 4
        ore_coords = set(ore_tiles)
        for x, y in ore_tiles:
            assert (15 - x, y) in ore_coords

    def test_impassable_terrain(self):
        m = GameMap(size=16, seed=42)
        m.tiles[8][8].terrain = Terrain.MOUNTAIN
        assert not m.is_passable(8, 8)
        m.tiles[8][8].terrain = Terrain.WATER
        assert not m.is_passable(8, 8)


class TestPathfinder:
    def test_simple_path(self):
        m = GameMap(size=16, seed=42)
        path = astar(m, (0, 0), (4, 4))
        assert path is not None
        assert path[-1] == (4, 4)

    def test_around_mountain(self):
        m = GameMap(size=16, seed=42)
        for y in range(3, 6):
            m.tiles[y][3].terrain = Terrain.MOUNTAIN
        path = astar(m, (2, 4), (5, 4))
        assert path is not None
        assert path[-1] == (5, 4)

    def test_no_path(self):
        m = GameMap(size=16, seed=42)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx != 0 or dy != 0:
                    m.tiles[8 + dy][8 + dx].terrain = Terrain.MOUNTAIN
        path = astar(m, (0, 0), (8, 8))
        assert path is None

    def test_impassable_goal(self):
        m = GameMap(size=16, seed=42)
        m.tiles[8][8].terrain = Terrain.MOUNTAIN
        path = astar(m, (7, 8), (8, 8))
        assert path is None


class TestCombat:
    def test_damage_calculation(self):
        m = GameMap(size=16, seed=42)
        gi = Unit.create(1, UnitType.GI, 0, 5.0, 5.0)
        target = Unit.create(2, UnitType.GI, 1, 8.0, 5.0)
        dmg = resolve_combat(gi, target, m)
        assert dmg > 0

    def test_type_multipliers(self):
        assert TYPE_MULTIPLIERS[(UnitType.GI, UnitType.PRISM)] == 1.5
        assert TYPE_MULTIPLIERS[(UnitType.GRIZZLY, UnitType.GI)] == 1.5
        assert TYPE_MULTIPLIERS[(UnitType.PRISM, UnitType.GRIZZLY)] == 1.5
        assert TYPE_MULTIPLIERS[(UnitType.GI, UnitType.GI)] == 1.0

    def test_terrain_defense(self):
        assert TERRAIN_DEFENSE["forest"] < TERRAIN_DEFENSE["plains"]

    def test_range_check(self):
        m = GameMap(size=16, seed=42)
        gi = Unit.create(1, UnitType.GI, 0, 0.0, 0.0)
        target = Unit.create(2, UnitType.GI, 1, 10.0, 10.0)
        dmg = resolve_combat(gi, target, m)
        assert dmg == 0


class TestFog:
    def test_initial_unknown(self):
        fog = FogManager(8, num_factions=2, vision_range=2)
        assert fog.visibility[0][0][0] == Visibility.UNKNOWN

    def test_unit_reveals(self):
        fog = FogManager(8, num_factions=1, vision_range=2)

        class FakeUnit:
            def __init__(self, x, y):
                self.x, self.y = float(x), float(y)
        fog.update(0, [FakeUnit(4, 4)])
        assert fog.is_visible(0, 4, 4)
        assert fog.is_visible(0, 5, 5)

    def test_visible_degrades_to_explored(self):
        fog = FogManager(8, num_factions=1, vision_range=2)

        class FakeUnit:
            def __init__(self, x, y):
                self.x, self.y = float(x), float(y)
        fog.update(0, [FakeUnit(4, 4)])
        assert fog.is_visible(0, 4, 4)
        fog.update(0, [FakeUnit(0, 0)])
        assert fog.visibility[0][4][4] == Visibility.EXPLORED


class TestStructures:
    def test_structure_defs(self):
        assert STRUCTURE_DEFS[StructureType.BARRACKS].produces == ["gi"]
        assert "grizzly" in STRUCTURE_DEFS[StructureType.WAR_FACTORY].produces

    def test_footprint(self):
        s = Structure.create(1, StructureType.BARRACKS, 0, 3, 3)
        assert len(s.footprint) == 4

    def test_create(self):
        s = Structure.create(1, StructureType.WAR_FACTORY, 0, 5, 5)
        assert s.is_alive


class TestEngine:
    def test_setup(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        assert len(e.factions[0].structures) == 2
        assert len(e.factions[1].structures) == 2

    def test_produce_action(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        barracks_id = next(s.id for s in e.factions[0].structures.values()
                          if s.structure_type == StructureType.BARRACKS)
        result = e.execute_action(0, {"action": "produce", "unit_type": "gi",
                                      "structure_id": barracks_id})
        assert result.success, result.message

    def test_move_action(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        barracks_id = next(s.id for s in e.factions[0].structures.values()
                          if s.structure_type == StructureType.BARRACKS)
        e.execute_action(0, {"action": "produce", "unit_type": "gi",
                             "structure_id": barracks_id})
        for _ in range(10):
            e.tick()
        uid = next(iter(e.factions[0].units))
        result = e.execute_action(0, {"action": "move", "unit_ids": [uid],
                                      "x": 8, "y": 8})
        assert result.success, result.message

    def test_get_state(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        state = e.get_state(0)
        assert "tick" in state
        assert "credits" in state

    def test_random_game_completes(self):
        e = GameEngine(map_size=16, max_ticks=50, seed=42)
        e.setup_two_player()
        ai0 = SimpleAI(0)
        ai1 = SimpleAI(1)
        start = time.time()
        while not e.game_over:
            for ai in (ai0, ai1):
                action = ai.get_action(e)
                if action:
                    e.execute_action(ai.fid, action)
            e.tick()
        elapsed = time.time() - start
        assert e.game_over
        assert elapsed < 0.5, f"Game took {elapsed:.3f}s"

    def test_win_by_destruction(self):
        e = GameEngine(map_size=16, max_ticks=300, seed=42)
        e.setup_two_player()
        for s in e.factions[1].structures.values():
            s.hp = 0
        e._check_win()
        assert e.game_over
        assert e.winner == 0

    def test_tick_limit(self):
        e = GameEngine(map_size=16, max_ticks=5, seed=42)
        e.setup_two_player()
        for _ in range(5):
            e.tick()
        assert e.game_over


class TestSimpleAI:
    def test_ai_produces_units(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        ai = SimpleAI(0)
        actions_taken = 0
        for _ in range(20):
            action = ai.get_action(e)
            if action:
                result = e.execute_action(0, action)
                if result.success:
                    actions_taken += 1
            e.tick()
        assert actions_taken > 0

    def test_ai_transitions_to_attack(self):
        e = GameEngine(map_size=16, seed=42)
        e.setup_two_player()
        ai = SimpleAI(0)
        ai.phase = 4  # skip to attack phase
        for i in range(5):
            uid = e.factions[0].next_unit_id
            e.factions[0].next_unit_id += 1
            e.factions[0].units[uid] = Unit.create(uid, UnitType.GI, 0, 3.0, 3.0)
        action = ai.get_action(e)
        assert action is not None
        assert action.get("action") in ("attack", "move")
