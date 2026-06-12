"""Scripted baseline AI: build order then attack."""

from __future__ import annotations
from ..unit import UnitType


class SimpleAI:
    def __init__(self, faction_id):
        self.fid = faction_id
        self.phase = 0
        self.gi_target = 3
        self.tank_target = 2
        self.tick_count = 0

    def get_action(self, engine):
        self.tick_count += 1
        fs = engine.factions[self.fid]
        efs = engine.factions[1 - self.fid]

        if self.phase == 0:
            return self._phase_build(fs, engine)
        elif self.phase == 1:
            return self._phase_produce_gi(fs, engine)
        elif self.phase == 2:
            return self._phase_build_factory(fs, engine)
        elif self.phase == 3:
            return self._phase_produce_tanks(fs, engine)
        elif self.phase == 4:
            return self._phase_attack(fs, efs, engine)

    def _phase_build(self, fs, engine):
        barracks = [s for s in fs.structures.values()
                    if s.structure_type.value == "barracks" and s.is_alive]
        if barracks:
            self.phase = 1
            return self._phase_produce_gi(fs, engine)
        if fs.credits >= 300:
            return {"action": "build", "structure_type": "barracks",
                    "x": 4, "y": 1}
        return None

    def _phase_produce_gi(self, fs, engine):
        gi_count = sum(1 for u in fs.units.values()
                       if u.unit_type == UnitType.GI and u.is_alive)
        if gi_count >= self.gi_target:
            self.phase = 2
            return self._phase_build_factory(fs, engine)
        barracks = [s for s in fs.structures.values()
                    if s.structure_type.value == "barracks" and s.is_alive
                    and not s.is_producing]
        if barracks and fs.credits >= 100:
            return {"action": "produce", "unit_type": "gi",
                    "structure_id": barracks[0].id}
        return None

    def _phase_build_factory(self, fs, engine):
        factory = [s for s in fs.structures.values()
                   if s.structure_type.value == "war_factory" and s.is_alive]
        if factory:
            self.phase = 3
            return self._phase_produce_tanks(fs, engine)
        if fs.credits >= 500:
            return {"action": "build", "structure_type": "war_factory",
                    "x": 4, "y": 4}
        return None

    def _phase_produce_tanks(self, fs, engine):
        tank_count = sum(1 for u in fs.units.values()
                         if u.unit_type == UnitType.GRIZZLY and u.is_alive)
        if tank_count >= self.tank_target:
            self.phase = 4
            return self._phase_attack(fs, engine.factions[1 - self.fid], engine)
        factory = [s for s in fs.structures.values()
                   if s.structure_type.value == "war_factory" and s.is_alive
                   and not s.is_producing]
        if factory and fs.credits >= 400:
            return {"action": "produce", "unit_type": "grizzly",
                    "structure_id": factory[0].id}
        return None

    def _phase_attack(self, fs, efs, engine):
        my_units = [u.id for u in fs.units.values() if u.is_alive]
        enemy_units = [u for u in efs.units.values() if u.is_alive]

        if len(my_units) < 2:
            return None

        if enemy_units:
            target = enemy_units[0]
            return {"action": "attack", "unit_ids": my_units,
                    "target_id": target.id}

        enemy_structs = [s for s in efs.structures.values() if s.is_alive]
        if enemy_structs:
            s = enemy_structs[0]
            cx = s.x + s.definition.width // 2
            cy = s.y + s.definition.height // 2
            return {"action": "move", "unit_ids": my_units,
                    "x": cx, "y": cy}
        return None
