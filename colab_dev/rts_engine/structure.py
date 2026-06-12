"""Structures: Barracks and War Factory with production queues."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class StructureType(Enum):
    BARRACKS = "barracks"
    WAR_FACTORY = "war_factory"


@dataclass
class StructureDef:
    width: int
    height: int
    hp: int
    cost: int
    produces: list


STRUCTURE_DEFS = {
    StructureType.BARRACKS: StructureDef(
        width=2, height=2, hp=500, cost=300, produces=["gi"]),
    StructureType.WAR_FACTORY: StructureDef(
        width=3, height=3, hp=800, cost=500, produces=["grizzly", "prism"]),
}


@dataclass
class Structure:
    id: int
    structure_type: StructureType
    faction_id: int
    x: int
    y: int
    hp: int
    production_queue: list = field(default_factory=list)
    production_timer: int = 0

    @staticmethod
    def create(struct_id, stype, faction_id, x, y):
        defn = STRUCTURE_DEFS[stype]
        return Structure(id=struct_id, structure_type=stype,
                        faction_id=faction_id, x=x, y=y, hp=defn.hp)

    @property
    def definition(self):
        return STRUCTURE_DEFS[self.structure_type]

    @property
    def is_alive(self):
        return self.hp > 0

    @property
    def is_producing(self):
        return len(self.production_queue) > 0

    @property
    def footprint(self):
        d = self.definition
        return [(self.x + dx, self.y + dy)
                for dy in range(d.height) for dx in range(d.width)]

    def short_label(self):
        return {"barracks": "BARRACKS", "war_factory": "WARFACT"}[self.structure_type.value]
