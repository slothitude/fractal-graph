"""Units: 3 rock-paper-scissors types with FSM states."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class UnitType(Enum):
    GI = "gi"
    GRIZZLY = "grizzly"
    PRISM = "prism"

    @property
    def short_id(self):
        return {"gi": "GI", "grizzly": "TZ", "prism": "ART"}[self.value]


class UnitState(Enum):
    IDLE = auto()
    MOVING = auto()
    ATTACKING = auto()
    FLEEING = auto()
    DEAD = auto()


@dataclass
class UnitStats:
    hp: int
    damage: int
    range: int
    speed: float
    attack_cooldown: int
    cost: int


UNIT_STATS = {
    UnitType.GI:      UnitStats(hp=50, damage=8, range=3, speed=2.0, attack_cooldown=3, cost=100),
    UnitType.GRIZZLY: UnitStats(hp=200, damage=25, range=4, speed=3.0, attack_cooldown=4, cost=400),
    UnitType.PRISM:   UnitStats(hp=80, damage=40, range=7, speed=1.0, attack_cooldown=6, cost=600),
}

TYPE_MULTIPLIERS = {
    (UnitType.GI, UnitType.PRISM): 1.5,
    (UnitType.GRIZZLY, UnitType.GI): 1.5,
    (UnitType.PRISM, UnitType.GRIZZLY): 1.5,
    (UnitType.GI, UnitType.GRIZZLY): 0.75,
    (UnitType.GRIZZLY, UnitType.PRISM): 0.75,
    (UnitType.PRISM, UnitType.GI): 0.75,
    (UnitType.GI, UnitType.GI): 1.0,
    (UnitType.GRIZZLY, UnitType.GRIZZLY): 1.0,
    (UnitType.PRISM, UnitType.PRISM): 1.0,
}


@dataclass
class Unit:
    id: int
    unit_type: UnitType
    faction_id: int
    x: float
    y: float
    hp: int
    state: UnitState = UnitState.IDLE
    target_id: Optional[int] = None
    path: list = field(default_factory=list)
    cooldown_remaining: int = 0
    move_progress: float = 0.0

    @staticmethod
    def create(unit_id, unit_type, faction_id, x, y):
        stats = UNIT_STATS[unit_type]
        return Unit(id=unit_id, unit_type=unit_type, faction_id=faction_id,
                    x=x, y=y, hp=stats.hp)

    @property
    def stats(self):
        return UNIT_STATS[self.unit_type]

    @property
    def is_alive(self):
        return self.state != UnitState.DEAD and self.hp > 0

    @property
    def can_attack(self):
        return self.cooldown_remaining <= 0 and self.is_alive

    def short_label(self):
        return f"{self.unit_type.short_id}#{self.id}"
