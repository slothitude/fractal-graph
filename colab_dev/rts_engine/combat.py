"""Combat resolution."""

from __future__ import annotations

TERRAIN_DEFENSE = {"plains": 1.0, "forest": 0.7, "mountain": 0.5, "water": 1.0, "road": 1.0}


def distance(x1, y1, x2, y2):
    return max(abs(x2 - x1), abs(y2 - y1))


def resolve_combat(attacker, defender, game_map):
    from .unit import TYPE_MULTIPLIERS, UnitState
    if not attacker.can_attack or not defender.is_alive:
        return 0
    dist = distance(attacker.x, attacker.y, defender.x, defender.y)
    if dist > attacker.stats.range:
        return 0
    base = attacker.stats.damage
    type_mult = TYPE_MULTIPLIERS.get((attacker.unit_type, defender.unit_type), 1.0)
    terrain = TERRAIN_DEFENSE.get(game_map.get_terrain(int(defender.x), int(defender.y)), 1.0)
    damage = max(1, int(base * type_mult * terrain))
    defender.hp -= damage
    attacker.cooldown_remaining = attacker.stats.attack_cooldown
    if defender.hp <= 0:
        defender.hp = 0
        defender.state = UnitState.DEAD
    return damage
