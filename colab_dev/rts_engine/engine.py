"""GameEngine: tick loop, win conditions, action execution."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .game_map import GameMap
from .unit import Unit, UnitType, UnitState, UNIT_STATS
from .structure import Structure, StructureType, STRUCTURE_DEFS
from .combat import resolve_combat, distance
from .pathfinder import astar
from .fog import FogManager


@dataclass
class ActionResult:
    success: bool
    message: str


@dataclass
class FactionState:
    faction_id: int
    credits: int = 2000
    next_unit_id: int = 1
    next_structure_id: int = 1
    units: dict = field(default_factory=dict)
    structures: dict = field(default_factory=dict)


def _production_time(unit_type):
    return {UnitType.GI: 5, UnitType.GRIZZLY: 10, UnitType.PRISM: 15}[unit_type]


class GameEngine:
    def __init__(self, map_size=16, max_ticks=300, seed=None):
        self.map = GameMap(size=map_size, seed=seed)
        self.max_ticks = max_ticks
        self.tick_count = 0
        self.fog = FogManager(map_size, num_factions=2, vision_range=4)
        self.factions = {0: FactionState(faction_id=0),
                        1: FactionState(faction_id=1)}
        self.game_over = False
        self.winner = None

    def setup_two_player(self):
        s = self.map.size
        self._place_base(0, 1, 1)
        self._place_base(1, s - 5, s - 7)
        self._update_fog()

    def _place_base(self, fid, bx, by):
        fs = self.factions[fid]
        for stype, dy_off in [(StructureType.BARRACKS, 0),
                               (StructureType.WAR_FACTORY, 3)]:
            sid = fs.next_structure_id; fs.next_structure_id += 1
            struct = Structure.create(sid, stype, fid, bx, by + dy_off)
            fs.structures[sid] = struct
            for tx, ty in struct.footprint:
                self.map.tiles[ty][tx].structure_id = sid

    def tick(self):
        if self.game_over:
            return False
        self.tick_count += 1
        self._tick_movement()
        self._tick_combat()
        self._tick_structures()
        self._tick_resources()
        self._update_fog()
        self._check_win()
        return not self.game_over

    def _tick_movement(self):
        for fs in self.factions.values():
            for unit in list(fs.units.values()):
                if unit.state == UnitState.DEAD:
                    continue
                if unit.cooldown_remaining > 0:
                    unit.cooldown_remaining -= 1
                if unit.state == UnitState.MOVING and unit.path:
                    wx, wy = unit.path[0]
                    speed = unit.stats.speed
                    dx, dy = wx - unit.x, wy - unit.y
                    dist = max(abs(dx), abs(dy), 0.01)
                    if dist <= speed:
                        unit.x, unit.y = float(wx), float(wy)
                        unit.path.pop(0)
                        if not unit.path:
                            unit.state = UnitState.IDLE
                    else:
                        step = min(speed, dist)
                        unit.x += (dx / dist) * step
                        unit.y += (dy / dist) * step

    def _tick_combat(self):
        alive = [u for fs in self.factions.values()
                 for u in fs.units.values() if u.is_alive]
        for attacker in alive:
            if attacker.state != UnitState.ATTACKING or attacker.target_id is None:
                continue
            target = self._find_unit(attacker.target_id)
            if target and target.is_alive:
                resolve_combat(attacker, target, self.map)
                if not target.is_alive:
                    target.state = UnitState.DEAD
                if not attacker.is_alive:
                    attacker.state = UnitState.DEAD
        for fs in self.factions.values():
            dead = [uid for uid, u in fs.units.items() if u.state == UnitState.DEAD]
            for uid in dead:
                del fs.units[uid]

    def _tick_structures(self):
        for fs in self.factions.values():
            for struct in fs.structures.values():
                if not struct.is_alive or not struct.is_producing:
                    continue
                struct.production_timer -= 1
                if struct.production_timer <= 0:
                    utype_val = struct.production_queue.pop(0)
                    utype = UnitType(utype_val)
                    uid = fs.next_unit_id; fs.next_unit_id += 1
                    sx = struct.x + struct.definition.width
                    sy = struct.y + struct.definition.height // 2
                    fs.units[uid] = Unit.create(uid, utype, fs.faction_id,
                                               float(sx), float(sy))
                    if struct.production_queue:
                        struct.production_timer = _production_time(
                            UnitType(struct.production_queue[0]))

    def _tick_resources(self):
        for fs in self.factions.values():
            fs.credits += sum(1 for s in fs.structures.values() if s.is_alive) * 50

    def _update_fog(self):
        for fid in self.factions:
            self.fog.update(fid, [u for u in self.factions[fid].units.values()
                                    if u.is_alive])

    def _check_win(self):
        if self.tick_count >= self.max_ticks:
            self.game_over = True
            scores = {}
            for fid, fs in self.factions.items():
                scores[fid] = (sum(1 for u in fs.units.values() if u.is_alive) * 10
                              + sum(1 for s in fs.structures.values() if s.is_alive) * 100)
            if scores[0] > scores[1]: self.winner = 0
            elif scores[1] > scores[0]: self.winner = 1
            return
        for fid in (0, 1):
            if not any(s.is_alive for s in self.factions[1 - fid].structures.values()):
                self.game_over = True
                self.winner = fid
                return

    def execute_action(self, faction_id, action):
        atype = action.get("action", "")
        fs = self.factions[faction_id]
        if atype == "build": return self._action_build(fs, action)
        if atype == "produce": return self._action_produce(fs, action)
        if atype == "move": return self._action_move(fs, action)
        if atype == "attack": return self._action_attack(fs, action)
        if atype == "stop": return self._action_stop(fs, action)
        if atype == "guard": return self._action_guard(fs, action)
        return ActionResult(False, f"Unknown: {atype}")

    def _action_build(self, fs, a):
        try:
            stype = StructureType(a.get("structure_type", ""))
        except ValueError:
            return ActionResult(False, f"Unknown: {a.get('structure_type')}")
        x, y = a.get("x", -1), a.get("y", -1)
        d = STRUCTURE_DEFS[stype]
        if fs.credits < d.cost:
            return ActionResult(False, f"Need {d.cost}, have {fs.credits}")
        if not self.map.can_place_structure(x, y, d.width, d.height):
            return ActionResult(False, f"Blocked ({x},{y})")
        fs.credits -= d.cost
        sid = fs.next_structure_id; fs.next_structure_id += 1
        struct = Structure.create(sid, stype, fs.faction_id, x, y)
        fs.structures[sid] = struct
        for tx, ty in struct.footprint:
            self.map.tiles[ty][tx].structure_id = sid
        return ActionResult(True, f"Built {stype.value}#{sid}")

    def _action_produce(self, fs, a):
        try:
            utype = UnitType(a.get("unit_type", ""))
        except ValueError:
            return ActionResult(False, f"Unknown: {a.get('unit_type')}")
        sid = a.get("structure_id")
        if sid is None or sid not in fs.structures:
            return ActionResult(False, f"Bad struct: {sid}")
        struct = fs.structures[sid]
        if not struct.is_alive:
            return ActionResult(False, "Destroyed")
        if a["unit_type"] not in struct.definition.produces:
            return ActionResult(False, f"Cannot produce {a['unit_type']}")
        cost = UNIT_STATS[utype].cost
        if fs.credits < cost:
            return ActionResult(False, f"Need {cost}, have {fs.credits}")
        fs.credits -= cost
        struct.production_queue.append(a["unit_type"])
        if len(struct.production_queue) == 1:
            struct.production_timer = _production_time(utype)
        return ActionResult(True, f"Queued {a['unit_type']}")

    def _action_move(self, fs, a):
        moved = 0
        for uid in a.get("unit_ids", []):
            if uid not in fs.units: continue
            unit = fs.units[uid]
            if not unit.is_alive: continue
            path = astar(self.map, (int(unit.x), int(unit.y)),
                         (a.get("x", 0), a.get("y", 0)))
            if path:
                unit.path = path; unit.state = UnitState.MOVING
                unit.target_id = None; moved += 1
        return ActionResult(moved > 0, f"Moved {moved}")

    def _action_attack(self, fs, a):
        target_id = a.get("target_id")
        target = self._find_unit(target_id)
        if target is None:
            return ActionResult(False, f"Target#{target_id} gone")
        targeted = 0
        for uid in a.get("unit_ids", []):
            if uid not in fs.units: continue
            unit = fs.units[uid]
            if not unit.is_alive: continue
            dist = distance(unit.x, unit.y, target.x, target.y)
            if dist > unit.stats.range:
                path = astar(self.map, (int(unit.x), int(unit.y)),
                             (int(target.x), int(target.y)))
                if path:
                    unit.path = path
            unit.state = UnitState.ATTACKING
            unit.target_id = target_id; targeted += 1
        return ActionResult(targeted > 0, f"{targeted} attacking")

    def _action_stop(self, fs, a):
        for uid in a.get("unit_ids", []):
            if uid in fs.units:
                u = fs.units[uid]
                u.state = UnitState.IDLE; u.target_id = None; u.path = []
        return ActionResult(True, "Stopped")

    def _action_guard(self, fs, a):
        gx, gy = a.get("x", -1), a.get("y", -1)
        for uid in a.get("unit_ids", []):
            if uid in fs.units:
                u = fs.units[uid]
                path = astar(self.map, (int(u.x), int(u.y)), (gx, gy))
                if path: u.path = path; u.state = UnitState.MOVING
        return ActionResult(True, "Guarding")

    def _find_unit(self, uid):
        for fs in self.factions.values():
            if uid in fs.units: return fs.units[uid]
        return None

    def get_state(self, faction_id):
        fs = self.factions[faction_id]
        efs = self.factions[1 - faction_id]
        my_units = [{"id": u.id, "type": u.unit_type.short_id,
                     "x": int(u.x), "y": int(u.y), "hp": u.hp,
                     "state": u.state.name, "cd": u.cooldown_remaining}
                    for u in fs.units.values()]
        enemy_units = [{"id": u.id, "type": u.unit_type.short_id,
                        "x": int(u.x), "y": int(u.y), "hp": u.hp,
                        "state": u.state.name}
                       for u in efs.units.values()
                       if self.fog.is_visible(faction_id, int(u.x), int(u.y))]
        my_structs = [{"id": s.id, "type": s.short_label(),
                       "x": s.x, "y": s.y, "hp": s.hp,
                       "max_hp": s.definition.hp,
                       "queue": list(s.production_queue),
                       "timer": s.production_timer}
                      for s in fs.structures.values()]
        enemy_structs = [{"id": s.id, "type": s.short_label(),
                         "x": s.x, "y": s.y, "hp": s.hp}
                        for s in efs.structures.values()
                        if any(self.fog.is_visible(faction_id, tx, ty)
                               for tx, ty in s.footprint)]
        ore = [{"x": x, "y": y, "amount": self.map.tiles[y][x].ore}
               for y in range(self.map.size) for x in range(self.map.size)
               if self.map.tiles[y][x].ore > 0
               and self.fog.is_known(faction_id, x, y)]
        v, e, u, t = self.fog.get_state(faction_id)
        return {"tick": self.tick_count, "max_ticks": self.max_ticks,
                "faction_id": faction_id, "credits": fs.credits,
                "my_units": my_units, "enemy_units": enemy_units,
                "my_structures": my_structs, "enemy_structures": enemy_structs,
                "ore": ore, "fog_visible": v, "fog_explored": e,
                "fog_unknown": u, "map_size": self.map.size,
                "game_over": self.game_over, "winner": self.winner}

    def run_game(self):
        while self.tick():
            pass
        return {"winner": self.winner, "ticks": self.tick_count}
