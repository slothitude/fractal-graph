"""Game map: 16x16 grid with terrain, ore fields, and mirror symmetry."""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Optional


class Terrain:
    PLAINS = "plains"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    WATER = "water"
    ROAD = "road"

    COSTS = {
        "plains": 1.0,
        "forest": 3.0,
        "mountain": float("inf"),
        "water": float("inf"),
        "road": 0.5,
    }

    PASSABLE = {"plains", "forest", "road"}


@dataclass
class Tile:
    terrain: str = Terrain.PLAINS
    ore: int = 0
    structure_id: Optional[int] = None


class GameMap:
    def __init__(self, size: int = 16, seed: Optional[int] = None):
        self.size = size
        self.rng = random.Random(seed)
        self.tiles = [[Tile() for _ in range(size)] for _ in range(size)]
        self._generate()

    def _generate(self):
        s = self.size
        for y in range(s):
            for x in range(s // 2):
                r = self.rng.random()
                if r < 0.10:
                    t = Terrain.FOREST
                elif r < 0.15:
                    t = Terrain.ROAD
                elif r < 0.18:
                    t = Terrain.WATER
                else:
                    t = Terrain.PLAINS
                self.tiles[y][x].terrain = t
                self.tiles[y][s - 1 - x].terrain = t
        for _ in range(self.rng.randint(2, 4)):
            cx = self.rng.randint(1, s // 2 - 2)
            cy = self.rng.randint(1, s - 2)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < s // 2 and 0 <= ny < s:
                        if self.rng.random() < 0.6:
                            self.tiles[ny][nx].terrain = Terrain.MOUNTAIN
                            self.tiles[ny][s - 1 - nx].terrain = Terrain.MOUNTAIN
        self._ensure_base_areas(s)
        self._place_ore(count=2)
        self._mirror_ore(s)

    def _ensure_base_areas(self, s):
        """Remove impassable terrain from base placement zones (mirrored)."""
        # Faction 0 base zone: x=0-4, y=0-7 (clear left, mirror to right)
        for dy in range(8):
            for dx in range(5):
                if self.tiles[dy][dx].terrain not in Terrain.PASSABLE:
                    self.tiles[dy][dx].terrain = Terrain.PLAINS
                    self.tiles[dy][s - 1 - dx].terrain = Terrain.PLAINS
        # Faction 1 base zone: x=0-5 left half, y=8-15 (clear left, mirror to right)
        for dy in range(8, s):
            for dx in range(6):
                if self.tiles[dy][dx].terrain not in Terrain.PASSABLE:
                    self.tiles[dy][dx].terrain = Terrain.PLAINS
                    self.tiles[dy][s - 1 - dx].terrain = Terrain.PLAINS

    def _place_ore(self, count=2):
        s = self.size
        for _ in range(count):
            for _attempt in range(50):
                x = self.rng.randint(s // 4, s // 2 - 1)
                y = self.rng.randint(s // 4, 3 * s // 4 - 1)
                if self.tiles[y][x].terrain == Terrain.PLAINS:
                    self.tiles[y][x].ore = self.rng.randint(1500, 3000)
                    break

    def _mirror_ore(self, s):
        """Enforce left-right ore symmetry."""
        for y in range(s):
            for x in range(s // 2):
                self.tiles[y][s - 1 - x].ore = self.tiles[y][x].ore

    def get_tile(self, x, y):
        return self.tiles[y][x]

    def get_terrain(self, x, y):
        return self.tiles[y][x].terrain

    def get_cost(self, x, y):
        return Terrain.COSTS[self.tiles[y][x].terrain]

    def is_passable(self, x, y):
        return self.tiles[y][x].terrain in Terrain.PASSABLE

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        out = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                out.append((nx, ny))
        return out

    def can_place_structure(self, x, y, w, h):
        for dy in range(h):
            for dx in range(w):
                nx, ny = x + dx, y + dy
                if not self.in_bounds(nx, ny):
                    return False
                tile = self.tiles[ny][nx]
                if tile.terrain not in Terrain.PASSABLE:
                    return False
                if tile.structure_id is not None:
                    return False
        return True

    def to_text(self):
        s = self.size
        char_map = {Terrain.PLAINS: ".", Terrain.FOREST: "T",
                     Terrain.MOUNTAIN: "^", Terrain.WATER: "~", Terrain.ROAD: "="}
        rows = []
        for y in range(s):
            row = []
            for x in range(s):
                tile = self.tiles[y][x]
                row.append("$" if tile.ore > 0 else char_map.get(tile.terrain, "?"))
            rows.append("".join(row))
        return "\n".join(rows)
