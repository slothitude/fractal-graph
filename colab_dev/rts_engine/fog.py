"""Per-faction fog of war."""

from __future__ import annotations


class Visibility:
    UNKNOWN = 0
    EXPLORED = 1
    VISIBLE = 2


class FogManager:
    def __init__(self, size, num_factions=2, vision_range=4):
        self.size = size
        self.num_factions = num_factions
        self.vision_range = vision_range
        self.visibility = [
            [[Visibility.UNKNOWN] * size for _ in range(size)]
            for _ in range(num_factions)
        ]

    def update(self, faction_id, units):
        s = self.size
        for y in range(s):
            for x in range(s):
                if self.visibility[faction_id][y][x] == Visibility.VISIBLE:
                    self.visibility[faction_id][y][x] = Visibility.EXPLORED
        vr = self.vision_range
        for unit in units:
            ux, uy = int(unit.x), int(unit.y)
            for dy in range(-vr, vr + 1):
                for dx in range(-vr, vr + 1):
                    if max(abs(dx), abs(dy)) <= vr:
                        nx, ny = ux + dx, uy + dy
                        if 0 <= nx < s and 0 <= ny < s:
                            self.visibility[faction_id][ny][nx] = Visibility.VISIBLE

    def is_visible(self, faction_id, x, y):
        return self.visibility[faction_id][y][x] == Visibility.VISIBLE

    def is_known(self, faction_id, x, y):
        return self.visibility[faction_id][y][x] != Visibility.UNKNOWN

    def get_state(self, faction_id):
        s = self.size
        visible = explored = unknown = 0
        for y in range(s):
            for x in range(s):
                v = self.visibility[faction_id][y][x]
                if v == Visibility.VISIBLE:
                    visible += 1
                elif v == Visibility.EXPLORED:
                    explored += 1
                else:
                    unknown += 1
        return visible, explored, unknown, s * s
