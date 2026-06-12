"""A* pathfinding on terrain cost map."""

from __future__ import annotations
import heapq


def astar(game_map, start, goal):
    sx, sy = start
    gx, gy = goal
    if not game_map.in_bounds(gx, gy) or not game_map.is_passable(gx, gy):
        return None
    counter = 0
    open_set = [(0, counter, sx, sy)]
    came_from = {}
    g_cost = {(sx, sy): 0}
    while open_set:
        f, _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) == (gx, gy):
            path = []
            node = (gx, gy)
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path
        current_g = g_cost.get((cx, cy), float("inf"))
        if f - _h(cx, cy, gx, gy) > current_g + 0.01:
            continue
        for nx, ny in game_map.get_neighbors(cx, cy):
            if not game_map.is_passable(nx, ny):
                continue
            new_g = current_g + game_map.get_cost(nx, ny)
            if new_g < g_cost.get((nx, ny), float("inf")):
                g_cost[(nx, ny)] = new_g
                came_from[(nx, ny)] = (cx, cy)
                counter += 1
                heapq.heappush(open_set, (new_g + _h(nx, ny, gx, gy), counter, nx, ny))
    return None

def _h(x, y, gx, gy):
    return max(abs(gx - x), abs(gy - y))
