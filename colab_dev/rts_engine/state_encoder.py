"""State encoder: game state -> compact token sequence."""

from __future__ import annotations


def encode_state(state: dict) -> str:
    lines = []
    lines.append(f"TICK:{state['tick']} CREDITS:{state['credits']}")

    for u in state["my_units"][:12]:
        lines.append(f"U:{u['type']}#{u['id']} hp={u['hp']} @{u['x']},{u['y']} {u['state']}")

    for u in state["enemy_units"][:12]:
        hp_str = "??" if u.get("hp") is None else str(u["hp"])
        lines.append(f"E:{u['type']}#{u['id']} hp={hp_str} @{u['x']},{u['y']} {u['state']}")

    for s in state["my_structures"]:
        q_str = ""
        if s.get("queue"):
            q_str = f" prod:{s['queue'][0]} {s['timer']}t"
        lines.append(f"S:{s['type']}#{s['id']} hp={s['hp']}/{s['max_hp']} @{s['x']},{s['y']}{q_str}")

    for s in state["enemy_structures"]:
        lines.append(f"ES:{s['type']}#{s['id']} hp={s['hp']} @{s['x']},{s['y']}")

    for o in state["ore"][:8]:
        lines.append(f"ORE:@{o['x']},{o['y']}:{o['amount']}")

    total = state["map_size"] ** 2
    lines.append(f"FOG:v={state['fog_visible']} e={state['fog_explored']} u={state['fog_unknown']} t={total}")

    if state["game_over"]:
        w = state.get("winner")
        lines.append(f"GAME_OVER WINNER:{w}")

    return "\n".join(lines)


def count_tokens(text: str) -> int:
    return len(text.split())
