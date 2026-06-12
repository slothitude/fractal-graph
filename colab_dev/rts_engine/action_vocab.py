"""Action vocabulary: 6 tool definitions for LLM function calling."""

ACTION_TOOLS = [
    {
        "name": "build",
        "description": "Build a structure at the given position",
        "parameters": {
            "structure_type": "barracks or war_factory",
            "x": "int, grid x position",
            "y": "int, grid y position"
        }
    },
    {
        "name": "produce",
        "description": "Queue a unit for production from a structure",
        "parameters": {
            "unit_type": "gi, grizzly, or prism",
            "structure_id": "int, the structure to produce from"
        }
    },
    {
        "name": "move",
        "description": "Move units to a target position",
        "parameters": {
            "unit_ids": "list of int, units to move",
            "x": "int, target x",
            "y": "int, target y"
        }
    },
    {
        "name": "attack",
        "description": "Order units to attack a target",
        "parameters": {
            "unit_ids": "list of int, attacking units",
            "target_id": "int, enemy unit to attack"
        }
    },
    {
        "name": "stop",
        "description": "Cancel current orders for units",
        "parameters": {
            "unit_ids": "list of int, units to stop"
        }
    },
    {
        "name": "guard",
        "description": "Move units to patrol/defend a position",
        "parameters": {
            "unit_ids": "list of int, units to guard with",
            "x": "int, patrol x",
            "y": "int, patrol y"
        }
    },
]

VALID_STRUCTURE_TYPES = {"barracks", "war_factory"}
VALID_UNIT_TYPES = {"gi", "grizzly", "prism"}


def parse_action(tool_name: str, params: dict) -> dict:
    """Convert tool call name + params into engine action dict."""
    action = {"action": tool_name}
    action.update(params)
    return action


def parse_action_text(text: str) -> dict | None:
    """Parse a model-generated text line into an action dict.

    Formats:
        build barracks 3 3
        produce gi 1
        move 1,2,3 8 8
        attack 1,2 5
        stop 1,2,3
        guard 1,2 5 10
    """
    text = text.strip().lower()
    if not text:
        return None
    parts = text.split()
    tool = parts[0]
    if tool not in ("build", "produce", "move", "attack", "stop", "guard"):
        return None
    try:
        if tool == "build":
            if parts[1] not in VALID_STRUCTURE_TYPES:
                return None
            return {"action": "build", "structure_type": parts[1],
                    "x": int(parts[2]), "y": int(parts[3])}
        if tool == "produce":
            if parts[1] not in VALID_UNIT_TYPES:
                return None
            return {"action": "produce", "unit_type": parts[1],
                    "structure_id": int(parts[2])}
        if tool == "move":
            uids = [int(x) for x in parts[1].split(",")]
            return {"action": "move", "unit_ids": uids,
                    "x": int(parts[2]), "y": int(parts[3])}
        if tool == "attack":
            uids = [int(x) for x in parts[1].split(",")]
            return {"action": "attack", "unit_ids": uids,
                    "target_id": int(parts[2])}
        if tool == "stop":
            uids = [int(x) for x in parts[1].split(",")]
            return {"action": "stop", "unit_ids": uids}
        if tool == "guard":
            uids = [int(x) for x in parts[1].split(",")]
            return {"action": "guard", "unit_ids": uids,
                    "x": int(parts[2]), "y": int(parts[3])}
    except (IndexError, ValueError):
        return None
    return None


def format_tools_prompt() -> str:
    """Return tool definitions as a compact prompt for the model."""
    lines = ["AVAILABLE ACTIONS:"]
    for t in ACTION_TOOLS:
        p = t["parameters"]
        args_str = ", ".join(f"{k}={v}" for k, v in p.items())
        lines.append(f"  {t['name']}({args_str})")
    return "\n".join(lines)
