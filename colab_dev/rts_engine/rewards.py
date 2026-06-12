"""Reward shaping functions for GRPO training."""


def game_outcome_reward(result: str) -> float:
    return {"win": 1.0, "loss": -1.0, "draw": 0.0}.get(result, 0.0)


def unit_balance_reward(my_units: int, enemy_units: int) -> float:
    return (my_units - enemy_units) * 0.01


def structure_health_reward(my_hp: int, enemy_hp: int) -> float:
    return (my_hp - enemy_hp) * 0.001


def combined_reward(result, my_units, enemy_units, my_structure_hp=0, enemy_structure_hp=0) -> float:
    r = unit_balance_reward(my_units, enemy_units)
    r += structure_health_reward(my_structure_hp, enemy_structure_hp)
    if result is not None:
        r += game_outcome_reward(result)
    return r
