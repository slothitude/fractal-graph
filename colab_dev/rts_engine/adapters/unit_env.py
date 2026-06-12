"""Unit-level environment: GRPO environment_factory for unit-type decisions.

A stripped-down environment where the model controls a single unit
in a pre-generated scenario. Used to train unit-type adapters
(infantry, tank, artillery).
"""

from __future__ import annotations
from ..engine import GameEngine
from ..state_encoder import encode_state
from ..action_vocab import format_tools_prompt, parse_action_text
from ..rewards import unit_balance_reward


class UnitEnv:
    """Single-unit control environment for GRPO training.

    The model controls one unit at a time. A scripted commander
    handles build/produce decisions. The model decides move/attack/stop/guard.
    """

    def __init__(self, map_size=16, max_ticks=200, ticks_per_action=3,
                 seed=None):
        self.map_size = map_size
        self.max_ticks = max_ticks
        self.ticks_per_action = ticks_per_action
        self.seed = seed
        self.engine: GameEngine | None = None
        self.controlled_unit_id: int | None = None
        self.tools_prompt = format_tools_prompt()
        self._total_reward = 0.0

    def reset(self, **kwargs) -> str:
        """Start a new scenario, assign a unit to control."""
        seed = kwargs.get("seed", self.seed)
        self.engine = GameEngine(
            map_size=self.map_size, max_ticks=self.max_ticks, seed=seed)
        self.engine.setup_two_player()
        self._total_reward = 0.0

        # Produce a GI unit for the model to control
        barracks_id = next(
            s.id for s in self.engine.factions[0].structures.values()
            if s.structure_type.value == "barracks")
        self.engine.execute_action(
            0, {"action": "produce", "unit_type": "gi",
                 "structure_id": barracks_id})
        for _ in range(10):
            self.engine.tick()
            if self.engine.factions[0].units:
                break

        if self.engine.factions[0].units:
            self.controlled_unit_id = next(iter(self.engine.factions[0].units))
        else:
            self.controlled_unit_id = None

        return self._observe()

    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        if self.engine is None or self.engine.game_over:
            return "Game is over.", 0.0, True, {}

        action = parse_action_text(action_text)
        if action and self.controlled_unit_id is not None:
            # Inject controlled unit into unit_ids if not present
            if "unit_ids" in action:
                action["unit_ids"] = [self.controlled_unit_id]
            self.engine.execute_action(0, action)

        done = False
        for _ in range(self.ticks_per_action):
            if self.engine.game_over:
                done = True
                break
            self.engine.tick()

        if self.engine.game_over:
            done = True

        reward = self._calc_reward()
        self._total_reward += reward
        obs = self._observe()
        info = {"tick": self.engine.tick_count,
                "total_reward": self._total_reward}
        if done:
            info["winner"] = self.engine.winner
        return obs, reward, done, info

    def _observe(self) -> str:
        if not self.engine:
            return ""
        state = self.engine.get_state(0)
        text = encode_state(state)
        if self.controlled_unit_id is not None:
            text += f"\nCONTROLLING: unit#{self.controlled_unit_id}"
        text += "\n" + self.tools_prompt
        return text

    def _calc_reward(self) -> float:
        if not self.engine:
            return 0.0
        fs = self.engine.factions[0]
        efs = self.engine.factions[1]
        my_units = sum(1 for u in fs.units.values() if u.is_alive)
        enemy_units = sum(1 for u in efs.units.values() if u.is_alive)
        r = unit_balance_reward(my_units, enemy_units)
        if self.engine.game_over:
            if self.engine.winner == 0:
                r += 1.0
            elif self.engine.winner == 1:
                r -= 1.0
        return r

    def reward(self) -> float:
        return self._total_reward
