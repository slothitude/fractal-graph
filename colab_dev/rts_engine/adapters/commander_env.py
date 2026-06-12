"""Commander environment: GRPO environment_factory for strategic decisions.

Wraps GameEngine as a text-based environment where the model issues
one action per model turn, and the engine simulates multiple ticks
(with opponent AI acting) before the next observation.

Rhythm: model acts -> engine runs N ticks (opponent acts during) -> model observes.
"""

from __future__ import annotations
from ..engine import GameEngine
from ..state_encoder import encode_state
from ..action_vocab import format_tools_prompt, parse_action_text
from ..opponents.simple_ai import SimpleAI
from ..rewards import combined_reward


class CommanderEnv:
    """Text-based RTS environment for GRPO training.

    The model plays faction 0, SimpleAI plays faction 1.
    """

    def __init__(self, map_size=16, max_ticks=300, ticks_per_action=5,
                 seed=None):
        self.map_size = map_size
        self.max_ticks = max_ticks
        self.ticks_per_action = ticks_per_action
        self.seed = seed
        self.engine: GameEngine | None = None
        self.opponent: SimpleAI | None = None
        self.faction_id = 0
        self.tools_prompt = format_tools_prompt()
        self._total_reward = 0.0
        self._episode_actions: list[str] = []

    def reset(self, **kwargs) -> str:
        """Start a new game, return initial observation."""
        seed = kwargs.get("seed", self.seed)
        self.engine = GameEngine(
            map_size=self.map_size, max_ticks=self.max_ticks, seed=seed)
        self.engine.setup_two_player()
        self.opponent = SimpleAI(1)
        self._total_reward = 0.0
        self._episode_actions = []
        return self._observe()

    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        """Execute model action, simulate ticks, return (obs, reward, done, info).

        Args:
            action_text: raw model output (e.g. "produce gi 1")

        Returns:
            (observation, reward, done, info_dict)
        """
        if self.engine is None or self.engine.game_over:
            return "Game is over.", 0.0, True, {}

        # Parse and execute model action
        action = parse_action_text(action_text)
        if action:
            result = self.engine.execute_action(self.faction_id, action)
        self._episode_actions.append(action_text.strip())

        # Simulate ticks with opponent acting
        done = False
        for _ in range(self.ticks_per_action):
            if self.engine.game_over:
                done = True
                break
            # Opponent acts
            opp_action = self.opponent.get_action(self.engine)
            if opp_action:
                self.engine.execute_action(1, opp_action)
            self.engine.tick()

        if self.engine.game_over:
            done = True

        # Calculate reward
        reward = self._calc_reward()
        self._total_reward += reward

        obs = self._observe()
        info = {
            "tick": self.engine.tick_count,
            "total_reward": self._total_reward,
            "actions_taken": len(self._episode_actions),
        }
        if done:
            info["winner"] = self.engine.winner
        return obs, reward, done, info

    def _observe(self) -> str:
        """Return current observation as text."""
        state = self.engine.get_state(self.faction_id)
        text = encode_state(state)
        text += "\n" + self.tools_prompt
        return text

    def _calc_reward(self) -> float:
        """Shaping reward based on current game state."""
        if not self.engine:
            return 0.0
        fs = self.engine.factions[self.faction_id]
        efs = self.engine.factions[1 - self.faction_id]
        my_units = sum(1 for u in fs.units.values() if u.is_alive)
        enemy_units = sum(1 for u in efs.units.values() if u.is_alive)
        my_hp = sum(s.hp for s in fs.structures.values() if s.is_alive)
        enemy_hp = sum(s.hp for s in efs.structures.values() if s.is_alive)

        result = None
        if self.engine.game_over:
            if self.engine.winner == self.faction_id:
                result = "win"
            elif self.engine.winner == 1 - self.faction_id:
                result = "loss"
            else:
                result = "draw"
        return combined_reward(result, my_units, enemy_units, my_hp, enemy_hp)

    def reward(self) -> float:
        """Cumulative reward for the episode (TRL interface)."""
        return self._total_reward


def environment_factory(seed=None, **kwargs):
    """TRL-compatible factory function.

    Usage with GRPOTrainer:
        trainer = GRPOTrainer(
            model=model,
            environment_factory=lambda: CommanderEnv(seed=42),
            ...
        )
    """
    env = CommanderEnv(seed=seed)
    env.reset()
    return env
