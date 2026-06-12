"""Stage 3 tests: GRPO environment wrappers."""

import pytest
from rts_engine.adapters.commander_env import CommanderEnv, environment_factory
from rts_engine.adapters.unit_env import UnitEnv


class TestCommanderEnv:
    def test_reset_returns_obs(self):
        env = CommanderEnv(map_size=16, seed=42)
        obs = env.reset()
        assert "TICK:0" in obs
        assert "CREDITS:2000" in obs
        assert "AVAILABLE ACTIONS:" in obs

    def test_step_returns_tuple(self):
        env = CommanderEnv(map_size=16, seed=42)
        env.reset()
        obs, reward, done, info = env.step("produce gi 1")
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_invalid_action_no_crash(self):
        env = CommanderEnv(map_size=16, seed=42)
        env.reset()
        obs, reward, done, info = env.step("fly away")
        assert isinstance(obs, str)
        assert done is False

    def test_empty_action_no_crash(self):
        env = CommanderEnv(map_size=16, seed=42)
        env.reset()
        obs, reward, done, info = env.step("")
        assert isinstance(obs, str)

    def test_episode_completes(self):
        env = CommanderEnv(map_size=16, max_ticks=30, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 20:
            _, _, done, _ = env.step("stop")
            steps += 1
        assert done, f"Game didn't end in {steps} steps"

    def test_reward_accumulates(self):
        env = CommanderEnv(map_size=16, max_ticks=30, seed=42)
        env.reset()
        total = 0.0
        done = False
        while not done:
            _, reward, done, _ = env.step("stop")
            total += reward
        assert env._total_reward == pytest.approx(total, abs=1e-9)

    def test_info_has_tick(self):
        env = CommanderEnv(map_size=16, max_ticks=30, seed=42)
        env.reset()
        _, _, _, info = env.step("stop")
        assert "tick" in info

    def test_game_over_returns_done(self):
        env = CommanderEnv(map_size=16, max_ticks=5, ticks_per_action=1, seed=42)
        env.reset()
        # Run to game over
        done = False
        while not done:
            _, _, done, _ = env.step("stop")
        assert done
        _, _, done2, _ = env.step("move 1 8 8")
        assert done2

    def test_environment_factory(self):
        env = environment_factory(seed=42)
        assert env is not None
        obs = env.reset()
        assert "TICK:" in obs


class TestUnitEnv:
    def test_reset_produces_unit(self):
        env = UnitEnv(map_size=16, seed=42)
        obs = env.reset()
        assert "CONTROLLING:" in obs

    def test_step_with_move(self):
        env = UnitEnv(map_size=16, seed=42)
        env.reset()
        if env.controlled_unit_id is not None:
            obs, reward, done, info = env.step("move 1 8 8")
            assert isinstance(obs, str)

    def test_episode_completes(self):
        env = UnitEnv(map_size=16, max_ticks=30, seed=42)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 20:
            _, _, done, _ = env.step("stop")
            steps += 1
        assert done
