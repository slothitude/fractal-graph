"""Stage 6 tests: evolutionary loop."""

import pytest
from rts_engine.tournament.evolve import (
    brier_score, build_order_entropy, adapter_capacity_efficiency,
    cull_candidates, mutate_config, AdapterConfig, EvolutionaryLoop,
)


class TestBrierScore:
    def test_perfect_prediction(self):
        preds = {"a": 0.8, "b": 0.2}
        actuals = {"a": 0.8, "b": 0.2}
        assert brier_score(preds, actuals) == 0.0

    def test_bad_prediction(self):
        preds = {"a": 0.9, "b": 0.1}
        actuals = {"a": 0.1, "b": 0.9}
        score = brier_score(preds, actuals)
        assert score > 0

    def test_empty(self):
        assert brier_score({}, {}) == 0.0

    def test_no_overlap(self):
        preds = {"a": 0.5}
        actuals = {"b": 0.5}
        assert brier_score(preds, actuals) == 0.0


class TestBuildOrderEntropy:
    def test_single_action(self):
        assert build_order_entropy(["move 1 8 8"]) == 0.0

    def test_uniform_distribution(self):
        actions = ["build a 1 1", "produce gi 1", "move 1 8 8", "attack 1 2"]
        entropy = build_order_entropy(actions)
        assert entropy == pytest.approx(2.0, abs=0.01)

    def test_empty(self):
        assert build_order_entropy([]) == 0.0


class TestCapacityEfficiency:
    def test_basic(self):
        eff = adapter_capacity_efficiency(0.8, 1_000_000)
        assert eff == 0.8

    def test_zero_params(self):
        assert adapter_capacity_efficiency(0.5, 0) == 0.0


class TestCull:
    def test_cull_bottom(self):
        configs = [AdapterConfig(name=f"c{i}") for i in range(10)]
        results = {f"c{i}": i * 0.1 for i in range(10)}
        survivors = cull_candidates(configs, results, bottom_pct=0.3)
        assert len(survivors) == 7
        assert survivors[0].name == "c3"

    def test_small_pool(self):
        configs = [AdapterConfig(name="a"), AdapterConfig(name="b")]
        survivors = cull_candidates(configs, {"a": 1.0, "b": 0.0}, 0.3)
        assert len(survivors) == 2


class TestMutate:
    def test_mutation_changes_name(self):
        parent = AdapterConfig(name="gen0", generation=0)
        child = mutate_config(parent)
        assert child.name == "gen0_g1"
        assert child.generation == 1

    def test_mutation_preserves_defaults(self):
        parent = AdapterConfig(name="gen0", generation=0)
        child = mutate_config(parent)
        assert child.rank in (4, 8, 16)
        assert child.learning_rate > 0


class TestEvolutionaryLoop:
    def test_single_round(self):
        configs = [AdapterConfig(name=f"c{i}") for i in range(5)]
        loop = EvolutionaryLoop(configs, max_rounds=1)
        result = loop.run_round()
        assert result.round_num == 1
        assert len(result.candidates) == 5

    def test_cull_and_mutate(self):
        configs = [AdapterConfig(name=f"c{i}") for i in range(5)]
        loop = EvolutionaryLoop(configs, max_rounds=1, bottom_cull_pct=0.3)
        loop.run_round()
        # Should have survivors + mutated children
        assert len(loop.candidates) > 5

    def test_full_loop(self):
        configs = [AdapterConfig(name=f"c{i}") for i in range(4)]
        loop = EvolutionaryLoop(configs, max_rounds=3)
        history = loop.run()
        assert len(history) == 3
        assert history[-1].round_num == 3

    def test_with_train_fn(self):
        call_count = [0]
        def mock_train(candidates):
            call_count[0] += 1
            return {c.name: 0.5 for c in candidates}

        configs = [AdapterConfig(name="c0"), AdapterConfig(name="c1")]
        loop = EvolutionaryLoop(configs, train_fn=mock_train, max_rounds=1)
        loop.run()
        assert call_count[0] == 1
