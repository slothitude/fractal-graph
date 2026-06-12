"""Evolutionary loop: train -> tournament -> cull -> mutate -> repeat."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import math


@dataclass
class AdapterConfig:
    """Hyperparameters for a LoRA adapter candidate."""
    name: str
    rank: int = 4
    learning_rate: float = 1e-5
    reward_weight: float = 1.0
    generation: int = 0


@dataclass
class EvolutionResult:
    """Summary of one evolution round."""
    round_num: int
    candidates: list[AdapterConfig]
    winner: Optional[str]
    elo_spread: float
    brier_score: float
    build_order_entropy: float
    win_rates: dict[str, float]


def brier_score(predicted_probs: dict[str, float], actual_results: dict[str, float]) -> float:
    """Compute mean Brier score for predicted win probabilities.

    Args:
        predicted_probs: {name: predicted_win_prob} (0-1)
        actual_results: {name: actual_win_rate} (0-1)
    """
    if not predicted_probs:
        return 0.0
    names = set(predicted_probs.keys()) & set(actual_results.keys())
    if not names:
        return 0.0
    total = 0.0
    for name in names:
        p = max(0.0, min(1.0, predicted_probs[name]))
        a = max(0.0, min(1.0, actual_results[name]))
        total += (p - a) ** 2
    return total / len(names)


def build_order_entropy(actions: list[str]) -> float:
    """Compute Shannon entropy of action types used.

    Higher entropy = more diverse strategies.
    """
    from collections import Counter
    if not actions:
        return 0.0
    counts = Counter()
    for a in actions:
        tool = a.split()[0] if a.strip() else "none"
        counts[tool] += 1
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def adapter_capacity_efficiency(win_rate: float, num_params: int) -> float:
    """Win rate per million LoRA parameters."""
    if num_params == 0:
        return 0.0
    return win_rate / (num_params / 1_000_000)


def cull_candidates(
    candidates: list[AdapterConfig],
    results: dict[str, float],
    bottom_pct: float = 0.3,
) -> list[AdapterConfig]:
    """Remove bottom performers. Returns surviving candidates."""
    if len(candidates) <= 2:
        return list(candidates)
    n_cull = max(1, int(len(candidates) * bottom_pct))
    ranked = sorted(candidates, key=lambda c: results.get(c.name, 0.0))
    return ranked[n_cull:]


def mutate_config(parent: AdapterConfig) -> AdapterConfig:
    """Create a mutated copy of an adapter config."""
    import random
    rng = random.Random()

    child = AdapterConfig(
        name=f"{parent.name}_g{parent.generation + 1}",
        rank=parent.rank,
        learning_rate=parent.learning_rate,
        reward_weight=parent.reward_weight,
        generation=parent.generation + 1,
    )

    # Mutate rank
    if rng.random() < 0.3:
        child.rank = rng.choice([4, 8, 16])

    # Mutate learning rate
    if rng.random() < 0.3:
        child.learning_rate = max(1e-6, min(1e-3, parent.learning_rate * rng.uniform(0.5, 2.0)))

    # Mutate reward weight
    if rng.random() < 0.3:
        child.reward_weight = max(0.1, min(2.0, parent.reward_weight * rng.uniform(0.5, 2.0)))

    return child


class EvolutionaryLoop:
    """Manage the train -> tournament -> cull -> mutate cycle."""

    def __init__(
        self,
        initial_configs: list[AdapterConfig],
        tournament_fn: Optional[Callable] = None,
        train_fn: Optional[Callable] = None,
        max_rounds: int = 10,
        bottom_cull_pct: float = 0.3,
    ):
        self.candidates = list(initial_configs)
        self.tournament_fn = tournament_fn
        self.train_fn = train_fn
        self.max_rounds = max_rounds
        self.bottom_cull_pct = bottom_cull_pct
        self.history: list[EvolutionResult] = []
        self.round_num = 0

    def run_round(self) -> EvolutionResult:
        """Run one evolution round: train -> tournament -> cull -> mutate."""
        self.round_num += 1
        results = {}

        # Train (if train_fn provided)
        if self.train_fn:
            results = self.train_fn(self.candidates)
        else:
            for c in self.candidates:
                results[c.name] = 0.5

        # Tournament (if tournament_fn provided)
        win_rates = {c.name: 0.5 for c in self.candidates}
        elo_spread = 0.0
        action_log = []

        if self.tournament_fn:
            tournament_results = self.tournament_fn(self.candidates)
            win_rates = tournament_results.get("win_rates", win_rates)
            elo_spread = tournament_results.get("elo_spread", 0.0)
            action_log = tournament_results.get("action_log", [])

        # Compute metrics
        bs = brier_score(
            {c.name: 0.5 for c in self.candidates},
            win_rates,
        )
        boe = build_order_entropy(action_log)

        # Find winner
        winner = max(win_rates, key=win_rates.get) if win_rates else None

        result = EvolutionResult(
            round_num=self.round_num,
            candidates=list(self.candidates),
            winner=winner,
            elo_spread=elo_spread,
            brier_score=bs,
            build_order_entropy=boe,
            win_rates=win_rates,
        )
        self.history.append(result)

        # Cull and mutate
        survivors = cull_candidates(self.candidates, results, self.bottom_cull_pct)
        new_candidates = list(survivors)
        for s in survivors:
            child = mutate_config(s)
            new_candidates.append(child)
        self.candidates = new_candidates

        return result

    def run(self) -> list[EvolutionResult]:
        """Run all rounds, return history."""
        for _ in range(self.max_rounds):
            self.run_round()
        return self.history
