"""Game replay capture for SFT warmstart data."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReplayStep:
    tick: int
    faction_id: int
    action: dict | None
    action_text: str
    reward: float = 0.0
    state_snapshot: dict = field(default_factory=dict)


@dataclass
class Replay:
    """Full game replay with state snapshots and actions."""
    seed: int
    map_size: int
    max_ticks: int
    winner: Optional[int] = None
    total_ticks: int = 0
    steps: list[ReplayStep] = field(default_factory=list)
    final_reward: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_sft_examples(self, faction_id: int) -> list[dict]:
        """Convert replay to SFT training examples (state -> action pairs)."""
        examples = []
        for step in self.steps:
            if step.faction_id != faction_id:
                continue
            if not step.action_text.strip():
                continue
            examples.append({
                "observation": step.state_snapshot,
                "action": step.action_text,
                "reward": step.reward,
            })
        return examples

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "map_size": self.map_size,
            "max_ticks": self.max_ticks,
            "winner": self.winner,
            "total_ticks": self.total_ticks,
            "num_steps": len(self.steps),
            "final_reward": self.final_reward,
        }
