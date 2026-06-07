"""Meeseeks — task-scoped ephemeral souls.

Inspired by Rick and Morty's Mr. Meeseeks: spawned for a single task,
inherits parent soul's graph as read-only context, monitors consistency
as existential state, auto-decomposes when suffering, and releases
(deletes its graph) on completion.

Core insight: consistency = existential state.
  High consistency -> content Meeseeks.
  Low consistency -> suffering Meeseeks -> task decomposition.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from uuid import uuid4

import db
from config import settings
from embedder import embed
from reasoning import decide_monte_carlo

logger = logging.getLogger(__name__)

# In-memory registry — ephemeral by design (lost on restart)
_registry: dict[str, "MeeseeksInstance"] = {}


@dataclass
class MeeseeksInstance:
    instance_id: str
    soul_id: str
    parent_soul: str
    task: str
    state: str  # spawned|working|complete|suffering|released|decomposed
    steps: int = 0
    consistency_history: list = field(default_factory=list)
    max_steps: int = 20
    suffering_threshold: float = 0.3
    suffering_window: int = 3
    personality: dict = field(default_factory=dict)
    sub_tasks: list = field(default_factory=list)
    sub_meeseeks: list = field(default_factory=list)
    outcome: str = ""
    created_at: float = field(default_factory=time.time)
    repo_soul_id: str = ""  # optional repo graph to inherit


# --- Public API ---


async def spawn_meeseeks(task: str, parent_soul: str = "coder",
                         repo_soul_id: str = None) -> dict:
    """Create a new Meeseeks instance for a task.

    1. Generate instance_id
    2. Load parent soul template for personality
    3. Create L0 identity node in DB + ChromaDB
    4. Return instance info

    Args:
        task: The task description
        parent_soul: Parent soul to inherit personality from
        repo_soul_id: Optional repo soul ID for code-aware graph inheritance
    """
    from souls import load_soul_template

    # Validate parent soul exists
    template = load_soul_template(parent_soul)

    instance_id = f"meeseeks-{uuid4().hex[:8]}"
    personality = template.get("personality", {})

    instance = MeeseeksInstance(
        instance_id=instance_id,
        soul_id=instance_id,
        parent_soul=parent_soul,
        task=task,
        state="spawned",
        max_steps=settings.meeseeks_max_steps,
        suffering_threshold=settings.meeseeks_suffering_threshold,
        suffering_window=settings.meeseeks_suffering_window,
        personality=personality,
        repo_soul_id=repo_soul_id or "",
    )

    # Create L0 identity node
    conn = db.get_db()
    identity_content = f"Meeseeks instance for: {task[:100]}"
    node_id = db.insert_node(
        conn, identity_content, resolution_level=0,
        confidence=0.5, soul_id=instance_id,
        metadata={"task": task[:200], "parent_soul": parent_soul},
    )

    # Embed and index in ChromaDB
    try:
        from chroma_store import upsert_node
        embedding = await embed(identity_content)
        upsert_node(node_id, identity_content, embedding, 0,
                    confidence=0.5, soul_id=instance_id)
    except Exception as e:
        logger.warning("Meeseeks ChromaDB indexing failed: %s", e)

    _registry[instance_id] = instance

    return _instance_summary(instance)


async def meeseeks_step(instance_id: str, options: list[str] = None) -> dict:
    """Execute one MC decision step for a Meeseeks.

    Tracks consistency, transitions state based on existential rules.
    """
    instance = _get_active(instance_id)

    # Build soul_ids: own graph + parent soul (read-only inheritance) + optional repo graph
    soul_ids = [instance_id, instance.parent_soul]
    if instance.repo_soul_id:
        soul_ids.append(instance.repo_soul_id)

    # Run MC decision
    decision = await decide_monte_carlo(
        instance.task,
        options=options,
        simulations=instance.personality.get("mc_simulations", 5),
        pool_size=instance.personality.get("mc_context_pool", 10),
        sample_k=instance.personality.get("mc_context_subset", 3),
        temperature=instance.personality.get("temperature", 0.3),
        soul_ids=soul_ids,
    )

    instance.steps += 1
    consistency = decision.get("consistency", 0.0)
    instance.consistency_history.append(consistency)

    # State transitions
    confidence = decision.get("confidence", 0.0)

    if confidence >= 0.7:
        instance.state = "complete"
        instance.outcome = decision.get("action", "")
    elif _is_suffering(instance):
        instance.state = "suffering"
    elif instance.steps >= instance.max_steps:
        instance.state = "suffering"  # give up
    else:
        instance.state = "working"

    result = {
        "instance_id": instance_id,
        "state": instance.state,
        "step": instance.steps,
        "consistency": consistency,
        "decision": decision,
        "consistency_history": instance.consistency_history,
    }
    return result


async def meeseeks_run(task: str, parent_soul: str = "coder",
                       max_steps: int = None) -> dict:
    """Full Meeseeks lifecycle: spawn -> step until done/decompose -> release/decompose."""
    # Spawn
    spawned = await spawn_meeseeks(task, parent_soul)
    instance_id = spawned["instance_id"]
    instance = _registry[instance_id]

    if max_steps:
        instance.max_steps = max_steps

    # Step loop
    last_result = spawned
    while instance.state in ("spawned", "working"):
        if instance.steps >= instance.max_steps:
            break
        last_result = await meeseeks_step(instance_id)

    # Finalize
    if instance.state == "complete":
        release_result = await release_meeseeks(instance_id)
        return {
            "status": "released",
            "outcome": instance.outcome,
            "steps": instance.steps,
            "consistency_history": instance.consistency_history,
            "release": release_result,
        }
    elif instance.state == "suffering":
        decompose_result = await decompose_meeseeks(instance_id)
        return {
            "status": "decomposed",
            "steps": instance.steps,
            "consistency_history": instance.consistency_history,
            "decompose": decompose_result,
        }
    else:
        return {
            "status": instance.state,
            "steps": instance.steps,
            "consistency_history": instance.consistency_history,
        }


async def release_meeseeks(instance_id: str) -> dict:
    """Write outcome to parent soul graph, delete Meeseeks graph."""
    instance = _get_required_state(instance_id, "complete")

    conn = db.get_db()

    # Write outcome node to parent soul graph
    outcome_text = f"Task completed: {instance.task[:80]}\nOutcome: {instance.outcome}"
    from chroma_store import upsert_node
    try:
        outcome_node_id = db.insert_node(
            conn, outcome_text, resolution_level=4,
            confidence=0.8, soul_id=instance.parent_soul,
            metadata={"meeseeks_instance": instance_id, "task": instance.task[:200]},
        )
        embedding = await embed(outcome_text)
        upsert_node(outcome_node_id, outcome_text, embedding, 4,
                    confidence=0.8, soul_id=instance.parent_soul)
    except Exception as e:
        logger.warning("Failed to write outcome to parent: %s", e)

    # Delete all Meeseeks nodes from SQLite
    nodes_deleted = db.delete_nodes_by_soul(conn, instance_id)

    # Delete all Meeseeks embeddings from ChromaDB
    from chroma_store import delete_by_soul
    embeddings_deleted = delete_by_soul(instance_id)

    instance.state = "released"

    return {
        "instance_id": instance_id,
        "state": "released",
        "outcome": instance.outcome,
        "parent_soul": instance.parent_soul,
        "nodes_deleted": nodes_deleted,
        "embeddings_deleted": embeddings_deleted,
    }


async def decompose_meeseeks(instance_id: str) -> dict:
    """Auto-decompose suffering Meeseeks into sub-tasks with sub-Meeseeks."""
    instance = _get_required_state(instance_id, "suffering")

    # Ask 2B to decompose the task
    decompose_prompt = (
        "You are a task decomposition agent. Break this task into 2-4 simpler sub-tasks. "
        "Each sub-task should be independently solvable.\n\n"
        f"Task: {instance.task}\n\n"
        "Return ONLY a JSON array of strings. Example:\n"
        '["sub-task 1", "sub-task 2", "sub-task 3"]\n'
    )

    from reasoning import _llm_call, _parse_json
    raw = await _llm_call(decompose_prompt, num_predict=512, timeout=30.0)
    parsed = _parse_json(raw)

    if not parsed or not isinstance(parsed, list):
        # Fallback: try to extract from raw text
        lines = [l.strip().strip('"').strip("'").strip("-").strip("*").strip()
                 for l in raw.split("\n") if l.strip() and len(l.strip()) > 5]
        sub_tasks = lines[:4] if lines else [instance.task]
    else:
        sub_tasks = [str(s) for s in parsed if str(s).strip()][:4]

    instance.sub_tasks = sub_tasks
    instance.sub_meeseeks = []

    # Spawn sub-Meeseeks for each sub-task
    for sub_task in sub_tasks:
        try:
            sub = await spawn_meeseeks(sub_task, parent_soul=instance.parent_soul)
            instance.sub_meeseeks.append(sub["instance_id"])
        except Exception as e:
            logger.warning("Failed to spawn sub-Meeseeks: %s", e)

    instance.state = "decomposed"

    return {
        "instance_id": instance_id,
        "state": "decomposed",
        "sub_tasks": sub_tasks,
        "sub_meeseeks": instance.sub_meeseeks,
    }


def list_meeseeks() -> list[dict]:
    """List all active Meeseeks (not released/decomposed)."""
    active = [m for m in _registry.values()
              if m.state not in ("released", "decomposed")]
    return [_instance_summary(m) for m in active]


def get_meeseeks(instance_id: str) -> dict:
    """Get full instance details including consistency history."""
    instance = _registry.get(instance_id)
    if not instance:
        return {"error": f"Meeseeks {instance_id} not found"}
    return _instance_summary(instance, full=True)


# --- Internal helpers ---


def _get_active(instance_id: str) -> MeeseeksInstance:
    """Get an active (not released/decomposed) Meeseeks instance."""
    instance = _registry.get(instance_id)
    if not instance:
        raise ValueError(f"Meeseeks {instance_id} not found")
    if instance.state in ("released", "decomposed"):
        raise ValueError(f"Meeseeks {instance_id} is already {instance.state}")
    return instance


def _get_required_state(instance_id: str, required_state: str) -> MeeseeksInstance:
    """Get a Meeseeks instance that must be in a specific state."""
    instance = _registry.get(instance_id)
    if not instance:
        raise ValueError(f"Meeseeks {instance_id} not found")
    if instance.state != required_state:
        raise ValueError(
            f"Meeseeks {instance_id} is in state '{instance.state}', "
            f"expected '{required_state}'"
        )
    return instance


def _is_suffering(instance: MeeseeksInstance) -> bool:
    """Check if a Meeseeks is suffering (low consistency for N consecutive steps)."""
    history = instance.consistency_history
    window = instance.suffering_window
    threshold = instance.suffering_threshold

    if len(history) < window:
        return False

    return all(c < threshold for c in history[-window:])


def _instance_summary(instance: MeeseeksInstance, full: bool = False) -> dict:
    """Convert instance to dict for MCP responses."""
    summary = {
        "instance_id": instance.instance_id,
        "soul_id": instance.soul_id,
        "parent_soul": instance.parent_soul,
        "task": instance.task,
        "state": instance.state,
        "steps": instance.steps,
        "max_steps": instance.max_steps,
        "outcome": instance.outcome,
        "created_at": instance.created_at,
        "repo_soul_id": instance.repo_soul_id,
    }
    if full:
        summary["consistency_history"] = instance.consistency_history
        summary["suffering_threshold"] = instance.suffering_threshold
        summary["suffering_window"] = instance.suffering_window
        summary["personality"] = instance.personality
        summary["sub_tasks"] = instance.sub_tasks
        summary["sub_meeseeks"] = instance.sub_meeseeks
        if instance.consistency_history:
            summary["avg_consistency"] = round(
                sum(instance.consistency_history) / len(instance.consistency_history), 3
            )
    return summary
