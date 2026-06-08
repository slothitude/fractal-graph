"""SWE-bench Evaluation Harness — Phase 23.

Bridges the fractal graph's code ingestion pipeline with Meeseeks task-scoped
reasoning to attempt solving real GitHub issues from SWE-bench.

Workflow:
    1. Download SWE-bench dataset (verified-lite split)
    2. Clone repos, checkout bug-introducing commits
    3. Ingest codebases into fractal graph (cached per commit)
    4. Attach issue descriptions to repo graphs
    5. Solve: spawn Meeseeks with swe_solver soul + repo graph context
    6. Parse natural-language actions into structured edits
    7. Generate patches via mother model (2B diagnoses, mother writes code)
    8. Score: exact match, file-level, function-level

Usage:
    python swe.py --ingest --max-repos 1 --max-files 100
    python swe.py --solve --instance-id <id> --max-steps 5
    python swe.py --score data/swe_bench/predictions.json
    python swe.py --run-eval --max-repos 2 --max-steps 5
"""

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

import db
from config import settings
import code_ingest
from meeseeks import spawn_meeseeks, meeseeks_step, release_meeseeks, decompose_meeseeks

logger = logging.getLogger(__name__)

# --- Dataset Loader ---


def load_swe_bench_split(split: str = "verified_lite", max_instances: int = 0) -> list[dict]:
    """Load SWE-bench dataset from local cache or HuggingFace.

    Returns list of instance dicts with keys:
        instance_id, repo, base_commit, problem_statement, hints_text,
        test_patch, patch, version, created_at

    Args:
        split: Dataset variant — "verified_lite" or "test"
        max_instances: Limit loaded instances (0 = all)
    """
    cache_dir = Path(settings.swe_bench_data_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{split}.json"

    if cache_file.exists():
        logger.info("Loading cached dataset from %s", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            instances = json.load(f)
        if max_instances:
            instances = instances[:max_instances]
        return instances

    # Load from HuggingFace datasets library (streaming, no full download)
    logger.info("Loading SWE-bench %s from HuggingFace", split)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench", split="test", streaming=True)

    # Filter: verified_lite is a subset (~300 instances)
    verified_lite_ids = None
    if split == "verified_lite":
        verified_lite_ids = _load_verified_lite_ids()

    instances = []
    for inst in ds:
        if verified_lite_ids is not None and inst["instance_id"] not in verified_lite_ids:
            continue
        # Keep only the fields we need
        instances.append({
            "instance_id": inst["instance_id"],
            "repo": inst["repo"],
            "base_commit": inst["base_commit"],
            "problem_statement": inst["problem_statement"],
            "hints_text": inst.get("hints_text", ""),
            "test_patch": inst.get("test_patch", ""),
            "patch": inst.get("patch", ""),
            "version": inst.get("version", ""),
            "created_at": inst.get("created_at", ""),
        })
        if max_instances and len(instances) >= max_instances:
            break

    logger.info("Loaded %d instances from HuggingFace", len(instances))

    # Cache locally
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(instances, f, indent=2)

    return instances


def _load_verified_lite_ids() -> set[str]:
    """Load the set of verified_lite instance IDs from HuggingFace parquet."""
    cache_dir = Path(settings.swe_bench_data_cache)
    cache_file = cache_dir / "verified_lite_ids.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return set(json.load(f))

    import urllib.request
    url = ("https://huggingface.co/api/datasets/"
           "princeton-nlp/SWE-bench/parquet/verified_lite/train/0.parquet")
    logger.info("Downloading verified_lite index from %s", url)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            import pyarrow.parquet as pq
            table = pq.read_table(resp)
            ids = set(table.column("instance_id").to_pylist())
    except Exception as e:
        logger.warning("Failed to load verified_lite IDs, loading all: %s", e)
        return None

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, indent=2)
    logger.info("Loaded %d verified_lite IDs", len(ids))
    return ids


# --- Git Operations ---


def clone_repo(repo_full_name: str) -> str:
    """Clone a GitHub repo to the local cache. Returns path to repo."""
    cache_dir = Path(settings.swe_bench_repo_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_path = cache_dir / repo_full_name.replace("/", "__")

    if repo_path.exists() and (repo_path / ".git").exists():
        logger.info("Repo already cloned: %s", repo_path)
        return str(repo_path)

    repo_url = f"https://github.com/{repo_full_name}.git"
    logger.info("Cloning %s -> %s", repo_url, repo_path)
    try:
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(repo_path)],
            check=True, capture_output=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        logger.error("Clone timeout for %s", repo_full_name)
        raise
    except subprocess.CalledProcessError as e:
        logger.error("Clone failed for %s: %s", repo_full_name, e.stderr.decode())
        raise

    return str(repo_path)


def checkout_commit(repo_path: str, commit_hash: str) -> bool:
    """Checkout a specific commit in a repo. Returns True on success."""
    try:
        subprocess.run(
            ["git", "checkout", commit_hash, "--quiet"],
            cwd=repo_path, check=True, capture_output=True, timeout=60,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        logger.error("Checkout failed for %s: %s", commit_hash[:8], stderr)
        return False


def get_commit_hash(repo_path: str) -> str:
    """Get current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, check=True, capture_output=True, timeout=10,
        )
        return result.stdout.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


# --- Cached Repo Ingestion ---


def _get_repo_l0_node(soul_id: str) -> dict | None:
    """Get the L0 node for a repo soul."""
    conn = db.get_db()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE soul_id = ? AND resolution_level = 0",
        (soul_id,),
    ).fetchall()
    if rows:
        d = dict(rows[0])
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        return d
    return None


def _get_cached_commit(soul_id: str) -> str:
    """Get the commit hash cached in the repo L0 metadata."""
    l0 = _get_repo_l0_node(soul_id)
    if l0 and l0.get("metadata"):
        return l0["metadata"].get("commit_hash", "")
    return ""


async def ingest_repo_cached(repo_path: str, repo_soul_id: str,
                             commit_hash: str,
                             max_files: int = None) -> dict:
    """Ingest a repo into the graph with commit-based caching.

    1. Check if soul_id has nodes
    2. If yes, check L0 metadata for cached commit hash
    3. If match -> skip (cache hit)
    4. If mismatch -> delete + re-ingest
    5. If no nodes -> ingest fresh

    Returns: {status, nodes_created, files_processed}
    """
    conn = db.get_db()
    existing_count = db.count_nodes_by_soul(conn, repo_soul_id)
    cached_commit = _get_cached_commit(repo_soul_id)

    if max_files is None:
        max_files = settings.swe_bench_max_files_per_repo

    if existing_count > 0:
        if cached_commit == commit_hash:
            # Also check if we need more files than previously ingested
            cached_files = l0_data.get("metadata", {}).get("max_files", 0) if (l0_data := _get_repo_l0_node(repo_soul_id)) else 0
            if cached_files >= max_files:
                logger.info("Cache HIT for %s at commit %s (%d files)",
                            repo_soul_id, commit_hash[:8], existing_count)
                return {"status": "cached", "nodes_created": 0,
                        "files_processed": existing_count}
            else:
                logger.info("Cache HIT commit but need more files (%d < %d), re-ingesting %s",
                            cached_files, max_files, repo_soul_id)
        else:
            logger.info("Cache MISS (commit changed %s -> %s), re-ingesting %s",
                        cached_commit[:8], commit_hash[:8], repo_soul_id)
            db.delete_nodes_by_soul(conn, repo_soul_id)
            try:
                from chroma_store import delete_by_soul
                delete_by_soul(repo_soul_id)
            except Exception as e:
                logger.warning("ChromaDB cleanup failed: %s", e)
    else:
        logger.info("No cache for %s, ingesting fresh", repo_soul_id)

    result = await code_ingest.ingest_codebase(repo_path, repo_soul_id, max_files)

    # Store commit hash in L0 node metadata
    l0 = _get_repo_l0_node(repo_soul_id)
    if l0:
        meta = l0.get("metadata", {})
        meta["commit_hash"] = commit_hash
        meta["max_files"] = max_files
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE nodes SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(meta), now, l0["id"]),
        )
        db._retry_commit(conn)

    result["status"] = "ingested"
    return result


async def attach_issue_to_repo(instance: dict, repo_soul_id: str) -> dict:
    """Attach a SWE-bench instance's issue to the repo graph.

    Creates L3 issue node + L4 symptoms + L5 evidence, then links
    to the repo L0 node.
    """
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "")
    issue_text = f"{problem}"
    if hints:
        issue_text += f"\n\nHints: {hints}"

    # Ingest the issue via code_ingest pipeline
    issue_result = await code_ingest.ingest_issue(issue_text, repo_soul_id)

    # Fix orphan: link issue node to repo L0 node
    conn = db.get_db()
    l0 = _get_repo_l0_node(repo_soul_id)
    if l0:
        try:
            db.insert_edge(
                conn, l0["id"], issue_result["issue_node_id"],
                edge_type="mentions", confidence=0.8,
            )
        except Exception as e:
            logger.warning("Failed to link issue to repo L0: %s", e)

    return issue_result


# --- Batch Ingestion Phase ---


async def run_ingest_phase(split: str = "verified_lite",
                           max_repos: int = None,
                           max_files: int = None) -> dict:
    """Batch ingest: group instances by (repo, commit), ingest once per group.

    Results saved to data/swe_bench/ingest_results.json
    """
    if max_repos is None:
        max_repos = settings.swe_bench_max_repos

    instances = load_swe_bench_split(split)
    logger.info("Loaded %d instances from %s", len(instances), split)

    # Group by (repo, base_commit)
    groups: dict[tuple, list[dict]] = {}
    for inst in instances:
        key = (inst["repo"], inst["base_commit"])
        if key not in groups:
            groups[key] = []
        groups[key].append(inst)

    results = []
    repos_done = 0

    for (repo, commit), group in sorted(groups.items()):
        if repos_done >= max_repos:
            break

        repo_soul_id = f"swe-{repo.replace('/', '-')}"
        logger.info("[%d/%d] Processing %s @ %s (%d instances)",
                     repos_done + 1, min(len(groups), max_repos),
                     repo, commit[:8], len(group))

        try:
            repo_path = clone_repo(repo)
            if not checkout_commit(repo_path, commit):
                results.append({
                    "repo": repo, "commit": commit,
                    "status": "checkout_failed", "instances": len(group),
                })
                continue

            ingest_result = await ingest_repo_cached(
                repo_path, repo_soul_id, commit, max_files,
            )

            # Attach all issues for this commit
            issues_attached = 0
            for inst in group:
                try:
                    await attach_issue_to_repo(inst, repo_soul_id)
                    issues_attached += 1
                except Exception as e:
                    logger.warning("Failed to attach issue %s: %s",
                                   inst.get("instance_id", "?"), e)

            results.append({
                "repo": repo,
                "commit": commit,
                "soul_id": repo_soul_id,
                "status": ingest_result["status"],
                "nodes_created": ingest_result.get("nodes_created", 0),
                "files_processed": ingest_result.get("files_processed", 0),
                "issues_attached": issues_attached,
                "instances": len(group),
            })
            repos_done += 1

        except Exception as e:
            logger.error("Failed to process %s: %s", repo, e)
            results.append({
                "repo": repo, "commit": commit,
                "status": "error", "error": str(e), "instances": len(group),
            })

    # Save results
    out_path = Path(settings.swe_bench_data_cache) / "ingest_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = {
        "total_instances": len(instances),
        "repos_processed": repos_done,
        "repos_total": len(groups),
        "results_path": str(out_path),
        "details": results,
    }
    logger.info("Ingest phase complete: %d/%d repos, saved to %s",
                repos_done, len(groups), out_path)
    return summary


# --- Action-to-Edit Parser ---


def parse_action_to_edit(action_text: str, repo_soul_id: str,
                         problem_text: str = "", repo_path: str = "") -> dict | None:
    """Parse a Meeseeks action string into a structured edit description.

    Uses multiple strategies to resolve the target file:
        1. Graph lookup — search L3/L4 nodes, walk parent chain to L2 file
        2. Filesystem grep — search repo for files containing target function/class
        3. Issue-based — extract key terms from issue text, search filesystem

    Args:
        action_text: Meeseeks action string
        repo_soul_id: Soul ID for the repo graph
        problem_text: Issue description (provides context for file resolution)
        repo_path: Path to the cloned repo on disk

    Returns dict with filepath, target, description or None.
    """
    conn = db.get_db()
    filepath = ""
    target = ""

    # --- Extract structured info from action ---

    # 1. Extract backtick-quoted code references: `ClassName.method` or `function`
    backtick_refs = re.findall(r'`([A-Za-z_]\w*(?:\.[A-Za-z_]\w+)*)`', action_text)

    # 2. Extract quoted file paths
    filepath_match = re.search(r'["\']([^"\']+\.\w+)["\']', action_text)

    # 3. Extract function/class keywords: "method to_string", "class Foo", "in `_rebuild`"
    kw_match = re.search(
        r'(?:function|method|class|in\s+|to\s+|in\s+`?)'
        r'(?:the\s+)?'
        r'`?([A-Za-z_]\w*(?:\.[A-Za-z_]\w+)*)`?',
        action_text, re.IGNORECASE,
    )

    # 4. Extract function-call patterns from action: `cse(...)`, `u.subs(...)`, `Mul(...)`
    call_refs = re.findall(r'\b([A-Za-z_]\w*)\s*\(', action_text)

    # Determine target from backtick refs or keyword match
    if backtick_refs:
        # Use the most specific backtick ref (longest, prefer ClassName.method)
        target = max(backtick_refs, key=len)
    elif kw_match:
        target = kw_match.group(1)
    elif call_refs:
        # Use first function-call ref as target (likely the main function)
        target = call_refs[0]

    # Collect ALL search terms: target, backtick refs, keywords from action + issue
    search_terms = set()
    if target:
        class_name, method_name = (target.split(".", 1) + [""])[:2]
        search_terms.add(target)
        if method_name:
            search_terms.add(method_name)
        if class_name:
            search_terms.add(class_name)
    for ref in backtick_refs:
        parts = ref.split(".")
        search_terms.update(parts)
    for ref in call_refs:
        search_terms.add(ref)

    # Extract key nouns/identifiers from issue text for filesystem search
    issue_terms = set()
    issue_fps = []
    issue_calls = []
    if problem_text:
        # Extract CamelCase identifiers and snake_case function names
        issue_terms.update(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', problem_text))
        issue_terms.update(re.findall(r'\b([a-z][a-z0-9_]*(?:_[a-z][a-z0-9_]+)+)\b', problem_text))
        # Extract identifiers that appear near code syntax (likely function/class names)
        # Matches: cse(, cse., cse[, "cse", etc.
        code_refs = re.findall(r'(?:^|[\s(,])\b([A-Za-z_]\w{2,})\b(?=[\s(,.\)\[/])', problem_text, re.MULTILINE)
        issue_terms.update(code_refs)
        # Extract quoted code references from issue
        issue_terms.update(re.findall(r'`([A-Za-z_]\w*)`', problem_text))
        # Extract function-call patterns from issue text: cse(...), Mul(...), etc.
        issue_calls = re.findall(r'\b([A-Za-z_]\w{2,})\s*\(', problem_text)
        issue_terms.update(issue_calls)
        # Extract the first word of the issue title (often the function/class name)
        title_line = problem_text.strip().split("\n")[0].strip()
        first_word = re.match(r'^([A-Za-z_]\w{2,})', title_line)
        if first_word:
            issue_terms.add(first_word.group(1))
        # Extract file paths mentioned in issue
        issue_fps = re.findall(r'(?:\bfile|in|from|import)\s+[`"\']?([A-Za-z_][\w/]*\.py)[`"\']?', problem_text, re.IGNORECASE)
        issue_fps += re.findall(r'\b(\w[\w/]*\.py)\b', problem_text)

    # If filepath was explicitly quoted in action, use it (highest priority)
    if filepath_match:
        filepath = filepath_match.group(1)

    # Collect candidates from all strategies with scores
    candidates = {}  # filepath -> score

    # Guard: filter empty strings from search_terms
    search_terms = {t for t in search_terms if t and len(t) > 1}
    issue_terms = {t for t in issue_terms if t and len(t) > 1}

    # --- Strategy 1: Graph lookup (score: node match count * 5) ---
    if search_terms and repo_soul_id:
        for term in search_terms:
            for level in [4, 3]:
                rows = conn.execute(
                    "SELECT id, content, metadata, parent_id FROM nodes "
                    "WHERE soul_id = ? AND resolution_level = ? "
                    "AND content LIKE ?",
                    (repo_soul_id, level, f"%{term}%"),
                ).fetchall()
                for row in rows:
                    fp = _resolve_filepath_from_node(conn, row)
                    if fp:
                        candidates[fp] = candidates.get(fp, 0) + 5

    # --- Strategy 2: Filesystem grep by target terms (score: term match * 10) ---
    if search_terms and repo_path and os.path.isdir(repo_path):
        fs_hits = _grep_repo_for_target(repo_path, search_terms, prefer_dirs=[],
                                        return_all=True)
        for fp, count in fs_hits.items():
            candidates[fp] = candidates.get(fp, 0) + count * 10

    # --- Strategy 3: Issue-based filesystem search (score: term match * 8) ---
    if issue_terms and repo_path and os.path.isdir(repo_path):
        prefer_dirs = _extract_module_paths(problem_text)
        fs_hits = _grep_repo_for_target(repo_path, issue_terms,
                                         prefer_dirs=prefer_dirs, return_all=True)
        for fp, count in fs_hits.items():
            candidates[fp] = candidates.get(fp, 0) + count * 8

    # --- Strategy 4: Issue-mentioned file paths (score: direct + 20) ---
    if issue_fps and repo_path:
        for fp in issue_fps:
            full = os.path.join(repo_path, fp)
            if os.path.exists(full):
                candidates[fp] = candidates.get(fp, 0) + 20
            # Try as basename in the repo
            for root, dirs, files in os.walk(repo_path):
                dirs[:] = [d for d in dirs if d not in settings.code_ingest_skip_dirs]
                if os.path.basename(fp) in files:
                    found = os.path.join(root, os.path.basename(fp))
                    candidates[found] = candidates.get(found, 0) + 15
                    break

    # --- Select best candidate ---
    if candidates:
        # Penalize non-source paths
        for fp in list(candidates.keys()):
            lower = fp.lower()
            for bad in ["/bin/", "/scripts/", "/setup", "/tests/", "/examples/",
                         "/docs/", "/build/", "/dist/", "/.tox/", "/__pycache__"]:
                if bad in lower:
                    candidates[fp] -= 15
        # Bonus: if search terms appear in the filepath itself (not just content)
        all_terms = search_terms | issue_terms
        for fp in list(candidates.keys()):
            basename = os.path.basename(fp).replace(".py", "").lower()
            for term in all_terms:
                if term.lower() in basename:
                    candidates[fp] += 25
        filepath = max(candidates, key=candidates.get) if candidates else ""

    # --- Validate: fuzzy match against L2 file nodes ---
    if filepath:
        validated = _validate_filepath_in_graph(conn, repo_soul_id, filepath)
        if validated:
            filepath = validated

    return {
        "filepath": filepath,
        "target": target,
        "description": action_text[:500],
        "raw_action": action_text,
    }


def _resolve_filepath_from_node(conn, node_row) -> str:
    """Walk up the parent chain from a graph node to find the L2 file path."""
    meta = json.loads(node_row["metadata"]) if node_row["metadata"] else {}

    # L4 nodes may have 'file' metadata directly (imports, methods)
    if "file" in meta:
        return meta["file"]

    # Walk parent chain: L4 -> L3 -> L2 (file)
    parent_id = node_row["parent_id"] if node_row["parent_id"] else None
    for _ in range(5):
        if not parent_id:
            break
        parent = conn.execute(
            "SELECT id, content, resolution_level, metadata, parent_id FROM nodes WHERE id = ?",
            (parent_id,),
        ).fetchone()
        if not parent:
            break
        pmeta = json.loads(parent["metadata"]) if parent["metadata"] else {}
        if parent["resolution_level"] <= 2 and "path" in pmeta:
            return pmeta["path"]
        if "file" in pmeta:
            return pmeta["file"]
        parent_id = parent["parent_id"]

    return ""


def _grep_repo_for_target(repo_path: str, search_terms: set[str],
                          prefer_dirs: list[str] = [],
                          return_all: bool = False) -> dict:
    """Search the repo filesystem for files containing the target terms.

    Uses subprocess grep for speed. Returns dict of filepath -> match_count.
    If return_all=True, returns all candidates. Otherwise returns just the
    best one as {best_path: score}.
    """
    result = {}  # filepath -> match_count
    if not search_terms or not repo_path:
        return result

    # Filter out overly generic terms
    generic = {"the", "a", "an", "is", "of", "to", "in", "for", "and", "or",
                "not", "with", "from", "that", "this", "are", "was", "were",
                "be", "been", "have", "has", "had", "can", "will", "would",
                "could", "should", "do", "if", "when", "by", "on", "at"}
    terms = [t for t in search_terms if t.lower() not in generic and len(t) > 2]
    if not terms:
        return result

    # Use up to 3 most specific terms (longest first)
    terms = sorted(terms, key=len, reverse=True)[:3]

    candidates = {}  # filepath -> match_count
    for term in terms:
        # Use word-boundary matching (-w) for short terms to avoid
        # substring false positives (e.g. "cse" matching "replace")
        grep_flags = ["grep", "-rl", "--include=*.py"]
        if len(term) <= 5:
            grep_flags.append("-w")
        grep_flags.extend([term, repo_path])
        try:
            proc = subprocess.run(
                grep_flags,
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        for line in proc.stdout.strip().split("\n"):
            if line:
                # Normalize to forward slashes
                norm = line.replace("\\", "/")
                candidates[norm] = candidates.get(norm, 0) + 1

    if not candidates:
        return result

    if return_all:
        return candidates

    # Score candidates: prefer source/lib directories, penalize non-source
    def _score(path: str, count: int) -> float:
        score = count * 10.0
        lower = path.lower()
        for d in prefer_dirs:
            if d.lower() in lower:
                score += 20.0
        for bad in ["/bin/", "/scripts/", "/setup", "/tests/", "/examples/",
                     "/docs/", "/build/", "/dist/", "/.tox/", "/__pycache__"]:
            if bad in lower:
                score -= 15.0
        score += count * 5.0
        return score

    ranked = sorted(candidates.keys(), key=lambda p: _score(p, candidates[p]), reverse=True)
    return {ranked[0]: _score(ranked[0], candidates[ranked[0]])}


def _extract_module_paths(text: str) -> list[str]:
    """Extract module directory paths from issue text.

    E.g. "django/core/management/commands" or "sympy/core"
    """
    paths = re.findall(r'([a-z][\w/]*(?:/[a-z][\w/]*)+)', text)
    return paths


def _validate_filepath_in_graph(conn, repo_soul_id: str, filepath: str) -> str:
    """Check if a filepath matches an L2 file node in the graph.

    If the graph has the exact file, return the graph's canonical path.
    If not, return the original filepath unchanged.
    """
    rows = conn.execute(
        "SELECT content, metadata FROM nodes "
        "WHERE soul_id = ? AND resolution_level = 2",
        (repo_soul_id,),
    ).fetchall()
    # Normalize both sides to forward slashes
    norm_fp = filepath.replace("\\", "/")
    best_match = ""
    best_score = 0
    for row in rows:
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        node_path = meta.get("path", "").replace("\\", "/")
        if not node_path:
            continue
        # Exact basename match is the minimum bar
        if os.path.basename(norm_fp) == os.path.basename(node_path):
            # Score by path component overlap
            score = sum(1 for a, b in zip(norm_fp.split("/"), node_path.split("/")) if a == b)
            if score > best_score:
                best_score = score
                best_match = node_path
    if best_match and best_score >= 1:
        return best_match
    return filepath


# --- Patch Generation ---


async def generate_patch(edit: dict, repo_path: str, issue_text: str) -> dict | None:
    """Generate a unified diff patch using the mother model.

    1. Read full file from filesystem (graph has snippets, not full files)
    2. Prompt mother with file content + issue + edit description -> diff
    3. Return structured patch info
    """
    filepath = edit.get("filepath", "")
    if not filepath:
        return None

    # Build full path — filepath from graph metadata is relative to project root
    if not os.path.isabs(filepath):
        # Try as repo-relative first, then project-root-relative
        repo_relative = os.path.join(repo_path, os.path.basename(filepath))
        if os.path.exists(repo_relative):
            full_path = repo_relative
        else:
            # Filepath is relative to project root (e.g., data/swe_repos/...)
            project_root = str(Path(__file__).parent)
            full_path = os.path.join(project_root, filepath)
    else:
        full_path = filepath
    if not os.path.exists(full_path):
        # Try finding it by searching common locations
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in settings.code_ingest_skip_dirs]
            if filepath.split("/")[-1] in files:
                full_path = os.path.join(root, filepath.split("/")[-1])
                break

    if not os.path.exists(full_path):
        logger.warning("File not found: %s", full_path)
        return None

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            file_content = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", full_path, e)
        return None

    # Truncate very large files for the prompt
    max_file_chars = 30000
    if len(file_content) > max_file_chars:
        # Include relevant section around target function/class
        target = edit.get("target", "")
        if target:
            idx = file_content.find(f"def {target}")
            if idx == -1:
                idx = file_content.find(f"class {target}")
            if idx >= 0:
                start = max(0, idx - 2000)
                end = min(len(file_content), idx + 15000)
                file_content = (
                    f"# ... {start} lines omitted ...\n"
                    + file_content[start:end]
                    + f"\n# ... {len(file_content) - end} lines omitted ..."
                )
        else:
            file_content = file_content[:max_file_chars] + "\n# ... truncated ..."

    # Call mother model to generate the patch
    from seed import _mother_generate

    prompt = (
        "You are a bug-fixing engineer. Given the following Python source file "
        "and a bug report, generate a minimal unified diff patch that fixes the issue.\n\n"
        f"## File: {filepath}\n\n"
        f"```python\n{file_content}\n```\n\n"
        f"## Bug Report\n{issue_text[:2000]}\n\n"
        f"## Requested Fix\n{edit['description']}\n\n"
        "## Instructions\n"
        "1. Generate ONLY a unified diff (git diff format)\n"
        "2. The diff should be minimal — fix only the root cause\n"
        "3. Include proper @@ hunk headers\n"
        "4. Output ONLY the diff, no explanation\n"
    )

    raw_patch = await _mother_generate(prompt)
    if not raw_patch:
        logger.warning("Mother model returned empty patch")
        return None

    # Clean up the patch output
    patch_text = raw_patch.strip()
    # Remove markdown fences if present
    patch_text = re.sub(r"^```(?:diff)?\s*\n?", "", patch_text)
    patch_text = re.sub(r"\n?```\s*$", "", patch_text)
    # Remove duplicate diff --git headers
    patch_text = re.sub(
        r"(diff --git a/[^\n]+\n){2,}",
        lambda m: m.group(0).split("\n")[0] + "\n",
        patch_text,
    )

    return {
        "filepath": filepath,
        "patch": patch_text,
        "model_patch": f"diff --git a/{filepath} b/{filepath}\n{patch_text}",
        "explanation": edit.get("description", ""),
    }


# --- Solve Loop ---


async def solve_instance(instance: dict, repo_soul_id: str,
                        max_steps: int = 10) -> dict:
    """Solve a single SWE-bench instance.

    1. Spawn Meeseeks with swe_solver soul + repo graph context
    2. Step loop: MC decision -> parse action -> generate patch
    3. Confidence >= 0.7 -> finalize patch
    4. Suffering -> decompose
    5. Release Meeseeks -> return result
    """
    instance_id = instance.get("instance_id", "unknown")
    problem = instance.get("problem_statement", "")
    hints = instance.get("hints_text", "")

    task = f"Fix this bug in the codebase:\n\n{problem}"
    if hints:
        task += f"\n\nHints: {hints}"

    logger.info("Solving instance %s (max_steps=%d)", instance_id, max_steps)

    # Spawn Meeseeks
    spawned = await spawn_meeseeks(
        task=task,
        parent_soul="swe_solver",
        repo_soul_id=repo_soul_id,
    )
    meeseeks_id = spawned["instance_id"]
    logger.info("Spawned Meeseeks %s for %s", meeseeks_id, instance_id)

    best_patch = None
    steps_log = []

    for step in range(max_steps):
        try:
            # Step the Meeseeks
            result = await meeseeks_step(meeseeks_id)
            state = result["state"]
            decision = result.get("decision", {})
            confidence = decision.get("confidence", 0.0)
            action = decision.get("action", "")

            # Handle nested JSON action — 2B sometimes returns the full JSON
            # object as a string in the action field
            if isinstance(action, dict):
                action = action.get("action", json.dumps(action))
            elif isinstance(action, str):
                try:
                    parsed = json.loads(action)
                    if isinstance(parsed, dict) and "action" in parsed:
                        action = parsed["action"]
                except (json.JSONDecodeError, TypeError):
                    pass

            step_log = {
                "step": step + 1,
                "state": state,
                "confidence": confidence,
                "action": action[:200] if action else "",
            }
            steps_log.append(step_log)

            logger.info("  Step %d: state=%s conf=%.2f action=%s",
                        step + 1, state, confidence, action[:80] if action else "none")

            # High confidence -> finalize
            if confidence >= 0.7 and action:
                # Get repo path for filesystem search
                l0 = _get_repo_l0_node(repo_soul_id)
                cur_repo_path = ""
                if l0 and l0.get("metadata"):
                    cur_repo_path = l0["metadata"].get("path", "")

                edit = parse_action_to_edit(action, repo_soul_id,
                                             problem_text=problem,
                                             repo_path=cur_repo_path)
                if edit and edit.get("filepath"):
                    repo_path = cur_repo_path

                    if repo_path:
                        patch = await generate_patch(edit, repo_path, problem)
                        if patch:
                            best_patch = patch
                            logger.info("Generated patch for %s", instance_id)
                            break

            # Suffering -> stop
            if state == "suffering":
                logger.info("Meeseeks suffering at step %d", step + 1)
                break

        except Exception as e:
            logger.warning("Step %d failed: %s", step + 1, e)
            steps_log.append({"step": step + 1, "error": str(e)})
            break

    # Release Meeseeks
    try:
        from meeseeks import _get_active
        instance_obj = _get_active(meeseeks_id)
        if instance_obj.state == "complete":
            await release_meeseeks(meeseeks_id)
        elif instance_obj.state == "suffering":
            await decompose_meeseeks(meeseeks_id)
    except Exception as e:
        logger.warning("Release failed: %s", e)

    return {
        "instance_id": instance_id,
        "model_patch": best_patch.get("model_patch", "") if best_patch else "",
        "patch_details": best_patch,
        "steps": steps_log,
        "status": "patch_generated" if best_patch else "no_patch",
    }


# --- Scoring ---


def score_results(predictions_path: str) -> dict:
    """Score predictions against ground truth.

    Metrics:
        - exact_match: generated patch == ground truth (string)
        - file_level: correct files modified (path overlap)
        - function_level: correct functions targeted (regex on + lines)
    """
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # Load ground truth
    instances = load_swe_bench_split(settings.swe_bench_split)
    truth_map = {inst["instance_id"]: inst for inst in instances}

    exact_matches = 0
    file_hits = 0
    file_total = 0
    func_hits = 0
    func_total = 0
    scored = 0
    details = []

    for pred in predictions:
        iid = pred.get("instance_id", "")
        model_patch = pred.get("model_patch", "")
        if not iid or not model_patch:
            continue

        truth = truth_map.get(iid)
        if not truth:
            continue

        scored += 1
        gt_patch = truth.get("patch", "")

        # Exact match
        is_exact = model_patch.strip() == gt_patch.strip()
        if is_exact:
            exact_matches += 1

        # File-level: extract files from both patches, normalize to forward slashes
        gt_files = {f.replace("\\", "/") for f in re.findall(r'diff --git a/(.*?) b/', gt_patch)}
        pred_files = {f.replace("\\", "/") for f in re.findall(r'diff --git a/(.*?) b/', model_patch)}
        # Also match repo-relative paths (e.g., astropy/coordinates/angles.py)
        pred_files |= {f.replace("\\", "/") for f in re.findall(r'diff --git a/\S+/(.*?) b/', model_patch)}
        if gt_files:
            file_total += 1
            if pred_files & gt_files:  # any overlap
                file_hits += 1

        # Function-level: extract from hunk @@ headers + def/class lines
        # Hunk headers contain the context function, e.g. @@ -314,10 +314,21 @@ def to_string(
        def _extract_hunk_targets(patch_text):
            """Extract function/class names from hunk @@ headers and +def lines."""
            targets = set()
            # From hunk headers: @@ ... @@ <optional> function_name(
            # Handles both real headers (@@ -314,10 +314,21 @@ def to_string()
            # and placeholders (@@ -X,Y +X,Z @@ def to_string())
            for m in re.finditer(
                r'@@[-0-9XYZ,+ ]+@@\s*(?:\w+\s+)?(?:def |class )(\w+)',
                patch_text,
            ):
                targets.add(m.group(1))
            # From + lines that add new function/class definitions
            for line in patch_text.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    m = re.search(r'(?:def |class )(\w+)', line)
                    if m:
                        targets.add(m.group(1))
            return targets

        gt_funcs = _extract_hunk_targets(gt_patch)
        pred_funcs = _extract_hunk_targets(model_patch)
        if gt_funcs:
            func_total += 1
            if pred_funcs & gt_funcs:
                func_hits += 1

        details.append({
            "instance_id": iid,
            "exact_match": is_exact,
            "file_overlap": bool(pred_files & gt_files) if gt_files else None,
            "func_overlap": bool(pred_funcs & gt_funcs) if gt_funcs else None,
            "gt_functions": sorted(gt_funcs),
            "pred_functions": sorted(pred_funcs),
        })

    return {
        "total_scored": scored,
        "exact_matches": exact_matches,
        "exact_match_pct": round(exact_matches / scored * 100, 1) if scored else 0,
        "file_precision": round(file_hits / file_total * 100, 1) if file_total else 0,
        "file_recall": round(file_hits / max(file_total, 1) * 100, 1),
        "func_precision": round(func_hits / func_total * 100, 1) if func_total else 0,
        "details": details,
    }


# --- Full Eval Runner ---


async def run_eval(max_repos: int = None, max_steps: int = 10,
                   split: str = "verified_lite") -> dict:
    """Run full SWE-bench eval: ingest -> solve -> save predictions."""
    if max_repos is None:
        max_repos = settings.swe_bench_max_repos

    # Phase 1: Ingest
    logger.info("=== Phase 1: Ingestion ===")
    ingest_summary = await run_ingest_phase(split, max_repos)
    logger.info("Ingestion complete: %s", json.dumps(ingest_summary, indent=2))

    # Phase 2: Solve
    logger.info("=== Phase 2: Solving ===")
    instances = load_swe_bench_split(split)
    predictions = []

    for inst in instances:
        repo = inst["repo"]
        soul_id = f"swe-{repo.replace('/', '-')}"

        # Check if this repo was actually ingested
        conn = db.get_db()
        node_count = db.count_nodes_by_soul(conn, soul_id)
        if node_count == 0:
            continue

        result = await solve_instance(inst, soul_id, max_steps)
        predictions.append(result)
        logger.info("Solved %s: %s", inst["instance_id"], result["status"])

    # Save predictions
    pred_path = Path(settings.swe_bench_data_cache) / "predictions.json"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    # Phase 3: Score
    logger.info("=== Phase 3: Scoring ===")
    scores = score_results(str(pred_path))
    logger.info("Scores: %s", json.dumps(scores, indent=2))

    # Save scores
    score_path = Path(settings.swe_bench_data_cache) / "scores.json"
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    return {
        "ingestion": ingest_summary,
        "predictions_path": str(pred_path),
        "predictions_count": len(predictions),
        "scores": scores,
    }


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Evaluation Harness")
    parser.add_argument("--ingest", action="store_true",
                        help="Run ingestion phase only")
    parser.add_argument("--solve", action="store_true",
                        help="Solve a single instance")
    parser.add_argument("--score", type=str, metavar="PATH",
                        help="Score predictions from JSON file")
    parser.add_argument("--run-eval", action="store_true",
                        help="Run full eval pipeline (ingest + solve + score)")
    parser.add_argument("--max-repos", type=int, default=None,
                        help="Max repos to ingest (default: config)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max files per repo (default: config)")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max Meeseeks steps per instance")
    parser.add_argument("--instance-id", type=str,
                        help="Instance ID to solve (requires --solve)")
    parser.add_argument("--split", type=str, default="verified_lite",
                        help="SWE-bench split (default: verified_lite)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.ingest:
        asyncio.run(run_ingest_phase(args.split, args.max_repos, args.max_files))

    elif args.solve:
        if not args.instance_id:
            print("Error: --instance-id required with --solve")
            sys.exit(1)

        instances = load_swe_bench_split(args.split)
        inst = next((i for i in instances if i["instance_id"] == args.instance_id), None)
        if not inst:
            print(f"Instance {args.instance_id} not found in {args.split}")
            sys.exit(1)

        soul_id = f"swe-{inst['repo'].replace('/', '-')}"

        # Auto-ingest if needed (wrong commit, no data, or need more files)
        async def _solve_with_ingest():
            cached = _get_cached_commit(soul_id)
            need_ingest = False

            if not cached:
                need_ingest = True
                print(f"No cached data for {soul_id}")
            elif cached != inst["base_commit"]:
                need_ingest = True
                print(f"Commit mismatch: cached={cached[:8]}, need={inst['base_commit'][:8]}")
            else:
                # Check if max_files is sufficient
                l0 = _get_repo_l0_node(soul_id)
                cached_files = l0.get("metadata", {}).get("max_files", 0) if l0 else 0
                requested_files = args.max_files or settings.swe_bench_max_files_per_repo
                if cached_files < requested_files:
                    need_ingest = True
                    print(f"Need more files: cached={cached_files}, requested={requested_files}")

            if need_ingest:
                max_files = args.max_files or settings.swe_bench_max_files_per_repo
                print(f"Ingesting {inst['repo']} @ {inst['base_commit'][:8]}...")
                repo_path = clone_repo(inst["repo"])
                if not checkout_commit(repo_path, inst["base_commit"]):
                    print(f"ERROR: Failed to checkout {inst['base_commit'][:8]}")
                    return {"instance_id": inst["instance_id"],
                            "status": "checkout_failed"}
                await ingest_repo_cached(repo_path, soul_id, inst["base_commit"],
                                         max_files)
                await attach_issue_to_repo(inst, soul_id)

            return await solve_instance(inst, soul_id, args.max_steps)

        result = asyncio.run(_solve_with_ingest())
        print(json.dumps(result, indent=2))

        # Save prediction to predictions.json
        pred_path = Path(settings.swe_bench_data_cache) / "predictions.json"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if pred_path.exists():
            try:
                existing = json.loads(pred_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        # Update or append
        found = False
        for i, p in enumerate(existing):
            if p.get("instance_id") == result["instance_id"]:
                existing[i] = result
                found = True
                break
        if not found:
            existing.append(result)
        pred_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info("Prediction saved to %s (%d total)", pred_path, len(existing))

    elif args.score:
        scores = score_results(args.score)
        print(json.dumps(scores, indent=2))

    elif args.run_eval:
        result = asyncio.run(run_eval(args.max_repos, args.max_steps, args.split))
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
