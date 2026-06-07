"""Code ingestion pipeline — per-repo graph seeding for SWE-bench.

Walks a codebase, extracts structural elements (modules, files, functions,
imports, code snippets), and stores them as a resolution-hierarchy in the
Fractal Graph. Also ingests GitHub issues and diffs into the repo graph.

Resolution mapping:
    L0: Repository identity
    L1: Module/package (top-level directory)
    L2: File (path + purpose)
    L3: Function/class (signature + docstring)
    L4: Import/dependency, issue symptom, diff hunk
    L5: Code snippet, stack trace, error message
"""

import logging
import os
import re
from pathlib import Path

import db
from config import settings
from embedder import embed, embed_batch
from chroma_store import upsert_node

logger = logging.getLogger(__name__)

# --- Regex patterns for code extraction ---

_PY_CLASS_RE = re.compile(r'^([ \t]*)(class\s+(\w+)[^\n]*(?::)[^\n]*)', re.MULTILINE)
_PY_FUNC_RE = re.compile(r'^([ \t]*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)', re.MULTILINE)
_PY_IMPORT_RE = re.compile(
    r'^(?:from\s+([\w.]+)\s+import\s+(.+)|import\s+([\w.,\s]+))', re.MULTILINE
)

_JS_FUNC_RE = re.compile(
    r'^(?:export\s+)?(?:async\s+)?(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?)',
    re.MULTILINE
)
_JS_CLASS_RE = re.compile(r'^(?:export\s+)?(?:[ \t]*)(?:class\s+(\w+))', re.MULTILINE)

# --- Skip patterns ---

_SKIP_DIRS = frozenset(settings.code_ingest_skip_dirs)
_SKIP_EXTS = frozenset(settings.code_ingest_skip_exts)

_TEXT_EXTS = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh", ".ps1",
    ".yaml", ".yml", ".toml", ".cfg", ".ini", ".json", ".md", ".rst", ".txt",
    ".html", ".css", ".scss", ".less", ".sql", ".r", ".lua", ".pl", ".ex",
    ".exs", ".erl", ".hs", ".ml", ".scala", ".kt", ".swift", ".dart",
})


def _is_source_file(filepath: str) -> bool:
    """Check if a file is a source file worth parsing."""
    p = Path(filepath)
    if p.name.startswith(".") or p.name.startswith("__"):
        return p.suffix in _TEXT_EXTS
    return p.suffix in _TEXT_EXTS


def _should_skip(dirpath: str) -> bool:
    """Check if a directory should be skipped."""
    parts = Path(dirpath).parts
    return any(part in _SKIP_DIRS for part in parts)


def _detect_language(filepath: str) -> str:
    """Detect language from file extension."""
    ext = Path(filepath).suffix.lower()
    lang_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".rs": "rust",
        ".go": "go", ".java": "java", ".c": "c", ".cpp": "cpp",
        ".h": "c", ".hpp": "cpp", ".cs": "csharp", ".rb": "ruby",
        ".php": "php", ".sh": "shell", ".bash": "shell", ".swift": "swift",
        ".kt": "kotlin", ".dart": "dart", ".hs": "haskell",
    }
    return lang_map.get(ext, "unknown")


def _guess_file_purpose(filepath: str) -> str:
    """Guess a file's purpose from its name/path."""
    name = Path(filepath).stem.lower()
    path_parts = Path(filepath).parts

    purpose_hints = {
        "test": "test suite",
        "spec": "specification/test",
        "config": "configuration",
        "setup": "setup/installation",
        "init": "initialization",
        "main": "entry point",
        "app": "application",
        "index": "module index",
        "util": "utility functions",
        "helper": "helper functions",
        "model": "data model",
        "view": "view/template",
        "ctrl": "controller",
        "controller": "controller",
        "route": "routing",
        "router": "routing",
        "middleware": "middleware",
        "handler": "request handler",
        "service": "service layer",
        "repo": "repository/data access",
        "repository": "repository/data access",
        "schema": "database schema",
        "migration": "database migration",
        "seed": "database seed data",
        "factory": "test factory",
        "mock": "test mock",
        "stub": "test stub",
        "cli": "command-line interface",
        "cmd": "command-line interface",
        "api": "API definition",
        "types": "type definitions",
        "interface": "interface definition",
        "enum": "enumeration",
        "constant": "constants",
        "const": "constants",
        "error": "error definitions",
        "exception": "exception definitions",
        "log": "logging",
        "logger": "logging",
        "auth": "authentication/authorization",
        "permission": "permissions",
        "policy": "policy definitions",
        "cache": "caching",
        "queue": "message queue",
        "worker": "background worker",
        "job": "background job",
        "task": "background task",
        "plugin": "plugin",
        "extension": "extension",
        "hook": "lifecycle hook",
        "event": "event handling",
        "listener": "event listener",
    }

    for part in reversed(path_parts):
        for key, purpose in purpose_hints.items():
            if key in part.lower():
                return purpose

    return "source file"


# --- Core extraction functions ---


def chunk_file(filepath: str, max_lines: int = 100) -> list[dict]:
    """Split a file into chunks at function/class boundaries.

    Each chunk: {content, start_line, end_line, name, type}
    where type is "function", "class", or "block".
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (IOError, OSError):
        return []

    if not lines:
        return []

    lang = _detect_language(filepath)
    chunks = []

    if lang == "python":
        chunks = _chunk_python(lines, max_lines)
    elif lang in ("javascript", "typescript"):
        chunks = _chunk_javascript(lines, max_lines)
    else:
        # Fallback: split by max_lines
        chunks = _chunk_generic(lines, max_lines)

    return chunks


def _chunk_python(lines: list[str], max_lines: int) -> list[dict]:
    """Chunk Python source at class/function boundaries."""
    chunks = []
    current_chunk_lines = []
    chunk_start = 1
    current_indent = 0
    current_name = "<module>"
    current_type = "block"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Check for class or function definition at module or same indent level
        m = _PY_CLASS_RE.match(line)
        if not m:
            m = _PY_FUNC_RE.match(line)

        if m and indent <= current_indent and current_chunk_lines:
            # Save previous chunk
            chunks.append({
                "content": "".join(current_chunk_lines),
                "start_line": chunk_start,
                "end_line": i,
                "name": current_name,
                "type": current_type,
            })
            current_chunk_lines = [line]
            chunk_start = i + 1
            current_indent = indent
            if "class " in line:
                current_name = f"class {_PY_CLASS_RE.match(line).group(3)}"
                current_type = "class"
            else:
                fm = _PY_FUNC_RE.match(line)
                current_name = f"def {fm.group(4)}"
                current_type = "function"
        else:
            current_chunk_lines.append(line)
            # Update indent if this is the first non-blank line of a block
            if stripped and not current_chunk_lines[:-1]:
                current_indent = indent

        i += 1

    # Final chunk
    if current_chunk_lines:
        chunks.append({
            "content": "".join(current_chunk_lines),
            "start_line": chunk_start,
            "end_line": len(lines),
            "name": current_name,
            "type": current_type,
        })

    return chunks


def _chunk_javascript(lines: list[str], max_lines: int) -> list[dict]:
    """Chunk JS/TS source at function/class/export boundaries."""
    chunks = []
    current_lines = []
    start = 1
    name = "<module>"
    ctype = "block"

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        m = _JS_CLASS_RE.match(line) if stripped.startswith("class") or stripped.startswith("export class") else None
        if not m:
            m = _JS_FUNC_RE.match(line)

        if m and current_lines:
            chunks.append({
                "content": "".join(current_lines),
                "start_line": start,
                "end_line": i,
                "name": name,
                "type": ctype,
            })
            current_lines = [line]
            start = i + 1
            if m and "class" in line:
                name = f"class {m.group(1)}"
                ctype = "class"
            else:
                func_name = m.group(1) or m.group(2) or "anonymous"
                name = f"function {func_name}"
                ctype = "function"
        else:
            current_lines.append(line)

    if current_lines:
        chunks.append({
            "content": "".join(current_lines),
            "start_line": start,
            "end_line": len(lines),
            "name": name,
            "type": ctype,
        })

    return chunks


def _chunk_generic(lines: list[str], max_lines: int) -> list[dict]:
    """Fallback: split by max_lines."""
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk_lines = lines[i:i + max_lines]
        chunks.append({
            "content": "".join(chunk_lines),
            "start_line": i + 1,
            "end_line": min(i + max_lines, len(lines)),
            "name": f"block_{i // max_lines + 1}",
            "type": "block",
        })
    return chunks


def extract_code_elements(filepath: str) -> list[dict]:
    """Parse a single file into structural elements.

    Returns: [{name, type, signature, docstring, start_line, end_line, filepath}]
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except (IOError, OSError):
        return []

    if not content.strip():
        return []

    lang = _detect_language(filepath)

    if lang == "python":
        return _extract_python(content, filepath)
    elif lang in ("javascript", "typescript"):
        return _extract_javascript(content, filepath)
    else:
        return []


def _extract_python(content: str, filepath: str) -> list[dict]:
    """Extract classes and functions from Python source."""
    elements = []
    lines = content.split("\n")

    # Extract imports at module level
    for i, line in enumerate(lines):
        m = _PY_IMPORT_RE.match(line)
        if m:
            module = m.group(1) or m.group(3) or ""
            what = m.group(2) or ""
            if module:
                elements.append({
                    "name": f"from {module} import {what}".strip(),
                    "type": "import",
                    "signature": line.strip(),
                    "docstring": "",
                    "start_line": i + 1,
                    "end_line": i + 1,
                    "filepath": filepath,
                })
            else:
                elements.append({
                    "name": f"import {module}".strip(),
                    "type": "import",
                    "signature": line.strip(),
                    "docstring": "",
                    "start_line": i + 1,
                    "end_line": i + 1,
                    "filepath": filepath,
                })

    # Extract classes
    for m in _PY_CLASS_RE.finditer(content):
        cls_name = m.group(3)
        sig = m.group(2).strip()
        start = content[:m.start()].count("\n") + 1

        # Find docstring
        docstring = _extract_docstring(content, m.end())

        # Find end of class (next line at same or lower indent)
        end = _find_block_end(lines, start - 1, 0)

        elements.append({
            "name": cls_name,
            "type": "class",
            "signature": sig,
            "docstring": docstring,
            "start_line": start,
            "end_line": end,
            "filepath": filepath,
        })

    # Collect class spans with their names for method extraction
    class_spans = []
    for cm in _PY_CLASS_RE.finditer(content):
        c_start = content[:cm.start()].count("\n") + 1
        c_end = _find_block_end(lines, c_start - 1, 0)
        cls_name = cm.group(3)
        class_spans.append((c_start, c_end, cls_name))

    for m in _PY_FUNC_RE.finditer(content):
        func_name = m.group(3)
        is_async = bool(m.group(2))
        params = m.group(4)
        prefix = "async " if is_async else ""
        sig = f"{prefix}def {func_name}({params})"
        start = content[:m.start()].count("\n") + 1

        docstring = _extract_docstring(content, m.end())
        end = _find_block_end(lines, start - 1, 0)

        # Check if inside a class span — extract as method
        parent_class = None
        for cs, ce, cls_name in class_spans:
            if cs <= start < ce:
                parent_class = cls_name
                break

        if parent_class:
            # Extract as method with qualified name
            elements.append({
                "name": f"{parent_class}.{func_name}",
                "type": "method",
                "signature": sig,
                "docstring": docstring,
                "start_line": start,
                "end_line": end,
                "filepath": filepath,
                "parent_class": parent_class,
            })
        else:
            # Top-level function
            elements.append({
                "name": func_name,
                "type": "function",
                "signature": sig,
                "docstring": docstring,
                "start_line": start,
                "end_line": end,
                "filepath": filepath,
            })

    return elements


def _extract_javascript(content: str, filepath: str) -> list[dict]:
    """Extract functions and classes from JS/TS source."""
    elements = []

    for m in _JS_CLASS_RE.finditer(content):
        cls_name = m.group(1)
        start = content[:m.start()].count("\n") + 1
        sig = m.group(0).strip()
        end = content.find("\n", m.end())
        end_line = content[:end].count("\n") + 1 if end > 0 else start + 10

        elements.append({
            "name": cls_name,
            "type": "class",
            "signature": sig,
            "docstring": "",
            "start_line": start,
            "end_line": end_line,
            "filepath": filepath,
        })

    for m in _JS_FUNC_RE.finditer(content):
        func_name = m.group(1) or m.group(2) or "anonymous"
        start = content[:m.start()].count("\n") + 1
        sig_line = content[m.start():content.find("\n", m.start())]
        end = content.find("\n", m.end())
        end_line = content[:end].count("\n") + 1 if end > 0 else start + 10

        elements.append({
            "name": func_name,
            "type": "function",
            "signature": sig_line.strip() if sig_line.strip() else f"function {func_name}",
            "docstring": "",
            "start_line": start,
            "end_line": end_line,
            "filepath": filepath,
        })

    return elements


def _extract_docstring(content: str, pos: int) -> str:
    """Extract the first docstring after a definition."""
    # Skip whitespace
    rest = content[pos:]
    rest = rest.lstrip("\n\r ")

    for quote in ('"""', "'''", '"""', "'''"):
        if rest.startswith(quote):
            end = rest.find(quote, len(quote))
            if end > 0:
                return rest[len(quote):end].strip()

    # Try single-line docstrings
    m = re.match(r'(["\'])(.+?)\1', rest)
    if m:
        return m.group(2).strip()

    return ""


def _find_block_end(lines: list[str], start_idx: int, base_indent: int) -> int:
    """Find the end of a code block by tracking indentation."""
    if start_idx >= len(lines):
        return len(lines)

    # Look past the definition line to find the body indent
    first_nonblank = start_idx
    while first_nonblank < len(lines) and not lines[first_nonblank].strip():
        first_nonblank += 1

    if first_nonblank >= len(lines):
        return len(lines)

    body_idx = first_nonblank + 1
    while body_idx < len(lines) and not lines[body_idx].strip():
        body_idx += 1

    if body_idx >= len(lines):
        return len(lines)

    body_indent = len(lines[body_idx]) - len(lines[body_idx].lstrip())
    if body_indent <= base_indent:
        return body_idx + 1

    end = body_idx + 1
    while end < len(lines):
        line = lines[end]
        if line.strip():
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= base_indent:
                break
        end += 1

    return end + 1


# --- Ingestion functions ---


async def ingest_codebase(repo_path: str, soul_id: str,
                          max_files: int = None) -> dict:
    """Ingest a codebase into the Fractal Graph.

    Creates a resolution hierarchy:
        L0: Repo identity
        L1: Module/package nodes
        L2: File nodes
        L3: Function/class nodes
        L4: Import/dependency nodes
        L5: Code snippet nodes

    All nodes are tagged with soul_id for scoped queries.

    Args:
        repo_path: Path to the repository root
        soul_id: Soul ID to tag all nodes with (e.g. "repo-django")
        max_files: Max files to process (default from config)

    Returns:
        {files_processed, nodes_created, edges_created, errors}
    """
    if max_files is None:
        max_files = settings.code_ingest_max_files

    repo = Path(repo_path)
    if not repo.exists():
        return {"error": f"Path does not exist: {repo_path}"}

    conn = db.get_db()
    nodes_created = 0
    edges_created = 0
    files_processed = 0
    errors = []

    # L0: Repo identity
    repo_name = repo.name
    repo_desc = f"{repo_name} — codebase at {repo_path}"

    # Try to get a better description from README
    readme = repo / "README.md"
    if readme.exists():
        try:
            with open(readme, "r", encoding="utf-8", errors="replace") as f:
                first_lines = [l.strip() for l in f.readlines(100) if l.strip()][:5]
            if first_lines:
                repo_desc = f"{repo_name} — {' '.join(first_lines[:3])}"
        except Exception:
            pass

    l0_id = db.insert_node(
        conn, repo_desc, resolution_level=0,
        confidence=0.9, soul_id=soul_id,
        metadata={"type": "repo", "name": repo_name, "path": str(repo.resolve())},
    )
    nodes_created += 1

    # Collect all source files grouped by top-level directory
    modules: dict[str, list[Path]] = {}
    root_files: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(repo):
        # Filter out skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]

        dir_p = Path(dirpath)
        rel = dir_p.relative_to(repo)

        for fname in filenames:
            fp = dir_p / fname
            if not _is_source_file(str(fp)):
                continue

            # Determine which module this belongs to
            if len(rel.parts) == 0 or (len(rel.parts) == 1 and dir_p == repo):
                root_files.append(fp)
            else:
                module_name = rel.parts[0]
                if module_name not in modules:
                    modules[module_name] = []
                modules[module_name].append(fp)

            if files_processed >= max_files:
                break
        if files_processed >= max_files:
            break

    # Process root files
    file_count = len(root_files)
    if file_count > 0:
        # L1: "root" module for top-level files
        l1_id = db.insert_node(
            conn, f"{repo_name} — root module",
            resolution_level=1, parent_id=l0_id,
            confidence=0.7, soul_id=soul_id,
            metadata={"type": "module", "name": "root", "path": str(repo)},
        )
        nodes_created += 1
        db.insert_edge(conn, l0_id, l1_id, edge_type="contains", confidence=0.9)
        edges_created += 1

        for fp in root_files[:max_files - files_processed]:
            result = _ingest_file(conn, fp, l1_id, soul_id)
            nodes_created += result["nodes"]
            edges_created += result["edges"]
            files_processed += 1
            errors.extend(result.get("errors", []))

    # Process each module
    for module_name, files in sorted(modules.items()):
        if files_processed >= max_files:
            break

        remaining = max_files - files_processed

        # L1: Module node
        module_path = repo / module_name
        module_purpose = _guess_file_purpose(str(module_path))
        l1_id = db.insert_node(
            conn, f"{repo_name}/{module_name} — {module_purpose}",
            resolution_level=1, parent_id=l0_id,
            confidence=0.7, soul_id=soul_id,
            metadata={"type": "module", "name": module_name, "path": str(module_path)},
        )
        nodes_created += 1
        db.insert_edge(conn, l0_id, l1_id, edge_type="contains", confidence=0.9)
        edges_created += 1

        for fp in sorted(files)[:remaining]:
            result = _ingest_file(conn, fp, l1_id, soul_id)
            nodes_created += result["nodes"]
            edges_created += result["edges"]
            files_processed += 1
            errors.extend(result.get("errors", []))

    # Batch embed all nodes for this soul_id
    try:
        await _embed_soul_nodes(soul_id)
    except Exception as e:
        logger.warning("Batch embedding failed for %s: %s", soul_id, e)
        errors.append(f"Embedding: {e}")

    return {
        "files_processed": files_processed,
        "nodes_created": nodes_created,
        "edges_created": edges_created,
        "errors": errors[-10:],  # cap error list
    }


def _ingest_file(conn, filepath: Path, parent_l1_id: int,
                 soul_id: str) -> dict:
    """Ingest a single file: L2 file node + L3 function/class nodes + L4 imports + L5 snippets."""
    nodes = 0
    edges = 0
    errors = []

    try:
        rel_path = str(filepath)
        purpose = _guess_file_purpose(str(filepath))

        # L2: File node
        l2_id = db.insert_node(
            conn, f"{rel_path} — {purpose}",
            resolution_level=2, parent_id=parent_l1_id,
            confidence=0.7, soul_id=soul_id,
            metadata={"type": "file", "path": rel_path, "purpose": purpose},
        )
        nodes += 1
        db.insert_edge(conn, parent_l1_id, l2_id, edge_type="contains", confidence=0.9)
        edges += 1

        # Extract elements
        elements = extract_code_elements(str(filepath))

        import_nodes = []  # track for cross-linking

        for elem in elements:
            if elem["type"] == "import":
                # L4: Import/dependency node
                sig = elem.get("signature", elem["name"])
                node_id = db.insert_node(
                    conn, f"imports: {sig}",
                    resolution_level=4, parent_id=l2_id,
                    confidence=0.6, soul_id=soul_id,
                    metadata={"type": "import", "file": rel_path, "line": elem["start_line"]},
                )
                nodes += 1
                db.insert_edge(conn, l2_id, node_id, edge_type="contains", confidence=0.9)
                edges += 1
                import_nodes.append((node_id, elem["name"]))

            elif elem["type"] in ("function", "class"):
                # L3: Function/class node
                docstring = elem.get("docstring", "")
                content = f"{elem['signature']}"
                if docstring:
                    content += f" — {docstring[:200]}"

                l3_id = db.insert_node(
                    conn, content,
                    resolution_level=3, parent_id=l2_id,
                    confidence=0.8, soul_id=soul_id,
                    metadata={
                        "type": elem["type"],
                        "name": elem["name"],
                        "file": rel_path,
                        "line": elem["start_line"],
                        "end_line": elem.get("end_line", elem["start_line"]),
                    },
                )
                nodes += 1
                db.insert_edge(conn, l2_id, l3_id, edge_type="contains", confidence=0.9)
                edges += 1

            elif elem["type"] == "method":
                # L4: Method node — child of parent class L3 node
                # Find the parent class L3 node for this file
                parent_class = elem.get("parent_class", "")
                docstring = elem.get("docstring", "")
                content = f"{elem['signature']}"
                if docstring:
                    content += f" — {docstring[:200]}"

                # We'll link to the class node later via a second pass.
                # For now, store as L4 under the file L2 node.
                method_id = db.insert_node(
                    conn, content,
                    resolution_level=4, parent_id=l2_id,
                    confidence=0.8, soul_id=soul_id,
                    metadata={
                        "type": "method",
                        "name": elem["name"],
                        "parent_class": parent_class,
                        "file": rel_path,
                        "line": elem["start_line"],
                        "end_line": elem.get("end_line", elem["start_line"]),
                    },
                )
                nodes += 1
                db.insert_edge(conn, l2_id, method_id, edge_type="contains", confidence=0.9)
                edges += 1

                # L5: Code snippet (first 10 lines of body)
                try:
                    with open(str(filepath), "r", encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()
                    start = elem["start_line"] - 1  # 0-indexed
                    end = min(start + 12, elem.get("end_line", start + 10), len(all_lines))
                    snippet = "".join(all_lines[start:end]).strip()
                    if snippet:
                        db.insert_node(
                            conn,
                            f"{rel_path}:{elem['start_line']}-{end}\n{snippet[:500]}",
                            resolution_level=5, parent_id=l3_id,
                            confidence=0.7, soul_id=soul_id,
                            metadata={
                                "type": "snippet",
                                "file": rel_path,
                                "start_line": elem["start_line"],
                                "end_line": end,
                            },
                        )
                        nodes += 1
                except Exception:
                    pass

    except Exception as e:
        errors.append(f"{filepath}: {e}")

    return {"nodes": nodes, "edges": edges, "errors": errors}


async def ingest_issue(issue_text: str, repo_soul_id: str) -> dict:
    """Ingest a GitHub issue into a repo's graph.

    Creates:
        L3: Issue node (title + description)
        L4: Symptom nodes (sentences with error keywords)
        L5: Stack trace / error message nodes

    Args:
        issue_text: Full issue text (title + body)
        repo_soul_id: Soul ID of the repo graph to attach to

    Returns:
        {issue_node_id, symptoms_extracted, evidence_nodes}
    """
    conn = db.get_db()

    # Split into title and body
    lines = issue_text.strip().split("\n")
    title = lines[0].strip() if lines else "Unknown Issue"
    body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    # L3: Issue node
    issue_content = f"Issue: {title[:100]}"
    if body:
        issue_content += f"\n{body[:500]}"

    l3_id = db.insert_node(
        conn, issue_content,
        resolution_level=3, confidence=0.9,
        soul_id=repo_soul_id,
        metadata={"type": "issue", "title": title},
    )

    symptoms_extracted = 0
    evidence_nodes = 0

    # L4: Extract symptoms from sentences with error keywords
    error_keywords = ["error", "bug", "fail", "crash", "broken", "wrong",
                      "exception", "traceback", "segmentation fault", "panic"]
    sentences = re.split(r'[.!?\n]', body) if body else []

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            continue
        if any(kw in sent.lower() for kw in error_keywords):
            l4_id = db.insert_node(
                conn, f"Symptom: {sent[:200]}",
                resolution_level=4, parent_id=l3_id,
                confidence=0.7, soul_id=repo_soul_id,
                metadata={"type": "symptom"},
            )
            db.insert_edge(conn, l3_id, l4_id, edge_type="mentions", confidence=0.8)
            symptoms_extracted += 1

    # L5: Extract quoted code / stack traces
    code_blocks = re.findall(r'```[\w]*\n(.*?)```', issue_text, re.DOTALL)
    inline_code = re.findall(r'`([^`]+)`', issue_text)

    for block in code_blocks[:5]:
        block = block.strip()[:500]
        if block:
            db.insert_node(
                conn, f"Code block:\n{block}",
                resolution_level=5, parent_id=l3_id,
                confidence=0.8, soul_id=repo_soul_id,
                metadata={"type": "code_block"},
            )
            evidence_nodes += 1

    for code in inline_code[:10]:
        code = code.strip()
        if code and len(code) > 3:
            db.insert_node(
                conn, f"Inline: {code[:200]}",
                resolution_level=5, parent_id=l3_id,
                confidence=0.6, soul_id=repo_soul_id,
                metadata={"type": "inline_code"},
            )
            evidence_nodes += 1

    # Embed the issue node
    try:
        embedding = await embed(issue_content)
        upsert_node(l3_id, issue_content, embedding, 3,
                    confidence=0.9, soul_id=repo_soul_id)
    except Exception as e:
        logger.warning("Issue embedding failed: %s", e)

    return {
        "issue_node_id": l3_id,
        "symptoms_extracted": symptoms_extracted,
        "evidence_nodes": evidence_nodes,
    }


async def ingest_diff(diff_text: str, repo_soul_id: str) -> dict:
    """Ingest a unified diff into a repo's graph.

    Creates:
        L4: Change nodes per hunk (file + line range + change type)
        L3: New function/class definitions from added lines

    Args:
        diff_text: Unified diff text
        repo_soul_id: Soul ID of the repo graph to attach to

    Returns:
        {hunks_processed, nodes_created}
    """
    conn = db.get_db()

    # Parse hunks: @@ -start,count +start,count @@
    hunk_re = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')
    file_re = re.compile(r'^diff --git a/(.*?) b/(.*)$')

    current_file = ""
    hunks_processed = 0
    nodes_created = 0

    for line in diff_text.split("\n"):
        fm = file_re.match(line)
        if fm:
            current_file = fm.group(1)
            continue

        hm = hunk_re.match(line)
        if hm:
            old_start = int(hm.group(1))
            old_count = int(hm.group(2)) if hm.group(2) else 1
            new_start = int(hm.group(3))
            new_count = int(hm.group(4)) if hm.group(4) else 1
            header = hm.group(5).strip()

            change_type = "modify"
            if old_count == 0:
                change_type = "add"
            elif new_count == 0:
                change_type = "delete"

            l4_id = db.insert_node(
                conn,
                f"Diff: {current_file} lines {new_start}-{new_start + new_count} ({change_type})",
                resolution_level=4, confidence=0.8,
                soul_id=repo_soul_id,
                metadata={
                    "type": "diff_hunk",
                    "file": current_file,
                    "old_start": old_start, "old_count": old_count,
                    "new_start": new_start, "new_count": new_count,
                    "change_type": change_type,
                    "header": header,
                },
            )
            hunks_processed += 1
            nodes_created += 1

        # Check for new function/class definitions in added lines
        elif line.startswith("+") and not line.startswith("+++"):
            added = line[1:].strip()
            for pattern in [_PY_FUNC_RE, _PY_CLASS_RE]:
                m = pattern.match(added)
                if m:
                    db.insert_node(
                        conn, added[:200],
                        resolution_level=3, confidence=0.7,
                        soul_id=repo_soul_id,
                        metadata={"type": "diff_definition", "file": current_file},
                    )
                    nodes_created += 1
                    break

    return {
        "hunks_processed": hunks_processed,
        "nodes_created": nodes_created,
    }


async def _embed_soul_nodes(soul_id: str) -> int:
    """Embed and upsert all un-embedded nodes for a soul."""
    conn = db.get_db()
    nodes = db.get_nodes_by_soul(conn, soul_id)

    if not nodes:
        return 0

    texts = [n["content"][:2000] for n in nodes]
    embeddings = await embed_batch(texts)

    upserted = 0
    for node, emb in zip(nodes, embeddings):
        if emb:
            upsert_node(
                node["id"], node["content"], emb,
                node["resolution_level"],
                parent_id=node.get("parent_id"),
                confidence=node.get("confidence", 0.5),
                soul_id=soul_id,
            )
            upserted += 1

    return upserted
