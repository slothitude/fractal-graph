"""Web UI for Fractal Graph visualization — Flask + D3.js force-directed graph."""

import json

import db
from config import settings
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph")
def api_graph():
    """Full graph as D3-compatible JSON: {nodes: [...], links: [...]}."""
    conn = db.get_db()
    rows = conn.execute("SELECT * FROM nodes").fetchall()
    nodes = []
    for row in rows:
        d = dict(row)
        if d.get("metadata"):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass
        nodes.append(d)

    edge_rows = conn.execute("SELECT * FROM edges").fetchall()
    links = []
    for row in edge_rows:
        links.append({
            "source": row["from_node_id"],
            "target": row["to_node_id"],
            "type": row["edge_type"],
            "confidence": row["confidence"],
        })

    return jsonify({"nodes": nodes, "links": links})


@app.route("/api/node/<int:node_id>")
def api_node(node_id):
    """Node details with children, parent, and edges."""
    conn = db.get_db()
    node = db.get_node(conn, node_id)
    if not node:
        return jsonify({"error": "Node not found"}), 404

    children = db.get_children(conn, node_id)
    edges = db.get_edges(conn, node_id)
    parent = db.get_node(conn, node["parent_id"]) if node.get("parent_id") else None

    return jsonify({
        "node": node,
        "parent": parent,
        "children": children,
        "edges": edges,
    })


@app.route("/api/stats")
def api_stats():
    """Graph statistics."""
    conn = db.get_db()

    node_counts = {}
    for level in range(6):
        count = conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE resolution_level = ?", (level,)
        ).fetchone()["c"]
        node_counts[level] = count
    total_nodes = sum(node_counts.values())

    edge_counts = {}
    for row in conn.execute(
        "SELECT edge_type, COUNT(*) as c FROM edges GROUP BY edge_type"
    ):
        edge_counts[row["edge_type"]] = row["c"]
    total_edges = sum(edge_counts.values())

    return jsonify({
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "nodes_per_level": node_counts,
        "edges_by_type": edge_counts,
    })


@app.route("/api/search")
def api_search():
    """Text search across nodes (LIKE + case-insensitive)."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"results": []})

    conn = db.get_db()
    rows = conn.execute(
        "SELECT * FROM nodes WHERE content LIKE ? COLLATE NOCASE ORDER BY resolution_level, confidence DESC LIMIT 50",
        (f"%{q}%",),
    ).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        if d.get("bbox"):
            try: d["bbox"] = json.loads(d["bbox"])
            except: pass
        if d.get("metadata"):
            try: d["metadata"] = json.loads(d["metadata"])
            except: pass
        results.append(d)

    return jsonify({"results": results, "query": q})


@app.route("/api/domains")
def api_domains():
    """L0 domains with child counts."""
    conn = db.get_db()
    domains = db.get_nodes_by_resolution(conn, 0)
    result = []
    for domain in domains:
        children = db.get_children(conn, domain["id"])
        # Count total descendants
        count = conn.execute(
            "SELECT COUNT(*) as c FROM nodes WHERE parent_id = ?", (domain["id"],)
        ).fetchone()["c"]
        result.append({
            "id": domain["id"],
            "content": domain["content"],
            "confidence": domain["confidence"],
            "direct_children": len(children),
            "total_descendants": count,
        })
    return jsonify(result)


if __name__ == "__main__":
    print(f"Fractal Graph Web UI -> http://localhost:{settings.server_port}")
    app.run(host="0.0.0.0", port=settings.server_port, debug=True)
