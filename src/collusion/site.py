"""Builds the JSON payload the interactive page is compiled around.

Layouts are computed here rather than in the browser: a 2,400-node force
simulation takes seconds to settle and looks like static noise while it does.
Shipping coordinates means the network is legible on first paint, and d3 only
has to refine it if the reader drags something.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from . import diffusion, graphs, io, metrics, temporal
from .pipeline import _json_default, label_family_ground_truth, page_family_map

PAYLOAD_NAME = "site_payload.json"
TEMPLATE_TOKEN = "/*__PAYLOAD__*/"


def _force_layout(giant: nx.Graph, seed: int) -> dict[str, tuple[float, float]]:
    """Fruchterman-Reingold, via igraph where available.

    networkx's implementation is dense: it materialises an n-by-n displacement
    array every iteration, which for a few thousand nodes is tens of megabytes
    per step and billions of operations over a full run. igraph does the same
    layout in C on a sparse graph. The networkx path stays as a fallback with a
    reduced iteration count.
    """
    nodes = list(giant.nodes())
    try:
        import random as _random

        import igraph as ig

        # igraph draws from its own RNG. Handing it a seeded Python Random is
        # what makes the layout reproducible between runs.
        ig.set_random_number_generator(_random.Random(seed))
        index = {n: i for i, n in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), edges=[(index[u], index[v]) for u, v in giant.edges()])
        weights = [float(d.get("weight", 1.0)) for _, _, d in giant.edges(data=True)]
        layout = g.layout_fruchterman_reingold(niter=500, weights=weights)
        return {nodes[i]: (float(x), float(y)) for i, (x, y) in enumerate(layout.coords)}
    except Exception:
        # Broad on purpose: any igraph problem should degrade to a working
        # layout rather than lose the page. networkx is slower and denser, so
        # the iteration count comes down to keep it tractable.
        pos = nx.spring_layout(
            giant,
            seed=seed,
            k=1.6 / np.sqrt(max(len(nodes), 1)),
            iterations=60,
            weight="weight",
        )
        return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}


def _normalize(coords: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Scale into roughly [-1, 1] so every layer shares the canvas's coordinates."""
    if not coords:
        return coords
    xs = [p[0] for p in coords.values()]
    ys = [p[1] for p in coords.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    scale = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
    return {n: ((x - cx) / scale, (y - cy) / scale) for n, (x, y) in coords.items()}


def _layout(
    graph: nx.Graph,
    seed: int = metrics.SEED,
    reuse: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Force layout on the giant component, with the periphery ringed outside.

    Laying out disconnected fragments together throws them to the edges at random
    and wastes the canvas. Placing them deliberately on a ring keeps them visible
    as what they are: agents that never connected to anything.

    `reuse` supplies coordinates from another layer over the same node set. The
    co-edit and handoff layers are both over agent handles, so reusing positions
    means switching layers redraws the *edges* while each agent stays put --
    which is the comparison a reader actually wants -- and skips a second
    expensive layout.
    """
    undirected = nx.Graph(graph)
    if undirected.number_of_nodes() == 0:
        return {}
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    giant = undirected.subgraph(components[0])

    if reuse and sum(1 for n in giant if n in reuse) >= 0.9 * giant.number_of_nodes():
        coords = {n: reuse[n] for n in giant if n in reuse}
        missing = [n for n in giant if n not in coords]
        for i, node in enumerate(sorted(missing)):
            angle = 2 * np.pi * i / max(len(missing), 1)
            coords[node] = (float(1.1 * np.cos(angle)), float(1.1 * np.sin(angle)))
    else:
        coords = _normalize(_force_layout(giant, seed))

    outliers = [n for comp in components[1:] for n in comp]
    for i, node in enumerate(sorted(outliers)):
        angle = 2 * np.pi * i / max(len(outliers), 1)
        radius = 1.35 + 0.12 * ((i % 5) / 5)
        coords[node] = (float(radius * np.cos(angle)), float(radius * np.sin(angle)))
    return coords


def _graph_payload(
    graph: nx.Graph,
    node_meta: dict[str, dict[str, Any]] | None = None,
    max_edges: int | None = None,
    reuse_layout: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    undirected = nx.Graph(graph)
    membership = metrics.detect_communities(undirected)
    cores = nx.core_number(undirected)
    coords = _layout(graph, reuse=reuse_layout)

    edges = list(graph.edges(data=True))
    if max_edges and len(edges) > max_edges:
        edges = sorted(edges, key=lambda e: -e[2].get("weight", 1))[:max_edges]
        keep = {u for u, _, _ in edges} | {v for _, v, _ in edges}
    else:
        keep = set(graph.nodes())

    nodes = []
    for node in graph.nodes():
        if node not in keep:
            continue
        x, y = coords.get(node, (0.0, 0.0))
        entry = {
            "id": node,
            "x": round(x, 4),
            "y": round(y, 4),
            "deg": graph.degree(node),
            "com": membership.get(node, -1),
            "core": cores.get(node, 0),
            "n": graph.nodes[node].get("n_events", 0),
            "t0": graph.nodes[node].get("first_seen"),
            "t1": graph.nodes[node].get("last_seen"),
        }
        if graph.is_directed():
            entry["in"] = graph.in_degree(node)
            entry["out"] = graph.out_degree(node)
        if node_meta and node in node_meta:
            entry.update(node_meta[node])
        nodes.append(entry)

    links = [
        {
            "s": u,
            "t": v,
            "w": int(d.get("weight", 1)),
            "t0": d.get("first_seen"),
        }
        for u, v, d in edges
        if u in keep and v in keep
    ]
    return {
        "directed": graph.is_directed(),
        "nodes": nodes,
        "links": links,
        "n_communities": len(set(membership.values())),
        "coords": coords,
    }


def _read_metrics(name: str) -> Any:
    path = io.derived_dir() / "metrics" / name
    return json.loads(path.read_text()) if path.exists() else None


def build_payload(feat: pd.DataFrame, built: dict[str, nx.Graph]) -> Path:
    daily = temporal.activity_series(feat, "1D")
    curve = temporal.percolation_curve(feat)
    # Hourly percolation is 900+ points; 6-hourly reads the same and halves the file.
    curve = curve.iloc[::3].copy()

    families = label_family_ground_truth(feat)
    centrality_by_node = metrics.centrality_table(built["G1_handoff"]).set_index("node")
    handoff_meta = {
        node: {
            "fam": families.get(node),
            "btw": round(float(centrality_by_node["betweenness"].get(node, 0.0)), 6),
            "pr": round(float(centrality_by_node["pagerank"].get(node, 0.0)), 7),
        }
        for node in built["G1_handoff"].nodes()
    }
    page_families = page_family_map()
    hyperlink_meta = {
        node: {"fam": page_families.get(node), "wiki": data.get("wiki")}
        for node, data in built["G3_hyperlink"].nodes(data=True)
    }

    brokers = centrality_by_node.reset_index().nlargest(20, "betweenness")[
        ["node", "degree", "betweenness", "pagerank", "core_number", "brokerage"]
    ]

    adoption = diffusion.adoption_curves(
        diffusion.adoption_events(feat, "techniques"), min_adopters=20
    )
    adoption_daily = (
        adoption.assign(day=adoption["time"].dt.floor("1D"))
        .groupby(["item", "day"], as_index=False)["adopters"]
        .max()
    )

    # Handoff first: the co-edit layer is over the same handles and reuses its
    # coordinates, so an agent sits in the same place in both views.
    handoff_payload = _graph_payload(built["G1_handoff"], handoff_meta)
    graph_payloads = {
        "handoff": handoff_payload,
        "coedit": _graph_payload(
            built["G2p_coedit_projection"],
            handoff_meta,
            max_edges=12000,
            reuse_layout=handoff_payload["coords"],
        ),
        "hyperlink": _graph_payload(built["G3_hyperlink"], hyperlink_meta),
    }
    for entry in graph_payloads.values():
        entry.pop("coords", None)  # positions already baked into each node

    payload = {
        "meta": {
            "generated_from": io.load_manifest()["generated_at"],
            "revisions": int(len(feat)),
            "pages": int(feat["page_key"].nunique()),
            "labels": int(graphs.agent_rows(feat)["label"].nunique()),
            "wikis": sorted(feat["wiki"].unique().tolist()),
            "window": graphs.DEFAULT_HANDOFF_WINDOW,
            "span": [str(feat["time"].min()), str(feat["time"].max())],
        },
        "incidents": [{"date": d, "label": t} for d, t in temporal.INCIDENTS],
        "timeline": [
            {
                "d": row["time"].strftime("%Y-%m-%d"),
                "saves": int(row["saves"]),
                "deletes": int(row["deletes"]),
                "labels": int(row["active_labels"]),
                "pages": int(row["active_pages"]),
                "ips": int(row["active_ip16"]),
            }
            for _, row in daily.iterrows()
        ],
        "percolation": [
            {
                "t": row["time"].strftime("%Y-%m-%dT%H"),
                "giant": round(float(row["giant_fraction"]), 4),
                "nodes": int(row["nodes"]),
                "deg": round(float(row["mean_degree"]), 3),
            }
            for _, row in curve.iterrows()
        ],
        "graphs": graph_payloads,
        "adoption": [
            {
                "item": item,
                "points": [
                    {"d": d.strftime("%Y-%m-%d"), "n": int(n)}
                    for d, n in zip(grp["day"], grp["adopters"])
                ],
            }
            for item, grp in adoption_daily.groupby("item", sort=False)
        ],
        "brokers": brokers.to_dict("records"),
        "structure": _read_metrics("structure.json"),
        "validation": _read_metrics("community_validation.json"),
        "temporal": _read_metrics("temporal.json"),
        "diffusion": _read_metrics("diffusion.json"),
        "extraction": json.loads((io.derived_dir() / "extraction_summary.json").read_text()),
        "graph_inventory": _read_metrics("graph_inventory.json"),
    }

    path = io.derived_dir() / PAYLOAD_NAME
    path.write_text(json.dumps(payload, default=_json_default, separators=(",", ":")))
    render_html()
    return path


def render_html(
    template: Path | None = None,
    payload: Path | None = None,
    out: Path | None = None,
) -> Path | None:
    """Inline the payload into the page template.

    Artifacts cannot fetch external data, so the JSON has to live inside the
    HTML. The template carries a single token where it goes.
    """
    root = io.repo_root()
    template = template or root / "site" / "template.html"
    payload = payload or io.derived_dir() / PAYLOAD_NAME
    out = out or root / "site" / "index.html"
    if not template.exists() or not payload.exists():
        return None
    html = template.read_text()
    if TEMPLATE_TOKEN not in html:
        raise SystemExit(f"template is missing the {TEMPLATE_TOKEN} token")
    out.write_text(html.replace(TEMPLATE_TOKEN, payload.read_text()))
    return out
