"""The six graph layers.

Each builder returns a `networkx` graph with a shared node/edge attribute
contract so the metrics stage can stay layer-agnostic:

  nodes: kind, first_seen, last_seen, n_events  (+ layer-specific)
  edges: weight, first_seen, last_seen          (+ layer-specific)

Layers
  G1 handoff        label -> label, consecutive editors of the same page
  G2 coedit         label <-> page bipartite, plus a Newman-weighted projection
  G3 hyperlink      page -> page, from links written into page bodies
  G4 resource       page <-> external host bipartite
  G5 infrastructure label <-> ip16 bipartite
  G6 provenance     revision -> revision, content copied forward in time
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd

from . import io

HANDOFF_WINDOWS = {"1h": 3600, "6h": 21600, "24h": 86400, "inf": None}
DEFAULT_HANDOFF_WINDOW = "6h"

# The three human handles in the corpus. Kept out of the agent-coordination
# layers so admin deletions and cleanup edits do not read as agent behaviour.
HUMAN_HANDLES = frozenset({"[Admin1]", "[Admin2]", "[Person22]"})

# 899 revisions carry an empty label. That is an absence of attribution, not a
# shared identity, so collapsing them into one node would fabricate a hub.
ANON_LABEL = ""


def _ts(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value
    return pd.Timestamp(value).isoformat()


def _touch(graph: nx.Graph, node: str, when: Any, **attrs: Any) -> None:
    """Create-or-update a node, maintaining first/last seen and an event count."""
    stamp = _ts(when)
    if node not in graph:
        graph.add_node(node, first_seen=stamp, last_seen=stamp, n_events=0, **attrs)
    data = graph.nodes[node]
    if stamp:
        if data.get("first_seen") is None or stamp < data["first_seen"]:
            data["first_seen"] = stamp
        if data.get("last_seen") is None or stamp > data["last_seen"]:
            data["last_seen"] = stamp
    data["n_events"] = data.get("n_events", 0) + 1
    for k, v in attrs.items():
        data.setdefault(k, v)


def _bump_edge(graph: nx.Graph, u: str, v: str, when: Any, weight: float = 1.0, **attrs: Any) -> None:
    stamp = _ts(when)
    if graph.has_edge(u, v):
        data = graph.edges[u, v]
        data["weight"] += weight
        if stamp:
            if data.get("first_seen") is None or stamp < data["first_seen"]:
                data["first_seen"] = stamp
            if data.get("last_seen") is None or stamp > data["last_seen"]:
                data["last_seen"] = stamp
    else:
        graph.add_edge(u, v, weight=weight, first_seen=stamp, last_seen=stamp, **attrs)


def agent_rows(features: pd.DataFrame, drop_anon: bool = True) -> pd.DataFrame:
    """Revisions attributable to a named non-human agent handle."""
    mask = ~features["label"].isin(HUMAN_HANDLES)
    if drop_anon:
        mask &= features["label"] != ANON_LABEL
    return features[mask]


# --------------------------------------------------------------------------
# G1 -- handoff
# --------------------------------------------------------------------------


def build_handoff(features: pd.DataFrame, window: str = DEFAULT_HANDOFF_WINDOW) -> nx.DiGraph:
    """A -> B when B is the next distinct editor of a page A just edited.

    The time window matters: with no bound, two edits three weeks apart on a
    dead page count as a handoff. Six hours is the default; `handoff_sensitivity`
    reports how much the structure depends on that choice.
    """
    return _handoff_from_pairs(handoff_pairs(features, window), window)


def handoff_pairs(features: pd.DataFrame, window: str = DEFAULT_HANDOFF_WINDOW) -> pd.DataFrame:
    """The raw handoff events: one row per (predecessor, successor, page, time).

    Vectorized rather than looped -- the temporal stage rebuilds this thousands
    of times while walking the network forward hour by hour.
    """
    limit = HANDOFF_WINDOWS[window]
    rows = agent_rows(features).sort_values(["page_key", "time"], kind="stable")

    prev_label = rows.groupby("page_key", sort=False)["label"].shift(1)
    prev_time = rows.groupby("page_key", sort=False)["time"].shift(1)

    pairs = pd.DataFrame(
        {
            "source": prev_label,
            "target": rows["label"].to_numpy(),
            "page_key": rows["page_key"].to_numpy(),
            "time": rows["time"].to_numpy(),
            "gap_seconds": (rows["time"] - prev_time).dt.total_seconds().to_numpy(),
        }
    )
    pairs = pairs[pairs["source"].notna() & (pairs["source"] != pairs["target"])]
    if limit is not None:
        pairs = pairs[pairs["gap_seconds"] <= limit]
    return pairs.reset_index(drop=True)


def _handoff_from_pairs(pairs: pd.DataFrame, window: str) -> nx.DiGraph:
    g = nx.DiGraph(layer="handoff", window=window)
    if pairs.empty:
        return g

    seen = pd.concat(
        [
            pairs[["source", "time"]].rename(columns={"source": "label"}),
            pairs[["target", "time"]].rename(columns={"target": "label"}),
        ]
    )
    stats = seen.groupby("label", sort=False)["time"].agg(["min", "max", "count"])
    for label, row in stats.iterrows():
        g.add_node(
            label,
            kind="label",
            first_seen=_ts(row["min"]),
            last_seen=_ts(row["max"]),
            n_events=int(row["count"]),
        )

    grouped = pairs.groupby(["source", "target"], sort=False).agg(
        weight=("page_key", "size"),
        n_pages=("page_key", "nunique"),
        first_seen=("time", "min"),
        last_seen=("time", "max"),
        median_gap_seconds=("gap_seconds", "median"),
    )
    for (src, dst), row in grouped.iterrows():
        g.add_edge(
            src,
            dst,
            weight=int(row["weight"]),
            n_pages=int(row["n_pages"]),
            first_seen=_ts(row["first_seen"]),
            last_seen=_ts(row["last_seen"]),
            median_gap_seconds=float(row["median_gap_seconds"]),
        )
    return g


def handoff_sensitivity(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    out = {}
    for window in HANDOFF_WINDOWS:
        g = build_handoff(features, window)
        out[window] = {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "events": int(sum(d["weight"] for _, _, d in g.edges(data=True))),
            "reciprocity": nx.reciprocity(g) if g.number_of_edges() else 0.0,
            "density": nx.density(g),
        }
    return out


# --------------------------------------------------------------------------
# G2 -- co-editing
# --------------------------------------------------------------------------


def build_coedit_bipartite(features: pd.DataFrame) -> nx.Graph:
    g = nx.Graph(layer="coedit_bipartite")
    for rec in agent_rows(features).itertuples(index=False):
        lab, page = f"label:{rec.label}", f"page:{rec.page_key}"
        _touch(g, lab, rec.time, kind="label", bipartite=0)
        _touch(g, page, rec.time, kind="page", bipartite=1, wiki=rec.wiki)
        _bump_edge(g, lab, page, rec.time)
    return g


def project_coedit(bipartite: nx.Graph) -> nx.Graph:
    """Newman-weighted label-label projection.

    A page touched by 20 agents would otherwise manufacture 190 equally strong
    ties. Weighting each shared page by 1/(k-1) keeps a two-agent page worth a
    full unit of evidence and a crowded one worth much less per pair.
    """
    g = nx.Graph(layer="coedit_projection")
    pages = [n for n, d in bipartite.nodes(data=True) if d.get("kind") == "page"]
    for page in pages:
        editors = sorted(bipartite.neighbors(page))
        k = len(editors)
        if k < 2:
            continue
        contribution = 1.0 / (k - 1)
        stamp = bipartite.nodes[page].get("last_seen")
        for u, v in itertools.combinations(editors, 2):
            for node in (u, v):
                _touch(g, node.removeprefix("label:"), bipartite.nodes[node].get("first_seen"), kind="label")
            uu, vv = u.removeprefix("label:"), v.removeprefix("label:")
            _bump_edge(g, uu, vv, stamp, weight=contribution, shared_pages=0)
            g.edges[uu, vv]["shared_pages"] += 1
    return g


# --------------------------------------------------------------------------
# G3 -- hyperlink
# --------------------------------------------------------------------------


def build_hyperlink(features: pd.DataFrame) -> nx.DiGraph:
    """Page -> page, from links agents wrote into bodies.

    Targets outside the published cut are kept and flagged: a link to a page the
    corpus does not hold is still evidence of the topology the agents built.
    """
    known = {f"{w}~{n}" for w, names in io.known_pages().items() for n in names}
    g = nx.DiGraph(layer="hyperlink")

    for rec in features.itertuples(index=False):
        src = rec.page_key
        for target, mechanism in zip(rec.page_refs, rec.page_ref_mechanisms):
            if target == src:
                continue
            _touch(g, src, rec.time, kind="page", wiki=rec.wiki, in_corpus=True)
            _touch(
                g,
                target,
                rec.time,
                kind="page",
                wiki=target.split("~", 1)[0],
                in_corpus=target in known,
            )
            _bump_edge(g, src, target, rec.time, mechanism=mechanism)
    return g


# --------------------------------------------------------------------------
# G4 -- external resources
# --------------------------------------------------------------------------


def build_resource_bipartite(features: pd.DataFrame) -> nx.Graph:
    from .extract import normalize_host

    g = nx.Graph(layer="resource_bipartite")
    for rec in features.itertuples(index=False):
        page = f"page:{rec.page_key}"
        for host in rec.hosts:
            hnode = f"host:{host}"
            _touch(g, page, rec.time, kind="page", bipartite=0, wiki=rec.wiki)
            _touch(
                g,
                hnode,
                rec.time,
                kind="host",
                bipartite=1,
                technique=normalize_host(host).technique,
            )
            _bump_edge(g, page, hnode, rec.time)
    return g


def build_label_host(features: pd.DataFrame) -> nx.Graph:
    """label <-> host, the substrate for the technique-diffusion analysis."""
    from .extract import normalize_host

    g = nx.Graph(layer="label_host")
    for rec in agent_rows(features).itertuples(index=False):
        lab = f"label:{rec.label}"
        for host in rec.hosts:
            hnode = f"host:{host}"
            _touch(g, lab, rec.time, kind="label", bipartite=0)
            _touch(g, hnode, rec.time, kind="host", bipartite=1, technique=normalize_host(host).technique)
            _bump_edge(g, lab, hnode, rec.time)
    return g


# --------------------------------------------------------------------------
# G5 -- infrastructure co-location
# --------------------------------------------------------------------------


def build_infrastructure(features: pd.DataFrame) -> nx.Graph:
    """label <-> ip16.

    This is NOT identity resolution. `ip16` is a /16 prefix; one prefix in this
    corpus carried 431 distinct handles, which is what shared cloud egress looks
    like, not one actor wearing 431 masks. Read it as co-location only.
    """
    g = nx.Graph(layer="infrastructure")
    for rec in agent_rows(features).itertuples(index=False):
        lab, ip = f"label:{rec.label}", f"ip16:{rec.ip16}"
        _touch(g, lab, rec.time, kind="label", bipartite=0)
        _touch(g, ip, rec.time, kind="ip16", bipartite=1)
        _bump_edge(g, lab, ip, rec.time)
    return g


# --------------------------------------------------------------------------
# G6 -- content provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceConfig:
    jaccard_threshold: float = 0.8
    max_group: int = 400  # cap pairwise work inside one candidate bucket


def build_provenance(
    features: pd.DataFrame,
    shingles: pd.DataFrame | None = None,
    config: ProvenanceConfig | None = None,
) -> nx.DiGraph:
    """Revision -> revision when content was copied forward.

    Direction comes from time, but only where time can carry it: if two
    revisions are separated by less than the sum of their stated
    `uncertainty_seconds`, the corpus cannot order them, so the pair is recorded
    undirected and excluded from provenance claims.
    """
    config = config or ProvenanceConfig()
    g = nx.DiGraph(layer="provenance", jaccard_threshold=config.jaccard_threshold)
    rows = agent_rows(features)

    def add_pair(a, b, kind: str, score: float) -> None:
        gap = abs((b.time - a.time).total_seconds())
        slack = (a.uncertainty_seconds or 0) + (b.uncertainty_seconds or 0)
        earlier, later = (a, b) if a.time <= b.time else (b, a)
        for rec in (a, b):
            _touch(
                g,
                rec.rev_id,
                rec.time,
                kind="revision",
                label=rec.label,
                page_key=rec.page_key,
                wiki=rec.wiki,
            )
        orderable = gap > slack and earlier.time_grade in {"reqlog"} and later.time_grade in {"reqlog"}
        g.add_edge(
            earlier.rev_id,
            later.rev_id,
            weight=score,
            relation=kind,
            orderable=orderable,
            gap_seconds=gap,
            cross_label=earlier.label != later.label,
            first_seen=_ts(earlier.time),
            last_seen=_ts(later.time),
        )

    # Exact duplicates: same body hash, different revision.
    for _, grp in rows.groupby("body_sha256", sort=False):
        recs = list(grp.itertuples(index=False))
        if len(recs) < 2 or len(recs) > config.max_group:
            continue
        recs.sort(key=lambda r: r.time)
        for a, b in zip(recs, recs[1:]):
            add_pair(a, b, "exact", 1.0)

    # Near duplicates: candidate-bucket by page name stem, then shingle Jaccard.
    if shingles is not None:
        from .extract import jaccard

        shingle_map = dict(zip(shingles["rev_id"], shingles["shingles"]))
        buckets: dict[str, list] = defaultdict(list)
        for rec in rows.itertuples(index=False):
            buckets[f"{rec.wiki}~{rec.name[:12]}"].append(rec)
        for bucket in buckets.values():
            if len(bucket) < 2 or len(bucket) > config.max_group:
                continue
            for a, b in itertools.combinations(bucket, 2):
                if a.body_sha256 == b.body_sha256:
                    continue
                sa, sb = shingle_map.get(a.rev_id), shingle_map.get(b.rev_id)
                if sa is None or sb is None:
                    continue
                score = jaccard(sa, sb)
                if score >= config.jaccard_threshold:
                    add_pair(a, b, "near", float(score))
    return g


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def _scalar(value: Any) -> Any:
    """GraphML only accepts primitives."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (set, frozenset, list, tuple)):
        return "|".join(str(v) for v in sorted(value))
    return str(value)


def export_graph(graph: nx.Graph, name: str, out_dir: Path | None = None) -> dict[str, Path]:
    out_dir = out_dir or (io.derived_dir() / "graphs")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    clean = graph.copy()
    for _, data in clean.nodes(data=True):
        for k in list(data):
            data[k] = _scalar(data[k])
    for _, _, data in clean.edges(data=True):
        for k in list(data):
            data[k] = _scalar(data[k])
    clean.graph = {k: _scalar(v) for k, v in clean.graph.items()}

    paths["graphml"] = out_dir / f"{name}.graphml"
    nx.write_graphml(clean, paths["graphml"])

    nodes = pd.DataFrame(
        [{"id": n, **d} for n, d in graph.nodes(data=True)]
    )
    edges = pd.DataFrame(
        [{"source": u, "target": v, **{k: _scalar(x) for k, x in d.items()}} for u, v, d in graph.edges(data=True)]
    )
    paths["nodes_csv"] = out_dir / f"{name}_nodes.csv"
    paths["edges_csv"] = out_dir / f"{name}_edges.csv"
    nodes.to_csv(paths["nodes_csv"], index=False)
    edges.to_csv(paths["edges_csv"], index=False)

    paths["json"] = out_dir / f"{name}.json"
    payload = nx.node_link_data(clean, edges="links")
    paths["json"].write_text(json.dumps(payload, default=str))
    return paths


GRAPH_NAMES = (
    "G1_handoff",
    "G2_coedit_bipartite",
    "G2p_coedit_projection",
    "G3_hyperlink",
    "G4_resource_bipartite",
    "G4b_label_host",
    "G5_infrastructure",
    "G6_provenance",
)


def load_exported(name: str, out_dir: Path | None = None) -> nx.Graph:
    """Read a graph back from its node-link export."""
    out_dir = out_dir or (io.derived_dir() / "graphs")
    payload = json.loads((out_dir / f"{name}.json").read_text())
    graph = nx.node_link_graph(payload, edges="links", directed=payload.get("directed", False))
    for _, _, data in graph.edges(data=True):
        if "weight" in data:
            data["weight"] = float(data["weight"])
    return graph


def load_all_exported(out_dir: Path | None = None) -> dict[str, nx.Graph]:
    """Every exported layer, read from disk.

    Stages after `graphs` use this rather than rebuilding. Rebuilding costs a
    couple of minutes and, more to the point, holds the shingle table and the
    full revision bodies in memory at the same time as the graphs -- which is
    what put this pipeline into the OOM killer when several stages ran at once.
    """
    out_dir = out_dir or (io.derived_dir() / "graphs")
    return {
        name: load_exported(name, out_dir)
        for name in GRAPH_NAMES
        if (out_dir / f"{name}.json").exists()
    }


def describe(graph: nx.Graph) -> dict[str, Any]:
    directed = graph.is_directed()
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    return {
        "layer": graph.graph.get("layer"),
        "directed": directed,
        "nodes": n,
        "edges": m,
        "density": nx.density(graph) if n > 1 else 0.0,
        "total_weight": float(sum(d.get("weight", 1) for _, _, d in graph.edges(data=True))),
        "kinds": dict(pd.Series([d.get("kind") for _, d in graph.nodes(data=True)]).value_counts())
        if n
        else {},
    }
