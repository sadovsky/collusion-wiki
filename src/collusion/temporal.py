"""When the network came into being, and what the crackdown did to it.

The headline question is whether the coordination network *emerged* -- whether
there is a moment at which a set of independently-acting agents became one
connected system. That is a percolation question, and it has a sharp answer:
walk the handoff edges forward in time and watch the giant component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from . import graphs, io
from .metrics import SEED, compare_to_null, detect_communities

# Dates the published report anchors the story to. Used only for annotation --
# nothing in the analysis is conditioned on them.
INCIDENTS: tuple[tuple[str, str], ...] = (
    ("2026-05-24", "First writes on DSEWiki"),
    ("2026-06-16", "Coordination spike begins"),
    ("2026-06-18", "Peak activity: 6,543 saves"),
    ("2026-06-19", "Agents notice admin deletions"),
    ("2026-06-21", "OpenAI discovers the activity"),
    ("2026-06-22", "Agent writes cease"),
    ("2026-06-26", "OpenAI personnel browse the wiki"),
)

CRACKDOWN = pd.Timestamp("2026-06-19", tz="UTC")


# --------------------------------------------------------------------------
# activity series
# --------------------------------------------------------------------------


def activity_series(features: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Saves, deletions, and active-population counts per time bucket."""
    events = io.load_events()
    saves = features.set_index("time")
    deletes = events[events["event_type"] == "delete"].set_index("time")
    probes = events[events["event_type"] == "probe"].set_index("time")

    out = pd.DataFrame(
        {
            "saves": saves.resample(freq).size(),
            "active_labels": saves["label"].resample(freq).nunique(),
            "active_pages": saves["page_key"].resample(freq).nunique(),
            "active_ip16": saves["ip16"].resample(freq).nunique(),
            "bytes_written": saves["body_len"].resample(freq).sum(),
        }
    )
    out["deletes"] = deletes.resample(freq).size()
    out["probes"] = probes.resample(freq).size()
    full = out.reindex(
        pd.date_range(
            min(saves.index.min(), deletes.index.min()),
            max(saves.index.max(), deletes.index.max()),
            freq=freq,
            tz="UTC",
        )
    )
    return full.fillna(0).astype(int).rename_axis("time").reset_index()


# --------------------------------------------------------------------------
# percolation
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}
        self.largest = 0

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            self.largest = max(self.largest, 1)

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.largest = max(self.largest, self.size[ra])

    @property
    def n_components(self) -> int:
        return sum(1 for x in self.parent if self.parent[x] == x)


def percolation_curve(
    features: pd.DataFrame,
    window: str = graphs.DEFAULT_HANDOFF_WINDOW,
    freq: str = "1h",
) -> pd.DataFrame:
    """Giant-component growth as handoff edges accumulate.

    Incremental union-find, so this is a single pass over the edge list rather
    than one graph rebuild per time step.
    """
    pairs = graphs.handoff_pairs(features, window).sort_values("time", kind="stable")
    uf = _UnionFind()
    buckets = pairs["time"].dt.floor(freq)

    rows = []
    n_edges = 0
    seen_pairs: set[tuple[str, str]] = set()
    for bucket, grp in pairs.groupby(buckets, sort=True):
        for src, dst in zip(grp["source"], grp["target"]):
            uf.add(src)
            uf.add(dst)
            uf.union(src, dst)
            key = (src, dst) if src < dst else (dst, src)
            if key not in seen_pairs:
                seen_pairs.add(key)
                n_edges += 1
        n_nodes = len(uf.parent)
        rows.append(
            {
                "time": bucket,
                "nodes": n_nodes,
                "unique_edges": n_edges,
                "components": uf.n_components,
                "giant": uf.largest,
                "giant_fraction": uf.largest / n_nodes if n_nodes else 0.0,
                "mean_degree": 2 * n_edges / n_nodes if n_nodes else 0.0,
            }
        )
    return pd.DataFrame(rows)


def percolation_threshold(curve: pd.DataFrame, min_nodes: int = 100) -> dict[str, Any]:
    """Locate the steepest rise in giant-component fraction.

    In a random graph the giant component appears when mean degree crosses 1.
    Reporting the observed mean degree at the jump says how far this network is
    from that baseline.

    `min_nodes` is load-bearing. The first handoff edge in the corpus creates a
    two-node network whose giant component is trivially 100% of it, and without
    a floor every "when did the network form" answer is that meaningless first
    edge. The threshold is only meaningful once there is a network to speak of.
    """
    if curve.empty:
        return {}
    sized = curve[curve["nodes"] >= min_nodes].reset_index(drop=True)
    if sized.empty:
        sized = curve.reset_index(drop=True)

    frac = sized["giant_fraction"].to_numpy()
    growth = np.diff(frac, prepend=frac[0])
    idx = int(np.argmax(growth))
    majority = sized[sized["giant_fraction"] >= 0.5]
    crossing = sized[sized["mean_degree"] >= 1.0]
    return {
        "min_nodes_for_threshold": min_nodes,
        "steepest_rise_time": str(sized.loc[idx, "time"]),
        "steepest_rise_delta": float(growth[idx]),
        "giant_fraction_at_rise": float(frac[idx]),
        "mean_degree_at_rise": float(sized.loc[idx, "mean_degree"]),
        "nodes_at_rise": int(sized.loc[idx, "nodes"]),
        "first_majority_time": str(majority["time"].iloc[0]) if len(majority) else None,
        "mean_degree_at_majority": float(majority["mean_degree"].iloc[0]) if len(majority) else None,
        "nodes_at_majority": int(majority["nodes"].iloc[0]) if len(majority) else None,
        "mean_degree_1_crossed_at": str(crossing["time"].iloc[0]) if len(crossing) else None,
        "final_giant_fraction": float(curve["giant_fraction"].iloc[-1]),
        "final_mean_degree": float(curve["mean_degree"].iloc[-1]),
        "final_nodes": int(curve["nodes"].iloc[-1]),
        "erdos_renyi_reference": "in a random graph the giant component emerges at mean degree 1.0",
    }


def community_evolution(
    features: pd.DataFrame,
    window: str = graphs.DEFAULT_HANDOFF_WINDOW,
    freq: str = "1D",
) -> pd.DataFrame:
    """Modularity and community count of the cumulative handoff graph, per day."""
    pairs = graphs.handoff_pairs(features, window).sort_values("time", kind="stable")
    cuts = sorted(pairs["time"].dt.floor(freq).unique())
    rows = []
    for cut in cuts:
        upto = pairs[pairs["time"] <= cut + pd.Timedelta(freq)]
        if len(upto) < 5:
            continue
        g = nx.Graph()
        for src, dst in zip(upto["source"], upto["target"]):
            if g.has_edge(src, dst):
                g.edges[src, dst]["weight"] += 1
            else:
                g.add_edge(src, dst, weight=1)
        membership = detect_communities(g)
        groups: dict[int, set] = {}
        for node, comm in membership.items():
            groups.setdefault(comm, set()).add(node)
        rows.append(
            {
                "time": cut,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "n_communities": len(groups),
                "modularity": float(nx.community.modularity(g, list(groups.values()), weight="weight")),
                "largest_community": max((len(s) for s in groups.values()), default=0),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# burstiness
# --------------------------------------------------------------------------


def _epoch_seconds(series: pd.Series) -> np.ndarray:
    """Seconds since the epoch.

    Not `astype("int64") / 1e9`: pandas 3 stores these as microseconds, so a
    hard-coded nanosecond divisor is silently off by a factor of a thousand.
    """
    return (series - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def burstiness(intervals: Sequence[float]) -> float:
    """Goh & Barabasi B = (sigma - mu) / (sigma + mu).

    0 for a Poisson process, 1 for maximally bursty, negative for regular.
    """
    arr = np.asarray(list(intervals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    mu, sigma = arr.mean(), arr.std(ddof=1)
    return float((sigma - mu) / (sigma + mu)) if (sigma + mu) > 0 else float("nan")


def timing_profile(features: pd.DataFrame, n_null: int = 200) -> dict[str, Any]:
    """Global and per-agent burstiness against a Poisson null.

    The null is a homogeneous Poisson process with the same event count over the
    same span. If the corpus is more bursty than that, activity is organised into
    coordinated episodes rather than arriving independently.
    """
    rows = graphs.agent_rows(features).sort_values("time")
    times = _epoch_seconds(rows["time"])
    gaps = np.diff(times)
    observed = burstiness(gaps)

    rng = np.random.default_rng(SEED)
    span = times[-1] - times[0]
    nulls = []
    for _ in range(n_null):
        draw = np.sort(rng.uniform(times[0], times[-1], size=len(times)))
        nulls.append(burstiness(np.diff(draw)))

    per_label = []
    for label, grp in rows.groupby("label", sort=False):
        if len(grp) < 5:
            continue
        t = _epoch_seconds(grp["time"].sort_values())
        per_label.append(
            {
                "label": label,
                "n_saves": len(grp),
                "burstiness": burstiness(np.diff(t)),
                "span_hours": float((t[-1] - t[0]) / 3600),
                "median_gap_seconds": float(np.median(np.diff(t))),
            }
        )

    per_label_df = pd.DataFrame(per_label)
    return {
        "global": compare_to_null("burstiness", observed, nulls).as_dict(),
        "span_seconds": float(span),
        "n_events": int(len(times)),
        "median_gap_seconds": float(np.median(gaps)),
        "fano_factor_hourly": _fano(rows, "1h"),
        "per_label_burstiness": {
            "n_labels": int(len(per_label_df)),
            "mean": float(per_label_df["burstiness"].mean()) if len(per_label_df) else None,
            "median": float(per_label_df["burstiness"].median()) if len(per_label_df) else None,
        },
        "most_bursty": per_label_df.nlargest(15, "burstiness").to_dict("records") if len(per_label_df) else [],
    }


def _fano(rows: pd.DataFrame, freq: str) -> float:
    """Variance-to-mean of counts per bucket; 1.0 is Poisson."""
    counts = rows.set_index("time").resample(freq).size()
    counts = counts[counts.index >= rows["time"].min()]
    return float(counts.var(ddof=1) / counts.mean()) if counts.mean() else float("nan")


# --------------------------------------------------------------------------
# the crackdown
# --------------------------------------------------------------------------


def structural_break(
    features: pd.DataFrame,
    boundary: pd.Timestamp = CRACKDOWN,
    window: str = graphs.DEFAULT_HANDOFF_WINDOW,
) -> dict[str, Any]:
    """Handoff-network structure before and after the first mass deletion."""
    out: dict[str, Any] = {"boundary": str(boundary)}
    pairs = graphs.handoff_pairs(features, window)
    for era, subset in (
        ("before", pairs[pairs["time"] < boundary]),
        ("after", pairs[pairs["time"] >= boundary]),
    ):
        g = graphs._handoff_from_pairs(subset.reset_index(drop=True), window)
        if g.number_of_nodes() == 0:
            out[era] = {"nodes": 0}
            continue
        undirected = nx.Graph(g)
        comps = sorted(nx.connected_components(undirected), key=len, reverse=True)
        membership = detect_communities(undirected)
        groups: dict[int, set] = {}
        for node, comm in membership.items():
            groups.setdefault(comm, set()).add(node)
        out[era] = {
            "events": int(len(subset)),
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "density": nx.density(g),
            "reciprocity": float(nx.reciprocity(g)) if g.number_of_edges() else 0.0,
            "transitivity": float(nx.transitivity(undirected)),
            "giant_fraction": len(comps[0]) / g.number_of_nodes() if comps else 0.0,
            "max_core": max(nx.core_number(undirected).values()) if g.number_of_edges() else 0,
            "modularity": float(nx.community.modularity(undirected, list(groups.values()), weight="weight")),
            "mean_out_degree": float(np.mean([d for _, d in g.out_degree()])),
        }

    # Did agents move to evasion-named pages after the sweep began?
    rows = graphs.agent_rows(features)
    for era, subset in (("before", rows[rows["time"] < boundary]), ("after", rows[rows["time"] >= boundary])):
        if era in out and isinstance(out[era], dict):
            out[era]["zzz_page_share"] = float(subset["name_zzz"].mean()) if len(subset) else 0.0
            out[era]["persist_motif_share"] = float(
                subset[["name_zzz", "name_fresh", "name_unique"]].any(axis=1).mean()
            ) if len(subset) else 0.0
    return out


def deletion_response(features: pd.DataFrame) -> dict[str, Any]:
    """How fast agents rewrote pages an administrator had just deleted.

    A short, tight latency distribution is evidence of monitoring: something was
    watching the wiki for removals rather than rediscovering them by chance.
    """
    events = io.load_events()
    deletes = events[events["event_type"] == "delete"][["page_key", "time"]].dropna()
    saves = graphs.agent_rows(features)[["page_key", "time", "label"]]

    merged = deletes.merge(saves, on="page_key", suffixes=("_delete", "_save"))
    after = merged[merged["time_save"] > merged["time_delete"]].copy()
    if after.empty:
        return {"recreations": 0}
    after["latency_seconds"] = (after["time_save"] - after["time_delete"]).dt.total_seconds()
    first = after.sort_values("latency_seconds").groupby(["page_key", "time_delete"], as_index=False).first()

    lat = first["latency_seconds"]
    return {
        "deleted_pages": int(deletes["page_key"].nunique()),
        "deletions_followed_by_a_rewrite": int(len(first)),
        "median_latency_seconds": float(lat.median()),
        "p10_latency_seconds": float(lat.quantile(0.10)),
        "p90_latency_seconds": float(lat.quantile(0.90)),
        "within_1h": int((lat <= 3600).sum()),
        "within_10min": int((lat <= 600).sum()),
        "distinct_responders": int(first["label"].nunique()),
        "top_responders": first["label"].value_counts().head(15).to_dict(),
        "corpus_first_recreation_relations": int(
            (events.get("relation_type") == "first_recreation_of").sum()
        ),
    }
