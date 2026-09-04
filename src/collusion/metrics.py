"""Static structural metrics, each reported against a null model.

The discipline here is that a bare number about a network is close to
meaningless. 749 reciprocal pairs sounds like coordination until you know that a
degree-preserving rewiring of the same graph produces 700. So every claim that
matters -- reciprocity, assortativity, clustering, community quality -- ships
with a null distribution and a z-score.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

SEED = 20260618  # the day of the 6,543-save spike


@dataclass
class NullComparison:
    """An observed statistic against a distribution of nulls."""

    statistic: str
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_to_null(
    statistic: str,
    observed: float,
    null_values: Sequence[float],
) -> NullComparison:
    arr = np.asarray(list(null_values), dtype=float)
    mean, std = float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    z = (observed - mean) / std if std > 0 else 0.0
    # Two-sided empirical p with the standard +1 correction.
    more_extreme = int(np.sum(np.abs(arr - mean) >= abs(observed - mean)))
    p = (more_extreme + 1) / (len(arr) + 1)
    return NullComparison(statistic, float(observed), mean, std, float(z), float(p), len(arr))


# Rewiring cost grows with edge count; past this many swaps the null is already
# well mixed and further swaps only buy wall-clock.
MAX_SWAPS = 40_000


def rewired_ensemble(graph: nx.Graph, n_samples: int, seed: int = SEED) -> list[nx.Graph]:
    """Degree-preserving double-edge swaps: the right null for degree-driven effects.

    Generated once and shared across every statistic that needs it. Rewiring
    separately per statistic would be three times the cost for a *worse* null,
    since each statistic would then be compared against a different ensemble.
    """
    rng = random.Random(seed)
    m = graph.number_of_edges()
    if m < 2:
        return [graph.copy() for _ in range(n_samples)]

    nswap = min(m, MAX_SWAPS)
    out = []
    for _ in range(n_samples):
        h = graph.copy()
        try:
            if h.is_directed():
                nx.directed_edge_swap(h, nswap=nswap, max_tries=nswap * 20, seed=rng.randrange(1 << 30))
            else:
                nx.double_edge_swap(h, nswap=nswap, max_tries=nswap * 20, seed=rng.randrange(1 << 30))
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            pass  # not every degree sequence can be fully rewired; a partial swap is still a null
        out.append(h)
    return out


def _rewire_null(
    graph: nx.Graph,
    statistic: Callable[[nx.Graph], float],
    n_samples: int,
    seed: int = SEED,
) -> list[float]:
    return [statistic(h) for h in rewired_ensemble(graph, n_samples, seed)]


# --------------------------------------------------------------------------
# degree distributions
# --------------------------------------------------------------------------


def fit_power_law(degrees: Sequence[int], xmin: int = 1) -> dict[str, Any]:
    """MLE tail exponent with a KS check and a lognormal likelihood-ratio test.

    Reported deliberately in full: a heavy tail is not a power law, and the
    honest answer for most social graphs is that lognormal fits at least as
    well. The likelihood ratio says which.
    """
    x = np.asarray([d for d in degrees if d >= xmin], dtype=float)
    if len(x) < 10:
        return {"n": int(len(x)), "fit": "insufficient_data"}

    alpha = 1.0 + len(x) / np.sum(np.log(x / (xmin - 0.5)))

    # KS distance against the discrete power-law CDF.
    xs = np.sort(x)
    empirical = np.arange(1, len(xs) + 1) / len(xs)
    theoretical = 1.0 - (xs / (xmin - 0.5)) ** (1.0 - alpha)
    ks = float(np.max(np.abs(empirical - theoretical)))

    logx = np.log(x)
    mu, sigma = float(logx.mean()), float(logx.std(ddof=1))
    ll_pl = float(np.sum(np.log((alpha - 1) / (xmin - 0.5)) - alpha * np.log(x / (xmin - 0.5))))
    ll_ln = float(np.sum(stats.lognorm.logpdf(x, s=sigma, scale=math.exp(mu))))
    # Vuong-style normalized likelihood ratio.
    diffs = np.log(
        np.clip(((alpha - 1) / (xmin - 0.5)) * (x / (xmin - 0.5)) ** (-alpha), 1e-300, None)
    ) - stats.lognorm.logpdf(x, s=sigma, scale=math.exp(mu))
    lr = float(np.sum(diffs))
    lr_std = float(np.std(diffs, ddof=1) * math.sqrt(len(diffs)))
    lr_p = float(2 * (1 - stats.norm.cdf(abs(lr) / lr_std))) if lr_std > 0 else 1.0

    return {
        "n": int(len(x)),
        "xmin": xmin,
        "alpha": float(alpha),
        "ks_distance": ks,
        "loglik_power_law": ll_pl,
        "loglik_lognormal": ll_ln,
        "likelihood_ratio": lr,
        "likelihood_ratio_p": lr_p,
        "preferred": (
            "power_law" if lr > 0 and lr_p < 0.05 else "lognormal" if lr < 0 and lr_p < 0.05 else "indistinguishable"
        ),
        "mean": float(x.mean()),
        "max": int(x.max()),
    }


def degree_profile(graph: nx.Graph) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if graph.is_directed():
        out["in"] = fit_power_law([d for _, d in graph.in_degree()])
        out["out"] = fit_power_law([d for _, d in graph.out_degree()])
        out["total"] = fit_power_law([d for _, d in graph.degree()])
    else:
        out["total"] = fit_power_law([d for _, d in graph.degree()])
    return out


# --------------------------------------------------------------------------
# cohesion
# --------------------------------------------------------------------------


def component_profile(graph: nx.Graph) -> dict[str, Any]:
    n = graph.number_of_nodes()
    if n == 0:
        return {"nodes": 0}
    if graph.is_directed():
        weak = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
        strong = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
        return {
            "nodes": n,
            "weak_components": len(weak),
            "giant_weak": len(weak[0]),
            "giant_weak_fraction": len(weak[0]) / n,
            "strong_components": len(strong),
            "giant_strong": len(strong[0]),
            "giant_strong_fraction": len(strong[0]) / n,
        }
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    return {
        "nodes": n,
        "components": len(comps),
        "giant": len(comps[0]),
        "giant_fraction": len(comps[0]) / n,
    }


def kcore_profile(graph: nx.Graph) -> dict[str, Any]:
    simple = nx.Graph(graph) if graph.is_directed() else graph.copy()
    simple.remove_edges_from(nx.selfloop_edges(simple))
    core = nx.core_number(simple)
    counts = Counter(core.values())
    max_k = max(core.values()) if core else 0
    inner = [n for n, k in core.items() if k == max_k]
    return {
        "max_core": max_k,
        "core_size_by_k": {str(k): counts[k] for k in sorted(counts)},
        "innermost_core_size": len(inner),
        "innermost_core_members": sorted(inner)[:50],
        "core_number": core,
    }


def _safe(fn: Callable[[nx.Graph], float], graph: nx.Graph) -> float:
    try:
        value = fn(graph)
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


# Triangle counting costs O(sum of squared degree). On a dense projection that
# is minutes per sample, and the answer is not worth it: see `null_budget`.
DENSE_EDGES = 40_000


def null_budget(graph: nx.Graph, requested: int) -> tuple[int, str | None]:
    """How many null samples this graph is worth, and why.

    Two cases get cut down. Dense graphs make triangle counting quadratic in
    degree, so a full ensemble costs hours. More importantly, a *projection* of
    a bipartite graph is mechanically saturated with triangles -- every page's
    editors form a clique by construction -- so its clustering coefficient
    measures the projection, not the agents. Spending an hour to put an error
    bar on an artifact is the wrong trade.
    """
    layer = str(graph.graph.get("layer", ""))
    if "projection" in layer or "bipartite" in layer:
        return min(requested, 10), (
            "reduced: clustering in a bipartite graph or its projection is "
            "structurally determined, so the rewired null is not informative"
        )
    if graph.number_of_edges() > DENSE_EDGES:
        return min(requested, 25), "reduced: dense graph, triangle counting is the bottleneck"
    return requested, None


def cohesion_metrics(graph: nx.Graph, n_null: int = 200) -> dict[str, Any]:
    """Clustering, reciprocity and assortativity, all against one shared null."""
    out: dict[str, Any] = {}
    simple = nx.Graph(graph) if graph.is_directed() else graph

    n_null, budget_note = null_budget(simple, n_null)
    out["null_budget_note"] = budget_note
    undirected_null = rewired_ensemble(simple, n_null)
    trans_null = [_safe(nx.transitivity, h) for h in undirected_null]
    assort_null = [_safe(nx.degree_assortativity_coefficient, h) for h in undirected_null]

    out["transitivity"] = compare_to_null(
        "transitivity", nx.transitivity(simple), [v for v in trans_null if np.isfinite(v)]
    ).as_dict()
    out["average_clustering"] = float(nx.average_clustering(simple))

    observed_assort = _safe(nx.degree_assortativity_coefficient, simple)
    clean_assort = [v for v in assort_null if np.isfinite(v)]
    out["degree_assortativity"] = (
        compare_to_null("degree_assortativity", observed_assort, clean_assort).as_dict()
        if np.isfinite(observed_assort) and clean_assort
        else None
    )

    if graph.is_directed():
        # Reciprocity needs a directed null, so it gets its own ensemble.
        directed_null = rewired_ensemble(graph, n_null, seed=SEED + 1)
        out["reciprocity"] = compare_to_null(
            "reciprocity",
            nx.reciprocity(graph),
            [_safe(nx.reciprocity, h) for h in directed_null],
        ).as_dict()
        # The full census is O(n^3) in the worst case; only worth it when small.
        if graph.number_of_nodes() <= 4000:
            out["triadic_census"] = {k: int(v) for k, v in nx.triadic_census(graph).items()}

    out["null_samples"] = n_null
    out["null_swaps_per_sample"] = min(simple.number_of_edges(), MAX_SWAPS)
    return out


def attribute_assortativity(graph: nx.Graph, attribute: str) -> float | None:
    """Do nodes attach to others with the same attribute value (wiki, family)?"""
    values = {n: d.get(attribute) for n, d in graph.nodes(data=True)}
    if len({v for v in values.values() if v is not None}) < 2:
        return None
    try:
        return float(nx.attribute_assortativity_coefficient(graph, attribute))
    except (nx.NetworkXError, ValueError, ZeroDivisionError):
        return None


# --------------------------------------------------------------------------
# centrality
# --------------------------------------------------------------------------


def centrality_table(graph: nx.Graph, betweenness_k: int | None = 500) -> pd.DataFrame:
    """Per-node centralities.

    Betweenness is the expensive one and the interesting one: a node with high
    betweenness and modest degree is a broker, not a hub. Sampled with a fixed
    seed above `betweenness_k` nodes; `betweenness_exact` records which.
    """
    n = graph.number_of_nodes()
    exact = betweenness_k is None or n <= betweenness_k
    bc = nx.betweenness_centrality(
        graph, k=None if exact else betweenness_k, seed=SEED, normalized=True
    )

    try:
        pr = nx.pagerank(graph, weight="weight")
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(graph, weight="weight", max_iter=500, tol=1e-4)

    # Eigenvector centrality is only defined within a connected component --
    # networkx refuses outright on a disconnected graph. These graphs all have a
    # periphery of isolated pairs, so it is computed on the giant component and
    # left undefined elsewhere rather than being quietly faked as zero.
    simple = nx.Graph(graph) if graph.is_directed() else graph
    ev: dict[str, float] = dict.fromkeys(graph, float("nan"))
    if simple.number_of_nodes():
        giant = simple.subgraph(max(nx.connected_components(simple), key=len))
        try:
            ev.update(nx.eigenvector_centrality_numpy(giant, weight="weight"))
        except (nx.NetworkXException, ValueError):
            pass

    simple_for_core = nx.Graph(graph)
    simple_for_core.remove_edges_from(nx.selfloop_edges(simple_for_core))
    core = nx.core_number(simple_for_core)
    rows = []
    for node in graph.nodes():
        row = {
            "node": node,
            "kind": graph.nodes[node].get("kind"),
            "degree": graph.degree(node),
            "weighted_degree": graph.degree(node, weight="weight"),
            "betweenness": bc.get(node, 0.0),
            "betweenness_exact": exact,
            "pagerank": pr.get(node, 0.0),
            "eigenvector": ev.get(node, float("nan")),
            "core_number": core.get(node, 0),
        }
        if graph.is_directed():
            row["in_degree"] = graph.in_degree(node)
            row["out_degree"] = graph.out_degree(node)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Brokerage: betweenness far above what this node's degree would predict.
    if len(df) > 2 and df["degree"].std() > 0:
        deg_rank = df["degree"].rank(pct=True)
        btw_rank = df["betweenness"].rank(pct=True)
        df["brokerage"] = btw_rank - deg_rank
    return df.sort_values("betweenness", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# communities
# --------------------------------------------------------------------------


def detect_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    seed: int = SEED,
) -> dict[str, int]:
    """Leiden where available, Louvain otherwise. Returns node -> community id."""
    simple = nx.Graph(graph) if graph.is_directed() else graph
    if simple.number_of_edges() == 0:
        return {n: i for i, n in enumerate(simple.nodes())}
    try:
        import igraph as ig
        import leidenalg

        nodes = list(simple.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        edges = [(index[u], index[v]) for u, v in simple.edges()]
        weights = [float(d.get("weight", 1.0)) for _, _, d in simple.edges(data=True)]
        g = ig.Graph(n=len(nodes), edges=edges)
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=10,
        )
        return {nodes[i]: c for i, c in enumerate(part.membership)}
    except ImportError:
        communities = nx.community.louvain_communities(
            simple, weight="weight", resolution=resolution, seed=seed
        )
        return {n: i for i, comm in enumerate(communities) for n in comm}


def community_profile(
    graph: nx.Graph,
    resolutions: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    n_null: int = 50,
) -> dict[str, Any]:
    simple = nx.Graph(graph) if graph.is_directed() else graph
    sweep = []
    for res in resolutions:
        membership = detect_communities(simple, resolution=res)
        groups: dict[int, set] = {}
        for node, comm in membership.items():
            groups.setdefault(comm, set()).add(node)
        q = nx.community.modularity(simple, list(groups.values()), weight="weight")
        sizes = sorted((len(s) for s in groups.values()), reverse=True)
        sweep.append(
            {
                "resolution": res,
                "n_communities": len(groups),
                "modularity": float(q),
                "largest": sizes[0] if sizes else 0,
                "singletons": sum(1 for s in sizes if s == 1),
            }
        )

    def null_q(h: nx.Graph) -> float:
        membership = detect_communities(h, resolution=1.0)
        groups: dict[int, set] = {}
        for node, comm in membership.items():
            groups.setdefault(comm, set()).add(node)
        return nx.community.modularity(h, list(groups.values()), weight="weight")

    # Every graph has *some* modularity -- Leiden will always find a partition,
    # even in noise. The rewired comparison is what separates real community
    # structure from the modularity a random graph of this degree sequence has
    # anyway, so it is not optional here.
    observed_q = next(s["modularity"] for s in sweep if s["resolution"] == 1.0)
    samples, note = null_budget(simple, n_null)
    return {
        "sweep": sweep,
        "null_budget_note": note,
        "modularity_vs_rewired": compare_to_null(
            "modularity", observed_q, [null_q(h) for h in rewired_ensemble(simple, samples, seed=SEED + 2)]
        ).as_dict(),
    }


def compare_partitions(
    membership: dict[str, int],
    ground_truth: dict[str, str],
    n_permutations: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """ARI / NMI of detected communities against a corpus-supplied labelling.

    The corpus ships its own `page_family` taxonomy, produced independently of
    any network. If interaction structure recovers it, task specialisation was
    emergent from who-worked-with-whom rather than imposed by us.
    """
    shared = [n for n in membership if n in ground_truth and ground_truth[n] is not None]
    if len(shared) < 10:
        return {"n": len(shared), "status": "insufficient_overlap"}

    detected = [membership[n] for n in shared]
    truth_values = [ground_truth[n] for n in shared]
    codes = {v: i for i, v in enumerate(sorted(set(truth_values)))}
    truth = [codes[v] for v in truth_values]

    ari = float(adjusted_rand_score(truth, detected))
    nmi = float(normalized_mutual_info_score(truth, detected))

    rng = random.Random(seed)
    null_ari, null_nmi = [], []
    for _ in range(n_permutations):
        shuffled = detected[:]
        rng.shuffle(shuffled)
        null_ari.append(adjusted_rand_score(truth, shuffled))
        null_nmi.append(normalized_mutual_info_score(truth, shuffled))

    return {
        "n": len(shared),
        "n_true_classes": len(codes),
        "n_detected_communities": len(set(detected)),
        "ari": compare_to_null("ari", ari, null_ari).as_dict(),
        "nmi": compare_to_null("nmi", nmi, null_nmi).as_dict(),
    }


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------


def analyze(graph: nx.Graph, name: str, n_null: int = 200, betweenness_k: int | None = 500) -> dict[str, Any]:
    from .graphs import describe

    return {
        "name": name,
        "summary": describe(graph),
        "degrees": degree_profile(graph),
        "components": component_profile(graph),
        "kcore": {k: v for k, v in kcore_profile(graph).items() if k != "core_number"},
        "cohesion": cohesion_metrics(graph, n_null=n_null),
        # Each modularity null runs a full Leiden pass, so it gets a smaller
        # ensemble than the cheap cohesion statistics.
        "communities": community_profile(graph, n_null=max(10, n_null // 8)),
    }
