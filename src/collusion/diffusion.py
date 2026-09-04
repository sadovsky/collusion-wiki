"""Did techniques spread through the network, or did agents rediscover them?

This is the sharpest test the corpus supports. Independent rediscovery and
social transmission both produce S-curves, so an S-curve alone proves nothing.
What separates them is whether an agent's probability of adopting at time t
depends on how many of its *network neighbours* had already adopted -- measured
against the same statistic on a degree-preserving shuffle of the network.
"""

from __future__ import annotations

from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import graphs
from .metrics import SEED, compare_to_null


# --------------------------------------------------------------------------
# adoption events
# --------------------------------------------------------------------------


def adoption_events(features: pd.DataFrame, column: str = "techniques") -> pd.DataFrame:
    """First time each label used each item (technique class, host, or token)."""
    rows = graphs.agent_rows(features)
    records = []
    for rec in rows.itertuples(index=False):
        for item in getattr(rec, column):
            records.append({"label": rec.label, "item": item, "time": rec.time})
    if not records:
        return pd.DataFrame(columns=["label", "item", "time"])
    df = pd.DataFrame(records)
    return (
        df.sort_values("time", kind="stable")
        .groupby(["item", "label"], as_index=False)
        .first()
        .rename(columns={"time": "first_use"})
    )


def adoption_curves(adoptions: pd.DataFrame, min_adopters: int = 20) -> pd.DataFrame:
    """Cumulative adopter counts per item, on a common hourly grid."""
    counts = adoptions["item"].value_counts()
    keep = counts[counts >= min_adopters].index
    subset = adoptions[adoptions["item"].isin(keep)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["item", "time", "adopters"])

    subset["bucket"] = subset["first_use"].dt.floor("1h")
    grid = pd.date_range(subset["bucket"].min(), subset["bucket"].max(), freq="1h", tz="UTC")
    frames = []
    for item, grp in subset.groupby("item", sort=False):
        series = grp.groupby("bucket").size().reindex(grid, fill_value=0).cumsum()
        frames.append(pd.DataFrame({"item": item, "time": grid, "adopters": series.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def fit_logistic(times: Sequence[float], adopters: Sequence[float]) -> dict[str, Any]:
    """Logistic fit K / (1 + exp(-r (t - t0))) on hours since first adoption."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(adopters, dtype=float)
    if len(t) < 8 or y.max() <= 1:
        return {"fit": "insufficient_data"}

    def model(x, k, r, t0):
        return k / (1.0 + np.exp(-r * (x - t0)))

    try:
        popt, _ = optimize.curve_fit(
            model,
            t,
            y,
            p0=[y.max(), 0.05, float(np.median(t))],
            bounds=([y.max() * 0.5, 1e-5, t.min()], [y.max() * 5, 5.0, t.max()]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return {"fit": "did_not_converge"}

    pred = model(t, *popt)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "fit": "logistic",
        "carrying_capacity": float(popt[0]),
        "growth_rate_per_hour": float(popt[1]),
        "midpoint_hours": float(popt[2]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "doubling_time_hours": float(np.log(2) / popt[1]) if popt[1] > 0 else None,
    }


def curve_summary(curves: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for item, grp in curves.groupby("item", sort=False):
        grp = grp.sort_values("time")
        active = grp[grp["adopters"] > 0]
        if active.empty:
            continue
        hours = (active["time"] - active["time"].iloc[0]).dt.total_seconds() / 3600
        summary = {
            "item": item,
            "final_adopters": int(active["adopters"].iloc[-1]),
            "first_adoption": str(active["time"].iloc[0]),
            "hours_to_half": float(
                hours[active["adopters"] >= active["adopters"].iloc[-1] / 2].iloc[0]
            )
            if (active["adopters"] >= active["adopters"].iloc[-1] / 2).any()
            else None,
        }
        summary.update(fit_logistic(hours.to_numpy(), active["adopters"].to_numpy()))
        out.append(summary)
    return sorted(out, key=lambda d: -d["final_adopters"])


# --------------------------------------------------------------------------
# network-conditioned adoption
# --------------------------------------------------------------------------


def exposure_test(
    features: pd.DataFrame,
    item: str,
    network: nx.Graph,
    column: str = "techniques",
    n_null: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """Were adopters disproportionately neighbours of earlier adopters?

    Statistic: the mean number of already-adopted neighbours an agent had at the
    moment it adopted.

    Two nulls, because they answer different questions and only the second is
    real evidence of transmission:

    `random_adopters` draws an adopter set of the same size at random from the
    network. Beating it shows adopters are clustered -- but a technique used
    only by the dense coordination core beats it automatically, whether or not
    anything spread. This null is reported for completeness, not as evidence.

    `shuffled_order` keeps the actual adopter set and permutes only *which of
    them adopted when*. It holds the network, the adopter set, and the adoption
    times all fixed, so it isolates the one thing at issue: whether agents
    adopted in an order that follows network proximity. This is the test.
    """
    adoptions = adoption_events(features, column)
    adopters = adoptions[adoptions["item"] == item]
    undirected = nx.Graph(network)
    inside = adopters[adopters["label"].isin(undirected.nodes)]
    if len(inside) < 10:
        return {"item": item, "status": "insufficient_adopters_in_network", "n": int(len(inside))}

    order = inside.sort_values("first_use")
    labels = order["label"].tolist()

    def mean_prior_exposure(assignment: Sequence[str]) -> float:
        adopted: set[str] = set()
        total = 0
        for node in assignment:
            total += sum(1 for nb in undirected.neighbors(node) if nb in adopted)
            adopted.add(node)
        return total / len(assignment)

    observed = mean_prior_exposure(labels)
    rng = np.random.default_rng(seed)

    candidates = np.array(list(undirected.nodes))
    random_set_null = [
        mean_prior_exposure(rng.choice(candidates, size=len(labels), replace=False))
        for _ in range(n_null)
    ]

    shuffled_order_null = []
    for _ in range(n_null):
        permuted = labels[:]
        rng.shuffle(permuted)
        shuffled_order_null.append(mean_prior_exposure(permuted))

    strict = compare_to_null("mean_prior_adopted_neighbours", observed, shuffled_order_null).as_dict()
    loose = compare_to_null("mean_prior_adopted_neighbours", observed, random_set_null).as_dict()

    return {
        "item": item,
        "observed": observed,
        "n_adopters_in_network": int(len(inside)),
        "network_nodes": undirected.number_of_nodes(),
        "shuffled_order": strict,
        "random_adopters": loose,
        "verdict": (
            "adoption order follows network proximity"
            if strict["z_score"] > 2 and strict["p_value"] < 0.05
            else "adoption order not distinguishable from chance among these adopters"
        ),
    }


def technique_diffusion(
    features: pd.DataFrame,
    network: nx.Graph,
    column: str = "techniques",
    min_adopters: int = 20,
    n_null: int = 200,
) -> dict[str, Any]:
    adoptions = adoption_events(features, column)
    curves = adoption_curves(adoptions, min_adopters=min_adopters)
    summaries = curve_summary(curves)
    tests = [
        exposure_test(features, s["item"], network, column=column, n_null=n_null)
        for s in summaries
    ]
    return {
        "column": column,
        "n_items": int(adoptions["item"].nunique()),
        "curves": summaries,
        "exposure_tests": tests,
    }


# --------------------------------------------------------------------------
# vocabulary and naming conventions
# --------------------------------------------------------------------------


def vocabulary_diffusion(features: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Adoption of edit-summary vocabulary across distinct agent handles.

    Agents wrote their own edit summaries. Words like "coordination" spreading
    across hundreds of independent handles is cultural transmission with no
    channel other than the wiki itself.
    """
    rows = graphs.agent_rows(features)
    records = []
    for rec in rows.itertuples(index=False):
        for token in set(rec.summary_tokens):
            records.append({"label": rec.label, "token": token, "time": rec.time})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    first = df.sort_values("time").groupby(["token", "label"], as_index=False).first()
    counts = first.groupby("token").agg(
        n_labels=("label", "nunique"),
        first_use=("time", "min"),
        last_use=("time", "max"),
    )
    counts["span_hours"] = (counts["last_use"] - counts["first_use"]).dt.total_seconds() / 3600
    counts["labels_per_day"] = counts["n_labels"] / (counts["span_hours"] / 24).clip(lower=1e-6)
    return counts.sort_values("n_labels", ascending=False).head(top_n).reset_index()


def naming_convention_diffusion(features: pd.DataFrame) -> pd.DataFrame:
    """Spread of page-naming motifs across handles over time."""
    rows = graphs.agent_rows(features)
    motif_cols = [c for c in rows.columns if c.startswith("name_")]
    records = []
    for col in motif_cols:
        hits = rows[rows[col]]
        if hits.empty:
            continue
        first = hits.sort_values("time").groupby("label", as_index=False)["time"].first()
        records.append(
            {
                "motif": col.removeprefix("name_"),
                "n_labels": int(first["label"].nunique()),
                "n_pages": int(hits["page_key"].nunique()),
                "n_revisions": int(len(hits)),
                "first_use": str(hits["time"].min()),
                "median_adoption": str(first["time"].median()),
                "last_use": str(hits["time"].max()),
            }
        )
    return pd.DataFrame(records).sort_values("n_labels", ascending=False).reset_index(drop=True)


def convention_timing_shift(
    features: pd.DataFrame,
    motif: str = "zzz",
    boundary: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Test whether an evasion motif concentrates after the deletion sweep began.

    Mann-Whitney on adoption times against every other motif's adoption times:
    if `zzz` is a *response* to deletion rather than background naming style, its
    adopters arrive later than the corpus baseline.
    """
    from .temporal import CRACKDOWN

    boundary = boundary or CRACKDOWN
    rows = graphs.agent_rows(features)
    col = f"name_{motif}"
    if col not in rows.columns:
        return {"motif": motif, "status": "unknown_motif"}

    hit = rows[rows[col]]["time"]
    other = rows[~rows[col]]["time"]
    if len(hit) < 10:
        return {"motif": motif, "status": "insufficient_data", "n": int(len(hit))}

    from .temporal import _epoch_seconds

    u, p = stats.mannwhitneyu(
        _epoch_seconds(hit), _epoch_seconds(other), alternative="greater"
    )
    return {
        "motif": motif,
        "n_revisions": int(len(hit)),
        "share_after_boundary": float((hit >= boundary).mean()),
        "baseline_share_after_boundary": float((other >= boundary).mean()),
        "median_time": str(hit.median()),
        "baseline_median_time": str(other.median()),
        "mannwhitney_u": float(u),
        "p_value": float(p),
        "alternative": "motif revisions occur later than the rest of the corpus",
    }
