"""Static figures.

Deliberately plain: one idea per panel, no chartjunk, and every axis labelled
with the population it is drawn from. Vector output, because several of these
are dense enough that a reader will want to zoom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import diffusion, graphs, io, metrics, temporal
from .pipeline import label_family_ground_truth

INK = "#1c1c1f"
MUTED = "#6b6b73"
GRID = "#e3e3e8"
ACCENT = "#c2410c"  # saves / agent activity
COUNTER = "#1d4ed8"  # deletions / administrator activity
SUPPORT = "#0f766e"
SERIES = ("#c2410c", "#1d4ed8", "#0f766e", "#7c3aed", "#b45309", "#be123c", "#4d7c0f", "#0369a1")

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    path = io.figures_dir() / name
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _annotate_incidents(ax: plt.Axes, ymax: float, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for date, text in temporal.INCIDENTS:
        if date in skip:
            continue
        x = pd.Timestamp(date, tz="UTC")
        ax.axvline(x, color=MUTED, lw=0.6, ls=(0, (2, 3)), zorder=0)
        ax.annotate(
            text,
            xy=(x, ymax),
            xytext=(2, -2),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="left",
            fontsize=6.5,
            color=MUTED,
        )


# --------------------------------------------------------------------------


def figure_timeline(feat: pd.DataFrame) -> Path:
    daily = temporal.activity_series(feat, "1D")
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, height_ratios=[2, 1], constrained_layout=True
    )

    ax.bar(daily["time"], daily["saves"], width=0.8, color=ACCENT, label="agent saves")
    ax.bar(
        daily["time"],
        -daily["deletes"],
        width=0.8,
        color=COUNTER,
        label="administrator deletions",
    )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("events per day")
    ax.set_title("Agent writes and administrator deletions")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", lw=0.5)
    _annotate_incidents(ax, daily["saves"].max())

    ax2.plot(daily["time"], daily["active_labels"], color=SUPPORT, lw=1.4, label="distinct handles")
    ax2.plot(daily["time"], daily["active_ip16"], color=MUTED, lw=1.0, ls="--", label="distinct /16 prefixes")
    ax2.set_ylabel("active per day")
    ax2.set_title("Active population")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", lw=0.5)

    peak = daily.loc[daily["saves"].idxmax()]
    ax.annotate(
        f"{int(peak['saves']):,} saves",
        xy=(peak["time"], peak["saves"]),
        xytext=(-46, -14),
        textcoords="offset points",
        fontsize=8,
        color=ACCENT,
        fontweight="bold",
    )
    return _save(fig, "01_timeline.svg")


def figure_percolation(feat: pd.DataFrame) -> Path:
    curve = temporal.percolation_curve(feat)
    evolution = temporal.community_evolution(feat)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    ax.plot(curve["time"], curve["giant_fraction"], color=ACCENT, lw=1.6, label="giant component share")
    ax.set_ylabel("share of agents in\nlargest component")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", lw=0.5)
    ax.set_title("Percolation: when independent agents became one connected system")

    twin = ax.twinx()
    twin.plot(curve["time"], curve["mean_degree"], color=MUTED, lw=1.0, ls="--")
    twin.axhline(1.0, color=COUNTER, lw=0.8, ls=":")
    twin.set_ylabel("mean degree (dashed)", color=MUTED)
    twin.spines["top"].set_visible(False)
    twin.annotate(
        "mean degree 1.0\n(random-graph threshold)",
        xy=(curve["time"].iloc[len(curve) // 6], 1.0),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=6.5,
        color=COUNTER,
    )
    _annotate_incidents(ax, 1.0)

    if not evolution.empty:
        ax2.plot(evolution["time"], evolution["modularity"], color=SUPPORT, lw=1.6, label="modularity Q")
        ax2.set_ylabel("modularity Q")
        ax2.grid(axis="y", lw=0.5)
        twin2 = ax2.twinx()
        twin2.plot(evolution["time"], evolution["n_communities"], color=MUTED, lw=1.0, ls="--")
        twin2.set_ylabel("communities (dashed)", color=MUTED)
        twin2.spines["top"].set_visible(False)
        ax2.set_title("Community structure of the cumulative handoff network")
    return _save(fig, "02_percolation.svg")


def figure_degree_distributions(built: dict[str, nx.Graph]) -> Path:
    panels = [
        ("G1_handoff", "out", "Handoff out-degree"),
        ("G1_handoff", "in", "Handoff in-degree"),
        ("G2p_coedit_projection", "total", "Co-edit degree"),
        ("G3_hyperlink", "in", "Page in-links"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    for ax, (name, direction, title) in zip(axes, panels):
        g = built.get(name)
        if g is None or g.number_of_edges() == 0:
            ax.set_visible(False)
            continue
        if direction == "in" and g.is_directed():
            degrees = [d for _, d in g.in_degree()]
        elif direction == "out" and g.is_directed():
            degrees = [d for _, d in g.out_degree()]
        else:
            degrees = [d for _, d in g.degree()]
        degrees = np.array([d for d in degrees if d > 0])
        if len(degrees) < 10:
            ax.set_visible(False)
            continue

        values = np.sort(degrees)
        ccdf = 1.0 - np.arange(len(values)) / len(values)
        ax.loglog(values, ccdf, ".", ms=3, color=ACCENT, alpha=0.7)

        fit = metrics.fit_power_law(degrees)
        if fit.get("alpha"):
            ax.set_title(f"{title}\n" + rf"$\alpha$={fit['alpha']:.2f}, best fit: {fit['preferred']}")
        else:
            ax.set_title(title)
        ax.set_xlabel("degree k")
        ax.set_ylabel("P(K ≥ k)")
        ax.grid(lw=0.4, which="both")
    fig.suptitle(
        "Degree distributions, with power-law vs lognormal model selection",
        x=0.005,
        ha="left",
        fontweight="bold",
    )
    return _save(fig, "03_degree_distributions.svg")


def figure_community_vs_family(feat: pd.DataFrame, built: dict[str, nx.Graph]) -> Path:
    g = nx.Graph(built["G2p_coedit_projection"])
    truth = label_family_ground_truth(feat)
    membership = metrics.detect_communities(g)

    shared = [n for n in membership if n in truth]
    if len(shared) < 20:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "insufficient overlap", ha="center")
        return _save(fig, "04_community_vs_family.svg")

    df = pd.DataFrame(
        {"community": [membership[n] for n in shared], "family": [truth[n] for n in shared]}
    )
    top_comms = df["community"].value_counts().head(14).index
    top_fams = df["family"].value_counts().head(14).index
    table = (
        df[df["community"].isin(top_comms) & df["family"].isin(top_fams)]
        .pivot_table(index="family", columns="community", aggfunc="size", fill_value=0)
    )
    # Order rows/cols so the diagonal, if there is one, is visible.
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table = table[table.idxmax(axis=0).sort_values().index]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    im = ax.imshow(np.log1p(table.to_numpy()), cmap="rocket_r" if False else "YlOrBr", aspect="auto")
    ax.set_xticks(range(len(table.columns)), [f"c{c}" for c in table.columns], fontsize=7)
    ax.set_yticks(range(len(table.index)), table.index, fontsize=7)
    ax.set_xlabel("detected community (co-edit network, Leiden)")
    ax.set_ylabel("page_family (corpus taxonomy, content-derived)")

    validation = metrics.compare_partitions(membership, truth)
    ari, nmi = validation["ari"]["observed"], validation["nmi"]["observed"]
    ax.set_title(
        f"Do interaction communities recover the corpus's task taxonomy?\n"
        f"ARI = {ari:.3f}, NMI = {nmi:.3f} (permutation null p < "
        f"{max(validation['ari']['p_value'], 1e-3):.3f})"
    )
    fig.colorbar(im, ax=ax, label="log(1 + agents)", shrink=0.7)
    return _save(fig, "04_community_vs_family.svg")


def figure_adoption(feat: pd.DataFrame) -> Path:
    adoptions = diffusion.adoption_events(feat, "techniques")
    curves = diffusion.adoption_curves(adoptions, min_adopters=20)
    if curves.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no curves", ha="center")
        return _save(fig, "05_adoption.svg")

    order = curves.groupby("item")["adopters"].max().sort_values(ascending=False)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    for color, item in zip(SERIES, order.index[: len(SERIES)]):
        sub = curves[curves["item"] == item]
        ax.plot(sub["time"], sub["adopters"], lw=1.6, color=color, label=item.replace("_", " "))
    ax.set_ylabel("cumulative agent handles")
    ax.set_title("Technique adoption across distinct handles")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", lw=0.5)
    _annotate_incidents(ax, order.max())

    naming = diffusion.naming_convention_diffusion(feat)
    naming = naming[naming["n_labels"] >= 5].head(10)
    ax2.barh(naming["motif"], naming["n_labels"], color=SUPPORT)
    ax2.invert_yaxis()
    ax2.set_xlabel("distinct handles using the motif")
    ax2.set_title("Page-naming conventions that spread")
    ax2.grid(axis="x", lw=0.5)
    return _save(fig, "05_adoption.svg")


def figure_structural_break(feat: pd.DataFrame) -> Path:
    br = temporal.structural_break(feat)
    keys = [
        ("density", "density"),
        ("reciprocity", "reciprocity"),
        ("transitivity", "transitivity"),
        ("giant_fraction", "giant component share"),
        ("modularity", "modularity Q"),
        ("zzz_page_share", "share of saves to ZZZ pages"),
    ]
    before = [br["before"].get(k, 0) or 0 for k, _ in keys]
    after = [br["after"].get(k, 0) or 0 for k, _ in keys]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    y = np.arange(len(keys))
    ax.barh(y - 0.2, before, height=0.38, color=MUTED, label="before 19 June")
    ax.barh(y + 0.2, after, height=0.38, color=ACCENT, label="19 June onward")
    ax.set_yticks(y, [lbl for _, lbl in keys], fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(axis="x", lw=0.5)
    ax.set_title("Handoff-network structure either side of the first mass deletion")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"{b:.3f}", (b, i - 0.2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=MUTED)
        ax.annotate(f"{a:.3f}", (a, i + 0.2), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=ACCENT)
    return _save(fig, "06_structural_break.svg")


def figure_core_network(built: dict[str, nx.Graph], min_core: int = 4) -> Path:
    g = nx.Graph(built["G1_handoff"])
    core_numbers = nx.core_number(g)
    keep = [n for n, k in core_numbers.items() if k >= min_core]
    sub = g.subgraph(keep)
    if sub.number_of_nodes() < 3:
        sub = g.subgraph(sorted(g.nodes, key=lambda n: -g.degree(n))[:150])

    membership = metrics.detect_communities(sub)
    pos = nx.spring_layout(sub, seed=metrics.SEED, k=1.2 / np.sqrt(max(sub.number_of_nodes(), 1)), iterations=200)

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    palette = {c: SERIES[i % len(SERIES)] for i, c in enumerate(sorted(set(membership.values())))}
    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.16, width=0.5, edge_color=INK)
    sizes = [18 + 5 * sub.degree(n) for n in sub.nodes]
    nx.draw_networkx_nodes(
        sub,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=[palette[membership[n]] for n in sub.nodes],
        linewidths=0.4,
        edgecolors="white",
    )
    hubs = sorted(sub.nodes, key=lambda n: -sub.degree(n))[:14]
    nx.draw_networkx_labels(sub, pos, {n: n for n in hubs}, font_size=6.5, ax=ax)
    ax.set_axis_off()
    ax.set_title(
        f"The coordination core: {sub.number_of_nodes()} handles in the {min_core}-core "
        f"of the handoff network, coloured by community"
    )
    ax.legend(
        handles=[Line2D([], [], marker="o", ls="", color=INK, alpha=0.4, label="node size = degree")],
        loc="lower right",
        fontsize=7,
    )
    return _save(fig, "07_core_network.svg")


def figure_brokerage(built: dict[str, nx.Graph]) -> Path:
    """Betweenness against degree: the gap identifies brokers, not hubs."""
    g = built["G1_handoff"]
    table = metrics.centrality_table(g)
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.scatter(
        table["degree"].clip(lower=1),
        table["betweenness"].clip(lower=1e-6),
        s=14,
        alpha=0.45,
        color=MUTED,
        edgecolors="none",
    )
    top = table.nlargest(12, "betweenness")
    ax.scatter(top["degree"], top["betweenness"], s=34, color=ACCENT, edgecolors="white", linewidths=0.5)
    for rec in top.itertuples():
        ax.annotate(rec.node, (rec.degree, rec.betweenness), xytext=(5, 2), textcoords="offset points", fontsize=6.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree (number of distinct handoff partners)")
    ax.set_ylabel("betweenness centrality")
    ax.set_title("Brokers vs hubs in the handoff network")
    ax.grid(lw=0.4, which="both")
    return _save(fig, "08_brokerage.svg")


def render_all(feat: pd.DataFrame, built: dict[str, nx.Graph]) -> list[Path]:
    return [
        figure_timeline(feat),
        figure_percolation(feat),
        figure_degree_distributions(built),
        figure_community_vs_family(feat, built),
        figure_adoption(feat),
        figure_structural_break(feat),
        figure_core_network(built),
        figure_brokerage(built),
    ]
