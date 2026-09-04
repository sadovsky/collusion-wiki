"""Stage orchestration: extract -> graphs -> metrics -> figures.

Each stage writes to `derived/` and reads only what earlier stages wrote, so any
stage can be re-run alone. Seeds are fixed throughout; re-running produces
identical JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from . import diffusion, features, graphs, io, metrics, temporal


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def write_json(payload: Any, name: str, subdir: str = "metrics") -> Path:
    out = io.derived_dir() / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return path


# --------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------


def stage_verify() -> dict[str, Any]:
    checksums = io.verify_checksums()
    bad_sums = [c.filename for c in checksums if not c.ok]
    if bad_sums:
        raise SystemExit(f"checksum mismatch, refusing to run: {bad_sums}")

    recon = io.reconcile_counts()
    failures = [f"{r.name}: expected {r.expected}, got {r.actual}" for r in recon if not r.ok]
    manifest_failures = io.manifest_self_checks()

    payload = {
        "checksums": [{"file": c.filename, "ok": c.ok} for c in checksums],
        "reconciliation": [
            {"name": r.name, "expected": r.expected, "actual": r.actual, "ok": r.ok} for r in recon
        ],
        "reconciliation_failures": failures,
        "manifest_self_check_failures": manifest_failures,
        "ok": not failures and not manifest_failures,
    }
    write_json(payload, "verification.json")
    if failures:
        raise SystemExit("population reconciliation failed: " + "; ".join(failures))
    return payload


def stage_extract() -> pd.DataFrame:
    df = features.build_features()
    features.build_shingles()
    features.write_summary(df)
    return df


# --------------------------------------------------------------------------
# stage 2
# --------------------------------------------------------------------------


def build_all_graphs(feat: pd.DataFrame) -> dict[str, nx.Graph]:
    shingles = features.load_shingles()
    bipartite = graphs.build_coedit_bipartite(feat)
    return {
        "G1_handoff": graphs.build_handoff(feat),
        "G2_coedit_bipartite": bipartite,
        "G2p_coedit_projection": graphs.project_coedit(bipartite),
        "G3_hyperlink": graphs.build_hyperlink(feat),
        "G4_resource_bipartite": graphs.build_resource_bipartite(feat),
        "G4b_label_host": graphs.build_label_host(feat),
        "G5_infrastructure": graphs.build_infrastructure(feat),
        "G6_provenance": graphs.build_provenance(feat, shingles),
    }


def stage_graphs(feat: pd.DataFrame) -> dict[str, nx.Graph]:
    built = build_all_graphs(feat)
    inventory = {}
    for name, g in built.items():
        graphs.export_graph(g, name)
        inventory[name] = graphs.describe(g)
    inventory["handoff_window_sensitivity"] = graphs.handoff_sensitivity(feat)
    write_json(inventory, "graph_inventory.json")
    return built


# --------------------------------------------------------------------------
# stage 3
# --------------------------------------------------------------------------


def label_family_ground_truth(feat: pd.DataFrame) -> dict[str, str]:
    """Each agent's dominant task family, from the corpus's own page taxonomy.

    `page_family` was produced by the corpus authors from page *content*, with no
    reference to who edited what. That independence is what makes it a usable
    ground truth for whether interaction structure recovers task organisation.
    """
    pages = io.load_pages()[["page_key", "page_family"]]
    classified = pages[pages["page_family"].notna() & ~pages["page_family"].isin(
        ["off_store_unclassified", "unknown", "source-or-unclassified"]
    )]
    joined = graphs.agent_rows(feat).merge(classified, on="page_key", how="inner")
    if joined.empty:
        return {}
    dominant = (
        joined.groupby(["label", "page_family"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .groupby("label")
        .first()
    )
    return dominant["page_family"].to_dict()


def page_family_map() -> dict[str, str]:
    pages = io.load_pages()
    return dict(zip(pages["page_key"], pages["page_family"]))


def stage_metrics(feat: pd.DataFrame, built: dict[str, nx.Graph], n_null: int = 200) -> dict[str, Any]:
    results: dict[str, Any] = {}

    heavy = {
        "G2_coedit_bipartite",
        "G2p_coedit_projection",
        "G4_resource_bipartite",
        "G4b_label_host",
        "G5_infrastructure",
        "G6_provenance",
    }
    for name, g in built.items():
        if g.number_of_edges() == 0:
            results[name] = {"name": name, "summary": graphs.describe(g)}
            continue
        results[name] = metrics.analyze(
            g,
            name,
            n_null=(30 if name in heavy else n_null),
            betweenness_k=300 if name in heavy else 500,
        )
    write_json(results, "structure.json")

    # Centrality tables for the layers a reader will actually interrogate.
    for name in ("G1_handoff", "G2p_coedit_projection", "G3_hyperlink"):
        g = built[name]
        if g.number_of_edges() == 0:
            continue
        table = metrics.centrality_table(g)
        table.to_csv(io.derived_dir() / "metrics" / f"centrality_{name}.csv", index=False)

    # Community validation against the corpus's own taxonomy.
    truth = label_family_ground_truth(feat)
    validation = {}
    for name in ("G1_handoff", "G2p_coedit_projection"):
        g = built[name]
        if g.number_of_edges() == 0:
            continue
        membership = metrics.detect_communities(nx.Graph(g))
        validation[name] = metrics.compare_partitions(membership, truth)
        validation[name]["assortativity_by_family"] = _family_assortativity(nx.Graph(g), truth)

    page_families = page_family_map()
    g3 = built["G3_hyperlink"]
    if g3.number_of_edges():
        membership = metrics.detect_communities(nx.Graph(g3))
        validation["G3_hyperlink"] = metrics.compare_partitions(membership, page_families)
        validation["G3_hyperlink"]["assortativity_by_family"] = _family_assortativity(
            nx.Graph(g3), page_families
        )
    write_json({"ground_truth_labels": len(truth), "layers": validation}, "community_validation.json")

    return {"structure": results, "validation": validation}


def _family_assortativity(graph: nx.Graph, truth: dict[str, str]) -> float | None:
    g = graph.copy()
    nx.set_node_attributes(g, {n: truth.get(n, "unknown") for n in g.nodes}, "family")
    return metrics.attribute_assortativity(g, "family")


def stage_temporal(feat: pd.DataFrame) -> dict[str, Any]:
    daily = temporal.activity_series(feat, "1D")
    hourly = temporal.activity_series(feat, "1h")
    daily.to_csv(io.derived_dir() / "metrics" / "activity_daily.csv", index=False)
    hourly.to_csv(io.derived_dir() / "metrics" / "activity_hourly.csv", index=False)

    curve = temporal.percolation_curve(feat)
    curve.to_csv(io.derived_dir() / "metrics" / "percolation_curve.csv", index=False)

    evolution = temporal.community_evolution(feat)
    evolution.to_csv(io.derived_dir() / "metrics" / "community_evolution.csv", index=False)

    payload = {
        "incidents": [{"date": d, "label": t} for d, t in temporal.INCIDENTS],
        "percolation": temporal.percolation_threshold(curve),
        "timing": temporal.timing_profile(feat),
        "structural_break": temporal.structural_break(feat),
        "deletion_response": temporal.deletion_response(feat),
        "peak_day": {
            "date": str(daily.loc[daily["saves"].idxmax(), "time"].date()),
            "saves": int(daily["saves"].max()),
        },
    }
    write_json(payload, "temporal.json")
    return payload


def stage_diffusion(feat: pd.DataFrame, built: dict[str, nx.Graph]) -> dict[str, Any]:
    network = nx.Graph(built["G1_handoff"])
    technique = diffusion.technique_diffusion(feat, network, column="techniques", min_adopters=20)
    hosts = diffusion.technique_diffusion(feat, network, column="hosts", min_adopters=40)

    curves = diffusion.adoption_curves(diffusion.adoption_events(feat, "techniques"), min_adopters=20)
    curves.to_csv(io.derived_dir() / "metrics" / "adoption_curves_techniques.csv", index=False)
    host_curves = diffusion.adoption_curves(diffusion.adoption_events(feat, "hosts"), min_adopters=40)
    host_curves.to_csv(io.derived_dir() / "metrics" / "adoption_curves_hosts.csv", index=False)

    vocabulary = diffusion.vocabulary_diffusion(feat)
    vocabulary.to_csv(io.derived_dir() / "metrics" / "vocabulary_diffusion.csv", index=False)
    naming = diffusion.naming_convention_diffusion(feat)
    naming.to_csv(io.derived_dir() / "metrics" / "naming_diffusion.csv", index=False)

    payload = {
        "techniques": technique,
        "hosts": hosts,
        "naming_conventions": naming.to_dict("records"),
        "vocabulary": vocabulary.to_dict("records"),
        "evasion_timing": [
            diffusion.convention_timing_shift(feat, motif)
            for motif in ("zzz", "fresh", "unique", "bridge", "relay")
        ],
    }
    write_json(payload, "diffusion.json")
    return payload


def stage_all(n_null: int = 200) -> dict[str, Any]:
    stage_verify()
    feat = stage_extract()
    built = stage_graphs(feat)
    stage_metrics(feat, built, n_null=n_null)
    stage_temporal(feat)
    stage_diffusion(feat, built)
    from . import viz

    viz.render_all(feat, built)
    from . import site

    site.build_payload(feat, built)
    return {"status": "complete"}
