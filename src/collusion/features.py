"""The one expensive pass over revision bodies.

Reads 27MB of page text once, writes a compact per-revision feature table that
every later stage reads instead. Keeping this isolated is what makes the graph
and metric stages cheap enough to iterate on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import extract, io


FEATURE_PATH = "revision_features.parquet"
BODY_HASH_PATH = "revision_shingles.parquet"


def build_features(save: bool = True) -> pd.DataFrame:
    revs = io.load_revisions(with_body=True)
    known = io.known_pages()

    rows = []
    for rec in revs.itertuples(index=False):
        body = rec.body or ""
        hosts = extract.extract_hosts(body)
        refs = extract.extract_page_refs(body, rec.wiki, known)
        tokens = extract.summary_tokens(rec.change_summary)
        naming = extract.naming_features(rec.name)

        row = {
            "rev_id": rec.rev_id,
            "page_key": rec.page_key,
            "wiki": rec.wiki,
            "name": rec.name,
            "seq": rec.seq,
            "label": rec.label,
            "ip16": rec.ip16,
            "time": rec.time,
            "time_grade": rec.time_grade,
            "uncertainty_seconds": rec.uncertainty_seconds,
            "body_len": rec.body_len,
            "body_sha256": rec.body_sha256,
            "lines": rec.lines,
            "change_summary": rec.change_summary if isinstance(rec.change_summary, str) else None,
            "summary_tokens": tokens,
            "n_urls": len(hosts),
            "hosts": sorted({h.host for h in hosts}),
            "techniques": sorted({h.technique for h in hosts}),
            "n_obfuscated_hosts": sum(1 for h in hosts if h.obfuscated),
            "page_refs": [r.page_key for r in refs],
            "page_ref_mechanisms": [r.mechanism for r in refs],
            "n_page_refs": len(refs),
        }
        row.update(naming)
        rows.append(row)

    df = pd.DataFrame(rows)
    if save:
        out = io.derived_dir() / FEATURE_PATH
        df.to_parquet(out, index=False)
    return df


def build_shingles(save: bool = True) -> pd.DataFrame:
    """Word-5-shingle sets per revision, for the near-duplicate copy graph.

    Stored separately from the feature table because the sets are large and only
    the provenance layer needs them.
    """
    revs = io.load_revisions(with_body=True)
    rows = [
        {
            "rev_id": rec.rev_id,
            "shingles": sorted(extract.shingles(rec.body or "")),
        }
        for rec in revs.itertuples(index=False)
    ]
    df = pd.DataFrame(rows)
    if save:
        df.to_parquet(io.derived_dir() / BODY_HASH_PATH, index=False)
    return df


def load_features() -> pd.DataFrame:
    path = io.derived_dir() / FEATURE_PATH
    if not path.exists():
        return build_features()
    return pd.read_parquet(path)


def load_shingles() -> pd.DataFrame:
    path = io.derived_dir() / BODY_HASH_PATH
    if not path.exists():
        return build_shingles()
    return pd.read_parquet(path)


def summarize(df: pd.DataFrame) -> dict:
    """Headline extraction counts, written alongside the table for provenance."""
    techniques: dict[str, int] = {}
    for lst in df["techniques"]:
        for t in lst:
            techniques[t] = techniques.get(t, 0) + 1
    mechanisms: dict[str, int] = {}
    for lst in df["page_ref_mechanisms"]:
        for m in lst:
            mechanisms[m] = mechanisms.get(m, 0) + 1
    hosts: dict[str, int] = {}
    for lst in df["hosts"]:
        for h in lst:
            hosts[h] = hosts.get(h, 0) + 1

    return {
        "revisions": int(len(df)),
        "total_urls": int(df["n_urls"].sum()),
        "distinct_hosts": len(hosts),
        "revisions_with_obfuscated_host": int((df["n_obfuscated_hosts"] > 0).sum()),
        "obfuscated_host_mentions": int(df["n_obfuscated_hosts"].sum()),
        "revisions_by_technique": dict(sorted(techniques.items(), key=lambda kv: -kv[1])),
        "page_refs_total": int(df["n_page_refs"].sum()),
        "page_refs_by_mechanism": dict(sorted(mechanisms.items(), key=lambda kv: -kv[1])),
        "top_hosts": dict(sorted(hosts.items(), key=lambda kv: -kv[1])[:40]),
        "naming_motifs": {
            col.removeprefix("name_"): int(df[col].sum())
            for col in df.columns
            if col.startswith("name_")
        },
    }


def write_summary(df: pd.DataFrame) -> Path:
    path = io.derived_dir() / "extraction_summary.json"
    path.write_text(json.dumps(summarize(df), indent=2, sort_keys=False) + "\n")
    return path
