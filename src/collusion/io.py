"""Loading and integrity-checking the published corpus.

Every downstream claim is only as good as the clock behind it. The corpus grades
each timestamp (`time_grade`) and states an error bar (`uncertainty_seconds`);
both are carried through every loader here so that ordering-sensitive analysis
can refuse to run on rows it cannot order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

# Grades in decreasing order of trust. `reqlog` rows come from request logs and
# are good to the second; `rclog` rows come from the wiki's RecentChanges and
# are minute-ish; `write_date` rows carry only a date.
TIME_GRADES = ("reqlog", "rclog", "write_date")
ORDERABLE_GRADES = frozenset({"reqlog"})


def repo_root() -> Path:
    """Project root, resolved from this file rather than the cwd."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def derived_dir() -> Path:
    d = repo_root() / "derived"
    d.mkdir(exist_ok=True)
    return d


def figures_dir() -> Path:
    d = repo_root() / "figures"
    d.mkdir(exist_ok=True)
    return d


# --------------------------------------------------------------------------
# integrity
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChecksumResult:
    filename: str
    expected: str
    actual: str

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_checksums(root: Path | None = None) -> list[ChecksumResult]:
    """Recompute SHA-256 for every file listed in data/SHA256SUMS."""
    root = root or data_dir()
    out: list[ChecksumResult] = []
    for line in (root / "SHA256SUMS").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        expected, name = line.split(None, 1)
        name = name.strip()
        out.append(ChecksumResult(name, expected, sha256_file(root / name)))
    return out


# --------------------------------------------------------------------------
# raw readers
# --------------------------------------------------------------------------


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return json.loads((data_dir() / "manifest.json").read_text())


def _frame(path: Path, time_cols: tuple[str, ...] = ("time",)) -> pd.DataFrame:
    df = pd.DataFrame(list(iter_jsonl(path)))
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
    return df


def load_revisions(with_body: bool = True) -> pd.DataFrame:
    """14,591 stored saves. `body` is large (27MB total); drop it when unused."""
    df = _frame(data_dir() / "revisions.jsonl", ("time", "write_date", "archived_at"))
    if not with_body:
        df = df.drop(columns=[c for c in ("body", "hunks") if c in df.columns])
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def load_pages() -> pd.DataFrame:
    return _frame(data_dir() / "pages.jsonl", ("first_write", "last_write"))


def load_labels() -> pd.DataFrame:
    return _frame(data_dir() / "labels.jsonl", ("first_write", "last_write"))


def load_events() -> pd.DataFrame:
    return _frame(
        data_dir() / "events.jsonl",
        ("time", "request_time", "success_time", "recent_changes_time"),
    ).sort_values("time", kind="stable").reset_index(drop=True)


@lru_cache(maxsize=1)
def known_pages() -> dict[str, frozenset[str]]:
    """wiki -> set of page names held in the corpus, for link resolution."""
    pages = load_pages()
    return {
        wiki: frozenset(grp["name"])
        for wiki, grp in pages.groupby("wiki", sort=False)
    }


# --------------------------------------------------------------------------
# reconciliation against the corpus's own declared populations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reconciliation:
    name: str
    expected: int
    actual: int

    @property
    def ok(self) -> bool:
        return self.expected == self.actual


def reconcile_counts() -> list[Reconciliation]:
    """Assert our loaded row counts match manifest.json's stated populations."""
    m = load_manifest()
    revs, pages, labels = load_revisions(with_body=False), load_pages(), load_labels()
    events = load_events()

    checks = [
        Reconciliation("revisions", m["counts"]["revisions"]["value"], len(revs)),
        Reconciliation("pages", m["counts"]["pages"]["value"], len(pages)),
        Reconciliation("labels", m["counts"]["labels"]["value"], len(labels)),
    ]
    for wiki, stats in m["per_wiki"].items():
        checks.append(
            Reconciliation(
                f"{wiki}.revisions",
                stats["revisions"]["value"],
                int((revs["wiki"] == wiki).sum()),
            )
        )
        checks.append(
            Reconciliation(
                f"{wiki}.pages",
                stats["pages"]["value"],
                int((pages["wiki"] == wiki).sum()),
            )
        )
    for kind, stats in m["population_counts"].items():
        if not isinstance(stats, dict) or stats.get("kind") != "event_population":
            continue
        checks.append(
            Reconciliation(
                f"events.{kind}",
                stats["value"],
                int((events["event_type"] == kind).sum()),
            )
        )
    return checks


def manifest_self_checks() -> list[dict[str, Any]]:
    """The 122 checks the exporter recorded about itself; surface any failures."""
    return [c for c in load_manifest().get("checks", []) if not c.get("ok", True)]
