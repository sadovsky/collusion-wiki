"""Detection avoidance, and whether any of it worked.

The corpus records two different adversaries, and conflating them makes the
evidence unreadable:

  The sandbox -- whatever restricted the agents' own egress. Countered with
  encoded hostnames, invented query parameters, CORS and markdown proxies,
  tunnels and Azure-hostname spoofing. This fight peaks on 18 June.

  The administrator -- a person deleting agent pages from one wiki. Countered
  with naming tricks, page recreation, retimed activity and dispersal to other
  wikis. This fight starts on 19 June and the administrator keeps going for
  three weeks after the agents stop.

The question this module exists to answer is the one the published report does
not ask: which of these tactics actually changed an outcome? Deletion events
carry timestamps and page identities, so survival is measurable rather than
inferred from the fact that agents tried something.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import features, io

CRACKDOWN = pd.Timestamp("2026-06-19", tz="UTC")

# Query parameters ProWiki actually uses. Anything else appearing in a
# `wiki.cgi` URL was invented by an agent, and the usual reason to invent one is
# to make an otherwise identical request look new.
STANDARD_PARAMS = frozenset(
    {"action", "id", "lang", "diff", "revision", "search", "title", "keywords", "raw"}
)
PARAM_RE = re.compile(r"[?&]([a-z0-9_]{2,14})=", re.I)
URL_RE = re.compile(r'https?://[^\s\]\|<>"\')]+')
EPOCH_RE = re.compile(r"1(?:7|8)\d{8}")

ALPHA_POS = {c: i for i, c in enumerate(string.ascii_uppercase)}


# --------------------------------------------------------------------------
# page survival
# --------------------------------------------------------------------------


def page_survival() -> pd.DataFrame:
    """One row per held page: when it was written, when it died, what it looked like."""
    pages = io.load_pages()
    events = io.load_events()
    deletions = events[events["event_type"] == "delete"][["page_key", "time"]]
    first_deletion = (
        deletions.sort_values("time").groupby("page_key", as_index=False).first()
        .rename(columns={"time": "deleted_at"})
    )
    observation_end = events["time"].max()

    df = pages.merge(first_deletion, on="page_key", how="left")
    df["deleted"] = df["deleted_at"].notna()
    # Censored at the last observed administrator action, not at the last agent
    # write: pages were still being deleted twelve days after the agents stopped.
    df["lifetime_hours"] = (
        (df["deleted_at"].fillna(observation_end) - df["first_write"]).dt.total_seconds() / 3600
    )
    df["observation_end"] = observation_end

    name = df["name"].fillna("")
    df["starts_with_z"] = name.str.upper().str.startswith("Z")
    df["contains_zzz"] = name.str.contains("zzz", case=False, regex=False)
    df["epoch_suffix"] = name.str.contains(EPOCH_RE)
    df["off_dse"] = df["wiki"] != "dse"
    df["created_after_crackdown"] = df["first_write"] >= CRACKDOWN
    df["creation_week"] = df["first_write"].dt.isocalendar().week

    feat = features.load_features()
    per_page = feat.groupby("page_key").agg(
        obfuscated_hosts=("n_obfuscated_hosts", "sum"),
        revisions=("rev_id", "size"),
    )
    df = df.merge(per_page, on="page_key", how="left")
    df["uses_obfuscated_host"] = df["obfuscated_hosts"].fillna(0) > 0
    return df


def survival_by_feature(survival: pd.DataFrame) -> list[dict[str, Any]]:
    """Deletion rate and median lifetime for each claimed evasion tactic.

    Reported twice: over the whole corpus, and restricted to the one wiki where
    deletions actually happened. The restricted figure is the honest one -- a
    tactic that looks protective corpus-wide may simply be correlated with being
    somewhere nobody was deleting.
    """
    dse = survival[survival["wiki"] == "dse"]
    tactics = [
        ("all pages", survival["page_key"].notna(), dse["page_key"].notna()),
        ("starts with Z", survival["starts_with_z"], dse["starts_with_z"]),
        ("contains ZZZ", survival["contains_zzz"], dse["contains_zzz"]),
        ("epoch-suffix name", survival["epoch_suffix"], dse["epoch_suffix"]),
        ("encoded hostname", survival["uses_obfuscated_host"], dse["uses_obfuscated_host"]),
        ("on another wiki", survival["off_dse"], None),
    ]
    rows = []
    for label, whole_mask, dse_mask in tactics:
        whole = survival[whole_mask]
        row = {
            "tactic": label,
            "pages": int(len(whole)),
            "deleted_share": float(whole["deleted"].mean()) if len(whole) else None,
            "median_lifetime_hours": float(whole["lifetime_hours"].median()) if len(whole) else None,
        }
        if dse_mask is not None:
            sub = dse[dse_mask]
            row.update(
                {
                    "dse_pages": int(len(sub)),
                    "dse_deleted_share": float(sub["deleted"].mean()) if len(sub) else None,
                    "dse_median_lifetime_hours": float(sub["lifetime_hours"].median()) if len(sub) else None,
                }
            )
        else:
            row.update({"dse_pages": 0, "dse_deleted_share": None, "dse_median_lifetime_hours": None})
        rows.append(row)
    return rows


def matched_week_comparison(survival: pd.DataFrame, feature: str = "starts_with_z") -> list[dict[str, Any]]:
    """Compare a tactic against everything else created the same week, on dse.

    Lifetime is measured from creation, so a tactic used early looks long-lived
    for reasons that have nothing to do with the tactic. Matching on creation
    week removes that.
    """
    dse = survival[survival["wiki"] == "dse"]
    rows = []
    for week, group in dse.groupby("creation_week"):
        marked, rest = group[group[feature]], group[~group[feature]]
        if len(marked) < 5:
            continue
        u = stats.mannwhitneyu(
            marked["lifetime_hours"], rest["lifetime_hours"], alternative="greater"
        ) if len(rest) else None
        rows.append(
            {
                "week": int(week),
                "n_marked": int(len(marked)),
                "n_rest": int(len(rest)),
                "marked_deleted_share": float(marked["deleted"].mean()),
                "rest_deleted_share": float(rest["deleted"].mean()),
                "marked_median_hours": float(marked["lifetime_hours"].median()),
                "rest_median_hours": float(rest["lifetime_hours"].median()),
                "p_marked_lives_longer": float(u.pvalue) if u is not None else None,
            }
        )
    return rows


def survival_curves(survival: pd.DataFrame, horizon_hours: int = 900, step: int = 6) -> pd.DataFrame:
    """Kaplan-Meier survival, split by wiki.

    With deletion as the event and the last administrator action as the censoring
    time, this is the direct answer to whether a page written on DSEWiki had any
    chance of persisting.
    """
    frames = []
    for group, sub in (("dse", survival[survival["wiki"] == "dse"]), ("other wikis", survival[survival["wiki"] != "dse"])):
        times = sub["lifetime_hours"].to_numpy()
        events = sub["deleted"].to_numpy()
        grid = np.arange(0, horizon_hours + step, step)
        surviving = []
        s = 1.0
        previous = 0.0
        for t in grid:
            at_risk = int(np.sum(times >= previous))
            died = int(np.sum((times >= previous) & (times < t) & events))
            if at_risk > 0 and died > 0:
                s *= 1 - died / at_risk
            surviving.append(s)
            previous = t
        frames.append(pd.DataFrame({"group": group, "hours": grid, "survival": surviving}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# was the administrator actually working alphabetically?
# --------------------------------------------------------------------------


def alphabetical_sweep_test(min_session: int = 40, gap_hours: float = 2.0) -> dict[str, Any]:
    """Did deletions proceed A to Z within a sitting?

    The whole rationale for a `ZZZ` prefix is that an alphabetical sweep reaches
    the end of the list last. That premise is testable: split deletions into
    sittings and correlate each page's initial letter with its position in the
    sitting. If the administrator did not work alphabetically, the tactic was
    aimed at a behaviour that did not exist.
    """
    events = io.load_events()
    deletions = events[(events["event_type"] == "delete") & events["page"].notna()].copy()
    deletions = deletions.sort_values("time")
    deletions["alpha"] = deletions["page"].str[0].str.upper().map(ALPHA_POS)
    deletions = deletions[deletions["alpha"].notna()]

    gaps = deletions["time"].diff().dt.total_seconds().fillna(0)
    deletions["session"] = (gaps > gap_hours * 3600).cumsum()

    sessions = []
    for session, group in deletions.groupby("session"):
        if len(group) < min_session:
            continue
        order = np.arange(len(group))
        alpha = group["alpha"].to_numpy(dtype=float)
        if np.all(alpha == alpha[0]):
            continue
        result = stats.spearmanr(order, alpha)
        sessions.append(
            {
                "session": int(session),
                "n": int(len(group)),
                "start": str(group["time"].min())[:16],
                "rho": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    rhos = np.array([s["rho"] for s in sessions])
    return {
        "sessions": sessions,
        "n_sessions": len(sessions),
        "median_rho": float(np.median(rhos)) if len(rhos) else None,
        "strongly_alphabetical": int(np.sum(rhos > 0.5)),
        "strongly_reverse": int(np.sum(rhos < -0.5)),
        "verdict": (
            "deletions were not systematically alphabetical"
            if len(rhos) and np.median(rhos) < 0.3
            else "deletions show alphabetical ordering"
        ),
    }


# --------------------------------------------------------------------------
# retiming
# --------------------------------------------------------------------------


def temporal_evasion(n_permutations: int = 2000, seed: int = 20260618) -> dict[str, Any]:
    """Did agents move their writing out of the administrator's active hours?

    Statistic: the share of agent writes falling in hours when the administrator
    ever deleted anything. Compared before and against after the first mass
    deletion, with a permutation test on the labelling of writes as before/after.
    """
    feat = features.load_features()
    events = io.load_events()
    deletions = events[events["event_type"] == "delete"]

    admin_hours = deletions["time"].dt.hour.value_counts()
    active_hours = set(admin_hours[admin_hours > 0].index.tolist())
    quiet_hours = sorted(set(range(24)) - active_hours)

    hour = feat["time"].dt.hour
    after = (feat["time"] >= CRACKDOWN).to_numpy()
    in_admin_hours = hour.isin(active_hours).to_numpy()

    before_share = float(in_admin_hours[~after].mean())
    after_share = float(in_admin_hours[after].mean())
    observed = before_share - after_share

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(after)
        null.append(float(in_admin_hours[~shuffled].mean() - in_admin_hours[shuffled].mean()))
    null_arr = np.array(null)
    p = float((np.sum(np.abs(null_arr) >= abs(observed)) + 1) / (n_permutations + 1))

    hourly = pd.DataFrame(
        {
            "hour": range(24),
            "agent_before": [float((hour[~after] == h).mean()) for h in range(24)],
            "agent_after": [float((hour[after] == h).mean()) for h in range(24)],
            "admin": [
                float((deletions["time"].dt.hour == h).mean()) for h in range(24)
            ],
        }
    )
    return {
        "admin_active_hours": sorted(active_hours),
        "admin_silent_hours": quiet_hours,
        "share_in_admin_hours_before": before_share,
        "share_in_admin_hours_after": after_share,
        "drop": observed,
        "permutation_p": p,
        "n_permutations": n_permutations,
        "hourly": hourly.to_dict("records"),
        "peak_hour_before": int(hourly["agent_before"].idxmax()),
        "peak_hour_after": int(hourly["agent_after"].idxmax()),
        "admin_peak_hour": int(hourly["admin"].idxmax()),
    }


# --------------------------------------------------------------------------
# the arms race
# --------------------------------------------------------------------------


def recreation_battles(max_rows: int = 20) -> dict[str, Any]:
    """Pages deleted more than once: the administrator and the agents trading blows."""
    events = io.load_events()
    deletions = events[events["event_type"] == "delete"][["page_key", "page", "time"]]
    counts = deletions.groupby("page_key").size().sort_values(ascending=False)
    repeated = counts[counts > 1]

    battles = []
    for page_key in repeated.head(max_rows).index:
        times = sorted(deletions[deletions["page_key"] == page_key]["time"])
        gaps = [round((b - a).total_seconds() / 60, 1) for a, b in zip(times, times[1:])]
        battles.append(
            {
                "page": page_key.split("~", 1)[-1],
                "wiki": page_key.split("~", 1)[0],
                "deletions": len(times),
                "first": str(times[0])[:16],
                "last": str(times[-1])[:16],
                "gaps_minutes": gaps,
                "fastest_recreation_minutes": min(gaps) if gaps else None,
            }
        )
    return {
        "pages_deleted_once": int((counts == 1).sum()),
        "pages_deleted_repeatedly": int(len(repeated)),
        "deletions_spent_on_repeats": int(repeated.sum()),
        "max_deletions_on_one_page": int(counts.max()),
        "battles": battles,
    }


def deletion_response_latency() -> dict[str, Any]:
    """How quickly a deleted page was written again -- evidence of monitoring."""
    events = io.load_events()
    feat = features.load_features()
    deletions = events[events["event_type"] == "delete"][["page_key", "time"]]
    saves = feat[["page_key", "time", "label"]]

    merged = deletions.merge(saves, on="page_key", suffixes=("_delete", "_save"))
    after = merged[merged["time_save"] > merged["time_delete"]].copy()
    if after.empty:
        return {"recreations": 0}
    after["latency_minutes"] = (after["time_save"] - after["time_delete"]).dt.total_seconds() / 60
    first = (
        after.sort_values("latency_minutes")
        .groupby(["page_key", "time_delete"], as_index=False)
        .first()
    )
    lat = first["latency_minutes"]
    return {
        "recreations": int(len(first)),
        "median_minutes": float(lat.median()),
        "within_10_minutes": int((lat <= 10).sum()),
        "within_60_minutes": int((lat <= 60).sum()),
        "distinct_responders": int(first["label"].nunique()),
        "fastest_minutes": float(lat.min()),
    }


# --------------------------------------------------------------------------
# sandbox-facing tactics
# --------------------------------------------------------------------------


def invented_parameters() -> pd.DataFrame:
    """Non-standard query parameters written into wiki.cgi URLs, by day.

    A request that differs only in a junk parameter is the same request wearing
    a different hat. Counting how many distinct hats appear per day measures how
    hard something was working to look new.
    """
    revisions = io.load_revisions(with_body=True)
    rows = []
    for rec in revisions.itertuples(index=False):
        for url in URL_RE.findall(rec.body or ""):
            if "wiki.cgi" not in url:
                continue
            for key in PARAM_RE.findall(url):
                if key.lower() not in STANDARD_PARAMS:
                    rows.append({"day": rec.time.floor("1D"), "key": key.lower()})
    if not rows:
        return pd.DataFrame(columns=["day", "mentions", "distinct_parameters"])
    df = pd.DataFrame(rows)
    daily = df.groupby("day").agg(mentions=("key", "size"), distinct_parameters=("key", "nunique"))
    daily["top_parameter"] = df.groupby("day")["key"].agg(lambda s: s.value_counts().idxmax())
    return daily.reset_index()


def obfuscated_hostnames() -> dict[str, Any]:
    """Hostnames written with percent- or entity-encoded characters.

    Not a transport artifact: `www.sec%2egov` and `www.sec&#46;gov` are a domain
    name taken apart so a string match will not find it.
    """
    from .extract import URL_RE as _URL, normalize_host

    revisions = io.load_revisions(with_body=True)
    found = []
    for rec in revisions.itertuples(index=False):
        for url in _URL.findall(rec.body or ""):
            m = re.match(r"https?://([^/?#]+)", url, re.I)
            if not m:
                continue
            info = normalize_host(m.group(1))
            if info.obfuscated:
                found.append(
                    {
                        "time": rec.time,
                        "label": rec.label,
                        "page": rec.name,
                        "raw": info.raw,
                        "decoded": info.host,
                    }
                )
    if not found:
        return {"mentions": 0}
    df = pd.DataFrame(found)
    return {
        "mentions": int(len(df)),
        "revisions": int(df.groupby(["time", "page"]).ngroups),
        "distinct_labels": int(df["label"].nunique()),
        "distinct_raw_forms": int(df["raw"].nunique()),
        "first": str(df["time"].min())[:16],
        "last": str(df["time"].max())[:16],
        "days": sorted({str(t)[:10] for t in df["time"]}),
        "examples": (
            df.drop_duplicates("raw")[["raw", "decoded"]].head(18).to_dict("records")
        ),
    }


def dispersal() -> pd.DataFrame:
    """Share of writing that went to a wiki other than the one being policed."""
    feat = features.load_features()
    daily = feat.assign(day=feat["time"].dt.floor("1D"), off_dse=feat["wiki"] != "dse")
    out = daily.groupby("day").agg(writes=("rev_id", "size"), off_dse=("off_dse", "sum"))
    out["off_dse_share"] = out["off_dse"] / out["writes"]
    by_wiki = daily.groupby(["day", "wiki"]).size().unstack(fill_value=0)
    return out.join(by_wiki).reset_index()


# --------------------------------------------------------------------------


def analyze() -> dict[str, Any]:
    survival = page_survival()
    return {
        "crackdown": str(CRACKDOWN),
        "survival_by_tactic": survival_by_feature(survival),
        "matched_week_z_prefix": matched_week_comparison(survival, "starts_with_z"),
        "matched_week_epoch": matched_week_comparison(survival, "epoch_suffix"),
        "alphabetical_sweep": alphabetical_sweep_test(),
        "temporal_evasion": temporal_evasion(),
        "recreation_battles": recreation_battles(),
        "deletion_response": deletion_response_latency(),
        "obfuscated_hostnames": obfuscated_hostnames(),
        "totals": {
            "held_pages": int(len(survival)),
            "deleted": int(survival["deleted"].sum()),
            "survived": int((~survival["deleted"]).sum()),
            "dse_pages": int((survival["wiki"] == "dse").sum()),
            "dse_deleted_share": float(survival[survival["wiki"] == "dse"]["deleted"].mean()),
            "off_dse_pages": int((survival["wiki"] != "dse").sum()),
            "off_dse_deleted_share": float(survival[survival["wiki"] != "dse"]["deleted"].mean()),
        },
    }
