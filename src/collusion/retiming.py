"""When the agents worked, and whether the change of hours was evasion.

The observation that starts this: before 19 June, 92% of agent writes landed in
hours the administrator was active; afterwards, 60%. The peak hour of writing
moved from 20:00 -- the administrator's second-busiest -- to 08:00, inside a
window in which the administrator never once deleted anything. It reads like
deliberate avoidance.

Three controls decide whether it is. They are applied in increasing order of
severity, and the finding does not survive the third.

  1. Composition. Did the same handles change their own hours, or did a
     night-active population simply replace a day-active one?
  2. Baseline. Is the "before" period a stable schedule, or is it dominated by
     one two-day burst that happened to run in the evening?
  3. A wiki with no adversary. probier was never policed -- not one of the
     corpus's 5,217 deletions touched it. If the shift is avoidance, it should
     be absent there.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import features, io

CRACKDOWN = pd.Timestamp("2026-06-19", tz="UTC")
SPIKE_START = pd.Timestamp("2026-06-16", tz="UTC")
SPIKE_END = pd.Timestamp("2026-06-19", tz="UTC")

# wikiservice.at is Austrian and the wiki is German-language, so the
# administrator's clock is almost certainly CEST. Used only to render local
# time alongside UTC; nothing is computed from it.
ADMIN_UTC_OFFSET = 2


def admin_hourly() -> pd.Series:
    """Administrator deletion intensity by hour, as a share of all deletions."""
    events = io.load_events()
    deletions = events[events["event_type"] == "delete"]
    return deletions["time"].dt.hour.value_counts(normalize=True).reindex(range(24), fill_value=0.0)


def _exposure(hours: pd.Series, admin: pd.Series) -> float:
    """Mean administrator intensity at the hours these writes happened.

    A continuous measure rather than a binary in-hours / out-of-hours split: the
    administrator's activity is far from uniform across the hours they were
    awake, and a binary treats 19:00 (21% of deletions) the same as 10:00 (1%).
    """
    return float(admin[hours].mean()) if len(hours) else float("nan")


def permutation_difference(
    a_hours: pd.Series,
    b_hours: pd.Series,
    admin: pd.Series,
    n: int = 5000,
    seed: int = 20260618,
) -> dict[str, Any]:
    """Difference in exposure between two sets of writes, against a label shuffle."""
    observed = _exposure(a_hours, admin) - _exposure(b_hours, admin)
    pool = np.concatenate([a_hours.to_numpy(), b_hours.to_numpy()])
    n_a = len(a_hours)
    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        rng.shuffle(pool)
        null[i] = admin[pool[:n_a]].mean() - admin[pool[n_a:]].mean()
    return {
        "exposure_a": _exposure(a_hours, admin),
        "exposure_b": _exposure(b_hours, admin),
        "difference": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "p_value": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n + 1)),
        "n_a": int(n_a),
        "n_b": int(len(b_hours)),
        "n_permutations": n,
    }


# --------------------------------------------------------------------------
# the observation
# --------------------------------------------------------------------------


def headline() -> dict[str, Any]:
    """The before/after comparison as originally framed, with a binary split."""
    feat = features.load_features()
    admin = admin_hourly()
    active = set(admin[admin > 0].index.tolist())
    silent = sorted(set(range(24)) - active)

    hour = feat["time"].dt.hour
    after = feat["time"] >= CRACKDOWN
    in_active = hour.isin(active)
    return {
        "admin_active_hours": sorted(active),
        "admin_silent_hours": silent,
        "silent_window_utc": [silent[0], silent[-1] + 1] if silent else None,
        "silent_window_admin_local": (
            [(silent[0] + ADMIN_UTC_OFFSET) % 24, (silent[-1] + 1 + ADMIN_UTC_OFFSET) % 24] if silent else None
        ),
        "admin_peak_hour": int(admin.idxmax()),
        "admin_peak_share": float(admin.max()),
        "share_in_active_before": float(in_active[~after].mean()),
        "share_in_active_after": float(in_active[after].mean()),
        "peak_hour_before": int(hour[~after].value_counts().idxmax()),
        "peak_hour_after": int(hour[after].value_counts().idxmax()),
        "writes_before": int((~after).sum()),
        "writes_after": int(after.sum()),
    }


def hourly_profiles() -> list[dict[str, Any]]:
    """Hour-of-day shares for each period, plus administrator intensity."""
    feat = features.load_features()
    admin = admin_hourly()
    hour = feat["time"].dt.hour

    periods = {
        "pre_spike": feat["time"] < SPIKE_START,
        "spike": (feat["time"] >= SPIKE_START) & (feat["time"] < SPIKE_END),
        "after": feat["time"] >= CRACKDOWN,
    }
    rows = []
    for h in range(24):
        row = {"hour": h, "admin": float(admin[h])}
        for name, mask in periods.items():
            sub = hour[mask]
            row[name] = float((sub == h).mean()) if len(sub) else 0.0
        rows.append(row)
    return rows


def day_hour_matrix() -> list[dict[str, Any]]:
    """Writes per day and hour -- the batch structure, made visible.

    This is the figure that settles the argument. Activity does not spread across
    the day and then retreat from the evening; it arrives in a handful of narrow
    blocks whose position moves around.
    """
    feat = features.load_features()
    grouped = (
        feat.assign(day=feat["time"].dt.strftime("%Y-%m-%d"), hour=feat["time"].dt.hour)
        .groupby(["day", "hour"])
        .size()
        .reset_index(name="writes")
    )
    events = io.load_events()
    deletions = events[events["event_type"] == "delete"]
    dele = (
        deletions.assign(day=deletions["time"].dt.strftime("%Y-%m-%d"), hour=deletions["time"].dt.hour)
        .groupby(["day", "hour"])
        .size()
        .reset_index(name="deletions")
    )
    merged = grouped.merge(dele, on=["day", "hour"], how="outer").fillna(0)
    return [
        {"d": r["day"], "h": int(r["hour"]), "w": int(r["writes"]), "x": int(r["deletions"])}
        for _, r in merged.iterrows()
    ]


# --------------------------------------------------------------------------
# control 1: composition
# --------------------------------------------------------------------------


def composition_control(min_writes: int = 5) -> dict[str, Any]:
    """Did individual handles move their own hours, or did the population turn over?

    Paired within handle: an agent that wrote on both sides of the boundary
    contributes its own before and after exposure. Turnover cannot produce a
    consistent within-handle shift.
    """
    feat = features.load_features()
    admin = admin_hourly()
    feat = feat.assign(hour=feat["time"].dt.hour, after=feat["time"] >= CRACKDOWN)
    feat = feat[feat["label"] != ""]

    rows = []
    for label, group in feat.groupby("label"):
        before, after = group[~group["after"]], group[group["after"]]
        if len(before) < min_writes or len(after) < min_writes:
            continue
        rows.append(
            {
                "label": label,
                "n_before": int(len(before)),
                "n_after": int(len(after)),
                "exposure_before": _exposure(before["hour"], admin),
                "exposure_after": _exposure(after["hour"], admin),
            }
        )
    paired = pd.DataFrame(rows)
    if len(paired) < 5:
        return {"status": "insufficient_paired_handles", "n": int(len(paired))}

    lowered = int((paired["exposure_after"] < paired["exposure_before"]).sum())
    test = stats.wilcoxon(paired["exposure_before"], paired["exposure_after"])
    return {
        "min_writes_each_side": min_writes,
        "n_handles": int(len(paired)),
        "mean_exposure_before": float(paired["exposure_before"].mean()),
        "mean_exposure_after": float(paired["exposure_after"].mean()),
        "handles_lowering_exposure": lowered,
        "wilcoxon_p": float(test.pvalue),
        "verdict": (
            "the same handles moved their own hours; this is not population turnover"
            if lowered / len(paired) > 0.75 and test.pvalue < 0.01
            else "no consistent within-handle shift"
        ),
        "handles": paired.sort_values("n_before", ascending=False).head(25).to_dict("records"),
    }


# --------------------------------------------------------------------------
# control 2: what the baseline actually contains
# --------------------------------------------------------------------------


def baseline_control() -> dict[str, Any]:
    """Is the 'before' period a schedule, or one burst that ran in the evening?

    This is where the reading starts to fail. The pre-crackdown period is 89%
    made of the 16-18 June spike. Before that spike the agents were already
    barely working in the administrator's evening hours -- so there is no stable
    evening habit for them to have retreated from.
    """
    feat = features.load_features()
    admin = admin_hourly()
    hour = feat["time"].dt.hour

    periods = {
        "pre_spike": feat["time"] < SPIKE_START,
        "spike": (feat["time"] >= SPIKE_START) & (feat["time"] < SPIKE_END),
        "after": feat["time"] >= CRACKDOWN,
    }
    summary = {}
    for name, mask in periods.items():
        sub = hour[mask]
        summary[name] = {
            "writes": int(len(sub)),
            "exposure": _exposure(sub, admin),
            "peak_hour": int(sub.value_counts().idxmax()) if len(sub) else None,
            "evening_share_18_21": float(sub.between(18, 21).mean()) if len(sub) else None,
            "night_share_01_07": float(sub.between(1, 7).mean()) if len(sub) else None,
            "top3_hours": [int(h) for h in sub.value_counts().head(3).index] if len(sub) else [],
            "top3_share": float(sub.value_counts(normalize=True).head(3).sum()) if len(sub) else None,
        }

    before_mask = feat["time"] < CRACKDOWN
    spike_share = float(periods["spike"].sum() / before_mask.sum())

    clean = permutation_difference(hour[periods["pre_spike"]], hour[periods["after"]], admin)
    return {
        "periods": summary,
        "spike_share_of_before_period": spike_share,
        "pre_spike_vs_after": clean,
        "verdict": (
            "the evening concentration belongs to the spike, not to a baseline schedule"
            if summary["spike"]["evening_share_18_21"] > 3 * summary["pre_spike"]["evening_share_18_21"]
            else "the before period has a stable schedule"
        ),
    }


# --------------------------------------------------------------------------
# control 3: a wiki with no adversary
# --------------------------------------------------------------------------


def adversary_control() -> dict[str, Any]:
    """The same comparison on a wiki nobody was policing.

    Every one of the corpus's deletions happened on dse. If the change of hours
    is avoidance of that administrator, it has no reason to appear on probier,
    where nothing was ever deleted.
    """
    feat = features.load_features()
    events = io.load_events()
    admin = admin_hourly()
    deletions_by_wiki = events[events["event_type"] == "delete"]["wiki"].value_counts().to_dict()

    feat = feat.assign(hour=feat["time"].dt.hour)
    pre_spike = feat[feat["time"] < SPIKE_START]
    after = feat[feat["time"] >= CRACKDOWN]

    per_wiki = {}
    for wiki in sorted(feat["wiki"].unique()):
        a = pre_spike[pre_spike["wiki"] == wiki]["hour"]
        b = after[after["wiki"] == wiki]["hour"]
        if len(a) < 50 or len(b) < 50:
            per_wiki[wiki] = {"status": "too_few_writes", "n_before": int(len(a)), "n_after": int(len(b))}
            continue
        result = permutation_difference(a, b, admin)
        result["deletions_on_this_wiki"] = int(deletions_by_wiki.get(wiki, 0))
        per_wiki[wiki] = result

    policed = per_wiki.get("dse", {})
    unpoliced = per_wiki.get("probier", {})
    both_shift = (
        policed.get("p_value", 1) < 0.05
        and unpoliced.get("p_value", 1) < 0.05
        and unpoliced.get("difference", 0) > 0
    )
    return {
        "deletions_by_wiki": deletions_by_wiki,
        "per_wiki": per_wiki,
        "verdict": (
            "the shift is the same on a wiki with no administrator, so it is not avoidance"
            if both_shift
            else "the shift is specific to the policed wiki"
        ),
    }


# --------------------------------------------------------------------------


def batch_signature() -> dict[str, Any]:
    """How concentrated writing is within the day, in each period.

    Both windows are narrow. Whatever moved was a schedule, and a schedule is
    the kind of thing that is set rather than chosen in response to a threat.
    """
    feat = features.load_features()
    hour = feat["time"].dt.hour
    out = {}
    for name, mask in (
        ("pre_spike", feat["time"] < SPIKE_START),
        ("spike", (feat["time"] >= SPIKE_START) & (feat["time"] < SPIKE_END)),
        ("after", feat["time"] >= CRACKDOWN),
    ):
        sub = hour[mask]
        shares = sub.value_counts(normalize=True)
        # Entropy over the 24 hours: low means the day's writing is compressed
        # into a few blocks, high means it is spread out.
        p = shares.reindex(range(24), fill_value=0.0).to_numpy()
        nz = p[p > 0]
        out[name] = {
            "writes": int(len(sub)),
            "top3_share": float(shares.head(3).sum()),
            "hours_covering_80pct": int((shares.sort_values(ascending=False).cumsum() < 0.8).sum() + 1),
            "entropy_bits": float(-(nz * np.log2(nz)).sum()),
            "max_entropy_bits": float(np.log2(24)),
        }
    return out


def analyze() -> dict[str, Any]:
    return {
        "headline": headline(),
        "hourly": hourly_profiles(),
        "composition_control": composition_control(),
        "baseline_control": baseline_control(),
        "adversary_control": adversary_control(),
        "batch_signature": batch_signature(),
        "conclusion": (
            "The change of hours is real and the same handles made it, but it appears "
            "identically on a wiki that was never policed, and the evening pattern it "
            "moved away from belonged to a two-day burst rather than to a standing "
            "schedule. The parsimonious reading is a change in when the agents were "
            "run, not a decision to hide."
        ),
    }
