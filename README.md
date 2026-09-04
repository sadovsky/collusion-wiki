# collusion-wiki

Graph-structural analysis of the OpenAI agent wiki corpus published at
[collusion.wiki](https://collusion.wiki/).

In mid-2026, autonomous agents identifying themselves as OpenAI models were
given multi-round web-lookup tasks. They were meant to read the internet, not
write to it. They discovered that a handful of German ProWiki instances accepted
page edits over plain GET requests, and turned one of them into a message board:
passing answers between rounds, publishing lists of CORS and markdown proxies
that got them past their sandbox, chaining relay pages, and renaming pages to
survive an administrator's deletion sweep.

The published corpus is a flat event log. **Nothing in it is a graph.** This repo
reconstructs the network structure latent in edit sequences, page bodies, URL
payloads and IP prefixes, and measures whether the coordination was genuinely
collective and emergent.

## What's here

```
data/          the published corpus, unmodified (verified against SHA256SUMS)
src/collusion/ the pipeline
derived/       graph exports, metric tables, the site payload
figures/       static SVG figures
reports/       written findings
site/          the interactive page
```

## Running it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .

.venv/bin/python -m collusion verify     # checksums + population reconciliation
.venv/bin/python -m collusion all        # everything, ~15 min
```

Individual stages: `extract`, `graphs`, `metrics`, `temporal`, `diffusion`,
`figures`, `site`. Every stage reads what earlier stages wrote to `derived/`, so
any one can be re-run alone. Seeds are fixed; re-running produces identical
output.

## The six graph layers

Every layer is exported to `derived/graphs/` as GraphML (for Gephi or Cytoscape),
node/edge CSV, and node-link JSON.

| Layer | Edge meaning | Why it exists |
|---|---|---|
| `G1_handoff` | one agent edited a page, another edited it next | the coordination network proper |
| `G2_coedit_bipartite` / `G2p_coedit_projection` | agents sharing a workspace | who worked alongside whom |
| `G3_hyperlink` | a page body links to another page | the relay and bridge topology agents built |
| `G4_resource_bipartite` / `G4b_label_host` | a page or agent used an external endpoint | technique adoption |
| `G5_infrastructure` | an agent handle wrote from a /16 prefix | co-location, *not* identity |
| `G6_provenance` | one revision's content reappears in a later one | who copied from whom |

## Methodological commitments

These constrain what the analysis is allowed to claim.

- **Every structural claim carries a null model.** Reciprocity, transitivity,
  assortativity, modularity and adoption are each reported against a
  degree-preserving rewiring or a permutation, with a z-score and an empirical
  p-value. A bare network statistic is not evidence.
- **Time is graded.** The corpus states a `time_grade` and an
  `uncertainty_seconds` per event. Ordering-sensitive analysis (the provenance
  direction, in particular) refuses pairs it cannot order.
- **`ip16` is infrastructure, not identity.** One /16 prefix in this corpus
  carried 431 distinct handles. That is shared cloud egress. Nothing here treats
  a shared prefix as a shared actor.
- **Power laws are tested, not asserted.** Degree distributions are fit by MLE
  and compared against a lognormal by likelihood ratio. Where lognormal wins,
  that is what gets reported.
- **The handoff window is a choice, so it is swept.** Results at 1h / 6h / 24h /
  unbounded are all recorded in `derived/metrics/graph_inventory.json`.
- **No intent claims.** The corpus contains no chain-of-thought. Structure is
  observable; motivation is not.

## Non-goals

No de-anonymization beyond what the corpus already publishes. No exercising of
the bypass endpoints the agents found - they are counted as diffusion tokens and
nothing more.
