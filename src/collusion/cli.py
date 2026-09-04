"""Command line entry point: `python -m collusion <stage>`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable

from . import features, graphs, io, pipeline

STAGES = ("verify", "extract", "graphs", "metrics", "temporal", "diffusion", "figures", "site", "all")


def _run(name: str, fn: Callable[[], object]) -> object:
    start = time.time()
    print(f"[{name}] start", file=sys.stderr, flush=True)
    result = fn()
    print(f"[{name}] done in {time.time() - start:.1f}s", file=sys.stderr, flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="collusion",
        description="Graph-structural analysis of the OpenAI agent wiki collusion corpus.",
    )
    parser.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    parser.add_argument(
        "--null-samples",
        type=int,
        default=200,
        help="null-model resamples per statistic (lower is faster, noisier)",
    )
    args = parser.parse_args(argv)

    if args.stage == "verify":
        payload = _run("verify", pipeline.stage_verify)
        print(json.dumps(payload, indent=2))
        return 0

    if args.stage == "all":
        _run("all", lambda: pipeline.stage_all(n_null=args.null_samples))
        print(f"outputs written to {io.derived_dir()} and {io.figures_dir()}")
        return 0

    if args.stage == "extract":
        _run("extract", pipeline.stage_extract)
        return 0

    feat = features.load_features()

    if args.stage == "graphs":
        _run("graphs", lambda: pipeline.stage_graphs(feat))
        return 0

    # Prefer the exports; only rebuild if the graphs stage has not run.
    built = graphs.load_all_exported()
    if len(built) < len(graphs.GRAPH_NAMES):
        print("[graphs] exports incomplete, rebuilding", file=sys.stderr)
        built = pipeline.build_all_graphs(feat)

    if args.stage == "metrics":
        _run("metrics", lambda: pipeline.stage_metrics(feat, built, n_null=args.null_samples))
    elif args.stage == "temporal":
        _run("temporal", lambda: pipeline.stage_temporal(feat))
    elif args.stage == "diffusion":
        _run("diffusion", lambda: pipeline.stage_diffusion(feat, built))
    elif args.stage == "figures":
        from . import viz

        _run("figures", lambda: viz.render_all(feat, built))
    elif args.stage == "site":
        from . import site

        _run("site", lambda: site.build_payload(feat, built))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
