#!/usr/bin/env python3
"""Train heuristic override tables from aggregated benchmark data.

Usage:
    rj_bench_train.py <aggregation.json> --uarch <uarch> --out <file>
    rj_bench_train.py --from-results [--target <sha>] [--hashes <f>] --uarch <uarch> --out <file>

Consumes the weighted pairwise comparisons from rj_bench_aggregate.py and
produces a heuristic override table per TRAINING.md:
  bench/heuristics/<uarch>.json

Rules:
  - Only usable comparisons with weight >= MIN_COMPARISON_WEIGHT count.
  - A (twinA over twinB) override needs dominance >= DOMINANCE_THRESHOLD
    (default 0.8) and total_weight >= MIN_WEIGHT (default 1.0).
  - Confidence = total_weight / (total_weight + SKEPTICISM_PRIOR),
    SKEPTICISM_PRIOR = 2.0 (TRAINING.md: training needs stronger evidence).
  - Overrides below CONFIDENCE_THRESHOLD (0.7) go to not_overridden.
  - RETIRED_CONFIG comparisons are excluded from training.
  - STALE_UNKNOWN comparisons contribute at most 10% of nominal weight.

This never runs benchmarks. It only learns from aggregated data.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_COMPARISON_WEIGHT = 0.1
DOMINANCE_THRESHOLD = 0.8
MIN_WEIGHT = 1.0
CONFIDENCE_THRESHOLD = 0.7
SKEPTICISM_PRIOR = 2.0  # TRAINING.md: higher than aggregation's 1.0
STALE_UNKNOWN_DISCOUNT = 0.1


def confidence(total_weight):
    return total_weight / (total_weight + SKEPTICISM_PRIOR)


def effective_weight(pair):
    """Apply STALE_UNKNOWN discount (defense in depth)."""
    w = pair.get("weight", 0)
    flags = pair.get("flags", [])
    if "STALE_UNKNOWN" in flags:
        w *= STALE_UNKNOWN_DISCOUNT
    return w


def train(agg_report, uarch, target_version=None, target_csv=None):
    """Train overrides from an aggregation report. Returns the table dict."""
    pairs = agg_report.get("pairwise", [])
    # Filter: usable, sufficient weight, not retired
    usable = []
    for p in pairs:
        if not p.get("usable"):
            continue
        if "RETIRED_CONFIG" in p.get("flags", []):
            continue
        w = effective_weight(p)
        if w < MIN_COMPARISON_WEIGHT:
            continue
        usable.append((p, w))

    # Group by (topology, op, isa). Topology supports glob in output;
    # for training we group exact, then generalize if the pattern holds
    # across topologies (future work: keep exact for now).
    groups = defaultdict(list)
    for p, w in usable:
        key = (p.get("topology"), p.get("op"), p.get("isa"))
        groups[key].append((p, w))

    overrides = []
    not_overridden = []
    ovr_id = 0

    for (topo, op, isa), plist in sorted(groups.items(), key=lambda x: str(x[0])):
        # For each twin-pair, accumulate wins
        # Key: (frozenset(twins_a), frozenset(twins_b)) with canonical order
        pair_stats = defaultdict(lambda: {"wins_a": 0.0, "wins_b": 0.0,
                                          "ties": 0.0, "total": 0.0,
                                          "sources": []})
        for p, w in plist:
            ta = frozenset(p["a"].get("twins", []))
            tb = frozenset(p["b"].get("twins", []))
            if ta == tb:
                continue  # same twins: not a twin comparison
            # Canonical order for the pair key
            key = tuple(sorted([ta, tb], key=lambda x: sorted(x)))
            st = pair_stats[key]
            winner = p.get("winner")
            if p.get("tied"):
                st["ties"] += w
            elif winner == p["a"]["plan_id"]:
                # a won; determine if a is the first or second in canonical key
                if ta == key[0]:
                    st["wins_a"] += w
                else:
                    st["wins_b"] += w
            elif winner == p["b"]["plan_id"]:
                if tb == key[0]:
                    st["wins_a"] += w
                else:
                    st["wins_b"] += w
            st["total"] += w
            st["sources"].append({
                "version": p.get("source_version", "?")[:12],
                "weight": round(w, 3),
                "flags": p.get("flags", []),
            })

        for (ta, tb), st in pair_stats.items():
            total = st["total"]
            if total < MIN_WEIGHT:
                not_overridden.append({
                    "match": {"topology": topo, "op": op, "isa": isa,
                              "twins_a": sorted(ta), "twins_b": sorted(tb)},
                    "reason": f"insufficient evidence (total_weight {total:.2f} < {MIN_WEIGHT})",
                })
                continue
            # Dominance: does one side win decisively?
            wa, wb = st["wins_a"], st["wins_b"]
            # wins_a corresponds to key[0], wins_b to key[1]
            dom_a = wa / total if total > 0 else 0
            dom_b = wb / total if total > 0 else 0
            conf = confidence(total)
            if dom_a >= DOMINANCE_THRESHOLD and conf >= CONFIDENCE_THRESHOLD:
                ovr_id += 1
                overrides.append(_make_override(
                    ovr_id, topo, op, isa, sorted(ta), sorted(tb),
                    "prefer", st, conf, total))
            elif dom_b >= DOMINANCE_THRESHOLD and conf >= CONFIDENCE_THRESHOLD:
                ovr_id += 1
                overrides.append(_make_override(
                    ovr_id, topo, op, isa, sorted(tb), sorted(ta),
                    "prefer", st, conf, total))
            elif max(dom_a, dom_b) < 0.6 and total >= MIN_WEIGHT * 2:
                # Neither dominates but we have substantial data: explicitly
                # record "no preference" so the planner doesn't wonder.
                not_overridden.append({
                    "match": {"topology": topo, "op": op, "isa": isa,
                              "twins_a": sorted(ta), "twins_b": sorted(tb)},
                    "reason": f"no dominance (a:{dom_a:.2f} b:{dom_b:.2f}, weight {total:.2f})",
                })
            else:
                not_overridden.append({
                    "match": {"topology": topo, "op": op, "isa": isa,
                              "twins_a": sorted(ta), "twins_b": sorted(tb)},
                    "reason": f"below dominance/confidence threshold "
                              f"(dom {max(dom_a, dom_b):.2f}, conf {conf:.2f})",
                })

    # Content hashes of inputs for reproducibility
    input_hashes = []
    # (The aggregation report doesn't carry content hashes yet; record what we have.)
    table = {
        "format_version": "heuristics/1",
        "uarch": uarch,
        "generated_at": _utcnow(),
        "generated_by": "rj_bench_train.py",
        "target_raidjit_version": target_version,
        "target_config_space_version": target_csv,
        "inputs": {
            "num_comparisons": len(usable),
            "total_weight": round(sum(w for _, w in usable), 3),
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "skepticism_prior": SKEPTICISM_PRIOR,
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "overrides": overrides,
        "not_overridden": not_overridden,
    }
    return table


def _make_override(oid, topo, op, isa, prefer_twins, over_twins, kind, st, conf, total):
    return {
        "id": f"ovr-{oid:03d}",
        "match": {"topology": topo, "op": op, "isa": isa},
        kind: {
            "twins": prefer_twins,
            "over_twins": [over_twins],
            "reason": (f"{prefer_twins or 'baseline'} beats {over_twins or 'baseline'} "
                       f"in {st['wins_a'] + st['wins_b']:.1f}/{total:.1f} weighted comparisons"),
        },
        "confidence": round(conf, 3),
        "evidence": {
            "total_weight": round(total, 3),
            "sources": st["sources"][:20],  # cap
            "flags": sorted({f for s in st["sources"] for f in s.get("flags", [])}),
        },
    }


def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser(description="Train heuristic overrides from aggregated data")
    ap.add_argument("aggregation", nargs="?", default=None,
                    help="Aggregation JSON (from rj_bench_aggregate.py)")
    ap.add_argument("--from-results", action="store_true",
                    help="Run aggregation inline from bench/results/")
    ap.add_argument("--target", default=None, help="Target raidjit SHA")
    ap.add_argument("--hashes", default=None, help="Hashdump output JSON")
    ap.add_argument("--target-csv", type=int, default=None)
    ap.add_argument("--uarch", required=True, help="Target uarch (e.g. zen4)")
    ap.add_argument("--out", required=True, help="Output table JSON")
    args = ap.parse_args()

    if args.from_results:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rj_bench_aggregate import load_results, aggregate, _load_machines
        from classify import _load_annotations
        results, _ = load_results()
        hashes = {}
        if args.hashes:
            with open(args.hashes, encoding="utf-8") as f:
                hashes = json.load(f).get("hashes", {})
        agg = aggregate(results, args.target, _load_annotations(),
                        hashes, _load_machines(), args.target_csv)
    elif args.aggregation:
        with open(args.aggregation, encoding="utf-8") as f:
            agg = json.load(f)
    else:
        ap.error("need aggregation JSON or --from-results")

    table = train(agg, args.uarch, args.target, args.target_csv)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}: {len(table['overrides'])} overrides, "
          f"{len(table['not_overridden'])} not overridden")


if __name__ == "__main__":
    sys.exit(main())
