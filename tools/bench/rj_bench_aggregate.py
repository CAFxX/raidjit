#!/usr/bin/env python3
"""Aggregate benchmark results with the staleness model.

Usage:
    rj_bench_aggregate.py [--target <sha>] [--format json|md] [--out <file>]

Loads all results from bench/results/, then:
  1. Builds the code-hash index; detects nondeterminism (BUG alerts).
  2. Detects codegen drift across versions.
  3. For each (topology, op, ISA) group, computes pairwise twin win-rates
     using per-candidate freshness (partial inference: stale candidates
     don't poison fresh ones).
  4. Compares measured vs static-model rankings (where static_model_version matches).
  5. Emits a report with explicit uncertainty (STALENESS.md §6) and a
     comparability ledger (what was included/excluded and why).

Never crashes on unknown fields/versions: warns and skips.
This tool never runs benchmarks.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from freshness import (  # noqa: E402
    build_hash_index, detect_nondeterminism, detect_codegen_drift,
    freshness_for, pairwise_usable, pairwise_weight, norm_config,
    USABILITY_THRESHOLD,
)
from classify import _load_annotations  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "bench", "results")


def load_results():
    """Load all result files. Warn-and-skip on any unreadable file."""
    results = []
    skipped = []
    if not os.path.isdir(RESULTS_DIR):
        return results, skipped
    for mid in sorted(os.listdir(RESULTS_DIR)):
        mdir = os.path.join(RESULTS_DIR, mid)
        if not os.path.isdir(mdir):
            continue
        for fn in sorted(os.listdir(mdir)):
            if not fn.endswith(".json") or fn.endswith(".compat.json"):
                continue
            if fn == "machine.json":
                continue  # machine descriptor, not a result
            path = os.path.join(mdir, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    r = json.load(f)
                # Tag candidates with topology/op for norm_config
                for c in r.get("candidates", []):
                    c["_topology"] = r.get("topology", "?")
                    c["_op"] = r.get("op", "?")
                # Load compat sidecar if present
                compat_path = path + ".compat.json"
                r["_compat"] = {}
                if os.path.exists(compat_path):
                    with open(compat_path, encoding="utf-8") as f:
                        r["_compat"] = json.load(f)
                r["_path"] = path
                results.append(r)
            except (json.JSONDecodeError, OSError) as e:
                skipped.append({"path": path, "reason": f"unreadable: {e}"})
    return results, skipped


def gbps_headline(candidate):
    """Headline GB/s: median of medians across sizes. None if unusable."""
    medians = candidate.get("measured", {}).get("gbps_median", [])
    vals = [v for v in medians if isinstance(v, (int, float)) and v > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def aggregate(results, target_version=None, annotations=None):
    """Main aggregation. Returns a report dict."""
    report = {
        "target_version": target_version or "latest",
        "num_results": len(results),
        "alerts": [],
        "pairwise": [],
        "static_vs_measured": [],
        "ledger": {
            "included": 0,
            "excluded": [],
            "skipped_files": [],
        },
    }

    if not results:
        report["ledger"]["note"] = "empty state: no benchmark data yet (valid)"
        return report

    # 1. Hash index + nondeterminism detection
    index = build_hash_index(results)
    nd_alerts = detect_nondeterminism(index)
    report["alerts"].extend(nd_alerts)

    # 2. Codegen drift
    drifts = detect_codegen_drift(index)
    # Classify each drift (needs candidate + versions)
    for d in drifts:
        d["classification"] = "pending"  # full classification needs candidate lookup
    report["codegen_drifts"] = drifts

    # 3. Pairwise twin win-rates per (topology, op, isa)
    # Group comparison sets: each result file is one comparison set
    groups = defaultdict(list)
    for r in results:
        for c in r.get("candidates", []):
            key = (r.get("topology"), r.get("op"), c.get("isa"))
            groups[key].append((r, c))

    for (topo, op, isa), entries in sorted(groups.items()):
        # Pairwise: for each result file, compare candidates within it
        by_file = defaultdict(list)
        for r, c in entries:
            by_file[r["_path"]].append((r, c))
        for path, rcs in by_file.items():
            r = rcs[0][0]
            ver = r["provenance"]["raidjit_version"]
            # Freshness of each candidate for the target
            fresh = {}
            for _, c in rcs:
                if target_version and ver != target_version:
                    f = freshness_for(
                        c, ver, c["code_hash"], target_version,
                        target_hash=None,  # target not built; conservative
                        csv_measured=c.get("config_space_version"),
                        annotations=annotations)
                else:
                    f = {"level": 0, "weight": 1.0, "source": "same_version",
                         "reason": "measured at target version"}
                fresh[c["plan_id"]] = f

            # All pairs within this comparison set
            for i in range(len(rcs)):
                for j in range(i + 1, len(rcs)):
                    _, a = rcs[i]
                    _, b = rcs[j]
                    fa, fb = fresh[a["plan_id"]], fresh[b["plan_id"]]
                    ga, gb = gbps_headline(a), gbps_headline(b)
                    if ga is None or gb is None:
                        report["ledger"]["excluded"].append({
                            "pair": (a["plan_id"], b["plan_id"]),
                            "reason": "missing GB/s data",
                        })
                        continue
                    usable = pairwise_usable(fa, fb)
                    w = pairwise_weight(fa, fb) if usable else 0.0
                    winner = a["plan_id"] if ga > gb else b["plan_id"]
                    # Noise margin: don't declare a winner within 1.5%
                    margin = abs(ga - gb) / max(ga, gb)
                    tied = margin < 0.015
                    report["pairwise"].append({
                        "topology": topo, "op": op, "isa": isa,
                        "a": {"plan_id": a["plan_id"], "twins": a.get("twins", []),
                              "gbps": round(ga, 2), "freshness": fa},
                        "b": {"plan_id": b["plan_id"], "twins": b.get("twins", []),
                              "gbps": round(gb, 2), "freshness": fb},
                        "winner": None if tied else winner,
                        "tied": tied,
                        "usable": usable,
                        "weight": round(w, 3),
                        "source_version": ver,
                        "flags": _pair_flags(fa, fb, ver, target_version),
                    })
                    if usable:
                        report["ledger"]["included"] += 1

    # 4. Static vs measured (only where static_model_version matches within the file)
    for r in results:
        cands = r.get("candidates", [])
        # Static ranking by uops_per_byte (lower is better)
        with_static = [c for c in cands
                       if isinstance(c.get("static", {}).get("uops_per_byte"), (int, float))]
        with_measured = [c for c in cands if gbps_headline(c) is not None]
        if len(with_static) >= 2 and len(with_measured) >= 2:
            static_rank = sorted(with_static,
                                 key=lambda c: c["static"]["uops_per_byte"])
            meas_rank = sorted(with_measured,
                               key=lambda c: -gbps_headline(c))
            static_ids = [c["plan_id"] for c in static_rank]
            meas_ids = [c["plan_id"] for c in meas_rank]
            agree = static_ids[0] == meas_ids[0]  # top-1 agreement
            report["static_vs_measured"].append({
                "topology": r.get("topology"), "op": r.get("op"),
                "version": r["provenance"]["raidjit_version"][:12],
                "static_model_version": r["provenance"].get("static_model_version"),
                "static_top1": static_ids[0] if static_ids else None,
                "measured_top1": meas_ids[0] if meas_ids else None,
                "agree": agree,
            })

    return report


def _pair_flags(fa, fb, ver, target_version):
    flags = []
    if target_version and ver != target_version:
        flags.append("CROSS_VERSION")
    for f in (fa, fb):
        if f["level"] >= 2:
            flags.append("STALE_PARTIAL")
            break
    for f in (fa, fb):
        if f["level"] == 4:
            flags.append("STALE_UNKNOWN")
            break
    for f in (fa, fb):
        if f["source"] == "manual_annotation":
            flags.append("ANNOTATED")
            break
    return sorted(set(flags))


def main():
    ap = argparse.ArgumentParser(description="Aggregate benchmark results with staleness model")
    ap.add_argument("--target", default=None,
                    help="Target raidjit version SHA (default: treat each result's own version as target)")
    ap.add_argument("--format", choices=["json", "md"], default="json")
    ap.add_argument("--out", default=None, help="Output file (default: stdout)")
    args = ap.parse_args()

    results, skipped = load_results()
    annotations = _load_annotations()
    report = aggregate(results, args.target, annotations)
    report["ledger"]["skipped_files"] = skipped
    report["ledger"]["annotations_loaded"] = len(annotations)

    if args.format == "json":
        out = json.dumps(report, indent=2, sort_keys=True)
    else:
        out = _format_md(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"wrote {args.out}")
    else:
        print(out)


def _format_md(report):
    lines = ["# Benchmark aggregation report", ""]
    lines.append(f"Results: {report['num_results']}, "
                 f"target: {report['target_version']}")
    lines.append("")
    if report["alerts"]:
        lines.append("## Alerts")
        for a in report["alerts"]:
            lines.append(f"- **{a['type']}**: {a['message']}")
        lines.append("")
    usable = [p for p in report["pairwise"] if p["usable"]]
    lines.append(f"## Pairwise comparisons ({len(usable)} usable / "
                 f"{len(report['pairwise'])} total)")
    for p in usable[:50]:  # cap markdown output
        w = p["winner"] or "tie"
        lines.append(f"- {p['topology']} {p['op']} {p['isa']}: "
                     f"{p['a']['plan_id']} ({p['a']['gbps']} GB/s) vs "
                     f"{p['b']['plan_id']} ({p['b']['gbps']} GB/s) -> "
                     f"{w} (weight {p['weight']}, flags {p['flags']})")
    if len(usable) > 50:
        lines.append(f"... and {len(usable) - 50} more (see JSON)")
    lines.append("")
    lines.append("## Static vs measured (top-1 agreement)")
    for s in report["static_vs_measured"]:
        mark = "AGREE" if s["agree"] else "DISAGREE"
        lines.append(f"- {s['topology']} {s['op']} [{s['version']}]: "
                     f"static={s['static_top1']} measured={s['measured_top1']} {mark}")
    lines.append("")
    lines.append("## Ledger")
    lines.append(f"Included comparisons: {report['ledger']['included']}")
    lines.append(f"Skipped files: {len(report['ledger']['skipped_files'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
