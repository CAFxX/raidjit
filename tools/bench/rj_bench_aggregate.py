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


def _safe_provenance(r):
    """CRITICAL-2 fix: defensive provenance access (never KeyError)."""
    p = r.get("provenance")
    return p if isinstance(p, dict) else {}


def _protocol_compatible(r, target_protocol=None):
    """MAJOR-2 fix: per-axis comparability gate for the measurement protocol.
    Returns (compatible: bool, reason: str, match_factor: float).
    GB/s values are only comparable when bench_protocol_version AND
    sizes_bytes match. A manual protocol map could allow 0.5; none exists yet."""
    prov = _safe_provenance(r)
    pv = prov.get("bench_protocol_version")
    proto = r.get("protocol", {})
    sizes = tuple(proto.get("sizes_bytes", [])) if isinstance(proto, dict) else ()
    if target_protocol is None:
        return (True, "no target protocol constraint", 1.0)
    tpv, tsizes = target_protocol
    if pv != tpv:
        return (False, f"bench_protocol_version {pv} != target {tpv}", 0.0)
    if sizes != tsizes:
        return (False, f"sizes_bytes {sizes} != target {tsizes}", 0.0)
    return (True, "protocol matches", 1.0)


def _machine_trust(r, machines):
    """MAJOR-2 fix: read machine_trust from machine.json (default 1.0)."""
    mid = r.get("machine_id", "")
    m = machines.get(mid, {})
    t = m.get("machine_trust", 1.0)
    if not isinstance(t, (int, float)) or isinstance(t, bool) or t < 0:
        return 1.0
    return float(t)


def _is_retired(candidate, target_csv, retired_map):
    """MAJOR-3 fix: detect retired configs.
    A candidate is retired if its config_space_version < target_csv and
    the version delta retired something this candidate uses.
    retired_map: {csv_version: {'retired_twins': [...], 'retired_isa': [...]}}.
    Returns (is_retired: bool, reason: str)."""
    c_csv = candidate.get("config_space_version")
    if not isinstance(c_csv, int) or not isinstance(target_csv, int):
        return (False, "")
    if c_csv >= target_csv:
        return (False, "")
    # Check each intermediate version for retirements affecting this candidate
    for v in range(c_csv + 1, target_csv + 1):
        retired = retired_map.get(v, {})
        for t in candidate.get("twins", []):
            if t in retired.get("retired_twins", []):
                return (True, f"twin {t!r} retired in config_space v{v}")
        isa = candidate.get("isa", "")
        if isa in retired.get("retired_isa", []):
            return (True, f"isa {isa!r} retired in config_space v{v}")
    return (False, "")


def _load_retired_map():
    """Load machine-readable retirements from bench/config_space/.
    Returns {version: {'retired_twins': [...], 'retired_isa': [...]}}.
    Currently the registry is prose-only (VERSIONS.md); this returns the
    hardcoded v1->v2 transition until the machine-readable form exists."""
    # TODO: parse bench/config_space/vN.json when it exists.
    # Hardcoded known transition: xorbit x86 restriction (config_space v1->v2).
    # This is the only retirement known at facility creation time.
    return {
        2: {
            "retired_twins": [],
            # xorbit not retired, but restricted: handled as structural change,
            # not retirement. True retirements go here.
            "retired_isa": [],
        },
    }


def aggregate(results, target_version=None, annotations=None, hashes=None,
              machines=None, target_csv=None):
    """Main aggregation. Returns a report dict.

    Args:
        results: list of result dicts
        target_version: raidjit SHA to evaluate freshness against (None = each
            result's own version, i.e. no cross-version inference)
        annotations: preloaded manual annotations
        hashes: {plan_id: {'code_hash': ..., 'status': ...}} from
            rj_bench_hashdump.py for the target version (CRITICAL-1 fix:
            makes graded freshness reachable)
        machines: {machine_id: machine.json dict} for machine_trust
        target_csv: target config_space_version (for retired detection)
    """
    machines = machines or {}
    hashes = hashes or {}
    retired_map = _load_retired_map()
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
            # MAJOR-2 fix: per-axis comparability accounting
            "per_axis": {
                "protocol_excluded": 0,
                "retired_excluded": 0,
                "stale_excluded": 0,
                "missing_data_excluded": 0,
            },
        },
    }

    if not results:
        report["ledger"]["note"] = "empty state: no benchmark data yet (valid)"
        return report

    # 1. Hash index + nondeterminism detection (defensive: skip malformed)
    try:
        index = build_hash_index(results)
    except (KeyError, TypeError, AttributeError) as e:
        report["alerts"].append({
            "type": "INDEX_ERROR",
            "severity": "ERROR",
            "message": f"hash index build failed (malformed results): {e}",
        })
        index = {}
    nd_alerts = detect_nondeterminism(index)
    report["alerts"].extend(nd_alerts)

    # 2. Codegen drift (classified where possible)
    drifts = detect_codegen_drift(index)
    report["codegen_drifts"] = drifts

    # Determine target protocol from the newest result (if any)
    target_protocol = None
    if results:
        newest = max(results,
                     key=lambda r: r.get("timestamp", ""),
                     default=None)
        if newest:
            prov = _safe_provenance(newest)
            proto = newest.get("protocol", {})
            sizes = tuple(proto.get("sizes_bytes", [])) if isinstance(proto, dict) else ()
            target_protocol = (prov.get("bench_protocol_version"), sizes)

    # 3. Pairwise comparisons (defensive throughout)
    groups = defaultdict(list)
    for r in results:
        cands = r.get("candidates")
        if not isinstance(cands, list):
            report["ledger"]["excluded"].append({
                "file": r.get("_path", "?"),
                "reason": "candidates is not a list",
            })
            report["ledger"]["per_axis"]["missing_data_excluded"] += 1
            continue
        for c in r.get("candidates", []):
            if not isinstance(c, dict):
                continue
            key = (r.get("topology"), r.get("op"), c.get("isa"))
            groups[key].append((r, c))

    for (topo, op, isa), entries in sorted(groups.items(), key=lambda x: str(x[0])):
        by_file = defaultdict(list)
        for r, c in entries:
            by_file[r.get("_path", "?")].append((r, c))
        for path, rcs in by_file.items():
            r = rcs[0][0]
            prov = _safe_provenance(r)
            ver = prov.get("raidjit_version", "unknown")

            # MAJOR-2: protocol gate per result file
            proto_ok, proto_reason, proto_factor = _protocol_compatible(r, target_protocol)
            trust = _machine_trust(r, machines)

            # Freshness per candidate (CRITICAL-1: use hashes when available)
            fresh = {}
            for _, c in rcs:
                pid = c.get("plan_id", "?")
                c_hash = c.get("code_hash")
                if target_version and ver != target_version:
                    # Look up target hash from hashdump output
                    hinfo = hashes.get(pid, {})
                    target_hash = hinfo.get("code_hash") if hinfo.get("status") in ("ok", "mismatch") else None
                    if hinfo.get("status") == "rejected":
                        # Config retired or rejected at target: freshness 0
                        f = {"level": 3, "weight": 0.0, "source": "target_rejected",
                             "reason": f"target rejects this config: {hinfo.get('message', '')}"}
                    else:
                        f = freshness_for(
                            c, ver, c_hash, target_version,
                            target_hash=target_hash,
                            csv_measured=c.get("config_space_version"),
                            csv_target=target_csv,
                            annotations=annotations)
                else:
                    f = {"level": 0, "weight": 1.0, "source": "same_version",
                         "reason": "measured at target version"}
                # MAJOR-3: retired check
                if target_csv is not None:
                    retired, rreason = _is_retired(c, target_csv, retired_map)
                    if retired:
                        f = {"level": 3, "weight": 0.0, "source": "retired",
                             "reason": rreason}
                        f["_retired"] = True
                fresh[pid] = f

            # All pairs within this comparison set
            for i in range(len(rcs)):
                for j in range(i + 1, len(rcs)):
                    _, a = rcs[i]
                    _, b = rcs[j]
                    a_pid = a.get("plan_id", "?")
                    b_pid = b.get("plan_id", "?")
                    fa, fb = fresh[a_pid], fresh[b_pid]
                    ga, gb = gbps_headline(a), gbps_headline(b)
                    if ga is None or gb is None:
                        report["ledger"]["excluded"].append({
                            "pair": (a_pid, b_pid),
                            "reason": "missing GB/s data",
                        })
                        report["ledger"]["per_axis"]["missing_data_excluded"] += 1
                        continue
                    # MAJOR-2: protocol gate
                    if not proto_ok:
                        report["ledger"]["excluded"].append({
                            "pair": (a_pid, b_pid),
                            "reason": f"protocol incompatible: {proto_reason}",
                        })
                        report["ledger"]["per_axis"]["protocol_excluded"] += 1
                        continue
                    # MAJOR-3: retired configs excluded from training signal
                    if fa.get("_retired") or fb.get("_retired"):
                        report["ledger"]["excluded"].append({
                            "pair": (a_pid, b_pid),
                            "reason": "retired config",
                        })
                        report["ledger"]["per_axis"]["retired_excluded"] += 1
                        # Still record the pair (marked) for historical visibility
                        usable, w = False, 0.0
                    else:
                        usable = pairwise_usable(fa, fb)
                        w = pairwise_weight(fa, fb, proto_factor, trust) if usable else 0.0
                        if not usable:
                            report["ledger"]["per_axis"]["stale_excluded"] += 1
                    winner = a_pid if ga > gb else b_pid
                    # Noise margin: use the result's own noise_window_pct if
                    # available, else default 1.5% (MAJOR-6 fix: was hardcoded)
                    noise_pct = 1.5
                    try:
                        nw = a.get("measured", {}).get("noise_window_pct")
                        if isinstance(nw, (int, float)) and 0 < nw < 50:
                            noise_pct = float(nw)
                    except (AttributeError, TypeError):
                        pass
                    margin = abs(ga - gb) / max(ga, gb)
                    tied = margin < noise_pct / 100.0
                    report["pairwise"].append({
                        "topology": topo, "op": op, "isa": isa,
                        "a": {"plan_id": a_pid, "twins": a.get("twins", []),
                              "gbps": round(ga, 2), "freshness": _clean_fresh(fa)},
                        "b": {"plan_id": b_pid, "twins": b.get("twins", []),
                              "gbps": round(gb, 2), "freshness": _clean_fresh(fb)},
                        "winner": None if tied else winner,
                        "tied": tied,
                        "usable": usable,
                        "weight": round(w, 3),
                        "source_version": ver,
                        "protocol_factor": proto_factor,
                        "machine_trust": trust,
                        "flags": _pair_flags(fa, fb, ver, target_version),
                    })
                    if usable:
                        report["ledger"]["included"] += 1

    # 4. Static vs measured (defensive)
    for r in results:
        cands = r.get("candidates")
        if not isinstance(cands, list):
            continue
        prov = _safe_provenance(r)
        with_static = [c for c in cands
                       if isinstance(c.get("static", {}).get("uops_per_byte"), (int, float))
                       and not isinstance(c["static"]["uops_per_byte"], bool)]
        with_measured = [c for c in cands if gbps_headline(c) is not None]
        if len(with_static) >= 2 and len(with_measured) >= 2:
            static_rank = sorted(with_static,
                                 key=lambda c: c["static"]["uops_per_byte"])
            meas_rank = sorted(with_measured,
                               key=lambda c: -gbps_headline(c))
            static_ids = [c.get("plan_id", "?") for c in static_rank]
            meas_ids = [c.get("plan_id", "?") for c in meas_rank]
            agree = static_ids[0] == meas_ids[0]
            ver = prov.get("raidjit_version", "unknown")
            report["static_vs_measured"].append({
                "topology": r.get("topology"), "op": r.get("op"),
                "version": ver[:12] if isinstance(ver, str) else "?",
                "static_model_version": prov.get("static_model_version"),
                "static_top1": static_ids[0] if static_ids else None,
                "measured_top1": meas_ids[0] if meas_ids else None,
                "agree": agree,
            })

    return report


def _clean_fresh(f):
    """Remove internal keys before serializing."""
    return {k: v for k, v in f.items() if not k.startswith("_")}

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
        if f.get("_retired"):
            flags.append("RETIRED_CONFIG")
            break
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


def _load_machines():
    """Load machine.json for all machines (for machine_trust)."""
    machines = {}
    if not os.path.isdir(RESULTS_DIR):
        return machines
    for mid in os.listdir(RESULTS_DIR):
        mp = os.path.join(RESULTS_DIR, mid, "machine.json")
        if os.path.exists(mp):
            try:
                with open(mp, encoding="utf-8") as f:
                    machines[mid] = json.load(f)
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    return machines


def main():
    ap = argparse.ArgumentParser(description="Aggregate benchmark results with staleness model")
    ap.add_argument("--target", default=None,
                    help="Target raidjit version SHA (default: treat each result's own version as target)")
    ap.add_argument("--hashes", default=None,
                    help="Hashdump output JSON (from rj_bench_hashdump.py) for target-version code hashes. "
                         "Enables graded freshness (levels 1-3); without it, cross-version pairs are level 4.")
    ap.add_argument("--target-csv", type=int, default=None,
                    help="Target config_space_version (for retired-config detection)")
    ap.add_argument("--format", choices=["json", "md"], default="json")
    ap.add_argument("--out", default=None, help="Output file (default: stdout)")
    args = ap.parse_args()

    results, skipped = load_results()
    annotations = _load_annotations()
    machines = _load_machines()
    hashes = {}
    if args.hashes:
        try:
            with open(args.hashes, encoding="utf-8") as f:
                hdata = json.load(f)
            hashes = hdata.get("hashes", {})
        except (json.JSONDecodeError, OSError, ValueError) as e:
            print(f"warning: cannot load hashes file: {e}", file=sys.stderr)
    report = aggregate(results, args.target, annotations, hashes, machines, args.target_csv)
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
