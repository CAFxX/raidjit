#!/usr/bin/env python3
"""Hash oracle wrapper: materialize (config -> code_hash) for the current checkout.

Usage:
    rj_bench_hashdump.py <result.json> [...] [--out hashes.json] [--hashdump-bin <path>]

For each candidate in each result file, invokes the hashdump C binary to
recompile the kernel from (topology, op, plan_id) and hash the machine code.
Outputs a JSON mapping: {plan_id: {"code_hash": ..., "status": ...}}.

Status values:
  - "ok": hash computed successfully
  - "rejected": config cleanly rejected by current version (e.g. retired twin)
  - "error": reconstruction failed (unsupported topology, etc.)
  - "mismatch": hash computed but differs from result file's recorded hash
    (indicates the result file's hash was for a different version — expected
    for old results; the new hash is authoritative for the current checkout)

This is NOT benchmarking: it only compiles, never times.
The C binary must be built first (see tools/bench/hashdump.c).
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BIN = os.path.join(REPO_ROOT, "build-x86_64", "hashdump")


def load_results(paths):
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        # Skip machine.json files
        if "candidates" not in r:
            continue
        out.append((p, r))
    return out


def hash_one(binpath, topology, op, failed, plan_id, timeout=120):
    """Returns (status, hash_or_message)."""
    failed_s = ",".join(str(x) for x in failed) if failed else "-"
    try:
        proc = subprocess.run(
            [binpath, topology, op, failed_s, plan_id],
            capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return ("error", f"hashdump binary not found: {binpath}")
    except subprocess.TimeoutExpired:
        return ("error", "hashdump timed out")
    out = proc.stdout.strip()
    if proc.returncode == 0 and len(out) == 64 and all(c in "0123456789abcdef" for c in out):
        return ("ok", out)
    if out.startswith("ERROR:"):
        # Distinguish clean rejection from internal error by exit code
        if proc.returncode == 1:
            return ("rejected", out[6:].strip())
        return ("error", out[6:].strip())
    return ("error", f"unexpected output: {out[:200]}")


def main():
    ap = argparse.ArgumentParser(description="Materialize code hashes for current checkout")
    ap.add_argument("results", nargs="+", help="Result JSON files")
    ap.add_argument("--out", default=None, help="Output JSON (default: stdout)")
    ap.add_argument("--hashdump-bin", default=DEFAULT_BIN)
    args = ap.parse_args()

    if not os.path.exists(args.hashdump_bin):
        print(f"error: hashdump binary not found: {args.hashdump_bin}\n"
              f"build it: gcc -O2 -o {args.hashdump_bin} tools/bench/hashdump.c "
              f"build-x86_64/libraidjit.a -lcrypto -lm",
              file=sys.stderr)
        return 2

    results = load_results(args.results)
    hashes = {}
    stats = {"ok": 0, "rejected": 0, "error": 0, "mismatch": 0}
    for path, r in results:
        topo = r.get("topology", "?")
        op = r.get("op", "?")
        failed = r.get("failed", [])
        for c in r.get("candidates", []):
            pid = c.get("plan_id")
            if not pid or pid in hashes:
                continue
            status, val = hash_one(args.hashdump_bin, topo, op, failed, pid)
            recorded = c.get("code_hash")
            entry = {"status": status, "topology": topo, "op": op}
            if status == "ok":
                entry["code_hash"] = val
                if recorded and recorded != val:
                    entry["status"] = "mismatch"
                    entry["recorded_hash"] = recorded
                    entry["note"] = ("recorded hash differs: result was measured "
                                     "on different code (expected for old versions)")
                    stats["mismatch"] += 1
                else:
                    stats["ok"] += 1
            else:
                entry["message"] = val
                stats[status] += 1
            hashes[pid] = entry

    output = {
        "hashdump_version": 1,
        "checkout": _git_sha(),
        "hashes": hashes,
        "stats": stats,
    }
    out_s = json.dumps(output, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_s)
        print(f"wrote {args.out}: {stats}", file=sys.stderr)
    else:
        print(out_s)
    return 0


def _git_sha():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=REPO_ROOT)
        return proc.stdout.strip()[:12]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
