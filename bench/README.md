# raidjit offline benchmark facility

Out-of-band, offline, in-repo storage and processing of benchmark results
from different machines (potentially user-submitted), used to train heuristic
selection for the planner.

**This facility never runs benchmarks.** (Benchmarking is postponed per the
2026-10-06 directive.) It stores, validates, aggregates, and learns from
results produced elsewhere.

## Layout

```
bench/
  README.md              # this file
  VERSIONING.md          # version-tolerance policy (read this first)
  STALENESS.md           # staleness model: trusting old data carefully
  schema.json            # JSON Schema for result files
  machine-schema.json    # JSON Schema for machine descriptors
  results/               # curated, committed benchmark data
    <machine-id>/
      machine.json       # machine descriptor (CPU, uarch, OS, compiler)
      <ts>-<topo>-<op>.json        # one benchmark run (comparison set)
      <ts>-<topo>-<op>.json.compat.json  # sidecar: version axes, warnings
  incoming/
    quarantine/          # files that can't be used yet (with reasons)
  config_space/
    VERSIONS.md          # what was legal when (twin x ISA x kind)
  annotations/           # manual change classifications (see STALENESS.md §5)
  heuristics/
    TRAINING.md          # heuristic-training output format (design)
    <uarch>.json         # trained override tables (generated, committed)
tools/bench/
  rj_bench_import.py     # validate + import submitted results
  rj_bench_aggregate.py  # aggregate with staleness model
  classify.py            # change classification (levels 0-4)
  freshness.py           # per-candidate freshness, drift detection
  migrate.py             # schema migration registry
```

## Quick start

```bash
# Import a submitted result file
python3 tools/bench/rj_bench_import.py results-from-zen4.json

# Import and register a new machine
python3 tools/bench/rj_bench_import.py --new-machine results-newbox.json
# then fill in bench/results/newbox/machine.json

# Aggregate everything (JSON report)
python3 tools/bench/rj_bench_aggregate.py --format json --out report.json

# Aggregate for a specific target version (staleness-aware)
python3 tools/bench/rj_bench_aggregate.py --target abc1234 --format md
```

## Key concepts

### 1. A result is a comparison set, not a number

Each result file holds one benchmark run: N candidates (different twins,
ISAs, unrolls) measured together under identical conditions for one
(topology, op). The pairwise rankings within a set are the primary signal —
more robust than absolute GB/s across machines.

### 2. Kernel identity is (config × version × code_hash)

The `code_hash` (SHA-256 of the generated machine code) is the ground truth
for "same kernel." Same config + different hash = the codegen changed; the
measurements are not of the same thing. Same config + same version +
different hash = nondeterminism **bug** (the aggregator flags it).

### 3. Version tolerance is per-axis, not per-file

Six independent version axes (`VERSIONING.md`): schema, raidjit SHA, plan-ID
format, static model, twin set, config space, bench protocol. Comparability
is decided per-axis: a result with a drifted twin set can still contribute
to ISA-level analysis. We degrade gracefully, never reject whole files
without reason.

### 4. Staleness: merge carefully, don't quarantine by default

Old data is valuable and sparse. The [staleness model](STALENESS.md) weights
old measurements by **what actually changed** (code-hash diff, classified
0–4), not by wall-clock age. Partial inference keeps fresh candidates'
signal even when siblings went stale. Every inferred value carries explicit
uncertainty (sources, weights, assumptions, flags).

### 5. Heuristics are data, not code

The trainer (future) outputs JSON override tables per uarch
(`bench/heuristics/<uarch>.json`). The planner applies them as score
adjustments, never forced selections. Empty table = static model everywhere.
See [TRAINING.md](heuristics/TRAINING.md).

## Result file format (summary)

```json
{
  "schema_version": 1,
  "format_version": "bench-result/1",
  "machine_id": "zen4-baremetal",
  "timestamp": "2026-10-15T12:00:00Z",
  "provenance": {
    "raidjit_version": "abc1234...",
    "planid_format_version": 1,
    "static_model_version": 1,
    "twin_set_version": 1,
    "config_space_version": 1,
    "bench_protocol_version": 1
  },
  "topology": "rs(6,3)",
  "op": "encode",
  "protocol": {
    "name": "bench_select", "version": 1,
    "warmup_iters": 2, "timed_iters": 11,
    "sizes_bytes": [16384, 65536, 262144],
    "noise_rule": "max(1.5%, pooled_noise)"
  },
  "candidates": [
    {
      "plan_id": "rs(6,3):encode:horner:avx512:gfni:U4",
      "twins": ["ternlog"],
      "isa": "avx512", "mul": "gfni", "u": 4,
      "code_hash": "deadbeef...",
      "code_size": 8192,
      "config_space_version": 1,
      "static": {"uops_per_byte": 0.42, "mem_per_byte": 2.1},
      "measured": {
        "gbps_median": [107.5, 98.2, 58.8],
        "gbps_iqr": [1.2, 0.9, 2.1],
        "noise_window_pct": 1.5
      },
      "selected": true
    }
  ]
}
```

Full field documentation: `bench/schema.json` (JSON Schema, authoritative).

## Import pipeline

1. **Parse** — invalid JSON rejected (clear error, no traceback).
2. **Schema version** — migrate forward via `migrate.py`, or quarantine
   (future versions, no migration path).
3. **Machine** — must be registered in `bench/results/<id>/machine.json`.
4. **Sanity** — ranges, consistency, required fields. Errors quarantine;
   warnings are recorded in the sidecar.
5. **Dedup** — same (machine, timestamp, topology, op) refused unless `--force`.
6. **Store** — to `bench/results/<id>/` with a `.compat.json` sidecar.

Quarantine is not deletion: files stay in `bench/incoming/quarantine/` with
a machine-readable reason, rescuable by future migrations or human review.

## Empty state

Zero benchmark data is valid. All tools handle it:
- `rj_bench_import.py` with no files: usage error (not a crash).
- `rj_bench_aggregate.py` with no results: report with `"note": "empty state"`.
- Planner with no heuristic tables: static model everywhere (unchanged behavior).

## Version bumping checklist

When anything versioned changes:
- [ ] Bump the axis in `bench/VERSIONING.md` (table in §6).
- [ ] Schema bump: add migration to `tools/bench/migrate.py`.
- [ ] Config-space bump: add section to `bench/config_space/VERSIONS.md`.
- [ ] Twin-set bump: note added/removed/renamed twins and their impact.
- [ ] Consider a manual annotation (`bench/annotations/`) if the automatic
      classifier would get the transition wrong.

A version bump without the corresponding entry is a test failure
(`tests/test_bench_facility.py`).

## Security: untrusted input

Result files may be user-submitted. The facility assumes adversarial input:
- JSON parsing never evals; deeply nested structures are bounded by Python's
  recursion limit (caught as Reject, not a crash).
- No shell invocation with file-derived data; filenames are sanitized
  (`re.sub(r"[^a-zA-Z0-9()+_-]", "_", ...)`).
- Numeric fields are range-checked; strings are length-bounded by schema.
- The importer never writes outside `bench/` (paths are constructed, not
  taken from input).
- Machine registration (`--new-machine`) creates a stub; a human fills it.

## What this is not

- Not a benchmark runner (no timing code here).
- Not a prediction engine (no extrapolation to unmeasured configs).
- Not real-time (training runs are explicit, versioned, reproducible).
- Not a replacement for the static model (heuristics are overrides on top).
