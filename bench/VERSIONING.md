# Benchmark facility: version-tolerance policy

This facility treats **version tolerance as a first-class requirement**.
Benchmark results arrive from different raidjit versions, over long periods,
from machines we don't control. The design below is what keeps old data
interpretable and new tools from breaking on it.


> **See also:** [STALENESS.md](STALENESS.md) — the staleness model for merging
> version-skewed data. Versioning tells you *what changed*; staleness tells you
> *how much to trust old measurements given the change*. Quarantine is the last
> resort, not the default: the facility prefers careful merging with explicit
> uncertainty over discarding data.

## 1. The four version axes

A result file carries four independent version numbers in `provenance`,
because the things that can drift do so independently:

| Axis | What it versions | Comparability rule |
|------|------------------|--------------------|
| `schema_version` | This JSON layout (`bench/schema.json`) | Ingestion migrates v1→current; unknown future versions are **quarantined**, never crashed on |
| `raidjit_version` | Git SHA of the producing tree | Informational anchor; never used for equality, only for archaeology |
| `planid_format_version` | The plan-ID grammar | `plan_id` strings are comparable **only** when equal. Mismatch → the `plan_id` field is quarantined (treated as opaque label), other fields may still aggregate |
| `static_model_version` | The static cost model (uops/byte, mem/byte) | Static figures comparable **only** when equal. Mismatch → static-vs-measured analysis skips the pair, records a `static_drift` note |
| `twin_set_version` | The `RJ_TWIN_*` enumeration | Twin names comparable **only** when equal (see §3). Mismatch → twin-level aggregation skips, file-level aggregation (topology/op/ISA) may continue |
| `bench_protocol_version` | The `bench_select` measurement protocol | Measured GB/s comparable **only** when protocol version **and** `sizes_bytes` match |

The key insight: **comparability is per-axis, not per-file.** A result from an
old raidjit with a new schema is still fully usable. A result with a drifted
twin set can still contribute to ISA-level (not twin-level) analysis. We
degrade gracefully, axis by axis, instead of rejecting whole files.

## 2. Breaking vs additive changes

### Schema (`schema_version`) — additive by default
- **Additive (minor):** new optional field, new optional enum value, new
  optional object. Old files validate unchanged. New tools ignore unknown
  fields (`additionalProperties: true` everywhere).
- **Breaking (major):** renaming/removing a required field, changing a
  field's type, changing the meaning of an existing value. Requires a
  `schema_version` bump **and** a migration function in
  `tools/bench/migrate.py`.

### Twin set (`twin_set_version`)
- **Minor:** a twin is **added**. Old results simply never mention it.
- **Major:** a twin is **removed or renamed**, or a twin's *meaning* changes
  (e.g. `xorbit` restricted to non-x86 ISAs changes what "xorbit wins" means).
  Old results naming that twin are quarantined from twin-level aggregation
  until a human maps them.

### Plan-ID format (`planid_format_version`)
- **Minor:** new optional token appended.
- **Major:** token reordering, token removal, grammar change. Old `plan_id`
  strings become opaque labels.

### Static model (`static_model_version`)
- **Any change to the uop tables, port model, or cost formula** is at least
  minor. Old static figures are never "wrong" — they're just from a different
  model, and cross-model comparison is meaningless.

### Bench protocol (`bench_protocol_version`)
- **Minor:** more iterations, tighter noise rule (results get *better*, still
  comparable with a note).
- **Major:** different working-set sizes, different timing method, different
  winner rule. Old measurements can't be ranked against new ones.

## 3. Ingestion behavior (migrate → tag → quarantine only as last resort, never crash)

`tools/bench/rj_bench_import.py` processes each submitted file through:

1. **Parse**: invalid JSON → reject with a clear error (not a traceback).
2. **Schema version check**:
   - `schema_version` ≤ current and a migration path exists → **migrate forward**,
     record `migrated_from: N` in the stored copy.
   - `schema_version` > current (from the future) → **quarantine** to
     `bench/incoming/quarantine/` with a `quarantine_reason`. Never crash.
   - `schema_version` ≤ current but no migration path → quarantine as legacy.
3. **Machine check**: `machine_id` must match `bench/results/<id>/machine.json`
   (or be registered with `--new-machine`). Unknown machine → quarantine.
4. **Sanity checks** (see README): out-of-range values → quarantine with reason.
5. **Axis tagging**: the stored copy gets a sidecar `_compat.json` recording
   which axes match current versions, so aggregation can filter without
   re-deriving.
6. **Staleness pre-computation**: ingestion computes the initial `code_hash`
   index (see STALENESS.md §1). It does NOT decide trust — that's the
   aggregation/training job. Ingestion only ensures the data needed for
   staleness decisions (hashes, versions, provenance) is present and valid.

Quarantine is **not** deletion. Quarantined files stay in the repo (or the
submitter's outbox) with a machine-readable reason, so a future migration or
human review can rescue them.

## 4. Aggregation behavior (warn and skip)

`tools/bench/rj_bench_aggregate.py`:

- Never assumes field presence: uses `.get()` with defaults everywhere.
- Unknown `twin` names, `isa` values, or `op` values → warning on stderr,
  excluded from that aggregation level, counted in the report's `skipped`
  section.
- Cross-version joins check the relevant axis first:
  - twin win-rates: require equal `twin_set_version`
  - static-vs-measured: require equal `static_model_version`
  - plan-ID matching: require equal `planid_format_version`
  - GB/s ranking: require equal `bench_protocol_version` + `sizes_bytes`
- Every report ends with a **comparability ledger**: how many results were
  included/excluded per axis and why. Silence about exclusions is a bug.

## 5. Re-interpreting old results

When a new raidjit version changes semantics (e.g. a twin's meaning), old
results are **not rewritten**. Instead:

1. The version axis is bumped (e.g. `twin_set_version` 1→2).
2. If a mechanical mapping exists (rename `foo`→`bar`), it's encoded as a
   migration in `tools/bench/migrate.py` and applied at **import** time,
   with `migrated_from` recorded.
3. If no mechanical mapping exists, old results are quarantined from the
   affected aggregation level. A human can later write a mapping and
   re-import.

History is append-only. We never mutate a stored result in place; migration
produces a new stored copy with provenance recording the migration.

## 6. Current versions (update on every bump)

| Axis | Current | Last changed | Reason |
|------|---------|--------------|--------|
| `schema_version` | 1 | 2026-10-11 | Initial schema |
| `planid_format_version` | 1 | 2026-10-11 | Initial |
| `static_model_version` | 1 | 2026-10-11 | Initial (ports.c uop tables) |
| `twin_set_version` | 1 | 2026-10-11 | 12 twins (nt..rec_cf) |
| `bench_protocol_version` | 1 | 2026-10-11 | bench_select as of 2026-10 |
| `format_version` | `bench-result/1` | 2026-10-11 | Initial |

When any axis bumps, update this table **and** add the migration (or
quarantine rule) to `tools/bench/migrate.py`. A version bump without a
migration entry is a CI failure (see `tests/test_bench_facility.py`).
