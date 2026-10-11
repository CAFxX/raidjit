# Spec: LRC locality-aware recovery planner

Status: spec (2026-10-10). Implements Carlo's directive that raidjit
offers the highest-performance implementation for each RAID-like use
case — for LRC, that means the planner must *understand* locality, not
merely stumble into it via cost-model dominance.

## 1. Problem

LRC topologies (A6: `rj_topo_lrc`, `rj_topo_lrc_azure`,
`rj_topo_lrc_optimal`) exist with docs (`docs/lrc.md`) and tests
(`tests/test_lrc.c`). The A5 survivor machinery enumerates parity-row
subsets and scores each with the exact static signal; for LRC single
failures the local-row subset strictly dominates on `(uops, mem)`, so
the pareto shortlist keeps it and the static pick prefers it.

This works **by accident of the cost model**, not by design. Fragility:

1. **Benchmark mode**: `bench_select` picks the measured cheapest. The
   measured local preference is *unverified* (benchmarks postponed per
   A6 brief). Measurement noise could flip a close call.
2. **Significance gate**: if the local subset's measured lead over the
   `is_today` anchor (a global row) is within the noise window, the
   gate falls back to today's global version — silently losing locality.
3. **Cost-model drift**: any future change to the static signal that
   narrows the local/global gap breaks the emergent preference with no
   explicit test to catch it (the red-green hook tests the *mechanism*,
   not the *guarantee*).

The planner should detect LRC structure explicitly and select the
local recovery path as a **planner rule**, not a cost-model accident.

## 2. LRC detection (from matrix structure, no topology tag needed)

A topology is LRC-detectable if its parity matrix contains rows that are:

- **Pure XOR**: every coefficient is 0 or 1, AND
- **Disjoint supports**: no data-disk column is nonzero in two such rows, AND
- **Partitioning**: every data disk is nonzero in exactly one such row.

These are the **local rows**; each defines a **local group** (its
support set). All other parity rows are **global rows**. This is
exactly the structure `test_lrc.c::matrix_structure` verifies
independently of the constructor, so detection is by structure, not by
name — it works for `rj_topo_lrc`, `rj_topo_lrc_azure`,
`rj_topo_lrc_optimal`, and any future LRC-like topology, including
user-supplied `rj_topo_custom` matrices that happen to have the shape.

Detection function (new, `src/survivor.c`):

```c
/* Returns 1 if ti's matrix has identifiable local rows; fills
 * group_of[d] (0..ngroups-1) per data disk and local_row[g] (parity-row
 * index) per group. Returns 0 if not LRC-shaped (caller uses the
 * existing generic path). Pure; no allocation. */
int survivor_lrc_detect(const topo_int_t *ti, int *group_of,
                        int *local_row, int *ngroups_out);
```

Complexity O(npar × ndata), plan-time only, negligible vs subset
enumeration.

## 3. Local cover for a failure pattern

Given LRC structure and failure list F (sorted, validated):

- A **data-disk** failure at disk d is locally covered by
  `local_row[group_of[d]]`.
- A **parity-disk** failure is never "local" in the LRC sense: a failed
  global parity is recomputed from data (generic path); a failed local
  parity is recomputed from its group (this *is* local: the local row's
  support disks are all survivors).
- F is **purely locally recoverable** if every failed disk is covered
  by a distinct local row (data failures in distinct groups, or failed
  local parities), i.e. the set
  `{ local_row[group_of[d]] : d in F, d data } ∪
   { local_row of group(g) : local parity of g in F }`
  has size == nf and each such row survives (not itself failed).

```c
/* Returns 1 if F admits a pure-local solution; fills cover_rows[0..nf)
 * with the covering local parity-row indices (sorted). Returns 0
 * otherwise (caller uses the generic path). */
int survivor_lrc_local_cover(const topo_int_t *ti, const int *F, int nf,
                             const int *group_of, const int *local_row,
                             int ngroups, int *cover_rows);
```

Note: `survivor_enumerate` already guarantees that any returned subset
is non-singular; the local cover is a *candidate* restriction, not a
correctness proof. If the local cover subset is singular (shouldn't
happen for valid LRC, but defensively), enumeration will have dropped
it and the restriction finds nothing → fall back to generic.

## 4. Integration in survivor_select

Insertion point: in `survivor_select`, after `survivor_enumerate`
succeeds and before `survivor_l2_signal`.

```c
/* LRC locality fast path (planner rule, not cost-model accident). */
int lrc_group_of[RJ_MAX_DISKS], lrc_local_row[RJ_MAX_DISKS], lrc_ng = 0;
int lrc_cover[RJ_MAX_DISKS], lrc_ncover = 0;
int use_lrc = survivor_lrc_detect(ti, lrc_group_of, lrc_local_row,
                                 &lrc_ng)
           && survivor_lrc_local_cover(ti, F, nf, lrc_group_of,
                                       lrc_local_row, lrc_ng,
                                       lrc_cover, &lrc_ncover);
if (use_lrc) {
    /* Restrict keep[] to subsets whose rows are all in lrc_cover.
     * The enumeration already validated non-singularity, so the
     * pure-local subset is present. */
    for (int i = 0; i < nsub; i++) {
        keep[i] = subset_rows_in(subs[i].rows, subs[i].nrows,
                                 lrc_cover, lrc_ncover);
    }
    /* Defensive: if restriction kept nothing (e.g. local subset was
     * singular and dropped by enumerate), fall back to generic. */
    int any = 0;
    for (int i = 0; i < nsub; i++) any |= keep[i];
    if (!any) use_lrc = 0;
}
if (!use_lrc) {
    survivor_l2_signal(subs, nsub, cpu, o);
    survivor_shortlist(subs, nsub, keep);
} else {
    /* Still run the L2 signal on the kept (local) subsets: the
     * shortlist is bypassed, but buildability must be verified.
     * A local subset that fails to build (l2_valid == 0) is dropped;
     * if none remain, fall back to the generic path. */
    survivor_l2_signal(subs, nsub, cpu, o);
    int any_valid = 0;
    for (int i = 0; i < nsub; i++) {
        if (keep[i] && !subs[i].l2_valid) keep[i] = 0;
        any_valid |= keep[i];
    }
    if (!any_valid) {
        /* Local build failed; fall back to generic shortlist over
         * all subsets. */
        survivor_shortlist(subs, nsub, keep);
        use_lrc = 0;
    }
}
```

Key properties:

- **Pl
...[truncated 4888 chars]