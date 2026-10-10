# LRC locality-aware recovery — coordinator checkpoint

## Status (2026-10-10, load 14+ blocking builds)

**Spec**: `hidden_files/lrc_locality_SPEC.md` — complete.
**Implementation**: `src/survivor.c` + `src/survivor.h` — complete, syntax-clean, committed (e93d28f, 5dcf9eb).

## What was built

**Problem**: LRC local recovery worked via emergent cost-model dominance
(local XOR subset strictly dominates global MULACC on both static axes),
not by design. Fragile to benchmark noise, significance-gate fallback,
and cost-model drift.

**Solution**: Explicit planner rule in `survivor_select`.

1. `survivor_lrc_detect(ti, group_of, local_row, ngroups_out)`:
   Identifies LRC-shaped topologies by matrix structure (pure-XOR
   disjoint rows partitioning data disks). Structure-based, not
   name-based — works for any present/future LRC-like matrix.

2. `survivor_lrc_local_cover(ti, F, nf, ...)`: Computes pure-local cover
   for failure pattern F (distinct surviving local row per failure).
   Handles: data failure → group's local row (must survive); failed
   local parity → itself (forced recompute, no survival check); failed
   global parity → not local (generic path); two failures one group →
   not purely local (generic path).

3. `survivor_select` integration: after `survivor_enumerate`, if
   pure-local cover exists, restrict `keep[]` to local subsets
   (bypass pareto shortlist). Still runs L2 signal to verify
   buildability; falls back to generic shortlist if local fails to
   build or restriction keeps nothing.

**No new cand_t axis** (subset selection, not codegen) → no prune.c change.
**Byte-identical for non-LRC** (detect returns 0 → generic path, zero behavior change).
**Bit-exact for LRC** (same subsets the cost model already picked; verified by existing tests).

## Remaining work (blocked on load)

1. Build + run `test_lrc` (verify no regression, same check counts).
2. Build + run full `make test` (or survivor-related subset).
3. Add unit tests for `survivor_lrc_detect` / `survivor_lrc_local_cover`
   (edge cases: non-LRC matrices, overlapping rows, incomplete partition,
   mixed failure patterns).
4. ASan/UBSan + QEMU (aarch64, riscv64) reruns.
5. 2 independent peer reviews + self-attack.
6. Integration via `wt.py integrate` after reviews.

## Commits

- e93d28f: spec + implementation
- 5dcf9eb: fix failed-local-parity survival check
