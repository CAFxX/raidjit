# Survivor Set Selection (Track A5)

For recovery operations with more surviving disks than strictly needed,
different subsets of survivors yield different solve costs. This
document describes the plan-time enumeration, pruning, and empirical
selection of the cheapest measured survivor set.

## 1. Enumeration space

For a recover op with `nf` failed disks (`F[0..nf-1]`, sorted/deduped,
`1 <= nf <= npar`):

- Unknowns: the `nf` failed disks.
- Equations: the `npar` parity rows. Row `j` is the equation
  `P_j = sum_i M[j,i] * D_i`.
- A valid survivor set is a subset of `nf` rows whose `nf x nf`
  subsystem (unknowns vs rows) is non-singular.

If parity disk `j` fails, row `j` must be in every valid subset: row `j`
is the only equation containing `P_j` as an unknown. Let `f_par` be the
number of failed parity disks, `Fp` their row indices (forced), `Rs`
the surviving parity rows, `f_data = nf - f_par`. The enumeration is
`C(|Rs|, f_data)` combinations of `Rs` plus the forced `Fp`.

Count: `C(npar - f_par, nf - f_par)`. When `nf == npar`, exactly one
subset (today's behavior, bit-identical).

Singular subsets are dropped silently (common for XOR array codes where
many row combinations are linearly dependent).

## 2. Pruning bound (L2 shortlist)

Raw enumeration is bounded by `SURVIVOR_ENUM_CAP = 8192` combinations.
If exceeded, `survivor_enumerate` returns -2 and the caller falls back
to the single all-rows plan (today's behavior) instead of failing.

The L2 shortlist applies A1-style static pruning across subsets, keyed
on the EXACT static signal -- not on plan shape. For each valid subset
`survivor_l2_signal` builds exactly ONE version (the canonical
`plan_candidates[0]`: best ISA, highest fitting U) via
`jit_build_version` and reads its real `(uops_per_byte, mem_per_byte)`
off the version (the twin-model lockstep rule forbids a second,
duplicated cost model). A `(kind_rank, nnz, ncols)` key was tried and
removed: it is vacuous on the RS space (2+ failures: all subsets share
identical triples, e.g. rs(6,3)/f02 all P_MULACC nnz=12 ncols=6), while
the exact signal discriminates ((0.660, 18.141) vs (0.691, 20.031)).

The shortlist keeps: the pareto frontier on the exact
`(l2_uops, l2_mem)` pair, plus the `is_today` subset (the no-regression
anchor) unconditionally, then the statically top-(K-1) pareto members
(lexicographic on (uops, mem); ties by subset index), with a HARD CAP
of `SURVIVOR_SHORTLIST_CAP = 3` subsets total:
keep = {today} ∪ top-(K-1) statically-ranked pareto members.
Subsets whose canonical version fails to build carry no signal and are
never treated as dominated (a missing signal can only keep a subset,
never drop it). Only shortlisted subsets pay the full candidate build
+ pooled bench. Measured shortlist sizes (x86-64): rs(6,3)/f0 = 1,
rs(6,3)/f02 = 2, rdp(5) whole-disk = 3 of 34.

`RJ_NO_SUBSET_PRUNE=1` disables L2 (same contract as `RJ_NO_PRUNE`):
every enumerated subset is benchmarked.

## 2b. Significance gate (P2)

After the pooled `bench_select`, `survivor_significance_gate` makes
"never measurably worse than today" a property of the selector. Each
subset's best score is recomputed from the already-measured pool (the
pool is appended in subset order; `pool_sub` tracks subset
boundaries). If the winning subset is not today's AND its lead over
today's best measured GB/s is within `bench_select`'s measured noise
window (max(1.5%, pooled noise), captured via the `noise_out`
parameter), the win is a noise flip: the selector falls back to
`select_by_time` restricted to today's pool entries with the measured
window -- the bench protocol's own selector, with 1/GB/s as the
relative-time proxy (exact: `vtot` is identical for every pooled
version). Within-window ties break by uops/byte, then mem/byte, then
code size, exactly as `bench_select` does. Genuine wins (XOR codes
show up to 2.25x spread) still ship; noise flips don't. The gate is
skipped when benchmarking is off (no measured window exists) and when
the anchor is absent from the pool (only under the
`RJ_SUBSET_PRUNE_INVERT` test hook).

## 2c. L2 signal vs NT/prefetch twins (P1.2 decision, 2026-10-03)

The L2 signal reads only the plain temporal canonical version per
subset, while the measured pool also contains NT/prefetch twins. A
subset L2-dominated on its plain version could in principle own the
measured winner through its NT twin and be dropped pre-benchmark.

Decision: empirical + documented (the "structural" route was evaluated
and rejected):

1. A static extension of the signal to the twin family is vacuous for
   NT/prefetch. NT twins tie their temporal original on the static
   counts by construction -- x86-64 `c_st`, AArch64-SVE `c_st_to`,
   and RVV `c_vse` count NT stores identically to temporal ones, and
   RVV's NTL hint is explicitly uncounted; on AArch64-NEON the only
   static difference is the STNP store-pairing uop discount, which
   reflects instruction count, not the cache-bypass benefit NT
   exists for. Prefetch twins are statically dominated by their own
   temporal original (+1 uop per stream per iteration, same mem; see
   api.c's note). The twin-model lockstep rule forbids a second
   static model of NT's memory-system benefit, so no static extension
   can see this failure mode; the only non-vacuous "extension to the
   measured pool" is measurement itself, which is what the shortlist
   exists to avoid.
2. The no-flip property therefore rests on the **NT-delta
   subset-independence** assumption: every subset's plan stores the
   same `nf` failed disks with the same layout, so the NT store-path
   benefit tracks uniformly across subsets, and a static dominance is
   not expected to flip into a measured NT upset. Second-order
   differences exist (different canonical U across subsets changes
   the store mix) but are small; the Part-7 tripwire is the backstop.
3. The Part-7 tripwire runs every corpus case twice: once plain and
   once with NT twins enabled (`nt_threshold=1`), with the
   head-to-head referee building under the same flags -- so the
   shortlisted-vs-exhaustive comparison actually exercises the NT-flip
   failure mode. (Note: before this fix the tripwire used zeroed
   options, so NT twins were never in play there despite a review
   note claiming otherwise; the extension closes that gap.)
4. **Re-audit trigger:** a future twin family with subset-dependent
   benefit (e.g. a read-path twin whose gain varies with the subset's
   input columns), or NT twins becoming statically distinguishable
   in a subset-dependent way, must re-audit the L2 signal. The
   `nt_static_tie` unit test (Part 2 of test_survivor) pins today's
   static relations -- NT never worse than temporal on (uops, mem),
   prefetch never better on uops -- and fails loudly if they change.

## 3. No-regression anchor

`is_today` marks the enumerated subset whose solution matrix is
bit-identical to the all-rows greedy solve (the pre-A5 behavior). It is
identified by matrix equality, NOT by pivot rows: the pivot rows are an
elimination implementation detail (partial-pivot order), while R is the
mathematical object the plan is built from. Matrix equality is robust
to bit-identical-R pathologies — two distinct subsets can in principle
yield bit-identical R (e.g. symmetric topologies), and any of them is a
valid anchor since identical R ==> identical plan — whereas pivot-row
identity would couple the anchor to elimination internals for no
benefit.

The anchor is always shortlisted, so the benchmark can always pick
today's solution. If another subset benchmarks faster *beyond the
noise window*, picking it is an improvement, not a regression; a
within-noise win falls back to today's subset via the significance
gate (§2b). The anchor marks the FIRST subset whose solution matrix is
bit-identical to the all-rows solve: two distinct subsets can in
principle yield bit-identical R, and either is a valid anchor, so only
the first match is marked (never double-marked).

## 4. Benchmark protocol

For each shortlisted subset:
1. Build the recovery plan via `plan_build_recover_rows` (existing
   solver, all computation at plan time per Carlo's invariant).
2. Generate candidates via `plan_candidates`, build versions via
   `jit_build_version` (all three backends).
3. Pool all versions across subsets; A1 `prune_versions` applies.
4. Single `bench_select` over the pool; winner is the measured cheapest.
5. Significance gate (`survivor_significance_gate`): a non-today win
   within the measured noise window falls back to today's best
   measured version (see §2b).
6. Every version is validated against its own plan's scalar `ref_run`
   before benchmarking (via `version_t.plan` borrow).

The winning subset is recorded in the plan desc as `[rows 0+2]`.

## 5. Dispatcher integration (A2)

The chosen subset bakes into the plan at compile time. The A2
length dispatcher (general / notail / masked-tail) operates on the
winning plan's versions; the dispatcher ABI is unchanged. The notail
twin is built from the winner's plan (`wpl`), so the subset choice
propagates to both dispatched variants.

## 6. Environment variables

- `RJ_NO_SUBSET_PRUNE=1`: disable L2 shortlist (benchmark all subsets).
- `RJ_NO_SURVIVOR_SELECT=1`: public disable knob for A5 survivor-set
  selection (A1 `RJ_NO_PRUNE` precedent). Non-empty and != "0" takes
  the legacy single-subset (all-rows) path in `rj_compile`'s recover
  branch: no subset enumeration, no pooled benchmark -- exactly
  today's behavior, behavior-identical plan (mat/kind bit-identical;
  the plan name differs: legacy `[rows 0+1+2]` vs anchor `[rows 0]`).
  "0" or empty means "not set".
- `RJ_SUBSET_PRUNE_INVERT=1`: test-only hook; inverts the pareto
  decision and skips the `is_today` anchor. Used for red-green mutation
  tests. Never set in production.
