# Loop-Structure Prototype Study

**Date:** 2026-10-02. **Scope:** x86-64 AVX-512 codegen loop-structure
alternatives for the raidjit vector body. **Method:** static uop model
(`version_t.uops_per_byte` / `mem_per_byte`, deterministic) via a scratch
harness; one JIT accounting variant built for tiling (P_HORNER only, never
executed — see P1 note); others evaluated analytically against the exact
model plus a tracking simulation. No wall-clock benchmarking was performed:
all rejections below are static-model-based, with margins (≥16% uops,
≥67% mem) that make reversal on hardware implausible. **Verdict: all four
prototypes REJECT. No codegen change.** Two independent peer reviews:
ACCEPT with 8 minor claim-precision findings, all remediated below.

## Baseline (AVX-512, measured via `jit_build_version`)

| Plan | uops/B | mem/B | regs |
|---|---|---|---|
| raid6-5 horner+gfni U=8 | 0.3281 | 7.000 | 19 |
| rs(6,3) horner+gfni U=8 | 0.5801 | 9.016 | 28 |
| raid5-4 xor U=8 | 0.0781 | 5.000 | 9 |

Hand-verified: raid6-5 = 168 uops / 512 B = 0.3281 exactly (i=0: 8u×3;
i=1..4: 8u×4; 16 stores). mem/B counts streaming traffic only
(inputs+outputs); the loop-invariant pool is excluded.

**U-(in)variance, with a bound on the unmodeled term.** uops/B is nearly
U-invariant in the model (raid6-5: 0.3281 for U=1..8; rs(6,3): 0.5938→0.5801,
the 2% slope is the non-hoisted 6th input's per-group pointer reload —
1 uop + 8 B/group, matching to 4 decimals). Caveat: the model does NOT count
per-group loop overhead (`lea`/`cmp`/`jcc`, ~5 uops/group ≈ 0.078 uops/B at
U=1, up to ~24% of counted uops/B). So "U reduction is free" is true in the
model but not on hardware. The dominance arguments below survive this:
the best conceivable counterexample (pressure forces untiled U=1, tiled
T=3/U=8) wins ~0.006 uops/B on counted+loop terms, but the tile loop's own
uncounted overhead (~3 uops/tile) erases it, and mem/B is ~5× worse.

## P1: Output-row tiling — BUILT (accounting variant), REJECTED

Forked `src/x86_64/jit.c` (`/tmp/proto/jit_tile.c`, not committed):
P_HORNER restructured as `for tile jt: for i: for u: for j in tile` with
remapped accumulators `(j-jt)*U+u` (subset of the normal range, no regfile
conflict) and per-tile stores. **Scope notes:** only the P_HORNER branch
was forked (P_XOR/P_MULACC/P_DELTA share the identical (i,u)-shared-input
nest shape, so the load-inflation math carries over analytically); the
variant was never executed (no ref-vs-JIT check) and `plan_need_regs` was
not adjusted — so only tiling's *cost* side was measured, not its
pressure-relief benefit. Cost alone rejects, so the outcome is unchanged.

| Config | raid6-5 uops/B | raid6-5 mem/B | rs(6,3) uops/B | rs(6,3) mem/B |
|---|---|---|---|---|
| untiled | 0.3281 | 7.000 | 0.5801 | 9.016 |
| T=1 | 0.4062 (+24%) | 12.000 (+71%) | 0.7715 (+33%) | 21.047 (+133%) |
| T=2 | 0.3281 (=) | 7.000 (=) | 0.6758 (+16%) | 15.031 (+67%) |

T=2 on raid6-5 (n_out=2, single tile) reproduces the baseline exactly —
a self-consistency check of the fork's accounting refactor, not a
validation. T=1's +0.078 uops/B matches the predicted extra-load cost
exactly (40 extra loads / 512 B), confirming the fork accounts honestly.

**Why:** tiling multiplies input loads by `n_out/T` (each tile reloads all
inputs) while the per-row op count is unchanged. It trades memory traffic
for register pressure, but pressure is already managed by U selection.

## P2: Hybrid (tile,U) product space — REJECTED (analytical)

The product space is degenerate: uops/B is nearly U-invariant and tiling
(T < n_out) strictly increases both uops/B and mem/B. The (T,U) optimum is
(n_out, any U) = the untiled baseline. **U-reduction dominates tiling**
*within the static model and for all buildable plans* (see Portability for
the precondition): both relieve pressure, but U reduction costs ~nothing
in the model (modulo the loop-overhead bound above) while tiling costs
`n_out/T ×` input loads. The planner should (and does) always prefer
smaller U over tiling.

## P3: Coefficient-aware scheduling — SIMULATED, REJECTED

Idea: sort rows by coefficient within a column to maximize F2 last-value
table-load tracking hits. Exact simulation of the tracking logic,
*including* cross-column/cross-u persistence, over 36 two-failure + 10
three-failure recovery plans and 3 P_DELTA plans:

- Duplicate non-{0,1} coefficients **do** occur in real recovery plans
  (e.g. rs(6,3) rec 0+1 column 5 has both rows sharing one). With n_out=2
  they are necessarily *consecutive* — already captured by the existing F2
  tracking.
- Across the 10 three-failure (n_out=3) plans — the smallest case where
  sorting could beat tracking — **zero columns had non-consecutive
  duplicates**, the only pattern sorting fixes.
- Even the best achievable global row permutation saves ≤1 pool load per
  group out of 80–192 (<1%).
- P_DELTA delta columns (Vandermonde subcolumns): no duplicates.
- Standard topologies (Vandermonde horner_c, all-{0,1} XOR): nothing to sort.

The REJECT stands (F2 already captures the exploitable cases; the residual
is <1%/group, not worth the complexity), on measured rather than assumed
grounds. (c==0 skip and c==1 plain-XOR are already implemented in both
P_HORNER and `mulacc_column`.) Caveat: `rj_topo_custom` matrices are
user-supplied and could contain adversarial duplicate patterns — but a
single static row order cannot exploit per-column optima anyway.

## P4: MULACC alternatives — REJECTED (analytical)

- **Row-major** (`for j: for i: for u`): loads each input n_out×
  (once per row) vs once per (i,u) in the current column-major order.
  Strictly worse; no uop reduction to compensate.
- **u/j interchange** (`for i: for j: for u`): same n_out× load inflation.
- **All 6 (i,u,j) permutations** (reviewer enumeration): `i,j,u` with input
  staging across j costs the same uops/mem but needs U extra live regs
  (cf. the RVV backend's `(n_out+1)×U` term); `u,i,j` repeats non-hoisted
  `body_inptr` loads U× (strictly worse); `u,j,i`, `j,u,i` degenerate;
  `j,i,u`/`j,u,i` = T=1 tiling (P1). None wins.
- **Table-domain accumulation**: no shareable structure across rows. Note
  the asymmetry the study originally misstated: the nibble split *is*
  shared across rows per (i,u) for **P_MULACC** (split on the input `t`,
  hoisted out of the j-loop), but for **P_HORNER nibble** the split operates
  on the per-row accumulator *inside* the j-loop (`acc = acc*c_j ^ t`
  multiplies per-row state — inherently per (i,j,u), 7 uops/row-step).
  Either way, both `acc` and the coefficient differ per row, so nothing is
  shareable across rows. GFNI is 1 uop per multiply with the same property.

## General result

The current input-major order (`for i: for u: for j`, all n_out×U
accumulators live) is at the Pareto frontier of the (loads × pressure)
tradeoff for every buildable plan: it attains the minimum possible input
load count — one per (i,u), information-theoretic minimum since each (i,u)
value is distinct (the i=0 Horner move-idiom and P_XOR `first_in`
direct-load already attain it) — and one store per (j,u), likewise minimal.
Register pressure is managed by U selection, which dominates tiling. No
loop-structure change can reduce uops/B; only per-row op fusion
(VPTERNLOGD, EOR3 — both already implemented) moves the needle.

## Portability

All four prototypes reject on x86-64 (32 vector regs). The U-dominance
argument carries a precondition: it holds **for all buildable plans, i.e.
where U=1 fits the register file**. On 16-register ISAs there is a regime
where no U≥1 fits (P_MULACC/P_DELTA need n_out×1+7 ≤ 16 → n_out ≥ 10;
P_HORNER remat needs n_out+6 ≤ 16 → n_out ≥ 11) — reachable via
`rj_topo_custom` / wide recovery — where tiling would be the only
loop-structure route to a buildable kernel. Every documented topology has
n_out ≤ 3 (x86-64 cliff: n_out ≥ 26), and the current behavior there is a
loud build failure, not silent wrongness. No prototype merits porting
today; the precondition is recorded here for the wide-n_out future.

## Conclusion

**No codegen change. All four prototypes REJECT with evidence.**
The loop structure is optimal for buildable plans; future work should
target per-row op fusion, not loop restructuring.

## Addendum (2026-10-03): wall-clock re-test on A3 CSE bodies — verdicts stand

The rejections above were static-model-only. Audit R1 (SOTA batch, A7
track) re-ran all four prototypes with wall-clock measurement on the
new A3 item-1 CSE bodies (P_XOR scheduled emitter, AVX-512, AMD EPYC
9D25), where the old study had no data. Full record:
`~/workspace/raidjit-tracks/A7/REPORT.md`; harness
`~/workspace/raidjit-tracks/A7/exp/a7_r1.c` (fork `exp/jit_tile.c` with
variant emitters V0–V7, built by `exp/surgery.py`).

Method in brief: CPU0-pinned, 51 interleaved rounds with per-round
shuffled variant order, batched calls at 256 KiB / 1 MiB / 4 MiB working
sets; verdicts on paired ratios t_variant/t_baseline (median [Q1,Q3]);
a variant wins only if median ratio < 1 − max(1.5%, pooled noise).
Pre-registered symmetric decontamination (drop a round for a size when
ANY variant's sample exceeds 3× its own median). Fork-fidelity gates
before timing: V0 byte-identical to `jit_build_version` per (config,U);
V6 (T=n_out tiling) byte-identical to V0; every variant differential
vs scalar/GF reference at boundary lengths incl. tails.

Corpus: `band10x4`, `tri9x3`, `chain12x4`, `stair14x5`, `dup8x3`
(P_XOR CSE bodies, ntemps 2–5, 36 cells — some U skipped where no
xorsched twin enumerates) + P4-literal leg on rs(6,3)
2-disk recovery (P_MULACC, GFNI + nibble, U=8).

Result: 144 paired verdicts on the CSE corpus, **0 wins above the
pre-registered gate** for any variant. Power caveat: several cells carry
20–50% pooled noise, so the gate has low power for small wins — this is
the safe direction for reject-verdicts (a true small win could be
missed, but no false win was declared).
Per-variant median-of-medians (ratio vs baseline):
V1 (T=1) 1.036, V2 (T=2) 1.023, V3 (reschedule) 1.006, V4 (row-major)
1.038 — every variant slower-or-equal in ≥26/36 cells; V1/V4 median
exceeds 1 + gate at low U (point-estimate gate crossing, not a
significance test — wide IQRs; e.g. tri9x3 U=1: V1 1.151
[1.043,1.288], V4 1.159 [1.036,1.272]). P_MULACC leg (evidenced runs
only: `r1_v7_run1.log` + the nibble row-major line embedded in
`r1_run4.log`; other reported runs' logs were not preserved — see
`~/workspace/raidjit-tracks/A7/REPORT.md` evidence-trail note):
row-major nibble 1.24 / 1.16, row-major GFNI 1.02,
(i,j,u)-staged GFNI 1.03 — no win; the staged permutation
(P4's "analytically equal" case) measures ≡ baseline at +U live regs,
confirming the analytical rejection empirically. Hybrid (T,U)
best-vs-best: baseline wins 4/5 configs (the one exception is
U-selection noise on a pure reschedule with an identical op multiset).

**All four prototypes REJECT, now with wall-clock evidence.** No
codegen change. The P4 portability precondition (wide-n_out future)
is unchanged.
