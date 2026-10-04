# Generator-matrix search (track A6)

- [What it is](#what-it-is) · [Design](#design) · [Cost metric](#cost-metric-dogfooding-the-planner) · [MDS proof method](#mds-proof-method)
- [Analytical result: the search's fixed point is the old matrix](#analytical-result-the-searchs-fixed-point-is-the-old-matrix) ([one-count bound](#the-one-count-bound-is-n3-not-n2-corrected) · [why the extra one does not win](#why-the-extra-one-does-not-win-placement-constraints--horner-floor) · [scope](#what-this-section-does-and-does-not-prove))
- [Old-vs-new cost table](#old-vs-new-cost-table-static-uopsbyte-mem-bytesbyte) · [Uncovered topologies](#uncovered-topologies) · [Falsifiable ship criterion](#falsifiable-ship-criterion) · [Tests](#tests) ([mutation evidence](#mutation-evidence-all-performed-all-reverted)) · [Perf guard](#perf-guard) · [Limitations](#limitations)

## What it is

`src/matrix_search.c` / `src/matrix_search.h`: a plan-time search for
low-cost MDS generator matrices over GF(2^8). `rj_topo_rs()` (and
`rj_topo_raid6()`, which shared the same default) now emit the searched
matrix **outright** — there is no opt-in, and no constructor hardcodes
the legacy default anymore (`rs_default_matrix()` survives as a
non-static helper in `src/gf.c`, used only by tests — see Uncovered
topologies).

## Design

- **Search space:** systematic-form matrices `P` (npar x ndata), row-major
  bytes, the same representation the planner consumes.
- **Seeds (deterministic):**
  1. the historical Vandermonde-Horner default `M[j][i] = 2^(j*(ndata-1-i))`
     (MDS by proof inside the envelope -- see below);
  2. a Cauchy matrix `C[j][i] = inv((i+1) ^ (128+j))` (MDS by proof --
     see below);
  3. a fixed-seed pseudorandom row-geometric matrix. This seed is
     **not** MDS by proof: random distinct-base row-geometric matrices
     are not MDS in general (sampling found singular 2x2 minors --
     e.g. two rows whose base ratio has small order dividing a small
     column gap), so the seed carries no proof. The true mechanism is
     rejection sampling: up to 1000 candidates from a fixed xorshift64
     stream, each verified by the full MDS check, first success wins;
     on total failure the seed is skipped, never emitted unproven.
- **Why seeds 0 and 1 are MDS by proof.** Seed 0: a 2x2 minor on rows
  r1<r2, columns i1<i2 factors as
  `2^(r1*k2+r2*k1) * (2^((r2-r1)(i2-i1)) - 1)` with `k_t = ndata-1-i_t`,
  nonsingular iff `ord(2) = 255` does not divide `(r2-r1)(i2-i1)`; the
  product is at most `2*50 = 100 < 255`. A 3x3 minor is Vandermonde in
  `x_t = 2^(ndata-1-i_t)`, nonsingular iff the bases are pairwise
  distinct, i.e. no column gap in `[1,50]` is a multiple of
  `ord(2) = 255`. Minors mixing data and identity columns reduce to
  these cases (a data column paired with unit vectors needs a nonzero
  entry, and `2^k != 0` always). Seed 1: the Cauchy determinant
  `det(inv(x_i ^ y_j)) = prod(x_b^x_a)(y_b^y_a)/prod(x_a^y_b)` is
  nonzero iff the x's are distinct, the y's are distinct, and no
  `x_a ^ y_b = 0`; here `x_i = i+1 in [1,51]`, `y_j = 128+j in
  [128,130]`, pairwise distinct with disjoint ranges.
- **Improvement step:** greedy single-entry 1-ization to fixpoint
  (at most 8 sweeps). This is complete for strict improvements *within
  one planner strategy*, not a heuristic: under the planner's cost
  model an entry's cost depends only on its class ({0}, {1}, other —
  all nontrivial values cost identically), so a strict improvement at
  fixed strategy must 1-ize some entry; and a 1-ization that breaks MDS
  is rejected by the check. `0` is never tried: every entry must be
  nonzero for MDS in the envelope (for npar == 1 the 1x1 minors of H
  are the entries themselves; for npar >= 2 the 2x2 minors pairing
  each data column with an identity column force every entry nonzero),
  so it is always rejected. NOT in the neighborhood: single-entry
  other→other' moves that would change the planner's strategy choice
  (e.g. completing a Vandermonde structure to flip P_MULACC to
  P_HORNER) — latent only, since any P_HORNER-coded matrix costs at
  least the default (see the analytical note).
- **Acceptance:** strict improvement in uops/byte (`u < best - 1e-12`).
  A challenger also wins on a uops-tie (within 1e-12) plus a mem/byte
  win; full ties keep seed 0 — no gratuitous parity-format churn.
- **Proof gate:** the winner is verified by *full* minor enumeration
  (`early_exit = 0`, honored by the code: every minor is examined and
  counted even after a singular minor is found) before emission, so a
  positive verdict always implies full enumeration — the tests assert
  the minor counts (35/35 … 24804/24804). On failure the search falls
  back to the verified seed instead of emitting an unproven matrix.
- **Budget / cache:** the work is bounded (8 greedy sweeps × 3 seeds,
  each step one MDS check + one `plan_build`); worst case is ~1–3 s on
  a shared VM for the first (51,3) call, dominated by vCPU contention
  rather than anything algorithmic — not a stable number, and it does
  not need to be: results are cached process-wide per (ndata, npar)
  (C11 `atomic_flag` spinlock, no pthread dependency), so repeat calls
  are a `memcpy` (~0 ms). `npar == 1` short-circuits to the all-ones
  row (unique optimum).
  For scale: `rj_compile` for rs(20,3) takes ~12 s, so the one-time
  search cost is noise next to codegen.

## Cost metric (dogfooding the planner)

`matrix_search_cost()` builds the **real** encode plan with `plan_build`
(the planner's actual P_XOR/P_HORNER/P_MULACC selection), then evaluates
a static model of the encode body transcribed 1:1 from the x86-64
nibble backend (`src/x86_64/jit.c`): per-u-lane vector uops/mem plus
per-group-iteration pointer reloads, at reference U=8, vl=64. Ranking of
same-shape MDS matrices is U/vl-independent (reload terms cancel), and
the transcription is pinned by an exact-equality cross-check against
real `jit_build_version` codegen (see Tests).

This is a *static* model used only for ranking. It is never presented
as a measurement; the shipped performance evidence is the perf-guard
before/after comparison (below).

## MDS proof method

`matrix_is_mds()` enumerates **all** C(ndata+npar, npar) npar x npar
minors of `H = [-P | I]` with closed-form 1x1/2x2/3x3 determinants
(`npar > 3` is rejected; the envelope is npar <= 3, ndata <= 51). The
enumeration is exact integer arithmetic over GF(2^8) — no sampling, no
randomization. An independent path (the library's own `solve_recovery`
over all erasure subsets) agrees on the test corpus.

## Analytical result: the search's fixed point is the old matrix

Derived before coding and confirmed empirically: within the envelope,
the search's fixed point is the historical default, so it emits
byte-identical matrices (no parity-format change). Read this section
as a *bound + mechanism* sketch, not a closed-form optimality proof:
the conclusion "the search emits the historical default" is carried
empirically by the dense envelope tripwire in
`tests/test_matrix_search.c` (`test_search_deterministic` asserts
byte-identity on every in-envelope shape — npar=2,3 × ndata=1..51,
npar=1 sampled), not by this analysis alone.

### The one-count bound is n+3, not n+2 (corrected)

For rows a<b let `D_ab = { i : P[a][i] = P[b][i] = 1 }`, the columns
holding a 1 in both rows. `|D_ab| ≤ 1`: two such columns would make
the 2×2 minor `|1 1; 1 1|` singular. Then:

- If some column is `(1,1,1)`, it sits in `D_01 ∩ D_02 ∩ D_12` and
  exhausts all three pair-budgets at once: no other column may hold
  two ones, so the total one-count is at most `3 + (n-1) = n+2`.
- Otherwise each pair admits at most one two-one column, and a
  two-one column belongs to exactly one pair: at most three two-one
  columns, every other column at most one one, so at most
  `2·3 + (n-3) = n+3`.

The maximal feasible one-count is therefore **n+3** for npar=3 (n+1
for npar=2, same argument on the single pair), and n+3 is tight.
Counterexample to the old, false n+2 bound — this 3×4 matrix has
7 = n+3 ones (`D_01 = {0}`, `D_02 = {1}`, `D_12 = {2}`) and is MDS
(full enumeration: 35/35 nonsingular minors):

        1   1 218   1
        1 147   1 207
       36   1   1 197

The historical default attains n+2 for npar=3: row 0 is all ones, and
rows 1 and 2 each hold a single 1 at `i = n-1` (both `2^(n-1-i) = 1`
and `2^(2(n-1-i)) = 1` force `i = n-1` for `n ≤ 51`). Its all-ones
column at `i = n-1` exhausts all three pair-budgets, so n+2 is the
maximum for any matrix *with an all-ones column* — and the planner's
own Horner detector (`mat_is_horner_vandermonde` in `src/gf.c`)
*requires* an all-ones last column, so every P_HORNER-coded matrix is
capped at n+2 ones exactly like the default.

### Why the extra one does not win: placement constraints + Horner floor

Per-nontrivial-entry costs under the transcribed model (x86-64 nibble,
VEX) — corrected; the old note's "MULACC ≥ Horner" was false on uops:

- P_HORNER: 7 uops, 0 extra mem bytes per nontrivial entry.
- P_MULACC: 5–6 uops **+ 128 mem bytes** (two 64 B constant-pool
  loads) per nontrivial entry, **plus** 3 uops per column containing
  any nontrivial entry (the nibble split, shared across rows) and
  1 uop + 64 B per such column (the input load).

Two structural facts do the real work:

1. **Horner floor = default.** Under P_HORNER the per-column cost is
   `1 + Σ_j t(c_j)` with `t = 1` for `c_j ∈ {0,1}` else 7, where
   `c_j` is row j's Horner coefficient (`0` is MDS-infeasible). The
   minimum is attained at `c = (1, nontrivial, nontrivial)` —
   i.e. row 0 all ones — and the default is such a matrix. Every
   other P_HORNER matrix with row 0 all ones ties it *exactly* (same
   per-row classes, same loads), and the acceptance rule keeps seed 0
   on full ties; any P_HORNER matrix with `c_0 ≠ 1` pays 7 instead of
   1 for row 0 and is strictly worse. The Horner basin is therefore
   closed: the default is optimal, up to ties, among all
   P_HORNER-coded matrices.
2. **An n+3-one matrix cannot be Horner-coded** (no all-ones column is
   possible — one would cap the count at n+2), so the planner codes it
   P_MULACC, where the `D_ab ≤ 1` placement constraint forces the
   nontrivial entries to *spread*: an n+3 attainer has `2n-3`
   nontrivial entries occupying all n columns (three columns with one
   nontrivial, n−3 with two), paying the 3-uop split in every column.
   The one extra 1 over the default cannot buy back the split terms
   and the 128 B-per-nontrivial mem penalty — and even a uops tie
   would lose on the mem tiebreak.

Worked example — reviewer 2 independently found a (6,3) MDS matrix
with 9 = n+3 ones; sampling found this one (full enumeration: 84/84
nonsingular minors):

        1   1   8   1 104 107
        1  75   1 241   1 185
       58   1   1 144 209   1

The planner codes it P_MULACC. Measured under the transcribed model:
challenger 1.3926 uops/B and 27.0156 mem/B vs default (P_HORNER)
1.3613 uops/B and 9.0156 mem/B — it loses on uops and 3× on mem.
Biased random sampling (90k matrices with one-count ≥ n+1 at (6,3)
and (10,3), MDS-verified) found no challenger beating the default on
uops either (best: 1.3770 vs 1.3613 at (6,3)).

### What this section does and does not prove

Proven: the one-count bound (n+3, tight), the Horner-basin optimum
(default, up to full ties), the per-class costs, and the placement
constraint forcing MULACC challengers to spread their nontrivial
entries across columns. NOT proven in closed form: that no
MULACC-coded MDS matrix anywhere in the envelope beats the default on
uops. The fixed-point conclusion therefore rests on this analysis
**plus** the empirical tripwire: `test_search_deterministic` pins
byte-identity on the full envelope, so any future planner or
cost-model change that *would* find something better fails loudly
instead of silently changing the parity format. The search exists so
that such a change re-derives the optimum instead of trusting a
hardcoded matrix.

Consequence: the old-vs-new table below is all ties, which is the
*expected* outcome.

## Old-vs-new cost table (static uops/byte; mem bytes/byte)

| shape    | old uops/B | new uops/B | old mem/B | new mem/B |
|----------|-----------:|-----------:|----------:|----------:|
| (2,1)    |     0.0469 |     0.0469 |    3.0000 |    3.0000 |
| (255,1)  |     4.4883 |     4.4883 |  259.9062 | 259.9062 |
| (3,2)    |     0.3594 |     0.3594 |    5.0000 |    5.0000 |
| (8,2)    |     1.0684 |     1.0684 |   10.0469 |  10.0469 |
| (51,2)   |     7.1992 |     7.1992 |   53.7188 |  53.7188 |
| (4,3)    |     0.8594 |     0.8594 |    7.0000 |    7.0000 |
| (10,3)   |     2.3691 |     2.3691 |   13.0781 |  13.0781 |
| (51,3)   |    12.6992 |    12.6992 |   54.7188 |  54.7188 |

(The search output is byte-identical to the historical default on every
tested shape; `test_search_deterministic` pins this with a tripwire:
any future strict improvement that changes the bytes fails loudly so
the format change is deliberate and documented.)

## Uncovered topologies

**None.** The search covers exactly the envelope the old default
covered (npar <= 3, ndata <= 51, ndata+npar <= 256); `rj_topo_rs`
returns NULL outside it exactly as before. The old generator survives
in two places: `rs_default_matrix()` is still a non-static function in
`src/gf.c` (declared in `src/gf.h`) — the canonical historical-default
implementation, legitimately used by `tests/test_remat.c` (custom
RS(6,4), outside the search envelope) — and the search's seed 0 plus
the static copy in `tests/test_matrix_search.c` replicate its formula
(`M[j][i] = 2^(j*(ndata-1-i))`) for seeding and old-vs-new cost
comparison.

## Falsifiable ship criterion

Ship iff **all** of these hold:

1. `tests/test_matrix_search.c` passes: every searched matrix in the
   envelope is proven MDS by full minor enumeration (minor count ==
   C(ndata+npar, npar)), the independent solver agrees, and searched
   cost <= old-default cost on the corpus.
2. The cost-model cross-check passes with exact equality against real
   `jit_build_version` uops/mem on the corpus (pins transcription drift).
3. The full suite is green: `make test`, `make test-asan`,
   `make test-qemu` (+aarch64, +aarch64-sve, +riscv64).
4. Perf guard: 64 KiB / 1 MiB medians on affected configs stay inside
   the baseline pass-to-pass [min,max] bands (no throughput
   regression; format is unchanged so no re-baseline is needed beyond
   recording the numbers).
5. All four mutation probes fail as designed (below).

## Tests

`tests/test_matrix_search.c` (7 groups, all must pass):

- **mds_proof**: searched matrices for 8 shapes incl. envelope corners
  (51,2), (51,3), (255,1) proven MDS by full enumeration; minor counts
  asserted; independent `solve_recovery` check over all (small shapes)
  or sampled (large) erasure subsets.
- **mds_rejects**: beyond-envelope default, duplicated columns, zero
  entry, all-zero matrix all rejected; bad args return -1; a corrupted
  searched matrix is rejected; plus the single-singular-minor corpus
  (two matrices with EXACTLY ONE singular minor, at enumeration indices
  1 and 44 of 45, counts verified by the independent solver) — the
  tripwire that catches count-preserving skipped-minor mutations.
- **cost_le_old**: searched cost <= old cost on 8 shapes (table above);
  Cauchy strictly pricier than default (cost is not constant).
- **cost_model_crosscheck** (x86-64): 9 matrices (Horner/MULACC/XOR,
  ndata <= 5 and > 5 to cover hoist/reload branches, plus the ndata=1
  hoist corner (1,1)/(2,1)) — model vs real codegen uops/mem per byte,
  exact equality (< 1e-9).
- **search_deterministic**: repeated calls byte-identical (cache path);
  dense envelope tripwire: search output byte-identical to the
  historical default on every in-envelope shape (npar=2,3 x
  ndata=1..51, npar=1 sampled) — the empirical guard behind the
  fixed-point conclusion.
- **end_to_end**: ISA x topology x op x length combination test via the
  public API (`rj_compile`) vs an independent scalar reference:
  raid6(8), rs(6,3), rs(10,2), raid5(4); encode + 11 recovery combos;
  lengths {1, 7, 63, 64, 65, 1000, 4096}; one benchmark=1 config.
  Runs natively and under QEMU (aarch64, riscv64).
- **envelope**: NULL outside the envelope, non-NULL at the corners.

### Mutation evidence (all performed, all reverted)

- **A** — `src/matrix_search.c`: Horner nontrivial entry cost 7 -> 6:
  `cost_model_crosscheck` FAILS
  (`default(8,2) horner: uops/byte model=0.9590 codegen=1.0684`).
- **B** — force the winner to the Cauchy matrix (still MDS):
  `cost_le_old` FAILS on all 6 multi-parity shapes (e.g.
  `(51,3): searched 17.6211 > old 12.6992`) and `search_deterministic`
  FAILS (format-change tripwire).
- **C** — neuter the singular-minor rejection in `matrix_is_mds`:
  `mds_rejects` FAILS on all 5 bad matrices, `mds_proof` FAILS via the
  independent solver check (defense in depth), and the search itself
  wanders to a non-default matrix (proof gate matters).
- **D** — count-preserving skipped-minor mutation in `matrix_is_mds`
  (`if (n != SKIP && !minor_nonsingular(...))`, the minor counter still
  counts every minor): with SKIP=44 the pre-hardening suite PASSED
  SILENTLY — full evasion, because every old bad matrix has singular
  minors at other positions. The hardened single-singular-minor corpus
  (`test_mds_rejects`: matrix B, whose only singular minor is {7,9} at
  enumeration index 44 of 45) FAILS the mutated build ("matrix B
  accepted as MDS"). With SKIP=1 the old suite failed only
  incidentally (the blinded search wandered for (3,2), caught by the
  independent solver check); the hardened suite fails directly on
  matrix A (only singular minor {0,1}, index 1). Mutation reverted;
  suite green.

## Perf guard

- BEFORE: `tools/perf_guard/baseline_2026-10-02.json` (18 configs,
  AMD EPYC 9D25, 0 failures) — true pre-change baseline.
- AFTER: run after the change; expected outcome is zero throughput
  change (parity format is byte-identical). Record the numbers.

## Limitations

- The search is single-threaded and uninterruptible; worst case ~1–3 s
  once per process for (51,3) on a shared VM (vCPU contention, not
  algorithmic). Bounded (8 sweeps × 3 seeds) and cached; acceptable
  next to multi-second `rj_compile` times for the same shapes.
- The static cost model is x86-64-nibble-specific in its constants.
  Ranking agreement with other backends is verified for the nibble
  path by the exact-equality cross-check against real codegen;
  GFNI/SSE ranking agreement was checked by hand only (the nibble
  5–6-vs-7 uop margin for MULACC-vs-Horner nontrivial entries
  collapses to 2-vs-2 under GFNI) — moot while the search output is
  byte-identical to the historical default, at which point every
  backend compiles the same bytes.
- `npar > 3` is not supported by the proof (or the old default);
  constructors return NULL there, unchanged.
