# Generator-matrix search (track A2 — matrix_search 2.0)

- [What it is](#what-it-is) · [The objective: bitmatrix weight](#the-objective-bitmatrix-weight) · [Design](#design) · [MDS proof method](#mds-proof-method)
- [Old-vs-new weight table](#old-vs-new-weight-table) · [Uncovered topologies](#uncovered-topologies) · [Falsifiable ship criterion](#falsifiable-ship-criterion) · [Tests](#tests) ([mutation evidence](#mutation-evidence-all-performed-all-reverted)) · [Perf guard](#perf-guard) · [Limitations](#limitations) · [History: the 1.0 uops objective](#history-the-10-uops-objective)

## What it is

`src/matrix_search.c` / `src/matrix_search.h`: a plan-time search for
low-bitweight MDS generator matrices over GF(2^8). `rj_topo_rs()` (and
`rj_topo_raid6()`, which shared the same default) emit the searched
matrix **outright** — there is no opt-in, and no constructor hardcodes
any fixed matrix (`rs_default_matrix()` survives as a non-static helper
in `src/gf.c`, used only by tests — see Uncovered topologies).

**2.0 is a deliberate parity-format change.** The 1.0 search minimized
a static nibble-backend uops model whose fixed point was the
historical Vandermonde default, so 1.0 emitted byte-identical matrices.
The 2.0 objective (below) is different and finds strictly lighter
matrices that are *not* Vandermonde, so the emitted bytes change. There
is no backward compatibility: the parity format is whatever the search
emits. Downstream tests that pinned the old format's consequences (e.g.
`tests/test_remat.c`'s `P_HORNER` strategy expectations) are the
tripwire firing as designed — they must be updated deliberately, not
worked around.

## The objective: bitmatrix weight

Each GF(2^8) entry `c` expands to the 8x8 binary matrix of the GF-linear
map `x -> c*x`: column `k` holds the bits of `gf_mul(c, 1<<k)`, i.e.
bit `r` of column `k` is 1 iff input bit `k` feeds output bit `r`. The
weight `wt(c)` is the number of 1s in that matrix; the matrix objective
is

    W(P) = Σ_{j,i} wt(P[j][i])

(`matrix_search_bitweight()`; pinned constants: `wt(0)=0`, `wt(1)=8`,
`wt(2)=11`, `wt(3)=19`, `wt(4)=14`, `wt(128)=30` — hand-derived in
`tests/test_matrix_search.c`, and `wt(c) > 8` for every other nonzero
`c`, so 1 is the unique minimizer).

**Why this objective.** In a bit-level (gate-level) XOR schedule each 1
is one input bit XORed into one parity bit. Every parity bit has
nonzero fan-in (each 8x8 bitmatrix is nonsingular, so every one of its
rows holds at least one 1), hence the straightforward bit-level
schedule uses exactly `W − 8·npar` two-input XOR gates — a structural
identity, not a measurement. Minimizing `W` minimizes the thing a
bit-level scheduler (track A1) has to work with; cross-coefficient
sharing can only do better from a lighter starting point. The objective
is static and deterministic; it is never presented as a measurement.

## Design

- **Search space:** systematic-form matrices `P` (npar x ndata), row-major
  bytes, the same representation the planner consumes.
- **Seeds (deterministic):**
  1. the historical Vandermonde-Horner default `M[j][i] = 2^(j*(ndata-1-i))`
     (MDS by proof inside the envelope -- see below);
  2. a Cauchy matrix `C[j][i] = inv((i+1) ^ (128+j))` (MDS by proof --
     see below);
  3. a fixed-seed pseudorandom row-geometric matrix, rejection-sampled
     to MDS (up to 1000 candidates from a fixed xorshift64 stream, each
     verified by the full MDS check, first success wins; on total
     failure the seed is skipped, never emitted unproven).
- **Why seeds 0 and 1 are MDS by proof.** `ord(2) = 51` in
  GF(2^8)/0x11B (src/gf.c documents this; an earlier version of this
  note wrongly said 255 — corrected 2026-10-06). Seed 0: a 2x2 minor on
  rows r1<r2, columns i1<i2 factors as
  `2^(r1*k2+r2*k1) * (2^((r2-r1)(i2-i1)) - 1)` with `k_t = ndata-1-i_t`,
  nonsingular iff `ord(2) = 51` does not divide `(r2-r1)(i2-i1)`; with
  `(r2-r1) ∈ {1,2}` and `(i2-i1) ∈ [1,50]` the product is never a
  multiple of 51 (51 = 3·17; the factor-2 case needs `(i2-i1)` itself
  to be a multiple of 51). A 3x3 minor is Vandermonde in
  `x_t = 2^(ndata-1-i_t)`, nonsingular iff the bases are pairwise
  distinct, i.e. no column gap in `[1,50]` is a multiple of 51. Minors
  mixing data and identity columns reduce to these cases (a data column
  paired with unit vectors needs a nonzero entry, and `2^k != 0`
  always). Seed 1: the Cauchy determinant
  `det(inv(x_i ^ y_j)) = prod(x_b^x_a)(y_b^y_a)/prod(x_a^y_b)` is
  nonzero iff the x's are distinct, the y's are distinct, and no
  `x_a ^ y_b = 0`; here `x_i = i+1 in [1,51]`, `y_j = 128+j in
  [128,130]`, pairwise distinct with disjoint ranges.
- **Improvement step:** greedy single-entry weight minimization to a
  fixpoint (at most 8 sweeps). For each entry in fixed row-major order
  the search computes the set of MDS-forbidden values and jumps
  straight to the lightest allowed one. The forbidden set is exact, not
  sampled: only minors containing the changed data column can change
  status (C(ndata+npar-1, npar-1) instead of C(ndata+npar, npar)), and
  each such minor's determinant is affine in the changed entry —
  `det(v) = A·v ^ B` with `A` the cofactor of the entry's position —
  so a minor forbids at most one value (`B/A` when `A != 0`; when
  `A == 0` the determinant is v-independent and, the matrix being
  MDS, nonzero for all `v`).
- **Completeness claim (exact scope).** The objective is separable per
  entry, so a strict *single-entry* improvement must replace some entry
  with a lighter allowed value, and the step moves each entry to its
  lightest allowed value. A fixpoint therefore has no single-entry
  strict improvement left — this is proven by the move definition, not
  heuristic. NOT in the neighborhood: multi-entry coordinated moves
  (changing two entries at once can preserve MDS where neither change
  alone does). `0` is always forbidden (for npar == 1 the 1x1 minors of
  H are the entries themselves; for npar >= 2 the 2x2 minors pairing
  each data column with an identity column force every entry nonzero).
- **Acceptance:** strict weight improvement (`w < best_w`); full ties
  keep seed 0 — no gratuitous parity-format churn beyond what the
  objective demands.
- **Proof gate:** the winner is verified by *full* minor enumeration
  (`early_exit = 0`) before emission, so a positive verdict always
  implies full enumeration — the tests assert the minor counts (10/10
  … 24804/24804). On failure the search falls back to the verified
  seed instead of emitting an unproven matrix.
- **Budget / cache:** the work is bounded (8 greedy sweeps × 3 seeds,
  each entry paying one forbidden-set computation over
  C(ndata+npar-1, npar-1) minors); worst case is a few seconds once per
  process for (51,3) on a shared VM (vCPU contention, not algorithmic —
  not a stable number, and it does not need to be: results are cached
  process-wide per (ndata, npar) (C11 `atomic_flag` spinlock, no pthread
  dependency), so repeat calls are a `memcpy`). `npar == 1`
  short-circuits to the all-ones row (unique optimum: MDS needs every
  entry nonzero and 1 uniquely minimizes the weight).

## MDS proof method

`matrix_is_mds()` enumerates **all** C(ndata+npar, npar) npar x npar
minors of `H = [-P | I]` with closed-form 1x1/2x2/3x3 determinants
(`npar > 3` is rejected; the envelope is npar <= 3, ndata <= 51). The
enumeration is exact integer arithmetic over GF(2^8) — no sampling, no
randomization. An independent path (the library's own `solve_recovery`
over all erasure subsets) agrees on the test corpus.

## Old-vs-new weight table

Static integer weights (the search is deterministic, so these are exact,
not measurements). "Old" is the historical default — the 1.0 objective's
proven fixed point, i.e. the old objective's selection:

| shape    | old W | new W |
|----------|------:|------:|
| (2,1)    |    16 |    16 |
| (255,1)  |  2040 |  2040 |
| (3,2)    |    57 |    54 |
| (8,2)    |   215 |   175 |
| (51,2)   |  1944 |  1518 |
| (4,3)    |   151 |   120 |
| (10,3)   |   582 |   373 |
| (51,3)   |  3480 |  3085 |

Every multi-parity shape is strictly lighter; npar=1 ties (all-ones is
the unique optimum under both objectives).

## Uncovered topologies

**None.** The search covers exactly the envelope the old default
covered (npar <= 3, ndata <= 51, ndata+npar <= 256); `rj_topo_rs`
returns NULL outside it exactly as before. The old generator survives
in two places: `rs_default_matrix()` is still a non-static function in
`src/gf.c` (declared in `src/gf.h`) — the canonical historical-default
implementation, legitimately used by `tests/test_remat.c` (custom
RS(6,4), outside the search envelope) — and the search's seed 0 plus
the static copy in `tests/test_matrix_search.c` replicate its formula
(`M[j][i] = 2^(j*(ndata-1-i))`) for seeding and old-vs-new weight
comparison.

## Falsifiable ship criterion

Ship iff **all** of these hold:

1. `tests/test_matrix_search.c` passes: every searched matrix in the
   envelope is proven MDS by full minor enumeration (minor count ==
   C(ndata+npar, npar)), the independent solver agrees, the weight
   table pins match, and the searched weight is strictly below the old
   default's on every multi-parity corpus shape.
2. The greedy's fixpoint is brute-force verified as a true
   single-entry local optimum on the corpus (independent check, not
   the greedy's own logic).
3. The full suite is green: `make test`, `make test-asan`
   (test_remat's `P_HORNER` expectations are the known deliberate
   tripwire — see "What it is" — and must be updated by its owner).
4. All three 2.0 mutation probes fail as designed (below).

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
- **bitweight_table**: weight-table pins against hand-derived constants
  (`wt(0)=0`, `wt(1)=8`, `wt(2)=11`, `wt(3)=19`, `wt(4)=14`,
  `wt(128)=30`; derivations in the test comments, independent of the
  table code); 1 pinned as the unique minimizer over nonzero values
  (exhaustive check); bad args return -1.
- **bitweight_beats_old**: the old-vs-new weight table above, asserted:
  searched weight <= old everywhere, strictly < on every multi-parity
  shape, == on npar=1.
- **search_deterministic**: repeated calls byte-identical (cache path);
  byte pins for (8,2) and (6,3) — deliberate-change detectors: any
  future search change that moves these bytes fails loudly so the
  parity-format change is deliberate and documented; moderate envelope
  sweep (npar=2,3 × ndata ∈ {1,2,5,13,25,37,51}): search succeeds and
  never exceeds the old default's weight.
- **end_to_end**: ISA x topology x op x length combination test via the
  public API (`rj_compile`) vs an independent scalar reference:
  raid6(8), rs(6,3), rs(10,2), raid5(4); encode + 11 recovery combos;
  lengths {1, 7, 63, 64, 65, 1000, 4096}; one benchmark=1 config.
  Runs natively and under QEMU (aarch64, riscv64).
- **envelope**: NULL outside the envelope, non-NULL at the corners.

### Mutation evidence (all performed, all reverted)

- **E** — `src/matrix_search.c`: bitweight table forced constant
  (`bitw[c] = 8` for all nonzero c): the greedy degenerates (no strict
  improvement is ever possible), the winner is seed 0 = the historical
  default, and `bitweight_beats_old` FAILS on all six strict-<
  checks (e.g. `(8,2): new weight 128 not < old weight 128` under the
  mutated table). The table pins in `bitweight_table` fail first.
  Mutation reverted; suite green.
- **F** — `src/matrix_search.c`: weight-table column computation
  perturbed (column k built from `1u << ((k+1) % 8)` instead of
  `1u << k`): `bitweight_table` FAILS on the pinned constants
  (`wt(2) = 14, want 11`). Mutation reverted; suite green.
- **G** — `src/matrix_search.c`: `forbidden_values` neutered to
  "nothing forbidden": the greedy 1-izes every entry into a non-MDS
  all-ones matrix; the full-enumeration proof gate catches it and falls
  back to the verified seed (defense in depth holds — `mds_proof`
  still passes), but the suite FAILS anyway via `bitweight_beats_old`
  (no strict improvement: new weight == old weight on all six shapes)
  and the `search_deterministic` byte pins. Mutation reverted; suite
  green.

## Perf guard

Benchmarks are postponed per standing rules — no measured numbers are
claimed here. The shipped evidence is static: the weight table above
(exact integers from the deterministic search) and the structural
`W − 8·npar` gate-count identity. Note the parity format changed
deliberately (2.0), so any before/after throughput comparison must
re-baseline rather than assume byte-identity.

## Limitations

- The search is single-threaded and uninterruptible; worst case a few
  seconds once per process for (51,3) on a shared VM (vCPU contention,
  not algorithmic). Bounded (8 sweeps × 3 seeds) and cached; acceptable
  next to multi-second `rj_compile` times for the same shapes.
- The greedy is complete only for single-entry moves; multi-entry
  coordinated improvements are outside the neighborhood (stated
  exactly, not hand-waved).
- The weight objective is backend-agnostic by construction (it counts
  bit-level XORs, not vector uops). The planner still codes the new
  matrices with its existing strategies (P_MULACC for the
  non-Vandermonde winners); the bit-level scheduling track (A1) is the
  intended consumer of the lighter matrices.
- `npar > 3` is not supported by the proof (or the old default);
  constructors return NULL there, unchanged.

## History: the 1.0 uops objective

The 1.0 search (2026-10-02 – 2026-10-06) minimized a static encode-cost
model: the real encode plan built by `plan_build` (so the
P_XOR/P_HORNER/P_MULACC strategy choice was the planner's actual code),
evaluated under a cost model transcribed 1:1 from the x86-64 nibble
backend (`src/x86_64/jit.c`) at reference U=8, vl=64. Its analytical
result: the search's fixed point was the historical default
(byte-identical on the whole envelope — proven empirically by a dense
tripwire, with a one-count bound + placement-constraint + Horner-floor
argument explaining the mechanism). 2.0 supersedes the objective; the
1.0 cost model (`matrix_search_cost`, `matrix_search_cost_parts`) was
removed. 1.0 mutation evidence (A: Horner cost 7→6 caught by the
codegen cross-check; B: forced Cauchy winner caught by cost_le_old and
the format tripwire; C: neutered minor rejection caught by mds_rejects
and the proof gate; D: count-preserving skipped-minor mutation caught
by the single-singular-minor corpus) is archived here for the record;
all were performed and reverted.
