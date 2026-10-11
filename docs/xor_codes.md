# XOR array codes for RAID-6: RDP, EVENODD, Liberation

## 1. Design

RAID-6 needs two parity disks. The default RS code (`rj_topo_raid6_rs`)
multiplies by GF(2^8) constants (Horner/GFNI path). The three XOR array
codes compute both parities with **XOR only**, no finite-field multiplies:

| code       | parameter | data disks k | rows/disk R | constraint (validity gate) |
|------------|-----------|--------------|-------------|----------------------------|
| RDP        | p         | p-1          | p-1         | p prime, p >= 3            |
| EVENODD    | p         | p            | p-1         | p prime, p >= 3            |
| Liberation | k         | k            | w           | w prime, w >= 3, 2 <= k <= w |

`w` for Liberation is the smallest prime >= max(k,3) (canonical choice;
larger w is asymptotically closer to optimal per Plank FAST'08 Fig. 11,
but also widens the stripe).

### 1.1 Lowering to the existing P_XOR plan path

Array codes mix bytes **across rows** of a data disk (diagonal parity),
which the disk-stream model (`out[j] ^= in[i]*M[j][i]` at one byte
offset) cannot express. Lowering: each **row** becomes a logical disk
("expanded model"):

- expanded `ndata = k*R`, `npar = 2*R`
- `in[d*R + r]` = row `r` of data disk `d` (`0 <= d < k`, `0 <= r < R`)
- `out[r]` = P parity row `r`; `out[R + r]` = Q parity row `r`

Every parity row is then a pure XOR of input rows, so the expanded
matrix is binary and `plan_build()` selects the existing **P_XOR** path
automatically — encode, recovery (the generic GF(2^8) erasure solver;
its solution of a binary system is binary, so recovery stays pure-XOR),
everything. **No separate plan path was built.** A whole-disk failure of
data disk `d` = failure of expanded disks `d*R .. d*R+R-1`
(`rj_xorcode_rows()` reports `R`; 0 for non-XOR topologies).

Trade-off of the expanded model: `2*R` parity outputs raise register
pressure (`P_XOR` needs `n_out*U + 1` regs), so large codes run at U=1.
The benchmark (§4) measures whether the XOR-only arithmetic still wins.

### 1.2 Constructions (from the primary sources)

**RDP** — Corbett et al., "Row-Diagonal Parity for Double Disk Failure
Correction", FAST '04, §4. `p` prime > 2, `k = p-1` data disks,
`R = p-1` rows (imaginary row `p-1` = 0):

- `P_j = XOR_{i=0}^{p-2} d[i][j]`
- Diagonal set `s = { (disk i, row k) : (i+k) mod p = s }`; diagonal
  `p-1` is the missing diagonal (not stored). Crucially, **each diagonal
  parity block covers the data AND row-parity blocks on its diagonal**
  (paper §4: "Each diagonal parity block contains the even parity of the
  data and row parity blocks in the same diagonal"). Substituting the
  row-parity definition (its diagonal-`s` block sits at row
  `(s-(p-1)) mod p = (s+1) mod p`):
- `Q_s = XOR_{i=0}^{p-2} d[i][(s-i) mod p] ^ XOR_{i=0}^{p-2} d[i][(s+1) mod p]`,
  dropping imaginary-row terms.

**EVENODD** — Blaum et al., IEEE Trans. Computers 1995 (formulation as in
the "Code Design for Dependable Systems" text, Example 14.2). `p` prime,
`k = p` data disks, `R = p-1` rows (imaginary row `p-1` = 0):

- `P_j = XOR_{i=0}^{p-1} d[i][j]`
- `S = XOR_{i=1}^{p-1} d[i][(p-1-i) mod p]` (the "missing diagonal")
- `Q_j = S ^ XOR_{i=0}^{p-1} d[i][(j-i) mod p]`, dropping imaginary terms.

Without the `S` adjustment the diagonal parities would not give distance
3; with it the code is MDS.

**Liberation** — Plank, "The RAID-6 Liberation Codes", FAST '08, §3.3.
`w` prime > 2, `w >= k`, `k` data disks, `R = w` rows:

- `P_j = XOR_{i=0}^{k-1} d[i][j]`
- `Q_r = XOR_{i=0}^{k-1} (X_i row r · d[i])`, with `X_0 = I_w` and
  `X_i = I_w^→i + O_w[y, y+i-1]`, `y = i*(w-1)/2 mod w`.
  `I_w^→i` = identity with columns rotated right by `i` (row `r` has its
  1 at column `(r+i) mod w`); `O_w[y,y+i-1]` adds one more 1 at
  `(row y, col (y+i-1) mod w)` for `i >= 1`. The two 1s never coincide.
  Check against the paper: the Q bitmatrix has exactly `kw + k - 1`
  ones (§3.3), verified in `test_matrices`.

### 1.3 Falsifiable ship criterion

This track ships iff **all** hold:

1. [tests/test_xorcode.c](../tests/test_xorcode.c) passes: validity boundaries (prime/non-prime,
   `k/w` envelope, 256-disk cap), matrix unit tests (binary, Liberation
   ones-count, exhaustive 2-disk MDS via `solve_recovery` for
   RDP/EVENODD/Liberation at small params), encode vs scalar `ref_run`
   across codes × params × boundary lengths × {benchmark, static}
   selection, recovery of every 2-data-disk pair (+ data+P, data+Q, P+Q,
   the parity-involved extras spot-sampled by design — check_mds is
   exhaustive over all whole-disk pairs through the same solver)
   bit-exact, max-config JIT smoke (rdp/evenodd/liberation(13): encode +
   one 2-disk recovery), synthetic-CPU × every-candidate × length combination
   matrix, and [docs/xor_code_default.md](xor_code_default.md) ↔ C table agreement.
2. Every new test is demonstrably able to fail (mutation evidence in
   §5).
3. `make test`, `make test-asan`, `make test-qemu` green in this copy
   (with the §6 caveat on pre-existing tests that assert the old
   `rj_topo_raid6` = RS default).
4. Code-choice benchmark (§4) run on the host; winners recorded in
   [docs/xor_code_default.md](xor_code_default.md); `rj_topo_raid6` dispatches accordingly.
5. Perf guard: no regression vs the checked-in baseline on 64 KiB/1 MiB
   for configs whose topology did not deliberately change (§7).

## 2. API

Constructors (declared in [include/raidjit.h](../include/raidjit.h); internal helpers —
choice table, matrix builders, validity predicates — in [src/xorcode.h](../src/xorcode.h)):

- `rj_topo_rdp(int p)` — NULL unless `p` prime ≥ 3 and `p^2-1 <= 256`.
- `rj_topo_evenodd(int p)` — NULL unless `p` prime ≥ 3 and
  `(p+2)(p-1) <= 256`.
- `rj_topo_liberation(int k)` — NULL unless `k >= 2` and
  `w*(k+2) <= 256` for `w` = smallest prime ≥ max(k,3).
- `rj_topo_raid6_rs(int ndata)` — the previous RS RAID-6, unchanged
  (name chosen to read as "raid6, RS variant"; alternatives considered:
  `rj_topo_raid6_gf` (less clear), keeping `rj_topo_rs(ndata,2)` (loses
  the RAID-6 naming). Documented here per the task.)
- `rj_topo_raid6(int ndata)` — now dispatches to the benchmark-selected
  default per [docs/xor_code_default.md](xor_code_default.md) (RS fallback when no XOR code
  applies/wins). Bounds unchanged (`3 <= ndata <= 51`, NULL outside).
  **When an XOR code wins, the topology uses the expanded row model**
  (pointer contract in §1.1), not one pointer per data disk.
  Table constraint: a non-RS entry must name a code whose constructor
  accepts the corresponding ndata (RDP: `ndata+1` prime; EVENODD:
  `ndata` prime; Liberation: `w*(ndata+2) <= 256`); as defense in depth
  `rj_topo_raid6()` falls back to RS rather than returning NULL if an
  entry ever violates this, and `test_default_table` rejects bad entries
  at test time.
- `int rj_xorcode_rows(const rj_topology *t)` — rows per disk `R` for
  XOR-code topologies, 0 otherwise (lets callers map whole-disk failures
  to expanded indices).

## 3. Verification summary

(See §5 for mutation evidence; full counts in the test report.)

- Exhaustive 2-disk MDS: every pair among {data disks, P, Q} is
  solvable by `solve_recovery` for RDP(3,5,7), EVENODD(3,5,7),
  Liberation (2,3),(3,3),(4,5),(5,5),(6,7),(7,7). This checks the
  constructions themselves, independent of the JIT.
- JIT encode vs `ref_run`: 8 topologies × 7 boundary lengths
  (0,1,63,64,65,1000,4096) × {benchmark, static} selection.
- JIT recovery bit-exact: every 2-data-disk pair for rdp(5),
  evenodd(5), liberation(4), liberation(6), plus data+P, data+Q, P+Q and
  single-disk, at 1000 B and 65536 B; survivors verified unclobbered.
- Combination matrix: every candidate offered by `plan_candidates` on
  synthetic x86 CPUs (sse3, avx2, avx512±gfni) — and neon/sve, rvv under
  QEMU — JITed directly and checked vs `ref_run` at all boundary
  lengths.
- Max-config JIT smoke: rdp(13)/evenodd(13)/liberation(13) (up to 26
  parity outputs) — encode + one 2-disk recovery end to end vs
  `ref_run`, static selection. CPUs that cannot encode a config
  (register pressure on 16-reg ISAs) must fail cleanly — `rj_compile`
  returns NULL — asserted via the `plan_candidates` oracle, so the
  failure mode is loud in both directions.
- `rj_topo_raid6` dispatch and the C default table agree with
  [docs/xor_code_default.md](xor_code_default.md) for every ndata 3..51 (parsed at test
  time).

## 4. Code-choice benchmark (host: AMD EPYC 9D25, AVX-512+VBMI+BITALG+GFNI)

Protocol: each (code, ndata) compiled with the standard candidate
benchmark (`benchmark=1`, `bench_bytes=128 KiB`); the winner verified
against the scalar reference, then timed at fixed **total data bytes**
(512 KiB and 8 MiB) so codes with different stream counts compare
fairly. 2 warmup + 11 timed iterations; median data-GB/s. Two full
sweeps plus a focused interleaved A/B for the contested ndata.

Data-GB/s at 8 MiB total (streaming), sweep 1 / sweep 2:

| ndata | RS | RDP | EVENODD | Liberation |
|-------|----|-----|---------|------------|
| 3 | 25.33 / 27.81 | — | 25.44 / 26.87 | 26.42 / 8.63 |
| 4 | 34.09 / 30.40 | 25.57 / 29.49 | — | 23.50 / 28.86 |
| 5 | 39.68 / 31.00 | — | 22.37 / 25.77 | 23.42 / 33.94 |
| 6 | 38.60 / 27.94 | 23.54 / 37.06 | — | 30.38 / 30.04 |
| 7 | 30.67 / 38.08 | — | 36.29 / 27.09 | 36.31 / 32.43 |
| 8 | 38.87 / 31.10 | — | — | 29.15 / 24.94 |
| 10 | 35.33 / 34.46 | 23.36 / 15.09 | — | 19.62 / 19.10 |
| 11 | 31.13 / 33.07 | — | 3.34 / 18.77 | 26.21 / 28.08 |
| 12 | 35.39 / 33.92 | 21.50 / 20.84 | — | 19.67 / 22.67 |
| 13 | 45.49 / 41.25 | — | 20.20 / 20.91 | 21.74 / 18.52 |

The EVENODD(11) sweep-1 cell of 3.34 GB/s is an obvious mis-pick by the
internal candidate benchmark (sweep 2 picks a sane kernel at 18.77);
treat that cell as noise, not signal.

### Contested cells (all four, disclosed)

The 8 MiB sweeps disagree at four ndata; each is listed here with the
numbers and the leg the resolution rests on:

- **ndata=3**: sweep 1 suggested Liberation (26.42 vs RS 25.33); sweep 2
  "reversed" it (8.63 vs 27.81) — but that reversal is the admitted
  mis-pick artifact (the sweep-2 run picked a `U=1` kernel for
  liberation(3); 26.42 with `U=5` in sweep 1), so sweep 2 carries no
  weight here. Resolution rests on the interleaved A/B (RS 20.82 vs
  Liberation 19.54 — quartiles [17.7,22.5] vs [9.2,20.9] overlap, so
  the A/B cell alone is **not** decisive) together with the 512 KiB leg
  (sweep 1), where RS wins (1.3–2× across ndata per that leg).
- **ndata=5**: sweep 2 shows Liberation 33.94 > RS 31.00. Resolution
  rests on the A/B (RS 27.19 vs Liberation 20.95) plus the 512 KiB leg.
- **ndata=6**: sweep 2 shows RDP 37.06 > RS 27.94. Resolution rests on
  the A/B (RS 22.31 vs RDP 19.17, Liberation 20.81) plus the 512 KiB leg.
- **ndata=7**: sweep 1 showed EVENODD 36.29 and Liberation 36.31 >
  RS 30.67; sweep 2 reversed (27.09 / 32.43 vs 38.08). Resolution rests
  on the A/B (RS 26.32 vs EVENODD 14.89, Liberation 12.65 — but note the
  A/B run picked an `sse U=1` kernel for evenodd(7) on this AVX-512
  host, a flagged methodology weakness) together with the 512 KiB leg
  (sweep 1: 69.42 vs 44.43/43.18).

At 512 KiB (**sweep 1**; kernels for this leg were not recorded, and it
is a single sweep, so read it as one noisy leg, not ground truth) RS
wins every ndata by 1.3–2× (e.g. ndata=13: 77.77 vs 40.85/45.63;
ndata=7: 69.42 vs 44.43/43.18).

Interleaved A/B (each code compiled once, then 15 timing rounds
rotating across codes to cancel machine drift), median [q1, q3]:

| ndata | RS | RDP | EVENODD | Liberation |
|-------|----|-----|---------|------------|
| 3 | 20.82 [17.7, 22.5] | — | 19.38 [3.4, 23.5] | 19.54 [9.2, 20.9] |
| 5 | 27.19 [21.5, 27.6] | — | 20.05 [2.1, 24.5] | 20.95 [3.5, 25.4] |
| 6 | 22.31 [13.2, 28.1] | 19.17 [1.3, 23.1] | — | 20.81 [18.9, 24.3] |
| 7 | 26.32 [3.5, 29.1] | — | 14.89 [3.2, 16.8] | 12.65 [1.4, 19.3] |

Kernel caveat on the A/B table: which kernel each code actually ran in
each cell was **not** recorded per cell — only the two flagged
mis-picks are known (liberation(3) ran a garbage `U=1` kernel in
sweep 2; evenodd(7) ran an `sse U=1` kernel on this AVX-512 host in the
A/B). Typical picks were RS → `horner avx512+gfni` and the XOR codes →
`xor avx512` (mostly `+ternlog`), but without per-cell records the A/B
cells at ndata=3 and ndata=7 — where the quartile ranges overlap
heavily — must **not** be read as decisive. The load-bearing leg at
ndata=3 and ndata=7 is the 512 KiB (sweep 1) leg, where RS leads by a
clear margin. The internal candidate benchmark is itself noisy for the
high-stream-count XOR topologies (the two mis-picks above); that is a
pre-existing benchmark-methodology weakness, flagged, not fixed here.

**Conclusion: no XOR code shows a stable, repeatable win over RS at
any ndata — for encode, on this host/ISA (AMD EPYC 9D25,
AVX-512+VBMI+BITALG+GFNI).** Sweep 1 suggested liberation(3) and
liberation/evenodd(7); the A/B and the 512 KiB leg resolve all four
contested ndata (3, 5, 6, 7) in RS's favor as detailed above — note the
sweep-2 "reversal" at ndata=3 is itself the admitted mis-pick artifact,
not independent evidence. RS wins decisively at ndata ∈ {4, 8, 10, 11,
12, 13} in both 8 MiB sweeps and at 512 KiB everywhere. Recovery was
not benchmarked (correctness only), and the default table is global,
not per-CPU. The default table is therefore all-RS, with measured
evidence (this section) instead of the provisional placeholder.

Why RS wins here is unsurprising: the XOR codes do ~3 XORs per data
byte with 2R parity streams (high register pressure, often U=1),
while RS does ~1 XOR + 1 GFNI multiply per byte — and `vpgfni` on this
host is fast enough that the multiply is not the bottleneck. The
classical XOR-code advantage dates from slower GF multipliers.

**Provenance gap (recorded 2026-10-02):** the raw benchmark logs and the
code-choice benchmark script from the original runs were **not
preserved** — only the summary tables in this section survive. The
numbers here therefore cannot be re-derived or audited cell by cell. The
integrator's post-merge re-run of the code-choice benchmark **must**
preserve both (the raw logs and the exact script used) alongside the
updated [docs/xor_code_default.md](xor_code_default.md) table. Designing that re-run protocol
is the integrator's job, not this track's.

> **Integrator: re-run this benchmark on the merged tree.** Track A3's
> XOR-schedule/CSE over the P_XOR path directly attacks the XOR codes'
> main cost (redundant XORs across the 2R parity rows) and is expected
> to make them more competitive — defaults recorded in
> [docs/xor_code_default.md](xor_code_default.md) may flip.

## 5. Mutation evidence

Each mutation below was applied to the track's code, the relevant test
re-run to show it FAILS, then reverted:

1. RDP Q: drop the substituted row-parity diagonal term
   (`r2` contribution) → exhaustive MDS check fails
   (e.g. rdp: disks 1+2 not recoverable).
2. EVENODD: drop the `S` (missing diagonal) term → MDS check fails.
3. Liberation: drop the `O_w` extra 1 → Q ones-count `kw+k-1` check
   fails and MDS check fails.
4. Default table: flip one [docs/xor_code_default.md](xor_code_default.md) entry →
   table-agreement test fails.
5. Constructor: force `rj_topo_rdp` to return NULL for prime `p` →
   validity test fails.
6. `rj_xorcode_rows` returning 0 for an XOR topology → validity test's
   rows check fails (would also break callers mapping whole-disk
   failures to expanded indices).

## 6. Known cross-track interaction

`rj_topo_raid6()` no longer means "RS RAID-6 with `ndata=k` 1-pointer-
per-disk streams" unconditionally — when the default table names an XOR
code, the topology uses the expanded row model (`ndata=k*R`, §1.1).
Today the table is all-RS so nothing breaks, but if the integrator's
re-run flips an entry, every pre-existing caller that assumes RS
semantics must be updated to the `rj_topo_raid6_rs()` opt-in. Only
`ndata <= 13` can ever flip (Liberation caps at `k <= 13`; RDP needs
`ndata+1` prime; EVENODD needs `ndata` prime); `ndata >= 14` callers are
structurally safe. Checked 2026-10-02; the integrator owns the fixes:

**Confirmed RS assumption (breaks loudly on a flip):**

1. [tests/test_boundaries.c::case_encode_raid6](../tests/test_boundaries.c) (~line 419):
   `rj_topo_raid6(5)` asserted to lower to `P_HORNER` — under an XOR
   default the plan kind becomes `P_XOR` and the assertion fails.
   Same file lines ~502, ~524 use `rj_topo_raid6(5)` for other cases.
2. [tests/test_codegen_shape.c](../tests/test_codegen_shape.c) F1/F5 (~lines 429, 531 — `rj_topo_raid6(6)`;
   also ~776, ~819, ~990 use `rj_topo_raid6(6)`/`rj_topo_raid6(4)`):
   asserts P_HORNER-specific codegen facts (one input load per (i,u),
   vzeroupper placement) that do not hold for a P_XOR plan.

**At-risk call sites (flippable ndata; assumption class noted — the
integrator checks each for plan-kind asserts, failed-index literals
assuming parity at `ndata`/`ndata+1`, and `ndata == k` stream counts):**

3. [tests/test_correct.c:286](../tests/test_correct.c) `rj_topo_raid6(8)` — failed-index
   literals `{8}`, `{9}`, `{8,9}` assume parity disks at indices 8, 9;
   under the expanded model those indices are data rows (the helper is
   self-consistent on buffer counts, so this would silently test the
   wrong disks, not crash). Also [`:304`](../tests/test_correct.c) `rj_topo_raid6(3)`.
4. [tests/test_matrix.c:1516-1517](../tests/test_matrix.c) (`rj_topo_raid6(4)`, `(8)`),
   [tests/test_benchreject.c:39](../tests/test_benchreject.c) (`(8)`), [tests/test_scan_align.c](../tests/test_scan_align.c):376, 405,422 (`(8)`), [tests/test_nt.c:345](../tests/test_nt.c) (`(8)`),
   [tests/test_pdelta.c:89-174](../tests/test_pdelta.c) (`(4)` ×8), [tests/test_pressure.c:282](../tests/test_pressure.c)
   (`(4)`), [tests/test_remat.c:265](../tests/test_remat.c) (`(8)`),
   [tests/test_sve_pdelta.c:179](../tests/test_sve_pdelta.c) (`(4)`), [tests/test_aarch64_remat.c:283](../tests/test_aarch64_remat.c)
   (`(8)`), [tests/test_sve_eor3.c:249,307,321](../tests/test_sve_eor3.c) (`(6)`, `(4)`),
   [tests/test_pdelta_edge.c](../tests/test_pdelta_edge.c):373,408,540 (`(8)`), :658,669 (`(4)`).
5. [tools/perf_guard.c](../tools/perf_guard.c) corpus `raid6(8+2)/*` — goes through
   `rj_topo_raid6()`; self-consistent (derives stream counts from the
   topology), but the measured object changes meaning under a flip.

**Structurally safe (can never be an XOR topology):** `ndata=51/52`
bounds checks ([tests/test_correct.c:396-399](../tests/test_correct.c),
[tests/test_pdelta_edge.c:308,689](../tests/test_pdelta_edge.c)), [tools/perf_guard.c](../tools/perf_guard.c)
`raid6(16+2)` (ndata=16 > 13), and the study/dump tools (not built by
`make`).

This file ownership did not include those tests, so they are untouched
here.

## 7. Perf guard

[tools/perf_guard.c](../tools/perf_guard.c) corpus configs `raid6(8+2)/*` go through
`rj_topo_raid6()`: if the default for ndata=8 flips to an XOR code, the
guard measures the expanded topology (self-consistent: it derives
stream counts from the topology). Throughput is compared per data byte
(`0.8 ×` reported GB/s both before and after, since both have 2
parities per 8 data disks). `raid6(16+2)` keeps RS (no XOR code applies
within the disk cap).

No before/after numbers were recorded in this track: with the all-RS
default table the dispatch resolves every guard config to
`rj_topo_raid6_rs()`, whose construction is byte-identical to the
pre-track `rj_topo_raid6()` (same name, same `rs_default_matrix` call —
verified against the base tree), so the measured topologies and their
kernels cannot differ from baseline; the only new code on the path is
the cold dispatch table lookup. The integrator re-runs the guard on the
merged tree, where flipped defaults can change the measured object.
