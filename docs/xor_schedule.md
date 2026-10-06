# XOR-schedule / cross-row CSE (track A3, item 1)

**Date:** 2026-10-02. **Status:** implemented, benchmark-gated twin.

## 1. What it is

For `P_XOR` plans (every matrix coefficient in {0,1}), the naive emitter
computes each parity row independently: for every input column it loads the
input once and XORs it into every row that uses it. When several rows share
a subexpression (e.g. rows `r1 = a^b^c`, `r2 = a^b^d` share `a^b`), the
naive body recomputes the shared XOR once per row.

This track builds, at **plan time**, an XOR DAG IR over the parity equations
and runs a **greedy cross-row common-subexpression elimination**, producing
an optional straight-line `xor_sched` program attached to the plan. A
schedule-driven P_XOR emitter exists in all three backends (x86-64,
AArch64, RISC-V). The scheduled emitter is a **candidate twin**:
`plan_candidates` enumerates it next to the naive per-row emitter and the
benchmark picks the winner empirically. It is never forced.

Carlo's invariant holds hard: all CSE/scheduling happens at plan/JIT time.
The emitted body contains no redundant XORs — accumulators are overwritten
(move / direct load) on first touch, never XORed with zero, and every
shared subexpression is computed exactly once per group.

## 2. IR and greedy algorithm (`src/xorsched.c`)

Atoms: inputs `0..n_in-1`, plus temps `t_k = in_a ^ in_b` (flat pairs only —
temps are never nested; see §6 for why).

1. **Row dedup.** Identical matrix rows are computed once; duplicates become
   a single vector move from the canonical row's accumulator. Pair
   co-occurrence is counted per unique row (unweighted): each unique row is
   computed once, so a pair's use-count is the number of distinct
   computations that use it.
2. **Greedy pair extraction.** Rows are multisets of atoms. Repeat: count
   co-occurrences of unordered atom pairs across (deduplicated,
   unweighted) rows — each unique row is computed once, so a pair's
   count is the number of distinct computations using it; take the pair
   with the highest count; if it is < 2, stop; create
   temp `t_new = x^y`; in every row containing both `x` and `y`, replace the
   pair by `t_new`. Stop also at 1024 temps (a larger schedule is never
   emittable at sane code size; the gate in §4 would reject it anyway), and
   skip schedule building entirely when `n_in * n_out > 16384` (pathological
   JIT-time guard for adversarial custom matrices).
3. **Row atom order.** Row `j`'s atoms are its temps in formation order
   followed by its surviving inputs in ascending order. `row_first[j]` is
   the atom materialized first in emission order (first temp, else first
   input; -1 for empty rows).

Correctness of the rewrite: in GF(2), replacing `x^y` by `t` with `t=x^y`
leaves every row's value unchanged; temps depend only on inputs, so the
formation order is a valid topological order. The schedule is validated by
differential tests (scheduled vs naive vs scalar reference), not by proof.

Emission order (per unroll lane, identical on all ISAs):

- **Phase 1 — temps**, in formation order: materialize `t_k = in_a ^ in_b`
  into the broadcast temp register, then fan out to its user rows: rows for
  which `t_k` is the first atom get a **move** (`acc = t`, overwrite —
  never xor-with-zero); all other user rows get `acc ^= t`.
- **Phase 2 — surviving input atoms**, ascending: column-major exactly like
  the naive emitter (load once per `(input, lane)` into the broadcast temp,
  fan out; single-user first-atom rows load straight into the accumulator).
- **Duplicate rows**: one vector move from the canonical row per lane.
- **Dead rows** (no atoms): zeroed by the existing prologue path, shared
  with the naive emitter.

Temp materialization per ISA (see §5 for why they differ):

| ISA | formation of `t_k = in_a ^ in_b` | temp regs |
|---|---|---|
| x86 AVX2/AVX-512 | `t = load(a); t ^= mem(b)` (folded memory XOR) | 1 (`t`) |
| x86 SSE | `t = load(a); t2 = load(b); t ^= t2` (no mem-XOR form; and `c_vpxor_mem` on SSE would clobber `t` via its internal load) | 2 (`t`, `t2`) |
| AArch64 NEON/SVE | `t = load(a); t2 = load(b); t ^= t2` (no vector XOR-with-memory) | 2 (`t`, `t2`) |
| RISC-V RVV | `vd0 = vle(a); vd1 = vle(b); vd0 ^= vd1` | 2 (`vd0`, `vd1`) |

Register pressure of the scheduled twin: `n_out*U + 1` on x86 AVX (same as
naive), `n_out*U + 2` on x86-SSE / AArch64 / RVV (one extra temp). Modeled
by `xorsched_need_regs()` (additive; `plan_need_regs` untouched — it stays
the naive twin's source of truth), asserted by every backend's register
assignment exactly like the existing pressure asserts.

## 3. Candidate integration

- `plan_t.xor_sched` (additive, nullable): the program, built in
  `plan_build` for `P_XOR` (NULL when no pair co-occurs ≥ 2, i.e. nothing
  to share — then no twin is enumerated).
- `cand_t.xorsched` (additive flag, precedent: `ternlog`/`nt`/`remat`):
  selects the schedule-driven body in the backends. Never combined with
  `ternlog` (a different P_XOR body); NT/prefetch twins of the scheduled
  twin are produced by the existing twin loops, so the full
  ISA × U × sched × nt × prefetch product is covered.
- `plan_candidates` appends one scheduled twin per fitting base candidate
  (after the ternlog twins, before the NT twins), gated per ISA by §4.
- Version `desc` gains a `+xsched` tag.

## 4. Static cost model and the ship gate

The twin reuses the existing `version_t` cost fields (`uops_per_byte`,
`mem_per_byte`); the emitters count with the same `c_*` helpers as the
naive body, so the numbers are exact, not estimated. Additionally,
`xorsched.c` predicts both twins' core vector uops at plan time under one
counting convention (per group: vector loads / moves / XORs; prefetch,
pointer setup and loop overhead excluded — prefetch is washed out by
construction: every contributing input is prefetched exactly once in both
bodies; pointer/loop overhead is identical in structure).

**Gate:** the scheduled twin is enumerated for an ISA class only if
`uops_sched < uops_naive` strictly under that convention. Per-temp saving
is `m·U` uops (`m` = user rows): naive spends `(2m+2)·U` on the pair
(2 loads + per-row 2 XORs, first row a move+1 XOR), scheduled spends
`(m+2)·U` (2 loads + 1 formation XOR + `m` fan-out ops). Inputs used both
in a temp and directly by a row are loaded twice, which the gate counts —
"dirty" sharing usually fails the gate, "clean" sharing (the temp's inputs
used nowhere else directly) keeps `mem_sched == mem_naive`.

**Falsifiable ship criterion.** The track ships iff ALL hold:

1. Differential correctness: scheduled ≡ naive ≡ scalar reference across
   topologies × lengths × alignments on x86-64, AArch64 (NEON + SVE at
   VL 128/256) and RVV (VLEN 128/256/512) — `tests/test_xorsched.c`.
   Status 2026-10-02: MET — native x86-64 702 iterations (416 full
   checks), 4 twins; NEON 234 (130 full), 3 twins; SVE 468 × 2 VLs
   (260 full each), 3 twins each; RVV 234 × 3 VLENs (156 full each),
   4 twins each; all 0 failures. Evidence: §5 and the qemurun logs.
2. Model honesty: for every enumerated twin on every backend,
   `sched.uops_per_byte < naive.uops_per_byte` strictly (same test).
3. No regression: `tools/perf_guard.c` before/after inside the baseline
   pass-to-pass bands on 64 KiB / 1 MiB.
4. Empirical relevance: on at least one sharing-heavy P_XOR config the
   scheduled twin is *measured* faster than naive in a benchmark duel
   (evidence in §7), or a documented rejection with evidence. Status
   2026-10-02: NEITHER, conclusively — the quiet-system duel (§7) shows
   +0.8% median at 64 KiB with overlapping IQRs and pure noise at 1 MiB
   on this shared VM; the earlier loaded-system "rejection" does not
   reproduce. The twin still ships: it is benchmark-gated, so
   `bench_select` can only select it where it measures faster on the
   target machine. The gate — not this criterion — is what makes
   shipping safe; the criterion is recorded as not-conclusively-met,
   not as met.

## 5. Why the temp budgets differ (ISA notes; QEMU correctness executed)

x86 is the only ISA here with a vector-XOR-with-memory form, so only x86
AVX can form `t_k = in_a ^ in_b` with one live temp (`t ^= mem(b)`); SSE
emulates mem-XOR as load-into-`t` + reg-XOR, which would clobber the `t`
holding `a`. NEON/SVE/RVV have register-only XOR. Hence the +1 temp on
SSE/AArch64/RVV. The second vector temp is `t2 = n_out*U+1` (NEON `v` /
SVE `z(16+t2)`), the second pointer is X3 (free in P_XOR bodies; the
P_DELTA X3 note applies identically). RVV uses `vd1 = vd0+1` and T4 as
the second pointer (`gen_body` already treats T2–T5 as scratch).

QEMU execution (correctness only, no perf claims) — 2026-10-02, all
0 failures, via `make test-qemu-aarch64`, `test-qemu-aarch64-sve`,
`test-qemu-riscv64` (`tests/test_xorsched.c`, bucketed counts are full
differential/model checks / no-schedule skips / reg-fit skips):
- qemu-aarch64 `-cpu max` (NEON): 234 iterations (130 / 65 / 39),
  3 twins (3 +nt, 3 +pf).
- qemu-aarch64 `-cpu max,sve-default-vector-length=128` and `=256`
  (SVE+NEON): 468 iterations each (260 / 130 / 78), 3 twins each.
- qemu-riscv64 `-cpu max` at VLEN 128/256/512 (RVV): 234 iterations
  each (156 / 78 / 0), 4 twins each (4 +nt, 4 +pf).
The combo exactness checks (xsched+nt / xsched+pf vs model), the OOM
unwind sweep (25 allocation sites), and the shipped-path e2e test
(`test_xorsched_e2e.c`) also ran on every QEMU config above, 0 failures.

## 6. Dominance-pruning safety (for the parallel pruning track)

Claim: **a dominance-pruning pass can never drop a winning scheduled twin
in favor of its naive twin.** Checkable form:

- **P1 (strict static win).** Whenever the twin is enumerated,
  `uops_sched < uops_naive` strictly — enforced at plan time by the §4
  gate under the emitter's own counting convention, and re-verified on the
  built versions by `test_xorsched.c`. Therefore the naive twin can never
  *dominate* the scheduled twin on `(uops, mem)`: it is strictly worse on
  the primary metric. Any rule of the form "drop B if A dominates B" with
  A = naive, B = scheduled can never fire.
- **P2 (same traffic shape).** `mem_sched` vs `mem_naive`: both count every
  vector load/store exactly; the twin loads each contributing input at
  least once per group like naive, plus at most one extra load per
  (temp, input) incidence for inputs shared between temps and direct row
  uses. The pruning track can read both numbers off the built versions;
  no hidden traffic exists (pool/loop overhead is plan-kind-constant).
- **P3 (no silent model drift).** The gate prediction and the emitter share
  the counting convention by construction (the test breaks if they drift:
  it compares built-version fields, which come from the emitter, against
  the strict inequality the gate promised).
- **P4 (twin-only scope).** The claim is relative to the naive twin only.
  Pruning *across* ISA levels or U values is the pruning track's general
  problem and is unaffected by this track: the scheduled twin is just
  another candidate with exact static fields.

Flat-pair restriction (no nested temps) is deliberate: it bounds the live
set to the two formation temps, keeps the pressure formula exact, and every
extracted temp provably saves `≥ 1` XOR uop (needs `≥ 2` user rows).

### Nested-extraction evaluation (2026-10-06): deliberately NOT implemented

A nested greedy (temp-input and temp-temp pairs, `t1 = t0^c`) was
prototyped standalone (`tools/nested_cse_experiment.c`: nested greedy +
honest AVX-convention static uop counting + scalar differential
correctness check per schedule) and measured on 12 topologies: the
repo's six xorsched differential-test topologies (t1-t6 from
`tests/test_xorsched.c`), six synthetic deeply-structured/dense plans,
and a real RDP-recovery P_XOR plan shape. Static uops per lane
(AVX: temp formation 2 uops with an input operand, 1 uop temp-temp;
fan-out 1 uop/row):

| topology | naive | flat | nested | delta | nested maxlive |
|---|---|---|---|---|---|
| repo t1 (4x20) | 24 | 26 | 25 | -1 | 3 |
| repo t2 (3x6) | 16 | 13 | 12 | -1 | 3 |
| repo t3 (3x6) | 10 | 11 | 11 | 0 | 1 |
| repo t5 (4x8) | 8 | 16 | 16 | 0 | 0 |
| repo t6 (6x12) | 36 | 34 | 34 | 0 | 1 |
| deep-share 4x7 | 19 | 19 | 16 | -3 | 2 |
| chain 4x7 | 21 | 19 | 17 | -2 | 2 |
| rs-like 3x9 | 27 | 21 | 18 | -3 | 2 |
| deep-core 8x20 | 52 | 52 | 45 | -7 | 3 |
| dense-rand 6x14 | 60 | 52 | 45 | -7 | 6 |
| dense70 4x20 | 75 | 53 | 42 | -11 | 6 |
| rdp-like 6x20 | 39 | 49 | 49 | 0 | 1 |

Verdict: on the differential-test topologies nested saves 0-1 uop/lane
(0-8%) -- not a win by the twin's own gate standard. The 11-21% wins
concentrate in synthetic dense plans, but those need up to 6
simultaneously-live temps vs flat's 1-2: at `n_out*U + 6` pressure the
feasible U drops (e.g. n_out=4: U=7 -> U=6 on AVX-512), and the unroll
loss can erase the static win in measured performance. Real P_XOR plans
do not show the dense structure either (RDP recovery is sparse: 2x46
with 6 ones/row; RAID-5/6 P_XOR is small). A production nested twin
would need a temp-stack emitter on all three backends, a liveness-based
pressure model replacing the exact `+1`/`+2`, and re-proof of
model-emitter exactness -- disproportionate risk for 0-1 uop on the
tested topologies. The flat bound stands; re-test trigger: a real
P_XOR workload class with dense row sharing (3+ common inputs across
many rows) where the twin is benchmark-relevant.

## 7. Measured evidence (2026-10-02, AMD EPYC 9D25)

Twin-vs-naive benchmark duel on a sharing-heavy synthetic P_XOR config:
20-in/4-out, 2 shared pairs → 2 temps (the `t1` matrix), ISA **AVX-512**,
**U=4**, through the real `bench_select` harness
(`tools/duel_xorsched.c`), taskset-pinned to one core, 12 runs/arm,
median+IQR per arm. Two working-set configs: `bench_bytes=64 KiB`
(harness sizes {16K, 64K, 256K}; 64K is cache-fit) and `bench_bytes=1
MiB` (sizes {256K, 1M, 4M}; 1M is streaming).

Static (model-exact, confirmed by `test_xorsched` on every backend):
**0.3711 vs 0.4648 uops/B** — a 20.1% static win for the scheduled twin.

- 64 KiB: naive median 82.98 GB/s, IQR [76.30, 84.23]; sched median
  83.66 GB/s, IQR [76.90, 84.43]. Sched faster in 10/12 paired runs,
  median +0.8%.
- 1 MiB: **inconclusive**. Shared-VM memory-bandwidth noise dominates:
  per-run throughput varies ~4x *within* each arm (10–64 GB/s).
  Medians 21.61 (naive) vs 19.10 (sched) GB/s with fully overlapping
  IQRs ([12.79, 52.40] vs [14.08, 60.48]).

The earlier loaded-system observation ("slower or tied, never faster",
n=3/arm, no pinning, high noise) does **not** reproduce under taskset
pinning: at 64 KiB the twin is never slower beyond noise. But the +0.8%
median edge is smaller than the run-to-run IQR width, so this is **not**
a conclusive measured win either. Honest status: no conclusive
measurement in either direction on this shared VM; the twin is retained
**benchmark-gated** — `bench_select` picks it only where it measures
faster on the target machine, never forced. Raw logs:
`docs/xor_duel_2026-10-02_64K.log`, `docs/xor_duel_2026-10-02_1M.log`.

## 8. Tiling-study reconsideration

`docs/tiling_study.md` rejections P1/P2/P4 concern loop tiling and row-major
restructuring (load inflation: tiling multiplies input loads by
`n_out/T`). This track does **not** change those premises: the scheduled
body keeps the input-major shared-input structure (each contributing input
loaded once per `(phase, lane)`; temps add only their two formation
loads). The study's "only per-row op fusion moves the needle" line is
superseded *for P_XOR* by construction: the twin is benchmark-gated, so if
cross-row CSE never wins on hardware, the twin simply never gets selected
— the empirical gate is the re-test. No tiling-study experiment is
re-run by this track; the later reconsideration-audit track owns that.

## 9. Limitations

- The twin only triggers for multi-row `P_XOR` with pair sharing across
  rows. Single-row P_XOR (RAID-5 encode, single-disk rebuild — i.e. every
  P_XOR config in the current perf-guard corpus) gets no twin: the schedule
  is NULL and enumeration is byte-identical to before.
- Greedy, not optimal (standard pair-frequency greedy; optimal XOR
  straight-line minimization is NP-hard). The benchmark gate, not
  optimality, decides. Caveat: the greedy can extract non-paying temps
  (a pair whose extraction costs more in formation loads than it saves
  in fan-out), and the strict static-win gate is the SOLE guard against
  that — it rejects the twin, so a non-paying extraction can never
  ship. Conversely, a config where the model says tie (or loss) but the
  twin would win on hardware is silently missed: a perf opportunity
  left on the table, never a miscompile. Pair counting is unweighted
  across deduplicated rows (multiplicity ignored), which can steer the
  greedy toward a lower-saving pair than a weighted count would.
- `n_in * n_out > 16384` or > 1024 extracted temps: no schedule (JIT-time
  guard); falls back to naive silently.
- Row-dedup moves assume the canonical row's accumulator is final before
  the duplicate move — enforced by emission order (all unique rows fully
  accumulated before duplicate moves).
