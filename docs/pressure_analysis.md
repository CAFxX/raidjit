# Register Pressure: Binding Analysis and Advanced Mitigations

Date: 2026-10-01. Context: register-pressure binding analysis; the
non-power-of-2 unroll study (U=9..12) is complete -- rigorous
repeated/interleaved measurement found no significant win over U=8, so
candidate lists stay {8,7,6,5,4,3,2,1} on x86/NEON and {5..1} on SVE/RVV.

## 1. Where pressure actually binds

Measured via `plan_need_regs()` (exact per-candidate accounting; asserted
by each backend for every variant it implements, including remat on all
three ISAs). "maxU" = largest U with need <= regfile (formula bound;
candidate lists additionally cap offered U at 8 on x86/NEON, 5 on
SVE/RVV).

**AVX-512 (32 regs):**

| Plan | n_out | Method | maxU | Status |
|---|---|---|---|---|
| rs(6,3) HORNER | 3 (2 nontrivial) | nibble | 7 | FIXED: U=7 added to candidate list |
| rs(6,3) HORNER | 3 | GFNI | 9 | U=8 offered; U=9 fits but skipped (diminishing returns) |
| rs(10,4) MULACC | 4 | nibble | 6 | TRULY BOUND: U=7 needs 35 regs |
| rs(10,4) MULACC | 4 | GFNI | 7 | FIXED: U=7 added |
| 5-out MULACC | 5 | GFNI | 5 | BOUND: U=6 needs 33 regs |

**16-reg files:** exact formula maxU from `plan_need_regs` (worst case
ntriv = n_out; regs used at maxU in parentheses; "--" = no U >= 1 fits).
SSE/AVX2 use the x86 formulas, NEON the AArch64 ones (HORNER differs).

SSE/AVX2 (XOR `nU+1`, MULACC `nU+7`, HORNER nibble `nU+6+2*ntriv`):

| n_out | XOR | MULACC | HORNER nibble |
|---|---|---|---|
| 1 | 15 (16) | 9 (16) | 8 (16) |
| 2 | 7 (15) | 4 (15) | 3 (16) |
| 3 | 5 (16) | 3 (16) | 1 (15) |
| 4 | 3 (13) | 2 (15) | -- (U=1 needs 18) |

NEON (XOR `nU+1`, MULACC `nU+7`, HORNER nibble `nU+5+2*ntriv`):

| n_out | XOR | MULACC | HORNER nibble |
|---|---|---|---|
| 1 | 15 (16) | 9 (16) | 9 (16) |
| 2 | 7 (15) | 4 (15) | 3 (15) |
| 3 | 5 (16) | 3 (16) | 1 (14) |
| 4 | 3 (13) | 2 (15) | -- (U=1 needs 17) |

RAID-5 (n_out=1) and RAID-6 (n_out=2) on AVX-512 never bind (maxU >= 12).

### 1b. Coefficient-aware allocation (2026-10-02)

`plan_need_regs` and all three backends now key the scratch-register
overhead off `plan_any_nontrivial()` (src/plan.h: 1 iff some
coefficient is outside {0,1}). When every coefficient is in {0,1} the
nibble split (and the GFNI affine table) never executes, so the
unconditionally-allocated temps were dead registers. Audit result: only
P_DELTA reaches this case from the planner. An all-{0,1} Vandermonde
matrix is itself all-{0,1}, so `plan_build` classifies it P_XOR before
the HORNER check (P_HORNER therefore always has a mask); P_MULACC is
guaranteed a nontrivial coefficient. HORNER/MULACC formulas are
unchanged; P_MULACC's all-trivial branch is unreachable from the
planner (kept consistent for hand-built plans).

Overhead under all-trivial coefficients (n = n_out; old in parens):

| backend / ISA | P_DELTA | P_MULACC (planner-unreachable) |
|---|---|---|
| x86 nibble, AVX2/AVX-512 | nU+1 (+7) | nU+1 |
| x86 nibble, SSE | nU+2 (+7); r1 survives as the delta temp | nU+1 |
| x86 GFNI | nU+1 (+3); t2/tbl dead | nU+1 |
| AArch64 nibble (NEON/SVE) | nU+2 (+7); r1 = delta temp | nU+1 |
| RVV nibble | (n+1)U+2 (+7); tlo/thi = delta temps | (n+1)U |

The delta temp is NOT dead: on SSE the two-operand old^new needs r1,
on AArch64/NEON+SVE r1 holds the delta, on RVV tlo/thi are borrowed as
delta temps; on x86-AVX the delta folds into a memory XOR and on GFNI
the SSE path never runs. Each backend asserts its numbering against
`plan_need_regs`, so allocation and model cannot drift.

Planner-visible effect (16-reg files, all-trivial P_DELTA, n_out=2,
e.g. a custom all-ones 2-parity topology): SSE goes from U<=4 to U<=7
offered (2U+7<=16 was, 2U+2<=16 now); AVX2 likewise U<=4 to U<=7
(2U+1<=16 now). GFNI is AVX-512-only (32 regs), where U=8 was already
offered before and after -- no change there. RAID-5 parity update
(n_out=1) already offered U=8 everywhere. RVV keeps the U<=5
candidate cap; the win there is headroom, not new candidates.
`tests/test_pressure.c` §§5-7 and `tests/test_codegen_shape.c`
`test_delta_trivial` pin the exact values, the empty constant pool,
and (SSE) execution of the kept delta temp.

## 2. Root-cause fix applied: U=7

The binding analysis revealed the actual gap was not a fundamental
pressure problem but a **missing candidate**: U=7 fits where U=8 does not
(rs(6,3) Horner nibble: 31 vs 34 regs; rs(10,4) MULACC GFNI: 31 vs 35
regs), yet the candidate list was {8,6,5,4,3,2,1}.

Fix: candidate lists are now {8,7,6,5,4,3,2,1} on x86 and NEON
(SVE/RVV stay at {5..1}; no binding evidence there yet). The pressure
filter drops U=7 where it doesn't fit, so this is purely additive.
`tests/test_pressure.c` pins the new boundary.

## 3. Advanced options evaluated

### 3a. Output-row tiling

Process `t < n_out` output rows per pass, rereading inputs per tile.
Register need drops from `n_out*U + overhead` to `t*U + overhead,
allowing larger U per tile.

Example: rs(10,4) MULACC nibble (n_out=4): untiled maxU=6 (31 regs);
t=2 tiling allows U=12 (31 regs) at 2x input reads.

When it helps: compute-bound kernels (nibble, large n_in) where extra
input reads are cheap relative to GF multiply uops. Likely ~neutral for
GFNI (1 uop/mul) and harmful for XOR (memory-bound).

Cost: new loop nest in every backend, tile-size candidate dimension,
benchmark integration. Significant change; win is speculative until
measured.

### 3b. Horner table rematerialization -- IMPLEMENTED on x86

Reload the table registers from the constant pool inside the loop
instead of keeping them live. Saves `2*n_nontrivial` regs for nibble
(`n_out*U+7` vs `n_out*U+6+2*ntriv` persistent) and `n_nontrivial` regs
for GFNI (`n_out*U+3` vs `n_out*U+2+ntriv`). The planner enumerates the
remat twin only when the persistent variant does not fit but the remat
one does (dominance: same-U remat is the persistent body plus extra pool
loads, so it can never win when persistent fits).

Example: 4-output HORNER nibble, all nontrivial: untiled maxU=4
(4*4+6+8=30); remat allows U=6 (4*6+7=31). Cost: 8 L1 vector loads per
inner iteration (nibble); GFNI remat reloads are deduped across the
unroll loop at JIT time.

Status: x86 backend implements nibble and GFNI remat (codegen, planner
gate, pressure asserts, `tests/test_remat.c` force-builds 5 remat
cases against the scalar reference). AArch64 backend implements nibble
remat (no GFNI on this ISA): codegen in `src/aarch64/jit.c`
(prologue skips table preloads, u-loop reloads tables into the `tbl`
scratch, 2 pool loads per nontrivial use), planner twin enumeration in
`src/plan.c` under the same dominance policy, register-count assert
(`n_out*U+6`) in `compute_regs`, and `tests/test_aarch64_remat.c`
force-builds 4 remat cases (NEON headline twin, all-trivial, ntriv=1
boundary, SVE) plus planner twin-policy checks. RVV backend implements
nibble remat (nibble is the only multiply method on this ISA): codegen
in `src/riscv/jit.c` (prologue skips table preloads, u-loop reloads
tables into the `tbl` scratch via a new single-table `c_ldtab1` helper
-- the MULACC/DELTA pair loader's fixed `(tbl_lo, tbl_hi)` registers are
not assigned in the HORNER map -- 2 pool loads per nontrivial use, no
cross-u dedup since a second scratch would change the proven
`(n_out+1)*U+6` formula), planner twin enumeration in `src/plan.c` under
the same dominance policy, register-count assert (`(n_out+1)*U+6`) at the
allocation site, and `tests/test_riscv_remat.c` force-builds 3 remat
cases (RS(6,3) U=6 headline twin: persistent 33 > 32 >= remat 30;
all-trivial; custom n_out=4/ntriv=4 Vandermonde) plus planner
twin-policy checks plus an over-pressure rejection check (U=8 remat
needs 38 regs > 32, must fail cleanly), run under QEMU at VLEN
128/256/512. Note: with the
current RVV unroll cap (U <= 5) and n_out <= 3 from the standard
constructors, the twin is unreachable by the planner (persistent always
fits); it becomes live for wider custom topologies.

### 3c. MULACC as HORNER alternative

For Vandermonde encode/recovery, the planner emits only HORNER. But
MULACC computes the same result with **lower pressure for n_out > 1**
(shared table pair: `n_out*U+7` vs `n_out*U+6+2*n_nontrivial`).

Example: rs(6,3) encode: HORNER nibble maxU=7 (31 regs); MULACC nibble
allows U=8 (31 regs). MULACC also streams inputs once (vs n_out passes
for HORNER), improving input locality.

Cost: planner change to emit both plan kinds for Vandermonde matrices
(`plan_build` currently returns one plan); benchmark picks the winner.
Moderate change; the win hypothesis (input reuse + 1 extra U step) is
concrete and testable.

## 4. Recommendation

1. **Done:** U=7 (closes the demonstrated gap; purely additive).
2. **Done:** 3b rematerialization on x86 (nibble + GFNI), AArch64
   (nibble), and RVV (nibble).
3. **Next, if pursuing:** 3c (MULACC alternative). Smallest remaining
   change with a concrete, testable hypothesis. Prototype by forcing
   MULACC on the RS(6,3) Vandermonde matrix and benchmarking vs HORNER.
4. **Defer:** 3a tiling until (3) is measured. It adds backend
   complexity for speculative gains; the binding tables above identify
   exactly which (plan, n_out) would benefit, so it can be targeted
   precisely if (3) underwhelms.

## 5. Experiment status

- Non-pow2 unroll study (U=9..12): complete. Rigorous
  repeated/interleaved design (10,600 samples, paired Wilcoxon +
  Holm-Bonferroni): no U in {9,10,11,12} significantly beat U=8 in any
  tested configuration; significant differences were losses (e.g.
  RAID6(8) nibble 1 MiB: U=9 -12.2%, U=12 -20.4%). Candidate lists stay
  {8,7,6,5,4,3,2,1}.
- `tests/test_pressure.c`: pins exact `plan_need_regs` values and the
  U=6/U=7/U=8 admission boundaries on AVX-512, plus the coefficient-aware
  P_DELTA values (§§5-7: all-trivial RAID-5 delta on every ISA, nontrivial
  RAID-6 control, and the newly-admitted U=5 SSE candidate).
- Full gates (`make test` 464,279 runs, ASan/UBSan, QEMU matrices) pass
  on the current tree; rerun after every change per the quality
  contract.
