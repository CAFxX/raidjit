# Spec: Single-Erasure Closed-Form Recovery Emitter (algos #4, phase 2)

## 1. Background

Track A4 phase 1 (planner-only) is DONE: `plan_recover_closed_form` fills the
recovery matrix R directly from e=1/e=2 closed forms, bit-identical to
`solve_recovery_rows`. The plans flow through the normal pipeline and
`plan_select_strategy` picks P_XOR/P_HORNER/P_MULACC.

This spec (phase 2) adds an **emitter-side** optimization: a specialized
codegen twin for closed-form recovery plans that fuses the syndrome
computation with the closed-form combine, instead of emitting the baked
R matrix as a generic MULACC.

## 2. Mathematical Analysis (honest win regions)

### e=1: `x_f = c_f^-1 * sum_s (E[r][s] * x_s)`

The generic path emits `x_f = sum_s R[s]*x_s` with `R[s] = E[r][s]*c_f^-1`.

**Win analysis:**
- If E[r][s] are all 1 (P row): R is all 1s → `plan_select_strategy` picks
  P_XOR. The generic emitter already does pure XOR. **No emitter win.**
- If E[r][s] are general (Q/RS row): generic does ns nibble-multiplies by
  `E[r][s]*c_f^-1`; specialized does ns nibble-multiplies by `E[r][s]` plus
  one by `c_f^-1`. **Same multiply count. No uop win.**

**Conclusion for e=1:** The A4 planner-side closed form + `plan_select_strategy`
is already optimal for uops. The phase-2 emitter has **no uop win** for e=1.

**Remaining e=1 value:**
- **Code size**: The specialized emitter can generate a smaller loop body
  (single syndrome reduction + one multiply vs per-coefficient multiply
  setup). This addresses the code-size heuristic weakness.
- **Strength reduction**: If `c_f^-1` is a "cheap" constant (power of 2 →
  xtime instead of nibble-split), the final multiply is cheaper than a
  generic nibble multiply. Win only when `c_f^-1` is cheap AND the
  generic path wouldn't also benefit (it wouldn't — generic folds
  `c_f^-1` into each coefficient).

**Decision:** Implement e=1 specialized emitter as a **code-size/strength-
reduction twin**, benchmark-gated. Expected win: small (5-15%) and only
when `c_f^-1` admits strength reduction. Document as such; do not claim
40-60%.

### e=2: shared syndromes

```
p = sum_s E[r1][s]*x_s    (syndrome from row r1)
q = sum_s E[r2][s]*x_s    (syndrome from row r2)
x1 = (d*p ^ b*q) * det^-1
x2 = (c*p ^ a*q) * det^-1
```

**Win analysis (RAID-6 P+Q case):**
- Generic: 2 rows × ns nibble-multiplies = 2ns multiplies.
- Specialized: p is pure XOR (P row, 0 multiplies) + q is ns multiplies
  + 6 multiplies for the 2x2 combine = ns+6.
- For ns=12: 18 vs 24 → **25% fewer multiplies**. ✓

**Win analysis (general RS, both rows non-trivial):**
- Generic: 2ns multiplies.
- Specialized: 2ns (syndromes) + 6 (combine) = 2ns+6. **Worse.**

**Conclusion for e=2:** Win only when at least one equation row has
simpler coefficients than the baked R (specifically, the P-row case).
The twin must check: `row r1 is all-0/1 OR row r2 is all-0/1`.

## 3. Design

### New cand_t axis: `rec_cf`

- `rec_cf = 1`: use closed-form emitter (syndrome + combine).
- `rec_cf = 0`: generic emitter (default).
- Enumerated ONLY when:
  - Plan is recovery (`pl->is_recover` or equivalent).
  - `nf` is 1 or 2.
  - `plan_recover_closed_form` would return 1 (closed form applies).
  - For e=1: `c_f^-1` admits strength reduction (else no win; skip).
  - For e=2: at least one equation row is all-0/1 (else no win; skip).
- Benchmark-gated per ISA: the twin is enumerated, benchmarks pick the
  winner. If the closed-form emitter doesn't win, generic is selected.
  **No regression possible.**

### Pin in prune.c::twin_dominates

Add `rec_cf` to the field-by-field axis comparison in the SAME commit
as the axis (AGENTS.md 2026-10-03 lesson).

### Emitter interface

New backend functions (per ISA):
- `gen_recover_cf_e1(cg, pl, cf_data)`: emit `x_f = c_inv * syndrome`.
  `cf_data` carries: equation row coefficients `E[r][s]`, `c_f^-1`,
  survivor list.
- `gen_recover_cf_e2(cg, pl, cf_data)`: emit syndromes p, q then 2x2 combine.

The `cf_data` is built at plan time (from the same inputs as
`plan_recover_closed_form`) and stored in `plan_t` (new fields).

### Static cost model

The cost model must accurately reflect:
- e=1: `(ns-1)` XORs + cost of syndrome multiplies (0 if P-row, else
  nibble) + cost of final `c_f^-1` multiply (strength-reduced if applicable).
- e=2: syndrome costs + 6 multiplies + 2 XORs for combine.

This addresses the code-size heuristic weakness: the model counts
actual emitted uops, not estimates.

## 4. Correctness

- **Differential**: closed-form emitter output vs generic emitter output,
  bit-identical for all (topology, failed set, row choice) where the
  twin applies. Test via `RJ_RECOVER_NO_CLOSEDFORM` to force generic.
- **Byte-identity**: non-recovery plans and e>=3 recovery plans must be
  byte-identical (twin not enumerated).
- **Red-green**: flip a coefficient in cf_data → test fails.

## 5. Test matrix

- All single-failure patterns for RAID-5, RAID-6, RS(6,3), RS(10,4).
- All double-failure patterns for RAID-6, RS(6,3).
- e>=3 patterns (twin not enumerated, generic path).
- Singular pivots (fallback to generic).
- ASan/UBSan, QEMU (aarch64, riscv64).

## 6. Reviews

- 2+ independent peer reviews.
- Self-attack pass.
- No benchmarks, no perf guard, no push.
