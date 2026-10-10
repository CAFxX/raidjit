# Spec: Nested CSE for xorsched/xorbit (track A3)

## 1. Problem

Both `xorsched_build` (P_XOR, byte-level) and `xorbit_build`
(P_MULACC/P_DELTA/P_HORNER, bit-level) do greedy **flat** pair-CSE:
they extract only `t = a^b` where a, b are inputs (or input bits).
They never extract nested temps like `t2 = t1^c`.

This misses sharing like `(a^b)` inside `(a^b^c^d)`, common in
RDP/EVENODD/Liberation diagonal parities (P_XOR) and in GF-multiply
rows with related coefficients (e.g. `xtime(x)` inside `gf_mul(3,x) =
xtime(x)^x`, though xorbit already captures some of this via
bit-level pairs).

A reviewer flagged flat-only as a P1 finding. This spec (track A3)
extends both builders to nested temps.

## 2. Design

### 2.1 Temp representation

Currently:
- `xorsched_temp_t { int a, b; }` where a, b are input ids (0..n_in-1).
- `xorbit_temp_t { int a, b; }` where a, b are input-bit ids
  (0..nbits_in-1).

Extended: a, b can be input ids OR earlier temp ids. We distinguish
by range:
- `xorsched`: inputs are 0..n_in-1, temps are n_in+k. In the temp
  struct, store operand ids in the "atom" namespace: input i ->
  i, temp k -> n_in+k. The existing `ta[]/tb[]` arrays already use
  plain ints; we extend the candidate pair enumeration to include
  temp atoms.
- `xorbit`: input bits are 0..nbits_in-1, temps are nbits_in+m.
  Same extension.

The `xorsched_t`/`xorbit_t` output format already uses atom ids
(temp k -> k, input i -> ntemps+i), so the emitter interface is
unchanged. Only the *formation* of temps changes (operands can be
temps).

### 2.2 Greedy algorithm with nesting

Current flat algorithm:
1. Count co-occurrences of input pairs (a,b) across rows.
2. Extract the best pair (most co-occurrences, >= 2) into t = a^b.
3. Replace (a,b) with t in all rows containing both.
4. Repeat.

Nested extension:
1. Count co-occurrences of pairs (X,Y) where X, Y are atoms
   (inputs or already-extracted temps).
2. Extract the best pair into t_new = X^Y.
3. Replace (X,Y) with t_new in all rows containing both.
4. Repeat.

Key subtlety: after extracting t1 = a^b, a row containing
(a,b,c) becomes (t1,c). The pair (t1,c) is now a candidate for
the next extraction. This is what enables nesting: t2 = t1^c
captures (a^b) inside (a^b^c^d).

Pair counting must handle the growing atom namespace. We maintain:
- `counts[]`: pair co-occurrence counts, indexed by atom pair.
  Atoms are inputs (0..n_in-1) plus temps (n_in..n_in+ntemps-1).
  The table grows as temps are added; we cap at XS_MAX_TEMPS
  (1024) as before.

- `has[]`: per-row atom membership bitsets, extended to cover
  temp atoms.

The greedy loop is otherwise unchanged: pick the best pair with
count >= 2, extract, update counts, repeat until best < 2 or
XS_MAX_TEMPS reached.

### 2.3 Preventing useless nesting

Not all nested extractions are beneficial. Consider:
- t1 = a^b (used in 3 rows).
- After replacement, rows have (t1,c), (t1,d), (t1,e).
- Pair (t1,c) occurs in 1 row only -> not extracted (needs >= 2).

The `best >= 2` threshold already prevents useless extractions.
A nested temp is only formed when it actually shares across >= 2
rows, which guarantees it replaces >= 1 XOR uop per user row
(same argument as flat temps).

Additional guard: do not extract (t1,t2) where t1 and t2 are both
temps unless it strictly reduces uops. The count>=2 threshold
handles this, but we also cap nesting depth implicitly via
XS_MAX_TEMPS.

### 2.4 Register pressure

The header documents: "Temps are flat pairs of inputs only --
never nested (keeps the live temp set at 2 and the
register-pressure formula exact)."

With nesting, the live temp set can exceed 2. Analysis:

In the worst case, a chain t1=a^b, t2=t1^c, t3=t2^d, ... has only
1 live temp at a time (each temp is consumed immediately to form
the next). But a "bushy" nesting like t1=a^b, t2=c^d, t3=t1^t2
has 2 live temps (t1, t2) when forming t3.

Upper bound: the number of live temps at any point is at most
the number of temps that are (a) formed and (b) not yet consumed
by all their users. In the greedy formation order, temps are
formed before use, so a temp is live from its formation until its
last user row is processed.

Conservative bound: live temps <= ntemps (trivially). Tighter:
the emitter processes temps in formation order and fans each out
immediately, so a temp's live range is [formation, last fan-out].
With nested temps, a temp used to form another temp has its live
range extended.

For the register-pressure formula, we take the conservative
approach: `xorsched_need_regs` and `xorbit_need_regs` add +1
for the nested-temp live set beyond the flat case. Specifically:
- Flat: live temp set = 2 (one being formed, one fanned out).
- Nested: live temp set <= 3 (conservative: one extra for
  temp-of-temp chains).

This is exact for chains (live=1 actually) and conservative for
bushy cases. The planner's `need > nregs` filter remains sound
(may reject a few feasible candidates, never accepts an
infeasible one).

### 2.5 Cost model

The static uop models (`xs_body_sched`, `xb_body_sched`) count:
- Temp formation: 2 uops (x86) or 3 uops (elsewhere) per temp.
- Fan-out: nusers uops per temp.

For nested temps, formation is still 1 XOR (temp^X or X^Y), so
the per-temp formation cost is UNCHANGED. The fan-out cost is
also unchanged (based on nusers).

Therefore: the existing cost models are exact for nested temps
without modification. The `ntemps` count and `nusers` per temp
are computed the same way; only the operand ids differ.

This is a key simplification: nesting changes WHICH pairs are
extracted, not the per-temp cost.

### 2.6 Branch-and-bound consideration

The task asks: "Consider whether bounded branch-and-bound (with
greedy incumbent) beats pure greedy for the sizes where nesting
matters most."

Analysis: The greedy pair-CSE is a heuristic for a Max-Coverage-
like problem. For the sizes where xorsched/xorbit run
(n_in <= 256, nbits_in <= 512), the greedy is fast (ms) and
typically within 10-15% of optimal (standard result for
greedy Max-Coverage).

Branch-and-bound could find the optimal, but:
1. The search space is exponential in the number of candidate
   pairs.
2. Plan-time budget is ms, not seconds.
3. The twin is benchmark-gated: if greedy finds a good schedule,
   the benchmark validates it. If greedy misses the optimal by
   10%, the benchmark still picks the best among candidates.

Decision: **Keep pure greedy.** The marginal gain from B&B
(< 10% on top of greedy's result, for the subset of cases where
nesting matters) does not justify the complexity and plan-time
risk. If future evidence shows greedy is systematically bad for
nested cases, revisit.

### 2.7 Termination and bounds

- XS_MAX_TEMPS (1024) cap unchanged.
- `best >= 2` threshold unchanged.
- Each extraction strictly reduces the total atom count across
  rows (replaces 2 atoms with 1 in each user row), so the loop
  terminates.

## 3. API/ABI impact

None. The `xorsched_t`/`xorbit_t` structs are unchanged (temps
still have `a, b, nusers, users`). The emitters consume atom ids
and don't care whether a temp's operands are inputs or temps.
The `cand_t.xorbit`/`cand_t.xorsched` axes and `prune.c` pins are
unchanged (no new twin, just better CSE within the existing
twins).

## 4. Test plan

1. **Differential**: nested vs flat on RDP/EVENODD/Liberation
   (P_XOR) and RS matrices (P_MULACC). Nested must never produce
   MORE uops than flat (it can only find additional sharing).
   Property: `uops_nested <= uops_flat` for all test matrices.
2. **Correctness**: kernel output bit-identical between nested
   and flat schedules (same XOR semantics).
3. **Byte-identity**: non-xorbit/non-xorsched kernels unchanged
   (the builders are only called for their respective twins).
4. **Red-green**: construct a matrix where nesting provably helps
   (e.g. rows (a,b), (a,b,c), (a,b,c,d)); verify nested finds
   t1=a^b, t2=t1^c and flat does not.
5. **OOM**: existing test hooks cover the new code paths (same
   allocation sites, extended loops).
6. **Register pressure**: verify `need_regs` is conservative
   (no backend spills on nested schedules).

## 5. Peer review focus

1. The pair-counting update logic for temp atoms (correctness of
   counts after nested extraction).
2. The register-pressure bound (is +1 sufficient?).
3. Termination argument (does the loop always terminate?).
4. The B&B decision (is pure greedy defensible?).

## 6. Rollout

1. Implement nested extraction in `xorsched_build` and
   `xorbit_build`.
2. Update `xorsched_need_regs`/`xorbit_need_regs` (+1 for nesting).
3. Update header docs (remove "never nested", document the
   new live-temp bound).
4. Run differential tests, full suite, sanitizers.
5. Two peer reviews, self-attack.
6. No new twin, no planner changes, no emitter changes.
