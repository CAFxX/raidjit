# Spec: xorbit/xorsched Emitter Redesign — Temp-Stack with Liveness

Status: spec (2026-10-10). Prerequisite for nested CSE (per 2026-10-06
verdicts). User directive: NEVER regress existing flat-temp performance.

## 1. Problem

Current emitters (`gen_body_xorbit` in `src/x86_64/jit.c:915`,
`gen_body_xorsched`, and AArch64/RISC-V counterparts) assume flat temps:

- xorbit: `t_m = in[a] ^ in[b]` where a,b are INPUT BIT ids.
- xorsched: `t_k = in_a ^ in_b` where a,b are INPUT indices.

The emitter forms each temp in a scratch register (`r1`), fans it out
to users immediately, then discards it. There is NO temp storage, NO
liveness tracking. A nested temp (`t2 = t1 ^ c`, where `t1` is a temp
id) cannot be formed because `t1`'s value is gone.

The 2026-10-06 verdicts (docs/xor_schedule.md §§6, 10) deliberately kept
flat-only because: (a) nested wins are 0-2.3% at affordable pressure
on real shapes; (b) uncapped nested needs up to 156 live temps
(unproducible). The flat-only operand bound is pinned by
`check_flat_bound` in `tests/test_xorbit.c`.

This redesign does NOT change that verdict. It builds the EMITTER
CAPABILITY for nested temps, with a hard no-regression guarantee for
flat programs, and a per-case cost model that only selects nested when
provably better.

## 2. Goals / Non-goals

- GOAL: Emitters handle arbitrarily nested temps via temp-stack +
  liveness, on all three backends.
- GOAL: Bit-identical machine code for ALL existing flat-temp
  programs (no-regression proof, not just aspiration).
- GOAL: Per-case flat-vs-nested cost model; nested emitted only when
  strictly better by static uop count.
- NON-GOAL: Changing the 2026-10-06 verdict. If the cost model rarely
  selects nested, that's the correct outcome.
- NON-GOAL: Benchmark-gated selection (benchmarks postponed). Static
  model only.

## 3. Temp-Stack Design

### 3.1 IR extension

Temp operand ids are extended: an operand is either an input (bit or
byte) or a temp id. The scheduler (future work, not this track) may
emit nested temps; this track only changes the EMITTER to consume
them.

Atom id encoding (xorbit, unchanged for flat):
- temp m -> m (0 <= m < ntemps)
- input bit q -> ntemps + q

For nested temps, `temps[m].a` / `temps[m].b` may be temp ids (< ntemps)
instead of input bit ids (>= ntemps). The emitter must distinguish.

### 3.2 Liveness analysis (plan-time, in xorsched.c)

For each temp m, compute:
- `def[m]`: program point where t_m is formed (its index in formation order).
- `lastuse[m]`: last program point where t_m is used (as operand of a
  later temp, or in a row's atom list).
- Live interval: `[def[m], lastuse[m]]`.

For FLAT temps: `lastuse[m] == def[m]` (used immediately in fanout,
never as operand of another temp). Live interval is a single point.
This is the KEY FACT for no-regression: flat temps need NO storage.

For NESTED temps: `lastuse[m] > def[m]`. The temp must be stored
somewhere (register or stack slot) between def and lastuse.

### 3.3 Temp-stack allocation

Maintain a stack of live temp values. When forming temp m:
1. If any temp dies at m (lastuse == m-1), pop it (free its storage).
2. Allocate storage for t_m (register if available, else stack slot).
3. Emit the XOR into that storage.
4. Record `loc[m] = storage location`.

When temp m is used as an operand (of a later temp or in fanout):
- Load from `loc[m]` (if not already in the desired register).

Storage classes (x86-64):
- Registers: use the existing `xbt_r1`, `xbt_r2` scratch regs plus
  additional regs from the pool, up to the pressure limit.
- Stack: spill to stack slots (emitter already has stack frame).

### 3.4 No-regression argument (flat temps)

For a flat program:
- Every temp m has `lastuse[m] == def[m]`.
- The temp-stack is EMPTY at each formation point (previous temp died).
- Allocation always returns the same scratch register (`r1`).
- The emitter emits: load a, load b, xor into r1, fanout from r1.
- This is EXACTLY the current emission order.

Therefore: the redesigned emitter, on flat input, MUST produce
byte-identical machine code. This is enforced by:
1. `test_xorbit` existing tests (must pass unchanged).
2. New `test_emitter_identity`: disassemble old vs new emitter on all
   flat-temp corpus matrices, diff must be empty.
3. The `check_flat_bound` test continues to pin the flat-only IR bound
   (the SCHEDULER still emits flat-only by default; nested is opt-in
   via the cost model).

If byte-identity fails for ANY flat program, the redesign is wrong.
Stop and report; do not proceed.

## 4. Per-Case Cost Model (Flat vs Nested)

Since "nested never worse than flat" is FALSE (peer review
counterexamples), the planner must compare per program.

Cost model (static uop count, same convention as
`xorbit_version_uops`):
- `cost_flat`: uops for the flat schedule (current model).
- `cost_nested`: uops for the nested schedule, INCLUDING:
  - Temp formation uops (same as flat per temp).
  - Temp storage uops (register moves / stack spills for live temps).
  - Increased register pressure (may reduce feasible U; model must
    account for U drop).

Selection rule: emit nested IFF `cost_nested < cost_flat` (strict).
Otherwise emit flat. This guarantees no-regression by construction:
the planner never chooses a worse static cost.

The cost model lives in `xorsched.c` (not the emitter). The emitter
consumes whichever schedule the planner selects.

## 5. Backend Implementation Order

1. **x86-64** (reference): `gen_body_xorbit`, `gen_body_xorsched`.
2. **AArch64**: NEON and SVE paths.
3. **RISC-V**: RVV path.

Each backend must pass:
- `test_xorbit` / `test_xorsched` unchanged (flat programs).
- `test_emitter_identity` (byte-identity old vs new on flat).
- New nested-temp differential tests (vs reference interpreter).

## 6. Register Pressure

Current: `xorbit_need_regs = n_out*U + 4` (exact for flat).
New: `n_out*U + 4 + maxlive` where `maxlive` = max simultaneously-live
temps (0 for flat programs).

The `plan_candidates` gate must use the new formula. For flat
programs, `maxlive == 0`, so the formula reduces to the old one —
no change in candidate enumeration for existing cases.

## 7. Test Plan

1. **No-regression**:
   - `test_xorbit` (existing): must pass unchanged.
   - `test_xorsched` (existing): must pass unchanged.
   - `test_emitter_identity` (new): byte-identity old vs new emitter
     on flat-temp corpus. FAILS if any diff.
   - `check_flat_bound` (existing): still pins flat-only scheduler
     default.
2. **New capability**:
   - Nested-temp differential: handcrafted nested programs vs scalar
     reference interpreter. Must match bit-exact.
   - Liveness unit tests: `def`/`lastuse` computation on known DAGs.
   - Cost model: verify `cost_nested < cost_flat` iff nested actually
     wins (on the 2026-10-06 corpus).
3. **Combination**: nested × {all twins, ISAs} builds or rejects
   cleanly.
4. **Mutation**: flip a temp operand → tests fail.

## 8. Peer Review Gates

- 2+ independent reviewers (one must verify the no-regression proof).
- Self-attack pass.
- The 2026-10-06 verdicts must be addressed: explain why this
  redesign does not contradict them (answer: it doesn't change the
  verdict; it builds the capability with a cost model that will
  rarely select nested, which is the correct outcome).

## 9. Open Questions

1. Should the temp-stack use a fixed set of registers or allocate
   dynamically? (Lean: fixed set for simplicity, spill to stack.)
2. For x86-64, can we use k-mask registers for temp liveness
   tracking? (Probably overkill; simple array suffices at plan time.)
3. Does the cost model need to account for I-cache effects of larger
   code? (Out of scope; static uop count only.)
