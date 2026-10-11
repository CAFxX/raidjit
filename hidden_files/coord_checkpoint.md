<<<<<<< HEAD
# LRC locality-aware recovery — coordinator checkpoint

## Status (2026-10-10, load 14+ blocking builds)

**Spec**: `hidden_files/lrc_locality_SPEC.md` — complete.
**Implementation**: `src/survivor.c` + `src/survivor.h` — complete, syntax-clean, committed (e93d28f, 5dcf9eb).

## What was built

**Problem**: LRC local recovery worked via emergent cost-model dominance
(local XOR subset strictly dominates global MULACC on both static axes),
not by design. Fragile to benchmark noise, significance-gate fallback,
and cost-model drift.

**Solution**: Explicit planner rule in `survivor_select`.

1. `survivor_lrc_detect(ti, group_of, local_row, ngroups_out)`:
   Identifies LRC-shaped topologies by matrix structure (pure-XOR
   disjoint rows partitioning data disks). Structure-based, not
   name-based — works for any present/future LRC-like matrix.

2. `survivor_lrc_local_cover(ti, F, nf, ...)`: Computes pure-local cover
   for failure pattern F (distinct surviving local row per failure).
   Handles: data failure → group's local row (must survive); failed
   local parity → itself (forced recompute, no survival check); failed
   global parity → not local (generic path); two failures one group →
   not purely local (generic path).

3. `survivor_select` integration: after `survivor_enumerate`, if
   pure-local cover exists, restrict `keep[]` to local subsets
   (bypass pareto shortlist). Still runs L2 signal to verify
   buildability; falls back to generic shortlist if local fails to
   build or restriction keeps nothing.

**No new cand_t axis** (subset selection, not codegen) → no prune.c change.
**Byte-identical for non-LRC** (detect returns 0 → generic path, zero behavior change).
**Bit-exact for LRC** (same subsets the cost model already picked; verified by existing tests).

## Remaining work (blocked on load)

1. Build + run `test_lrc` (verify no regression, same check counts).
2. Build + run full `make test` (or survivor-related subset).
3. Add unit tests for `survivor_lrc_detect` / `survivor_lrc_local_cover`
   (edge cases: non-LRC matrices, overlapping rows, incomplete partition,
   mixed failure patterns).
4. ASan/UBSan + QEMU (aarch64, riscv64) reruns.
5. 2 independent peer reviews + self-attack.
6. Integration via `wt.py integrate` after reviews.

## Commits

- e93d28f: spec + implementation
- 5dcf9eb: fix failed-local-parity survival check
=======
# xorbit AArch64/RISC-V Implementation Checkpoint

Status: body implementations complete (2026-10-10). Builds/tests deferred (load backoff).

## Completed

### AArch64 (`src/aarch64/jit.c`)
- [x] Helpers: `c_xbt_srl`, `c_xbt_and1`, `c_xbt_sll` (from prior commit)
- [x] `c_xbt_zero` (vector zeroing)
- [x] `xorbit_fanout` (mirrors x86 exactly, using c_copy/c_xor/c_xbt_sll)
- [x] `gen_body_xorbit` (full: dead-byte zeroing, DELTA seeding, phase 1 temps, phase 2 input bits)
- [x] Register allocation in `compute_regs` (xbt_in/r1/r2/mask)
- [x] 0x01 mask setup in `gen_const_setup`
- [x] Dispatch in `gen_body` (xorbit check first, before kind dispatch)

### RISC-V (`src/riscv/jit.c`)
- [x] Helpers: `c_xbt_srl` (vsrl.vi), `c_xbt_and1` (vand.vi imm), `c_xbt_sll` (vsll.vi)
- [x] `xorbit_fanout` (LMUL-aware: acc indices scaled by lmul)
- [x] `gen_body_xorbit` (full, with LMUL scaling, T2/T1/T0 pointer management)
- [x] Register allocation in `jit_build_version` (xbt_in/r1/r2, no mask reg needed)
- [x] Dispatch in `gen_body` (xorbit check first)

### Cost model (`src/xorsched.c`)
- [x] `xorbit_version_uops` accepts ISA_NEON/SVE/RVV (was x86-only)
- [x] `xb_body_sched` and `xb_fanout_uops` take `isa_t` (was `is_sse` bool)
- [x] Bit-extraction uops: ISA-aware (SSE: 3, others: 2)
- [x] Shift cost: ISA-aware (SSE: 2, others: 1)

### Enumeration (`src/plan.c`)
- [x] Gate extended to NEON/SVE/RVV (was x86-only)
- [x] Comment updated (removed "x86-64-only for now")

## Deferred (load backoff)

1. **Builds**: `make` in worktree to check for compile errors
2. **QEMU tests**: AArch64 and RISC-V differential tests
3. **GAS verification**: `em_sve_lsl_unpred` base (0x0420B400) derived, not verified
4. **Lockstep test**: extend `tests/test_xorbit.c` for new ISAs
5. **Peer reviews**: 2+ independent + self-attack
6. **Integration**: via `wt.py integrate` after reviews

## Key design decisions

1. **Lockstep**: Bodies mirror x86 emission order exactly. The k=0 case does
   copy+and (2 uops) to match the model's 2-uop count for non-SSE.
2. **RISC-V LMUL**: All vector register indices scaled by `cg->lmul`.
   The xbt registers are logical (xbt_in, etc.); physical is `* lmul`.
3. **No mask reg for RVV**: `vand.vi` takes immediate, so no xbt_mask needed.
4. **AArch64 unpredicated**: The xorbit body is lane-independent, so
   unpredicated forms are sound (same argument as c_and3/c_lsr4).

## Files changed
- `src/aarch64/jit.c`: +~250 lines (helpers, fanout, body, regs, dispatch)
- `src/riscv/jit.c`: +~250 lines (helpers, fanout, body, regs, dispatch)
- `src/xorsched.c`: cost model ISA extensions
- `src/plan.c`: enumeration gate extension
>>>>>>> fc87cba (xorbit-arm-rv: AArch64/RISC-V backends, cost model, verification fixes)
