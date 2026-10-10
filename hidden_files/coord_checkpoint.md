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
