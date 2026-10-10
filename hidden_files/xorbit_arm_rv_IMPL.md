# xorbit AArch64/RISC-V Implementation Checkpoint

Status: in progress (2026-10-10). Continuing from spec + emitters (commit a1ec9ce).

## Completed
- [x] Spec read and understood
- [x] x86 gen_body_xorbit structure analyzed
- [x] AArch64/RISC-V helper inventory done

## In Progress
- [ ] AArch64 gen_body_xorbit + helpers
- [ ] RISC-V gen_body_xorbit + helpers
- [ ] Cost model extension (xorbit_version_uops)
- [ ] plan.c enumeration gate
- [ ] Backend jit_build_version gates

## Design Notes

### AArch64 helpers needed (mirroring x86 c_*):
- `c_xbt_srl(cg, vd, vs, k)`: d = s >> k, k in 1..7. NEON: USHR; SVE: LSR unpred.
  For k=0: skip (caller handles).
- `c_xbt_and1(cg, vd, vs)`: d = vs & 0x01 using xbt_mask reg.
- `c_xbt_sll(cg, vd, vs, b)`: d = vs << b, b in 1..7. NEON: em_neon_shl; SVE: em_sve_lsl_unpred.
  For b=0: use c_copy.
- Loads: `c_ld_to(cg, r, u, ptr)` (already exists, takes u offset).
- Input ptr: `c_inptr(cg, dst, idx)` (already exists).
- XOR: `c_xor(cg, d, s)` (d ^= s, already exists).
- Copy: `c_copy(cg, d, s)` (already exists).
- Zero: inline em_sve_dup / em_neon_movi (like gen_body does).

### RISC-V helpers needed:
- `c_xbt_srl(cg, vd, vs, k)`: `vsrl.vi vd, vs, k`.
- `c_xbt_and1(cg, vd, vs)`: `vand.vi vd, vs, 1` (no mask reg needed!).
- `c_xbt_sll(cg, vd, vs, b)`: `vsll.vi vd, vs, b` (new emitter).
- Loads: need to check RVV load helper (c_vle?).
- XOR: `c_vxor(cg, vd, vs1, vs2)` (3-operand, already exists).
- Move: `c_vmv(cg, vd, vs)` (already exists).

### Cost model:
- `xorbit_version_uops`: add ISA_NEON, ISA_SVE, ISA_RVV to the ISA gate.
- Replace `is_sse` with ISA-aware shift cost. For ARM/RV: sh=1 (like AVX, not SSE).
  - NEON: EOR is destructive (2-operand), but fanout uses copy+xor = 2 uops for b=0 case... 
    Actually x86 AVX: c_vmov (1) + c_vpxor (1) = 2 uops for the b=0 multi-user case.
    The `sh` param only affects the shift cost (1 vs 2). ARM/RV shifts are 1 uop (3-operand or non-destructive).
  - So: `sh = (isa == ISA_SSE) ? 2 : 1`.
- Bit extraction: x86 counts `c_srlw (2) + c_and (1)` = 3. For SSE, srlw is 2 uops (mov+shift).
  For ARM/RV: shift (1) + and (1) = 2 uops. Need to parameterize this too.
  - Actually looking at xb_body_sched: `uops += 3; /* c_srlw (2) + c_and (1) */`
    For non-SSE: c_srlw is 1 uop, so it should be 2. But the code adds 3 unconditionally!
    Let me re-read... The comment says "c_srlw (2) + c_and (1)" = 3. This is the SSE cost.
    For AVX, c_srlw is 1 uop, so it should be 2. This looks like a bug, or the model is conservative.
    I need to check the lockstep test to see what's expected.

Let me verify against the lockstep test before changing the model.
