# Spec: xorbit twin for AArch64 and RISC-V

Status: spec (2026-10-10). Extends the Phase C item 1 bit-level XOR scheduling
(xorbit) twin from x86-only to AArch64 (NEON/SVE) and RISC-V (RVV).

## 1. Background

The xorbit twin drives the vector body from `pl->bit_sched` (an `xorbit_t`
straight-line bit program built by `xorbit_build`) instead of the naive
nibble-table MULACC emitter. The x86 implementation (`gen_body_xorbit` in
`src/x86_64/jit.c:915`) is shipped and verified. This spec extends it to
AArch64 and RISC-V.

Backend contract (from `src/xorsched.h`, ISA-agnostic):
- Materialize each input bit as a bit-0-normalized byte vector `((in_i >> k) & 1)`.
- Evaluate each temp with vector XORs: `t_m = in[a_m] ^ in[b_m]`.
- Assemble output bytes: `out_j = XOR_b (P(j,b) << b)`.
- Accumulators start zero (P_MULACC/P_HORNER) or seeded from oldpar (P_DELTA).
- All CSE/scheduling is plan-time; the emitter walks the program linearly.

## 2. AArch64 emitter design

### 2.1 Required operations

| Operation | NEON | SVE | uops |
|-----------|------|-----|------|
| `d = s >> k` (k=1..7) | `USHR d.16b, s.16b, #k` | `LSR d.b, s.b, #k` (unpred) | 1 |
| `d = s & 1` | `AND d.16b, s.16b, mask1` | `AND d.b, s.b, mask1` (unpred) | 1 |
| `d = s << b` (b=1..7) | `SHL d.16b, s.16b, #b` | `LSL d.b, s.b, #b` (unpred) | 1 |
| `d = s` (move) | `ORR d.16b, s.16b, s.16b` | `MOV d.b, s.b` (unpred) | 1 |
| `d ^= s` | `EOR d.16b, d.16b, s.16b` | `EOR d.b, d.b, s.b` (unpred) | 1 |
| `d = a ^ b` | (via mov + eor) | (via mov + eor) | 2 |

### 2.2 New emitters needed

- `em_neon_shl(e, vd, vn, sh)`: `SHL vd.16b, vn.16b, #sh`, sh in 1..7.
  Encoding: `0x6F000400 | (((sh) & 63) << 16) | (vn << 5) | vd`
  (SHL uses immh:immb = sh directly, unlike USHR which uses 16-sh).
- `em_sve_lsl_unpred(e, zd, zn, sh)`: `LSL zd.b, zn.b, #sh`, unpredicated,
  sh in 1..7.
  Encoding: SVE LSL (unpredicated): `0x04209C00 | (((sh) & 15) << 16) | (zn << 5) | zd`
  (verify against ARM ARM; LSL encoding differs from LSR).

Both must be GAS cross-checked across randomized operands per the AGENTS.md
emitter lesson (clear ALL operand fields when deriving bases).

### 2.3 Mask materialization

The `& 1` mask needs a vector of 0x01 bytes. Options:
- (a) Dedicated mask register loaded from literal pool (like x86's CK_ONE).
- (b) `DUP` from a GPR holding 1.

Option (a) is simpler and matches x86. The AArch64 backend already has
`cg->mask` (0x0F for nibble-split); xorbit needs a separate 0x01 mask.
Add `cg->xbt_mask1` or reuse the literal pool.

For k=0 (no shift): skip the shift, just AND.
For b=0 (no shift): skip the shift, just MOV/XOR.

### 2.4 SVE vs NEON

- NEON (128-bit fixed): straightforward, mirrors x86 SSE path.
- SVE (variable length): use unpredicated forms (like `c_and3`/`c_lsr4` do).
  The xorbit body is lane-independent (each byte processed identically),
  so unpredicated is sound. `cg->vl` gives the vector length in bytes.

### 2.5 Register allocation

Mirroring x86: `n_out * U` accumulators + 4 temps (inreg, r1, r2, mask1).
`xorbit_need_regs` already returns `pl->n_out * U + 4` (ISA-agnostic).
Add `xbt_in, xbt_r1, xbt_r2, xbt_mask1` fields to AArch64 `cg_t`.

## 3. RISC-V emitter design

### 3.1 Required operations

| Operation | RVV | uops |
|-----------|-----|------|
| `d = s >> k` (k=1..7) | `vsrl.vi d, s, k` | 1 |
| `d = s & 1` | `vand.vi d, s, 1` | 1 |
| `d = s << b` (b=1..7) | `vsll.vi d, s, b` | 1 |
| `d = s` (move) | `vmv.v.v d, s` | 1 |
| `d = a ^ b` | `vxor.vv d, a, b` | 1 |

### 3.2 New emitters needed

- `em_vsll_vi(e, vd, vs2, uimm)`: `vsll.vi vd, vs2, uimm`, uimm < 32.
  Encoding: OPIVI class, funct6=100111 (vsll.vi).
  `em_opv(e, 0x27 /*100111*/, vs2, uimm, 0x3 /*OPIVI*/, vd)`.
  (Verify funct6 against RISC-V V spec: vsll.vi = 100111.)

Existing emitters cover the rest:
- `em_vsrl_vi` (shift right) ✓
- `em_vand_vi` (AND immediate, for `& 1`) ✓
- `em_vxor_vv` (XOR) ✓
- `em_vmv_v_v` (move) ✓

### 3.3 Mask materialization

`vand.vi d, s, 1` uses the immediate directly — no mask register needed!
This is simpler than x86/AArch64. For k=0: skip shift, just `vand.vi`.

### 3.4 Register allocation

Same as AArch64: `n_out * U + 4`. Add `xbt_*` fields to RISC-V `cg_t`.

### 3.5 LMUL considerations

RVV uses LMUL to group registers. The xorbit body uses the same `cg->vl`
mechanism as other bodies. The 4 temp registers must respect LMUL grouping
(follow the existing `r1`/`r2`/`mask` allocation pattern in `src/riscv/jit.c`).

## 4. Cost model

Extend `xorbit_version_uops` for AArch64/RISC-V. The x86 model has two
variants (SSE vs AVX: `sh = 2` vs `1` for `c_sllw` when `vd != s1`).

For AArch64/RISC-V:
- All ops are 1 uop (no SSE-style mov+shift penalty; both have 3-operand
  or non-destructive forms... actually NEON EOR is destructive (d=d^s),
  but the xorbit_fanout pattern uses `c_vmov` + `c_vpxor` which is 2 uops
  for the "shift then fanout" case. Model exactly like x86 AVX (sh=1).
- Key difference: RISC-V `vand.vi` for `& 1` is 1 uop (no separate mask
  load). x86/AArch64 need the mask register (loaded once in prologue,
  uncounted).
- P_DELTA seeding: same as x86 (n_out * U loads).

New function or extended: `xorbit_version_uops(s, pl, isa, U, sched, nt)`
- Returns -1 for unsupported ISAs (keep the gate).
- For ISA_NEON, ISA_SVE, ISA_RVV: compute using the ARM/RV emitter's
  counting convention.
- The `is_sse` parameter becomes `isa_class` (x86-SSE vs others).

Lockstep test: the existing lockstep test (emitter vs model) must be
extended to AArch64/RISC-V. This is the "tests never tautological"
requirement — the model and emitter must agree.

## 5. Enumeration

In `src/plan.c`, extend the xorbit enumeration gate:
- Current: `if (isa != ISA_SSE && isa != ISA_AVX2 && isa != ISA_AVX512) continue;`
- New: also allow `ISA_NEON`, `ISA_SVE`, `ISA_RVV` (check exact enum names).
- Keep the other gates: kind in {P_MULACC, P_DELTA, P_HORNER}, `bit_sched`
  non-NULL, `mul == MUL_NIBBLE`, register check, strict uop win
  (`u_sched < u_naive`).
- The `xorbit_version_uops` returning -1 for unsupported ISAs is the
  backstop; the enumeration gate is the primary filter.

## 6. Prune pin

`src/prune.c:228` already pins `xorbit` (ISA-agnostic comparison).
Verify it covers the new ISAs (it should — it compares `cand.xorbit`
values directly, no ISA check). No change needed, but verify with a test.

## 7. Backend gates

Each backend's `jit_build_version` must:
- Accept `cand->xorbit == 1` for its ISA (currently may assert or reject).
- Validate: `pl->bit_sched != NULL`, kind in {P_MULACC, P_DELTA, P_HORNER},
  `mul == MUL_NIBBLE`.
- Reject cleanly (error return, not assert) for invalid combos — per the
  crashfix lesson (user-reachable invalid combos must not abort).

Check AArch64/RISC-V `jit_build_version` for existing xorbit handling.

## 8. Test plan

1. **Differential**: xorbit kernel output vs reference (`ref.c` or naive
   MULACC) for random topologies, lengths, alignments — per ISA.
   Run under QEMU for AArch64 and RISC-V.
2. **Lockstep**: emitter uop count == model uop count (extend existing
   x86 lockstep test to ARM/RV).
3. **Byte-identity**: non-xorbit kernels unchanged (sha256 over kernels
   doc sections, or targeted disassembly diff).
4. **Combination**: xorbit × {nt, prefetch, U values, LMUL (RVV)} —
   builds or rejects cleanly.
5. **Negative**: invalid xorbit combos (wrong kind, NULL bit_sched)
   return errors, don't abort (fork-isolated, like `test_pinned_negative`).
6. **QEMU**: AArch64 and RISC-V full test suites for xorbit builds.

## 9. Peer review

2+ independent reviewers + self-attack, per the world-class contract.
Reviewers must check: emitter correctness (bit-exactness), cost model
lockstep, ISA gating, register allocation soundness.

## 10. Open questions

- SVE predicate for the xorbit body: unpredicated throughout (like
  `c_and3`/`c_lsr4`), or does the tail need predication? (The xorbit
  body is called for the main loop; tails use the existing tail logic.)
- RISC-V `vsll.vi` funct6 verification against the spec.
- AArch64 SVE LSL unpredicated encoding verification.
- Should `xorbit_version_uops` be split per-ISA or parameterized?
  (Recommend parameterized with `isa_class` to avoid duplication.)
