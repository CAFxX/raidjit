# RVV Zvbc/Zvkg carryless multiply (B3) and Zicond (B6)

Implemented 2026-10-03. This document records the instruction
selection, the gating design, the exact GF(2^8) formulation, the
evidence that the emitted code is correct, and the Zicond rejection.

**No perf claim is made for RVV in this wave** — there is no RISC-V
silicon here to measure on (QEMU 8.2.2 only). The planner's
compile-time benchmarking will decide on real hardware.

## B3 — Zvbc/Zvbc32e carryless multiply: SHIPPED

### Instruction selection

| Instruction | Extension | Semantics (SEW=8) |
|---|---|---|
| `vclmul.vv vd, vs1, vs2` | Zvbc (ratified); SEW=8 form needs Zvbc32e | `vd[i]` = low 8 bits of carryless `vs1[i]`×`vs2[i]` |
| `vclmulh.vv vd, vs1, vs2` | Zvbc (ratified); SEW=8 form needs Zvbc32e | `vd[i]` = bits 15..8 of carryless `vs1[i]`×`vs2[i]` |

Both instructions are Zvbc (Vector Crypto §36.2/36.3), which defines
them for SEW=64 only. The SEW=8 forms the backend emits require the
separate Zvbc32e extension (fast-track vector-crypto-extra §33),
which is independent of Zvbc -- Zvbc does NOT imply Zvbc32e, and
Zvbc32e is legal standalone (ELEN >= 32). (An earlier version of this
document misattributed `vclmulh.vv` to Zvkg; Zvkg actually defines
`vghsh`/`vgmul`, not `vclmulh`. The old `rvv_zvkg` gate was built on
that misattribution and has been dropped -- it bought no safety.)
The twin is therefore gated on **`rvv_zvbc32e` alone**. (No twin is
enumerated on Zvbc-without-Zvbc32e cores: the SEW=8 encoding is
reserved there, and a SEW=64 carryless formulation would process 1/8
the lanes with a reduction that does not transfer. Such cores keep
the nibble path.)

### GF(2^8) formulation

For input byte `a`, coefficient byte `b`, with `K = 0x1B`:

```
lo = vclmul(a, b) & 0xFF          # low 8 bits of the 16-bit product
hi = vclmulh(a, b)               # high 8 bits
q  = vclmul(hi, K) & 0xFF
qh = vclmulh(hi, K)
n  = vclmul(qh, K) & 0xFF
res = lo ^ q ^ n
```

This is the standard carryless reduction for the AES polynomial
`x^8+x^4+x^3+x+1` (0x11B), restricted to SEW=8 lanes. The SEW=8 lane
width performs the `& 0xFF` masking for free. Verified exhaustive over
all 65536 input pairs against `gf_mul` (see Tests).

Per multiply-add the backend emits 5 carryless + 2 xor + 1 move/xor =
**8 vector uops** (5 `vclmul`/`vclmulh`, 2 `vxor`, then `vmv.v.v` for
the first accumulate or `vxor` into the accumulator; vs 6 for the
nibble path), trading the vrgather/table loads/mask/constant-pool/
registers. The planner's benchmark gate decides on silicon; the static
`uops_per_byte` model counts 8 per multiply-add honestly.

### Gating

1. **Detection** ([src/riscv/cpu.c](../src/riscv/cpu.c)): no `riscv_hwprobe` key exposes
   Zvbc/Zvbc32e/Zvkg/Zicond, so the existing Zicbop cpuinfo-token scan
   was generalized into `has_ext_cpuinfo()`. New `rj_cpu_info` fields:
   `rvv_zvbc`, `rvv_zvbc32e`, `rvv_zvkg`, `rvv_zicond`
   (`rvv_zvbc`/`rvv_zvkg` are informational only now). Until a
   kernel/QEMU reports the `zvbc32e` token, the twin stays off (fail
   closed); QEMU 8.2.2 does not know Zvbc32e at all (see below).
2. **Planner** ([src/plan.c](../src/plan.c)): `MUL_ZVBC` twins are enumerated iff
   `rvv_zvbc32e` (mirrors the x86-64 AVX-512 GFNI pattern).
   `MUL_NIBBLE` is always enumerated as the fallback. Remat twins stay
   nibble-only (nothing to rematerialize on the Zvbc path).
3. **Backend** ([src/riscv/jit.c](../src/riscv/jit.c)): `jit_build_version` re-checks
   `rvv_zvbc32e` and fails cleanly (no code, error message) when it is
   missing — defense in depth against force-built candidates.

The twin is a third `mul_t` method (`MUL_ZVBC`), not a cand flag: `mul`
already distinguishes multiply methods, and [prune.c](../src/prune.c)'s twin relation
plus `plan_need_regs` key off it.

### Register allocation (`plan_need_regs`, single source of truth)

| Plan kind | ZVBC regs |
|---|---|
| P_HORNER | `(n_out+1)*U + 4 + horner_ntriv` (r1,r2,r3,kvec + 1 cvec/row) |
| P_MULACC nontrivial | `(n_out+1)*U + 5` (r1,r2,r3,cvec,kvec) |
| P_DELTA nontrivial | `(n_out+1)*U + 7` (+ tlo/thi delta temps) |
| P_DELTA all-trivial | `(n_out+1)*U + 2` (tlo/thi only; e.g. RAID-5) |

No nibble tables, no 0x0F mask. Every formula is asserted against the
backend's actual allocation (`assert(r == plan_need_regs(...))`).

### Encoding

`vclmul.vv`/`vclmulh.vv` are OPMVV (funct3=010), funct6 `001100` /
`001101`. Operand placement follows the standard convention (first
source in bits[24:20], second in bits[19:15]), verified byte-for-byte
against GAS. A dedicated emitter (`em_vclmul_vv`/`em_vclmulh_vv`) keeps
the field assignment explicit; the shape test pins 9 register combos
including v0/v31 edge regs.

### QEMU 8.2.2 limitations

1. **Cannot execute** `vclmul.vv`/`vclmulh.vv` (illegal instruction;
   the vector-crypto TCG support postdates 8.2.2). The Zvbc twin's
   execution is validated via a minimal interpreter over the actual
   emitted bytes (see Tests), not via QEMU execution.
2. **cpuinfo quirk**: `-cpu max` advertises `zvbc`/`zvkg` in the guest
   /proc/cpuinfo "isa" line while being unable to execute the
   instructions (verified SIGILL). It does not know Zvbc32e at all --
   no `zvbc32e` token is advertised and `-cpu ...,zvbc32e=false` is
   rejected as an unknown property -- so the detector's `rvv_zvbc32e`
   gate keeps the MUL_ZVBC twin off under QEMU 8.2.2 regardless. The
   Makefile QEMU recipes still pass `-cpu max,zvbc=false,zvkg=false`
   (`QEMU_RVV_CPU`), retained as documentation of the
   advertise-but-can't-execute mismatch. Zicond executes correctly and
   stays enabled.

### Tests ([tests/test_riscv_zvbc.c](../tests/test_riscv_zvbc.c))

All run at vlen=128/256/512 under QEMU (via the `QEMU_RVV_CPU`
recipes):

- **A. Shape**: 9 GAS-verified `(vd,vs1,vs2)` combos, byte-exact.
- **B. Lane model**: C model of the 5-clmul sequence vs `gf_mul`,
  exhaustive over all 65536 pairs + zero/one/all-ones/0x80/K patterns.
- **C. Planner**: all 8 `(zvbc,zvkg,zvbc32e)` combos; ZVBC twins iff
  `zvbc32e` -- `(1,1,0)` must NOT enumerate (SEW=8 on
  Zvbc-without-Zvbc32e is a reserved encoding), `(0,0,1)` MUST
  (Zvbc32e standalone is legal); nibble always present; zvbc+remat
  never enumerated.
- **D. Backend gates**: missing `zvbc32e` (whatever the legacy
  `zvbc`/`zvkg` gates say) and zvbc+remat all fail cleanly with an
  error; `zvbc32e` alone (legacy gates clear) builds with `+zbc` in
  the desc; the **real emitted bytes** are checked — (i) the first
  `vclmul.vv` has the expected register wiring, (ii) the 9-word backend
  sequence is executed by the interpreter and must equal
  `gf_mul(c,a)^t` per lane. (This check caught a real backend bug
  during development: the Horner path used accumulate instead of
  overwrite semantics.)
- **D2. MULACC/DELTA semantic**: one MULACC and one DELTA ZVBC twin's
  real multiply sequences (found by register-wiring scan, coefficient
  decoded from the preceding `addi`/`vmv.v.x` broadcast) run through
  the interpreter vs `gf_mul` -- the historically buggy paths (P0
  lived here), previously covered only structurally (G).
- **D3. Prologue broadcasts**: every adjacent `addi`/`vmv.v.x` pair in
  the emitted code is decoded and the broadcast scalars checked --
  `0x1B` -> `kvec` (and nothing else binds `kvec`), row coefficients
  -> HORNER cvec slots, occurring plan coefficients -> MULACC/DELTA
  `cvec`. D/D2 prime the broadcast regs themselves and would miss a
  wrong scalar or target.
- **E. Interpreter differential**: emitter-built ZVBC and nibble
  sequences run on the same random/edge inputs; both must match
  `gf_mul` and agree with each other.
- **F. Zicond absence**: see B6 below — now scans nibble AND zvbc
  twins, with a positive control pinning the czero matcher to
  GAS-verified czero words so the scan can never pass vacuously.
- **G. Zvbc MULACC/DELTA: no constant-pool table loads**: builds ZVBC
  twins for Horner (RS(6,3)), MULACC (custom non-Vandermonde 4+2), and
  P_DELTA (RS(4,2) parity update) and asserts no `vle8.v` with the
  pool-address scratch (T3) as base — the Zvbc path needs no tables.
  Regression test for the closeout bug below (red-green verified:
  fails with "N pool table loads" on the unfixed code, for both the
  MULACC and DELTA call sites independently).

Red-green evidence: mutating the emitter funct6, the backend register
assignment, and the Horner overwrite/accumulate semantics each produce
a failing test with a precise diagnostic.

### Closeout findings (2026-10-03, Brvv-CLOSEOUT)

Two genuine bugs were found by the closeout review in code the
original implementer left untested, both fixed and regression-tested:

1. **Zvbc MULACC/DELTA emitted nibble-table loads into v0**
   ([src/riscv/jit.c](../src/riscv/jit.c)): the `c_ldtab` calls in the P_MULACC and
   P_DELTA bodies ran unconditionally, but on the Zvbc path
   `tbl_lo`/`tbl_hi` are never assigned (0 = v0), so the loads
   clobbered accumulator v0 and added phantom uops/mem to the static
   model. The existing tests never built a Zvbc MULACC/DELTA twin
   semantically (only Horner), so it slipped through. Fix: guard both
   call sites with `!cg->zvbc`; harden with the codebase's -1
   tripwire pattern (`tbl_lo`/`tbl_hi` init to -1, assert in
   `c_ldtab` — the same silent-v0 hazard `c_ldtab1` already guards).
2. **The Zicond-absence matcher was vacuous** (`is_czero` in the
   test): the mask kept the funct3 bits while the constant had
   funct3=0, so no czero word could ever match — section F passed
   trivially. Fix: mask funct7+opcode only, check funct3 separately;
   added a positive control on the two GAS-verified czero words plus
   near-miss negatives, and extended the scan to ZVBC twins.

## B6 — Zicond: REJECTED with evidence

### Branch inventory of the RVV backend ([src/riscv/jit.c](../src/riscv/jit.c))

The JITed function contains exactly 5 conditional branches:

1. `len == 0` early-out guard (prologue; `c_beq_long`).
2. `niter == 0` skip around the main counted loop (`c_beq_long`).
3. The main counted loop's back-edge (`c_bne_long`).
4. `remaining/U == 0` skip around the tail section (`c_beq_long`).
5. `remaining == 0` skip inside tail handling (`c_beq_long`).

(Plus 1 unconditional `jal` over the tail and the `ret` via `jalr` --
neither is conditional.)

(`c_beq_long`/`c_bne_long` wrap beq/bne+jal to span >±4 KiB bodies.)

### Why `czero.eqz`/`czero.nez` cannot help

`czero` is a conditional-move (select 0 or rs1 based on a condition),
not a branch eliminator for control flow with side effects:

- Branches 1, 2, 4, 5 guard **side-effecting bodies** (loop bodies,
  stores). A cmov cannot skip stores; predicating them would require
  masked vector ops, a different (larger) change.
- Branch 3 is the counted loop back-edge; `czero` cannot replace a
  loop branch.
- The unconditional `jal` is not a conditional branch.

The `vl = 0` trick (setting VL to zero to nullify a body) also fails:
`vsetvli` with AVL=0 yields VLMAX, not 0 — there is no way to get a
zero VL from the length-setting instructions.

No profitable branch-kill was found. Zicond detection
(`rvv_zicond`) is still implemented (cheap, and documents the CPU
capability for future work), but no backend path consumes it.

### Regression pin

[tests/test_riscv_zvbc.c](../tests/test_riscv_zvbc.c) section F builds every RVV candidate the
planner's RVV branch enumerates for RS(6,3) and RS(4,2) (both Horner)
and RS(3,1) (XOR, RAID-5-like) — with `rvv_zicond` advertised, nibble
and ZVBC twins, every U — and asserts zero `czero.eqz`/`czero.nez`
words in the emitted code. NOTE on scope: the planner's RVV branch
only sets `.isa`/`.mul`/`.U`/`.remat`, so nt/prefetch/notail variants
(api.c-level options, never enumerated by `plan_candidates`) are not
covered here; P_DELTA plans are not in the F set either (they are
covered for the table-load scan in G). A positive control pins the
matcher to GAS-verified czero words (added in closeout: the original
matcher was vacuous). If a future change introduces a Zicond use,
this test forces it to be
a deliberate, reviewed decision.

## Files changed

- [include/raidjit.h](../include/raidjit.h): `rvv_zvbc`, `rvv_zvbc32e`, `rvv_zvkg`,
  `rvv_zicond` fields.
- [src/plan.h](../src/plan.h): `MUL_ZVBC` in `mul_t`.
- [src/plan.c](../src/plan.c): `plan_need_regs` ZVBC branch; RVV candidate
  enumeration with the ZVBC twin.
- [src/riscv/cpu.c](../src/riscv/cpu.c): `has_ext_cpuinfo()` generalization; four new
  detections.
- [src/riscv/emit.h](../src/riscv/emit.h), [src/riscv/emit.c](../src/riscv/emit.c): `em_vclmul_vv`,
  `em_vclmulh_vv`.
- [src/riscv/jit.c](../src/riscv/jit.c): ZVBC codegen (`c_gfmul_acc_zvbc`, prologue
  broadcasts, register allocation, gate re-check, `+zbc` desc).
- [tests/test_riscv_zvbc.c](../tests/test_riscv_zvbc.c): new (A–G above).
- [tests/test_pressure.c](../tests/test_pressure.c): ZVBC exact pressure values.
- [Makefile](../Makefile): test registration; QEMU recipes at 3 vlens;
  `QEMU_RVV_CPU` with the zvbc/zvkg quirk workaround.
- [docs/rvv_zvbc.md](rvv_zvbc.md): this file.
- `~/workspace/raidjit-tracks/Brvv/DESIGN.md`: design note.

RV-P was not touched.
