# AArch64 PMULL-based GF(2⁸) multiply: survey, analysis, decision

**Date:** 2026-10-01 · **Status:** study complete — **do NOT integrate** (evidence below)
**Scope:** NEON and SVE backends in [src/aarch64/jit.c](../src/aarch64/jit.c). x86/RISC-V untouched.

## Decision

No `MUL_PMULL` method is added. The nibble-TBL path wins by a margin no
microarchitecture can plausibly overturn: **23 vs 7 vector instructions per
16 bytes (3.3×)**, a ~2× longer dependency chain, worse register pressure,
and stricter ISA requirements. The planner's candidate set is unchanged.

## 1. Architectural facts (verified, not assumed)

Two premises in the task brief turned out to be wrong; both were checked
against the assembler and ARM documentation rather than taken on faith.

| Claim | Verdict |
|---|---|
| "NEON PMULL is 64-bit lanes only" | **False.** `PMULL Vd.8H, Vn.8B, Vm.8B` exists: 8 lanes of 8-bit → 16-bit carryless multiply. Assembles fine (`aarch64-linux-gnu-as`, needs `+crypto` on ARMv8.0-A). ARM's Cortex-A76 Software Optimization Guide lists it explicitly ("PMULL Vd.8H, Vn.8B, Vm.8B", note 3, §3.15). |
| "SVE has no carryless multiply; SVE2 doesn't add PMULL" | **False for SVE2+PMULL.** Base SVE and plain SVE2 have none (assembler rejects `pmull` on z-regs), but the SVE PMULL extension (`ID_AA64ZFR0_EL1.AES == 0b0010`, `HWCAP2_SVEPMULL`) adds `PMULLB`/`PMULLT`: 8-bit → 16-bit carryless multiply on even/odd lanes. Verified: assembles under `-march=armv8.2-a+sve2-aes` and `armv9-a`, rejected under `armv8.6-a+sve`. It is an *optional* extension on top of SVE2, so its addressable machine
set is strictly smaller than base SVE's. |

Consequence: a PMULL construction is *expressible* on both NEON and
(SVE2+PMULL) SVE, so the study could not stop at "no such instruction".

## 2. Baseline: current nibble-TBL cost

From [src/aarch64/jit.c](../src/aarch64/jit.c) HORNER body, per nontrivial `(row j, unroll u)`
— one 16-byte (NEON) / VL-byte (SVE) vector:

```
AND  tlo, acc, mask      # 1
USHR thi, acc, #4        # 1
TBL  r2, {hreg_lo}, tlo  # 1
MOV  acc, r2             # 1
TBL  r2, {hreg_hi}, thi  # 1
EOR  acc, acc, r2        # 1  (lo ^ hi)
EOR  acc, acc, input     # 1  (Horner add)
```
**7 vector insns per vector** (persistent tables). Remat mode adds 2 pool
`LDR`s → **9**. The MULACC column path has the same per-multiply cost.

## 3. Constructions surveyed

### 3a. NEON PMULL.8H + Barrett reduction mod 0x11B (the best PMULL method)

`p = clmul(a,b)` is 16 bits (degree ≤ 14). Reduce with
x⁸ ≡ x⁴+x³+x+1 (`0x1B`), two folds (second fold needed: first fold leaves
degree ≤ 10):

```
q1 = p >> 8;  t1 = clmul(q1, 0x1B);  r  = (p & 0xFF) ^ t1
q2 = r >> 8;  t2 = clmul(q2, 0x1B);  res = (r & 0xFF) ^ t2
```

NEON sequence per 8-byte half (`vc` = broadcast `0x1B`, `vff` = `0x00FF`
per 16-bit lane, both hoisted):

```
PMULL p.8H,  a.8B, b.8B      # 1
USHR  q1.8H, p.8H, #8        # 1
PMULL t1.8H, q1.8B, c.8B     # 1
AND   pl.16B, p.16B, ff.16B  # 1
EOR   r.16B, pl.16B, t1.16B  # 1
USHR  q2.8H, r.8H, #8        # 1
PMULL t2.8H, q2.8B, c.8B     # 1
AND   rl.16B, r.16B, ff.16B  # 1
EOR   s.16B, rl.16B, t2.16B  # 1
XTN   s.8B, s.8H             # 1  (narrow to bytes)
EOR   acc.8B, acc.8B, s.8B   # 1  (accumulate)
```
11 per half → 22, plus one 16B `EOR acc ^= input` → **23 insns per
16 bytes** (vs 7 for TBL). Correctness of the reduction was verified by a
scalar model against `gf_mul` for all 65,536 `(a,b)` pairs — 0 mismatches.

A shift-XOR fold variant (replace each `clmul(q,0x1B)` by
`(q<<4)^(q<<3)^(q<<1)^q`) costs 6 insns per fold instead of 1 → 27/16B.
Strictly dominated; discarded.

`PMULL.1Q` (64-bit lanes) processes 2 bytes/insn vs 8 for `.8H`.
Strictly dominated; discarded.

### 3b. SVE PMULLB/PMULLT + Barrett

Same math, predicated (`p0` all-true; the backend already uses full
vectors + scalar tail):

```
PMULLB p.h, a.b, b.b   # even lanes -> 16-bit
PMULLT p.h, a.b, b.b   # odd lanes (merge)          } 2
USHR q1.h, p.h, #8                                      1
PMULLB/PMULLT t1.h, q1.b, c.b                           2
AND  pl.h, p.h, #0xFF                                   1
EOR  r.h, pl.h, t1.h                                    1
USHR q2.h, r.h, #8                                      1
PMULLB/PMULLT t2.h, q2.b, c.b                           2
AND  rl.h, r.h, #0xFF                                   1
EOR  s.h, rl.h, t2.h                                    1
UZP1 s.b, s.h, s.h     # narrow                         1
```
**13 insns per VL** vs **7** for the SVE TBL path (1.9×), and it requires
the SVE2+PMULL extension — an optional extension beyond SVE2, so its
addressable machine set is strictly smaller than base SVE's. Discarded.

### 3c. Log/exp tables

`TBL log[a]`, `TBL log[b]`, `ADD`, `TBL exp[]`, plus zero-operand
special-casing (`CMEQ`×2, select) ≈ 8–9 insns/16B, needs 512 B of tables,
and has a data-dependent zero path. Never beats nibble-TBL's 7.
Discarded.

### 3d. Composite field GF((2⁴)²)

A tower-field multiply needs ≥3 GF(2⁴) multiplies plus extension-constant
folds; each GF(2⁴) multiply is itself a small Barrett or table op. Lower
bound exceeds the 7-insn nibble path before writing a single instruction.
Discarded on counting grounds.

## 4. Microarchitectural argument (Cortex-A76, ARM's own numbers)

From the *Cortex-A76 Software Optimization Guide* v10.0, §3.15:

| insn | latency | throughput | pipes |
|---|---|---|---|
| TBL (1–2 table regs) | 2 | **2/cycle** | V0+V1 |
| PMULL.8H (Qform) | 4 | **1/2 cycle** | V0 only |
| AND/EOR/MOV (ASIMD logical) | 2 | 2/cycle | V0+V1 |
| USHR/XTN (shift basic) | 2 | 1/cycle | V1 only |

Throughput floor per 16 bytes:
- **TBL path** (7 insns): ~7 insns at ~2/cycle ≈ **3.5 cycles** (the lone
  V1-only USHR is the only structural hazard).
- **PMULL path** (23 insns): 6× PMULL.8H at ½/cycle on V0 alone =
  **≥12 cycles**, before counting the remaining shift/narrow/logical
  instructions.
- **≥3.4× throughput advantage for TBL** on a core with a *relatively
  good* PMULL. The 3.3× instruction-count ratio is microarchitecture-
  independent; no core makes a multiply-pipe instruction retire 3× faster
  than a permute.

Dependency chain per multiply: PMULL ≈ 4+2+4+2+2+2+4+2+2+2+2+2 ≈ **28
cycles** vs TBL 7×2 = **14 cycles**. Unrolling hides throughput but not a
2× latency gap on the accumulator chain.

Register pressure: PMULL needs ~10 live vector regs per multiply
(a, b, c, p, q1, t1, r, q2, t2, s) vs 6 for TBL
(acc, tlo, thi, r1/input, r2, mask). PMULL is *worse* under pressure, so
there is no "tables don't fit" regime where it wins — remat TBL (9/16B)
still beats PMULL (23/16B) by 2.5×.

ISA availability: PMULL.8H needs the crypto extension (`HWCAP_PMULL`);
SVE PMULLB/T needs SVE2+PMULL (`HWCAP2_SVEPMULL`). TBL is baseline
ARMv8.0 / base SVE. A `MUL_PMULL` candidate would also need hwcap gating
in the planner for machines where it cannot even execute.

## 5. Corroboration

The x86-64 backend faced the identical choice (`PCLMULQDQ` vs
nibble-`PSHUFB`) and likewise never adopted carryless multiply: without a
single-instruction GF multiply (x86 `GFNI`, which has no ARM equivalent),
Barrett reduction always loses to tables. Independent convergence.

## 6. What would change this decision

- An ARM ISA extension providing single-instruction GF(2⁸) multiply
  (analogous to x86 GFNI) — then the comparison is 1–2 insns vs 7 and the
  study must be redone.
- A core where `PMULL.8H` sustains ≥2/cycle *and* the 23-vs-7 count is
  offset — arithmetically impossible without the count changing, so this
  trigger is listed only for completeness.

## 7. Measurement note

QEMU cannot measure real ARM throughput (it does not model pipelines,
issue widths, or latencies), so no cycle measurement under QEMU could
confirm or refute §4. The decision rests on instruction counts (exact,
from the emitted sequences), a machine-checked reduction proof (§3a),
and ARM's published per-instruction characteristics. On real hardware,
the falsifying measurement would be: a PMULL-based Horner kernel
sustaining higher bytes/cycle than the TBL kernel on the same core —
the §4 model predicts it cannot (≥3.4× deficit on Cortex-A76).
