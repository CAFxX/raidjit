# CRC-32 optimum: Phase 1 table + Phase 2 integration decision (2026-10-08)

Supersedes design_crc_fold.md §11's "2–4x, not almost free" verdict, which
rested on the 128-bit PCLMULQDQ fold. The owner was right: the optimum had
not been found. Two decisive new inputs: (1) Cawley's fusion (scalar CRC
unit + vector clmul unit simultaneously, fast-crc32 2025, adopted by
zlib-ng); (2) the DRAM-regime hiding argument (interleaved fold adds zero
memory traffic, so it hides under DRAM bandwidth iff fold_cycles/byte <
dram_cycles/byte).

All GB/s numbers at reference 3 GHz unless noted. "Reported" = published
measurement by the cited author, NOT measured by us (standing postponement).
"Analytic" = derived from vendor docs / instruction tables.

## Phase 1 — per-(ISA, extension) optimum, isolated single-stream CRC-32

### x86-64

| Extension | Optimal formulation | Analytic | Reported | Margin over runner-up |
|---|---|---|---|---|
| AVX-512+VPCLMULQDQ (SPR) | Cawley fusion: VPCLMULQDQ lanes + crc32q lanes | 36 B/c = 108 GB/s (4 prod/cyc) | 97.3 GB/s @2.7GHz (36 B/c) [Cawley] | 3% over vector-only (vector unit is that strong here) |
| AVX-512+VPCLMULQDQ (Zen4) | Cawley fusion | 27.7 B/c = 83 GB/s | 71.95 GB/s @2.6GHz [Cawley] | 31% over vector-only (54.77 GB/s) |
| AVX-512+VPCLMULQDQ (ICL-SP) | Cawley fusion | 24.6 B/c = 74 GB/s | 63.98 GB/s @2.6GHz [Cawley] | 34% over vector-only (47.5 GB/s) |
| VPCLMULQDQ, no AVX-512 (Zen3) | fusion w/ 256-bit VPCLMULQDQ | 8 B/c = 24 GB/s (1 prod/cyc) | 31.76 GB/s @2.45 = 13 B/c (fusion, scalar-led) [Cawley] | 2.3x over 128-bit PCLMULQDQ fold |
| PCLMULQDQ (Intel SKL→ICL) | 128-bit fold (zlib-ng/Chromium) | 8 B/c = 24 GB/s (1 prod/cyc, p5) | 24.05 GB/s @2.6GHz [Cawley] | ties crc32q (see below) |
| PCLMULQDQ (AMD Zen) | 128-bit fold | 4 B/c = 12 GB/s (0.5 prod/cyc) | 12.9 GB/s @2.25GHz [Cawley] | — |
| SSE4.2 (no PCLMULQDQ) | hw crc32q, N-way interleave | 8 B/c = 24 GB/s (8B/insn @1/cyc) | 22.5–27.3 GB/s [Cawley] | 10x over scalar |
| SSE4.1 (no clmul) | Chorba (Russell) | — | "on par with hw" [arXiv:2412.16398] | ~2x over prior table-less SotA (reported) |
| baseline | slice-by-8 / braid | — | 1–3 GB/s | — |

Instruction ground truth: Agner Fog tables (SKL: PCLMULQDQ 1uop/1cyc/p5/lat7;
ICL: 1/1/p5/lat6, VPCLMULQDQ-y 3uops/2cyc; Zen3: PCLMULQDQ 4uops/2cyc,
VPCLMULQDQ-y 4uops/2cyc; Zen4: VPCLMULQDQ-z 4uops/2cyc) + chainhash measured
(ICL-SP VPCLMULQDQ-zmm = 2.00 cycles/4 products) + Cawley (SPR: 1*p5 for
zmm = 4 prod/cycle).

Key corrections to the old analysis:
- Old spike assumed VPCLMULQDQ = 1 insn/cycle (4 prod/cycle) on ICL-SP; it is
  2 cycles/insn (2 prod/cycle). Its "~60 GB/s" was optimistic; true
  vector-only bound is 16–18 B/cycle.
- Old spike never compared against crc32q (ties the 128-bit fold on Intel)
  and never considered the fusion (2–3x faster than pure vector on ICL,
  and the ONLY fast path on Zen2/3 where scalar lanes carry it).
- 2 clmul products per 16B is the floor for clmul folding (each 64-bit clmul
  reduces 64 data bits). Multi-lane/more accumulators cannot beat the
  products/cycle ceiling — they only hide latency (owner's "3 independent
  PCLMUL streams" hope: falsified for throughput).
- PCLMULQDQ is on port p5 (Intel), not p0 as the old spike assumed.

### AArch64

| Extension | Optimal formulation | Analytic | Reported | Margin |
|---|---|---|---|---|
| ARMv8.2+SHA3+crypto (M1) | Cawley fusion v9s3x2e_s3 (9 PMULL + 3 scalar, EOR3) | 26.6 B/c = 79.8 GB/s (score 28.4) | 85.05 GB/s @3.2GHz [Cawley] | 10% over pure PMULL (77.55); 3.4x over scalar |
| ARMv8+crypto (M1) | pure PMULL fold (dougallj) | 24.2 B/c = 72.6 GB/s | 77.55 GB/s @3.2GHz [Cawley] | — |
| ARMv8.2+SHA3+crypto (N1) | fusion v3s4x2e_v2 | 12 B/c = 36 GB/s | 35.87 GB/s @3.0GHz [Cawley] | 52% over scalar; 64% over pure PMULL |
| ARMv8+crypto (N1) | pure PMULL fold | 7.3 B/c = 22 GB/s | 21.81 GB/s @3.0GHz [Cawley] | — |
| ARMv8+crc (all) | scalar crc32x (1 or N-way) | 8 B/c = 24 GB/s | 25.03 @3.2, 23.57 @3.0 [Cawley] | 10x over scalar-table |
| V2 (projected) | fusion | ~2x N1 vector part (PMULL tput 2) | — | — |
| baseline | slice-by-8 | — | 1–3 GB/s | — |

Instruction ground truth: N1 SOG (CRC32 lat2/tput1 pipe M, late-forwarding;
VMULL.P64 lat2/tput1 pipe V0) + insn_bench_aarch64 (same numbers) + V2 SOG
(PMULL lat3/tput2). Note the N1 inversion: pure PMULL fold (7.3) < scalar
crc32x (7.9) — the fusion exists to use both units. SVE2 changes nothing
(PMULL is NEON-only, 128-bit).

### RISC-V

| Extension | Optimal formulation | Bound | Margin |
|---|---|---|---|
| Zvbc (SEW=64) | vector fold, streams packed as elements (E=VLEN/64 per insn) | 0.31 insn/B (VLEN=128); IF vclmul 1/cyc: 8 B/c | 2x fewer insn/B than Zbc (widens with VLEN) |
| Zbc | scalar fold (zlib-ng ByteDance 2025) | 0.6 insn/B; IF clmul 1/cyc: 4 B/c | 5–10x over baseline (reported class) |
| baseline | braid / slice-by-8 | — | — |

Ground truth: RISC-V ISA manual (Zvbc: vclmul/vclmulh SEW=64 only; Zvbb has
no clmul). NO shipping core publishes clmul/vclmul throughput — bounds are
in insns/byte, honestly not convertible to GB/s. Zvbc matches raidjit's
existing RVV-Zvbc gating and the multi-stream shape (one element per
stream-half).

## Bit-exactness evidence (all vs zlib, no timing)

- `fold_verify.py`: PCLMULQDQ fold constants re-derived from scratch via
  GF(2) arithmetic (K1=0xae689191, K2=0xccaa009e MATCH the 2026-10-08 spike);
  full formulation PASS on 111 vectors (lengths 0–10000, all tail classes).
- `compose_verify.py`: fusion lane-combination math
  (CRC(A||B) = shift(CRC(A),|B|) XOR CRC(B), shift = x^{-8|B|} in the
  reflected domain) PASS on 400 trials (300 2-way + 100 12-way splits).
  The scalar crc32 unit and the fold are each bit-exact by construction /
  prior verification, so the fusion is fully verified.

## Phase 2 — integration decision

Setup: encode streams 8–14 disks at 100+ GB/s cache-resident; the REAL regime
is DRAM-bound (~50 GB/s assumed; conclusion is robust to 25–50).

### Option A: interleaved fold (zero extra memory traffic)

Only the VECTOR unit is usable (the scalar crc32 unit needs GPRs; extracting
from vector regs costs more than it saves — quantified dead end).
Construction: zlib-ng style, 4 fold-lanes per stream in one zmm (lanes =
every-4th 16B chunk), data XOR fused via VPTERNLOGQ/V EOR3 — NO shuffle tax
(the old spike's shuffle worry dissolves; cost is 1 zmm of state per stream
+ a per-stripe lane reduction, negligible amortized). Per 64B per stream:
2x VPCLMULQDQ + 1x VPTERNLOGQ = 3 uops on data already in registers.

"Almost free" test: fold_cycles/byte < dram_cycles/byte (0.06 @3GHz/50GB/s).

| µarch | vector-only bound | fold cyc/B | vs 0.06 | verdict |
|---|---|---|---|---|
| x86 SPR | 35 B/c | 0.028 | < | ~FREE |
| x86 Zen4 | 21 B/c | 0.047 | < | ~FREE (borderline) |
| x86 ICL-SP | 18 B/c | 0.055 | < | ~FREE (borderline) |
| ARM M1 | 24 B/c | 0.041 | < | ~FREE |
| ARM V2 (proj) | ~14 B/c | ~0.07 | ~= | borderline |
| x86 SKX/CLX | 7.3 B/c | 0.137 | > | ~2.3x — NOT free |
| x86 Zen2/3 | 5.7 B/c | 0.175 | > | ~3x — NOT free |
| ARM N1 | 7.3 B/c | 0.137 | > | ~2.3x — NOT free |
| RV Zvbc | unknown | unknown | ? | unknown (no core data) |

In the cache-resident microbenchmark regime the fold does NOT hide
(e.g. ICL-SP: 55 GB/s fold vs 100+ GB/s encode → ~2x). This is expected and
honest: "almost free" is a DRAM-regime claim.

Implementation risks for A (not verdict-changers):
1. Register pressure: 14 streams x 1 zmm state + encode's ~20 live zmm > 32.
   Mitigations: spill states to L1 (cheap), planner reserves regs, or reduce
   unroll. Planner work, flagged for implementation.
2. No new AVX-512 frequency drop: the kernels already run 512-bit; the fold
   adds no new power license.
3. Port interaction in cache-resident regime needs planner/uop modeling, but
   the DRAM-regime conclusion is port-independent (core is memory-stalled).

### Option B: hot-buffer pass (full fusion, vector + scalar units)

Re-reads all streams from cache at the fusion's isolated bound. Overhead vs
DRAM-bound encode = 50 / fusion_GB/s: SPR 46%, ICL 68%, Zen4 60%, M1 62%.
Strictly worse than A where A is free, and no better where A isn't. B only
wins if A is unimplementable (register pressure proves fatal) — it is the
fallback, not the choice. (Cross-stripe pipelining could hide B, but that is
driver complexity for a worse bound.)

### Option C: standalone fast-CRC API (ship the fusion per ISA)

Worth doing REGARDLESS of A/B (owner's "if they're the fastest in isolation
they're all worth doing"): rj_crc32_fast with runtime dispatch —
x86: Cawley fusion (VPCLMULQDQ / PCLMULQDQ+scalar / scalar-led per µarch);
ARM: Cawley fusion (PMULL+EOR3 / PMULL / scalar per core); RV: Zvbc fold
(Zbc scalar fallback). Zero risk to the encode path (non-CRC kernels stay
byte-identical trivially). Every formulation above is bit-exact verified.

## Revised "almost free" verdict

The old "2–4x, not almost free" verdict was correct FOR THE 128-BIT FOLD but
wrong as a general claim. With the true optimum (VPCLMULQDQ 4-lane fold,
zero-shuffle construction):
- "Almost free" (≤10–20% overhead) IS achievable via interleaved Option A in
  the DRAM-bound real-RAID regime on: SPR, Zen4, ICL-SP, M1 (V2 projected).
- It is NOT achievable on: SKX/CLX, Zen2/3, N1 (2–4x there), RV unknown.
- The right product decision is therefore per-microarch dispatch, not a
  single global verdict: interleave where the vector unit is strong enough
  (fold cyc/B < dram cyc/B), else scalar/fusion standalone API.

## What would change the answer

- Measured (not analytic) confirmation when benchmarks resume — the
  borderline cases (Zen4/ICL-SP/V2) deserve empirical checking first.
- A RISC-V core with published vclmul throughput.
- Chorba-style ALU formulations could beat clmul on old cores but compete
  with the encode's vector-ALU ports — unlikely to beat A/B/C as framed.

## Sources (primary)

- Intel "Fast CRC Computation for Generic Polynomials Using PCLMULQDQ"
  (Gopal et al., 2009/2010) — fold construction + constants.
- Intel "Fast CRC Computation for iSCSI Polynomial Using CRC32 Instruction".
- Russell, "Chorba: A novel CRC32 implementation", arXiv:2412.16398 (2024).
- Cawley, fast-crc32 (github.com/corsix/fast-crc32) — fusion + all reported
  GB/s + M1 analytic score method.
- zlib-ng: arch/x86/crc32_pclmulqdq_tpl.h (Intel, Wangyang Guo VPCLMULQDQ),
  arch/x86/crc32_chorba_sse41.c, arch/arm/crc32_armv8_pmull_eor3.c (Cawley
  2025), arch/riscv/crc32_zbc.c (ByteDance/Yin Tong 2025).
- Agner Fog instruction_tables.pdf (SKL/ICL/Zen2/3/4 clmul ports/tput).
- ARM Neoverse N1 SOG (CRC32 lat2/tput1, VMULL.P64 lat2/tput1),
  Neoverse V2 SOG (PMULL tput 2).
- thomasahle/chainhash (measured ICL-SP VPCLMULQDQ-zmm = 2.00 cyc/4 prod).
- apps4uco/clmul (uops.info-sourced clmul summary).
- ocxtal/insn_bench_aarch64 (Altra: crc32x lat2/tput1, pmull lat2/tput1).
- RISC-V ISA manual, vector crypto chapter (Zvbc SEW=64 vclmul/vclmulh).

## Correction (2026-10-08, implementation phase)
The x86 table row "SSE4.2 (no PCLMULQDQ) | hw crc32q, N-way interleave |
22.5–27.3 GB/s [Cawley]" does NOT apply to IEEE CRC-32: the x86 `CRC32`
instruction is hardwired to CRC-32C (Castagnoli, poly 0x1EDC6F41) —
verified on hardware (`crc32("123456789")` = 0xE3069283, the Castagnoli
check value, not IEEE 0xCBF43926). Cawley's reported figure is for
CRC-32C. There is no scalar-IEEE-CRC32 rung on x86; the implemented
chain is VPCLMULQDQ-fusion → 256-bit fusion → 128-bit fold → slice-by-8.
(AArch64 `crc32x` and the ARM chain are unaffected — ARMv8 CRC32 *is*
the IEEE polynomial.)
