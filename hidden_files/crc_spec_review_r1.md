# CRC impl spec review — Reviewer 1 (fidelity & design correctness)

Reviewer: coordinator doing the reviewer-1 angle directly (the two dispatched
reviewer agents were killed by repeated runtime restarts before producing
findings; their partial checkpoint files held only headers). All claims below
verified against primary sources. Date: 2026-10-08.

## Verified sound (with source)
- x86 CPUID bits: PCLMULQDQ = CPUID.1:ECX bit 1, VPCLMULQDQ = CPUID.7:ECX bit
  10 — confirmed in zlib-ng `arch/x86/x86_features.c`
  (`ecx & 0x2`, `ecx & 0x400`). Spec R3 correct.
- ARM HWCAP bits: HWCAP_PMULL = bit 4, HWCAP_CRC32 = bit 7,
  HWCAP_SHA3 = bit 17, all AT_HWCAP — confirmed in the aarch64 sysroot
  `/usr/aarch64-linux-gnu/include/asm/hwcap.h`. (No HWCAP2_SHA3 on Linux;
  zlib-ng's HWCAP2_SHA3 probe is a FreeBSD/portability fallback.) Spec R3
  correct after the earlier fix.
- µarch gate (R10) membership matches FINAL_SYNTHESIS verdict table
  (SPR/Zen4/ICL-SP in; SKX/Zen3/N1 out; Zen5 excluded = conservative
  correct; RV forced off = correct).
- Public API chaining contract: sound; init/final reuse correct.
- Internal contract (`rj_crc32_fast_<isa>`, backend picks best internally):
  coherent; seam testable via backend test hooks.
- `rj_cpu_info` flag set complete for all dispatch chains.
- Branch plan sane.

## Findings

### R1-1 (MAJOR) — ARM dispatch tier order contradicts the N1 inversion
Spec R2 orders AArch64 as (1) pmull&&sha3 → fusion, (2) pmull → pure PMULL
fold, (3) crc32 → scalar crc32x. FINAL_SYNTHESIS §Phase 1 ARM table documents
the N1 inversion: pure PMULL fold = 7.3 B/c < scalar crc32x = 7.9 B/c
("the fusion exists to use both units"). On any pmull-without-sha3 core
(N1, most Cortex-A7x), the spec's order picks the SLOWER variant.
Apple cores always take tier 1 (they have SHA3), so no Apple regression from
reordering. FIX: tier 2 = `crc32` → scalar crc32x N-way interleave; tier 3 =
`pmull` → pure PMULL fold (only when no CRC32 hw).

### R1-2 (MAJOR) — x86 tier 2 must specify 256-bit VPCLMULQDQ lanes on Zen3-class
FINAL_SYNTHESIS x86 row "VPCLMULQDQ, no AVX-512 (Zen3)": optimal formulation
is "fusion w/ 256-bit VPCLMULQDQ" (reported 13 B/c scalar-led, 2.3x over the
128-bit fold). Spec R2 tier 2 says "scalar+vector fusion (Cawley v-lane
structure)" without pinning the vector lane width. On a Zen3 (vpclmulqdq=1,
avx512f=0) the spec as written lets the worker pick 128-bit PCLMULQDQ lanes
and leave performance on the floor. FIX: in the pclmulqdq tier, vector lanes
are 256-bit VPCLMULQDQ when `vpclmulqdq` is set (independent of avx512f),
else 128-bit PCLMULQDQ. Worker documents the choice.

### R1-3 (MAJOR) — split-point chaining tests missing from R4
R1 promises per-call equivalence, which implies chaining, but R4 never
explicitly tests `f(c,A||B) == f(f(c,A),B)` at odd split points. The fused
driver accumulates CRC per tile — chaining is load-bearing, and a variant
that is correct on whole buffers but wrong across a chunk boundary (e.g. a
fold whose lane-combination assumes buffer start) would pass R4 as written.
FIX: add split-point tests at every split offset mod 64 over a range of
lengths and seeds.

### R1-4 (MAJOR) — ARM fold gate is untestable as specified
R10's ARM gate reads /proc/cpuinfo for MIDR implementer 0x61; R13 requires a
differential proof of the ARM fold. Neither the dev host (x86-64) nor QEMU
aarch64 (no Apple MIDR) can ever satisfy the gate, so the ARM emitter would
ship untested. FIX: the gate takes the MIDR implementer (and feature flags)
as parameters — production passes parsed values, tests inject 0x61+pmull+sha3
— and the ARM fold is then differential-tested under QEMU aarch64
(`-cpu max`; worker verifies max exposes PMULL+SHA3 HWCAPs first, else the
test reports SKIP with a reason, never silent pass).

### R1-5 (MINOR) — `rj_crc32()` one-shot should route to the fast path
`rj_crc32()` (src/crc.h) currently has no callers, but leaving it on the
slice-by-8 path while `rj_crc32_fast` exists is API incoherence. FIX: route
it through `rj_crc32_fast` (init + fast + final). One line; test the KAT
through it.

### R1-6 (MINOR) — x86 tier-1 flag set may need avx512bw
Cawley's AVX-512 fusion may use byte shuffles for lane management. The
worker justifies the exact flag set from the source; if avx512bw (already
detected) is required, it joins the tier-1 condition. Not a blocker.

### R1-7 (MINOR) — R16 ARM doc sections cannot execute-verify locally
The kernels generator verifies bytes by execution. The dev host
(AMD EPYC 9D25, vpclmulqdq present — verified /proc/cpuinfo) CAN
execute-verify x86_64 fold sections. ARM fold sections cannot run locally;
only include them if the generator can execute-verify under QEMU aarch64,
else omit with a "Deliberately not represented" note. FIX: scope R16's
MUST to x86_64; ARM sections conditional.

### R1-8 (MINOR) — mirror zlib-ng's dual SHA3 HWCAP probe
Linux: HWCAP_SHA3 (bit 17, AT_HWCAP) is correct and sufficient. For
portability, also probe HWCAP2_SHA3 under `#ifdef`, mirroring zlib-ng
`arch/arm/arm_features.c` (their HWCAP2 check targets FreeBSD). Follow the
existing local-`#ifndef`-define pattern in src/aarch64/cpu.c.

### R1-9 (NIT) — R5's SIZE_MAX test is unexecutable
len=SIZE_MAX would hang, not wrap. The real bug class is 32-bit truncation
in block-count arithmetic. FIX: test lengths around 2^32 (2^32±16) for
truncation; drop the SIZE_MAX execution requirement (keep a code-review
assertion of no pointer wraparound).
