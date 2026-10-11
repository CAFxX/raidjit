/* AArch64 fast CRC-32 variants (Option C, Phase 1). Private header.
 *
 * Entry points (exact names per the cross-worker contract in
 * hidden_files/crc_impl_SPEC.md R2.5):
 *
 *   rj_crc32_fast_arm          best available ARM variant (internal
 *                              runtime selection, thread-safe,
 *                              first-call cached).
 *   rj_crc32_fast_arm_fusion   Cawley 2025 fusion v9s3x2e_s3 (M1): 9 PMULL
 *                              vector lanes (144-byte stride) + 3 scalar
 *                              crc32 lanes, EOR3-combined.
 *                              Needs pmull && sha3 && crc32.
 *   rj_crc32_fast_arm_fusion_n1
 *                              Cawley 2025 fusion v3s4x2e_v2 (N1): 3 PMULL
 *                              vector lanes (48-byte stride) + 4 scalar
 *                              crc32 lanes, no EOR3. Needs pmull && sha3
 *                              && crc32.
 *                              (Finding 13: per-core lane variants from
 *                              the synthesis; selected by MIDR
 *                              implementer — 0x61 Apple → M1 variant,
 *                              else N1 variant. Rationale: N1's PMULL
 *                              throughput is 1/cycle, so the M1
 *                              variant's 9 vector lanes would saturate
 *                              the vector unit (~10.7 B/c analytic)
 *                              while the N1 variant balances 6 PMULL
 *                              against 8 crc32d (~14 B/c analytic,
 *                              12 B/c reported). Both are bit-exact;
 *                              the difference is purely throughput.)
 *   rj_crc32_fast_arm_crc32hw  scalar crc32x 3-way interleave (24 B/iter).
 *                              Needs the CRC32 instructions.
 *   rj_crc32_fast_arm_pmull    pure PMULL fold (dougallj structure,
 *                              4 chains x 16 B). Needs pmull only.
 *   rj_crc32_fast_arm_class    0=scalar slice-by-8, 1=crc32hw, 2=pmull,
 *                              3=fusion: which variant rj_crc32_fast_arm
 *                              selects on this machine.
 *
 * All five share rj_crc32_update's chaining contract: they take the
 * running (uncomplemented) CRC and return the updated running CRC, so
 * each is a drop-in replacement for rj_crc32_update. The scalar
 * slice-by-8 fallback is rj_crc32_update itself (reused, not
 * duplicated).
 *
 * Dispatch order (spec R2, remediated review R1-1): fusion first
 * (pmull && sha3 && crc32 — the crc32 conjunct is explicit because both
 * fusion sub-variants execute the scalar crc32 instructions; without it
 * they would SIGILL, violating the "never SIGILL" contract),
 * then scalar crc32x, then pure PMULL fold, then slice-by-8. The scalar
 * tier deliberately precedes the pure PMULL fold: on Neoverse N1 the
 * pure PMULL fold measures 7.3 B/cycle while scalar crc32x measures 7.9
 * B/cycle (the "N1 inversion" documented in
 * hidden_files/crc_optimum_research/FINAL_SYNTHESIS.md); the fusion
 * exists to use both units at once. Apple cores (which always have
 * pmull+sha3+crc32) take tier 1, so no Apple regression.
 *
 * Lane-count justification for the crc32x tier: the CRC32 instruction
 * retires at most 1/cycle on every documented core (N1 SOG: latency 2,
 * throughput 1, pipe M; the same 1/cycle ceiling on M1), i.e. 8
 * bytes/cycle is the hard ceiling. A 3-way interleave hides the
 * latency-2/3 chain and reaches ~7.9 B/cycle in Cawley's measurements
 * (23.57 GB/s @3.0 GHz on N1, 25.03 GB/s @3.2 GHz on M1); a 4th lane
 * cannot pass the 1/cycle ceiling and only adds combine cost.
 *
 * The translation unit compiles on any host: every intrinsic-using
 * function carries a function-level target attribute enabling exactly
 * the extensions it needs (+crc / +crypto / +sha3), and all
 * AArch64-specific code is inside #ifdef __aarch64__. The aarch64 file
 * is only linked into aarch64 builds (Makefile ARCH_SRCS); on other
 * hosts the TU compiles to nothing.
 */
#pragma once
#include <stddef.h>
#include <stdint.h>

uint32_t rj_crc32_fast_arm(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_fast_arm_fusion(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_fast_arm_fusion_n1(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_fast_arm_crc32hw(uint32_t crc, const void *data, size_t len);
uint32_t rj_crc32_fast_arm_pmull(uint32_t crc, const void *data, size_t len);
int rj_crc32_fast_arm_class(void);

/* Which fusion sub-variant this machine selected: 0 = M1 v9s3x2e_s3,
 * 1 = N1 v3s4x2e_v2 (Finding 13). Test-visible; only meaningful when
 * rj_crc32_fast_arm_class() == 3. */
int rj_crc32_fast_arm_fusion_sub(void);

/* Pure function of the three feature flags -> variant class. Test-only
 * visibility for the dispatch-chain decision table (all 8 flag combos);
 * production selection goes through rj_cpu_detect. */
int rj_crc32_fast_arm_pick(int crc32, int pmull, int sha3);

/* Pure function of the MIDR implementer -> fusion sub-variant
 * (0 = M1 v9s3x2e_s3 iff implementer == 0x61, else 1 = N1 v3s4x2e_v2).
 * Test-only visibility for the per-core selection (Finding 13). */
int rj_crc32_fast_arm_pick_fusion(int implementer);
